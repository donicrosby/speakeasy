"""ReferenceVoiceStore tests — the VoxCPM2 clone store on disk.

The store keeps a reference as `<voices_root>/voxcpm2/<id>.wav` (16 kHz mono
PCM_16) plus a metadata `<id>.json`, written WAV first and JSON second so the
JSON is the existence marker. These tests touch only the filesystem; no model
and no `voxcpm` import are involved.
"""

import json

import numpy as np
import pytest
import soundfile as sf

from speakeasy import voices


def _samples(n=16000):
    return np.linspace(-0.5, 0.5, n, dtype=np.float32)


@pytest.fixture()
def store(tmp_path):
    return voices.ReferenceVoiceStore(tmp_path / "voices")


def _write_wav(path, n=16000, sample_rate=16000):
    sf.write(path, _samples(n), sample_rate, subtype="PCM_16", format="WAV")


def test_store_layout_and_roundtrip(store):
    """After save: both files, a 16 kHz mono PCM_16 WAV, four-key JSON."""
    voice_id = "clone-1"
    samples = _samples()

    assert store.exists(voice_id) is False

    created = store.save(voice_id, samples, 16000, "hello there")

    wav = store.root / f"{voice_id}.wav"
    meta = store.root / f"{voice_id}.json"
    assert wav.is_file()
    assert meta.is_file()

    data, sample_rate = sf.read(wav)
    info = sf.info(wav)
    assert sample_rate == 16000
    assert data.ndim == 1  # mono
    assert data.shape[0] == samples.shape[0]  # same number of samples
    assert info.subtype == "PCM_16"

    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert set(payload) == {"engine", "transcript", "seconds", "created"}
    assert payload["engine"] == "voxcpm2"
    assert payload["transcript"] == "hello there"
    assert payload["seconds"] == pytest.approx(1.0)
    assert payload["created"] == created
    assert isinstance(payload["created"], int)

    assert store.exists(voice_id) is True
    assert store.reference_path(voice_id) == wav
    assert store.info(voice_id) == payload
    assert store.ids() == [voice_id]
    assert store.listing() == [(voice_id, payload)]


def test_transcript_may_be_null(store):
    store.save("clone-2", _samples(), 16000, None)
    info = store.info("clone-2")
    assert info["transcript"] is None


def test_directory_created_on_first_use(tmp_path):
    """Constructing the store touches nothing; the first save creates it 0o700."""
    root = tmp_path / "voices"
    store = voices.ReferenceVoiceStore(root)
    assert not store.root.exists()
    store.save("clone-1", _samples(), 16000, None)
    assert store.root.is_dir()
    assert store.root.stat().st_mode & 0o077 == 0


def test_empty_listing_before_anything_is_saved(store):
    assert store.listing() == []
    assert store.ids() == []


def test_wav_without_json_is_not_listed(store):
    """The crash state between the WAV and JSON renames is invisible to listing."""
    store.save("clone-1", _samples(), 16000, None)
    (store.root / "clone-1.json").unlink()

    assert store.listing() == []
    assert store.ids() == []


@pytest.mark.parametrize(
    "bad_id",
    [
        "../x",  # traversal
        "../etc/passwd",
        "A_b",  # uppercase + underscore
        "x" * 65,  # too long
        "clone\n",  # trailing newline
    ],
)
def test_invalid_ids_rejected_before_touching_disk(tmp_path, bad_id):
    root = tmp_path / "voices"
    store = voices.ReferenceVoiceStore(root)
    before = {p for p in tmp_path.rglob("*")}

    with pytest.raises(ValueError):
        store.save(bad_id, _samples(), 16000, None)

    # Nothing was created anywhere, not even the store directory.
    assert {p for p in tmp_path.rglob("*")} == before


def test_delete_twice_and_unknown(store):
    store.save("clone-1", _samples(), 16000, None)

    assert store.delete("clone-1") is True
    assert set(p for p in store.root.iterdir()) == set()
    assert store.delete("clone-1") is False
    assert store.delete("never-saved") is False
    # A non-filename id is answered, never raised on.
    assert store.delete("../etc/passwd") is False


def test_delete_wav_gone_json_present_removes_json(store):
    """A leftover JSON whose WAV vanished is cleaned up; the return says no WAV."""
    store.save("clone-1", _samples(), 16000, None)
    (store.root / "clone-1.wav").unlink()

    assert store.delete("clone-1") is False
    assert not (store.root / "clone-1.json").exists()
    assert set(p for p in store.root.iterdir()) == set()


def test_listing_skips_corrupt_json(store):
    store.save("good-1", _samples(), 16000, None)
    (store.root / "bad-1.json").write_text("{not json", encoding="utf-8")
    _write_wav(store.root / "bad-1.wav")

    assert store.ids() == ["good-1"]


def test_listing_skips_invalid_stem(store):
    store.save("good-1", _samples(), 16000, None)
    # An uppercase stem fails the id regex; both files exist but it is skipped.
    _write_wav(store.root / "Bad_1.wav")
    (store.root / "Bad_1.json").write_text(
        json.dumps(
            {"engine": "voxcpm2", "transcript": None, "seconds": 1.0, "created": 0}
        ),
        encoding="utf-8",
    )

    assert store.ids() == ["good-1"]


def test_listing_is_sorted_by_id(store):
    for voice_id in ("clone-a-b", "clone-z", "clone-a"):
        store.save(voice_id, _samples(), 16000, None)

    assert store.ids() == ["clone-a", "clone-a-b", "clone-z"]
    assert [i for i, _ in store.listing()] == ["clone-a", "clone-a-b", "clone-z"]


def test_info_raises_for_missing_or_corrupt(store):
    with pytest.raises(ValueError):
        store.info("never-saved")

    store.save("clone-1", _samples(), 16000, None)
    (store.root / "clone-1.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        store.info("clone-1")


def test_reference_path_rejects_traversal(store):
    with pytest.raises(ValueError):
        store.reference_path("../etc/passwd")


def test_reverb_store_ignores_the_voxcpm2_subtree(tmp_path):
    """The Reverb VoiceStore on the same root neither lists nor deletes VoxCPM2 files."""
    root = tmp_path / "voices"
    voxcpm2 = voices.ReferenceVoiceStore(root)
    voxcpm2.save("clone-1", _samples(), 16000, None)

    reverb = voices.VoiceStore(root)
    assert reverb.listing() == []
    assert reverb.ids() == []
    assert reverb.delete("clone-1") is False

    # The VoxCPM2 clone is untouched by the Reverb store's listing and delete.
    assert (voxcpm2.root / "clone-1.wav").is_file()
    assert (voxcpm2.root / "clone-1.json").is_file()
    assert voxcpm2.ids() == ["clone-1"]
