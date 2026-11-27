"""VoiceStore tests — the on-disk half of the TTS suite.

The real `reverb_tts` package is never imported for real: its `Voice` on-disk
format is mirrored by a local stand-in and the module is faked through
`sys.modules`, so the suite passes in CI's no-extras job where the `tts` extra
is not installed. Endpoint TTS scenarios are added by later slices.
"""

import sys
import types

import numpy as np
import pytest

from speakeasy import voices


class Voice:
    """Mirrors reverb_tts.Voice's on-disk `.npz` format without the package."""

    def __init__(self, language="en", text="ref words"):
        self.language, self.text = language, text
        self.codes = np.zeros((8, 250), np.int16)

    def save(self, path):
        np.savez(path, codes=self.codes, text=self.text, language=self.language)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            v = cls(str(data["language"]), str(data["text"]))
            v.codes = data["codes"]
            return v


@pytest.fixture()
def fake_reverb(monkeypatch):
    mod = types.ModuleType("reverb_tts")
    mod.Voice = Voice
    monkeypatch.setitem(sys.modules, "reverb_tts", mod)
    return mod


@pytest.fixture()
def store(tmp_path):
    return voices.VoiceStore(tmp_path / "voices")


def test_voices_do_not_import_reverb_tts_at_import_time():
    """src/ must import reverb_tts lazily — CI has no `tts` extra installed."""
    assert "reverb_tts" not in sys.modules


def test_voice_store_roundtrip(store, fake_reverb):
    voice_id = "my-voice-1"
    assert store.exists(voice_id) is False

    store.save(voice_id, Voice(language="de", text="hallo welt"))

    assert store.ids() == [voice_id]
    assert store.exists(voice_id) is True

    info = store.info(voice_id)
    assert info["language"] == "de"
    assert info["transcript"] == "hallo welt"
    assert isinstance(info["created"], int)

    loaded = store.load(voice_id)
    assert loaded.language == "de"
    assert loaded.text == "hallo welt"
    assert loaded.codes.shape == (8, 250)

    assert store.delete(voice_id) is True
    assert store.exists(voice_id) is False
    assert store.ids() == []
    assert not (store.root / f"{voice_id}.npz").exists()


@pytest.mark.parametrize(
    "bad_id",
    [
        "../etc/passwd",
        "../x",
        "..",
        "En_X",  # uppercase
        "a",  # too short
        "x" * 65,  # too long
        "",  # empty
        "clone x",  # space
        "a/b",  # nested
        "clone\n",  # trailing newline — `$` would let this slip past a ^...$ match
    ],
)
def test_invalid_voice_ids_rejected(store, bad_id):
    with pytest.raises(ValueError):
        voices.check_voice_id(bad_id)
    # exists() answers, never raises, for anything that is not a valid id.
    assert store.exists(bad_id) is False
    # delete() answers too.
    assert store.delete(bad_id) is False


def test_delete_unknown_returns_false(store):
    assert store.delete("never-saved") is False
    # An id that is not even a valid filename is refused the same way.
    assert store.delete("../etc/passwd") is False


def test_voices_dir_created_private(store):
    """§5: the voices directory is created with mode 0o700 (no group/other bits)."""
    assert store.root.is_dir()
    assert store.root.stat().st_mode & 0o077 == 0


def test_corrupt_npz_skipped_in_listing(store):
    """A garbage `.npz` in the voices dir is skipped, never raised on.

    §8.3 files the scenario at the endpoint, but the endpoint lands in slice 3.
    Here we test the store half it depends on: `ids()`/`info()` never surface a
    file that cannot be read, so `GET /v1/audio/voices` cannot 500 on one.
    """
    store.save("good-1", Voice(text="fine"))
    (store.root / "bad-1.npz").write_bytes(b"this is not a numpy archive")

    assert store.ids() == ["good-1"]

    # Building the listing shape slice 3 will return must not raise.
    listed = [{"id": i, **store.info(i)} for i in store.ids()]
    assert listed == [
        {
            "id": "good-1",
            "language": "en",
            "transcript": "fine",
            "created": listed[0]["created"],
        }
    ]


def test_save_is_atomic_via_temp_file(store):
    """The final path only ever holds a complete file: temp then os.replace."""
    voice_id = "atomic-1"
    seen = {}

    class ProbeVoice:
        def save(self, path):
            seen["path"] = path
            # The writer is handed a temporary in the store dir, never the
            # final path — so the final path is only ever a complete file.
            assert path.parent == store.root
            assert path.name.startswith(f".{voice_id}.")
            assert path.name.endswith(".tmp.npz")
            assert path != store.root / f"{voice_id}.npz"
            np.savez(path, codes=np.zeros((8, 2), np.int16), text="t", language="en")

    store.save(voice_id, ProbeVoice())

    assert seen["path"].parent == store.root
    assert store.exists(voice_id) is True
    # No temporary file survives the replace.
    assert [p.name for p in store.root.iterdir()] == [f"{voice_id}.npz"]


@pytest.mark.parametrize("bad_id", ["../etc/passwd", "../x", "..", "a/b", "En_X", ""])
def test_rejected_ids_create_no_file_outside_root(tmp_path, bad_id):
    root = tmp_path / "voices"
    store = voices.VoiceStore(root)
    before = {p for p in tmp_path.rglob("*")}

    with pytest.raises(ValueError):
        store.save(bad_id, Voice())

    assert {p for p in tmp_path.rglob("*")} == before
