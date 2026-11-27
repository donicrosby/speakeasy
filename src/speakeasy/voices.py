"""Saved voice clones on disk: `<root>/<voice_id>.npz`, one file per clone.

An id is `^[a-z0-9][a-z0-9-]{1,63}$` and becomes a filename, so anything else is
rejected outright — never sanitized into a name that might collide or traverse.
`reverb_tts` is imported lazily (only `load` needs it) so this module imports
cleanly without the `tts` extra.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

VOICE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")


def check_voice_id(voice_id: str) -> str:
    """Return `voice_id` when it is a safe filename stem, else raise ValueError.

    `fullmatch`, not `match`: `$` matches just before a trailing newline, so a
    `match` would accept `"clone\\n"` and turn it into a filename.
    """
    if not VOICE_ID.fullmatch(voice_id):
        raise ValueError(
            f"invalid voice id {voice_id!r}: 2-64 chars of a-z, 0-9 and '-', "
            "starting with a letter or digit"
        )
    return voice_id


class VoiceStore:
    """Clones as `<root>/<id>.npz`. Ids are validated before they become paths."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _path(self, voice_id: str) -> Path:
        return self.root / f"{check_voice_id(voice_id)}.npz"

    def _read(self, path: Path) -> dict | None:
        """The listing dict for a readable clone, or None if it cannot be read.

        A corrupt or non-numpy `.npz` (concurrent write, manual edit, fs
        corruption) must not break a listing, so every read goes through here.
        """
        import numpy as np

        try:
            with np.load(path) as data:
                return {
                    "language": str(data["language"]),
                    "transcript": str(data["text"]),
                    "created": int(path.stat().st_mtime),
                }
        except Exception:  # noqa: BLE001 — a bad file is skipped, not fatal
            return None

    def listing(self) -> list[tuple[str, dict]]:
        """`(id, info)` for every readable clone, sorted by id — one read per file.

        A clone deleted mid-listing (its file gone before the read) or a corrupt
        file is skipped, never raised on, so a listing cannot 500 on the race.
        """
        entries: list[tuple[str, dict]] = []
        for path in sorted(self.root.glob("*.npz")):
            if not VOICE_ID.fullmatch(path.stem):
                continue
            info = self._read(path)
            if info is not None:
                entries.append((path.stem, info))
        return entries

    def ids(self) -> list[str]:
        """Saved clone ids, sorted. Unreadable files are skipped, never raised."""
        return [voice_id for voice_id, _ in self.listing()]

    def exists(self, voice_id: str) -> bool:
        """True only for a valid id backed by a file; never raises."""
        return bool(VOICE_ID.fullmatch(voice_id)) and self._path(voice_id).is_file()

    def save(self, voice_id: str, voice) -> int:
        """Atomically write a clone: temp file in the same dir, then `os.replace`.

        The final path only ever holds a complete file, so a concurrent reader
        (or a crash mid-write) never sees a partial `.npz`. Returns the clone's
        `created` timestamp — the file's mtime, read from the temp file before
        the rename publishes it, so a concurrent delete can never make the
        caller's read fail.
        """
        path = self._path(voice_id)  # validates the id before touching any file
        tmp = self.root / f".{voice_id}.{uuid.uuid4().hex}.tmp.npz"
        try:
            voice.save(tmp)  # np.savez keeps the .npz suffix exactly as given
            created = int(tmp.stat().st_mtime)  # rename preserves this mtime
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return created

    def load(self, voice_id: str):
        from reverb_tts import Voice

        return Voice.load(self._path(voice_id))

    def info(self, voice_id: str) -> dict:
        result = self._read(self._path(voice_id))
        if result is None:
            raise ValueError(f"voice {voice_id!r} is corrupt or unreadable")
        return result

    def delete(self, voice_id: str) -> bool:
        """True when a saved clone was there to remove.

        A delete that loses a race to another delete — the file is already gone
        when this request unlinks it — still reports success: the request asked
        for the clone to be gone, and it is.
        """
        if not self.exists(voice_id):
            return False
        self._path(voice_id).unlink(missing_ok=True)
        return True
