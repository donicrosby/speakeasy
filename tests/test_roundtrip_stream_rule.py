"""The streamed-take equivalence rule, pinned.

`scripts/tts_roundtrip.py` is not installed and nothing imports it, so it is
loaded from its file here to reach the rule directly. The script imports only
the standard library and runs nothing at import.
"""

from __future__ import annotations

import array
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tts_roundtrip.py"
_spec = importlib.util.spec_from_file_location("tts_roundtrip", _SCRIPT)
assert _spec is not None and _spec.loader is not None
tts_roundtrip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tts_roundtrip)


def _take(*samples: int) -> array.array:
    """A hand-made take of 16-bit samples."""
    return array.array("h", samples)


def test_one_step_apart_is_the_same_take():
    """One sample a single step away is rounding, not a different take."""
    a = _take(0, 1000, -1000, 32767)
    b = _take(0, 1001, -1000, 32767)
    assert tts_roundtrip._same_take(a, b)


def test_two_steps_apart_is_not_the_same_take():
    """A sample two steps away breaks the rule."""
    a = _take(0, 1000, -1000, 32767)
    b = _take(0, 1002, -1000, 32767)
    assert not tts_roundtrip._same_take(a, b)


def test_a_length_difference_is_not_the_same_take():
    """Takes of different length are never the same take."""
    a = _take(0, 1000, -1000)
    b = _take(0, 1000)
    assert not tts_roundtrip._same_take(a, b)


def test_stream_tolerance_is_one_step():
    """The tolerance is exactly one step of the delivered 16-bit format."""
    assert tts_roundtrip.STREAM_TOLERANCE == 1 / 32768
