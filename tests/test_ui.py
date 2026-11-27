"""Static UI shell — the page, its assets, its headers, and the source-level
guarantees the CSP depends on.

The page is not executed here: the repo ships no JS harness and the workers
have no browser. These tests read the shipped source as text and pin the
contract the orchestrator re-checks in a browser at landing.
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import speakeasy.server as server

STATIC = Path(server.__file__).parent / "static"

# The exact string `GET /` must return.
CSP = (
    "default-src 'self'; media-src 'self' blob:; connect-src 'self'; "
    "img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'"
)

FILES = ("index.html", "app.js", "style.css")


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server, "whisper", None)
    monkeypatch.setattr(server, "moonshine", None)
    monkeypatch.setattr(server, "startup_error", None)
    monkeypatch.setattr(server.settings, "api_key", "")
    return TestClient(server.app)


# --- UI -----------------------------------------------------------------


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "speakeasy" in r.text
    assert r.headers["content-security-policy"] == CSP
    # The CSP plus the two companion headers.
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"


def test_static_assets(client):
    js = client.get("/static/app.js")
    assert js.status_code == 200
    js_type = js.headers["content-type"].split(";")[0]
    assert js_type in ("text/javascript", "application/javascript")
    css = client.get("/static/style.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")


def test_static_traversal_blocked(client):
    assert client.get("/static/../pyproject.toml").status_code == 404
    assert client.get("/static/..%2fpyproject.toml").status_code == 404


def test_root_and_static_require_no_auth(monkeypatch):
    # `/` and `/static/*` stay open even when a key is configured.
    monkeypatch.setattr(server.settings, "api_key", "unit-test-key")
    c = TestClient(server.app)
    assert c.get("/").status_code == 200
    assert c.get("/static/app.js").status_code == 200
    assert c.get("/static/style.css").status_code == 200


# --- source-level guarantees ------------------------

INLINE_HANDLER = re.compile(
    r"\son(?:click|dblclick|change|input|submit|reset|load|error|focus|blur|"
    r"keydown|keyup|keypress|mousedown|mouseup|mousemove|mouseover|mouseout|"
    r"touchstart|touchend|touchmove|pointerdown|pointerup|contextmenu|scroll)"
    r"\s*=",
    re.IGNORECASE,
)
INLINE_STYLE_ATTR = re.compile(r"\sstyle\s*=", re.IGNORECASE)
SCRIPT_TAG = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.IGNORECASE | re.DOTALL)
REMOTE_URL = re.compile(r"https?://", re.IGNORECASE)


def test_static_source_is_csp_safe():
    # No innerHTML, no remote URL, no inline style attribute, and no inline
    # event handler anywhere in the three shipped files.
    for name in FILES:
        text = _read(name)
        assert "innerHTML" not in text, f"{name}: innerHTML is forbidden"
        assert not REMOTE_URL.search(text), f"{name}: remote URL is forbidden"
        assert not INLINE_STYLE_ATTR.search(text), f"{name}: inline style= forbidden"
        assert not INLINE_HANDLER.search(text), f"{name}: inline handler forbidden"
    # index.html carries no <style> block and every <script> is external.
    html = _read("index.html")
    assert "<style" not in html.lower(), "index.html: inline <style> block forbidden"
    for attrs, body in SCRIPT_TAG.findall(html):
        assert "src=" in attrs.lower(), f"inline <script> without src: {attrs!r}"
        assert body.strip() == "", "inline <script> body is forbidden"


def test_page_implements_three_panels_and_states():
    html = _read("index.html")
    js = _read("app.js")
    # Three panels, each with its own error line.
    for panel in ("listen", "speak", "clone"):
        assert f'id="panel-{panel}"' in html
        assert f'id="{panel}-error"' in html
    assert html.count('class="error"') >= 3
    # The mic-unavailable path says why, and upload still works.
    assert "mic needs HTTPS or localhost" in js
    assert 'type="file"' in html
    # Every state message is present, plus the in-flight label.
    for message in (
        "Wrong or missing API key",
        "TTS is still loading",
        "TTS is not available on this server.",
        "TTS encountered an error",
        "Could not reach the server",
        "working",
    ):
        assert message in js, message


def test_mic_capture_is_16k_mono_wav():
    js = _read("app.js")
    # Mono WAV at 16 kHz, with the canonical 16-bit PCM header fields.
    assert "16000" in js
    assert "OfflineAudioContext" in js
    for field in ("RIFF", "WAVE", "fmt ", "data"):
        assert f'"{field}"' in js
    assert "setInt16" in js


def test_api_key_uses_sessionstorage_and_bearer():
    js = _read("app.js")
    html = _read("index.html")
    assert "sessionStorage" in js
    assert "Bearer" in js
    assert 'type="password"' in html
    # Never in a URL: no query-string or fragment key handling.
    assert "?key=" not in js
    assert "location.search" not in js
    assert "location.hash" not in js


def test_voice_selects_fill_from_voices_endpoint():
    js = _read("app.js")
    assert "/v1/audio/voices" in js
    assert "languages" in js
    # Presets and saved clones populate the select through optgroups.
    assert "optgroup" in js


def test_503_is_endpoint_aware():
    """A Whisper-side 503 must not render as a TTS error (round-1 finding 1).

    The transcription endpoint 503s from Whisper (failed to load, or still
    loading), so only the TTS endpoints may map a 503 through the TTS reason
    table. The page is not executed here, so this pins the shipped
    source the way the rest of this file does.
    """
    js = _read("app.js")
    # The round-1 bug: every 503 went straight into the TTS reason mapper.
    assert "throw new Error(await ttsReason(response))" not in js
    # The 503 branch now dispatches on the request path...
    assert "isTtsPath(path)" in js
    assert "sttReason(response)" in js
    # ...and the TTS path set is exactly the two TTS endpoints.
    match = re.search(r"TTS_PATHS\s*=\s*\[(.*?)\]", js, re.DOTALL)
    assert match, "TTS_PATHS not found in app.js"
    tts_paths = match.group(1)
    assert "/v1/audio/speech" in tts_paths
    assert "/v1/audio/voices" in tts_paths
    assert "transcriptions" not in tts_paths
    # The Whisper side carries its own, non-TTS line.
    assert "The speech models are still loading" in js
