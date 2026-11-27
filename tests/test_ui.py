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


# --- default voice and language -----------------------------------------


def _function_body(js: str, name: str) -> str:
    """The text of the top-level ``function name(...)`` up to its closing brace."""
    start = js.index(f"function {name}(")
    end = js.index("\n}", start)
    return js[start:end]


def test_default_voice_and_language_constants():
    # The page opens on the API's default voice and on English, not on
    # whichever option a rebuilt <select> happens to land on ("de" sorts first).
    js = _read("app.js")
    assert re.search(r'DEFAULT_VOICE\s*=\s*"en_male_narrator_1"', js)
    assert re.search(r'DEFAULT_LANGUAGE\s*=\s*"en"', js)


def test_fill_languages_keeps_the_current_choice():
    body = _function_body(_read("app.js"), "fillLanguages")
    # Reads the value before the rebuild and keeps it when still offered...
    assert "select.value" in body
    assert re.search(r"languages\.includes\(\s*current\s*\)", body)
    # ...else falls back to the default language, then the first option.
    assert re.search(r"languages\.includes\(\s*DEFAULT_LANGUAGE\s*\)", body)
    assert re.search(r"languages\[\s*0\s*\]", body)


def test_fill_voices_keeps_the_current_voice_and_defaults_to_the_api_default():
    body = _function_body(_read("app.js"), "fillVoices")
    # Keeps the selected voice when it is still in the list...
    assert "select.value" in body
    assert re.search(r"ids\.includes\(\s*current\s*\)", body)
    # ...else falls back to the API's default voice, then the first option...
    assert re.search(r"ids\.includes\(\s*DEFAULT_VOICE\s*\)", body)
    assert re.search(r"ids\[\s*0\s*\]", body)
    # ...and the language follows the chosen voice.
    assert "syncSpeakLanguage()" in body


# --- VoxCPM2 engine selector and controls ---------------------------------


def _element(html: str, element_id: str) -> str:
    """The opening tag of the element carrying ``id="<element_id>"``."""
    match = re.search(r'<[a-z]+[^>]*\bid="%s"[^>]*>' % re.escape(element_id), html)
    assert match, f'no element with id="{element_id}"'
    return match.group(0)


def _select_options(html: str, select_id: str) -> list[tuple[str, str]]:
    match = re.search(
        r'<select id="%s">(.*?)</select>' % re.escape(select_id), html, re.DOTALL
    )
    assert match, f'no <select id="{select_id}">'
    return re.findall(r'<option value="([^"]+)"([^>]*)>', match.group(1))


def test_engine_selectors_exist_and_default_to_reverb():
    html = _read("index.html")
    for select_id in ("speak-engine", "clone-engine"):
        options = _select_options(html, select_id)
        assert [value for value, _ in options] == ["reverb-tts", "voxcpm2"]
        assert "selected" in options[0][1], f"{select_id}: reverb-tts is the default"
        assert "selected" not in options[1][1], f"{select_id}: only one default"
    js = _read("app.js")
    assert re.search(r'ENGINE_REVERB\s*=\s*"reverb-tts"', js)
    assert re.search(r'ENGINE_VOXCPM2\s*=\s*"voxcpm2"', js)


def test_cfg_control_bounds_and_default():
    html = _read("index.html")
    tag = _element(html, "speak-cfg")
    assert 'type="range"' in tag
    assert 'min="1.0"' in tag
    assert 'max="5.0"' in tag
    assert 'step="0.1"' in tag
    assert 'value="2"' in tag
    assert 'id="speak-cfg-value"' in html
    # The display follows the slider.
    body = _function_body(_read("app.js"), "syncCfgDisplay")
    assert "speak-cfg-value" in body
    assert "toFixed(1)" in body


def test_design_and_style_controls_are_present_and_bounded():
    html = _read("index.html")
    design = _element(html, "speak-design")
    style = _element(html, "speak-style")
    assert design.startswith("<textarea")
    assert style.startswith("<textarea")
    assert 'maxlength="300"' in design
    assert 'maxlength="200"' in style
    # VoxCPM2-only: hidden while Reverb is selected.
    assert "hidden" in _element(html, "speak-voxcpm2-controls")


def test_engine_switch_shows_the_voxcpm2_controls_and_hides_language():
    body = _function_body(_read("app.js"), "applySpeakEngine")
    assert "ENGINE_VOXCPM2" in body
    assert "speak-voxcpm2-controls" in body
    assert "hidden" in body
    assert "speak-language-label" in body
    assert "fillVoices(voices)" in body


def test_voice_list_is_filtered_to_the_selected_engine():
    body = _function_body(_read("app.js"), "fillVoices")
    # The synthetic default voice comes first under voxcpm2...
    assert '"default"' in body
    # ...and only the selected engine's objects are added.
    assert "ENGINE_VOXCPM2" in body
    assert "v.engine" in body


def test_loading_state_polls_readiness_at_five_seconds():
    js = _read("app.js")
    assert re.search(r"LOAD_POLL_MS\s*=\s*5000", js)
    body = _function_body(js, "pollWhileLoading")
    # One loop, never two.
    assert re.search(r"if \(loadPollTimer !== null\) return;", body)
    # Reads readiness through the voxcpm2 state reader...
    assert "voxcpm2State()" in body
    # ...keeps polling while the engine reports loading...
    assert re.search(r'state === "loading"', body)
    assert "setTimeout(tick, LOAD_POLL_MS)" in body
    # ...and stops on every other status.
    assert re.search(r'if \(state !== "ready"\) return;', body)


def test_the_poll_token_is_held_across_the_readiness_probe():
    # A request held while the readiness probe is in flight must join the
    # running loop; releasing the token before the await lets it start a second
    # loop — duplicate polls and, once ready, a duplicate status refresh.
    body = _function_body(_read("app.js"), "pollWhileLoading")
    probe = body.index("await voxcpm2State()")
    release = body.index("loadPollTimer = null")
    assert release > probe
    # The release is the loop ending, before the pending request is taken, so a
    # resend that reports loading again can start the next loop.
    assert release < body.index("const pending = pendingRequest;")


def test_readiness_reader_returns_the_voxcpm2_state():
    body = _function_body(_read("app.js"), "voxcpm2State")
    assert "/health/ready" in body
    assert "body.voxcpm2" in body


def test_pending_request_is_taken_once_and_resent_once():
    js = _read("app.js")
    body = _function_body(js, "pollWhileLoading")
    taken = body.index("const pending = pendingRequest;")
    cleared = body.index("pendingRequest = null;", taken)
    sent = body.index("sendPending(pending)", cleared)
    assert taken < cleared < sent, "take, clear, then send — one resend only"
    send = _function_body(js, "sendPending")
    assert "runSpeech(pending.payload)" in send
    assert "runCloneSave(pending.payload)" in send


def test_both_panels_hold_the_request_while_the_engine_loads():
    js = _read("app.js")
    for name in ("speak", "saveClone"):
        body = _function_body(js, name)
        assert "holdPending(" in body, name
        assert re.search(r'err\.voxcpm2State === "loading"', body), name
    held = _function_body(js, "holdPending")
    assert "pendingRequest = request" in held
    assert "pollWhileLoading()" in held


def test_voxcpm2_503_messages_are_the_pages_own_table():
    js = _read("app.js")
    for message in (
        "VoxCPM2 is disabled — enable it in the server config and restart.",
        "VoxCPM2 is still loading (first start downloads ~4.7 GB of weights) — "
        "check back in a minute.",
        "VoxCPM2 encountered an error and is unavailable — check the server logs.",
        "VoxCPM2 could not generate a good take for this input — try shorter text "
        "or a different seed.",
    ):
        assert message in js, message
    message_body = _function_body(js, "voxcpm2Message")
    for state in ('"disabled"', '"error"', '"ready"'):
        assert state in message_body, state
    # The 503 mapper decides from the readiness state, never from the API body.
    api_body = _function_body(js, "api")
    assert "voxcpm2State()" in api_body
    assert "voxcpm2Message(state)" in api_body
    # A 400 still shows the API's own message, verbatim.
    assert "body.error.message" in api_body


def test_speech_payload_sends_voxcpm2_fields_only_for_voxcpm2():
    body = _function_body(_read("app.js"), "speechPayload")
    assert body.index("ENGINE_VOXCPM2") < body.index("ENGINE_REVERB")
    voxcpm2_part, reverb_part = body.split("ENGINE_REVERB", 1)
    for field in ("cfg", "design", "style", '"default"'):
        assert field in voxcpm2_part, field
    # Neither engine switch nor Reverb's body carries VoxCPM2-only fields.
    for field in ("speed", "language"):
        assert field not in voxcpm2_part, field
    for field in ("cfg", "design", "style", "speed"):
        assert field not in reverb_part, field
    assert "language" in reverb_part
    assert 'response_format: "wav"' in reverb_part


def test_clone_panel_engine_selector_and_optional_transcript():
    html = _read("index.html")
    assert 'id="clone-text-label"' in html
    assert 'id="clone-language-label"' in html
    js = _read("app.js")
    assert 'const TRANSCRIPT_OPTIONAL = "Transcript (optional for VoxCPM2)";' in js
    switch = _function_body(js, "applyCloneEngine")
    assert "TRANSCRIPT_OPTIONAL" in switch
    assert "clone-language-label" in switch
    assert "hidden" in switch
    save = _function_body(js, "saveClone")
    # The transcript is required for Reverb and optional for VoxCPM2.
    assert re.search(r"engine !== ENGINE_VOXCPM2 && !text", save)
    # The engine rides the multipart as its model field, for both engines.
    assert "model: engine" in save


def test_clone_list_shows_each_engine_and_deletes_both():
    body = _function_body(_read("app.js"), "renderClones")
    assert "voice.engine" in body
    assert "deleteVoice(voice.id)" in body
    # No engine filter: every clone gets its delete button.
    assert "ENGINE_" not in body


def test_engine_selector_hides_voxcpm2_when_disabled():
    """While readiness reports `disabled`, neither selector offers voxcpm2."""
    js = _read("app.js")
    availability = _function_body(js, "applyEngineAvailability")
    assert 'state === "disabled"' in availability
    for select_id in ("speak-engine", "clone-engine"):
        assert select_id in availability, select_id
    # The option is removed from the list, not merely marked: hidden and
    # disabled both follow the reported state.
    assert "option.disabled" in availability
    assert "option.hidden" in availability
    assert "ENGINE_VOXCPM2" in availability
    # The rule is applied to what `/health/ready` reports.
    status = _function_body(js, "refreshStatus")
    assert "applyEngineAvailability(body.voxcpm2" in status


# --- the listen request names its model ----------------------------------


def test_listen_sends_moonshine_for_short():
    js = _read("app.js")
    assert re.search(r"LISTEN_MOONSHINE_MAX_SECONDS\s*=\s*10\b", js)
    body = _function_body(js, "runTranscription")
    # The listen request carries the model, and the short arm is Moonshine.
    assert re.search(r'form\.append\(\s*"model"\s*,', body)
    assert re.search(
        r'LISTEN_MOONSHINE_MAX_SECONDS\s*\?\s*"moonshine-base"\s*:\s*"whisper-1"',
        body,
    )


def test_listen_sends_whisper_for_long():
    body = _function_body(_read("app.js"), "runTranscription")
    # The same expression's other arm sends the long recording to Whisper, and
    # the choice is made from the recording's duration.
    assert '"whisper-1"' in body
    assert re.search(r"\bduration\b", body)
