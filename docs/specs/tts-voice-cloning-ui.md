# Speakeasy TTS, voice cloning and web UI

> Spec v1.0. Validated against `reverb-tts 1.0.1`, spike 004 (gfx1151), speakeasy main `5700710`.

## 1. Goals and non-goals

### Goals

Serve Reverb-TTS from speakeasy — preset voices, zero-shot voice clones saved from a reference clip, and an OpenAI-compatible `POST /v1/audio/speech`. Ship a single static web page at `/` that demos the round trip: record or upload → transcribe → speak the text in a preset or a cloned voice.

### Non-goals

- Audio streaming (chunked transfer, Server-Sent Events).
- SSML markup.
- True playback-rate control — `speed` is a length factor, not a time-stretch.
- Voice training, fine-tuning, or speaker adaptation beyond the library's zero-shot `voice()`.
- User accounts, per-client API keys, or multi-tenancy beyond the existing single bearer token.
- Opus or OGG output — `response_format` covers wav, mp3, flac, pcm.
- A UI framework (React, Vue, etc.), a build step, or a bundler.
- Gradio integration.
- CUDA runtime verification — the ROCm host covers it; CUDA builds pass the import sanity check only.
- Any change to the existing transcription endpoints' behaviour, response shape, or engine routing.

## 2. Owner's decisions

Copied from the brief's decisions table (rows 1, 2, 4, 5, 6, 7, 8). These are settled; the spec restates them and does not reopen them.

| #  | Decision                | Answer                                                                                                                                                                                                                |
|----|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | UI tech                 | Single static page served by speakeasy — vanilla HTML/JS/CSS at `/`, no build step, no new deps.                                                                                                                      |
| 2  | Cloned voices           | Saved on disk + list/delete API — persist in the `/models` volume as `.npz`, survive restarts.                                                                                                                        |
| 4  | Process model           | TTS in the server process (not a worker) — set by the coexistence smoke below, not a preference.                                                                                                                      |
| 5  | `/v1/audio/speech` shape | OpenAI JSON body (`model`, `input`, `voice`, `response_format`, `speed`) + two extensions: `language`, `seed`. Agent decision, overridable.                                                                           |
| 6  | Cloning API             | `POST /v1/audio/voices` multipart (`file`, `text`, `language`, optional `voice_id`). OpenAI has no clone endpoint; this is an extension. Agent decision.                                                              |
| 7  | `response_format` default | `mp3`, same as OpenAI; the UI asks for `wav`. Agent decision.                                                                                                                                                       |
| 8  | TTS load                | Background thread at startup; endpoints answer 503 with a reason until ready. Agent decision.                                                                                                                         |

## 3. Verified facts

### VERIFIED — `reverb-tts 1.0.1`

- **Weights revision.** `tts.py:29` — `REVISION = "v1.0.0"`; the default tag for `software-mansion/Reverb-TTS`.
- **Preset count.** 16 preset voices (`spike/004-reverb-tts-rocm/results/result.json:presets`); `preset_voices()` enumerates them (`tts.py:156-164`); `preset_voice(name)` raises `ValueError` on unknown (`tts.py:180-181`).
- **Voice save/load.** `Voice.save(path)` → `np.savez(path, codes=…int16, text=…, language=…)` (`tts.py:67-73`). `Voice.load(path)` reads `.npz` with `codes` (int64, `[8, frames]`), `text`, `language` (`tts.py:83-87`).
- **`generate` return type.** `np.ndarray` float32 at the model's sample rate (`tts.py:268` — `return out[0] if single else out`).
- **Sample rate.** `tts.py:116-117` — `self.codec.config.sample_rate`; measured 24000 on gfx1151.
- **`speed` semantics.** `tts.py:228` — "a factor on the estimated length", not a time-stretch.
- **Languages.** `text.py:14-22` — `LANGUAGES: Dict[str, str]` lists 7: de, en, es, fr, it, pl, pt. `check_language(language)` raises `ValueError` (`text.py:28-31`).
- **Reference decode (lib path).** `audio.py:34-36` — `sf.read(str(audio), dtype="float32", always_2d=True)` reads via libsndfile (WAV, FLAC, MP3; no Opus).
- **`TTS.from_pretrained` signature.** `tts.py:120-154` — `(name, *, variant="int8", revision=None, device=None, dtype=None, sampler=None, codec=CODEC, codec_revision=None)`.
- **`voice()` signature.** `tts.py:183-204` — `(audio, text, language, *, sample_rate=None, prepare=True)`. Accepts a path or float32 array.
- **Variants.** `tts.py:32` — `VARIANTS = ("fp32", "fp16", "int8", "int4")`; "every variant … download[s] faster but run[s] at the same speed" (`tts.py:140-141`).

### VERIFIED — speakeasy commit `5700710`

- **Auth.** `server.py:41-53` — `require_auth` uses `secrets.compare_digest`; bearer token on `/v1/*` when `SPEAKEASY_API_KEY` is set; health probes stay open.
- **Error shape.** `server.py:159-161` — `JSONResponse(status_code=503, content={"error": {"message": …}})`.
- **Config prefix.** `config.py:10` — `SettingsConfigDict(env_prefix="SPEAKEASY_")`.
- **CI test job.** `test.yml:22` — `uv sync --all-groups` *(no extras)*; any new test must pass without the `tts` extra installed.
- **Docker sanity.** `Dockerfile.cuda:49`, `Dockerfile.rocm:167` — both import `reverb_tts` in their final sanity step.
- **Existing endpoints unaffected.** `server.py:143-235` — `/v1/audio/transcriptions`; `responses.py` defines `FORMATS`; no existing test file is modified.

### VERIFIED — spike 004 (gfx1151 ROCm)

- **Seed reproducibility.** `spike/004-reverb-tts-rocm/results/result.json:seed_reproducible` — `true`.
- **GPU memory.** `result.json:gpu_mem_peak_mb` — 1984 MB peak; after load 1430 MB.
- **MIOpen.** `torch.backends.cudnn.enabled = False` required on HIP; find mode costs 1.5–19 s per new output length and intermittently raises `miopenStatusUnknownError`.

### ASSUMED

- **CUDA TTS at runtime.** The CUDA image's sanity step imports `reverb_tts` against cu130 torch, but no NVIDIA host has run inference. **Settled by:** orchestrator runs `scripts/tts_roundtrip.py` on an NVIDIA host at landing; if it fails the PR ships with a CUDA limitation note.
- **`speed` quality at extremes (0.25, 4.0).** Unverified. **Settled by:** Task 12 on the AMD host reports audio duration and transcript similarity; if quality degrades badly, clamp to a documented range in a follow-up.
- **Clone quality after PyAV resample (24 kHz swresample).** The server decodes uploads with `faster_whisper.audio.decode_audio` (PyAV) rather than the library's own `soundfile` path (kaiser-best). **Settled by:** Task 12 step 4 (clone → speak → transcribe → similarity ≥ 0.8).

## 4. HTTP contract

All `/v1/*` endpoints require the bearer token (existing `require_auth`). `/health`, `/health/ready`, `GET /`, `/static/*` stay open.

### 4.1 `GET /v1/audio/voices`

**Auth:** bearer token. **TTS state:** 503 when `tts is None`.

**Response 200:**

```json
{
  "object": "list",
  "languages": ["de", "en", "es", "fr", "it", "pl", "pt"],
  "data": [
    {
      "id": "en_male_narrator_1",
      "object": "voice",
      "type": "preset",
      "language": "en"
    },
    {
      "id": "clone-a1b2c3",
      "object": "voice",
      "type": "clone",
      "language": "en",
      "transcript": "exact reference text",
      "created": 1700000000
    }
  ]
}
```

- `data`: presets first (sorted by id), then clones (sorted by id). Presets carry `type: "preset"` and `language`. Clones carry `type: "clone"`, `language`, `transcript`, `created` (Unix seconds, from file mtime).
- `languages`: the top-level key is the engine's `languages` list so the UI never hardcodes them.

**Response 401** (wrong/missing key): `{"detail": "Invalid or missing API key"}` — existing FastAPI `HTTPException` shape.

**Response 503** (TTS loading/disabled): `{"error": {"message": "tts is still loading (first start downloads about 1 GB of weights)"}}` or `"…tts is disabled…"`.

### 4.2 `POST /v1/audio/speech`

**Auth:** bearer token. **TTS state:** 503 when `tts is None`.

**Request body** (`application/json`):

| Field              | Type                              | Default                  | Validation                                                   |
|--------------------|-----------------------------------|--------------------------|--------------------------------------------------------------|
| `input`            | `str`                             | *(required)*             | Min length 1 (non-whitespace), max 4096 chars.               |
| `voice`            | `str`                             | `"en_male_narrator_1"`   | A preset id or a saved clone id.                             |
| `response_format`  | `"wav"` \| `"mp3"` \| `"flac"` \| `"pcm"` | `"mp3"` | —                                                            |
| `speed`            | `float`                           | `1.0`                    | `0.25 ≤ speed ≤ 4.0`. A length factor, not a time-stretch.  |
| `model`            | `str`                             | `""`                     | Accepted for OpenAI clients; the server has one TTS model.   |
| `language`         | `str` \| `null`                   | `null`                   | One of the 7 supported codes. Defaults to the voice's language. |
| `seed`             | `int` \| `null`                   | `null`                   | Reproducible take when set; drawn randomly when null.        |

**Response 200:**

- Body: the encoded audio bytes.
- Headers:
  - `Content-Type`: `audio/mpeg` (mp3), `audio/wav` (wav), `audio/flac` (flac), `audio/pcm` (pcm).
  - `X-Speakeasy-Engine`: `"reverb-tts"`.
  - `X-Speakeasy-Elapsed`: float seconds of inference time.
  - `X-Speakeasy-Audio-Duration`: float seconds of output audio.
  - `X-Speakeasy-RTF`: real-time factor (`elapsed / duration`).

**Response 400:** `{"error": {"message": "…"}}` — unknown voice, unsupported language, empty input.

**Response 422:** FastAPI validation error (`speed` out of range, `response_format` not in the enum) — existing Pydantic shape.

**Response 401, 503:** as in 4.1.

### 4.3 `POST /v1/audio/voices`

**Auth:** bearer token. **TTS state:** 503 when `tts is None`.

**Request body** (`multipart/form-data`):

| Field       | Type     | Default                | Validation                                                                   |
|-------------|----------|------------------------|------------------------------------------------------------------------------|
| `file`      | file     | *(required)*           | Reference audio recording, max 20 MB.                                        |
| `text`      | `str`    | *(required)*           | The reference's exact transcript — non-empty after stripping.                |
| `language`  | `str`    | `"en"`                 | One of the 7 supported codes.                                                |
| `voice_id`  | `str`    | `clone-<hex8>`         | `^[a-z0-9][a-z0-9-]{1,63}$`. Auto-generated when omitted.                   |

**Response 201:**

```json
{
  "id": "clone-a1b2c3",
  "object": "voice",
  "type": "clone",
  "language": "en",
  "transcript": "exact reference text",
  "created": 1700000000
}
```

**Response 400:** `{"error": {"message": "…"}}` — text empty; unsupported language; invalid `voice_id`; reference under 1 s or over 30 s ("reference is X.Xs; give it 3-10 s of speech (1-30 accepted)"); could not decode the reference audio (message fixed, exception text never leaked).

**Response 409:** `{"error": {"message": "…"}}` — `voice_id` collides with a preset or an existing clone.

**Response 413:** `{"error": {"message": "reference audio over 20 MB"}}` — the raw upload exceeds 20 MB before any decode attempt.

**Response 401, 422, 503:** as in 4.1.

### 4.4 `DELETE /v1/audio/voices/{voice_id}`

**Auth:** bearer token. **TTS state:** 503 when `tts is None`.

**Response 200:**

```json
{"id": "clone-a1b2c3", "object": "voice", "deleted": true}
```

**Response 404:** `{"error": {"message": "no saved voice '…'"}}` — not a saved clone (unknown id, preset, or invalid id).

**Response 401, 503:** as in 4.1.

### 4.5 Additions to existing endpoints

**`GET /health/ready`** gains `"tts": tts_state` (`"disabled"` | `"loading"` | `"ready"` | `"error"`) in its JSON body. Existing `status`/`device`/`models` keys unchanged.

**`GET /v1/models`** appends `{"id": "reverb-tts", "object": "model", "created": 0, "owned_by": "speakeasy"}` when `tts is not None`. Existing entries unchanged.

## 5. On-disk contract

- **Directory:** `SPEAKEASY_VOICES_DIR` when set; otherwise `<hf_home or ~/.cache/speakeasy>/voices`.
- **File:** `<voice_id>.npz`, written by `reverb_tts.Voice.save()` (`tts.py:67-73`).
- **Contents:** `codes` (int16 `[8, frames]`), `text` (the transcript string), `language` (language code string), optional `speech_frames` (int).
- **Voice id regex:** `^[a-z0-9][a-z0-9-]{1,63}$`. An id is validated before it ever becomes a path; invalid ids are rejected, never sanitized.
- **Mode:** the voices directory is created with `0o700` at store init.

## 6. Security requirements

Each requirement is named and testable.

1. **SEC-01 — voice id is validated before any path is built.** `check_voice_id()` enforces the regex; `VoiceStore._path()` calls it first. Test: `test_voice_id_rejects_traversal` (`test_voices.py`).

2. **SEC-02 — clone id collides with preset → 409.** `POST /v1/audio/voices` checks presets before saving. Test: `test_clone_id_preset_collision_returns_409` (`test_tts.py`).

3. **SEC-03 — clone id collides with existing clone → 409.** `VoiceStore.exists()` checked before save. Test: `test_clone_id_duplicate_returns_409` (`test_tts.py`).

4. **SEC-04 — reference upload cap 20 MB → 413.** Raw body read capped at `MAX_REFERENCE_BYTES + 1`; exceeds → 413 before any decode. Test: `test_clone_reference_over_20mb_returns_413` (`test_tts.py`).

5. **SEC-05 — reference duration bounds 1–30 s → 400.** Duration measured after decode; out of range → 400 with the exact duration in the message. Test: `test_clone_reference_too_short_returns_400`, `test_clone_reference_too_long_returns_400` (`test_tts.py`).

6. **SEC-06 — decode failure → 400, no exception text leaked.** Exception caught; message is a fixed string. Test: `test_clone_undecodable_reference_returns_400` (`test_tts.py`).

7. **SEC-07 — one inference at a time.** `ReverbEngine` holds a `threading.Lock`; both `synthesize` and `clone` acquire it. The speech endpoint is `def` (not `async def`), so FastAPI runs it on a threadpool — the lock blocks a worker thread, never the event loop. Test: `test_engine_lock_serialises_calls` (`test_tts.py`).

8. **SEC-08 — model loads off the startup path.** `start_tts()` spawns a daemon thread; readiness checks only Whisper + Moonshine. Test: `test_ready_reports_loading_while_tts_boots` (`test_tts.py`).

9. **SEC-09 — CSP with no `unsafe-inline`.** `GET /` returns `Content-Security-Policy: default-src 'self'; media-src 'self' blob:; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'`. Test: `test_index_has_csp_header` (`test_ui.py`).

10. **SEC-10 — no inline script or style.** The CSP forbids them; the static HTML `/` page has no `<script>` or `<style>` blocks — only `<link rel="stylesheet">` and `<script src="app.js">`. Verify by reading `index.html` source in `test_ui.py`.

11. **SEC-11 — no third-party script.** `default-src 'self'` covers scripts. The page loads only local resources.

12. **SEC-12 — no `innerHTML` with server or user data.** UI uses `textContent` for all dynamic text (transcripts, errors, voice names). Documented in `app.js`; verified by review.

13. **SEC-13 — API key in `sessionStorage`, never in a URL.** `app.js` stores the key in `sessionStorage.getItem("speakeasy_key")`; `api()` helper adds it as an `Authorization` header. No `?key=` or hash parameters. Test: `test_index_has_no_key_in_url` (review check).

14. **SEC-14 — MIOpen flag set only on HIP.** `ReverbEngine.__init__` disables `torch.backends.cudnn.enabled` only when `torch.version.hip is not None`. Test: `test_cudnn_disabled_only_on_hip` (`test_tts.py`).

15. **SEC-15 — no auth on health/root/static.** `/health`, `/health/ready`, `GET /`, `/static/*` have no `Depends(require_auth)`. Test: existing `test_health_probes_stay_open` (`test_auth.py`) + new assertions for `/` and `/static/*` in `test_ui.py`.

## 7. UI behaviour

A single static page at `/` with three panels in a grid. Dark theme, compact. No framework, no build step, no third-party scripts.

### 7.1 Shell (`index.html`)

- A `<header>` with the title "speakeasy", a status line, and an API-key `<input type="password">`.
- Status line reads from `GET /health/ready` on load and every 30 s: device, models loaded, TTS state.
- API key: stored in `sessionStorage` on change; restored on page load. Never in a URL.

### 7.2 Panel: Listen

- **Trigger:** Record ⏺ / Stop buttons, or an Upload `<input type="file">`.
- **Recording:** `navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, …}})`, `MediaRecorder`. Recorded blob decoded to mono WAV at 16 kHz via `AudioContext` + `OfflineAudioContext`. When `navigator.mediaDevices` is undefined (plain HTTP on a LAN IP), the Record buttons are replaced with the text "mic needs HTTPS or localhost — upload a file instead".
- **Request:** `POST /v1/audio/transcriptions` multipart with the WAV blob.
- **Result:** transcript displayed in an editable `<textarea>` (used by Clone panel). Engine name, audio duration, elapsed time, RTF shown below.
- **Edge:** keeps the recording blob for the Clone panel's "use last recording" prefill.

### 7.3 Panel: Speak

- **Controls:** a textarea, a voice `<select>` (optgroups Presets / Saved, populated from `GET /v1/audio/voices`), a language `<select>` (defaults to chosen voice's language), a Speak button.
- **Request:** `POST /v1/audio/speech` JSON with `response_format: "wav"`.
- **Result:** `<audio controls>` from the blob URL, a Download link, audio duration, elapsed time, × real time.
- **Loading state:** button disabled, reads "working…". TTS on CPU can take tens of seconds; other panels stay interactive.

### 7.4 Panel: Clone a voice

- **Controls:** Record/Upload (reuses the Listen panel's recording), a transcript textarea with "use last transcript" button, language `<select>`, an optional voice name input, Save button.
- **Client-side:** enforces 1–30 s; hints 3–10 s for best quality.
- **Request:** `POST /v1/audio/voices` multipart.
- **Result:** new voice appears in the Speak select. Saved voices listed below with Delete buttons → `DELETE /v1/audio/voices/{id}`, confirmed with `window.confirm`.

### 7.5 Error handling

Every panel renders errors in a `<p class="error">` specific to that panel.

| State                        | Handling                                                                                  |
|------------------------------|-------------------------------------------------------------------------------------------|
| 401                          | "Wrong or missing API key — enter it above."                                              |
| 503 (TTS loading)            | "TTS is still loading (first start downloads ~1 GB of weights) — check back in a minute." |
| 503 (TTS disabled)           | "TTS is not available on this server."                                                    |
| Network error / fetch fail   | "Could not reach the server — check the connection."                                      |
| Mic unavailable              | Record buttons replaced with the HTTPS/localhost message (see 7.2).                       |

## 8. Acceptance scenarios

Every scenario runs without a GPU. Tests fake `reverb_tts` through `sys.modules`; no test imports `reverb_tts` for real.

### 8.1 Config

**Scenario: TTS config defaults.** Given no env vars. When `Settings()` is created. Then `tts_enabled` is `True`, `tts_model` is `"software-mansion/Reverb-TTS"`, `tts_variant` is `"int8"`, `tts_device` is `"auto"`, `voices_dir` is `None`. **Test:** `test_tts_defaults` (`tests/test_config.py`).

**Scenario: voices dir follows HF_HOME.** Given `SPEAKEASY_VOICES_DIR` unset and `hf_home="/models"`. When `voices_dir_default()` is called. Then it returns `"/models/voices"`. **Test:** `test_voices_dir_follows_hf_home` (`tests/test_config.py`).

### 8.2 Engine

**Scenario: engine loads, presets and synthesize.** Given a fake `reverb_tts` module. When `ReverbEngine` is constructed with `device="cuda"`. Then `name` is `"reverb-tts"`, `sample_rate` is 24000, `languages` is `["de", "en"]`, `presets()` returns preset dicts. `synthesize("Hi.", …)` returns float32 audio and elapsed ≥ 0. **Test:** `test_engine_loads_presets_and_speaks` (`tests/test_tts.py`).

**Scenario: engine clones from samples.** Given a fake engine. When `clone(samples, 24000, "text", "de")` is called. Then the returned voice has `language="de"` and `FakeTTS.calls` records the clone. **Test:** `test_engine_clones_from_samples` (`tests/test_tts.py`).

**Scenario: cudnn disabled only on HIP.** Given `torch.version.hip` set. When `ReverbEngine` is constructed. Then `torch.backends.cudnn.enabled` is `False`. When `hip` is `None`, `cudnn.enabled` is left alone. **Test:** `test_cudnn_disabled_only_on_hip` (`tests/test_tts.py`).

### 8.3 Voice store

**Scenario: valid id round-trip.** Given a valid id `"my-voice-1"`. When saved, loaded, read via `info()`, listed via `ids()`, then deleted. Then all operations succeed and `exists()` returns `False` after delete. **Test:** `test_voice_store_roundtrip` (`tests/test_voices.py`).

**Scenario: invalid ids rejected.** Given ids `"../etc/passwd"`, `"En_X"` (uppercase), `"a"` (too short), 65 chars (too long). When `check_voice_id()` is called or they are passed to `VoiceStore`. Then `ValueError` is raised; `exists()` returns `False` without raising. **Test:** `test_invalid_voice_ids_rejected` (`tests/test_voices.py`).

**Scenario: delete unknown returns False.** Given an id that was never saved. When `delete()` is called. Then it returns `False`. **Test:** `test_delete_unknown_returns_false` (`tests/test_voices.py`).

### 8.4 Lifespan and state

**Scenario: readiness reports TTS state.** With `tts=None` and `tts_state="disabled"`. When `GET /health/ready` is called. Then the response includes `"tts": "disabled"`. **Test:** `test_ready_reports_tts_state` (`tests/test_tts.py`).

**Scenario: models lists reverb-tts when loaded.** With `tts` monkeypatched to a stub. When `GET /v1/models` is called. Then `"reverb-tts"` appears in the model ids. **Test:** `test_models_lists_reverb_tts` (`tests/test_tts.py`).

**Scenario: models omits reverb-tts when unloaded.** With `tts=None`. When `GET /v1/models` is called. Then `"reverb-tts"` is absent. **Test:** `test_models_omits_tts_when_none` (`tests/test_tts.py`).

### 8.5 Voice list

**Scenario: list voices returns presets.** Given a stub TTS engine and a `VoiceStore(tmp_path)` with one saved clone. When `GET /v1/audio/voices` is called with a valid key. Then response is 200, `data` contains presets with `type: "preset"` and clones with `type: "clone"`, `transcript`, `created`. **Test:** `test_list_voices` (`tests/test_tts.py`).

**Scenario: list voices 503 when TTS is None.** With `tts=None`. When `GET /v1/audio/voices` is called. Then response is 503. **Test:** `test_list_voices_503_when_tts_none` (`tests/test_tts.py`).

**Scenario: list voices 401 without key.** With `settings.api_key` set. When called without `Authorization`. Then response is 401. **Test:** `test_list_voices_401` (`tests/test_tts.py`).

### 8.6 Speech

**Scenario: speech returns audio in the requested format.** With a stub engine returning 0.5 s of sine. When `POST /v1/audio/speech` is called with `{"input": "Hi"}`. Then response is 200, `Content-Type: audio/mpeg`, body starts with MP3 magic. Wav → `RIFF`, flac → `fLaC`, pcm → 24000 bytes with `audio/pcm`. **Test:** `test_speech_response_formats` (`tests/test_tts.py`).

**Scenario: speech response headers.** With a stub engine. When speech is generated. Then response includes `X-Speakeasy-Engine: reverb-tts`, `X-Speakeasy-Audio-Duration`, `X-Speakeasy-RTF`. **Test:** `test_speech_response_headers` (`tests/test_tts.py`).

**Scenario: speech language defaults to voice language.** With a stub that records the call. When `POST /v1/audio/speech` is called without `language`. Then the stub received the preset's language. When called with an explicit `language`, that wins. **Test:** `test_speech_language_default` (`tests/test_tts.py`).

**Scenario: speech 400 on unknown voice, bad language, empty input.** Each returns 400 with a message. **Test:** `test_speech_validation_errors` (`tests/test_tts.py`).

**Scenario: speech 422 on out-of-range speed, invalid format.** Each returns 422 (Pydantic). **Test:** `test_speech_pydantic_validation` (`tests/test_tts.py`).

**Scenario: speech uses saved clone.** Given a `VoiceStore(tmp_path)` with a saved clone. When speech is called with the clone's id. Then the stub engine receives the loaded voice's object. **Test:** `test_speech_with_clone` (`tests/test_tts.py`).

**Scenario: speech 503, 401.** As in 8.5. **Test:** `test_speech_503_401` (`tests/test_tts.py`).

### 8.7 Clone and delete

**Scenario: clone creates a voice.** Given a 3 s WAV uploaded via multipart with text and language. When `POST /v1/audio/voices` is called. Then response is 201 with `type: "clone"`, the `.npz` file exists on disk, and the voice appears in the list. **Test:** `test_clone_creates_voice` (`tests/test_tts.py`).

**Scenario: clone 400 on bad input.** Empty text, unsupported language, invalid voice_id → 400. **Test:** `test_clone_validation_errors` (`tests/test_tts.py`).

**Scenario: clone 409 on collision.** Voice id equals a preset or an existing clone → 409. **Test:** `test_clone_collision_409` (`tests/test_tts.py`).

**Scenario: clone duration bounds.** Reference under 1 s → 400; over 30 s → 400. **Test:** `test_clone_duration_bounds` (`tests/test_tts.py`).

**Scenario: clone undecodable bytes → 400.** The message is a fixed string, no exception text. **Test:** `test_clone_undecodable_400` (`tests/test_tts.py`).

**Scenario: clone over 20 MB → 413.** **Test:** `test_clone_too_large_413` (`tests/test_tts.py`).

**Scenario: delete clone.** Given a saved clone. When `DELETE /v1/audio/voices/{id}` is called. Then response is 200 with `deleted: true`, the `.npz` is gone, and the voice is absent from the list. **Test:** `test_delete_clone` (`tests/test_tts.py`).

**Scenario: delete 404.** Unknown id, preset id, or invalid id → 404. **Test:** `test_delete_404` (`tests/test_tts.py`).

### 8.8 UI

**Scenario: index page served with CSP.** `GET /` returns 200, `text/html`, body contains "speakeasy", CSP header matches the string from SEC-09. **Test:** `test_index_page` (`tests/test_ui.py`).

**Scenario: static assets served.** `GET /static/app.js` returns 200 `text/javascript` (or `application/javascript`). `GET /static/style.css` returns 200 `text/css`. **Test:** `test_static_assets` (`tests/test_ui.py`).

**Scenario: path traversal blocked.** `GET /static/../pyproject.toml` returns 404 (Starlette `StaticFiles` handles this). **Test:** `test_static_traversal_blocked` (`tests/test_ui.py`).

## 9. Slices

Seven coder slices, in dependency order, each a card in the `speakeasy` board with a conventional-commit deliverable on `feat/tts-voice-cloning-ui`.

| Slice | Tasks    | Files                                                                      | Tests                     |
|-------|----------|----------------------------------------------------------------------------|---------------------------|
| 1     | 1–2      | `src/speakeasy/config.py`, `src/speakeasy/engines.py`                      | `tests/test_config.py`, `tests/test_tts.py` (engine unit tests) |
| 2     | 3        | `src/speakeasy/voices.py`                                                  | `tests/test_voices.py`    |
| 3     | 4–5      | `src/speakeasy/server.py` (lifespan, state, `GET /v1/audio/voices`)        | `tests/test_tts.py` (state, list) |
| 4     | 6        | `src/speakeasy/server.py` (`POST /v1/audio/speech`, encode helpers)        | `tests/test_tts.py` (speech scenarios) |
| 5     | 7        | `src/speakeasy/server.py` (`POST/DELETE /v1/audio/voices`, decode helper)  | `tests/test_tts.py` (clone/delete scenarios) |
| 6     | 8–9      | `src/speakeasy/static/index.html`, `app.js`, `style.css`; `server.py` (mount, `/` route, CSP) | `tests/test_ui.py` |
| 7     | 10–12    | `Dockerfile.cuda`, `Dockerfile.rocm`, `README.md`, `README.docker.md`, `scripts/tts_roundtrip.py` | Existing suite + Docker sanity (manual) |

Gate for every slice: `git add` every new file, then `pre-commit run --all-files`; `uv run --frozen --extra cpu --extra tts pytest tests/ -q`; and `UV_PROJECT_ENVIRONMENT="$PWD/.venv-ci" uv sync --frozen --all-groups && UV_PROJECT_ENVIRONMENT="$PWD/.venv-ci" uv run --frozen pytest tests/ -q` (CI's no-extras job). No slice adds, removes, or bumps a package.

## 10. Risks and open questions

| #  | Risk / question                                                                             | Who settles it                    | When       |
|----|---------------------------------------------------------------------------------------------|-----------------------------------|------------|
| 1  | CUDA TTS unverified at runtime — the CUDA sanity step imports `reverb_tts` but no NVIDIA host has run inference. | Orchestrator at landing           | Task 13    |
| 2  | `speed` quality at extremes (0.25, 4.0). May degrade; if so, clamp to a documented range in a follow-up. | Orchestrator — reports Task 12 results | Task 12–13 |
| 3  | Clone quality after PyAV resample (24 kHz swresample) vs. the library's kaiser-best path.   | Orchestrator — Task 12 step 4     | Task 12    |
| 4  | First-start download (~1.1 GB) blocks first TTS request until the background load finishes. | Coder — 503 with clear reason     | Slice 3    |
| 5  | Two engines contending for GPU memory — TTS serialized by one lock; peak ~2 GB.             | Coder — lock in engine            | Slice 1    |
| 6  | Mic in browser needs HTTPS or `localhost` (`getUserMedia` restriction).                     | Coder — UI message instead of silent failure | Slice 6    |
| 7  | `response_format` default `mp3` requires libsndfile with MP3 support. Both Docker images confirm this (sanity step). | Coder — `encode` helper           | Slice 4    |
| 8  | `pcm` format is raw 24 kHz s16le mono, matching OpenAI's definition. Unusual — document it. | Coder — README                    | Slice 7    |
