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
| 4  | Process model           | TTS in the server process (not a worker) — set by the coexistence smoke in the brief, not a preference.                                                                                                        |
| 5  | `/v1/audio/speech` shape | OpenAI JSON body (`model`, `input`, `voice`, `response_format`, `speed`) + two extensions: `language`, `seed`. Agent decision, overridable.                                                                           |
| 6  | Cloning API             | `POST /v1/audio/voices` multipart (`file`, `text`, `language`, optional `voice_id`). OpenAI has no clone endpoint; this is an extension. Agent decision.                                                              |
| 7  | `response_format` default | `mp3`, same as OpenAI; the UI asks for `wav`. Agent decision.                                                                                                                                                       |
| 8  | TTS load                | Background thread at startup; endpoints answer 503 with a reason until ready. Agent decision.                                                                                                                         |

## 3. Verified facts

### VERIFIED — `reverb-tts 1.0.1`

- **Weights revision.** `tts.py:29` — `REVISION = "v1.0.0"`; the default tag for `software-mansion/Reverb-TTS`.
- **Preset count.** 16 preset voices (`spikes/004-reverb-tts-rocm/results/result.json:presets`); `preset_voices()` enumerates them (`tts.py:156-164`); `preset_voice(name)` raises `ValueError` on unknown (`tts.py:180-181`).
- **Voice save/load.** `Voice.save(path)` → `np.savez(path, codes=…int16, text=…, language=…)` (`tts.py:67-73`). `Voice.load(path)` reads `.npz` with `codes` (int64, `[8, frames]`), `text`, `language` (`tts.py:83-87`).
- **`generate` return type.** `np.ndarray` float32 at the model's sample rate (`tts.py:269` — `return out[0] if single else out`).
- **Sample rate.** `tts.py:116-117` — `self.codec.config.sample_rate`; measured 24000 on gfx1151.
- **`speed` semantics.** `tts.py:228` — "a factor on the estimated length", not a time-stretch.
- **Languages.** `text.py:14-22` — `LANGUAGES: Dict[str, str]` lists 7: de, en, es, fr, it, pl, pt. `check_language(language)` raises `ValueError` (`text.py:28-31`).
- **Reference decode (lib path).** `audio.py:34-36` — `sf.read(str(audio), dtype="float32", always_2d=True)` reads via libsndfile (WAV, FLAC, MP3; no Opus).
- **`TTS.from_pretrained` signature.** `tts.py:120-154` — `(name, *, variant="int8", revision=None, token=None, device=None, dtype=None, sampler=None, codec=CODEC, codec_revision=None)`.
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

- **Seed reproducibility.** `spikes/004-reverb-tts-rocm/results/result.json:seed_reproducible` — `true`.
- **GPU memory.** `spikes/004-reverb-tts-rocm/results/result.json:gpu_mem_peak_mb` — 1984 MB peak; after load 1430 MB.
- **MIOpen.** `spikes/004-reverb-tts-rocm/README.md:7,32` — find mode costs 1.5–19 s per new output length, so `torch.backends.cudnn.enabled = False` is required on HIP. The spike's `miopenStatusUnknownError` came from an invalid `MIOPEN_FIND_MODE=default` value, not find-mode flakiness; the README records it precisely so nobody reads it as MIOpen being flaky.

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

**Response 503** (TTS loading/disabled/error): `{"error": {"message": "tts is still loading (first start downloads about 1 GB of weights)"}}` for loading, `"tts is disabled: SPEAKEASY_TTS_ENABLED=false, or the tts extra is not installed"` for disabled, and `"tts encountered an error and is unavailable"` for error. All three messages are fixed — exception details are logged (`log.exception`) but never returned in the response body.

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
  - `X-Speakeasy-Audio-Duration`: float seconds of output audio. (The transcriptions endpoint uses `X-Speakeasy-Duration` for the same concept; this is intentional — a future reconciler may unify them.)
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

**Response 409:** `{"error": {"message": "…"}}` — `voice_id` collides with a preset or an existing clone. The create path decodes the reference and validates duration *before* acquiring the engine lock; only the exists re-check, clone, and save are serialized under the engine's `threading.RLock` (nested acquire in `clone` is safe). `VoiceStore.save` uses a unique temporary filename (`.{voice_id}.{uuid}.tmp.npz`) so concurrent requests for the same `voice_id` cannot corrupt the `.npz`: exactly one returns 201, the other 409.

**Response 413:** `{"error": {"message": "reference audio over 20 MB"}}` — the raw upload exceeds 20 MB before any decode attempt. Known limitation: the 20 MB check runs after Starlette has spooled the multipart body to disk, same as the existing transcriptions endpoint.

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

- **Directory:** `SPEAKEASY_VOICES_DIR` when set (the value **is** the voices root — no `/voices` suffix is appended); otherwise `<hf_home or ~/.cache/speakeasy>/voices`.
- **File:** `<voice_id>.npz`, written by `reverb_tts.Voice.save()` (`tts.py:67-73`).
- **Contents:** `codes` (int16 `[8, frames]`), `text` (the transcript string), `language` (language code string), optional `speech_frames` (int).
- **Voice id regex:** `^[a-z0-9][a-z0-9-]{1,63}$`. An id is validated before it ever becomes a path; invalid ids are rejected, never sanitized.
- **Mode:** the voices directory is created with `0o700` at store init.
- **Note for coder slices:** the brief's Task 1 `voices_dir_default()` snippet appends `/voices` in both branches — the spec's semantics above (set value IS the root) win. Adjust the snippet, don't copy it.

## 6. Security requirements

Each requirement is named and testable.

1. **SEC-01 — voice id is validated before any path is built.** `check_voice_id()` enforces the regex; `VoiceStore._path()` calls it first. Test: `test_invalid_voice_ids_rejected` (`tests/test_voices.py`, §8.3 — includes the `../etc/passwd` traversal id).

2. **SEC-02 — clone id collides with preset → 409.** `POST /v1/audio/voices` checks presets before saving. Test: `test_clone_collision_409` (`tests/test_tts.py`, §8.7).

3. **SEC-03 — clone id collides with existing clone → 409.** `VoiceStore.exists()` checked before save. Test: `test_clone_collision_409` (`tests/test_tts.py`, §8.7 — the same test covers both collision kinds).

4. **SEC-04 — reference upload cap 20 MB → 413.** Raw body read capped at `MAX_REFERENCE_BYTES + 1`; exceeds → 413 before any decode. Test: `test_clone_too_large_413` (`tests/test_tts.py`, §8.7).

5. **SEC-05 — reference duration bounds 1–30 s → 400.** Duration measured after decode; out of range → 400 with the exact duration in the message. Known limitation: decoded length is not bounded before decode (20 MB of compressed audio can decode to hours of float32). Test: `test_clone_duration_bounds` (`tests/test_tts.py`, §8.7).

6. **SEC-06 — decode failure → 400, no exception text leaked.** Exception caught; message is a fixed string. Test: `test_clone_undecodable_400` (`tests/test_tts.py`, §8.7).

7. **SEC-07 — one operation at a time; clone-create atomic.** `ReverbEngine` holds a `threading.RLock`; `synthesize` and `clone` acquire it. The create endpoint decodes the reference and validates duration *before* acquiring the engine lock; the locked region covers only the exists re-check, clone, and save — decode never stalls speech inference. Within the locked region `clone` re-acquires the same `RLock` (nested acquire is safe). `VoiceStore.save` writes to a unique temporary filename (`.{voice_id}.{uuid}.tmp.npz`) then atomically `os.replace`s it into place. The speech and clone-create endpoints are `def` (not `async def`), so FastAPI runs them on a threadpool — the lock blocks a worker thread, never the event loop. Test: `test_engine_lock_serialises_calls` and `test_concurrent_clone_create_one_wins` (`tests/test_tts.py`).

8. **SEC-08 — model loads off the startup path.** `start_tts()` spawns a daemon thread; readiness checks only Whisper + Moonshine. Test: `test_ready_reports_loading_while_tts_boots` (`test_tts.py`).

9. **SEC-09 — CSP with no `unsafe-inline`.** `GET /` returns `Content-Security-Policy: default-src 'self'; media-src 'self' blob:; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'`. Test: `test_index_page` (`tests/test_ui.py`, §8.8).

10. **SEC-10 — no inline script, style, or event handlers.** The CSP forbids inline scripts and styles; the static HTML `/` page has no `<script>` or `<style>` blocks — only `<link rel="stylesheet">` and `<script src="app.js">`. The page also has no inline event-handler attributes (`onclick=`, `onload=`, etc.). Verify by reading `index.html` source in `test_ui.py`.

11. **SEC-11 — no third-party script.** `default-src 'self'` covers scripts. The page loads only local resources.

12. **SEC-12 — no `innerHTML` with server or user data.** UI uses `textContent` for all dynamic text (transcripts, errors, voice names). Documented in `app.js`; verified by review.

13. **SEC-13 — API key in `sessionStorage`, never in a URL.** `app.js` stores the key in `sessionStorage.getItem("speakeasy_key")`; `api()` helper adds it as an `Authorization` header. No `?key=` or hash parameters. Check: review of the shipped `app.js`/`index.html` source (no automated test — the repo has no JS harness); behaviour specified in §7.1.

14. **SEC-14 — MIOpen flag set only on HIP.** `ReverbEngine.__init__` disables `torch.backends.cudnn.enabled` only when `torch.version.hip is not None`. Test: `test_cudnn_disabled_only_on_hip` (`test_tts.py`).

15. **SEC-15 — no auth on health/root/static.** `/health`, `/health/ready`, `GET /`, `/static/*` have no `Depends(require_auth)`. Test: existing `test_health_probes_stay_open` (`test_auth.py`) + new assertions for `/` and `/static/*` in `test_ui.py`.

16. **SEC-16 — no exception text in any response body.** Every error response body carries a fixed message; exception details, stack traces, file paths, HF repo URLs, and local paths are logged (`log.exception`) but never returned to the client. This covers 400, 500, and 503 bodies from all endpoints. Test: `test_no_exception_text_in_error_bodies` (`tests/test_tts.py` — spot-checks the error-state 503, decode 400, and an unhandled 500 path).

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

- **Controls:** a textarea, a voice `<select>` (optgroups Presets / Saved, populated from `GET /v1/audio/voices`), a language `<select>` (defaults to chosen voice's language), a Speak button. The voice select opens on `en_male_narrator_1` (the API default) and its language follows it; a reload of the voices list keeps whatever the user chose.
- **Request:** `POST /v1/audio/speech` JSON with `response_format: "wav"`.
- **Result:** `<audio controls>` from the blob URL, a Download link, audio duration, elapsed time, × real time.
- **Loading state:** button disabled, reads "working…". TTS on CPU can take tens of seconds; other panels stay interactive.

### 7.4 Panel: Clone a voice

- **Controls:** Record/Upload (reuses the Listen panel's recording), a transcript textarea with "use last transcript" button, language `<select>`, an optional voice name input, Save button. The language select opens on `en`; a reload of the voices list keeps whatever the user chose.
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
| 503 (TTS error)              | "TTS encountered an error and is unavailable — check the server logs."                     |
| Network error / fetch fail   | "Could not reach the server — check the connection."                                      |
| Mic unavailable              | Record buttons replaced with the HTTPS/localhost message (see 7.2).                       |

## 8. Acceptance scenarios

Every scenario runs without a GPU. Tests fake `reverb_tts` through `sys.modules`; no test imports `reverb_tts` for real.

### 8.1 Config

**Scenario: TTS config defaults.** Given no env vars. When `Settings()` is created. Then `tts_enabled` is `True`, `tts_model` is `"software-mansion/Reverb-TTS"`, `tts_variant` is `"int8"`, `tts_device` is `"auto"`, `voices_dir` is `None`. **Test:** `test_tts_defaults` (`tests/test_config.py`).

**Scenario: voices dir follows HF_HOME.** Given `SPEAKEASY_VOICES_DIR` unset and `hf_home="/models"`. When `voices_dir_default()` is called. Then it returns `"/models/voices"`. **Test:** `test_voices_dir_follows_hf_home` (`tests/test_config.py`).

**Scenario: voices dir set uses the value as-is.** Given `SPEAKEASY_VOICES_DIR="/custom"`. When `voices_dir_default()` is called. Then it returns `"/custom"` (no `/voices` suffix). **Test:** `test_voices_dir_set_uses_value_as_is` (`tests/test_config.py`).

### 8.2 Engine

**Scenario: engine loads, presets and synthesize.** Given a fake `reverb_tts` module. When `ReverbEngine` is constructed with `device="cuda"`. Then `name` is `"reverb-tts"`, `sample_rate` is 24000, `languages` is `["de", "en"]`, `presets()` returns preset dicts. `synthesize("Hi.", …)` returns float32 audio and elapsed ≥ 0. **Test:** `test_engine_loads_presets_and_speaks` (`tests/test_tts.py`).

**Scenario: engine clones from samples.** Given a fake engine. When `clone(samples, 24000, "text", "de")` is called. Then the returned voice has `language="de"` and `FakeTTS.calls` records the clone. **Test:** `test_engine_clones_from_samples` (`tests/test_tts.py`).

**Scenario: cudnn disabled only on HIP.** Given `torch.version.hip` set. When `ReverbEngine` is constructed. Then `torch.backends.cudnn.enabled` is `False`. When `hip` is `None`, `cudnn.enabled` is left alone. **Test:** `test_cudnn_disabled_only_on_hip` (`tests/test_tts.py`).

**Scenario: presets resolved once at construction.** Given a fake `reverb_tts` module whose `preset_voice()` records every call. When `ReverbEngine` is constructed. Then every preset the fake module offers is fetched during `__init__` (the real engine: 16). When `voice()` is called for a known preset. Then `preset_voice()` is NOT called again — the cached `Voice` object is returned. No `hf_hub_download` network call can fire on the request path. **Test:** `test_presets_cached_at_construction` (`tests/test_tts.py`).

### 8.3 Voice store

**Scenario: valid id round-trip.** Given a valid id `"my-voice-1"`. When saved, loaded, read via `info()`, listed via `ids()`, then deleted. Then all operations succeed and `exists()` returns `False` after delete. **Test:** `test_voice_store_roundtrip` (`tests/test_voices.py`).

**Scenario: invalid ids rejected.** Given ids `"../etc/passwd"`, `"En_X"` (uppercase), `"a"` (too short), 65 chars (too long). When `check_voice_id()` is called or they are passed to `VoiceStore`. Then `ValueError` is raised; `exists()` returns `False` without raising. **Test:** `test_invalid_voice_ids_rejected` (`tests/test_voices.py`).

**Scenario: delete unknown returns False.** Given an id that was never saved. When `delete()` is called. Then it returns `False`. **Test:** `test_delete_unknown_returns_false` (`tests/test_voices.py`).

**Scenario: corrupt npz skipped in listing.** Given a voices directory containing one valid clone and one garbage `.npz` (not a valid numpy archive). When `GET /v1/audio/voices` is called. Then response is 200; the valid clone is listed; the corrupt file is skipped — no 500. **Test:** `test_corrupt_npz_skipped_in_listing` (`tests/test_voices.py`).

### 8.4 Lifespan and state

**Scenario: readiness reports TTS state.** With `tts=None` and `tts_state="disabled"`. When `GET /health/ready` is called. Then the response includes `"tts": "disabled"`. **Test:** `test_ready_reports_tts_state` (`tests/test_tts.py`).

**Scenario: models lists reverb-tts when loaded.** With `tts` monkeypatched to a stub. When `GET /v1/models` is called. Then `"reverb-tts"` appears in the model ids. **Test:** `test_models_lists_reverb_tts` (`tests/test_tts.py`).

**Scenario: models omits reverb-tts when unloaded.** With `tts=None`. When `GET /v1/models` is called. Then `"reverb-tts"` is absent. **Test:** `test_models_omits_tts_when_none` (`tests/test_tts.py`).

**Scenario: ready reports loading while TTS boots.** With `tts=None` and `tts_state="loading"`. When `GET /health/ready` is called. Then the response includes `"tts": "loading"` and the endpoint returns 200 immediately — it does not block waiting for the TTS thread. **Test:** `test_ready_reports_loading_while_tts_boots` (`tests/test_tts.py`).

**Scenario: TTS disabled when extra not installed.** With `reverb_tts` absent from `sys.modules`. When `start_tts()` is called. Then `tts_state` is `"disabled"`, no background thread is started, and `tts` remains `None`. When `GET /health/ready` is called. Then the response includes `"tts": "disabled"`. **Test:** `test_tts_disabled_on_import_error` (`tests/test_tts.py`).

**Scenario: ready reports error state.** With `tts=None` and `tts_state="error"`. When `GET /health/ready` is called. Then the response includes `"tts": "error"`. When a TTS endpoint is called. Then the 503 body is the fixed error message (no exception text). **Test:** `test_ready_reports_error_state` (`tests/test_tts.py`).

**Scenario: error bodies contain no exception text.** Given three forced error paths: `tts_state="error"` (503), an undecodable upload (400), and a handler that raises → 500. When each response body is read. Then every body contains only its fixed message; no exception text, stack trace, file path, or URL appears in any response. **Test:** `test_no_exception_text_in_error_bodies` (`tests/test_tts.py`).

### 8.5 Voice list

**Scenario: list voices returns presets.** Given a stub TTS engine and a `VoiceStore(tmp_path)` with one saved clone. When `GET /v1/audio/voices` is called with a valid key. Then response is 200, `data` contains presets with `type: "preset"` and clones with `type: "clone"`, `transcript`, `created`. **Test:** `test_list_voices` (`tests/test_tts.py`).

**Scenario: list voices 503 when TTS is None.** With `tts=None`. When `GET /v1/audio/voices` is called. Then response is 503. **Test:** `test_list_voices_503_when_tts_none` (`tests/test_tts.py`).

**Scenario: list voices 401 without key.** With `settings.api_key` set. When called without `Authorization`. Then response is 401. **Test:** `test_list_voices_401` (`tests/test_tts.py`).

### 8.6 Speech

**Scenario: speech returns audio in the requested format.** With a stub engine returning 0.5 s of sine. When `POST /v1/audio/speech` is called with `{"input": "Hi"}`. Then response is 200, `Content-Type: audio/mpeg`, body starts with MP3 magic. Wav → `RIFF`, flac → `fLaC`, pcm → 24000 bytes with `audio/pcm`. **Test:** `test_speech_response_formats` (`tests/test_tts.py`).

**Scenario: speech response headers.** With a stub engine. When speech is generated. Then response includes `X-Speakeasy-Engine: reverb-tts`, `X-Speakeasy-Elapsed`, `X-Speakeasy-Audio-Duration`, `X-Speakeasy-RTF`. **Test:** `test_speech_response_headers` (`tests/test_tts.py`).

**Scenario: speech language defaults to voice language.** With a stub that records the call. When `POST /v1/audio/speech` is called without `language`. Then the stub received the preset's language. When called with an explicit `language`, that wins. **Test:** `test_speech_language_default` (`tests/test_tts.py`).

**Scenario: speech 400 on unknown voice, bad language, empty input.** Each returns 400 with a message. "Empty" includes whitespace-only `input` (stripped before validation). **Test:** `test_speech_validation_errors` (`tests/test_tts.py`).

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

**Scenario: clone 401 without key.** With `settings.api_key` set. When `POST /v1/audio/voices` is called without `Authorization`. Then response is 401. **Test:** `test_clone_401` (`tests/test_tts.py`).

**Scenario: clone 422 on missing multipart fields.** When `POST /v1/audio/voices` is called without `file` or without `text`. Then response is 422 (FastAPI/Pydantic validation error). **Test:** `test_clone_missing_fields_422` (`tests/test_tts.py`).

**Scenario: clone 503 when TTS is None.** With `tts=None`. When `POST /v1/audio/voices` is called. Then response is 503 with the appropriate fixed message. **Test:** `test_clone_503` (`tests/test_tts.py`).

**Scenario: concurrent clone-create same id — one wins.** Given two concurrent `POST /v1/audio/voices` with the same `voice_id`. When both complete. Then exactly one returns 201 with a valid `.npz` on disk, the other returns 409, and the `.npz` loads cleanly (not corrupted by interleaved writes). **Test:** `test_concurrent_clone_create_one_wins` (`tests/test_tts.py`).

**Scenario: engine lock serialises synthesis.** Given a stub engine whose `synthesize` records concurrency (thread count in-flight). When two concurrent `POST /v1/audio/speech` calls are made. Then at most one synthesis is in-flight at any moment. **Test:** `test_engine_lock_serialises_calls` (`tests/test_tts.py`).

**Scenario: delete clone.** Given a saved clone. When `DELETE /v1/audio/voices/{id}` is called. Then response is 200 with `deleted: true`, the `.npz` is gone, and the voice is absent from the list. **Test:** `test_delete_clone` (`tests/test_tts.py`).

**Scenario: delete 404.** Unknown id, preset id, or invalid id → 404. **Test:** `test_delete_404` (`tests/test_tts.py`).

**Scenario: delete 401 without key.** With `settings.api_key` set. When `DELETE /v1/audio/voices/{id}` is called without `Authorization`. Then response is 401. **Test:** `test_delete_401` (`tests/test_tts.py`).

**Scenario: delete 503 when TTS is None.** With `tts=None`. When `DELETE /v1/audio/voices/{id}` is called. Then response is 503. **Test:** `test_delete_503` (`tests/test_tts.py`).

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

### Slice 7 contract

**Docker sanity assertions (Task 10).** Both `Dockerfile.cuda` and `Dockerfile.rocm` end with a `RUN` that imports `reverb_tts` and additionally asserts:

1. `import importlib.resources as r, soundfile as sf`
2. `assert (r.files('speakeasy') / 'static' / 'index.html').is_file()` — the static UI shell is present in the installed wheel.
3. `assert (r.files('speakeasy') / 'static' / 'app.js').is_file()` — the JS bundle is present in the installed wheel.
4. `assert (r.files('speakeasy') / 'static' / 'style.css').is_file()` — the stylesheet is present in the installed wheel.
5. `assert 'MP3' in sf.available_formats()` — libsndfile has MP3 support; the default `response_format` would fail without it.

**Docs requirements (Task 11).** `README.md` and `README.docker.md` extended with:

- Five `SPEAKEASY_TTS_*` config rows (`ENABLED`, `MODEL`, `VARIANT`, `DEVICE`, `VOICES_DIR`).
- The `speed`-is-a-length-factor caveat (not a playback rate).
- Mic needs HTTPS or `localhost` (`getUserMedia` restriction).
- **Reverb weights are CC BY-NC 4.0 (non-commercial).**
- **A saved clone stores its transcript on disk** in the `.npz` file.
- `/v1/models` extension note.
- First-start download of ~1.1 GB for TTS weights.

**Round-trip script (Task 12).** `scripts/tts_roundtrip.py` (run inside the image; not installed) orchestrates the real GPU path. Its contract:

1. **Wait:** poll `GET /health/ready` until `tts == "ready"` (timeout 120 s).
2. **Preset speech → wav:** `POST /v1/audio/speech` with preset `en_male_narrator_1`, a 20-word line, `response_format: "wav"` → save to `t1.wav`.
3. **Transcribe:** `POST /v1/audio/transcriptions` with `t1.wav` → compute `difflib.SequenceMatcher` similarity between transcription and the original line; assert ≥ 0.8.
4. **Clone:** `POST /v1/audio/voices` multipart with `t1.wav`, its transcript, `voice_id: "roundtrip"` → assert 201.
5. **Clone speech:** `POST /v1/audio/speech` with `voice: "roundtrip"` → save to `t2.wav`; transcribe → similarity ≥ 0.8.
6. **Speed report:** `POST /v1/audio/speech` at speeds 0.5, 1.0, 2.0 → report audio duration and transcript similarity for each.
7. **Delete:** `DELETE /v1/audio/voices/roundtrip` → assert `deleted: true`.
8. **Output:** one JSON line per step (`{"step": "…", "ok": true, …}`). Exit code 0 only when all steps pass.

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
| 9  | Corrupt `.npz` in the voices dir (from concurrent writes, manual edits, or fs corruption) can raise inside `VoiceStore.info`/`load` → unhandled 500; one bad file can break `GET /v1/audio/voices` entirely. | Coder — skip/quarantine unreadable files in `ids()` and `info()`; document the 500 for direct-access corruption. | Slice 2    |
| 10 | Header naming drift: transcriptions returns `X-Speakeasy-Duration` (`server.py:218`), speech returns `X-Speakeasy-Audio-Duration`. Both name the audio length, but the names differ. | Documented in §4.2 and brief — intentional as of this spec; a future reconciler can rename one. | —          |
