# Speakeasy

OpenAI-compatible audio transcription server. Whisper large-v3-turbo as the
primary engine, with a Moonshine fast path for short English voice notes.
Runs on CPU, NVIDIA CUDA, and AMD ROCm.

Speakeasy implements the
[OpenAI audio transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription),
so any client that speaks `POST /v1/audio/transcriptions` works against it
unchanged.

## Features

- **Two engines, one API** — Whisper `large-v3-turbo` (faster-whisper /
  CTranslate2) for accuracy and multilingual support; Moonshine base for
  fast transcription of short English voice notes.
- **You choose the engine per request** — the `model` field names it:
  `whisper-1` (or the configured `SPEAKEASY_WHISPER_MODEL`) for Whisper,
  `moonshine-base` for the fast English path. There is no default engine.
- **GPU acceleration** — CUDA (NVIDIA) and ROCm (AMD) via CTranslate2. The
  ROCm wheel is HIP-backed but exposes the same `device="cuda"` API, so one
  code path covers both.
- **OpenAI response formats** — `json`, `verbose_json`, `text`, `srt`, `vtt`.
- **Text-to-speech and voice cloning** — Reverb-TTS preset voices and
  zero-shot clones saved from a reference clip, on an OpenAI-compatible
  `POST /v1/audio/speech`, plus a static web page at `/` that demos the round
  trip: record or upload → transcribe → speak.
- **A second TTS engine (VoxCPM2)** — opt in per request with
  `"model": "voxcpm2"`: 48 kHz output, 30 languages with no language tag, a
  voice designed from a written description, style hints on a clone, and
  transcript-assisted cloning. Loaded on first use, unloaded after an idle
  period — see [Second TTS engine: VoxCPM2](#second-tts-engine-voxcpm2).
- **Optional auth** — set `SPEAKEASY_API_KEY` and the `/v1/*` endpoints demand it as a bearer token; unset means open, for local dev.
- **Silero VAD** built in (on by default) to skip silence before
  transcription.
- **Observability** — per-request `X-Speakeasy-Engine` and
  `X-Speakeasy-Elapsed` response headers; `/health` and `/health/ready`
  probes.

## API

### `POST /v1/audio/transcriptions`

Multipart form fields (OpenAI-compatible):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | *(required)* | Audio file to transcribe |
| `model` | string | *(required)* | A listed transcription id from `GET /v1/models`: `whisper-1` (or the configured name) for Whisper, `moonshine-base` for the fast path |
| `language` | string | *(none)* | Language code (e.g. `en`). Only `en` (or unset) is valid with `moonshine-base`; anything else is a `400` |
| `response_format` | string | `json` | `json`, `verbose_json`, `text`, `srt`, or `vtt` |
| `temperature` | float | `0.0` | Sampling temperature (accepted for API compatibility) |

Query parameters:

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `fast` | bool | `false` | Selects Moonshine; only valid with `model=moonshine-base` — with a Whisper id it is a `400` |
| `vad` | bool | `true` | Silero VAD filter (Whisper path only) |
| `word_timestamps` | bool | `false` | Word-level timestamps (Whisper path only) |

Example:

```bash
curl -F file=@note.wav -F model=whisper-1 -F response_format=json \
     http://localhost:8080/v1/audio/transcriptions
```

### `POST /v1/audio/speech`

OpenAI-compatible text-to-speech. `model` is required and must name an engine
the deployment lists in `GET /v1/models`: `"reverb-tts"` for Reverb-TTS,
`"voxcpm2"` for the [second engine](#second-tts-engine-voxcpm2). There is no
default — a missing or empty `model` is a `422`, and an unlisted one is a
`404`. JSON body:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | *(required)* | Text to speak, 1–4096 chars (whitespace-only is rejected) |
| `voice` | string | `en_male_narrator_1` | A preset id or a saved clone id, at most 64 chars. Under `voxcpm2`: `"default"` (its own voice) or a saved VoxCPM2 clone; omitted or `null` resolves to `"default"` |
| `response_format` | string | `mp3` | `mp3`, `wav`, `flac`, or `pcm` |
| `speed` | float | `1.0` | Reverb: `0.25`–`4.0`, scaling the target length as `1 / speed` (`2.0` is shorter, `0.5` is longer) — see below. `voxcpm2`: must be `1.0` |
| `model` | string | *(required)* | A listed TTS engine id: `"reverb-tts"` or `"voxcpm2"`. Missing or empty is a `422`; any other value is a `404` |
| `language` | string | *(the voice's language)* | Reverb: one of `de`, `en`, `es`, `fr`, `it`, `pl`, `pt`. `voxcpm2`: accepted and ignored |
| `seed` | integer | *(random)* | Reproducible take when set |
| `cfg` | float | `2.0` | VoxCPM2 guidance strength, `1.0`–`5.0` (outside that range is a `422`). VoxCPM2 only |
| `design` | string | *(none)* | VoxCPM2 voice built from a description, 1–300 chars, with `voice: "default"`. VoxCPM2 only |
| `style` | string | *(none)* | VoxCPM2 style hint, 1–200 chars, on `"default"` or a clone. VoxCPM2 only |

Response headers: `X-Speakeasy-Engine` (`reverb-tts` or `voxcpm2`),
`X-Speakeasy-Elapsed`, `X-Speakeasy-Audio-Duration` (the output length in
seconds), and `X-Speakeasy-RTF` (real-time factor). `pcm` is raw signed 16-bit
little-endian mono (`audio/pcm`), matching OpenAI's definition — at **24 kHz**
for Reverb and **48 kHz** for VoxCPM2, the sample rate of the engine that
produced it (`wav`, `mp3` and `flac` carry that rate in the container).

```bash
curl -s -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "reverb-tts", "input": "Hello from speakeasy.", "voice": "en_male_narrator_1", "response_format": "wav"}' \
  -o out.wav
```

> **`speed` caveat.** Reverb applies `speed` as a factor on the length it aims
> for, not as a time-stretch of the finished audio: it generates about
> `1 / speed` times the estimated length, so `2.0` gives roughly half the
> duration and `0.5` roughly double, the same direction as OpenAI's `speed`.
> Measured on one 7.5 s line: `0.5` gave 13.8 s, `1.0` gave 7.5 s, `2.0` gave
> 3.7 s, and the `2.0` take came back missing several words when transcribed.
> Pacing and prosody change with it, so it is not a reliable rate control:
> stay near `1.0` when every word matters.

### `GET /v1/audio/voices`

Lists preset and cloned voices. `data` holds presets first (sorted by id), then
clones (each with its `transcript` and `created`); the top-level `languages`
array is the engine's supported codes, so clients never hardcode them.

```bash
curl -s -H "Authorization: Bearer $SPEAKEASY_API_KEY" \
     http://localhost:8080/v1/audio/voices | jq
```

### `POST /v1/audio/voices`

Creates a voice clone from a reference recording and its exact transcript
(multipart). This is a speakeasy extension — OpenAI has no clone endpoint.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | *(required)* | Reference audio, ≤ 20 MB, 1–30 s of speech (3–10 s recommended) |
| `text` | string | *(required for Reverb)* | The reference's exact transcript. Required when `model` is `"reverb-tts"`, optional for `"voxcpm2"` |
| `language` | string | `en` | One of the supported language codes (stored verbatim for `voxcpm2`) |
| `voice_id` | string | `clone-<hex8>` | `^[a-z0-9][a-z0-9-]{1,63}$`; `409` if it collides |
| `model` | string | *(required)* | A listed TTS engine id: `"reverb-tts"` or `"voxcpm2"`. Missing or empty is a `422`; any other value is a `404` |

```bash
curl -s -X POST http://localhost:8080/v1/audio/voices \
  -H "Authorization: Bearer $SPEAKEASY_API_KEY" \
  -F file=@reference.wav \
  -F model=reverb-tts \
  -F text="the exact words spoken in reference.wav" \
  -F language=en | jq
```

### `DELETE /v1/audio/voices/{voice_id}`

Deletes a saved clone (`{"deleted": true}`); `404` if it is not a saved clone.

```bash
curl -s -X DELETE -H "Authorization: Bearer $SPEAKEASY_API_KEY" \
     http://localhost:8080/v1/audio/voices/clone-a1b2c3 | jq
```

> **TTS weights and saved clones.** The Reverb-TTS weights are licensed
> **CC BY-NC 4.0 (non-commercial)** — do not use this stack commercially without
> a separate license. A saved clone stores its reference **transcript on disk**
> in the `.npz` file alongside the codec codes; treat the voices directory as
> sensitive.
>
> **Corrupt saved clones.** A clone whose `.npz` no longer reads (a manual edit,
> or filesystem corruption) is skipped by `GET /v1/audio/voices`, but
> `POST /v1/audio/speech` that names it fails with a `500` — the speech path
> loads the file directly rather than listing it, so the guard that skips a
> bad file in the listing does not apply. Delete the `.npz` and re-clone to
> recover.

### Other endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List loaded models |
| `GET /health` | Liveness probe |
| `GET /health/ready` | Readiness probe — 503 until models are loaded |

`GET /v1/models` lists every id this deployment offers, from configuration: the
transcription ids (`whisper-1`, the configured `SPEAKEASY_WHISPER_MODEL` when it
differs, and `moonshine-base` while `SPEAKEASY_MOONSHINE_ENABLED` is true),
followed by the TTS engine ids: `reverb-tts` while `SPEAKEASY_TTS_ENABLED` is
true, `voxcpm2` while `SPEAKEASY_TTS_VOXCPM_ENABLED` is true — whether or not an
engine has finished loading. A listed engine that is still loading answers its own
`503`; a disabled one is absent from the listing and every request naming it is a
`404`.

### Response headers

Every transcription response carries:

- `X-Speakeasy-Engine` — which engine produced the transcript (`whisper` or `moonshine`)
- `X-Speakeasy-Elapsed` — transcription time in seconds

## Second TTS engine: VoxCPM2

VoxCPM2 is a second, **opt-in** text-to-speech engine. Reverb-TTS is the engine
for everything that is not VoxCPM2, and a request selects it by name: send
`"model": "reverb-tts"`. No request is routed by default — a request with no
`model` is a `422`.

**Selecting it.** Send `"model": "voxcpm2"` on `POST /v1/audio/speech` (or the
`model` form field on `POST /v1/audio/voices`). `voxcpm2` appears in
`GET /v1/models` whenever `SPEAKEASY_TTS_VOXCPM_ENABLED` is true, and as
`"voxcpm2"` in `GET /health/ready` in every state. It is the only other engine
id speakeasy serves; any other `model` value is a `404`, and there is no default.

### Request fields

`cfg`, `design` and `style` are VoxCPM2-only; any of them sent to Reverb is a
`400` naming `voxcpm2`.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `cfg` | float | `2.0` | Guidance strength, `1.0`–`5.0` (outside that range is a `422`). Higher is louder and follows the text more literally |
| `design` | string | *(none)* | A voice built from a description, 1–300 chars. Only with `voice: "default"` |
| `style` | string | *(none)* | A style hint, 1–200 chars, on `"default"` or a clone |

`design` and `style` are mutually exclusive, and neither may contain `(`, `)`
or a newline (the library's syntax is a parenthesized prefix; an `input` that
starts with `(` is rejected for the same reason).

- **`speed`** has no effect under VoxCPM2 and must be `1.0`; anything else is a
  `400` (`{"error": {"message": "speed must be 1.0 for the voxcpm2 engine"}}`).
  Reverb's `speed` is unchanged.
- **`language`** is accepted and ignored — the model needs no language tag and
  covers 30 languages.
- **`voice`** is `"default"` (the model's own voice) or the id of a saved
  VoxCPM2 clone; an omitted or `null` `voice` resolves to `"default"`. A Reverb
  preset id is a `400`. The field is capped at 64 characters (`422` past that).

### Voices and clones

A VoxCPM2 clone stores the reference clip; the model runs at speech time, not
at clone-create time, so `text` (the reference's transcript) is **optional** on
the clone form. With a transcript, speech for that clone uses the library's
combined mode: the reference is isolated behind its own audio tokens and its
audio is continued in context — the closest match. Without one it is plain
voice cloning. The upload limits are Reverb's (20 MB, 1–30 s), and a clone id
is unique across both engines (a collision with a Reverb clone is a `409`).
`GET /v1/audio/voices` carries an `engine` field on every object, and `DELETE`
returns the `engine` that owned the clone.

### Sample rate

VoxCPM2 synthesizes at **48 kHz** (Reverb at 24 kHz). The `wav`, `mp3` and
`flac` responses carry that rate, and `pcm` is raw s16le mono at the same rate,
so a client that hardcodes 24 kHz will misread a VoxCPM2 `pcm` stream.

### Loading and idle unload

VoxCPM2 loads **on the first request**, not at startup: that request answers
`503` and starts the load, and `GET /health/ready` reports
`"voxcpm2": "loading"` until it is `"ready"` (the web UI polls and re-sends).
The first use also downloads about **4.7 GB** of weights from HuggingFace — on
a cold `/models` volume that can take minutes. After
`SPEAKEASY_TTS_VOXCPM_IDLE_SECONDS` (default `600`) with no VoxCPM2 speech the
engine unloads to free its memory, and the next request reloads it (the same
`503`-then-ready path); set the setting to `0` to keep it loaded. A load that
fails sets the engine to `error` and is **not** retried — restart the server to
recover.

### Cost

VoxCPM2 runs at roughly real time — about **1 second of wall time per second of
audio** (RTF 1.03–1.07 at the default CFG 2.0 and 10 steps, flat across length
and mode; about 17 characters of text per second of audio). Fewer diffusion
steps are faster (5 steps: 0.91) and more are slower (16: 1.24). It holds
**about 5–8 GB** of the GPU's shared memory while loaded: ~5.0 GB right after
load, ~5.4 GB peak on short takes, ~7.9 GB on a ~50 s take (reserved grows to
~11 GB over a long session). Reverb is 20–30× faster and needs 1.4–2.0 GB.
Measured on the ROCm host (gfx1151); see
[spike 005](spikes/005-voxcpm2-rocm/README.md).

### Bad takes

VoxCPM2 occasionally produces a take that is mostly silence or noise for the
input. The server does not retry silently — it answers `503` with a fixed
reason (`{"error": {"message": "voxcpm2 could not generate a good take for this
input; check the input length and try again"}}`), so the caller can ask again
(with a different `seed` if one was used).

### Limits

- **No rate control.** `speed` must be `1.0`; VoxCPM2 has no playback-rate
  knob. Use Reverb-TTS when the rate matters.
- **Loudness varies.** Output level changes with `cfg` (about 17 dB from 1.0 to
  5.0) and by a few dB with the `seed`; there is no normalization.
- **CUDA is verified at build only.** The engine is verified at runtime on ROCm
  (gfx1151) only; the CUDA image imports it but no NVIDIA GPU has run inference
  (see [README.docker.md](README.docker.md)).
- **Weights licence.** VoxCPM2's weights are **Apache-2.0** (commercial use
  permitted); Reverb-TTS's are CC BY-NC 4.0 (non-commercial) — see
  [License](#license).

## Streaming speech

A speech request can ask for its audio **as it is generated** instead of
waiting for the whole take. Add `stream_format` to the JSON body:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stream_format` | string | *(none)* | `"sse"` for server-sent events, `"audio"` for a raw chunked body. Any other value is a `422`. Omitted means today's single non-streaming response |

Only `pcm` can be streamed. A container format is written by seeking back to
patch its header once the length is known, and a stream cannot seek, so asking
to stream one is a `422`:

```
{"error": {"message": "streaming not supported for format 'wav'. Supported: pcm"}}
```

| `response_format` | Streams | Why |
|-------------------|---------|-----|
| `pcm` | yes | raw s16le mono — no container, no seeking |
| `wav` | no (`422`) | libsndfile refuses to open a WAV on a sink that cannot seek |
| `mp3` | no (`422`) | the assembled bytes decode to a fraction of the audio |
| `flac` | no (`422`) | the assembled bytes do not decode |

> **Difference from OpenAI.** OpenAI's `stream_format` defaults to `audio`
> (raw chunked audio). Speakeasy's omitted `stream_format` means today's
> non-streaming response; a raw stream is asked for explicitly with
> `"stream_format": "audio"`.

Both streaming forms carry `X-Speakeasy-Sample-Rate` — the engine's output
rate in hertz (`24000` for Reverb, `48000` for VoxCPM2) — so a client can
decode `pcm` without already knowing which engine answered.

### Response forms

| Form | `Content-Type` | Headers |
|------|----------------|---------|
| non-streaming (today) | per `response_format` | `X-Speakeasy-Engine`, `-Elapsed`, `-Audio-Duration`, `-RTF` |
| raw chunked (`stream_format: "audio"`) | per `response_format` | `X-Speakeasy-Engine`, `X-Speakeasy-Sample-Rate` |
| server-sent events (`stream_format: "sse"`) | `text/event-stream` | `Cache-Control: no-store`, `X-Accel-Buffering: no`, `X-Speakeasy-Sample-Rate` |

On a raw chunked stream only `X-Speakeasy-Engine` and
`X-Speakeasy-Sample-Rate` are sent as headers: the timing statistics are not
known before the first byte, and speakeasy sends no trailers. On an event
stream they arrive in the final event instead:

| Statistic | Non-streaming | Raw chunked | Server-sent events |
|-----------|---------------|-------------|--------------------|
| engine | `X-Speakeasy-Engine` | `X-Speakeasy-Engine` | `usage.engine` |
| sample rate | — | `X-Speakeasy-Sample-Rate` | `X-Speakeasy-Sample-Rate` |
| elapsed | `X-Speakeasy-Elapsed` | — | `usage.elapsed` |
| audio duration | `X-Speakeasy-Audio-Duration` | — | `usage.audio_duration` |
| real-time factor | `X-Speakeasy-RTF` | — | `usage.rtf` |

### Server-sent events

`stream_format: "sse"` answers `text/event-stream`. The body is one `data:`
line per event, and there are exactly two event types — zero or more
`speech.audio.delta`, then one `speech.audio.done`:

```
data: {"type":"speech.audio.delta","audio":"<base64 pcm>"}
data: {"type":"speech.audio.done","usage":{"engine":"voxcpm2","elapsed":1.234,"audio_duration":3.5,"rtf":0.35}}
```

The `usage` object carries speakeasy's four statistics — `engine`, `elapsed`,
`audio_duration`, `rtf` — in place of OpenAI's token counts. There is no
`data: [DONE]` line and no keepalive comment; the connection closes after
`speech.audio.done`. A failure after the first byte ends the stream with one
`{"type":"error","error":{"message":"…"}}` event.

Read the events as they arrive with `curl --no-buffer`:

```bash
curl -s -N http://localhost:8080/v1/audio/speech \
  -H "Authorization: Bearer $SPEAKEASY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"voxcpm2","input":"Hello world","voice":"default","response_format":"pcm","stream_format":"sse"}'
```

### Raw chunked audio

`stream_format: "audio"` answers with the format's own content type and
`Transfer-Encoding: chunked`, the encoded audio as it is generated:

```bash
curl -s -N http://localhost:8080/v1/audio/speech \
  -H "Authorization: Bearer $SPEAKEASY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"reverb-tts","input":"Hello world","voice":"en_male_narrator_1","response_format":"pcm","stream_format":"audio"}' \
  --output -
```

### Engines

| Engine | `stream_format: "sse"` | `stream_format: "audio"` |
|--------|------------------------|--------------------------|
| `voxcpm2` | one `speech.audio.delta` per ~0.16 s chunk of generated audio | chunked bytes, one chunk per ~0.16 s |
| `reverb-tts` | one delta carrying the finished take, then `done` | one chunk carrying the finished take |

Reverb-TTS has no streaming entry point, so a streaming request to it returns
the finished take as a single chunk when it is ready — documented behaviour,
not an error.

### In the web page

The page plays a stream live only while the engine keeps up with playback. It
waits until 1.5 seconds of audio has arrived. If that took no longer than 1.5
seconds it plays from there, each chunk after the previous one. If it took
longer, the page keeps receiving and plays the whole take when the stream ends.
If live playback ever runs dry, it finishes receiving and plays the rest.
VoxCPM2 generates about one second of audio per 1.05 seconds of wall time — a
little slower than playback — so on most hardware the page plays its takes
whole. The finished take is always intact.
Reverb-TTS delivers a take as one piece, so for it the page behaves as before.

## Engine routing

There is no routing: the `model` field names the engine, and the request reaches
only that engine. `model` is required and must be one of the ids
`GET /v1/models` lists.

1. **`model=whisper-1`** or the configured `SPEAKEASY_WHISPER_MODEL` — Whisper.
2. **`model=moonshine-base`** — Moonshine, the fast English path.
3. **Any other value** — `404`, naming the ids that are listed.

A request with no `model` is a `422`. `fast=true` selects Moonshine and is only
valid with `model=moonshine-base`; sent with a Whisper id it is a `400`.

Moonshine is **English-only** — `moonshine-base` with any `language` other than
`en` (or unset) is a `400`, never a quiet fall back to Whisper. It also
**drops punctuation** in its output. An engine that is configured but failed to
load stays listed and answers `503` with the reason; a request that names the
other engine is unaffected.

### Migration

The `model` field is required and is no longer auto-routed:

- A request with no `model` → add one, e.g. `-F model=whisper-1`.
- A request with a name that is not listed → use one from `GET /v1/models`.
- `fast=true` with a Whisper model → send `-F model=moonshine-base` instead.
- `model=moonshine-base` with a language other than `en` → unset `language`.

## Quickstart

### Local (uv)

```bash
uv sync
uv run speakeasy
```

The server listens on `0.0.0.0:8080`. Models are downloaded from HuggingFace
on first start; set `SPEAKEASY_HF_HOME` to control the cache location.

### Docker

```bash
docker run --gpus all -p 8080:8080 ghcr.io/donicrosby/speakeasy
```

Drop `--gpus all` for CPU. Mount a volume at `/models` to persist the model
cache across container restarts:

```bash
docker run --gpus all -p 8080:8080 -v speakeasy-models:/models \
    ghcr.io/donicrosby/speakeasy
```

## Web UI

Speakeasy serves a small single-page UI at `/` — open
**http://localhost:8080/**. It demos the round trip: record or upload audio →
transcribe → speak the text in a preset or a cloned voice. It is three static
files served from the package: no build step, no framework, no third-party
script.

- **Microphone capture** needs a secure context: `https://` or
  `http://localhost`. Over plain `http://<lan-ip>:8080` the browser blocks
  `getUserMedia`; the page says so and keeps file upload working.
- **API key.** When `SPEAKEASY_API_KEY` is set, enter it in the key field at the
  top of the page. It is kept in `sessionStorage` for the tab and sent as a
  bearer token; it never appears in a URL.

The first TTS use waits for the background model load (see
[Configuration](#configuration)); `/health/ready` reports `"tts": "loading"`
until it is ready, and the page shows that state.

## Configuration

All settings are environment variables with the `SPEAKEASY_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEAKEASY_PORT` | `8080` | Port to listen on |
| `SPEAKEASY_BIND` | `0.0.0.0` | Bind address |
| `SPEAKEASY_WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model name |
| `SPEAKEASY_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. `cuda` covers both NVIDIA CUDA and AMD ROCm (CTranslate2's ROCm wheel is HIP-backed but exposes the `cuda` device API). `auto` probes for a GPU |
| `SPEAKEASY_COMPUTE_TYPE` | `auto` | `auto`, `float16`, `int8`, or `int8_float16`. `auto` resolves to `float16` on GPU, `int8` on CPU |
| `SPEAKEASY_MOONSHINE_ENABLED` | `true` | Load the Moonshine fast-path engine at startup |
| `SPEAKEASY_MOONSHINE_MODEL` | `UsefulSensors/moonshine-base` | HuggingFace model ID for the fast path |
| `SPEAKEASY_BEAM_SIZE` | `5` | Beam size for Whisper decoding |
| `SPEAKEASY_VAD_FILTER` | `true` | Silero VAD filter default (Whisper path) |
| `SPEAKEASY_HF_HOME` | *(unset)* | HuggingFace cache directory — models persist across restarts when set |
| `SPEAKEASY_TTS_ENABLED` | `true` | Load Reverb-TTS in a background thread at startup. A no-op unless the `tts` extra is installed |
| `SPEAKEASY_TTS_MODEL` | `software-mansion/Reverb-TTS` | HuggingFace model ID for TTS |
| `SPEAKEASY_TTS_VARIANT` | `int8` | `fp32`, `fp16`, `int8`, or `int4` — download size only, same speed |
| `SPEAKEASY_TTS_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `SPEAKEASY_TTS_VOXCPM_ENABLED` | `true` | Enable the VoxCPM2 engine. A no-op unless the `tts-voxcpm` extra is installed |
| `SPEAKEASY_TTS_VOXCPM_MODEL` | `openbmb/VoxCPM2` | HuggingFace model ID for VoxCPM2 |
| `SPEAKEASY_TTS_VOXCPM_IDLE_SECONDS` | `600` | Unload VoxCPM2 after this many seconds with no speech; `0` never unloads |
| `SPEAKEASY_VOICES_DIR` | *(unset)* | Voices root for saved clones. When set, the value **is** the root (no `/voices` suffix); otherwise `<SPEAKEASY_HF_HOME or ~/.cache/speakeasy>/voices` |
| `SPEAKEASY_API_KEY` | *(unset)* | Bearer token required by the `/v1/*` endpoints. Unset or empty = no auth — see [Authentication](#authentication) |

> `SPEAKEASY_FAST_PATH_MAX_SECONDS` has been removed: `model` names the engine,
> so the duration auto-route is gone and the setting is no longer read.

## Authentication

Auth is off by default: with `SPEAKEASY_API_KEY` unset or empty, every
endpoint is open. That is for local dev — don't expose a keyless server on
an untrusted network.

Set it to require a bearer token on the `/v1/*` endpoints. This is the
header every OpenAI client already sends, so pointing an OpenAI SDK at
speakeasy with the same key needs no client changes:

```bash
export SPEAKEASY_API_KEY="$(openssl rand -hex 32)"

curl -H "Authorization: Bearer $SPEAKEASY_API_KEY" \
     -F file=@note.wav -F model=whisper-1 \
     http://localhost:8080/v1/audio/transcriptions
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key=os.environ["SPEAKEASY_API_KEY"])
```

| Endpoint | Without a valid key |
|----------|---------------------|
| `POST /v1/audio/transcriptions`, `GET /v1/models` | `401` — `{"detail": "Invalid or missing API key"}` |
| `GET /health`, `GET /health/ready` | Open — probes must work without credentials |

It is one shared secret, read at startup and compared in constant time.
There are no per-client keys, no rotation, and no TLS — terminate TLS and
anything richer at the reverse proxy.

## Development

```bash
uv sync --all-groups --extra cpu       # add --extra tts for the TTS stack
uv run --extra cpu pytest tests/       # test suite (mock engines, no GPU or model downloads)
uvx ruff check src/ tests/             # lint
```

`uv run` syncs the environment to *its own* extras before running, so pass
the extras you synced with (or `--no-sync`). A bare `uv run` targets the
no-accelerator set, which resolves to the CPU torch — in a `rocm101` or
`cu130` environment that swaps the GPU torch out from under you. The
accelerator extras are mutually exclusive and no accelerator means CPU;
`tests/test_lock_default.py` pins both.

## Design notes

The architecture and the Whisper/Moonshine split are backed by measured
spikes, kept in [`spikes/`](spikes/):

- [`001-faster-whisper-rocm`](spikes/001-faster-whisper-rocm/README.md) —
  faster-whisper + CTranslate2 ROCm wheel, OpenAI API shape.
- [`002-vad-denoise-pipeline`](spikes/002-vad-denoise-pipeline/README.md) —
  Silero VAD vs. MetricGAN+ denoise at low SNR.
- [`003-moonshine-fast-path`](spikes/003-moonshine-fast-path/README.md) —
  Moonshine is ~11x faster than Whisper large-v3-turbo on CPU at the cost of
  punctuation and English-only support — the basis for the 10s fast-path
  threshold.

## License

[Apache-2.0](LICENSE)

The Reverb-TTS weights speakeasy downloads at runtime are licensed separately:
**CC BY-NC 4.0 (non-commercial)** — see the model card for
`software-mansion/Reverb-TTS`. The VoxCPM2 weights are licensed **Apache-2.0**
(commercial use permitted). The Apache-2.0 license above covers speakeasy's own
code, not those weights.
