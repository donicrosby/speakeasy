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
- **Automatic engine routing** — audio under ~10 seconds is routed to
  Moonshine automatically; everything else goes to Whisper. Clients can
  override per-request.
- **GPU acceleration** — CUDA (NVIDIA) and ROCm (AMD) via CTranslate2. The
  ROCm wheel is HIP-backed but exposes the same `device="cuda"` API, so one
  code path covers both.
- **OpenAI response formats** — `json`, `verbose_json`, `text`, `srt`, `vtt`.
- **Text-to-speech and voice cloning** — Reverb-TTS preset voices and
  zero-shot clones saved from a reference clip, on an OpenAI-compatible
  `POST /v1/audio/speech`, plus a static web page at `/` that demos the round
  trip: record or upload → transcribe → speak.
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
| `model` | string | `""` | Model name. `moonshine-base` selects the fast path; empty string auto-routes |
| `language` | string | *(none)* | Language code (e.g. `en`). Anything other than `en` forces Whisper |
| `response_format` | string | `json` | `json`, `verbose_json`, `text`, `srt`, or `vtt` |
| `temperature` | float | `0.0` | Sampling temperature (accepted for API compatibility) |

Query parameters:

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `fast` | bool | `false` | Force the Moonshine fast path regardless of audio duration |
| `vad` | bool | `true` | Silero VAD filter (Whisper path only) |
| `word_timestamps` | bool | `false` | Word-level timestamps (Whisper path only) |

Example:

```bash
curl -F file=@note.wav -F response_format=json \
     http://localhost:8080/v1/audio/transcriptions
```

### `POST /v1/audio/speech`

OpenAI-compatible text-to-speech (Reverb-TTS). JSON body:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | *(required)* | Text to speak, 1–4096 chars (whitespace-only is rejected) |
| `voice` | string | `en_male_narrator_1` | A preset id or a saved clone id |
| `response_format` | string | `mp3` | `mp3`, `wav`, `flac`, or `pcm` |
| `speed` | float | `1.0` | `0.25`–`4.0`. Scales the target length as `1 / speed` (`2.0` is shorter, `0.5` is longer) — see below |
| `model` | string | `""` | Accepted for OpenAI clients; the server has one TTS model |
| `language` | string | *(the voice's language)* | One of `de`, `en`, `es`, `fr`, `it`, `pl`, `pt` |
| `seed` | integer | *(random)* | Reproducible take when set |

Response headers: `X-Speakeasy-Engine` (`reverb-tts`), `X-Speakeasy-Elapsed`,
`X-Speakeasy-Audio-Duration` (the output length in seconds), and
`X-Speakeasy-RTF` (real-time factor). `pcm` is raw 24 kHz signed 16-bit
little-endian mono (`audio/pcm`), matching OpenAI's definition.

```bash
curl -s -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from speakeasy.", "voice": "en_male_narrator_1", "response_format": "wav"}' \
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
| `text` | string | *(required)* | The reference's exact transcript |
| `language` | string | `en` | One of the supported language codes |
| `voice_id` | string | `clone-<hex8>` | `^[a-z0-9][a-z0-9-]{1,63}$`; `409` if it collides |

```bash
curl -s -X POST http://localhost:8080/v1/audio/voices \
  -H "Authorization: Bearer $SPEAKEASY_API_KEY" \
  -F file=@reference.wav \
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

`GET /v1/models` lists `reverb-tts` once the TTS engine has finished loading
(the load runs in the background; see [Configuration](#configuration)).

### Response headers

Every transcription response carries:

- `X-Speakeasy-Engine` — which engine produced the transcript (`whisper` or `moonshine`)
- `X-Speakeasy-Elapsed` — transcription time in seconds

## Engine routing

For each request, the engine is picked as follows:

1. **`fast=true` query param** — forces Moonshine, regardless of duration.
2. **`model=moonshine-base` form field** — selects Moonshine by name.
3. **Auto-route** — if `model` is empty, the audio duration is probed and
   anything ≤ `SPEAKEASY_FAST_PATH_MAX_SECONDS` (default 10s) goes to
   Moonshine; longer audio goes to Whisper.
4. **Otherwise** — Whisper.

Moonshine is only eligible when `language` is unset or `en` — it is
**English-only**. It also **drops punctuation** in its output. If Moonshine
fails to load at startup the server continues with Whisper only; requests
that would have taken the fast path fall back automatically.

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
| `SPEAKEASY_FAST_PATH_MAX_SECONDS` | `10.0` | Audio at or under this duration is auto-routed to Moonshine |
| `SPEAKEASY_BEAM_SIZE` | `5` | Beam size for Whisper decoding |
| `SPEAKEASY_VAD_FILTER` | `true` | Silero VAD filter default (Whisper path) |
| `SPEAKEASY_HF_HOME` | *(unset)* | HuggingFace cache directory — models persist across restarts when set |
| `SPEAKEASY_TTS_ENABLED` | `true` | Load Reverb-TTS in a background thread at startup. A no-op unless the `tts` extra is installed |
| `SPEAKEASY_TTS_MODEL` | `software-mansion/Reverb-TTS` | HuggingFace model ID for TTS |
| `SPEAKEASY_TTS_VARIANT` | `int8` | `fp32`, `fp16`, `int8`, or `int4` — download size only, same speed |
| `SPEAKEASY_TTS_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `SPEAKEASY_VOICES_DIR` | *(unset)* | Voices root for saved clones. When set, the value **is** the root (no `/voices` suffix); otherwise `<SPEAKEASY_HF_HOME or ~/.cache/speakeasy>/voices` |
| `SPEAKEASY_API_KEY` | *(unset)* | Bearer token required by the `/v1/*` endpoints. Unset or empty = no auth — see [Authentication](#authentication) |

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
     -F file=@note.wav http://localhost:8080/v1/audio/transcriptions
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
`software-mansion/Reverb-TTS`. The Apache-2.0 license above covers speakeasy's
own code, not those weights.
