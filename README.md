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

### Other endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List loaded models |
| `GET /health` | Liveness probe |
| `GET /health/ready` | Readiness probe — 503 until models are loaded |

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

## Configuration

All settings are environment variables with the `SPEAKEASY_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEAKEASY_PORT` | `8080` | Port to listen on |
| `SPEAKEASY_HOST` | `0.0.0.0` | Bind address |
| `SPEAKEASY_WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model name |
| `SPEAKEASY_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. `cuda` covers both NVIDIA CUDA and AMD ROCm (CTranslate2's ROCm wheel is HIP-backed but exposes the `cuda` device API). `auto` probes for a GPU |
| `SPEAKEASY_COMPUTE_TYPE` | `auto` | `auto`, `float16`, `int8`, or `int8_float16`. `auto` resolves to `float16` on GPU, `int8` on CPU |
| `SPEAKEASY_MOONSHINE_ENABLED` | `true` | Load the Moonshine fast-path engine at startup |
| `SPEAKEASY_MOONSHINE_MODEL` | `UsefulSensors/moonshine-base` | HuggingFace model ID for the fast path |
| `SPEAKEASY_FAST_PATH_MAX_SECONDS` | `10.0` | Audio at or under this duration is auto-routed to Moonshine |
| `SPEAKEASY_BEAM_SIZE` | `5` | Beam size for Whisper decoding |
| `SPEAKEASY_VAD_FILTER` | `true` | Silero VAD filter default (Whisper path) |
| `SPEAKEASY_HF_HOME` | *(unset)* | HuggingFace cache directory — models persist across restarts when set |

## Development

```bash
uv sync --all-groups
uv run pytest tests/   # test suite (mock engines, no GPU or model downloads)
uvx ruff check src/ tests/   # lint
```

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
