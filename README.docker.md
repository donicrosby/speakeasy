# Speakeasy — Docker builds

Production container images for speakeasy, in two GPU flavors:

| Variant | Dockerfile | Accelerator stack | GPU |
|---|---|---|---|
| CUDA | `Dockerfile.cuda` | torch from `download.pytorch.org/whl/cu130` (`cu130` extra) | NVIDIA (via NVIDIA Container Toolkit) |
| ROCm | `Dockerfile.rocm` | ROCm 10.1 runtime as pip wheels (`rocm101` extra), CTranslate2 built from source (HIP) | AMD gfx1151 (Strix Halo) |

Both images run the same server (`speakeasy` → OpenAI-compatible
transcription API) on port **8080**, and both are built with the TTS stack
(`--extra tts`: reverb-tts + torchaudio). On first start the Whisper
(and optionally Moonshine) models are downloaded into the Hugging Face
cache; mount a volume at `/models` to avoid re-downloading.

## Pulling prebuilt images

CI builds and pushes both variants on every push to `main`:

```bash
docker pull ghcr.io/donicrosby/speakeasy:cuda    # also tagged :latest
docker pull ghcr.io/donicrosby/speakeasy:rocm
```

Per-commit tags are also published: `cuda-<sha>`, `rocm-<sha>`. Version tags
(`v1.2.3`) additionally publish `<version>-cuda` / `<version>-rocm`.

## Building locally

```bash
docker build -f Dockerfile.cuda -t speakeasy:cuda .
docker build -f Dockerfile.rocm -t speakeasy:rocm .
```

Both builds start from `python:3.12-slim` and install the locked dependency
tree with `uv sync --frozen --no-dev --extra <accelerator> --extra tts`. The
GPU stacks come from pip, selected by the conflicting extras in
`pyproject.toml` (one universal `uv.lock`):

1. **CUDA.** torch and its `nvidia-*` wheels come from the cu130 index;
   ctranslate2 stays on the PyPI wheel, which bundles its CUDA kernels.
2. **ROCm 10.1.** torch and the whole ROCm 10.1 user-space runtime (rocBLAS,
   MIOpen, the gfx1151 kernel shims) come from AMD's wheel index
   (`stable.repo.amd.com/rocm/whl-next`) — no apt ROCm packages in the
   runtime image. ctranslate2's PyPI wheel is CUDA-only, so a builder stage
   (Ubuntu 24.04 + the ROCm 10.1 SDK from AMD's apt repo) compiles
   CTranslate2 with the HIP backend for gfx1151, and the runtime image swaps
   that wheel in for the locked PyPI one *after* its last `uv sync`.

Each build ends with a sanity step that imports the stack and asserts the
right builds are present: HIP torch and the HIP source-built ctranslate2 on
ROCm, CUDA torch on CUDA, and the torchaudio/reverb-tts imports on both.

## Running

### NVIDIA (CUDA)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker run --rm --gpus all \
  -p 8080:8080 \
  -v speakeasy-models:/models \
  ghcr.io/donicrosby/speakeasy:cuda
```

### AMD (ROCm)

Requires a ROCm-capable host kernel/amdgpu driver.

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --ipc=host --shm-size 8g \
  -p 8080:8080 \
  -v speakeasy-models:/models \
  ghcr.io/donicrosby/speakeasy:rocm
```

### Docker Compose

One compose file, two services — start the one matching your GPU:

```bash
docker compose up cuda    # or: docker compose up rocm
```

## Configuration (environment variables)

All settings use the `SPEAKEASY_` prefix. Defaults shown are the
application defaults; the Dockerfiles additionally set
`SPEAKEASY_HF_HOME=/models`, `SPEAKEASY_BIND=0.0.0.0`, `SPEAKEASY_PORT=8080`.

| Variable | Default | Description |
|---|---|---|
| `SPEAKEASY_PORT` | `8080` | Listen port |
| `SPEAKEASY_BIND` | `0.0.0.0` | Bind address |
| `SPEAKEASY_WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model id (Hugging Face or CTranslate2 name) |
| `SPEAKEASY_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu` (use `cuda` for **both** NVIDIA and ROCm) |
| `SPEAKEASY_COMPUTE_TYPE` | `auto` | `auto` \| `float16` \| `int8` \| `int8_float16` |
| `SPEAKEASY_MOONSHINE_ENABLED` | `true` | Enable the Moonshine fast path for short audio |
| `SPEAKEASY_MOONSHINE_MODEL` | `UsefulSensors/moonshine-base` | Moonshine model id |
| `SPEAKEASY_FAST_PATH_MAX_SECONDS` | `10.0` | Audio shorter than this may use Moonshine |
| `SPEAKEASY_BEAM_SIZE` | `5` | Whisper beam size |
| `SPEAKEASY_VAD_FILTER` | `true` | Enable Silero VAD filtering |
| `SPEAKEASY_HF_HOME` | unset (image: `/models`) | Hugging Face cache directory — persist this |

Example:

```bash
docker run --rm --gpus all -p 8080:8080 \
  -e SPEAKEASY_WHISPER_MODEL=distil-large-v3 \
  -e SPEAKEASY_MOONSHINE_ENABLED=false \
  -v speakeasy-models:/models \
  ghcr.io/donicrosby/speakeasy:cuda
```

## Verifying a running container

Liveness (process up):

```bash
curl -sf http://localhost:8080/health
```

Readiness — returns `503` until the models finish loading, then `200`
(first start downloads several GB of models, so this can take a while;
watch `docker logs`):

```bash
curl -s http://localhost:8080/health/ready | jq
```

List loaded models:

```bash
curl -s http://localhost:8080/v1/models | jq
```

Transcribe a WAV file (OpenAI-compatible endpoint):

```bash
# make a 2-second test wav if you don't have one handy
python3 -c "
import math, struct, wave
w = wave.open('/tmp/test.wav', 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
w.writeframes(b''.join(struct.pack('<h', int(12000*math.sin(2*math.pi*440*i/16000))) for i in range(32000)))
w.close()"

curl -s http://localhost:8080/v1/audio/transcriptions \
  -F file=@/tmp/test.wav \
  -F model=large-v3-turbo | jq
```
