# Speakeasy — Docker builds

Production container images for speakeasy, in two GPU flavors:

| Variant | Dockerfile | Base image | GPU |
|---|---|---|---|
| CUDA | `Dockerfile.cuda` | `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime` | NVIDIA (via NVIDIA Container Toolkit) |
| ROCm | `Dockerfile.rocm` | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` | AMD (gfx1151/Strix Halo and other ROCm 7.2-supported targets) |

Both images run the same server (`speakeasy` → OpenAI-compatible
transcription API) on port **8080**. On first start the Whisper
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

The two builds are symmetric: each uses the official PyTorch container for
its GPU vendor as the base (`pytorch/pytorch` for CUDA, `rocm/pytorch` for
ROCm) and installs the locked dependency tree with `uv sync --frozen
--no-dev`, skipping the GPU stack with `--no-install-package` so the base
image's torch is the single source of truth.

1. **torch comes from the base image (both variants).** The locked
   dependency tree would otherwise pull the default PyPI torch plus ~7 GB
   of `nvidia-*` CUDA wheels — duplicate weight, and wrong on AMD. On the
   CUDA base, torch (`2.14.1+cu130`) lives in the system environment and
   the app is synced into a project venv created with
   `--system-site-packages`; on the ROCm base, torch
   (`2.10.0+rocm7.2.4`, HIP) lives in `/opt/venv` itself, so the app is
   synced directly into `/opt/venv` (`UV_PROJECT_ENVIRONMENT`). The skip
   list (torch, triton, every `nvidia-*`/`cuda-*` package in uv.lock) must
   be repeated on every `uv sync` — without it a later sync would
   reinstall torch from PyPI. If the lockfile gains another
   `nvidia-*`/`cuda-*` package, add it to the skip list in both
   Dockerfiles.
2. **ctranslate2.** CUDA: the default PyPI wheel is correct — its CUDA
   variant bundles the kernels it needs on top of the base image's CUDA
   stack. ROCm: the PyPI wheel is CUDA-only, so the build skips it at sync
   time and installs the official CTranslate2 v4.7.1 ROCm wheel
   (`rocm-python-wheels-Linux.zip` from the CTranslate2 GitHub release,
   cp312 manylinux x86_64) instead. The wheel is HIP-backed but still uses
   `device="cuda"` internally — no code changes needed. The ROCm base
   image provides the HIP/rocBLAS/MIOpen shared libraries it links
   against; ROCm 7.2.x supports gfx1151 (Strix Halo) natively, so no
   `HSA_OVERRIDE_GFX_VERSION` workaround is required.

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
`SPEAKEASY_HF_HOME=/models`, `SPEAKEASY_HOST=0.0.0.0`, `SPEAKEASY_PORT=8080`.

| Variable | Default | Description |
|---|---|---|
| `SPEAKEASY_PORT` | `8080` | Listen port |
| `SPEAKEASY_HOST` | `0.0.0.0` | Listen address |
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
