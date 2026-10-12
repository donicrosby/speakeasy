# Spike 001: faster-whisper on ROCm

**Question:** Given the Strix Halo host with ROCm, when faster-whisper large-v3-turbo transcribes a WAV via an OpenAI-compatible endpoint, then it works and the API shape matches what Cortex's `OpenAiProvider` already speaks.

**Approach:** FastAPI + faster-whisper (CTranslate2). CPU for the sandbox spike, ROCm in the Docker container via the official CTranslate2 ROCm wheel.

**Key finding:** CTranslate2 v4.7.0+ added ROCm support (PR #1989). The ROCm wheel still exposes `device="cuda"` — it's HIP-backed, not CUDA-backed. Same API, different runtime. The nabe2030 recipe confirms gfx1151 works with ROCm 7.2.2 + CTranslate2 4.7.1 official wheel at ~11.5x realtime for large-v3.

## What this validates

- [x] OpenAI `/v1/audio/transcriptions` response shape matches
- [x] `response_format` values: `json`, `verbose_json`, `srt`, `vtt`, `text`
- [x] Model loads and transcribes on CPU (sandbox)
- [ ] Model loads and transcribes on ROCm (container — validated in Docker build)
- [x] Latency is acceptable for Cortex use case (<5s for 30s voice note — 11s audio in ~2s on CPU)

## Verdict: VALIDATED (CPU path)

### What worked
- faster-whisper 1.2.1 + CTranslate2 4.8.2 transcribes correctly on CPU
- OpenAI API shape: `/v1/audio/transcriptions` with multipart form data works
- All 5 response formats (`json`, `verbose_json`, `srt`, `vtt`, `text`) return correctly
- `/v1/models` endpoint returns the expected list shape
- 11s JFK clip transcribes in ~2s on CPU (5.5x realtime) — acceptable for voice notes

### What didn't
- PyAV 19.x breaks faster-whisper 1.2.1 (`metadata_errors` kwarg removed) — pinned `av<19`
- `io.BytesIO` doesn't work reliably for all audio formats — temp file workaround needed

### Surprises
- CTranslate2's ROCm wheel still uses `device="cuda"` in the API — it's HIP-backed, not CUDA-backed. Same code path, different runtime. This means the CUDA and ROCm Docker images can share the same application code.

### Recommendation for the real build
- Pin `av<19` in dependencies until faster-whisper catches up
- Use temp files for audio ingestion, not BytesIO
- For ROCm: install CTranslate2 from the official ROCm wheel (v4.7.1+), not PyPI
- For CUDA: standard PyPI ctranslate2 works
- Same FastAPI app code works for both — the only difference is which ctranslate2 wheel gets installed
