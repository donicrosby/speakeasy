# Spike 002: VAD + denoise pipeline

**Question:** Given noisy audio, when MetricGAN+ denoise runs before Whisper transcription, then accuracy improves vs VAD-only.

**Approach:** faster-whisper (with built-in Silero VAD) + SpeechBrain MetricGAN+ enhancement. Test at 10dB, 5dB, 0dB SNR.

**Key finding:** faster-whisper already has Silero VAD built in (`vad_filter=True`). The question is whether additional MetricGAN+ denoising helps for genuinely noisy audio (G2 glasses recordings, outdoor environments).

## What this validates

- [x] MetricGAN+ loads and runs on CPU
- [x] Denoise → VAD → transcribe tested vs VAD-only on noisy audio
- [x] Latency overhead of denoise step measured
- [x] Pipeline works end-to-end

## Verdict: PARTIAL

### What worked
- MetricGAN+ loads and runs on CPU without issues
- At 10dB and 5dB SNR, both VAD-only and Denoise+VAD produce correct transcriptions
- Latency overhead of denoise is minimal (~0.2s on 11s audio)

### What didn't
- At 0dB SNR (very noisy), MetricGAN+ denoise actually **degraded** transcription accuracy
- VAD-only: "And so, my fellow Americans, ask not what your country can do for you..."
- Denoise+VAD: "Oh my God, that's not what your country can do to you..."
- The denoiser introduced artifacts that confused Whisper more than the original noise

### Surprises
- faster-whisper's built-in Silero VAD is surprisingly robust on its own
- MetricGAN+ is optimized for speech enhancement (human perception), not ASR preprocessing
- White noise is easier for Whisper to handle than denoiser artifacts

### Recommendation for the real build
- **Don't add MetricGAN+ denoise as a default preprocessing step** — it can hurt accuracy on very noisy audio
- **Do keep VAD enabled** (`vad_filter=True`) — it's already built into faster-whisper and helps
- If denoise is needed later, make it opt-in per-request (`?denoise=true`), not default
- Consider testing with real-world noise (wind, traffic, cafe) rather than synthetic white noise
- The built-in VAD + Whisper large-v3-turbo is already quite robust for voice notes
