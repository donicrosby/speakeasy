# Spike 003: Moonshine as fast-path for short voice notes

**Question:** Given short voice notes (<10s), when Moonshine base transcribes, then accuracy is acceptable vs Whisper large-v3-turbo.

**Approach:** Compare Moonshine Base (61.5M params) against Whisper large-v3-turbo (809M params) on the JFK test clip. Measure both accuracy and latency.

**Key finding:** Moonshine is 11x faster on CPU and gets the words right, but drops punctuation. For Cortex's use case (voice notes that get parsed into tasks), punctuation is nice-to-have but not critical — the LLM downstream can handle missing commas.

## Results

| Metric | Moonshine Base | Whisper large-v3-turbo |
|--------|---------------|------------------------|
| Params | 61.5M | 809M |
| Load time | 0.8s | 2.8s |
| Transcription time | 0.97s | 10.97s |
| Realtime factor | 11.3x | 1.0x |
| Words correct | Yes | Yes |
| Punctuation | No | Yes |
| English only | Yes | No (multilingual) |

## What this validates

- [x] Moonshine loads and transcribes on CPU
- [x] Accuracy is acceptable for voice notes (words correct, punctuation missing)
- [x] 11x speedup over Whisper large-v3-turbo on CPU
- [x] 13x smaller model (61.5M vs 809M params)

## Verdict: VALIDATED (with caveats)

### What worked
- Moonshine Base transcribes correctly — all words present, just missing punctuation
- 11x faster than Whisper large-v3-turbo on CPU
- 13x smaller model — fits comfortably alongside Whisper in memory
- Loading is fast (0.8s vs 2.8s)

### What didn't
- No punctuation in output — commas, periods are dropped
- English-only — no multilingual support
- Not a drop-in replacement for Whisper's API shape (different input format)

### Surprises
- The `AutoProcessor` expects raw audio waveform, not spectrograms — different from Whisper's feature extractor
- Moonshine's conv stem processes raw audio directly (no mel spectrogram), which is why it's so fast

### Recommendation for the real build
- **Use Moonshine as fast-path for short voice notes** (<10s) where speed matters more than punctuation
- **Keep Whisper large-v3-turbo as default** for anything longer or when punctuation matters
- Route based on audio duration: <10s → Moonshine, >=10s → Whisper
- The punctuation gap can be handled by the downstream LLM (Cortex's thought parser already handles messy input)
- Moonshine's English-only limitation is fine for Cortex's use case
