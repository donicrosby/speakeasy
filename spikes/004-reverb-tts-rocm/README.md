# Spike 004: Reverb-TTS (voice cloning) on ROCm

**Question:** Given the Strix Halo host with the ROCm 10.1 pip stack speakeasy already ships, when `reverb-tts` 1.0.1 loads on the HIP device and clones a voice, does it run unmodified, say the right words, and run at a usable speed at batch 1?

**Approach:** speakeasy's `pyproject.toml` + `uv.lock` copied to a scratch dir with a `tts` extra (reverb-tts, torchaudio), `uv sync --extra rocm101 --extra tts`, run on gfx1151 from the sandbox (it can see `/dev/kfd` and `/dev/dri`). Oracle: faster-whisper large-v3-turbo on CPU transcribes Reverb's output and the word error rate against the input text is counted. Controls: fp32 on the GPU, several seeds, MIOpen on/off/fast, latency per output length, load profile.

**Key finding:** Runs unmodified. 21x to 31x real time at batch 1 with MIOpen disabled (RTF 0.03 to 0.05). Whisper round-trip: 5 edits in 211 words (2.4%), every one of them number or spelling formatting, plus the brand token "ROCm". One real gotcha: MIOpen's default find mode makes every new output length cost 1.5 to 19 seconds of kernel search in the codec decoder. `torch.backends.cudnn.enabled = False` removes it and is the fastest path.

## Results

Environment: torch 2.14.0+rocm10.1.0, transformers 5.19.0, torchaudio 2.11.1+rocm10.1.0, reverb-tts 1.0.1, AMD Radeon 8060S (gfx1151). Network bf16, codec fp32, 24 kHz mono. GPU memory 1.43 GB after load, 1.98 GB peak.

Speed at batch 1, MIOpen disabled, warm (`results/lengths_nocudnn.json`):

| Text | Audio | Elapsed | x real time | RTF |
|---|---:|---:|---:|---:|
| 12 words | 4.4 s | 0.21 s | 21 | 0.047 |
| 25 words | 8.9 s | 0.32 s | 28 | 0.036 |
| 40 words | 17.4 s | 0.58 s | 30 | 0.033 |
| 60 words | 23.3 s | 0.74 s | 31 | 0.032 |

Batch 4 (17 s texts): 11.5x real time against 11x single, both measured with MIOpen on. Batching buys nothing on this iGPU; the model is launch-bound at these sizes.

MIOpen (`results/lengths_*.json`), first take for a length not seen before, then warm:

| Mode | 17 s text, first / warm | 23 s text, first / warm | codec decode, new / warm shape |
|---|---:|---:|---:|
| default | 10.6 s / 1.56 s | 18.9 s / 2.06 s | 12.4 s / 1.1 s |
| `MIOPEN_FIND_MODE=FAST` | 0.95 s / 0.58 s | 1.13 s / 0.72 s | 0.6 s / 0.2 s |
| MIOpen off | 0.59 s / 0.58 s | 0.74 s / 0.74 s | 0.3 s / 0.2 s |

`MIOPEN_FIND_MODE=default` is not a valid value and dies with `miopenStatusUnknownError` in the codec's `conv1d`. That one cost three runs during the spike; it is recorded here so nobody reads it as MIOpen being flaky.

Oracle (`results/oracle.json`), faster-whisper large-v3-turbo, CPU int8:

| Take | Edits / words | What whisper heard |
|---|---:|---|
| reference (preset voice) | 0 / 17 | exact |
| short, cloned | 2 / 12 | "cloned on ROCm" as "cloned on Aero Team" |
| medium, cloned | 2 / 40 | "three hundred million" as "300 million" |
| long, cloned | 1 / 133 | "optimization" as "optimisation" |
| Polish, cloned | 0 / 9 | exact |
| **aggregate** | **5 / 211** | 2.4% |

"ROCm" is a text-normalization gap, not numerics: Verbalis leaves the token as "ROCm", and whisper hears "Aero Team", "Aerotem", "Rootsim" from it across four seeds and from the fp32 control alike (`results/variants_oracle.json`). Brand tokens and acronyms need a glossary or pre-normalized text.

Other facts measured:

- SDPA: the flash kernel refuses a non-null `attn_mask`, so the dispatcher picks the mem-efficient (CK) kernel for the sampler's 4D bf16 mask. The math kernel differs from it by at most 0.0156 (bf16 noise); all outputs finite.
- Same seed gives byte-identical audio on this device.
- Voice encode: 5.6 s reference becomes 141 frames; the saved voice is a 3.6 KB npz (codec codes + transcript, no audio).
- Load from cache: ~75 s in the sandbox. 55 s is the int8 dequantize on CPU, 17 s the host-to-device copy, 2 s the codec. The sandbox cgroup caps CPU at 2 cores (`cpu.max 200000 100000`, 72% of periods throttled), so the real container should be well under that. First ever load was 102 s including the 1.1 GB download (307 MB int8 weights + 806 MB codec + tokenizer).
- Lock impact: 101 to 104 packages. Added reverb-tts 1.0.1, verbalis 0.1.1, torchaudio 2.11.1+rocm10.1.0. Nothing moved. The diff is in `results/pyproject.diff`. AMD's index also lists torchaudio 2.11.1+rocm10.2.0, which sorts above the 10.1 build, hence the exact pin.
- Unrelated, found on the way: `av==18.1.0` in the lock is yanked (double free on custom channel layouts); 18.1.1 is the fix and still under the `av<19` pin.

## What this validates

- [x] reverb-tts 1.0.1 loads and generates on gfx1151 with the rocm101 stack, no code changes
- [x] Output is intelligible and faithful: whisper round-trip 2.4% WER, all formatting
- [x] 21x to 31x real time at batch 1 (MIOpen off), 2 GB of GPU memory
- [x] torchaudio resolves from AMD's index with a `torch>=2.11` floor; transformers stays at 5.19.0
- [x] Seed-reproducible takes on this device
- [ ] CPU fp32 control: abandoned, one 4.4 s clip did not finish in 15 minutes under the 2-core quota

## Verdict: VALIDATED (with the MIOpen caveat)

### What worked

- Device pick, bf16 dtype, SDPA with the additive mask, the sampler, the codec: all as-is.
- `uv sync --extra rocm101 --extra tts` resolves in one universal lock with no changes to the existing stack.
- The fp32 control on the GPU produced the same words as bf16.

### What didn't

- MIOpen default mode: a server would stall 1.5 to 19 s on any text length it has not decoded yet.
- "ROCm" and similar tokens come out as sounds whisper cannot map back to a word.
- Load is slow wherever CPU is scarce; the int8 dequantize is single-threaded CPU work.

### Surprises

- Disabling "cuDNN" (MIOpen on ROCm) is both the safe and the fast setting here: the native HIP convolution path is 2.5x to 3x faster warm and has no per-shape search.
- The batch numbers in the model card do not transfer: batch 4 gained 5%.
- Verbalis does not expand brand acronyms; the TTS reads "ROCm" as a word.

### Recommendation for the real build

- Add the `tts` extra exactly as in `results/pyproject.diff`; bump `av` to 18.1.1 while touching the lock.
- In the TTS engine, before loading: `torch.backends.cudnn.enabled = False`. Readiness = one real generate at startup, not a load.
- Budget +1.1 GB in `/models` (int8 weights, codec, tokenizer). Vendor the 16 preset voice JSONs from tag v1.0.0 into the image so presets work offline.
- Live smoke test for the endpoint: generate, then transcribe with speakeasy's own whisper and diff the words.
- Offer a glossary or accept pre-normalized text (`prepare=False`) for brand tokens.
- Weights are CC BY-NC 4.0: homelab yes, product no.

## Reproduce

```
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --frozen --no-dev --no-install-project --extra rocm101 --extra tts
SP=.venv/lib/python3.12/site-packages
LD_LIBRARY_PATH=$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib .venv/bin/python run.py out
.venv/bin/python oracle.py out            # needs out/texts.json with the expected texts
HF_HUB_OFFLINE=1 LD_LIBRARY_PATH=... MODE=nocudnn .venv/bin/python lengths.py out
```

`UV_PROJECT_ENVIRONMENT` is not optional when running from the Hermes sandbox: it exports `/opt/venv` for that variable, and a bare `uv sync` syncs the sandbox's tool venv instead of the project. Put the override on every `uv sync` / `uv run` command line.

Scripts: `run.py` (load, clone, timed generation, SDPA probe), `oracle.py` + `whisper_files.py` (whisper round-trip), `variants_gen.py` (seed and fp32 controls), `lengths.py` (MIOpen modes, per-length latency), `loadprof.py` (load profile), `probe.py` (HIP stack). Raw output in `results/`; the scripts write to `out/` (argv[1]). Audio from the run stayed in the scratch dir (`/workspace/spike-004-scratch/out/`).
