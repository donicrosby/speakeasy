> **Repository copy.** The listening set stayed out of the repository; so did the model card, the scratch audio and the host directories the *Reproduce* section names.

# Spike 005: VoxCPM2 (second TTS engine, voice cloning) on ROCm

**Question:** Given the Strix Halo host with the ROCm 10.1 pip stack speakeasy already ships, when `voxcpm` 2.0.3 loads on the HIP device and clones a voice at its high-quality default (CFG 2.0, 10 steps), does it run unmodified, say the right words, coexist with the engines already in the server, and run faster than real time?

**Approach:** The published rocm image (`ghcr.io/donicrosby/speakeasy:rocm` at `c20645e`) as a throwaway container on the host GPU with all 32 cores (not the 2-core sandbox: the uncompiled path is CPU-overhead sensitive). `voxcpm` installed into the image's own venv. Oracle: faster-whisper large-v3-turbo on CPU transcribes every take and the word error rate against the input text is counted, as in spike 004. Controls: MIOpen on/off, diffusion steps 5/10/16, `torch.compile` on/off, determinism, the library's silent retry, dependency resolution at both the install and the lock level, and all three engines in one process. Listening set: the same reference clip and the same four sentences through Reverb and VoxCPM2.

**Key finding:** Runs unmodified and is intelligible, but it is not faster than real time at high quality on this GPU: RTF 1.03 to 1.07, flat across lengths and modes, against Reverb's 0.03 to 0.05. The quality dial barely moves it (5 steps: 0.91), because the cost is the two language-model decode steps per audio patch, not the diffusion head. `torch.compile`, their 2x lever on CUDA, runs here once the image has a C compiler and is 27% slower than eager. It wants 5 to 8 GB of the APU's shared memory against Reverb's 2 GB. The owner listened to the A/B set: VoxCPM2 is much clearer and has more depth, so it goes in as an opt-in second engine (his decisions are recorded below).

## Results

Environment: the speakeasy rocm image at `c20645e` (torch 2.14.0+rocm10.1.0, HIP 7.16.26385, triton 3.8.0 rocm build, torchaudio 2.11.1+rocm10.1.0, transformers 5.19.0) plus voxcpm 2.0.3 from PyPI. AMD Radeon 8060S (gfx1151), 128 GB RAM, 32 cores, host docker daemon. Other GPU tenants were up and idle (lemonade-server, photo-ranker, the stack's speakeasy container); repeated takes agreed within 3%.

Model: 2B parameters in bf16, 4.7 GB download (4.27 GB `model.safetensors`, 0.35 GB `audiovae.pth`). One audio patch is 0.16 s (6.25 patches per second of audio); each patch runs the diffusion head `inference_timesteps` times with CFG (a batch of 2) and one decode step of each of two language models. Reference in at 16 kHz, output at 48 kHz. Load from a warm cache: 15 s, 4.95 GB of GPU memory after load.

Speed, eager, MIOpen off, high quality (CFG 2.0, 10 steps), batch 1 (`results/timing.json`):

| Take | Audio | Elapsed | RTF |
|---|---:|---:|---:|
| short, plain (12 words) | 4.0 to 4.5 s | 4.1 to 4.6 s | 1.03 |
| short, cloned | 5.1 to 5.3 s | 5.4 to 5.6 s | 1.07 |
| medium, plain (40 words) | 11.7 to 13.6 s | 12.2 to 14.3 s | 1.05 |
| medium, cloned | 15.8 to 16.6 s | 16.9 to 17.8 s | 1.07 |
| long, plain (133 words) | 39.4 to 41.0 s | 41.5 to 43.3 s | 1.06 |
| long, cloned | 47.4 to 49.0 s | 50.2 to 52.1 s | 1.06 |

Other modes, short sentence: voice design 1.04, styled clone 1.06, continuation-only clone 1.07 (labelled "ultimate" here at first; the addendum on the three clone modes below corrects that), Polish clone 1.07. RTF does not improve with length. Cloning costs about 3% over plain.

Diffusion steps, short clone, warm (`results/timing.json`, `steps_sweep`):

| Steps | RTF | Whisper errors |
|---:|---:|---:|
| 5 | 0.91 | 2 / 12 |
| 10 (default) | 1.06 | 1 / 12 |
| 16 | 1.24 | 2 / 12 |

Each step costs about 0.03 RTF; extrapolated to zero steps the pipeline still sits near 0.76. Faster than real time needs 6 steps or fewer, and every error in the table is the word "ROCm", so the oracle cannot rank the steps. The owner ranked them by ear: 5 is fine, 10 clearly better, 16 best but not by much.

Guidance strength (`cfg_value`), short clone, one seed for every take (`results/cfg_sweep.json`, `results/cfg/oracle.json`). The owner asked for cfg as a request parameter, so it needed a measured range:

| cfg | Audio | Loudness (RMS) | Whisper errors |
|---:|---:|---:|---:|
| 0.5 | 9.3 s | -28.6 dBFS | 12 / 12: the text is ignored |
| 1.0 | 6.9 s | -25.7 dBFS | 1 / 12 |
| 1.5 | 7.0 s | -18.8 dBFS | 1 / 12 |
| 2.0 (their default) | 6.6 s | -17.4 dBFS | 2 / 12 |
| 2.5 | 5.3 s | -15.3 dBFS | 1 / 12 |
| 3.0 | 5.3 s | -14.4 dBFS | 2 / 12 |
| 4.0 | 5.4 s | -11.5 dBFS | 0 / 12 |
| 5.0 | 5.4 s | -9.0 dBFS | 2 / 12 |

Time does not move: RTF 1.09 at every value. Below about 1.0 the guidance is too weak to follow the text; from 1.0 up every error is the word "ROCm". Loudness rises 17 dB from 1.0 to 5.0, and low values produce quiet takes (-26 dBFS at 1.0). The same default with two other seeds gave 4.3 s and 6.2 s of audio at -21.7 and -20.0 dBFS: on the seed alone a take's length varies by about a third and its loudness by about 4 dB, so the server cannot promise a duration for this engine.

`torch.compile` (`optimize=True`, their default: inductor, `reduce-overhead`, `fullgraph`):

- In the shipped image it dies at load, in the warm-up generation the constructor runs: Triton finds no C compiler (`python:3.12-slim` has none). Their docs say to disable it on ROCm.
- With `gcc` and `libc6-dev` added it runs, and is slower: short plain 1.31 against 1.03 eager, short clone 1.33 against 1.06, medium clone 1.34 against 1.07 (`results/compile_partial.json`). It did engage (no fallback warning, 183 MB in the inductor cache).
- It also moves cost to startup: load takes 88 s against 15 s eager, because with `optimize=True` the constructor generates a warm-up sentence and that is where the four compiled callables compile (cold inductor cache).
- The default is a trap: `optimize` defaults to True, the library's guard only catches a failure of the wrap step, and the compile error surfaces in that warm-up. In the shipped image the default therefore crashes the process at load instead of degrading. Always pass `optimize=False`.

MIOpen (`results/miopen_default.json`, `results/miopen_off.json`), same seed, so lengths repeat:

| Mode | short, first / warm | medium, first / warm |
|---|---:|---:|
| default | 15.9 s / 4.5 s | 24.7 s / 14.0 s |
| MIOpen off | 6.8 s / 4.7 s | 14.4 s / 14.2 s |

Same trap as spike 004: the audio VAE decodes the whole utterance in fp32 convs, and every output length not seen before costs a 10 to 11 s kernel search. `torch.backends.cudnn.enabled = False` before loading removes it.

First request after load: 5.7 s for 4.3 s of audio (plain); the first clone pays 9.3 s for 6.2 s (reference encode plus warm-up).

Memory: 4.95 GB after load, 5.4 GB peak on short takes, 7.9 GB on a 49 s take (allocated); reserved grows to 11.1 GB over a long session. This is the APU's shared memory.

Oracle (`results/oracle.json`), faster-whisper large-v3-turbo, CPU int8, 10 takes in 7 modes:

| Take | Edits / words | What whisper heard |
|---|---:|---|
| short, cloned | 1 / 12 | "ROCm" as "Rakim" |
| short, plain; design; styled | 0 / 12 each | exact |
| medium, cloned | 2 / 40 | "three hundred" as "300" |
| long, cloned | 1 / 133 | "coarse" as its homophone "course" |
| Polish, cloned | 0 / 9 | exact |
| **aggregate** | **9 / 266 (3.4%)** | all brand-token, number formatting or a homophone |

Determinism: 2.0.3 has no `seed` argument (the GitHub README shows one; the PyPI `_generate` does not take it). The sampler draws `torch.randn` on the device, so `torch.manual_seed(n)` right before the call gives byte-identical takes (verified), and a different seed differs.

Silent retry: the library regenerates a take up to 3 times when audio patches exceed 6 per text token. Normal takes here land at 1.45 to 1.62, and it never fired (`results/retry_probe.json`).

Coexistence (`results/coexist.json`): Whisper large-v3-turbo (HIP CTranslate2, fp16), Reverb (int8) and VoxCPM2 loaded in one process, then interleaved in both orders for two rounds. Alive after every step; 7.0 GB allocated, 7.6 GB reserved with all three loaded. In that process: Reverb 0.22 s for 4.6 s of audio, VoxCPM2 5.96 s for 5.8 s.

Dependencies:

- Naive install into the image's venv (`results/install-dryrun-summary.txt`): resolves in 6.5 s, adds 100 packages, and downgrades one: fsspec 2026.9.0 to 2025.3.0, because `datasets<4` caps it. fsspec is what huggingface_hub loads Whisper and Moonshine through. No other existing pin moves.
- `torchcodec` is a hard dependency with no ROCm build. Its PyPI wheel installs and fails to load against HIP torch. Worse, torchaudio 2.11's `load` is torchcodec-only, so with torchcodec present `torchaudio.load` raises for the whole process (their FAQ's ROCm workaround). Without it, `torchaudio.load` raises "TorchCodec is required" instead: it was already unusable in this image and nothing in speakeasy calls it. VoxCPM2 reads its reference with `librosa.load`, so the path we need is unaffected.
- voxcpm's metadata pulls packages its runtime never imports: gradio, funasr, spaces are imported nowhere in the wheel; datasets and argbind are training-only; modelscope is the optional denoiser's.
- Lock level (`results/pyproject.diff`, `results/req-*.txt`): a `tts-voxcpm` extra plus `override-dependencies` entries with `sys_platform == 'never'` for those packages and torchcodec. `uv lock` in 4 s (uv 0.12.5): 104 to 192 lock entries, zero existing versions changed. On this platform the extra then adds 40 packages (84 to 122) and fsspec stays at 2026.9.0.

### Addendum: the three clone modes (measured 27 November, after the build lane started)

**Correction.** The modes run above labelled the continuation-only clone "ultimate". That was this harness's label, not the vendor's. The model card (`model-card.md:135-144`) defines Ultimate Cloning as reference audio plus its exact transcript, and says: "Pass the same clip to both `reference_wav_path` and `prompt_wav_path` for highest similarity." That is the library's combined mode (`model/voxcpm2.py:484-515`, cache mode `ref_continuation`): the reference is isolated behind its own audio tokens, and the prompt audio is continued in context. `run.py` never ran it. The build uses it for every clone saved with a transcript.

The library has four modes, decided by which of the three reference arguments are set:

| Mode | Arguments | What the model does |
|---|---|---|
| zero-shot | none | the model's own voice |
| reference-only | `reference_wav_path` | voice cloning: the reference sits in an isolated prefix |
| continuation-only | `prompt_wav_path` and `prompt_text` | in-context continuation of the prompt audio |
| combined | all three | both: an isolated reference and an in-context prompt |

**Speed, paired** (`modes_pair.py`, `results/modes_pair.json`). A first pass (`modes.py`, `results/modes.json`) ran while two build workers shared the host, and its combined takes on the medium text came out at 1.540 and 1.108: too far apart to quote. The second pass alternates the two modes the server uses, same text and same seed within a pair, three pairs per length, so drift in host load lands on both alike. Warm takes, `retry_badcase` off, MIOpen find mode off, host load about 5 on 32 cores.

| Length | Audio | Reference-only, mean of 3 | Combined, mean of 3 | Combined minus reference-only, per pair |
|---|---:|---:|---:|---|
| short (12 words) | 5.3 s | 1.141 | 1.127 | −0.044, −0.024, +0.028 |
| medium (40 words) | 15.8 s | 1.046 | 1.097 | +0.011, +0.034, +0.108 |
| long (133 words) | 47 s | 1.140 | 1.106 | −0.077, −0.031, +0.004 |

The mean of the nine differences is +0.001. The spread between pairs (−0.08 to +0.11) is as wide as the spread between repeated takes of one mode (reference-only on the long text: 1.059 to 1.264), so this run measures no cost for the combined mode. The quietest pairs sit at 1.02 to 1.09 in both modes; the timing table at the top of this record, taken on a quieter host, gives 1.03 to 1.07. Peak memory is the same in every mode: 5.7 GB short, 6.2 GB medium, 7.7 to 8.2 GB long.

**Not measured: which mode clones better.** Nothing on this host scores speaker similarity, and that call is the owner's ear. The listening set has one reference-only take per length and no combined-mode take. `ab-modes/` holds one warm take of each mode per length for whoever wants to listen. The vendor's claim that the combined mode gives the highest similarity is a claim, not a measurement.

## Head-to-head: Reverb-TTS vs VoxCPM2 on this GPU

| Dimension | Reverb-TTS (shipped) | VoxCPM2 |
|---|---|---|
| RTF, high quality | 0.03 to 0.05 (21x to 31x real time) | 1.03 to 1.07 (about real time) |
| Fastest usable setting | as-is | 5 steps: 0.91 |
| GPU memory | 1.4 to 2.0 GB | 5 to 8 GB |
| Load (warm cache, GPU) | about 7 s in the container | 15 s |
| Output | 24 kHz | 48 kHz |
| Languages | 7 | 30 |
| Cloning | voice encoded once, saved as a 3.6 KB npz | reference re-encoded on every request; a saved clone would be a stored wav |
| Extras | 16 preset voices | voice design from a description, style hints, transcript-assisted cloning |
| Determinism | `seed` parameter | global `torch.manual_seed` |
| License (weights) | CC BY-NC 4.0 | Apache-2.0 |
| Dependency footprint | +3 packages | +40 with overrides (+100 raw, one downgrade) |
| `torch.compile` | n/a | 27% slower than eager here |

**Winner on speed and footprint:** Reverb, by 20x. **Winner on license, languages, output rate and cloning features:** VoxCPM2. Quality of the clone: the listening set decides.

## What this validates

- [x] voxcpm 2.0.3 loads and generates on gfx1151 with the rocm101 stack, no code changes
- [x] Output is intelligible: 3.4% WER, every error a brand token, a number format or a homophone
- [x] Reproducible takes through `torch.manual_seed`
- [x] Coexists with Whisper and Reverb in one process
- [x] Dependency footprint controllable at the lock level with no downgrade to the shipped stack
- [ ] Faster than real time at high quality: no (1.05x real time)
- [ ] `torch.compile` as a speedup: no (slower)
- [ ] Streaming (`generate_streaming`) and time to first audio: not measured

## Verdict: PARTIAL

### What worked

- Device pick, bf16, the sampler, both language models, the audio VAE: all as-is.
- One line fixes the only performance trap: `torch.backends.cudnn.enabled = False` before load.
- Speed is predictable: RTF is flat across length and mode, so capacity planning is trivial (one request at a time, about real time).

### What didn't

- Faster than real time at high quality. The 4090's 0.30 comes from a compiled path that does not pay off here.
- `torch.compile`: crashes at load in the shipped image; with a compiler it is slower than eager and adds 73 s to startup.
- A lean install out of the box: it takes lock overrides to keep 60 unused packages and an fsspec downgrade out.

### Surprises

- The diffusion steps are not the cost: 10 steps cost 0.03 RTF each, the two LM decode steps are the rest.
- Compiling made it slower, by more than the diffusion dial can recover.
- The GitHub README documents a `seed` argument the released package rejects.
- The iGPU beats the published CPU-only number for this same chip class (RTF 1.6 to 1.9) by about 1.7x, and still only reaches real time.

### Recommendation for the real build

- Decide on the listening set, not on the model card. If VoxCPM2's clone is clearly better for the owner's use, add it as a second engine; if not, keep Reverb alone.
- If added: opt-in and lazily loaded (`SPEAKEASY_TTS_ENGINES` or load on first request), because 5 to 8 GB stay resident on a shared APU and every request takes about the audio's duration. Never a replacement for Reverb.
- Engine selection through the speech request's `model` field. A VoxCPM2 "saved clone" is a stored reference wav (no encode step); the `.npz` store stays Reverb's.
- `cudnn.enabled = False` before load; `optimize=False` always (it crashes at load without a compiler and is slower with one); `retry_badcase=False` on the server path (or surface a retry in the response); seed with `torch.manual_seed` under the engine lock.
- The `tts-voxcpm` extra exactly as `results/pyproject.diff`, including the overrides. Never install torchcodec.
- If a streaming feel matters, measure `generate_streaming` and time to first audio before building; that path exists and was not tested here.
- For longer texts, expect about 1 s of wall time per second of audio, and cap `input` for this engine accordingly.

## Owner's verdict and decisions (2026-11-27, from the listening set)

- Quality: VoxCPM2 is much clearer and has more depth than Reverb on the cloned pairs. The reference was a voice Reverb generated itself, which favours Reverb, so a clear VoxCPM2 win is robust. Go: integrate as an opt-in second engine.
- Loading: on first request, unloaded after 10 minutes idle.
- Scope: plain speech, clone, voice design, style hints and transcript-assisted cloning.
- Steps: fixed at 10.
- `cfg`: exposed as a request parameter. Measured range above; the default stays their 2.0 until the owner rules by ear on the cfg clips.
- `input`: the same 4096-character cap as Reverb, about four minutes of waiting at worst.

## Reproduce

```
# container from the published image, GPU passed through, scratch dir mounted at /work
docker run -d --name spike5 --device=/dev/kfd --device=/dev/dri \
  -v /opt/data/workspace/spike-005-scratch:/work -e HF_HOME=/work/hf \
  --entrypoint sleep ghcr.io/donicrosby/speakeasy:rocm-c20645e infinity
docker exec spike5 /work/bin/uv pip install --python /app/.venv/bin/python --dry-run voxcpm
docker exec spike5 /work/bin/uv pip install --python /app/.venv/bin/python voxcpm
docker exec spike5 /work/bin/uv pip uninstall --python /app/.venv/bin/python torchcodec
docker exec spike5 /app/.venv/bin/python /work/run.py --out /work/out/timing --cudnn off --retry 0 --phase load,timing,modes
docker exec spike5 /app/.venv/bin/python /work/miopen_probe.py default
docker exec spike5 /app/.venv/bin/python /work/coexist.py
docker exec spike5 /app/.venv/bin/python /work/cfg_sweep.py
docker exec -e HF_HUB_OFFLINE=1 spike5 /app/.venv/bin/python /work/modes.py        # the three clone modes, first pass
docker exec -e HF_HUB_OFFLINE=1 spike5 /app/.venv/bin/python /work/modes_pair.py   # paired timing of the two the server uses
/workspace/spike-004-scratch/.venv/bin/python oracle.py out/timing     # needs out/timing/oracle_texts.json
/usr/bin/python3 ab.py                                                  # the listening set
```

The harness writes `result.json` into its `--out` directory after every step, so a crash keeps what came before. Audio stays in the scratch dir (`/workspace/spike-005-scratch/out/`, listening set in `ab/`).

Scripts: `run.py` (load, smoke, determinism, timing matrix, steps sweep, modes), `miopen_probe.py` (MIOpen on/off, first vs warm per length), `coexist.py` (three engines, one process), `modes.py` (the three clone modes, first pass), `modes_pair.py` (paired timing of the two the server uses), `retry_probe.py` (the silent retry), `cfg_sweep.py` (the guidance-strength sweep), `oracle.py` (whisper round-trip), `ab.py` (listening set and `listen.html`). Raw output in `results/`.
