# Speakeasy VoxCPM2 (second TTS engine)

> Spec v1.4 (rev4 — model required and listed, voice length cap, no default engine). Validated against `voxcpm` 2.0.3 (PyPI wheel, `openbmb/VoxCPM2` weights), spike 005 (gfx1151), speakeasy `main` `c20645e`. Contract for Reverb: `docs/specs/tts-voice-cloning-ui.md` — this spec extends it; one row is superseded (the `model` row of its §4.2 speech request table).

### Revision summary (r4)

**Owner's ruling (2026-11-27) — `model` required and listed, `voice` length cap, no default engine.** His words, verbatim:

> Yes add cap
> I want you also return 404 if model isn't listed. There should be no default model.

The first line answers the question put to him: cap the speech request's `voice` at 64 characters. Three changes, binding:

- `model` is required on speech and clone-create (5.1, 5.3). Its value must be a listed TTS engine id — `reverb-tts` or `voxcpm2`. Missing/empty → 422. Unlisted → 404 `unknown model`. The check runs before any engine gate, reads no state, takes no lock. No default — a typo'd or empty `model` no longer falls through to Reverb.
- `voice` capped at 64 characters (5.1 request table). A saved clone id is at most 64 characters and the longest preset id is 17, so no valid voice is excluded; the bound keeps an unbounded string from entering a body that names the unknown voice.
- `GET /v1/models` lists TTS engine ids by **configuration**, not load state (5.5, state table): `reverb-tts` whenever `SPEAKEASY_TTS_ENABLED` is true, `voxcpm2` whenever `SPEAKEASY_TTS_VOXCPM_ENABLED` is true. A listed engine that is not ready keeps its 503s; a disabled engine is absent from the listing and answers 404.
- MF-22's `disabled` 503 for speech and clone-create is superseded — disabled engines are unlisted and answer 404 (§5.1, state table). Delete of a VoxCPM2 clone while disabled stays 404 "no saved voice"; the voice listing keeps its gate.
- One coder slice added (§13): `feat/voxcpm2-model-required`. The Reverb spec gets one supersession note on its §4.2 `model` row.
- Reviewer mechanical completions (rev4 review): the Reverb-side state table's `disabled` row corrected to the ruling (speech and clone create 404, delete 503) and its heading's "unchanged" dropped; the owner's words quoted verbatim above; §12.10's clone-create cells completed (typo'd `model` on every engine state, empty `model` 422); the hostile-id test's adaptation to the voice cap named; cross-references fixed to point at the Reverb spec's §4.2 row.

### Revision summary (r3)

**MF-25 — delete gate keeps Reverb's pinned contract.** The VoxCPM2-store half answers when the store exists (filesystem operation — no engine needed) and the state is not `disabled`; the Reverb-store fall-through gates on `tts_unavailable()` exactly as today — a delete of a Reverb clone while Reverb is not ready answers Reverb's fixed 503 in every VoxCPM2 state, no Reverb status change. §5.4 states it once; the state table's delete column, §7.5, and §10 agree.

**MF-26 — idle-unload timer guarded by engine state; loader resets last-used at completion.** The timer unloads only when `voxcpm2_state == 'ready'` — a tick in any other state skips. The loader sets `voxcpm2_last_used = time.monotonic()` at load completion so the idle clock measures from the ready transition. Two race rows added to §7.5: (a) timer tick during load (pre-`tts_lock` phase) guarded by the state check — no clobber of `loading`; (b) timer tick just after load completes guarded by the fresh last-used — no reload thrash. Both interleavings get a §12.4 scenario and test (`test_voxcpm2_timer_skips_during_load`, `test_voxcpm2_timer_skips_after_load`).

### Revision summary (r2)

**MF-13 — listing restores Reverb's `languages` and 503 strings (O-R7a).** `languages` present whenever Reverb is ready, omitted otherwise; the page keeps its content and refills. Pinned 503 strings stay byte-identical when the 503 is Reverb's alone.

**MF-14 — clone-create gating locked to the state table (O-R1a).** `not_loaded`/`loading`/`error` → 503 with that state's reason; load starts in `not_loaded`; 201 only in `ready`. §5.3, §12.7, and the state table now agree.

**MF-15 — voice-listing gate stated once (O-R8a).** 200 whenever voxcpm2 is enabled (any state; clones from disk; presets only while Reverb ready) or Reverb is ready; 503 only when voxcpm2 disabled and Reverb not ready. §5.2, §6, §12.8 agree.

**MF-16 — last-used bump removed from clone-create, listing, and delete (O-R4a).** Only VoxCPM2 speech bumps `voxcpm2_last_used`. State table ready row corrected.

**MF-17 — fake module pinned to `(audio, bad_case)` tuple (R12/SEC-V8).** §12.2 asserts the tuple shape; §12.6 drives the stub to return a bad tuple and asserts the route's 503 mapping.

**MF-18 — clone-create decode path named (O-R6a).** `faster_whisper.audio.decode_audio(data, sampling_rate=16000)`; written with `soundfile.write(..., 16000, subtype='PCM_16')`; Reverb's limits and fixed 400s unchanged.

**MF-19 — load trigger uses dedicated state lock (O-R2a/O-R9a).** Starting a load never takes `tts_lock` and never skips. A shared helper holds a small dedicated `voxcpm2_state_lock` for the `not_loaded`→`loading` check-and-set; the first caller that wins it spawns the loader. Speech and clone-create both call the helper. The loader thread takes `tts_lock` for the load itself; while it waits behind a Reverb take the state is `loading` and requests answer the loading 503. §12.4 extended to create+speech pair.

**MF-20 — §7.4 split into per-engine protocols (O-R4a).** VoxCPM2 speech: gate → acquire → re-check → seed → generate → map bad_case → bump `voxcpm2_last_used`. Reverb speech: Reverb's own gate → acquire `tts_lock` → synthesize. Only VoxCPM2 speech bumps `voxcpm2_last_used`.

**MF-21 — rule 7 and leading-`(` 400 get exact JSON messages; rule 8 added to §5.1 (O-R13a).**

**MF-22 — 503 bodies pinned once in §5.1; page maps states to its own strings (like Reverb's `app.js:17-18`). `disabled` body added to §5.1 (O-R7a).**

**MF-23 — README recovery note added to c8 contract.**
**MF-24 — 2-minute poll cap removed; 5-second poll continues while server reports `loading` (O-R3a).**

## 1. Goals and non-goals

### Goals

Add VoxCPM2 as a **second, opt-in TTS engine** next to Reverb. A client selects an engine per request with the `model` field (required): `"reverb-tts"` for Reverb, `"voxcpm2"` for VoxCPM2.

VoxCPM2 brings: 48 kHz output, 30 languages with no language tag, a voice designed from a written description (`design`), style hints on a clone (`style`), and cloning that continues the reference audio when its transcript is known (ultimate / transcript-assisted cloning). Its cost is roughly real time per request (Reverb is 20 to 30 times faster) and 5 to 8 GB of the GPU's shared memory while loaded.

### Non-goals

- Audio streaming (chunked transfer, Server-Sent Events).
- SSML markup.
- Playback-rate control for this engine — `speed` other than 1.0 is refused.
- `torch.compile` — passed as `optimize=False` always; the shipped image has no C compiler, and even with one it is 27% slower than eager on this GPU.
- Voice training, fine-tuning, or LoRA loading.
- User accounts, per-client API keys, or multi-tenancy beyond the existing single bearer token.
- Opus or OGG output.
- UI frameworks, a build step, a bundler.
- Gradio integration.
- Audio loudness normalization — output loudness varies with `cfg` and the seed; the owner wants to hear it before a decision.
- CUDA runtime verification — the ROCm host covers it; the CUDA image's sanity step imports the engine but has no NVIDIA GPU.
- Any change to Reverb's contract (`docs/specs/tts-voice-cloning-ui.md`), to the transcription endpoints, or to the Reverb clone files on disk (`.npz`).
- Deploying to the owner's stack — a separate go.

## 2. Owner's decisions (settled)

Copied verbatim from the brief's decisions table. The spec restates them; it does not reopen them.

| #  | Decision            | Owner's words (2026-11-27)                                                                                                                                                                                                                                                              |
|----|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| D1 | VoxCPM2 goes in, as a second engine | "the VoxCPM2 voice is much clearer and has more depth than the Reverb voice"                                                                                                                                                                                           |
| D2 | Diffusion steps     | "5 is fine but 10 is clearly better, 16 is best but not by much"                                                                                                                                                                                                                         |
| D3 | Loading             | "Load on first request, unload after 10 min idle: frees the memory between uses, and the next request after idle pays the reload"                                                                                                                                                         |
| D4 | Scope               | "Everything: the above plus transcript-assisted cloning for the closest match" (plain speech, voice clone, voice design, style hints, transcript-assisted clone)                                                                                                                          |
| D5 | Guidance strength   | "Allow cfg to be a parameter passed to the API."                                                                                                                                                                                                                                         |
| D6 | Input length        | "Same 4096 as Reverb: up to about four minutes of waiting"                                                                                                                                                                                                                               |

## 3. Orchestrator's rulings (restated as requirements)

Each ruling from the brief's section 5, restated as a requirement with the check that proves it.

| #  | Requirement                                                                                                                                                                                                                                                                           | Check                                                                                              |
|----|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| R1 | **Lazy and asynchronous.** A `voxcpm2` request (speech **or** clone-create) while the engine is `not_loaded` starts one background load (single-flight) and answers 503 at once — it never waits. The page polls `/health/ready` and sends once the state is ready.                 | `test_voxcpm2_not_loaded_triggers_load_and_503` — a speech request returns 503 immediately; the loader starts. |
| R2 | **Idle unload.** While `ready`, every **speech** request updates a last-used time under the TTS lock. A daemon timer (every 15 s) unloads when `now - last_used` exceeds `tts_voxcpm_idle_seconds`, and only if it can take the TTS lock without waiting. `0` disables the timer. Clone-create and voice listing do not bump last-used — only model-running requests do. | `test_voxcpm2_idle_unload` — a mock last-used time older than the timeout causes unload; a request during unload gets the `not_loaded` 503. |
| R3 | **One TTS lock for both engines**, created at startup in `server.py` and held by every speech request around its engine call. Clone-create does not take this lock — it uses a dedicated create lock. Reverb's own `RLock` inside `ReverbEngine` stays.                              | `test_shared_tts_lock_serialises_both_engines` — two concurrent requests (one `voxcpm2`, one Reverb) never run engine inference at once. |
| R4 | **Engine gates its own requests.** A `voxcpm2` request does not depend on Reverb being loaded, and a Reverb request does not depend on VoxCPM2. The voice listing answers when either is enabled. The VoxCPM2 clone store must not be created by Reverb's loader.                     | `test_voxcpm2_requests_work_with_reverb_disabled` and `test_reverb_requests_work_with_voxcpm2_disabled`. |
| R5 | **Selecting the engine.** `model` is required on both routes that take it: speech (§5.1) and clone create (§5.3). Its value must be a TTS engine id this deployment lists in `GET /v1/models` — `reverb-tts` selects Reverb, `voxcpm2` selects VoxCPM2. There is no default: a missing value is rejected, and no other value ever falls through to an engine. | `test_model_field_selects_engine` — `"reverb-tts"` routes to Reverb, `"voxcpm2"` routes to VoxCPM2; missing/empty/unlisted → 422/404. `test_models_lists_both_tts_engines`. |
| R6 | **`voice` under `voxcpm2`.** `"default"` is the model's voice for the seed. Any other id must be a saved VoxCPM2 clone. A Reverb preset id is a 400 that names `default` and the saved clones. `language` is accepted and ignored. `seed` seeds the take (under the TTS lock). An omitted `voice` under `voxcpm2` means `"default"`.      | `test_voxcpm2_voice_default_and_clone` — `"default"` works; a saved clone works; a Reverb preset id returns 400. |
| R7 | **`speed` under `voxcpm2`** must be 1.0 — anything else is a 400 that says the engine has no rate control.                                                                                                                                                                            | `test_voxcpm2_speed_rejected` — `speed: 1.5` returns 400.                                          |
| R8 | **`cfg`, `design`, `style` with Reverb.** A value other than the default (`cfg` != 2.0 or non-null string) is a 400 that names `voxcpm2`. `cfg` outside 1.0 to 5.0 is a validation error.                                                                                             | `test_voxcpm2_fields_rejected_on_reverb` — sending `cfg` or `design` or `style` to Reverb returns 400. |
| R9 | **`design` and `style` constraints.** At most one. `design` only with `voice: "default"`. `style` with `"default"` or a clone. Neither may contain `(`, `)` or a newline — that is a 400. The server builds the prefix `"(" + value + ")" + text`. `input` whose first character (after stripping) is `(` is a 400 that names `design` and `style`. | `test_voxcpm2_design_or_style_validation` — rejects `(` / `)` / `\n`; rejects both `design` and `style`; rejects `design` with a non-default voice; rejects `input` starting with `(`. |
| R10 | **Clone create — store only, no model call.** `POST /v1/audio/voices` with `model=voxcpm2` stores the reference as 16 kHz mono WAV and a metadata JSON. `text` is optional. The model runs at speech time, not at clone-create time.                                                  | `test_voxcpm2_clone_stores_reference` — the clone create succeeds with `model=voxcpm2`; `.wav` and `.json` files exist; no VoxCPM inference ran. |
| R11 | **Ids unique across engines.** A VoxCPM2 clone under an id that exists as a Reverb clone (or the reverse) is 409. `DELETE` removes the clone from whichever engine owns it and returns `"engine"` in the body.                                                                         | `test_clone_id_unique_across_engines` — creating a VoxCPM2 clone under a Reverb id returns 409; a delete returns the engine. |
| R12 | **Bad take → 503, never a silent retry.** `VoxCPM2Engine.generate` passes `retry_badcase=False` to the library, computes the audio-patch count (`floor(audio_seconds / 0.16)`, where 0.16 s is the model's patch duration from spike README "Results") and the text-token count via the model's tokenizer over the same flattened text the library generates from (V-fact: `voxcpm-src/model/voxcpm2.py:633`). If `patches >= 6 * tokens`, the take is bad. `generate` returns `(audio, bad_case)`; the route maps `bad_case` to 503. | `test_voxcpm2_bad_take_503` — a stub that simulates a bad take returns 503 with a fixed reason. |
| R13 | **Fixed library arguments.** `load_denoiser=False`, `optimize=False`, `retry_badcase=False`, `inference_timesteps=10`, `max_len=4096`, `min_len=2`, no text normalisation, `denoise=False`. MIOpen disabled on HIP as Reverb does.                                                     | `test_voxcpm2_constructor_args` — the fake module records every argument passed to `from_pretrained` and `generate`. |
| R14 | **Encoding at 48 kHz.** `wav`, `mp3`, `flac`, `pcm` at the engine's sample rate (Reverb: 24 kHz; VoxCPM2: 48 kHz). `pcm` is raw s16le mono.                                                                                                                                            | `test_voxcpm2_response_formats` — each format's byte count matches 48 kHz.                          |
| R15 | **Both images get the engine.** The `tts-voxcpm` extra installs beside `tts` in `Dockerfile.cuda` and `Dockerfile.rocm`. The ROCm image runs a sanity step; the CUDA image imports the engine and nothing more.                                                                        | CI: `test_lock_default.py` covers the extra; Docker build at landing verifies the sanity steps.      |
| R16 | **No audio normalisation.** Output loudness varies. Changing it is a follow-up for the owner.                                                                                                                                                                                          | The encode path applies no gain or normalisation.                                                   |

## 4. Verified facts

Every fact is cited at an exact version and every citation was re-read before marking.

### VERIFIED — speakeasy `main` `c20645e`

- **S1 — TTS architecture.** `src/speakeasy/server.py:81-84` — one global `tts` (ReverbEngine or None), `voices_store`, `tts_state` in `disabled | loading | ready | error`. `server.py:124-142` (`start_tts`) spawns `_load_tts` on a daemon thread; the store is published before the engine.
- **S2 — TTS gate.** `server.py:170-179` — `tts_unavailable()` returns 503 with a fixed per-state reason; gates every TTS route.
- **S3 — Health and models.** `server.py:212-225` — `GET /health/ready` reports `"tts": tts_state`. `server.py:239-247` — `GET /v1/models` lists `"reverb-tts"` only while `tts is not None`.
- **S4 — Voice list.** `server.py:250-268` — `_voice_objects()`: presets sorted by id then clones sorted by id; clones carry `type`, `language`, `transcript`, `created`.
- **S5 — SpeechRequest.** `server.py:279-286` — `input` (max 4096), `voice` (default `"en_male_narrator_1"`), `response_format`, `speed` (0.25-4.0), `model` (default `""`, accepted), `language`, `seed`. Pydantic drops unknown fields by default.
- **S6 — Speech routing.** `server.py:324-363` — `speech()` resolves voice via `_resolve_voice()`, checks language, calls `tts.synthesize()`, encodes at `tts.sample_rate`.
- **S7 — Reference decode.** `server.py:370-384` — `_decode_reference()` via `faster_whisper.audio.decode_audio` to `tts.sample_rate`; 20 MB cap, 1 to 30 s duration.
- **S8 — Clone create.** `server.py:387-461` — multipart `file`, `text` (required), `language`, `voice_id`. Clone runs under `tts._lock`; 201 body built without reading the file back.
- **S9 — Clone delete.** `server.py:464-473`, `src/speakeasy/voices.py:115-125` — 404 when nothing is there; concurrent deletes of the same clone both report success.
- **S10 — VoiceStore.** `src/speakeasy/voices.py:16-102` — `^[a-z0-9][a-z0-9-]{1,63}$`, `fullmatch` before any path; `listing()` one-pass sorted; `save()` atomic with `os.replace`.
- **S11 — ReverbEngine.** `src/speakeasy/engines.py:148-213` — `from_pretrained`, disables `cudnn` on HIP, holds one `RLock` over `synthesize` and `clone`.
- **S12 — Test faking pattern.** `tests/test_tts.py:112-116` — `sys.modules` fake; `tests/test_tts.py:121-123` — asserts `reverb_tts` not in `sys.modules` at import time. Baseline: 127 passed, 7 pre-commit hooks.
- **S13 — CI.** `.github/workflows/test.yml:22` — `uv sync --all-groups` (no extras). Both Dockerfiles sync `--extra <accel> --extra tts` and sanity-step import.
- **S14 — Config prefix and lock.** `src/speakeasy/config.py:12` — `SPEAKEASY_` prefix. `tests/test_lock_default.py` — no-extras resolves to CPU torch.
- **S15 — No request timeout.** `server.py:577` — uvicorn has no timeout set on the run; `app.js:85` — `fetch` has no abort controller.
- **S16 — UI structure.** `src/speakeasy/static/index.html` — three panels (listen, speak, clone); `app.js` — `speak()` at line 351, `fillVoices()` at 327, `renderClones()` at 394, `saveClone()` at 419, `refreshStatus()` at 477; `textContent` for all data; `sessionStorage` for key.

### VERIFIED — voxcpm 2.0.3 source (`ref/voxcpm2/voxcpm-src/`)

- **V1 — `VoxCPM.from_pretrained` signature.** `core.py:104-172` — `(hf_model_id, load_denoiser, zipenhancer_model_id, cache_dir, local_files_only, optimize, device)`. `optimize` defaults to `True`.
- **V2 — Constructor warm-up when `optimize=True`.** `core.py:96-101` — `if optimize:` generates a warm-up sentence; in the shipped image (no C compiler) this crashes the constructor.
- **V3 — `_generate` signature.** `core.py:180-196` — `(text, prompt_wav_path, prompt_text, reference_wav_path, cfg_value=2.0, inference_timesteps=10, min_len=2, max_len=4096, normalize=False, denoise=False, retry_badcase=True)`.
- **V4 — Text flattening.** `core.py:241-242` — `text.replace("\n", " ")` then `re.sub(r"\s+", " ", text)`. The server must preserve intentional whitespace, but the library flattens it anyway.
- **V5 — `prompt_wav_path` and `prompt_text` coupling.** `core.py:234-235` — both or neither; mismatch raises `ValueError`.
- **V6 — Reference and prompt path validation.** `core.py:227-232` — `os.path.exists` checked for each path; raises `FileNotFoundError` if absent.
- **V7 — Silent retry.** `core.py:192` — `retry_badcase` defaults to `True`. The check is in `model/voxcpm2.py:658-669`: when `pred_audio_feat.shape[0] >= target_text_length * 6.0`, it retries up to 3 times.
- **V8 — No `seed` argument.** `core.py:180-196` — no `seed` parameter. The sampler draws from torch's global generator (`torch.randn`); reproducible via `torch.manual_seed(n)` right before the call.
- **V9 — Audio loading in the model.** `model/voxcpm2.py:412` — `librosa.load(wav_path, sr=self._encode_sample_rate, mono=True)`, where `_encode_sample_rate` is 16000.
- **V10 — `build_prompt_cache`.** `model/voxcpm2.py:681-740` — encodes reference and prompt paths into feature tensors; returns a dict with mode (`reference`, `continuation`, `ref_continuation`) and the encoded features.
- **V11 — Modes.** `model/voxcpm2.py:484-626` — four modes: zero-shot (no paths, `text` alone), reference-only (voice cloning), continuation-only (`prompt_wav_path` + `prompt_text`), and combined (all three — reference isolation + continuation).
- **V12 — `optimize` guard.** `model/voxcpm2.py:274-296` — `optimize(disable=False)` wraps four callables with `torch.compile`; catches `Exception`, prints a warning, and returns `self`. The constructor's warm-up is *outside* this guard, so a compile failure there crashes the process.
- **V13 — `torchcodec`.** `model/voxcpm2.py` has no `torchcodec` import; it uses `librosa.load` (line 412). But `voxcpm`'s PyPI metadata lists `torchcodec` as a hard dependency; when installed it breaks `torchaudio.load` for the whole process. The lock overrides remove it.

### VERIFIED — spike 005 results (`ref/voxcpm2/results/`)

- **V14 — Speed.** `timing.json` — RTF 1.03 to 1.07 at 10 steps (CFG 2.0), flat across length and mode. 5 steps: 0.91; 16 steps: 1.24. About 17 characters of text per second of audio.
- **V15 — `cfg` sweep.** `cfg_sweep.json` — RTF ~1.09 at every accepted value (1.0 to 5.0; 1.22 at 0.5, which validation rejects). At 0.5 the text is ignored (12/12 words wrong). Loudness: -25.7 dBFS at 1.0, -17.4 at 2.0, -9.0 at 5.0. Takes vary ±4 dB on seed alone.
- **V16 — Load and memory.** `timing.json` — load 15.29 s from warm cache; 4.95 GB after load. Peak 5.4 GB (short), 7.9 GB (49 s take); reserved grows to 11.1 GB over a session.
- **V17 — MIOpen.** `miopen_default.json`, `miopen_off.json` — first take at each new length costs 10-11 s extra with MIOpen default; `cudnn.enabled = False` removes it.
- **V18 — Coexistence.** `coexist.json` — Whisper + Reverb + VoxCPM2 in one process, interleaved both ways, alive throughout: 7.0 GB allocated, 7.6 GB reserved.
- **V19 — Dependencies.** `pyproject.diff` — the `tts-voxcpm` extra plus lock overrides; `install-dryrun-summary.txt` — 100 packages raw, fsspec downgrade fixed by the overrides.
- **V20 — Oracle.** `oracle.json` — 9/266 words wrong (3.4%), every error a brand token, number format, or homophone.
- **V21 — Determinism.** Spike README, "Determinism" — `torch.manual_seed(n)` before the call gives byte-identical takes; different seeds differ.

### ASSUMED

- **A1 — Reference length for VoxCPM2.** The library's `_encode_wav` (`model/voxcpm2.py:412`) loads with `librosa.load` and pads to a patch-aligned length — it imposes no minimum or maximum. The spec keeps Reverb's 1 to 30 s bounds. **Settled by:** coder verifies the pad-to-patch logic does not crash on a 1 s clip; if it does, the minimum is raised.
- **A2 — CUDA runtime.** No NVIDIA GPU runs inference in this lane. The CUDA image's sanity step imports the engine. **Settled by:** the orchestrator at landing runs a real CUDA take; if it fails, the lane ships with a CUDA limitation note.
- **A3 — `torch.manual_seed` on GPU.** `torch.manual_seed` seeds the CPU RNG state **and every accelerator generator** (CUDA, ROCm, MPS — `torch.cuda`, `torch.hip`, etc. are all seeded). The spike verifies this on gfx1151: the sampler draws `torch.randn` on the device, and two takes with the same seed produce byte-identical audio (spike README "Determinism"). **Settled by:** the orchestrator verifies two takes with the same seed produce byte-identical audio on the target GPU at landing.
- **A4 — 48 kHz `pcm` format.** 48 kHz s16le mono follows the same convention as Reverb's 24 kHz; client libraries that hardcode a sample rate may misinterpret it. **Settled by:** documented in the README.

## 5. HTTP contract

All `/v1/*` endpoints require the bearer token (existing `require_auth`). `/health`, `/health/ready`, `GET /`, `/static/*` stay open.

### 5.1 `POST /v1/audio/speech` — both engines

**Auth:** bearer token.

**SpeechRequest fields:**

| Field              | Type                              | Default                  | Validation                                                                                           | Engine        |
|--------------------|-----------------------------------|--------------------------|------------------------------------------------------------------------------------------------------|---------------|
| `input`            | `str`                             | *(required)*             | Min length 1 (non-whitespace), max 4096 chars.                                                       | both          |
| `voice`            | `str`                             | `"en_male_narrator_1"`   | A preset id, a saved clone id, or `"default"` (VoxCPM2 only). At most 64 characters (Pydantic `max_length`). **Under `voxcpm2`, an omitted or null `voice` resolves to `"default"`** — never the Reverb preset. A saved clone id is at most 64 characters by the id rule and the longest preset id is 17, so no valid voice is excluded; the bound keeps an unbounded string from entering a body that names the unknown voice. | both          |
| `response_format`  | `"wav"` \| `"mp3"` \| `"flac"` \| `"pcm"` | `"mp3"` | —                                                                                                    | both          |
| `speed`            | `float`                           | `1.0`                    | 0.25 to 4.0 for Reverb; must be 1.0 for `voxcpm2` (else 400).                                        | Reverb only   |
| `model`            | `str`                             | *(required)*             | A listed TTS engine id from `GET /v1/models`: `"reverb-tts"` for Reverb, `"voxcpm2"` for VoxCPM2. No default — missing or empty is 422. | both          |
| `language`         | `str` \| `null`                   | `null`                   | For Reverb: one of the 7 codes. For VoxCPM2: accepted and ignored (the model needs no tag).           | both          |
| `seed`             | `int` \| `null`                   | `null`                   | Reproducible take when set. For VoxCPM2: applied via `torch.manual_seed(n)` under the TTS lock.       | both          |
| `cfg`              | `float` \| `null`                 | `null` (→ 2.0)           | CFG guidance strength, 1.0 to 5.0 inclusive. For VoxCPM2 only; a non-null value on Reverb is 400.     | VoxCPM2 only  |
| `design`           | `str` \| `null`                   | `null`                   | Voice design description, 1 to 300 chars. For VoxCPM2 only; non-null on Reverb is 400.                | VoxCPM2 only  |
| `style`            | `str` \| `null`                   | `null`                   | Style hint, 1 to 200 chars. For VoxCPM2 only; non-null on Reverb is 400.                              | VoxCPM2 only  |

**Validation rules (new fields, VoxCPM2-specific):**

1. `cfg` outside 1.0 to 5.0 → 422 (Pydantic `Field(ge=1.0, le=5.0)`).
2. `speed` != 1.0 for `voxcpm2` → 400 `{"error": {"message": "speed must be 1.0 for the voxcpm2 engine"}}`.
3. `cfg`, `design`, or `style` sent to Reverb (any non-default value) → 400 `{"error": {"message": "cfg, design and style are voxcpm2-only fields; set model=voxcpm2"}}`.
4. `design` and `style` both non-null → 400 `{"error": {"message": "at most one of design and style may be set"}}`.
5. `design` non-null with a voice other than `"default"` → 400 `{"error": {"message": "design requires voice: 'default'"}}`.
6. `design` or `style` contains `(`, `)`, or a newline → 400 `{"error": {"message": "design and style must not contain parentheses or newlines"}}`. The length caps (300 and 200) are Pydantic `max_length`.
7. Voice is `"default"` → plain (or designed) speech, no reference. Any other voice must be a saved VoxCPM2 clone. A Reverb preset id is 400: `{"error": {"message": "unknown voice '<id>' for voxcpm2; use 'default' or a saved clone: <clone-ids>"}}`.
8. `input` whose first character (after stripping) is `(` → 400 `{"error": {"message": "input starting with '(' is reserved for design and style; remove the leading parenthesis or use the design field"}}`.
9. `model` missing or empty → 422 (Pydantic `Field(..., min_length=1)` with `model: str = ...` — no default). This is the same 422 class as `input` over 4096 characters.
10. `model` not a listed TTS engine id → **404** `{"error": {"message": "unknown model '<value>'. Known: <listed ids, comma-separated>"}`. `<value>` echoes at most the first 64 characters of what was sent, with an ellipsis when it was cut (e.g. `"voxcp..."`). The check runs before any engine gate, reads no engine state and takes no lock: an unlisted name answers 404 whatever the engines' states. Listed ids are `reverb-tts` and `voxcpm2`.

The server builds the library input:
- **Plain / voice design:** `text` = the request's `input`. For `design` non-null: prepend `"(" + design + ")"`.
- **Style on default:** same as voice design — `"(" + style + ")" + input`, no reference path. A styled default voice is a designed voice.
- **Clone (no transcript):** `reference_wav_path` set to the stored WAV; `text` = the request's `input` (optionally prefixed with style).
- **Clone (with transcript):** both `prompt_wav_path` and `reference_wav_path` set to the stored WAV; `prompt_text` = the stored transcript; `text` = the request's `input` (optionally prefixed with style).

**Response 200:**

- Body: the encoded audio bytes at 48 kHz for VoxCPM2, 24 kHz for Reverb.
- Headers:
  - `Content-Type`: `audio/mpeg` (mp3), `audio/wav` (wav), `audio/flac` (flac), `audio/pcm` (pcm).
  - `X-Speakeasy-Engine`: `"voxcpm2"` or `"reverb-tts"`.
  - `X-Speakeasy-Elapsed`: float seconds of inference time.
  - `X-Speakeasy-Audio-Duration`: float seconds of output audio.
  - `X-Speakeasy-RTF`: real-time factor.

**Response 400:** `{"error": {"message": "…"}}` — validation errors as above; unknown voice; clone file vanished.

**VoxCPM2 503 bodies (fixed, byte-exact — tested as such):**

| State                         | Body                                                                                                      |
|-------------------------------|-----------------------------------------------------------------------------------------------------------|
| `not_loaded` / `loading`      | `{"error": {"message": "voxcpm2 is still loading (first start downloads about 4.7 GB of weights)"}}`     |
| `error`                       | `{"error": {"message": "voxcpm2 encountered an error and is unavailable"}}`                              |
| bad take                      | `{"error": {"message": "voxcpm2 could not generate a good take for this input; check the input length and try again"}}` |

**Note:** `disabled` is no longer a 503 on speech and clone-create — when `SPEAKEASY_TTS_VOXCPM_ENABLED` is false the engine is absent from `GET /v1/models`, so `model: "voxcpm2"` answers 404 (unknown model). Delete of a VoxCPM2 clone while disabled stays 404 "no saved voice" (§5.4); the voice listing keeps its 503 when both engines are unavailable (§5.2).

**Response 401, 422:** as in the Reverb spec.

### 5.2 `GET /v1/audio/voices`

**Auth:** bearer token. **State gate (O-R8a):** The listing answers 200 whenever voxcpm2 is enabled — in any state (clones appear from disk; presets appear only while Reverb is ready) — **or** Reverb is ready (presets + clones). 503 only when voxcpm2 is **disabled** *and* Reverb is not ready. A deployment with `tts_voxcpm_enabled=False` and Reverb not ready answers exactly as today.

**Response 200:** Every voice object gains an `"engine"` field.

```json
{
  "object": "list",
  "engines": ["reverb-tts", "voxcpm2"],
  "languages": ["de", "en", "es", "fr", "it", "pl", "pt"],
  "data": [
    {"id": "en_male_narrator_1", "object": "voice", "type": "preset", "language": "en", "engine": "reverb-tts"},
    {"id": "clone-a1b2c3", "object": "voice", "type": "clone", "language": "en", "transcript": "Hello world", "created": 1700000000, "engine": "voxcpm2"}
  ]
}
```

- `engines`: a sorted array of engine ids whose voices appear in this response (up to two: `"reverb-tts"` and `"voxcpm2"`). The page may ignore this key; it is additive surface.
- `languages` **(O-R7a):** present with today's value whenever Reverb is ready; **omitted** otherwise (Reverb disabled or not ready). The page keeps the language-select content across re-renders and refills it when a later listing carries the key. The pinned 503 strings stay byte-identical whenever the 503 is Reverb's alone — the response bodies in a deployment with voxcpm2 disabled are bit-for-bit the same as today.
- Races: a clone deleted mid-listing is absent; no 500.
- Reverb presets + clones first (sorted by id), then VoxCPM2 clones (sorted by id).

**Response 401, 503:** as in the Reverb spec (the Reverb 503 strings are unchanged — §4.1:114 of the Reverb spec).

### 5.3 `POST /v1/audio/voices`

**Auth:** bearer token.

**Request body** (`multipart/form-data`):

| Field       | Type     | Default                | Validation                                                                                          |
|-------------|----------|------------------------|-----------------------------------------------------------------------------------------------------|
| `file`      | file     | *(required)*           | Reference audio, max 20 MB.                                                                         |
| `text`      | `str`    | optional for `voxcpm2` | The reference's transcript, non-empty after stripping. **Required when `model` is `"reverb-tts"`; optional for `voxcpm2`.** |
| `language`  | `str`    | `"en"`                 | For Reverb: one of the 7 codes. For VoxCPM2: accepted but not validated (stored in the metadata).    |
| `voice_id`  | `str`    | `clone-<hex8>`         | `^[a-z0-9][a-z0-9-]{1,63}$`. Unique across both engines — a collision with a Reverb clone is 409.   |
| `model`     | `str`    | *(required)*           | A listed TTS engine id: `"reverb-tts"` for a Reverb clone, `"voxcpm2"` for a VoxCPM2 clone. No default — missing or empty is 422; unlisted is 404 (see §5.1). |

**For VoxCPM2:** The route **stores the reference and does not run the model**. The reference is decoded with `faster_whisper.audio.decode_audio(data, sampling_rate=16000)` (PyAV — never through a helper that reads the Reverb engine's sample rate), written as `<voices_dir>/voxcpm2/<voice_id>.wav` at 16 kHz PCM_16 with `soundfile.write(..., 16000, subtype='PCM_16')` via `os.replace`, then the metadata JSON written second. No `tts_lock` is held — the create's own critical section (both-stores existence check through the save) holds `voxcpm2_create_lock` (§7.5, §8). The single-flight load trigger (§7.2) is called through a shared helper that holds a dedicated state lock for the check-and-set — the create does not take `tts_lock` for the load trigger.

**Response 201:**

```json
{
  "id": "clone-a1b2c3",
  "object": "voice",
  "type": "clone",
  "engine": "voxcpm2",
  "language": "en",
  "transcript": "exact reference text or null",
  "created": 1700000000
}
```

**Response 400:** text empty when `model` is `"reverb-tts"` (text is optional for `voxcpm2`); invalid `voice_id`; reference under 1 s or over 30 s; could not decode the reference.

**Response 409:** `voice_id` collides with a preset or an existing clone (either engine).

**Response 413:** reference over 20 MB.

**Response 503:** clone-create is gated by VoxCPM2 state (O-R1a): in `not_loaded` / `loading` / `error` the response is 503 with that state's reason; in `not_loaded` the single-flight load starts; 201 only in `ready`. See the state table (§6) and race rows (§7.5).

**Response 401, 422:** as in the Reverb spec.

### 5.4 `DELETE /v1/audio/voices/{voice_id}`

**Auth:** bearer token. **State gate — per store.** The delete is a two-store operation: the route resolves the id against the VoxCPM2 store first. A VoxCPM2 clone is deleted from disk immediately (a filesystem operation that does not need a loaded engine) and answers per the state table (§6). Any id not found in the VoxCPM2 store falls through to the Reverb path exactly as on `main` today — `tts_unavailable()` gates the Reverb-store half, so a delete of a Reverb clone while Reverb is not ready answers Reverb's fixed 503 in every VoxCPM2 state, and the VoxCPM2 state is irrelevant to it. This means: a voxcpm2-enabled deployment, for a Reverb clone while Reverb is loading, disabled, or in error, answers the same 503 as today — no Reverb status change.

**Response 200:**

```json
{"id": "clone-a1b2c3", "object": "voice", "deleted": true, "engine": "voxcpm2"}
```

`"engine"` is the engine that owned the clone (`"voxcpm2"` or `"reverb-tts"`). Two concurrent deletes of the same clone both report success.

**Response 404:** `{"error": {"message": "no saved voice '…'"}}` — unknown id in either store, invalid id, or a preset name.

**Response 401, 503:** as above.

### 5.5 Additions to existing endpoints

**`GET /health/ready`** — gains `"voxcpm2"` in the response body, parallel to `"tts"`:

```json
{
  "status": "ready",
  "device": "cuda",
  "models": ["large-v3-turbo", "reverb-tts", "voxcpm2"],
  "tts": "ready",
  "voxcpm2": "ready"
}
```

`"voxcpm2"` is `"disabled"` | `"not_loaded"` | `"loading"` | `"ready"` | `"error"`.

**`GET /v1/models`** — lists TTS engine ids by **configuration**, not load state: `reverb-tts` whenever `SPEAKEASY_TTS_ENABLED` is true, `voxcpm2` whenever `SPEAKEASY_TTS_VOXCPM_ENABLED` is true. Each listed model is `{"id": "<engine-id>", "object": "model", "created": 0, "owned_by": "speakeasy"}`. Previously `reverb-tts` appeared only once loaded (S3); under the "unlisted is 404" ruling every speech request during Reverb's startup would 404 instead of answering the loading 503, so the listing needs the names present at config time. A listed engine that is not ready keeps its 503s exactly as pinned. Transcription models keep today's behaviour and are outside this revision.

## 6. State table

### VoxCPM2 engine state

| State         | Speech (`voxcpm2`)                    | Clone create (`voxcpm2`)             | Voice list (any engine)             | Delete (`voxcpm2` clone)            | `/health/ready`                     | `/v1/models`                        |
|---------------|---------------------------------------|--------------------------------------|-------------------------------------|-------------------------------------|-------------------------------------|-------------------------------------|
| `disabled`    | 404 — engine not listed (see §5.1)     | 404 — engine not listed (see §5.1)   | VoxCPM2 clones omitted; Reverb ok   | 404 — "no saved voice"              | `"voxcpm2": "disabled"`             | `voxcpm2` absent                    |
| `not_loaded`  | 503 — "is still loading" + **starts load** | 503 — "is still loading" + **starts load** | VoxCPM2 clones from disk; load NOT triggered | 200 — delete from disk (no load needed) | `"voxcpm2": "not_loaded"`       | `voxcpm2` present                  |
| `loading`     | 503 — "is still loading"              | 503 — "is still loading"             | VoxCPM2 clones from disk            | 200 — delete from disk               | `"voxcpm2": "loading"`              | `voxcpm2` present                  |
| `ready`       | 200 — generate audio; **update last-used** | 201 — store reference (no model call) | VoxCPM2 clones from disk | 200 — delete from disk | `"voxcpm2": "ready"`          | `voxcpm2` present                  |
| `error`       | 503 — "encountered an error"          | 503 — "encountered an error"         | VoxCPM2 clones from disk            | 200 — delete from disk               | `"voxcpm2": "error"`                | `voxcpm2` present                  |

The delete column covers VoxCPM2-clone deletes only. The Reverb-store fall-through (any id not found in the VoxCPM2 store) gates on `tts_unavailable()` exactly as on `main` today — delete of a Reverb clone follows the Reverb contract unchanged, answering Reverb's fixed 503 in every VoxCPM2 state.

### Reverb side (shown for comparison)

| State     | Speech (Reverb)               | Clone create/delete (Reverb)   | Voice list                         |
|-----------|-------------------------------|--------------------------------|------------------------------------|
| `disabled`| 404 — engine not listed (see §5.1) | create 404 — engine not listed; delete 503 — fixed reason | Reverb presets and clones absent   |
| `loading` | 503 — fixed reason            | 503 — fixed reason             | Reverb presets and clones absent   |
| `ready`   | 200 — generate audio          | 201 / 200                      | Reverb presets + clones present    |
| `error`   | 503 — fixed reason            | 503 — fixed reason             | Reverb presets and clones absent   |

## 7. Lifecycle

### 7.1 Shared objects at startup

At server startup (in `lifespan`), three new globals are created:

```python
tts_lock = threading.Lock()             # one lock shared by both engines for inference
voxcpm2_state_lock = threading.Lock()   # small dedicated lock for state check-and-set
voxcpm2_create_lock = threading.Lock()  # dedicated lock for clone-create serialization
voxcpm2_state = "disabled" if not settings.tts_voxcpm_enabled else "not_loaded"
voxcpm2: VoxCPM2Engine | None = None
voxcpm2_store: VoxCPM2VoiceStore | None = None
voxcpm2_error: str | None = None
voxcpm2_last_used: float = 0.0
```

**Store lifecycle.** The VoxCPM2 clone store is created eagerly at startup **iff `tts_voxcpm_enabled`** — it is a filesystem tree under `<voices_dir>/voxcpm2/` and does not need a loaded engine. The store is never created by the loader (the loader only loads the model) and is never dropped by an idle unload — `voxcpm2_store` survives `ready → not_loaded` transitions. When `tts_voxcpm_enabled` is `False`, `voxcpm2_state` is `"disabled"` and the store is never created; all endpoints answer as shown in the `disabled` state-table row. Reverb's `start_tts()` remains exactly as-is and does not touch VoxCPM2.

### 7.2 Single-flight load

A shared helper, `_ensure_voxcpm2_loading()`, is called by every VoxCPM2 speech request and every VoxCPM2 clone-create request. Starting a load **never takes `tts_lock`** and **never skips** because the lock is busy — both request kinds call the same helper.

The helper:

1. Acquires `voxcpm2_state_lock` (a small dedicated lock — never `tts_lock`).
2. Checks `voxcpm2_state`. If `not_loaded`: sets it to `loading`, spawns `_load_voxcpm2()` on a daemon thread, releases the state lock, and returns the loading 503.
3. If `loading`: releases the state lock and returns 503 — the load is already in flight.
4. If `ready` or `disabled` or `error`: releases the state lock and falls through (the caller proceeds with its own gate logic).

The loader thread, `_load_voxcpm2()`:

- Imports `voxcpm`, `torch` inside the function body (never at module import time).
- Acquires `tts_lock` for the load itself (model construction is not thread-safe).
- Constructs `VoxCPM2Engine` and publishes it (`voxcpm2 = engine`).
- Sets state to `ready`, sets `voxcpm2_last_used = time.monotonic()` (the idle clock starts at the moment the engine became ready — under `tts_lock`, so no tick races the ready transition), then releases `tts_lock`.
- While the loader waits for `tts_lock` behind a Reverb synthesis, the state is `loading` — arriving VoxCPM2 requests see `loading` and answer the loading 503.
- On failure: sets state to `error` under `tts_lock`, stores the exception text in `voxcpm2_error` (for logs only — never returned to clients). Nothing auto-retries.

The loader never creates the store — the store already exists eagerly from startup (§7.1). The loader never touches `voxcpm2_state_lock` after the initial check-and-set; the state transition to `ready` or `error` is under `tts_lock`.

### 7.3 Idle unload

A daemon thread, started after the first successful load, runs every 15 s:

1. Acquires `tts_lock` **without blocking** (`acquire(blocking=False)`). If it cannot take the lock, sleeps and tries again — a running request owns it.
2. Checks `voxcpm2_state == 'ready'` **first**. In any other state (`loading`, `error`, `disabled`, `not_loaded`) the tick has nothing to unload and skips — this guard prevents a tick in the loader's pre-`tts_lock` phase from clobbering `loading` and prevents a tick just after a load completes from unloading a freshly loaded engine whose last-used is still on its way (the page's auto-send may be up to 5 s behind).
3. Checks `time.monotonic() - voxcpm2_last_used > tts_voxcpm_idle_seconds`. When the loader sets `voxcpm2_last_used` at load completion (§7.2), the idle clock measures from the ready transition — no stale-last-used window after the load.
4. If the idle time is exceeded: drops the model (`voxcpm2 = None`), calls `torch.cuda.empty_cache()` and `gc.collect()`, sets state to `not_loaded`. The store (`voxcpm2_store`) is **not** dropped — it survives the unload.
5. Releases the lock.

When `tts_voxcpm_idle_seconds` is `0`, the timer thread is never started — the engine stays loaded.

### 7.4 TTS lock protocol — per engine

Both engines share `tts_lock` for inference serialisation, but their protocols are separate.

#### VoxCPM2 speech protocol

1. Call `_ensure_voxcpm2_loading()` (§7.2). If it returned 503 (state was `not_loaded` or `loading`), return that 503.
2. **Gate check** (no lock). If the state is `error` or `disabled`, return 503 immediately. If `ready`, proceed. This check never blocks and a pass is not a promise.
3. Acquire `tts_lock`.
4. **State/engine re-check** under the lock. If the state is no longer `ready` (an unload completed between the gate check and the lock acquisition): set `not_loaded` and start a new load; return 503 `not_loaded`, never 500. If the state is `error`: return 503 `error`. This re-check guarantees the engine is present before the model call.
5. Seed the global generator (`torch.manual_seed(seed)`) if a `seed` is given, under the lock, immediately before the library call.
6. Call `VoxCPM2Engine.generate()` which returns `(audio, bad_case)`.
7. Map `bad_case` → 503 (see §5.1). Only a successful take flows through.
8. **Update `voxcpm2_last_used`** (only VoxCPM2 speech bumps this — clone-create, listing, delete, and Reverb speech do not).
9. Release `tts_lock`.
10. Encode the result outside the lock.

#### Reverb speech protocol

1. **Gate check** (no lock) — Reverb's own gate as today (`tts_unavailable()`, reading `tts_state`). VoxCPM2's state is irrelevant.
2. Acquire `tts_lock`.
3. Call `tts.synthesize()` (which re-acquires Reverb's own `RLock` inside the engine).
4. Release `tts_lock`.
5. Encode the result outside the lock.

Reverb speech does **not** bump `voxcpm2_last_used` — only VoxCPM2 speech does (O-R4a).

Clone-create requests do **not** acquire `tts_lock`. They use `voxcpm2_create_lock` instead (see §7.5). They call `_ensure_voxcpm2_loading()` for the load trigger through `voxcpm2_state_lock`, never through `tts_lock`.

This ensures at most one inference runs at any moment across both engines, each engine's gate only reads its own state, the `torch.manual_seed` call is not interleaved, and only VoxCPM2 speech keeps the engine warm against idle unload.

### 7.5 Races

| Race                                                          | Outcome                                                                                          |
|---------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Two first requests arrive while `not_loaded` (speech or clone-create) | One starts the load (sets `loading`), the other sees `loading` and returns 503.                   |
| Request (speech) passes the gate, unload takes the lock and completes before the request acquires | Gate sees `ready` → unload timer takes `tts_lock`, drops model, releases → request acquires, re-checks state → sees `not_loaded` → **starts a new load** and returns 503. Never 500. |
| Request arrives during unload                                 | Unload holds `tts_lock`; request blocks, then finds `not_loaded` → 503 + starts load. Or request holds the lock first; unload skips. |
| Load fails                                                    | `_load_voxcpm2()` catches the exception, sets state to `error` under `tts_lock`. Requests polling see `error` → 503 with the fixed error message. Nothing auto-retries — recovery is a server restart (noted in the README). |
| Clone deleted between store listing and speech voice resolution | `exists()` passes, `load`/path read finds file gone → 400 (unknown voice), never 500.             |
| Clone deleted mid-load or mid-unload (unload does not touch files) | Delete is a filesystem op; it succeeds at any moment. |
| Two concurrent deletes                                        | Both report `deleted: true`; see §10.                                                            |
| Clone-create in `not_loaded` / `loading` / `error`           | Answers 503 with that state's reason; in `not_loaded` starts the single-flight load (same path as speech). |
| Clone-create in `ready`                                      | Answers 201 — store reference, no model call. Does not take `tts_lock`; uses `voxcpm2_create_lock`; does not bump last-used. |
| Two concurrent clone-creates for the same id                  | Serialized under `voxcpm2_create_lock` from the both-stores existence check through the save. Exactly one returns 201; the other 409. Cross-engine collision (Reverb `.npz` vs VoxCPM2 `.json`) is covered: both stores are checked under the create lock. |
| Delete of a Reverb clone while Reverb is not ready (loading / disabled / error) | The id is not found in the VoxCPM2 store; the fall-through to the Reverb path gates on `tts_unavailable()` — answers Reverb's fixed 503 in every VoxCPM2 state. No Reverb status change. |
| Timer tick while a load is in flight (loader's pre-`tts_lock` phase, no lock held) | The timer skips — `voxcpm2_state != 'ready'` (§7.3 step 2 guards the tick). Without the guard a tick would take `tts_lock` (free during imports), find a stale last-used, and clobber `loading → not_loaded`, breaking the single-flight invariant (SEC-V6) and misreporting `/health/ready`. |
| Timer tick just after a load completes (state `ready`, last-used stale before first speech) | The timer skips — `voxcpm2_last_used` was set to `time.monotonic()` at load completion under `tts_lock` (§7.2), so the idle clock started at the ready transition. Without the reset a tick could unload the freshly loaded engine before the page's ≤5 s auto-send bumps `last_used`, causing reload thrash that degrades D3's one-reload contract into a probabilistic multiple. |

## 8. On-disk contract

### VoxCPM2 clone store

- **Directory:** `<voices_dir>/voxcpm2/`, created eagerly at startup with `0o700`.
- **Reference file:** `<voice_id>.wav` — 16 kHz mono PCM_16 WAV, decoded from the upload with `faster_whisper.audio.decode_audio(data, sampling_rate=16000)` (PyAV) and written with `soundfile.write(..., 16000, subtype='PCM_16')`. The decode never reads the Reverb engine's sample rate (O-R6a).
- **Metadata file:** `<voice_id>.json` — `{"engine": "voxcpm2", "language": <string or null>, "transcript": <string or null>, "seconds": <float>, "created": <int>}`.
  - `language` is the value from the multipart form (accepted but not validated — stored verbatim).
  - `transcript` is the form's `text` value or `null` when omitted.
  - `seconds` is the decoded audio duration in seconds.
- **Write order:** WAV first (`os.replace` from a temp file), JSON second. The JSON is the authoritative existence marker: `listing()` reads `.json` files only; if only the WAV exists (a crash between the two writes), the clone is not listed.
- **Delete order:** JSON first (`os.remove`, `missing_ok=True`), then WAV (`os.remove`, `missing_ok=True`). An orphan `.wav` after a crash between the two deletes is unlisted and harmless (the JSON is gone, so `listing()` never finds it).
- **Concurrent creates serialized:** The both-stores `exists` check and the file writes hold `voxcpm2_create_lock` — exactly one create for a given id wins (201); the other is 409, cross-engine included. Delete and listing take no new lock.

### Id uniqueness

Ids are validated with `fullmatch` of `^[a-z0-9][a-z0-9-]{1,63}$` before any path is built. A clone-create checks both stores before saving: a collision with a Reverb `.npz` or a VoxCPM2 `.json` is 409.

### Reverb store

`<voices_dir>/<voice_id>.npz` — unchanged. `ReverbEngine`'s loader creates the root `voices_dir`; VoxCPM2 creates only the `voxcpm2/` subdirectory.

## 9. Security requirements

Each named and testable.

1. **SEC-V1 — clone id validated before any path is built.** `fullmatch` of the regex for both stores. A VoxCPM2 clone id passes the same `check_voice_id()` and additionally `Path(.../voxcpm2/<id>.wav)` is only built after validation. Test: `test_invalid_voxcpm2_voice_ids_rejected`.

2. **SEC-V2 — `design` and `style` cannot inject the library's syntax.** `(`, `)`, and newlines are rejected; length caps hold. Additionally, `input` whose first character (after stripping) is `(` is rejected — the library treats a leading `(` as a design prefix. Test: `test_voxcpm2_design_style_syntax_rejected`.

3. **SEC-V3 — reference path from validated id only.** The WAV path handed to the library comes from `voxcpm2_store._path(voice_id)` which calls `check_voice_id()` — the path is always under the store root. Test: `test_invalid_ids_rejected_before_touching_disk` (validates store-level rejection of traversal ids before anything is written); `test_reference_path_rejects_traversal` asserts `_path()` raises `ValueError` on a traversal id, and a speech request with a traversal id is a 400.

4. **SEC-V4 — upload limits unchanged.** >20 MB → 413; <1 s or >30 s → 400; decode failure → 400 with no exception text. Test: `test_voxcpm2_clone_upload_limits`.

5. **SEC-V5 — one inference at a time across both engines.** `tts_lock` covers every call to `generate` and `synthesize`. Seeding is under the same lock. Test: `test_shared_tts_lock_serialises_both_engines`.

6. **SEC-V6 — single-flight load.** Two concurrent first requests start exactly one loader. The loader never runs on the startup path or blocks `/health/ready`. Test: `test_voxcpm2_single_flight_load`.

7. **SEC-V7 — never a crash from missing engine or unload race.** The state/engine re-check under `tts_lock` guarantees no code path reaches a model call with `voxcpm2 is None`. The gate-pass-then-unload race starts a new load and answers 503, never 500. A failed load sets `error`; polling requests see 503; nothing auto-retries — the README states recovery is a restart. Test: `test_voxcpm2_state_recheck_under_lock` and `test_voxcpm2_load_failure_sets_error`.

8. **SEC-V8 — bad-take check before audio returned.** `VoxCPM2Engine.generate` passes `retry_badcase=False`, computes the audio-patch count (`floor(audio_seconds / 0.16)`, citing the 0.16 s model patch duration from spike README "Results") and the text-token count via the model's tokenizer over the same flattened text the library generates from (V-fact: `voxcpm-src/model/voxcpm2.py:633`). Returns `(audio, bad_case)` where `bad_case = (patches >= 6 * tokens)`. The route maps `bad_case` to 503. Test: `test_voxcpm2_bad_take_503`.

9. **SEC-V9 — no exception text in 400/503/500 bodies.** Fixed strings per state; exceptions logged but never returned. Test: `test_no_exception_text_in_voxcpm2_error_bodies`.

10. **SEC-V10 — CSP unchanged, no new inline script/style, no `innerHTML`.** The engine selector and new fields are added with `textContent` and `createElement`; CSP does not change. Test: `test_index_page` (existing, extended to check for voxcpm2 additions).

11. **SEC-V11 — supply chain.** Lock overrides keep `torchcodec`, `gradio`, `gradio-client`, `hf-gradio`, `funasr`, `spaces`, `datasets`, `argbind`, `modelscope`, and `modelscope-hub` out. No existing pinned version changes (`fsspec==2026.9.0` and each torch pin unchanged). `uv sync --all-groups` (no extras) still resolves to CPU torch. Test: `test_lock_default.py` (existing, must pass unchanged). **New test (in `tests/test_lock_default.py`):** `test_voxcpm2_extra_excludes_supply_chain_leaks` — runs `uv export --frozen` for each accelerator extra plus `--extra tts-voxcpm` and asserts: (a) none of the ten excluded packages appear in the resolution, and (b) `fsspec==2026.9.0` and the torch pins are unchanged from the lockfile. The `uv lock --dry-run` diff is the landing check; this test is the re-runnable proof.

12. **SEC-V12 — `src/` never imports `voxcpm` at module import time.** All `import voxcpm` and `import torch` calls are inside function bodies (`_load_voxcpm2`, `VoxCPM2Engine.__init__`). CI has no `tts-voxcpm` extra. Test: `test_engines_do_not_import_voxcpm_at_import_time` mirrors the existing Reverb check.

## 10. Check-then-use census

For every pair where a clone is checked and then used — the clone can vanish between the two calls.

### Model listing check (speech and clone-create)

The first step of both the speech path (§5.1) and the clone-create path (§5.3) is a model listing check: the request's `model` is matched against the listed TTS engine ids from `GET /v1/models`. Unlisted answers 404 with the `unknown model` body (§5.1 rule 10). The check runs before any engine gate, reads no engine state and takes no lock — it is a simple set-membership test over the configured engines. The listed ids are `reverb-tts` (when `SPEAKEASY_TTS_ENABLED` is true) and `voxcpm2` (when `SPEAKEASY_TTS_VOXCPM_ENABLED` is true).

### Speech path (`VoxCPM2Engine`)

| Pair                                          | File:line in spec (server.py)           | Vanished between calls → outcome                                           |
|-----------------------------------------------|-----------------------------------------|----------------------------------------------------------------------------|
| `exists(voice_id)` → `load(voice_id)`         | speech handler, voice resolution        | `FileNotFoundError` caught → 400 `"unknown voice"`, same as Reverb's path  |
| `listing()` → per-clone `info()` in response  | `_voice_objects()` (shared listing)     | Clone absent from the response — no 500                                    |
| `exists(voice_id)` → `_path(voice_id)` → library | speech handler, build library args   | Library raises `FileNotFoundError` → caught → 400 `"unknown voice"`        |

### Clone create path (VoxCPM2 — filesystem only)

| Pair                                              | File:line   | Vanished between calls → outcome                                           |
|---------------------------------------------------|-------------|----------------------------------------------------------------------------|
| `exists(voice_id)` before save → `save()`         | create handler | Serialized under `voxcpm2_create_lock` — the `exists` re-check and the write are atomic with respect to concurrent creates. A delete could race: `exists` passes, file deleted, `save` writes normally (no model runs, no dependency on the pre-existing file). |

### Delete path

| Pair                                              | Vanished between calls → outcome                                           |
|---------------------------------------------------|----------------------------------------------------------------------------|
| `exists(voice_id)` → `unlink()` (VoxCPM2 store)   | `unlink(missing_ok=True)` — both deletes report success. Same as Reverb.   |
| `exists(voice_id)` (Reverb store) → `delete()`    | Id not found in VoxCPM2 store → fall through to Reverb path. The Reverb path gates on `tts_unavailable()` exactly as on `main` today — while Reverb is not ready the delete answers Reverb's fixed 503 before reaching the filesystem check. |

### Delete during load / unload

The delete is a filesystem operation on the voice store, which is separate from the engine's load/unload lifecycle. A delete during an active load or unload succeeds — the loader does not hold the store's lock (there is no store lock). The store directory exists eagerly and survives engine unload.

## 11. UI behaviour

The existing three-panel page (`index.html` + `app.js`) gains a VoxCPM2 engine selector and three new controls.

### 11.1 Engine selector

A `<select id="speak-engine">` above or beside the voice select, with two options: `reverb-tts` (default) and `voxcpm2`. **The page never offers an engine the server does not list:** the `voxcpm2` option is present only while `/health/ready` reports `voxcpm2` as something other than `disabled` — when the engine is disabled it is absent from the listing and the page does not offer it. The page already sends `model` on every speech and clone request; that stays. Switching engines:

- Rebuilds the voice list filtered to that engine.
- Reverb selection: shows the language `<select>`. The page keeps the select's content across re-renders and refills it from `data.languages` when the key is present (as today — O-R7a: `languages` is present whenever Reverb is ready, omitted otherwise). VoxCPM2 selection: hides or disables the language `<select>` (VoxCPM2 needs no language tag), and the page keeps the select's existing content for when the user switches back.
- Shows/hides the `cfg`, `design`, and `style` controls (visible only for `voxcpm2`).
- The Speak button sends `model` matching the selected engine.

The clone panel gains a similar engine selector (`model` field in the multipart). For `voxcpm2`: the language field is hidden (the stored value passes through to metadata but is not user-visible). The `text` field is optional (its label changes to "Transcript (optional for VoxCPM2)").

### 11.2 New controls (VoxCPM2 only)

| Control           | Element                                    | Behaviour                                                                                      |
|-------------------|--------------------------------------------|------------------------------------------------------------------------------------------------|
| CFG (guidance)    | `<input type="range" min="1.0" max="5.0" step="0.1">` + display | Default 2.0. Sent as `cfg` in the speech JSON.                    |
| Design            | `<textarea placeholder="describe a voice, e.g. 'A young woman, gentle and sweet voice'">` | Up to 300 chars. Sent as `design`. Disabled when a clone is selected. |
| Style             | `<textarea placeholder="style hint, e.g. 'slightly faster, cheerful tone'">` | Up to 200 chars. Sent as `style`. Mutually exclusive with `design`. |

### 11.3 Voice list filtering

`GET /v1/audio/voices` returns `engine` on every object. The voice `<select>` is rebuilt with only the selected engine's voices. For `voxcpm2`: only clones appear (plus `"default"` as a synthetic first option).

### 11.4 First-use loading state

When a `voxcpm2` request (speech or clone-create) returns 503 with the loading reason:

- The Speak button (or clone Save button) re-enables (the 503 is a normal rejection, not a hang).
- The error area shows the **engine-specific** loading message from the page's own per-state mapping (§11.6) — engine-specific means the page checks `voxcpm2` state on a 503 and never shows the Reverb loading message for a VoxCPM2 request.
- The pending request (speech input, voice, cfg, design, style — and for clone-create: the file blob, text, language, voice_id, model) is saved.
- The page polls `/health/ready` **every 5 seconds** during the load. When `body.voxcpm2` becomes `"ready"`, the pending request is **auto-sent exactly once** — the page re-sends the speech or clone-create request the user made while the engine was loading. The voices are reloaded and the error is cleared. A one-shot "ready" notification is shown in the status display — the user sees `"tts: ready, voxcpm2: ready"`.
- The 5-second poll continues every 5 seconds for as long as the server reports `voxcpm2` as `"loading"` — the page polls at least every 5 seconds during the entire load (O-R3a). The poll stops only when `voxcpm2` becomes `"ready"` (auto-send pending request), `"error"` (discard pending request, revert to 30 s), or any other non-loading status. The load always ends in `ready` or `error` by the loader's own `finally` block, so the poll always terminates.

### 11.5 Idle unload state

After an idle unload, the next `voxcpm2` request gets the `not_loaded` 503, which starts a new load. The UI shows the same loading message. This is identical to the first-use path.

### 11.6 Error handling (new messages)

The page maps each VoxCPM2 state to its own display string — exactly as Reverb's page does today (`app.js:17-18` maps `tts_state` to a per-state message). The API's 503 bodies (§5.1) are the authoritative contract and are tested byte-exact; the page's strings are the UI contract and are tested in UI scenarios. The page never claims to show the API string.

| State                            | Page display message                                                                             |
|----------------------------------|--------------------------------------------------------------------------------------------------|
| 401                              | "Wrong or missing API key — enter it above." (unchanged)                                        |
| 503 (voxcpm2 `disabled`)         | "VoxCPM2 is disabled — enable it in the server config and restart."                              |
| 503 (voxcpm2 `not_loaded`/`loading`) | "VoxCPM2 is still loading (first start downloads ~4.7 GB of weights) — check back in a minute." |
| 503 (voxcpm2 `error`)            | "VoxCPM2 encountered an error and is unavailable — check the server logs."                       |
| 503 (voxcpm2 bad take)           | "VoxCPM2 could not generate a good take for this input — try shorter text or a different seed."  |
| 400 (VoxCPM2 field on Reverb)    | `error.body.error.message` — shown verbatim.                                                     |
| 400 (`speed` != 1.0 on VoxCPM2)  | `error.body.error.message` — shown verbatim.                                                     |
| 400 (input starts with `(`)      | `error.body.error.message` — shown verbatim.                                                     |
| Network error                    | "Could not reach the server — check the connection." (unchanged)                                 |

### 11.7 CSP and security

No change to the CSP. No new third-party scripts. All engine names, voice names, and error messages are written with `textContent`. The API key stays in `sessionStorage`.

## 12. Acceptance scenarios

Every scenario runs without a GPU. Tests fake `voxcpm` through `sys.modules`; no test imports `voxcpm` for real (`torch` is imported for real, as the existing suite already does).

### 12.1 Config

**Scenario: VoxCPM2 config defaults.** Given no env vars. When `Settings()` is created. Then `tts_voxcpm_enabled` is `True`, `tts_voxcpm_model` is `"openbmb/VoxCPM2"`, `tts_voxcpm_idle_seconds` is 600. **Test:** `test_voxcpm2_defaults` (`tests/test_config.py`).

### 12.2 Engine

**Scenario: engine construction with fixed args.** Given a fake `voxcpm` module. When `VoxCPM2Engine` is constructed. Then `from_pretrained` is called with `load_denoiser=False`, `optimize=False`; `sample_rate` is 48000; `name` is `"voxcpm2"`. **Test:** `test_voxcpm2_engine_constructor_args` (`tests/test_voxcpm2_engine.py`).

**Scenario: engine generates plain speech.** Given a fake engine. When `generate(text, voice="default", cfg=2.0, seed=None)` is called. Then the fake `generate` is called with `retry_badcase=False`, `inference_timesteps=10`, `max_len=4096`. It returns `(audio, bad_case)` — a tuple of float32 audio and `bool`. A normal take returns `(audio, False)`. **Test:** `test_voxcpm2_engine_generates` (`tests/test_voxcpm2_engine.py`).

**Scenario: engine applies seed.** Given a fake engine. When `generate(text, seed=42)` is called. Then `torch.manual_seed(42)` was called before the library call, under the TTS lock. **Test:** `test_voxcpm2_engine_seeds` (`tests/test_voxcpm2_engine.py`).

**Scenario: cudnn disabled only on HIP.** Given `torch.version.hip` set / not set. When `VoxCPM2Engine` is constructed. Then `torch.backends.cudnn.enabled` is `False` only on HIP. **Test:** `test_voxcpm2_cudnn_disabled_on_hip` (`tests/test_voxcpm2_engine.py`).

**Scenario: voxcpm not imported at module import time.** With `sys.modules` clean. When `import speakeasy.engines` is run. Then `"voxcpm"` is not in `sys.modules`. **Test:** `test_engines_do_not_import_voxcpm_at_import_time` (`tests/test_voxcpm2_engine.py`).

### 12.3 VoxCPM2 voice store

**Scenario: store round-trip.** Given a valid id. When a WAV + JSON is saved, listed via `listing()`, read via `info()`, then deleted. Then all operations succeed; after delete `exists()` returns `False`. **Test:** `test_voxcpm2_store_roundtrip` (`tests/test_voxcpm2_store.py`).

**Scenario: store rejects invalid ids.** Given ids with path traversal, uppercase, too short, too long. When passed to `VoxCPM2VoiceStore`. Then `ValueError` is raised; `exists()` returns `False`. **Test:** `test_invalid_ids_rejected_before_touching_disk` (`tests/test_voxcpm2_store.py`).

**Scenario: store lists only complete clones.** Given a directory with a complete clone (`.wav` + `.json`) and an incomplete one (`.wav` only). When `listing()` is called. Then only the complete clone appears. **Test:** `test_voxcpm2_store_incomplete_clone_skipped` (`tests/test_voxcpm2_store.py`).

**Scenario: store writes JSON last.** Given a mock filesystem. When `save()` is called. Then `.wav` is written before `.json`. A crash between the two leaves the clone unlisted. **Test:** `test_voxcpm2_store_atomic_write` (`tests/test_voxcpm2_store.py`).

### 12.4 Lifecycle and state

**Scenario: state transitions.** Given a server with VoxCPM2 disabled. When `start_tts()` runs (VoxCPM2 path). Then state is `"disabled"` when `tts_voxcpm_enabled=False`. When a speech request triggers a load. Then state moves `not_loaded` → `loading` → `ready` (or `error` on failure). **Test:** `test_voxcpm2_state_transitions` (`tests/test_voxcpm2_api.py`).

**Scenario: readiness reports voxcpm2 state.** With `voxcpm2_state="ready"`. When `GET /health/ready` is called. Then the response includes `"voxcpm2": "ready"`. **Test:** `test_ready_reports_voxcpm2_state` (`tests/test_voxcpm2_api.py`).

**Scenario: models lists voxcpm2 when not disabled.** With `voxcpm2_state="ready"`. When `GET /v1/models` is called. Then `"voxcpm2"` appears in model ids. **Test:** `test_models_lists_voxcpm2` (`tests/test_voxcpm2_api.py`).

**Scenario: models omits voxcpm2 when disabled.** With `voxcpm2_state="disabled"`. When `GET /v1/models` is called. Then `"voxcpm2"` is absent. **Test:** `test_models_omits_voxcpm2_when_disabled` (`tests/test_voxcpm2_api.py`).

**Scenario: single-flight load.** With `voxcpm2_state="not_loaded"`. When two concurrent speech requests arrive. Then exactly one load thread is spawned; both requests return 503. When one speech and one clone-create request arrive concurrently. Then exactly one load thread is spawned (both call `_ensure_voxcpm2_loading()`); the speech returns 503; the clone-create returns 503. **Test:** `test_voxcpm2_single_flight_load` (`tests/test_voxcpm2_api.py`).

**Scenario: idle unload.** With a mock last-used time older than `tts_voxcpm_idle_seconds`. When the timer ticks. Then the model is unloaded, `torch.cuda.empty_cache()` is called, state is `"not_loaded"`. **Test:** `test_voxcpm2_idle_unload` (`tests/test_voxcpm2_api.py`).

**Scenario: idle unload with tts_voxcpm_idle_seconds=0.** Timer never starts; engine stays loaded indefinitely. **Test:** `test_voxcpm2_idle_unload_disabled` (`tests/test_voxcpm2_api.py`).

**Scenario: unload skips when a request holds the lock.** Timer tries to acquire `tts_lock` with `blocking=False` and fails; it sleeps and retries next cycle. The running request completes normally. **Test:** `test_voxcpm2_unload_skips_under_request` (`tests/test_voxcpm2_api.py`).

**Scenario: unload timer skips while a load is in flight.** With `voxcpm2_state="loading"` (the loader in its pre-`tts_lock` phase, `tts_lock` free) and a `voxcpm2_last_used` older than `tts_voxcpm_idle_seconds`. When the timer ticks. Then the tick skips — the state is not `ready` (§7.3 step 2), so it is never clobbered to `not_loaded`: a subsequent speech request still sees `loading` and returns the loading 503, and `/health/ready` still reports `loading`. **Test:** `test_voxcpm2_timer_skips_during_load` (`tests/test_voxcpm2_api.py`).

**Scenario: unload timer skips just after a load completes.** Given a `voxcpm2_last_used` stale from before the load (older than `tts_voxcpm_idle_seconds`) and a load that just completed (state `ready`, no speech since). When the timer ticks immediately after the ready transition. Then the engine is not unloaded — the loader set `voxcpm2_last_used` at the ready transition under `tts_lock` (§7.2), so the idle clock restarted; no reload thrash. **Test:** `test_voxcpm2_timer_skips_after_load` (`tests/test_voxcpm2_api.py`).

**Scenario: request during unload gets not_loaded 503.** When the unload holds `tts_lock`, an arriving speech request blocks on `tts_lock`, then finds `voxcpm2_state="not_loaded"` → 503. **Test:** `test_voxcpm2_request_during_unload` (`tests/test_voxcpm2_api.py`).

**Scenario: state re-check under lock catches unload race.** With `voxcpm2_state="ready"`. When a speech request passes the gate, an unload completes before the request acquires `tts_lock`. Then the request's re-check sees `not_loaded` → sets `not_loaded`, starts a new load, returns 503 `not_loaded` — never 500. **Test:** `test_voxcpm2_state_recheck_under_lock` (`tests/test_voxcpm2_api.py`).

**Scenario: load failure sets error, no auto-retry.** When `_load_voxcpm2()` raises an exception. Then state is set to `error`; polling requests see 503 `error`; the page shows the error message (§11.6); the pending request is discarded; nothing auto-retries. **Test:** `test_voxcpm2_load_failure_sets_error` (`tests/test_voxcpm2_api.py`).

### 12.5 Shared TTS lock

**Scenario: shared lock serialises both engines.** Given a stub Reverb engine and a stub VoxCPM2 engine both recording concurrency. When two concurrent requests (one Reverb speech, one VoxCPM2 speech) are made. Then at most one engine's inference is in-flight at any moment. **Test:** `test_shared_tts_lock_serialises_both_engines` (`tests/test_voxcpm2_api.py`).

**Scenario: voxcpm2 seeding is under the shared lock.** Given a stub that records when `torch.manual_seed` is called relative to the lock. When two concurrent seeded VoxCPM2 requests are made. Then each `manual_seed` call and its corresponding `generate` call run as an atomic pair. **Test:** `test_voxcpm2_seeding_atomic` (`tests/test_voxcpm2_api.py`).

### 12.6 Speech (VoxCPM2)

**Scenario: model=voxcpm2 routes to VoxCPM2 engine.** With both stubs present. When `POST /v1/audio/speech` is called with `{"model": "voxcpm2", "input": "Hi", "voice": "default"}`. Then the VoxCPM2 stub's `generate` is called; `X-Speakeasy-Engine` is `"voxcpm2"`. **Test:** `test_speech_voxcpm2_routing` (`tests/test_voxcpm2_api.py`).

**Scenario: voxcpm2 speed != 1.0 → 400.** With `{"model": "voxcpm2", "speed": 1.5}`. Then response is 400 with the fixed message. **Test:** `test_voxcpm2_speed_rejected` (`tests/test_voxcpm2_api.py`).

**Scenario: voxcpm2 fields on Reverb → 400.** With `{"model": "reverb-tts", "cfg": 3.0}`. Then response is 400 naming `voxcpm2`. **Test:** `test_voxcpm2_fields_rejected_on_reverb` (`tests/test_voxcpm2_api.py`).

**Scenario: cfg out of range → 422.** With `{"model": "voxcpm2", "cfg": 0.5}` or `"cfg": 6.0`. Then response is 422 (Pydantic). **Test:** `test_voxcpm2_cfg_bounds` (`tests/test_voxcpm2_api.py`).

**Scenario: design with parentheses or newline → 400.** With `design` containing `(`, `)`, or `\n`. Then response is 400. **Test:** `test_voxcpm2_design_style_syntax` (`tests/test_voxcpm2_api.py`).

**Scenario: design with non-default voice → 400.** With `{"design": "...", "voice": "clone-xy"}`. Then response is 400. **Test:** `test_voxcpm2_design_requires_default` (`tests/test_voxcpm2_api.py`).

**Scenario: both design and style → 400.** With `{"design": "...", "style": "..."}`. Then response is 400. **Test:** `test_voxcpm2_design_style_mutual_exclusion` (`tests/test_voxcpm2_api.py`).

**Scenario: voice "default" works.** With a stub VoxCPM2 engine. When speech is called with `voice: "default"`. Then the stub's `generate` is called with no reference paths. **Test:** `test_voxcpm2_voice_default` (`tests/test_voxcpm2_api.py`).

**Scenario: voice omitted under voxcpm2 is default.** With a stub VoxCPM2 engine. When `POST /v1/audio/speech` is called with `{"model": "voxcpm2", "input": "Hi"}` and no `voice` field. Then the request is plain speech — the voice resolves to `"default"` (no reference paths, never a 400, never the Reverb preset `en_male_narrator_1`). **Test:** `test_voxcpm2_voice_omitted_is_default` (`tests/test_voxcpm2_api.py`).

**Scenario: clone voice with transcript → continuation mode.** Given a VoxCPM2 store with a clone that has a transcript. When speech is called with that clone's id. Then the stub's `generate` is called with `prompt_wav_path`, `prompt_text`, and `reference_wav_path` all set. **Test:** `test_voxcpm2_clone_with_transcript` (`tests/test_voxcpm2_api.py`).

**Scenario: clone voice without transcript → reference-only mode.** Given a clone stored without transcript (`transcript: null`). When speech is called. Then the stub's `generate` is called with only `reference_wav_path`. **Test:** `test_voxcpm2_clone_without_transcript` (`tests/test_voxcpm2_api.py`).

**Scenario: style prefix prepended.** With `{"style": "cheerful", "voice": "clone-xy", "input": "Hello."}`. Then the library receives `text = "(cheerful)Hello."`. **Test:** `test_voxcpm2_style_prefix` (`tests/test_voxcpm2_api.py`).

**Scenario: design prefix prepended.** With `{"design": "A young woman", "voice": "default", "input": "Hello."}`. Then the library receives `text = "(A young woman)Hello."`. **Test:** `test_voxcpm2_design_prefix` (`tests/test_voxcpm2_api.py`).

**Scenario: bad take → 503.** Given a VoxCPM2 stub whose `generate` returns `(audio, True)` — a bad take. When the route maps `bad_case`. Then response is 503 with the bad-take message. The stub's audio is discarded; only a `(audio, False)` tuple flows through to encoding. **Test:** `test_voxcpm2_bad_take_503` (`tests/test_voxcpm2_api.py`).

**Scenario: response formats at 48 kHz.** With a stub returning 1 s of float32 audio. When speech is called with each format. Then `wav` byte count matches 48000 * 2 bytes; `pcm` is exactly 96000 bytes with `audio/pcm`. **Test:** `test_voxcpm2_response_formats` (`tests/test_voxcpm2_api.py`).

**Scenario: 503, 401 states.** `not_loaded`/`loading`/`error` each return 503 with the correct fixed message (§5.1); `disabled` answers 404 (engine not listed); missing key returns 401. **Test:** `test_voxcpm2_speech_503_401` (`tests/test_voxcpm2_api.py`).

**Scenario: clone deleted between resolution and library call.** Given a store whose `exists` returns `True` but whose `.wav` is missing. When speech is called with that clone's id. Then `FileNotFoundError` from the library → caught → 400 `"unknown voice"`, never 500. **Test:** `test_voxcpm2_clone_deleted_before_library_call` (`tests/test_voxcpm2_api.py`).

### 12.7 Clone create and delete (VoxCPM2)

**Scenario: clone create stores reference, no model call.** Given a 3 s WAV uploaded with `model=voxcpm2` and a transcript. When `POST /v1/audio/voices` is called. Then response is 201 with `engine: "voxcpm2"`; `.wav` and `.json` files exist; the VoxCPM2 engine stub's `generate` was never called. **Test:** `test_voxcpm2_clone_create_stores_only` (`tests/test_voxcpm2_api.py`).

**Scenario: clone create with text optional for voxcpm2.** Given an upload with `model=voxcpm2` and no `text`. Then response is 201; `transcript` is `null` in the body. **Test:** `test_voxcpm2_clone_create_no_text_ok` (`tests/test_voxcpm2_api.py`).

**Scenario: clone create with text required for Reverb.** Given an upload with `model=reverb-tts` (Reverb) and no `text`. Then response is 400. (Existing Reverb test covers this; the VoxCPM2 path does not break it. The test sends `model=reverb-tts` since `model` is now required.) **Test:** the existing Reverb clone-validation test (`test_clone_validation_errors`, adapted to send the required model field).

**Scenario: clone create id collision across engines.** Given a Reverb clone saved as `"shared-id"`. When `POST /v1/audio/voices` is called with `model=voxcpm2` and `voice_id="shared-id"`. Then response is 409. **Test:** `test_voxcpm2_clone_cross_engine_collision` (`tests/test_voxcpm2_api.py`).

**Scenario: delete returns engine.** Given a VoxCPM2 clone. When `DELETE /v1/audio/voices/{id}` is called. Then response is 200 with `"engine": "voxcpm2"`. When a Reverb clone is deleted. Then response includes `"engine": "reverb-tts"`. **Test:** `test_delete_returns_engine` (`tests/test_voxcpm2_api.py`).

**Scenario: delete traverses both stores.** When `DELETE` is called with an id that exists in neither store. Then response is 404. The delete checks the VoxCPM2 store first, then the Reverb store. **Test:** `test_delete_checks_both_stores` (`tests/test_voxcpm2_api.py`).

**Scenario: clone create during load returns 503 and starts load in not_loaded.** With `voxcpm2_state="loading"`. When `POST /v1/audio/voices` is called with `model=voxcpm2`. Then response is 503 with the loading reason. With `voxcpm2_state="not_loaded"`. When the clone-create is called. Then response is 503 with the not_loaded reason and the single-flight load starts. With `voxcpm2_state="ready"`. When the clone-create is called. Then response is 201 — the reference is stored and no model runs. **Test:** `test_voxcpm2_clone_during_load` (`tests/test_voxcpm2_api.py`).

**Scenario: clone upload limits (VoxCPM2 path).** Over 20 MB → 413; under 1 s → 400; over 30 s → 400; undecodable → 400. **Test:** `test_voxcpm2_clone_upload_limits` (`tests/test_voxcpm2_api.py`).

### 12.8 Voice list

**Scenario: list includes both engines.** With a Reverb engine (presets + 1 clone) and a VoxCPM2 store (1 clone). When `GET /v1/audio/voices` is called. Then `engines` is `["reverb-tts", "voxcpm2"]`; `data` has `engine` on every object; Reverb presets + clones appear first, then VoxCPM2 clones. **Test:** `test_list_voices_both_engines` (`tests/test_voxcpm2_api.py`).

**Scenario: list when Reverb is loading, VoxCPM2 ready.** Reverb presets absent; VoxCPM2 clones present; `engines` is `["voxcpm2"]`. **Test:** `test_list_voices_mixed_states` (`tests/test_voxcpm2_api.py`).

**Scenario: list when both engines unavailable → 503.** With `tts_voxcpm_enabled=False` (voxcpm2 disabled) and `tts_state` not `ready`. When `GET /v1/audio/voices` is called. Then response is 503. With voxcpm2 enabled in any state (even `not_loaded` — clones from disk), the listing answers 200. **Test:** `test_list_voices_503_both_unavailable` (`tests/test_voxcpm2_api.py`).

**Scenario: clone deleted mid-listing absent.** A clone whose files vanish between the listing scan and the read is absent from `data` — no 500. **Test:** `test_list_voices_voxcpm2_clone_deleted_mid_listing` (`tests/test_voxcpm2_api.py`).

**Scenario: languages present when Reverb ready, omitted otherwise.** With `tts_state="ready"` (Reverb loaded). When `GET /v1/audio/voices` is called. Then `languages` is present with today's pinned value `["de","en","es","fr","it","pl","pt"]`. With `tts_state="disabled"` (Reverb not ready). Then `languages` is absent from the response body — the page keeps the language-select content across re-renders and refills it when a later listing carries the key. **Test:** `test_list_voices_languages_present_when_reverb_ready` (`tests/test_voxcpm2_api.py`).

### 12.9 UI

**Scenario: index page unchanged CSP.** `GET /` returns same CSP as before; no new inline script/style. **Test:** `test_index_page` (`tests/test_ui.py` — existing, extended).

**Scenario: new controls present.** With the page loaded. Then the engine `<select>` exists, `cfg` range input exists, `design` and `style` text areas exist. **Test:** `test_voxcpm2_controls_present` (`tests/test_ui.py`).

**Scenario: engine switch filters voices.** Switching engine to `voxcpm2` removes Reverb presets and shows only `"default"` + VoxCPM2 clones in the voice `<select>`. **Test:** `test_voxcpm2_engine_filter` (`tests/test_ui.py`).

### 12.10 Model selection and voice length

**Scenario: typo'd model on speech → 404 on every engine state.** With any engine state (disabled, not_loaded, loading, ready, error). When `POST /v1/audio/speech` is called with `{"model": "voxpcm2", "input": "Hi"}`. Then response is 404 with the `unknown model` body (§5.1 rule 10); no engine is touched (no state read, no lock). **Test:** `test_typo_model_speech_404` (`tests/test_voxcpm2_api.py`). With a Reverb typo (`"rever-tts"`): same 404, attributed to `tests/test_tts.py`.

**Scenario: typo'd model on clone create → 404 before any write.** With any voxcpm2 engine state (disabled, not_loaded, loading, ready, error). When `POST /v1/audio/voices` is called with `model=voxcpm` and a valid audio file. Then response is 404 with the `unknown model` body on every state — the check runs before any engine gate (§10); no file is written. **Test:** `test_typo_model_clone_create_404` (`tests/test_voxcpm2_api.py`).

**Scenario: missing or empty model → 422 on both routes.** With `POST /v1/audio/speech` body `{"input": "Hi"}` (no `model` field) → 422. With `{"model": "", "input": "Hi"}` → 422. Same for `POST /v1/audio/voices` multipart with no `model` field or an empty one → 422. **Test:** `test_model_missing_empty_422` (`tests/test_voxcpm2_api.py`).

**Scenario: 65-character voice → 422.** With `POST /v1/audio/speech` body `{"model": "voxcpm2", "input": "Hi", "voice": "<65 chars>"}`. Then response is 422 (Pydantic `max_length`). A 64-character voice passes validation and proceeds to voice resolution. The existing `test_voxcpm2_speech_rejects_invalid_voice_ids` keeps its traversal, uppercase and newline cases as 400s; its over-length case is superseded by this 422 and is dropped from that test's 400 loop, so the cap costs none of the hostile-id route coverage (§9 SEC-V3). **Test:** `test_voice_max_length` (`tests/test_voxcpm2_api.py`).

**Scenario: voxcpm2 disabled → absent from listing, speech/clone 404.** With `tts_voxcpm_enabled=False` (voxcpm2 disabled). When `GET /v1/models` is called. Then `voxcpm2` is absent from the model ids. When `POST /v1/audio/speech` is called with `{"model": "voxcpm2", "input": "Hi"}`. Then response is 404 (model unlisted), never 503. When `POST /v1/audio/voices` is called with `model=voxcpm2`. Then response is 404. **Test:** `test_voxcpm2_disabled_unlisted_404` (`tests/test_voxcpm2_api.py`).

**Scenario: both engines enabled, nothing loaded → listing names both.** With `tts_voxcpm_enabled=True` and `tts_enabled=True` but neither engine loaded. When `GET /v1/models` is called. Then both `reverb-tts` and `voxcpm2` appear in the listing (by configuration, not load state — §5.5). When `POST /v1/audio/speech` is called with `{"model": "reverb-tts", "input": "Hi"}`. Then response is 503 with the Reverb loading reason — never a 404. **Test:** `test_models_lists_both_when_unloaded` (`tests/test_voxcpm2_api.py`) and `test_listed_reverb_answers_loading_503` (`tests/test_tts.py`).

**Scenario: Reverb disabled → absent from listing, speech 404.** With `tts_enabled=False`. When `GET /v1/models` is called. Then `reverb-tts` is absent. When `POST /v1/audio/speech` is called with `{"model": "reverb-tts", "input": "Hi"}`. Then response is 404 (model unlisted). **Test:** `test_reverb_disabled_unlisted_404` (`tests/test_tts.py`).

**Scenario: 404 body echoes at most 64 chars of model value.** With a model value longer than 64 characters. When `POST /v1/audio/speech` is called. Then the 404 body's `<value>` is truncated to 64 characters with an ellipsis. **Test:** `test_unknown_model_truncates_value` (`tests/test_voxcpm2_api.py`).

**Scenario: page engine selector hides voxcpm2 when disabled.** With `tts_voxcpm_enabled=False`. When the page loads and `/health/ready` reports `voxcpm2: "disabled"`. Then the engine selector's `voxcpm2` option is absent. **Test:** `test_engine_selector_hides_voxcpm2_when_disabled` (`tests/test_ui.py`).

## 13. Slices

Nine coder slices in dependency order. Each produces a conventional commit on its branch; each branch starts from the previous slice's merge. The last slice's tip is the lane deliverable.

| Slice | Key | Branch | Files | Tests |
|-------|-----|--------|-------|-------|
| c1 | Dependencies | `build/tts-voxcpm-extra` | `pyproject.toml`, `uv.lock`, `Dockerfile.cuda`, `Dockerfile.rocm` | `tests/test_lock_default.py` (existing, must pass unchanged, plus the new supply-chain test), `tests/test_images.py` (new) |
| c2 | Engine class | `feat/voxcpm2-engine` | `src/speakeasy/engines.py` (new `VoxCPM2Engine` class) | `tests/test_voxcpm2_engine.py` (engine unit tests) |
| c3 | Voice store | `feat/voxcpm2-voice-store` | `src/speakeasy/voices.py` (new `VoxCPM2VoiceStore` class) | `tests/test_voxcpm2_store.py` (store tests) |
| c4 | Lifecycle | `feat/voxcpm2-lifecycle` | `src/speakeasy/config.py` (new settings), `src/speakeasy/server.py` (states, load, unload, shared lock, readiness, models) | `tests/test_voxcpm2_api.py` (lifecycle half), `tests/test_config.py` (defaults scenario) |
| c5 | Speech | `feat/voxcpm2-speech` | `src/speakeasy/server.py` (speech routing, validation, encoding) | `tests/test_voxcpm2_api.py` (speech half) |
| c6 | Clone routes | `feat/voxcpm2-clone-routes` | `src/speakeasy/server.py` (clone create, delete, listing for both engines) | `tests/test_voxcpm2_api.py` (clone half) |
| c7 | Web UI | `feat/voxcpm2-web-ui` | `src/speakeasy/static/index.html`, `app.js`, `style.css` | `tests/test_ui.py` |
| c8 | Docs | `docs/voxcpm2-and-spike-005` | `README.md`, `README.docker.md`, `spikes/005-voxcpm2-rocm/` (imported from `ref/voxcpm2/`), `scripts/tts_roundtrip.py` (existing, extended with `--engine` and `--cfg`) | Docker sanity (manual at landing) |
| c9 | Model required | `feat/voxcpm2-model-required` | `src/speakeasy/server.py` (model listing check, model=required on SpeechRequest and clone-create, voice max_length=64, disabled → unlisted → 404), `src/speakeasy/static/app.js` (engine selector hides voxcpm2 when disabled), `README.md` (add model to every speech request), `scripts/tts_roundtrip.py` (send model on every speech request, Reverb included) | `tests/test_voxcpm2_api.py` (model listing, voice cap, disabled-unlisted scenarios), `tests/test_tts.py` (Reverb model scenarios), `tests/test_ui.py` (selector hides when disabled). **Review:** delta review on top of the lane review in flight. |

**Gate for every slice:** `git add` every new file, then `pre-commit run --all-files`; `uv run --frozen --extra cpu --extra tts pytest tests/ -q`; and `UV_PROJECT_ENVIRONMENT="$PWD/.venv-ci" uv sync --frozen --all-groups && UV_PROJECT_ENVIRONMENT="$PWD/.venv-ci" uv run --frozen pytest tests/ -q`. Only c1 may change `pyproject.toml`, `uv.lock`, or the Dockerfiles.

### Slice c1 contract (dependencies)

- Add a `tts-voxcpm` extra as in `ref/voxcpm2/results/pyproject.diff`.
- Add the ten override-dependencies entries; `sys_platform == 'never'` for each.
- `uv lock` — verify zero existing pinned versions changed and fsspec stays at 2026.9.0. Use `uv lock --dry-run` diff.
- `Dockerfile.cuda`: add `--extra tts-voxcpm` to both `uv sync` lines; add `import voxcpm` to the sanity step.
- `Dockerfile.rocm`: add `--extra tts-voxcpm` to both `uv sync` lines; add a two-line sanity step: `import voxcpm` and `from voxcpm import VoxCPM` (a warm import, no model load).
- `tests/test_lock_default.py` must keep passing — the CPU build is unchanged — and gains `test_voxcpm2_extra_excludes_supply_chain_leaks` (SEC-V11's supply-chain proof test).
- `tests/test_images.py` (new): asserts both Dockerfiles carry `--extra tts-voxcpm` on both `uv sync` lines, both sanity steps import the engine, and no `uv sync` follows the source-built CTranslate2 wheel in `Dockerfile.rocm`.

### Slice c8 contract (docs and round-trip)

- `README.md`: add `SPEAKEASY_TTS_VOXCPM_*` config rows (enabled, model, idle_seconds). Add 48 kHz `pcm` note, VoxCPM2 model description, CC BY-NC 4.0 vs Apache-2.0 license note, and a line stating recovery from a failed VoxCPM2 load is a server restart (no auto-retry).
- `README.docker.md`: note that VoxCPM2 is verified on ROCm only; the CUDA image imports it but no NVIDIA GPU has run inference.
- `scripts/tts_roundtrip.py` (existing): extended with `--engine` and `--cfg`. Run inside the ROCm image with `--engine voxcpm2`, it polls `/health/ready` until `voxcpm2 == "ready"`, then tests plain speech, design, clone, styled clone, transcript-assisted clone, bad-take path, `cfg` bounds, and idle unload. Each step outputs one JSON line. No new round-trip script.
- Import `ref/voxcpm2/README.md` (spike record) and `ref/voxcpm2/results/` into the repo as `spikes/005-voxcpm2-rocm/`, with a one-line README linking to the spike record.

## 14. Risks and open questions

| #  | Risk / question                                                                                           | Who settles it                     | When       |
|----|-----------------------------------------------------------------------------------------------------------|------------------------------------|------------|
| 1  | **Loudness.** Output varies by up to 17 dB across `cfg` and ~4 dB on seed. Normalising would change the sound the owner approved. | Owner — after using it             | post-merge |
| 2  | **Reference length for VoxCPM2.** The library's `_encode_wav` pads to patch-aligned length and imposes no minimum. The spec keeps Reverb's 1–30 s. If a 1 s clip crashes the pad-to-patch logic, the minimum is raised. | Coder (c5) — verify at implementation | c5         |
| 3  | **Reproducibility.** A request without `seed` is not reproducible. With the shared lock two engines never run at once, so a seed sequence is deterministic across requests. The spec states both. | Documented — no action             | —          |
| 4  | **CUDA runtime.** No NVIDIA GPU has run VoxCPM2 in speakeasy. The CUDA image imports the engine; the orchestrator runs a real CUDA take at landing. If it fails, the lane ships with a CUDA limitation note. | Orchestrator at landing            | landing    |
| 5  | **First-use wait on a cold cache.** The 4.7 GB download takes minutes; the page polls `/health/ready`. If the owner's volume is cold, the first successful request can be minutes away. The 503 message names the ~4.7 GB download. | Coder (c4, c7) — clear loading message | c4, c7     |
| 6  | **`torch.manual_seed` determinism across GPU architectures.** Verified on gfx1151 only; a CUDA or different HIP arch may produce different audio for the same seed. | Orchestrator at landing            | landing    |
| 7  | **48 kHz `pcm` format.** 48 kHz s16le mono is unusual — most clients expect 24 kHz from OpenAI's TTS API. The README documents it; users of the `pcm` format must handle the higher rate. | Documented in README (c8)          | c8         |
| 8  | **`design` and `style` quality variance.** The model card notes results "may vary between runs; generating 1–3 times is recommended." The server does not retry on quality; the user can re-request with a different seed. | Documented — no action             | —          |
| 9  | **Voice list `engines` and backward compatibility.** The `engines` array and per-object `engine` field are new. Existing clients that read the voice list will see new fields; they are additive and should not break a client that ignores unknown keys. | Documented — no action             | —          |
| 10 | **Preset-name collision across engines.** While the Reverb engine is not loaded, a VoxCPM2 clone can be created under the id of a Reverb preset — preset ids exist only on a loaded engine, so the cross-engine check cannot see them; once Reverb loads, the voice listing shows two objects with that id. Accepted for now: the alternative is a hand-copied preset list that goes stale. | Accepted — no action               | —          |
