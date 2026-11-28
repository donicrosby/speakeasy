# Speakeasy streaming speech

> Spec v2.0. Validated against speakeasy `main` `301cf43`, the brief `brief-no-default-and-streaming.md` §5C with its addendum (rulings S12–S16), and OpenAI's published interface (OpenAPI spec `openai.yaml` on `master`, the Rust crate `async-openai`, and the reference page at `developers.openai.com`, all read 2026-11-27). Contracts: `docs/specs/tts-voice-cloning-ui.md` (Reverb) and `docs/specs/tts-voxcpm2-engine.md` (VoxCPM2) — this spec extends both and changes neither except for one cross-reference line each (S11).
>
> **Revision from v1.0:** S12–S16 from the addendum are carried in. Lock ownership rewritten (S12). Buffer cap is a hard stop, never a park (S13). Buffer bound recalculated from the true worst case (includes Reverb's `speed` factor). The page plays through Web Audio from `pcm` in SSE events (S14). `X-Speakeasy-Sample-Rate` on both streaming forms (S15). `data: [DONE]` and keepalive comments are withdrawn — the stream is exactly OpenAI's two events (S16). Citations ground the interface in real sources. The probe sink is defined. The equivalence pass rule is one rule. Slice attribution corrected. Reviewer mechanical corrections carried from the review branch.

## 1. Goals and non-goals

### Goal

Speech can be delivered as it is generated, for callers that want first audio sooner than the whole take (brief §1, piece (c)). A request without the new field behaves exactly as today, byte for byte (S1).

### Non-goals

- Streaming transcription (brief §5C, non-goals).
- Server-sent events on any route other than speech.
- OpenAI's Realtime API.
- HTTP trailers — behaviour through the owner's proxy is unverified (S5).
- Changing what an engine generates or any non-streaming response.
- New engines, new dependencies, any change to `pyproject.toml`, `uv.lock` or a Dockerfile (C1).
- A nano-vLLM backend — the owner's decision is pending (brief §1, not a goal).

## 2. Owner's decisions

Copied verbatim from the brief §2. These are settled; the spec restates them and does not reopen them.

His message of 2026-11-27, verbatim:

> Fix the README, yes no default model, yes to streaming endpoint, 10 step default is probably better.

| #  | His words | The answer that matters here | What it settles |
|----|-----------|------------------------------|------------------|
| D3 | "yes to streaming endpoint" | The parked option "streaming endpoint: server work and the page" | It is built. |
| D4 | "10 step default is probably better" | VoxCPM2's diffusion steps, 5 against 10 | Stays 10 (`src/speakeasy/engines.py:287`). No card changes it, and it is not made configurable. |

## 3. Orchestrator's rulings (restated as requirements)

Each ruling from the brief §5C and its addendum, restated as a requirement with the check that proves it.

| #  | Requirement | Check |
|----|-------------|-------|
| S1 | **The interface is OpenAI's, verified.** The request field is `stream_format` with values `"sse"` and `"audio"`. When absent, behaviour is byte-for-byte unchanged from today. The SSE events are `speech.audio.delta` and `speech.audio.done`. Source: the upstream OpenAPI spec `openai/openai-openapi`, `openapi.yaml` on `master` (read 2026-11-27) — `CreateSpeechResponseStreamEvent` = `SpeechAudioDeltaEvent`(`type`, `audio`, both required) or `SpeechAudioDoneEvent`(`type`, `usage`, both required). The Rust crate `async-openai` `types/audio/stream.rs` (docs.rs, read 2026-11-27) confirms the same schema. At OpenAI the field's default is `audio`; Speakeasy's absent-means-today is ruling S1 and is documented as a deliberate difference — the README names it. | A test sends `stream_format: "sse"` and asserts each `data:` line's JSON matches the schema. A test sends `stream_format: "audio"` and asserts chunked transfer with the content type of the response format. A test omits `stream_format` and asserts the response is bit-for-bit unchanged from today. |
| S2 | **Formats stream by evidence, not hope.** `pcm` streams always. For `wav`, `mp3` and `flac`, a probe encodes to a non-seekable sink (a `BytesIO` subclass whose `seek` raises `OSError`) with libsndfile (soundfile 0.14.0), decodes the concatenated bytes with an independent reader, and records the result. The sink is defined here because a plain `BytesIO` is seekable by definition. A format that cannot be encoded incrementally and decoded correctly is refused for streaming with 422 and a fixed message naming the formats that can. **Escape hatch: none; a format streams only if the probe shows it. ASSUMED until the probe runs.** Reviewer evidence (2026-11-27, this venv, the same sink, three 0.16 s chunks at 24 kHz): WAV refuses to open for writing (libsndfile asks for the file length up front); MP3 encodes but decodes to 15552 of 11520 samples; FLAC encodes but does not decode at all; only `pcm` streamed intact (23040 bytes, 11520 samples exactly). A sink that answers the initial length probe but raises on later seeks fails the same three formats with different artefacts (WAV decodes to 11564 of 11520 and stdlib `wave` rejects it; MP3 decodes 9432 of 11520). | `test_streaming_format_probe` — calls `_encode_chunk` on each format's first 0.16 s with the seek-raising sink and verifies the concatenated bytes decode correctly; formats that fail are absent from the allowed list. |
| S3 | **Engines.** VoxCPM2 streams through `generate_streaming` (F16). Reverb has no streaming entry point (F17): a streaming request to `reverb-tts` returns the finished take as one chunk when it is ready. That is documented behaviour, not an error, and the response is not labelled with it. | `test_reverb_stream_returns_finished_take` — the Reverb path to a streaming request answers 200 with the full audio in one chunk. |
| S4 | **Delivery changes, the take does not.** The audio of a stream is the take the non-streaming route would return for the same request and seed, to within float rounding: same sample count, largest absolute sample difference at most 1e-5 (measured 1.2e-6, F10). Byte identity is promised by neither path. Coder pins it with fake engines; the orchestrator measures it on the real engine at landing. | `test_stream_and_non_stream_same_take` — a fake VoxCPM2 that records its output returns identical samples from both paths (fake data, so byte identity holds). The orchestrator runs `ref/nds/stream_eq.py` on the real engine at landing and asserts one pass rule: same sample count and `max_abs_diff <= 1e-5`. |
| S5 | **Statistics.** On an SSE stream the final `speech.audio.done` event carries the four values of today's headers as JSON. On a raw chunked stream only what is known before the first byte is sent as a header (`X-Speakeasy-Engine` and `X-Speakeasy-Sample-Rate`); the other three are absent and the README says so. No HTTP trailers. | `test_sse_stream_final_event_carries_stats` — parses the last event and asserts `engine`, `elapsed`, `audio_duration` and `rtf` fields. `test_raw_stream_headers` — asserts `X-Speakeasy-Engine` and `X-Speakeasy-Sample-Rate` are present, the other three absent. |
| S6 | **The engine never waits on the network.** One generating thread drives the engine and writes chunks to an in-memory bounded buffer; the handler reads that buffer as chunks arrive. The buffer is bounded for the largest allowed take (see §6.1), so generation runs to completion whatever the client does and the engine lock is held for the generation time only. A slow or stalled client never holds the engine. | `test_engine_runs_to_completion_when_client_stalls` — a fake engine that takes 0.2 s; the test reads one chunk, closes the connection, then asserts the engine finished and the lock was released. |
| S7 | **A client that leaves is an ordinary outcome.** A disconnect mid-stream produces no 500 and no traceback at error level (one informational line). The generating thread finishes and releases the lock. The idle timer is bumped exactly as for a completed request. The "clone deleted mid-request is never a 500" contract holds for streams. An engine failure after the first byte ends the stream cleanly: an error event with a fixed message on an SSE stream, a truncated body on a raw one, and one error line in the log. | `test_disconnect_mid_stream_no_500` — the client disconnects; the next healthy request succeeds. `test_engine_failure_mid_stream_clean_close` — a fake engine that raises after two chunks; the SSE stream ends with an error event. |
| S8 | **The event loop is never blocked.** Handlers follow the existing pattern: plain `def` on the threadpool. A test pins it: `/health/ready` answers while a stream is mid-take behind a fake engine that waits on an event. | `test_health_responds_during_stream` — a fake engine blocks on a `threading.Event`; a concurrent `/health/ready` call returns 200 within 2 s. |
| S9 | **The page plays speech through Web Audio, from `pcm` carried in events.** For Speak, the page sends `response_format: "pcm"` and `stream_format: "sse"`, for both engines. It decodes each `speech.audio.delta` with `atob`, turns s16le into float samples, and schedules them on one `AudioContext` timeline as they arrive, each chunk starting where the previous one ends. The `AudioContext` is created at the rate of S15 and resumed inside the click handler. At `speech.audio.done` the page builds a WAV Blob from the samples (writing the 44-byte header itself) for the `<audio>` element and the download link exactly as today, and shows the four statistics from `usage`. Reverb arrives as one delta and takes the same path. The fallback is today's non-streaming request where `ReadableStream` is absent. No MediaSource, no dependency, no CSP change. Evidence: headless Chromium 128 `MediaSource.isTypeSupported` is false for `audio/wav`, `audio/flac`, `audio/pcm`; an `AudioContext` constructs at an arbitrary rate; a Blob cannot grow (`ref/nds/stream-capabilities.txt`). Proven in Chromium at landing; other browsers stay `ASSUMED`. | `test_csp_unchanged` — the existing UI test that pins the CSP header stays green unchanged. The orchestrator verifies play-as-it-arrives in a real browser at landing. |
| S10 | **Unchanged by this workstream:** every non-streaming response, auth, the 64-character echo bound, `speed`, `cfg`, `design` and `style` semantics, the listing rules, the idle unload, and D4. | Every existing test stays green. |
| S11 | **Cross-references.** Each TTS spec's speech section gains at most one line saying that streaming delivery is specified in this spec. Nothing else in those files changes. | `grep` of both spec files shows exactly one new line each. |
| S12 | **One thread owns the engine lock for a stream.** The generating thread does what the non-streaming route does today, in the same order: take `tts_lock`; check the engine's state and, if it is not ready, hand the 503 reason to the handler and stop (today's route re-checks under the lock, `server.py:714-724`); generate; hand the first chunk to the handler; bump the idle timer; release `tts_lock` in a `finally` when the generator is exhausted, stopped under S13, or raises. The handler starts that thread, waits for its first item before building the response, and never takes the lock itself. The handler's wait has no timeout of its own: today's route blocks on the lock in the same way. | The lifecycle diagram in §6.1 reflects this. `test_lock_held_by_generating_thread` — the generating thread, not the handler, acquires `tts_lock`. |
| S13 | **The buffer cap ends a generation; it never parks one.** When a put would exceed the bound: the generating thread stops pulling from the engine, closes the engine's generator, releases the lock, bumps the idle timer, and puts the fixed mid-stream error on the queue. Nothing may block on a full queue. A legal request never reaches the cap, so S6 still holds for every legal request. | SEC-6 test (§10) asserts four things: the client gets the mid-stream error (or a truncated raw body), the lock is free (the next speech request succeeds at once), the log holds one error line, and no step waits on a timeout. |
| S14 | **The page plays through Web Audio from `pcm`.** See S9 above. | Page scenarios in §12. |
| S15 | **`X-Speakeasy-Sample-Rate` header on both streaming forms.** The engine's output rate in hertz, an integer, known before the first byte is sent. Added to the header sets of both the event stream and the raw stream, and to the README. | `test_raw_stream_headers` asserts the header is present and correct. `test_sse_stream_headers` asserts it on the event-stream response. |
| S16 | **The event stream is exactly OpenAI's two events.** Zero or more `speech.audio.delta`, then one `speech.audio.done`, then the connection closes. Speakeasy sends no `data: [DONE]` line and no keepalive comment. The `usage` object carries speakeasy's four statistics in place of OpenAI's token counts; the README names that difference. | Scenario 2 asserts the final event is `done` and the stream ends with no `[DONE]`. |

## 4. Verified facts

### VERIFIED — OpenAI's streaming speech interface (read 2026-11-27)

- **O1 — Parameter name and values.** `openai-python` `src/openai/types/audio/speech_create_params.py` (main): `stream_format: Literal["sse", "audio"]`. The OpenAI API reference at `https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create` documents the same parameter with the same values. At OpenAI the documented default is `audio`; Speakeasy's absent-means-no-streaming is ruling S1, a deliberate difference, and the README names it.

- **O2 — SSE event names and payloads.** The upstream OpenAPI spec `openai/openai-openapi`, `openapi.yaml` on `master` (read 2026-11-27) defines `CreateSpeechResponseStreamEvent` as a discriminated union of:
  - `SpeechAudioDeltaEvent`: `{"type": "speech.audio.delta", "audio": "<base64-encoded audio chunk>"}`
  - `SpeechAudioDoneEvent`: `{"type": "speech.audio.done", "usage": {"input_tokens": N, "output_tokens": N, "total_tokens": N}}`
  The Rust crate `async-openai` `types/audio/stream.rs` (docs.rs, read 2026-11-27) confirms the same two-event schema. No `[DONE]` line belongs to speech streaming — the ones present in the OpenAPI spec belong to chat completions, the threads API, and beta endpoints. No keepalive comment is specified for the speech stream. The stream is exactly the two event types above, then the connection closes.

- **O3 — Raw audio stream.** When `stream_format = "audio"`, the response is chunked transfer of raw audio bytes in the requested `response_format`, with `Content-Type` matching the format. The `openai-python` SDK consumes it with `with_streaming_response.create` returning `HttpxBinaryResponseContent` — iterable bytes.

- **O4 — OpenAI also offers `opus` and `aac` formats, and the `instructions` field.** Speakeasy does not implement these; they are non-goals.

### VERIFIED — speakeasy `main` `301cf43`

- **F1 — Speech route.** `server.py:595-605` — `speech()` is a plain `def` (threadpool), checks the model listing first, dispatches to `_speech_voxcpm2` or `_speech_reverb`.

- **F2 — Speech response.** `server.py:752-770` — `_speech_response()` encodes the finished array with `_encode()` and returns one `Response` with four headers: `X-Speakeasy-Engine`, `-Elapsed`, `-Audio-Duration` and `-RTF`.

- **F3 — TTS lock.** `server.py:99` — `tts_lock = threading.Lock()` serialises inference across both engines. `server.py:636` and `714` — both speech paths hold it around the engine call.

- **F4 — VoxCPM2 lifecycle.** `server.py:285-319` — `_ensure_voxcpm2_loading()` starts a single-flight loader. `server.py:329-360` — `_voxcpm2_unload_loop()` consumes the lock every 15 s and unloads when idle. `server.py:714-746` — the speech path re-checks state under the lock.

- **F5 — Idle timer bump.** `server.py:745` — `voxcpm2_last_used = time.monotonic()` under `tts_lock` after a successful take.

- **F6 — Starlette StreamingResponse.** Starlette 1.7.0 `starlette/responses.py:224-257`: `StreamingResponse.__init__` passes an `AsyncIterable` through and wraps any other iterable with `iterate_in_threadpool` (`starlette/concurrency.py:51-59`), which runs every `next()` through `anyio.to_thread.run_sync` — a sync-iterator response is iterated on the threadpool, never the event loop. `stream_response` then sends each chunk as an `http.response.body` ASGI message; the chunked transfer encoding on the wire is uvicorn's. Source: installed Starlette 1.7.0, `responses.py:224-257` and `concurrency.py:51-59`.

- **F7 — libsndfile open paths.** soundfile 0.14.0 `SoundFile._open` (reached from `__init__`) dispatches on the input: `sf_open` for paths, `sf_open_fd` for file descriptors, `sf_open_virtual` for file objects (soundfile.py:1269-1287). No format claim is made here: which container survives a sink that cannot seek is exactly what the probe in S2 decides. Reviewer evidence (2026-11-27, this venv, a `BytesIO` subclass whose every `seek` raises, three 0.16 s chunks at 24 kHz): WAV refuses to open for writing; MP3 encodes but decodes to 15552 of 11520 samples; FLAC encodes but does not decode at all ("array is too big"); only `pcm` streamed intact. A sink that answers the initial length probe and raises only on later seeks fails the same three formats with different artefacts (WAV decodes to 11564 of 11520 and stdlib `wave` rejects it, "fmt chunk and/or data chunk missing"; MP3 decodes 9432 of 11520). Note the probe must name its sink: a `BytesIO` is seekable by definition, so "non-seekable `BytesIO`" names no object.

- **F8 — CSP.** `server.py:369-372` — `default-src 'self'; media-src 'self' blob:; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'`.

- **F9 — CI.** `.github/workflows/test.yml:22` — `uv sync --all-groups` (no extras). The streaming tests must pass without the `tts` extra installed.

### VERIFIED — `ref/nds/` (2026-11-27)

- **F10 — Stream equivalence.** `ref/nds/stream_eq.json` — streamed take and non-streamed take for the same request and seed: identical sample count; `max_abs_diff` 8.3e-7 (short), 1.2e-6 (medium); `max_diff_over_rms` 6.5e-6 and 8.5e-6; zero samples differ by more than 1e-3. Chunks are 0.16 s; the first arrives in 0.10 to 0.37 s (nine takes, spike 006).

- **F11 — Stream determinism.** `ref/nds/stream_eq.json` — the streamed path reproduced itself with the same seed in both takes (`stream_same_seed_identical: true` on both lines). The non-streamed short take did not (`non_stream_same_seed_identical: false`). Neither path may promise byte identity.

- **F12 — `generate_streaming` signature.** `ref/nds/stream_eq.py:34,36` — `model.generate_streaming(*args, **kwargs) -> Generator[np.ndarray, None, None]` takes the same arguments as `generate`.

- **F13 — Engine sample rates.** VoxCPM2: 48,000 Hz (`engines.py:259`, `self._model.tts_model.sample_rate` — the library's sample rate, 48 kHz speech). Reverb: 24,000 Hz (Higgs Audio V2 codec config `sample_rate: int = 24000`, read from installed transformers `configuration_higgs_audio_v2_tokenizer.py`; accessed as `self.codec.config.sample_rate` in `reverb_tts/tts.py:117`).

- **F14 — Reverb speed factor.** `server.py:546, 637-639` — `speed` is valid 0.25–4.0. VoxCPM2 rejects `speed` != 1.0 as a 400. Reverb's `TTS.generate` applies `speed` as a length factor: at `speed = 0.25` the take is 4× the base duration.

### ASSUMED

- **A1 — WAV incremental encoding.** libsndfile on a non-seekable sink must patch the RIFF header size after writing. Whether the WAV is decodable without that patch is determined by the probe (S2). **Settled by:** coder c2b runs the probe; if WAV fails, it is removed from the streaming-allowed list.

- **A2 — Browser play-as-it-arrives.** The Web Audio + `pcm` + SSE path works in the owner's browsers. **Settled by:** orchestrator at landing in a real browser. Other browsers stay `ASSUMED`.

- **A3 — Character rate floor for buffer bound.** The buffer bound uses a floor of 15 characters per second of audio as the slowest synthesis rate. This is slower than both engines' measured output on normal input at speed 1.0 and is a conservative floor for VoxCPM2 and Reverb at speed 1.0. **Settled by:** measuring the duration of a 4096-character input on both engines at speed 1.0 at landing. If either engine's measured rate is below 15 chars/s, the bound is recalculated and the new value is written into this spec.

## 5. HTTP contract

All `/v1/*` endpoints require the bearer token (existing `require_auth`). The non-streaming speech path is unchanged.

### 5.1 `POST /v1/audio/speech` — new request field

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `stream_format` | `"sse"` \| `"audio"` \| absent | absent (no streaming) | Case-sensitive; any other value → 422 (Pydantic `Literal`). |

When `stream_format` is absent: the route follows today's code path exactly, returning one `Response` with four headers. Byte-for-byte unchanged.

### 5.2 Streaming response forms

#### Non-streaming (today)

| Status | Content-Type | Headers |
|--------|-------------|---------|
| 200 | per `response_format` | `X-Speakeasy-Engine`, `-Elapsed`, `-Audio-Duration`, `-RTF` |

#### Raw chunked audio (`stream_format: "audio"`)

| Status | Content-Type | Headers |
|--------|-------------|---------|
| 200 | per `response_format` | `Transfer-Encoding: chunked`, `X-Speakeasy-Engine`, `X-Speakeasy-Sample-Rate` |

`X-Speakeasy-Elapsed`, `-Audio-Duration` and `-RTF` are absent: they are not known until the stream completes (S5). `X-Speakeasy-Sample-Rate` is the engine's output rate in hertz, an integer (S15).

#### Server-sent events (`stream_format: "sse"`)

| Status | Content-Type | Headers |
|--------|-------------|---------|
| 200 | `text/event-stream` | `Cache-Control: no-store`, `X-Accel-Buffering: no`, `X-Speakeasy-Sample-Rate` |

`X-Speakeasy-Sample-Rate` is the engine's output rate in hertz, an integer (S15).

The body is a sequence of `data:` lines — exactly the two events from O2, with no `[DONE]` line and no keepalive comments (S16):

1. Zero or more `speech.audio.delta` events:
   ```
   data: {"type":"speech.audio.delta","audio":"<base64>"}
   ```
   Each carries one chunk of audio in the requested `response_format`, base64-encoded.

2. One `speech.audio.done` event:
   ```
   data: {"type":"speech.audio.done","usage":{"engine":"<name>","elapsed":<float>,"audio_duration":<float>,"rtf":<float>}}
   ```
   The `usage` object carries the server's four statistics (S5), with fields re-keyed:
   - `engine` → today's `X-Speakeasy-Engine`
   - `elapsed` → today's `X-Speakeasy-Elapsed`
   - `audio_duration` → today's `X-Speakeasy-Audio-Duration`
   - `rtf` → today's `X-Speakeasy-RTF`

After `speech.audio.done` the connection closes.

#### Per-format streaming table (S2)

| Format | Streams | Probe result |
|--------|---------|-------------|
| `pcm` | Always | Raw s16le at engine rate — no container, no seeking, streams by construction. |
| `wav` | **ASSUMED until probe runs** | libsndfile writes WAV to the seek-raising sink (§3 S2). Probe encodes → decodes → checks header. Reviewer evidence (F7): fails. |
| `mp3` | **ASSUMED until probe runs** | libsndfile writes MP3 to the seek-raising sink. Probe encodes → decodes → checks frames. Reviewer evidence (F7): fails. |
| `flac` | **ASSUMED until probe runs** | libsndfile writes FLAC to the seek-raising sink. Probe encodes → decodes → checks frames. Reviewer evidence (F7): fails. |

A format that fails the probe answers 422: `{"error": {"message": "streaming not supported for format '<fmt>'. Supported: pcm, <others>"}}`.

#### Engine behaviour per format (S3)

| Engine | `stream_format: "sse"` | `stream_format: "audio"` | Notes |
|--------|------------------------|--------------------------|-------|
| `voxcpm2` | SSE events, 0.16 s chunks, base64 per chunk | Chunked bytes in response format, 0.16 s chunks | Uses `generate_streaming`. First chunk in 0.10-0.37 s (F10). |
| `reverb-tts` | SSE events, one delta with the full take, then done | Chunked bytes, one chunk with the full take | No streaming entry point (F17); finished take returned when ready. Documented, not an error. |

### 5.3 Errors before the stream starts

Every existing error is answered exactly as today, before any byte is sent:

| Code | Condition | Body |
|------|-----------|------|
| 401 | Missing or invalid API key | `{"detail": "Invalid or missing API key"}` |
| 404 | `model` not a listed TTS engine id | `_unknown_model_404(req.model)` — echoes at most 64 chars, names known ids |
| 422 | `model` missing or empty, `input` missing or over 4096 chars, `speed` outside 0.25-4.0, invalid `response_format`, invalid `stream_format` | Pydantic validation detail |
| 422 | streaming in either form (`"sse"` or `"audio"`) on a format that cannot stream (per S2 probe) | `{"error": {"message": "streaming not supported for format '<fmt>'. Supported: pcm, <others>"}}` |
| 503 | `voxcpm2` not loaded/loading/error | Fixed per-state 503 (existing VoxCPM2 contract) |
| 503 | `reverb-tts` not ready | Fixed per-state 503 (existing Reverb contract) |
| 400 | `cfg`/`design`/`style` on Reverb, `speed` != 1.0 on VoxCPM2, etc. | Existing validation errors (VoxCPM2 spec §5.1) |

The `stream_format` check runs after model listing and auth, before any engine state is read.

### 5.4 Errors mid-stream

| Condition | SSE | Raw |
|-----------|-----|-----|
| Engine raises after first byte | One error event `{"type":"error","error":{"message":"<fixed reason>"}}`, then connection closes | Connection closed; the body ends truncated |
| Buffer bound exceeded (S13) | Error event as above; lock released; one error line logged | Truncated body; lock released; one error line logged |
| Client disconnects | Generating thread finishes, lock released, idle timer bumped | Same |

Mid-stream errors carry only a fixed message — never exception text, paths, or user input (SEC-5).

## 6. Lifecycle and concurrency

### 6.1 Buffer, generating thread and lock (S12, S13)

```
Client            Handler (threadpool)     Generating thread
  |                      |                    |
  |-- POST /v1/speech -->|                    |
  |                      |-- start gen-thread->|
  |                      |   (wait for first)  |
  |                      |                    |-- tts_lock.acquire()
  |                      |                    |-- check engine state
  |                      |                    |-- (503 reason, stop if not ready)
  |                      |                    |-- generate_streaming()
  |                      |<-- first chunk -----|  yield first chunk
  |  <-- 200 headers ----|                    |
  |  <-- chunk 1 --------|<-- queue.get() ----|  queue.put(chunk)
  |  <-- chunk 2 --------|<-- queue.get() ----|  queue.put(chunk)
  |  ...                 |                    |
  |  <-- chunk N ---------|<-- queue.get() ----|  queue.put(last)
  |                      |                    |-- bump idle timer
  |                      |                    |-- tts_lock.release()  (finally)
  |                      |   (generator ends)
  |  <-- done / EOF -----|                    |
```

- The handler (plain `def`, threadpool) starts a generating thread and waits for its first item before building the response. The handler never acquires `tts_lock`. Its wait has no timeout of its own — today's route blocks on the lock in the same way.
- The generating thread does what the non-streaming route does today, in the same order: take `tts_lock`; check the engine's state and, if it is not ready, hand the 503 reason to the handler and stop (today's route re-checks under the lock, `server.py:714-724`); generate; hand the first chunk to the handler; bump the idle timer; release `tts_lock` in a `finally` when the generator is exhausted, stopped under S13, or raises.
- **Queue bound:** 67,108,864 bytes (64 MiB). Derived from the true worst case across both engines: 4096 input chars ÷ 15 chars/s (A3 floor) ÷ 0.25 (slowest legal `speed`, Reverb only) = 1092.3 s of audio samples; 1092.3 s × 24,000 Hz (Reverb's rate, F13) × 2 bytes per sample (s16le pcm) = 52.4 MB (50.0 MiB); rounded up to the next power-of-2 boundary. The factors: VoxCPM2's 48 kHz rate gives a smaller value (26.2 MB) because VoxCPM2 rejects `speed` != 1.0; Reverb at speed 0.25 with its 24 kHz codec rate is the worst case. Reverb fits in one chunk, so the full take lands in the queue at once.
- **S13: the cap ends a generation, never parks one.** When a `put()` would exceed the bound: the generating thread stops pulling from the engine, closes the engine's generator, releases the lock, bumps the idle timer, and puts the fixed mid-stream error on the queue. Nothing may block on a full queue. A legal request never reaches the cap, so S6 still holds for every legal request.

### 6.2 Concurrency guarantees

- The single `tts_lock` serialises all engine inference (S3 of the VoxCPM2 spec). A streaming request holds it during generation exactly like a non-streaming one.
- The idle-unload timer (`_voxcpm2_unload_tick`) acquires `tts_lock` with `blocking=False`; it never unloads under a running request.
- A clone deleted mid-stream follows the shipped clone-deletion contract: the reference is read at synthesis start under the lock (Reverb holds the loaded voice object in memory; VoxCPM2's library reads the reference file at generation start), so a delete after synthesis start cannot affect the take — the stream completes normally. A reference that cannot be read at synthesis start is answered with today's status for that case (the unknown-voice 400, `server.py:737-739`) before the first chunk is sent — never a 500, never a mid-stream error (S7, S12).

### 6.3 Race table

| Interleaving | Outcome | Test |
|-------------|---------|------|
| Streaming client + concurrent `/health/ready` | `/health/ready` answers 200 within 2 s | `test_health_responds_during_stream` |
| Streaming client disconnects mid-stream + next speech request | Next request succeeds; no 500 | `test_disconnect_mid_stream_no_500` |
| Engine fails after two chunks | SSE: error event, then close. Raw: truncated body. Log has one error line. | `test_engine_failure_mid_stream_clean_close` |
| Clone deleted while a stream uses it | Deletion after synthesis start cannot affect the take — the stream completes normally (the shipped clone-deletion contract). A reference that cannot be read at synthesis start answers today's unknown-voice 400 before the first chunk. No 500, no mid-stream error. | `test_clone_deleted_mid_stream_no_500` |
| Idle-unload tick fires during stream | Unload skipped (lock not free). Stream completes normally. | `test_idle_unload_skipped_during_stream` |
| Two concurrent streaming requests | Serialised by `tts_lock`; second waits. Both complete. | `test_two_concurrent_streams_serialised` |
| Slow client — reads slower than generation | Generation runs to completion; lock released; buffer drains at client's pace. No stall. | `test_slow_client_no_stall` |
| Request body over the buffer cap | Mid-stream error per S13; lock freed; one error line logged. Next request succeeds at once. | `test_buffer_over_cap_mid_stream_error` |

## 7. Equivalence contract (S4)

**Promise:** the audio of a stream is the take the non-streaming route would return for the same request and seed, to within float rounding: same sample count, largest absolute sample difference at most 1e-5 (F10 measured 1.2e-6, which is 8× smaller). Byte identity is promised by neither path (F11).

**Test with fakes:** `test_stream_and_non_stream_same_take` — a fake VoxCPM2 that records its output returns identical samples from both paths. The test asserts `np.array_equal(streamed, non_streamed)`.

**Measure on real engine at landing:** the orchestrator runs `ref/nds/stream_eq.py` with one pass rule: the same sample count and `max_abs_diff <= 1e-5`.

## 8. Statistics (S5)

| Response form | `X-Speakeasy-Engine` | `-Sample-Rate` | `-Elapsed` | `-Audio-Duration` | `-RTF` |
|--------------|---------------------|----------------|------------|-------------------|--------|
| Non-streaming (today) | Header | Absent | Header | Header | Header |
| Raw chunked (`stream_format: "audio"`) | Header | Header | Absent | Absent | Absent |
| SSE (`stream_format: "sse"`) | In `speech.audio.done` `usage.engine` | `X-Speakeasy-Sample-Rate` response header | In `speech.audio.done` `usage.elapsed` | In `speech.audio.done` `usage.audio_duration` | In `speech.audio.done` `usage.rtf` |

On a raw chunked stream the three absent timing values are not known before the first byte and are not sent as trailers (S5). The README documents this.

## 9. The page (S9, S14)

### 9.1 Streaming playback

For Speak, the page sends `response_format: "pcm"` and `stream_format: "sse"`, for both engines. The streaming path:

1. **Request:** `fetch('/v1/audio/speech', {body: JSON.stringify({...stream_format:'sse', response_format:'pcm'...})})`.
2. **Read SSE events:** use `response.body.getReader()` on the `ReadableStream`.
3. **Decode each `speech.audio.delta`:** base64-decode the `audio` field (`atob`), convert each s16le sample to float by dividing by 32768.0.
4. **Schedule on `AudioContext`:** one `AudioContext` created at the rate of S15 (`X-Speakeasy-Sample-Rate` from the response headers). Schedule each chunk's `AudioBuffer` on the timeline with `start(chunk_start_time)` where `chunk_start_time` = previous chunk's end time. Resume the `AudioContext` inside the click handler (the user gesture browsers require).
5. **At `speech.audio.done`:** build a WAV Blob from the accumulated float samples (write the 44-byte RIFF/WAV header in JS — sample rate, channel count, bit depth). Set the `<audio>` element's `src` to the Blob URL and show the download link exactly as today. Show the four statistics from `usage` in `speak-meta`.
6. **Reverb:** arrives as one delta and takes the same path — one decode, one schedule, one AudioBuffer.

### 9.2 What the page shows

- **While receiving:** an elapsed counter (seconds since first chunk). The download link is hidden or disabled.
- **At completion:** the final statistics (duration, elapsed, RTF) shown in `speak-meta`. The download link appears on the complete audio (WAV Blob).
- **Error:** the existing error display in `speak-error`. No additional UI element.

### 9.3 Fallback

If the browser does not support `ReadableStream` (the `fetch` body reader — `if (!response.body || !response.body.getReader)`), the page falls back to today's non-streaming request exactly (S14): `response_format: "wav"` — the value today's `speechPayload` sends — with no `stream_format` field, the four `X-Speakeasy-*` headers read as today, the body turned into a Blob for the `<audio>` element and the download link as today. The route's default `response_format` is `mp3`, not `pcm`, and a raw `pcm` Blob does not play in an `<audio>` element.

### 9.4 Content-Security-Policy

The CSP at `server.py:369-372` must not change. The page's streaming code must not need `unsafe-eval`, `unsafe-inline`, or a new `connect-src` entry. The existing `connect-src 'self'` already covers `fetch` to the same origin. No MediaSource (not supported for speakeasy's audio formats in Chromium), no new dependency.

## 10. Security requirements (SEC-5 to SEC-9)

Each with the test that fails when it is broken. Source: brief §6.

| # | Requirement | Test |
|---|-------------|------|
| SEC-5 | A stream carries only audio and the fixed JSON of its events: no exception text, no path, no user text beyond the bounded echoes. | `test_stream_leaks_no_exception_text` — a fake engine that raises `RuntimeError("secret /path/to/weights")` after two chunks; the SSE error event body contains only the fixed message, and the log has one error line. |
| SEC-6 | The per-request buffer is bounded by the stated cap (64 MiB). A request over the cap produces a mid-stream error and does not hold the lock. | `test_stream_buffer_bounded` — a fake engine that yields more than 64 MiB. Asserts four things: (1) the client gets the mid-stream error on the event stream (or a truncated body on the raw stream); (2) the lock is free — a subsequent speech request succeeds at once; (3) the log holds exactly one error line; (4) no step waits on a timeout. The test completes in bounded wall time. |
| SEC-7 | An SSE response is marked `Cache-Control: no-store` and is authenticated. | `test_sse_stream_cache_control` — asserts `Cache-Control: no-store` header. `test_sse_stream_401_without_key` — a request without a bearer token to an API-key-protected server returns 401 before any event. |
| SEC-8 | The diff of every branch touches none of `pyproject.toml`, `uv.lock`, `Dockerfile.cuda`, `Dockerfile.rocm`. | Not a test; the reviews check it and so does the orchestrator. |
| SEC-9 | The page adds no external URL, no inline handler, and no CSP change. | The existing `test_csp_unchanged` (from `test_ui.py`) stays green. `test_page_no_external_urls` — greps `index.html` and `app.js` for `http:` / `https:` URLs; only the connect-src-covered origin is allowed. |

## 11. Docs

### 11.1 README

A new section after the existing VoxCPM2 section in `README.md`. It includes:

- A short description of the `stream_format` parameter.
- The deliberate difference from OpenAI: OpenAI's default is `audio` (raw chunked streaming); Speakeasy's absent `stream_format` means today's non-streaming response.
- A table of supported formats per engine (from §5.2 above).
- The `X-Speakeasy-Sample-Rate` header: present on both streaming forms, the engine's output rate in hertz.
- On a raw chunked stream, only `X-Speakeasy-Engine` and `X-Speakeasy-Sample-Rate` are sent as headers; the timing statistics are absent.
- An SSE example with `curl --no-buffer`:
  ```bash
  curl -s -N http://localhost:8080/v1/audio/speech \
    -H "Authorization: Bearer $SPEAKEASY_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"voxcpm2","input":"Hello world","voice":"default","response_format":"pcm","stream_format":"sse"}'
  ```
- A raw chunked audio example with `curl --no-buffer` and `--output -`.
- The `usage` object in `speech.audio.done`: carries speakeasy's four statistics (`engine`, `elapsed`, `audio_duration`, `rtf`) in place of OpenAI's token counts.

### 11.2 Round-trip script

`scripts/tts_roundtrip.py` gains a `--stream` flag that sends `stream_format: "sse"`, reads the SSE events, decodes the base64 audio, verifies at least one `speech.audio.delta` and one `speech.audio.done` event, asserts there is no `data: [DONE]` line, and checks that `usage.audio_duration` matches the decoded audio length to within 0.01 s.

### 11.3 Cross-reference lines (S11)

In `docs/specs/tts-voice-cloning-ui.md` §4.2 (the speech route table), add:

> Streaming delivery is specified in `docs/specs/tts-streaming.md`.

In `docs/specs/tts-voxcpm2-engine.md` §5.1 (the speech route table), add:

> Streaming delivery is specified in `docs/specs/tts-streaming.md`.

Nothing else changes in either file.

## 12. Acceptance scenarios

Every scenario runs without a GPU. Test file: `tests/test_tts_streaming.py`.

### Scenario 1: Non-streaming request unchanged

**Given** a fake VoxCPM2 engine is loaded and ready.
**When** a `POST /v1/audio/speech` with `model: "voxcpm2"` and no `stream_format` field is sent.
**Then** the response is 200 with `Content-Type: audio/mpeg` (default format) and four `X-Speakeasy-*` headers. The body is a complete MP3, byte-for-byte the response the unchanged route gives for the same request (a golden fixture recorded before the change). The engine's `synthesize` was called, not `generate_streaming`.
**Test:** `test_non_streaming_unchanged`

### Scenario 2: SSE stream with VoxCPM2

**Given** a fake VoxCPM2 engine provides three 0.16 s chunks via `generate_streaming`.
**When** a `POST /v1/audio/speech` with `model: "voxcpm2"`, `response_format: "pcm"` and `stream_format: "sse"` is sent.
**Then** the response is 200 with `Content-Type: text/event-stream`, `Cache-Control: no-store`, and `X-Speakeasy-Sample-Rate`. The body contains three `data: {"type":"speech.audio.delta","audio":"..."}` lines, one `data: {"type":"speech.audio.done","usage":{...}}` line, and no `data: [DONE]` line. The `usage` object carries `engine`, `elapsed`, `audio_duration` and `rtf`.
**Test:** `test_sse_stream_voxcpm2`

### Scenario 3: Raw chunked stream

**Given** a fake VoxCPM2 engine provides chunks via `generate_streaming`.
**When** a `POST /v1/audio/speech` with `model: "voxcpm2"`, `response_format: "pcm"` and `stream_format: "audio"` is sent.
**Then** the response is 200 with `Content-Type: audio/pcm` and `Transfer-Encoding: chunked`. `X-Speakeasy-Engine` and `X-Speakeasy-Sample-Rate` are present in the headers. The body is the concatenated raw PCM chunks.
**Test:** `test_raw_stream_pcm`

### Scenario 4: Reverb returns finished take

**Given** a fake Reverb engine is loaded and ready.
**When** a `POST /v1/audio/speech` with `model: "reverb-tts"` and `stream_format: "sse"` is sent.
**Then** the response is 200 with one `speech.audio.delta` carrying the full take, followed by `speech.audio.done`. No 400, no error.
**Test:** `test_reverb_stream_returns_finished_take`

### Scenario 5: Invalid stream_format → 422

**Given** any engine.
**When** a request with `stream_format: "invalid_value"` is sent.
**Then** 422 with Pydantic validation detail referencing `stream_format`.
**Test:** `test_invalid_stream_format_422`

### Scenario 6: Client disconnect mid-stream

**Given** a fake VoxCPM2 engine that takes 0.5 s to generate (yielding chunks every 0.1 s).
**When** the client reads one chunk and closes the connection.
**Then** the server logs one informational line ("client disconnected during stream"). The generating thread finishes. A subsequent speech request succeeds. No 500 is served.
**Test:** `test_disconnect_mid_stream_no_500`

### Scenario 7: Engine failure mid-stream

**Given** a fake VoxCPM2 engine that raises `RuntimeError` after yielding two chunks.
**When** a `POST /v1/audio/speech` with `stream_format: "sse"` is sent.
**Then** the response contains two `speech.audio.delta` events, one error event `{"type":"error","error":{"message":"<fixed message>"}}`, and no `data: [DONE]` line. The connection closes. No exception text appears in any event.
**Test:** `test_engine_failure_mid_stream_clean_close`

### Scenario 8: `/health/ready` during stream

**Given** a fake VoxCPM2 engine that blocks on a `threading.Event`.
**When** a streaming request is sent and the engine is blocked mid-generation.
**Then** a concurrent `GET /health/ready` returns 200 within 2 s.
**Test:** `test_health_responds_during_stream`

### Scenario 9: Idle unload skipped during stream

**Given** a fake VoxCPM2 engine whose `generate_streaming` takes 0.5 s, and an idle timeout of 0.1 s.
**When** a streaming request arrives right after the idle timer fires.
**Then** the unload is skipped (lock not free); the stream completes normally; the engine remains loaded.
**Test:** `test_idle_unload_skipped_during_stream`

### Scenario 10: Stream and non-stream equivalence

**Given** a fake VoxCPM2 engine that records its output.
**When** the same request is sent once with `stream_format: "sse"` and once without.
**Then** `np.array_equal(streamed_audio, non_streamed_audio)` is true.
**Test:** `test_stream_and_non_stream_same_take`

### Scenario 11: CSP unchanged

**Given** the server is running.
**When** `GET /` is requested.
**Then** the `Content-Security-Policy` header matches the pinned string from `test_ui.py:20-23`.
**Test:** `test_csp_unchanged` (existing test in `tests/test_ui.py`)

### Scenario 12: Buffer over-cap produces mid-stream error (S13)

**Given** a fake VoxCPM2 engine that yields more than 64 MiB of data.
**When** a `POST /v1/audio/speech` with `stream_format: "sse"` is sent.
**Then** the stream ends with an error event; the lock is free (a subsequent speech request succeeds at once); the log holds one error line; the test completes in bounded wall time.
**Test:** `test_buffer_over_cap_mid_stream_error`

### Scenario 13: Slow client — buffer drains, lock free (S12, S13)

**Given** a fake VoxCPM2 engine that generates at normal speed.
**When** the client reads chunks slowly (slower than generation).
**Then** the engine runs to completion; the lock is released; the buffer drains at the client's pace. No stall. A subsequent speech request succeeds at once.
**Test:** `test_slow_client_no_stall`

### Scenario 14: Page streams and plays audio (S9, S14)

**Given** a fake VoxCPM2 engine providing three `pcm` chunks via SSE.
**When** the page's Speak handler is triggered.
**Then** an `AudioContext` is created at the rate from `X-Speakeasy-Sample-Rate`; each chunk is scheduled on the timeline as it arrives; at `speech.audio.done`, a WAV Blob is set on the `<audio>` element and the download link shows; the four statistics from `usage` appear in `speak-meta`; the CSP header is unchanged.
**Test:** `test_page_streams_pcm_web_audio` — the orchestrator verifies in a real browser at landing; the existing UI tests stay green. Other browsers stay `ASSUMED`.

### Scenario 15: Page fallback where ReadableStream absent

**Given** a browser without `ReadableStream` support.
**When** the page's Speak handler is triggered.
**Then** the page sends no `stream_format` field; the request follows today's non-streaming path exactly. The `<audio>` element and download link behave as today.
**Test:** `test_page_fallback_no_readable_stream` — the orchestrator verifies in a real browser at landing.

## 13. Slices

In order; each merged by its coder against the tip of the completed slice before it.

| Slice | Branch | Files | Tests |
|-------|--------|-------|-------|
| c2a — engines | `feat/tts-stream-engine` | `src/speakeasy/engines.py` (add `generate_streaming` and `synthesize_streaming` to `VoxCPM2Engine`; add `synthesize_streaming` to `ReverbEngine` that wraps `synthesize`), `tests/test_tts_streaming.py` (engine-level unit tests of the new streaming methods: `test_voxcpm2_generate_streaming_chunk_shape`, `test_reverb_synthesize_streaming_single_chunk`) | `pytest tests/test_tts_streaming.py -q` |
| c2b — route | `feat/tts-stream-route` | `src/speakeasy/server.py` (add `stream_format` to `SpeechRequest`, streaming response builders, the probe for S2, SSE event encoding, the generating-thread/handler lifecycle per S12/S13, the bounded queue with non-blocking cap), `tests/test_tts_streaming.py` (route tests: scenarios 1–10 and 12–13, plus `test_streaming_format_probe` (S2), `test_sse_stream_final_event_carries_stats` and `test_sse_stream_headers` (S5, S15), `test_raw_stream_headers` (S5, S15), `test_engine_runs_to_completion_when_client_stalls` (S6), `test_lock_held_by_generating_thread` (S12), `test_two_concurrent_streams_serialised` and `test_clone_deleted_mid_stream_no_500` (§6.3), plus SEC-5 `test_stream_leaks_no_exception_text`, SEC-6 `test_stream_buffer_bounded`, SEC-7 tests `test_sse_stream_cache_control` and `test_sse_stream_401_without_key`) | `pytest tests/test_tts_streaming.py -q` + full suite |
| c2c — page | `feat/tts-stream-ui` | `src/speakeasy/static/app.js` (Web Audio streaming path per S14, WAV Blob assembly from pcm, fallback, elapsed display, usage statistics), `tests/test_ui.py` (SEC-9: `test_csp_unchanged` stays green; `test_page_no_external_urls` stays green) | `pytest tests/test_ui.py -q` — every existing test green |
| c2d — docs | `docs/tts-streaming` | `README.md` (streaming section + examples, with `X-Speakeasy-Sample-Rate` and the deliberate OpenAI difference), `scripts/tts_roundtrip.py` (streaming path with `--stream`), `docs/specs/tts-voice-cloning-ui.md` (one cross-ref line), `docs/specs/tts-voxcpm2-engine.md` (one cross-ref line) | `pytest tests/ -q` (everything green), manual README review |

## 14. Risks and open questions

Carried from the brief §10. Each with who settles it and the planner's recommendation.

| # | Risk/Question | Settled by | Recommendation |
|---|---------------|------------|----------------|
| 1 | **OpenAI's streaming interface unverified at spec time (F20).** | Planner | **SETTLED.** The `stream_format` parameter and SSE event schema are confirmed in `openai-python` types, the upstream OpenAPI spec, and the `async-openai` Rust crate (§4, O1-O3). `data: [DONE]` and keepalive comments are confirmed absent from the speech stream by the same sources (S16). |
| 2 | **Whether `wav`, `mp3` or `flac` can be encoded incrementally.** | Coder c2b, by the probe in S2 | `pcm` streams always. For the other three, the probe decides. Reviewer evidence (F7) shows all three fail the seek-raising sink; only `pcm` survived. Recommend the probe run first in c2b, and the spec table updated with its result before the route is built. The page already uses `pcm` (S14), so a pcm-only outcome needs no second revision. |
| 3 | **The browser mechanism for play-as-it-arrives.** | Planner + orchestrator at landing | **SETTLED for Chromium** by S14: `AudioContext` with scheduled `AudioBuffer`s from base64-decoded `pcm`, WAV Blob assembly at `done`. No MediaSource (not supported by Chromium), no dependency, no CSP change. Proven at landing. Other browsers stay `ASSUMED`. |
| 4 | **The 15 chars/s floor for the buffer bound (A3).** | Orchestrator at landing | **ASSUMED.** It is a conservative floor below both engines' measured rates at speed 1.0. Settled by measuring the duration of a 4096-char input on both engines at landing. If either is slower, recalculate the bound. |
