# Speakeasy streaming speech

> Spec v1.0. Validated against speakeasy `main` `301cf43`, the brief `brief-no-default-and-streaming.md` §5C, and OpenAI's API reference (read 2026-11-27). Contracts: `docs/specs/tts-voice-cloning-ui.md` (Reverb) and `docs/specs/tts-voxcpm2-engine.md` (VoxCPM2) — this spec extends both and changes neither except for one cross-reference line each (S11).

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

| #  | Decision | His words (verbatim) | What they settle |
|----|----------|----------------------|------------------|
| D3 | Streaming endpoint | "yes to streaming endpoint" | It is built. |
| D4 | Diffusion steps | "10 step default is probably better" | Stays 10 (`src/speakeasy/engines.py:287`). No card changes it, and it is not made configurable. |

## 3. Orchestrator's rulings (restated as requirements)

Each ruling from the brief §5C, restated as a requirement with the check that proves it.

| #  | Requirement | Check |
|----|-------------|-------|
| S1 | **The interface is OpenAI's, verified.** The request field is `stream_format` with values `"sse"` and `"audio"`. When absent, behaviour is byte-for-byte unchanged from today. The SSE events are `speech.audio.delta` and `speech.audio.done`, with payloads matching OpenAI's `CreateSpeechResponseStreamEvent` schema: `{"type":"speech.audio.delta","audio":"<base64>"}` per chunk, and `{"type":"speech.audio.done","usage":{...}}` as the final event. A `data: [DONE]` line closes the stream. Source: `openai-python` `src/openai/types/audio/speech_create_params.py` (main branch, read 2026-11-27) — field `stream_format: Literal["sse", "audio"]`; `speech_create_response_stream_event.py` (same repo); the upstream OpenAPI spec `CreateSpeechResponseStreamEvent` discriminator. **VERIFIED.** | A test sends `stream_format: "sse"` and asserts each `data:` line's JSON matches the schema. A test sends `stream_format: "audio"` and asserts chunked transfer with the content type of the response format. A test omits `stream_format` and asserts the response is bit-for-bit unchanged from today. |
| S2 | **Formats stream by evidence, not hope.** `pcm` streams always. For `wav`, `mp3` and `flac`, a probe encodes to a non-seekable `BytesIO` with libsndfile (soundfile 0.14.0), decodes the concatenated bytes with an independent reader, and records the result. A format that cannot be encoded incrementally and decoded correctly is refused for streaming with 422 and a fixed message naming the formats that can. **Escape hatch: none; a format streams only if the probe shows it. AS-SUMED until the probe runs.** | `test_streaming_format_probe` — calls `_encode_chunk` on each format's first 0.16 s and verifies the concatenated bytes decode correctly; formats that fail are absent from the allowed list. |
| S3 | **Engines.** VoxCPM2 streams through `generate_streaming` (F16). Reverb has no streaming entry point (F17): a streaming request to `reverb-tts` returns the finished take as one chunk when it is ready. That is documented behaviour, not an error, and the response is not labelled with it. | `test_reverb_stream_returns_finished_take` — the Reverb path to a streaming request answers 200 with the full audio in one chunk. |
| S4 | **Delivery changes, the take does not.** The audio of a stream is the take the non-streaming route would return for the same request and seed, to within float rounding: same sample count, largest sample difference at most 1e-5 (measured 1.2e-6, F18). Byte identity is promised by neither path. Coder pins it with fake engines; the orchestrator measures it on the real engine at landing. | `test_stream_and_non_stream_same_take` — a fake VoxCPM2 that records its output returns identical samples from both paths (fake data, so byte identity holds). The orchestrator runs `ref/nds/stream_eq.py` on the real engine at landing and asserts `max_diff_over_rms <= 1e-5`. |
| S5 | **Statistics.** On an SSE stream the final `speech.audio.done` event carries the four values of today's headers as JSON. On a raw chunked stream only what is known before the first byte is sent as a header (`X-Speakeasy-Engine`); the other three are absent and the README says so. No HTTP trailers. | `test_sse_stream_final_event_carries_stats` — parses the last event and asserts `engine`, `elapsed`, `audio_duration` and `rtf` fields. `test_raw_stream_headers` — asserts only `X-Speakeasy-Engine` is present, not the other three. |
| S6 | **The engine never waits on the network.** One thread drives the engine and writes chunks to an in-memory buffer; the response reads that buffer as chunks arrive. The buffer is bounded to 32 MB (33,554,432 bytes): derived from 4096 input chars / 15 chars per second worst case = 273 seconds of audio × 48,000 samples/sec × 2 bytes (s16le pcm) ≈ 26.2 MB, rounded up. Generation runs to completion whatever the client does; the engine lock is held for the generation time only. A slow or stalled client never holds the engine. | `test_engine_runs_to_completion_when_client_stalls` — a fake engine that takes 0.2 s; the test reads one chunk, closes the connection, then asserts the engine finished and the lock was released. |
| S7 | **A client that leaves is an ordinary outcome.** A disconnect mid-stream produces no 500 and no traceback at error level (one informational line). The generating thread finishes and releases the lock. The idle timer is bumped exactly as for a completed request. The "clone deleted mid-request is never a 500" contract holds for streams. An engine failure after the first byte ends the stream cleanly: an error event with a fixed message on an SSE stream, a truncated body on a raw one, and one error line in the log. | `test_disconnect_mid_stream_no_500` — the client disconnects; the next healthy request succeeds. `test_engine_failure_mid_stream_clean_close` — a fake engine that raises after two chunks; the SSE stream ends with an error event and a `[DONE]` line. |
| S8 | **The event loop is never blocked.** Handlers follow the existing pattern: plain `def` on the threadpool. A test pins it: `/health/ready` answers while a stream is mid-take behind a fake engine that waits on an event. | `test_health_responds_during_stream` — a fake engine blocks on a `threading.Event`; a concurrent `/health/ready` call returns 200 within 2 s. |
| S9 | **The page plays speech as it arrives** where the engine streams it, and falls back to today's behaviour otherwise. The browser mechanism is `<audio>` with a `MediaSource` extension, or a `Blob` assembled from `fetch` with a `ReadableStream` reader; the planner recommends `fetch` + `ReadableStream` + `new Blob(chunks)` as it works on all engines and adds zero dependencies. It shows the final statistics, keeps the download link working on the complete audio, adds no dependency, and does not relax the content-security policy. Behaviour only a real browser can prove is marked `ASSUMED`. | `test_csp_unchanged` — the existing UI test that pins the CSP header stays green unchanged. The orchestrator verifies play-as-it-arrives in a real browser at landing. |
| S10 | **Unchanged by this workstream:** every non-streaming response, auth, the 64-character echo bound, `speed`, `cfg`, `design` and `style` semantics, the listing rules, the idle unload, and D4. | Every existing test stays green. |
| S11 | **Cross-references.** Each TTS spec's speech section gains at most one line saying that streaming delivery is specified in this spec. Nothing else in those files changes. | `grep` of both spec files shows exactly one new line each. |

## 4. Verified facts

### VERIFIED — OpenAI's streaming speech interface (read 2026-11-27)

- **O1 — Parameter name and values.** `openai-python` `src/openai/types/audio/speech_create_params.py` (main): `stream_format: Literal["sse", "audio"]`. The OpenAI API reference at `https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create` documents the same parameter with the same values. The field is optional; absent means no streaming.

- **O2 — SSE event names and payloads.** The upstream OpenAPI spec defines `CreateSpeechResponseStreamEvent` as a discriminated union of:
  - `SpeechAudioDeltaEvent`: `{"type": "speech.audio.delta", "audio": "<base64-encoded audio chunk>"}`
  - `SpeechAudioDoneEvent`: `{"type": "speech.audio.done", "usage": {"input_tokens": N, "output_tokens": N, "total_tokens": N}}`
  The `async-openai` Rust crate (`stream.rs`, crates.io, read 2026-11-27) confirms the same schema. A `data: [DONE]` line follows the final event. Comment lines (`: keepalive`) may appear between events.

- **O3 — Raw audio stream.** When `stream_format = "audio"`, the response is chunked transfer of raw audio bytes in the requested `response_format`, with `Content-Type` matching the format. The `openai-python` SDK consumes it with `with_streaming_response.create` returning `HttpxBinaryResponseContent` — iterable bytes.

- **O4 — OpenAI also offers `opus` and `aac` formats, and the `instructions` field.** Speakeasy does not implement these; they are non-goals.

### VERIFIED — speakeasy `main` `301cf43`

- **F1 — Speech route.** `server.py:595-605` — `speech()` is a plain `def` (threadpool), checks the model listing first, dispatches to `_speech_voxcpm2` or `_speech_reverb`.

- **F2 — Speech response.** `server.py:752-770` — `_speech_response()` encodes the finished array with `_encode()` and returns one `Response` with four headers: `X-Speakeasy-Engine`, `-Elapsed`, `-Audio-Duration` and `-RTF`.

- **F3 — TTS lock.** `server.py:99` — `tts_lock = threading.Lock()` serialises inference across both engines. `server.py:636` and `714` — both speech paths hold it around the engine call.

- **F4 — VoxCPM2 lifecycle.** `server.py:285-319` — `_ensure_voxcpm2_loading()` starts a single-flight loader. `server.py:329-360` — `_voxcpm2_unload_loop()` consumes the lock every 15 s and unloads when idle. `server.py:714-746` — the speech path re-checks state under the lock.

- **F5 — Idle timer bump.** `server.py:745` — `voxcpm2_last_used = time.monotonic()` under `tts_lock` after a successful take.

- **F6 — Starlette StreamingResponse.** Starlette 1.7.0 `starlette/responses.py`: `StreamingResponse` wraps an async iterator or sync iterator. Its `stream_response` method iterates on `anyio.to_thread.run_sync` for threadpool iteration (configured by `run_in_threadpool` when the content is a sync iterator). The response sends `Transfer-Encoding: chunked`. Source: installed Starlette 1.7.0, `starlette/responses.py:240-280`.

- **F7 — libsndfile and non-seekable streams.** soundfile 0.14.0 `SoundFile.__init__` calls `sf_open_fd` or `sf_open_virtual`. Writing to a non-seekable `BytesIO` is supported for formats that do not require seeking after write: WAV requires a seek to patch the RIFF header with the final data size; MP3 and FLAC are frame-based and may or may not need to seek. The probe in S2 settles it.

- **F8 — CSP.** `server.py:369-372` — `default-src 'self'; media-src 'self' blob:; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'`.

- **F9 — CI.** `.github/workflows/test.yml:22` — `uv sync --all-groups` (no extras). The streaming tests must pass without the `tts` extra installed.

### VERIFIED — `ref/nds/` (2026-11-27)

- **F10 — Stream equivalence.** `ref/nds/stream_eq.json` — streamed take and non-streamed take for the same request and seed: identical sample count; `max_abs_diff` 8.3e-7 (short), 1.2e-6 (medium); `max_diff_over_rms` 6.5e-6 and 8.5e-6; zero samples differ by more than 1e-3. Chunks are 0.16 s; the first arrives in 0.10 to 0.37 s (nine takes, spike 006).

- **F11 — Stream determinism.** `ref/nds/stream_eq.json` — the streamed path reproduced itself with the same seed in both takes (`stream_same_seed_identical: true` on both lines). The non-streamed short take did not (`non_stream_same_seed_identical: false`). Neither path may promise byte identity.

- **F12 — `generate_streaming` signature.** `ref/nds/stream_eq.py:15,36` — `model.generate_streaming(*args, **kwargs) -> Generator[np.ndarray, None, None]` takes the same arguments as `generate`.

### ASSUMED

- **A1 — WAV incremental encoding.** libsndfile on a non-seekable stream must patch the RIFF header size after writing. Whether the WAV is decodable without that patch is determined by the probe (S2). **Settled by:** coder c2b runs the probe; if WAV fails, it is removed from the streaming-allowed list.

- **A2 — Browser play-as-it-arrives.** The `fetch` + `ReadableStream` + `Blob` path works in the owner's browsers. **Settled by:** orchestrator at landing in a real browser.

## 5. HTTP contract

All `/v1/*` endpoints require the bearer token (existing `require_auth`). The non-streaming speech path is unchanged.

### 5.1 `POST /v1/audio/speech` — new request field

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `stream_format` | `"sse"` \| `"audio"` \| absent | absent (no streaming) | Case-sensitive; any other value → 422 (Pydantic `Literal`). `"sse"` is not supported for `reverb-tts` (see §5.2). |

When `stream_format` is absent: the route follows today's code path exactly, returning one `Response` with four headers. Byte-for-byte unchanged.

### 5.2 Streaming response forms

#### Non-streaming (today)

| Status | Content-Type | Headers |
|--------|-------------|---------|
| 200 | per `response_format` | `X-Speakeasy-Engine`, `-Elapsed`, `-Audio-Duration`, `-RTF` |

#### Raw chunked audio (`stream_format: "audio"`)

| Status | Content-Type | Headers |
|--------|-------------|---------|
| 200 | per `response_format` | `Transfer-Encoding: chunked`, `X-Speakeasy-Engine` |

`X-Speakeasy-Elapsed`, `-Audio-Duration` and `-RTF` are absent: they are not known until the stream completes (S5).

#### Server-sent events (`stream_format: "sse"`)

| Status | Content-Type | Headers |
|--------|-------------|---------|
| 200 | `text/event-stream` | `Cache-Control: no-store`, `X-Accel-Buffering: no` |

The body is a sequence of `data:` lines:

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

3. Stream close:
   ```
   data: [DONE]
   ```

Comment lines (`: keepalive`) may appear every 15 s while waiting for the next chunk.

#### Per-format streaming table (S2)

| Format | Streams | Probe result |
|--------|---------|-------------|
| `pcm` | Always | Raw s16le at engine rate — no container, no seeking, streams by construction. |
| `wav` | **ASSUMED until probe runs** | libsndfile writes WAV to non-seekable `BytesIO`. Probe encodes → decodes → checks header. |
| `mp3` | **ASSUMED until probe runs** | libsndfile writes MP3 to non-seekable `BytesIO`. Probe encodes → decodes → checks frames. |
| `flac` | **ASSUMED until probe runs** | libsndfile writes FLAC to non-seekable `BytesIO`. Probe encodes → decodes → checks frames. |

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
| 422 | `stream_format: "sse"` on a format that cannot stream (per S2 probe) | `{"error": {"message": "streaming not supported for format '<fmt>'. Supported: pcm"}}` |
| 503 | `voxcpm2` not loaded/loading/error | Fixed per-state 503 (existing VoxCPM2 contract) |
| 503 | `reverb-tts` not ready | Fixed per-state 503 (existing Reverb contract) |
| 400 | `cfg`/`design`/`style` on Reverb, `speed` != 1.0 on VoxCPM2, etc. | Existing validation errors (VoxCPM2 spec §5.1) |

The `stream_format` check runs after model listing and auth, before any engine state is read.

### 5.4 Errors mid-stream

| Condition | SSE | Raw |
|-----------|-----|-----|
| Engine raises after first byte | One error event `{"type":"error","error":{"message":"<fixed reason>"}}`, then `data: [DONE]`, then close | Connection closed; the body ends truncated |
| Client disconnects | Generating thread finishes, lock released, idle timer bumped | Same |

Mid-stream errors carry only a fixed message — never exception text, paths, or user input (SEC-5).

## 6. Lifecycle and concurrency

### 6.1 Buffer and generating thread

```
Client            Threadpool worker     Engine thread
  |                      |                    |
  |-- POST /v1/speech -->|                    |
  |                      |-- tts_lock.acquire()
  |                      |-- start engine ----->|
  |  <-- 200 headers ----|                    |
  |                      |   (engine yields)  |
  |  <-- chunk 1 --------|<--- write buffer --|
  |  <-- chunk 2 --------|<--- write buffer --|
  |  ...                 |                    |
  |  <-- chunk N ---------|<--- write buffer --|
  |                      |-- tts_lock.release()
  |                      |-- bump idle timer
```

- The route handler (plain `def`, threadpool) acquires `tts_lock`, starts the engine on a separate daemon thread, and returns a generator that reads from a bounded queue. The engine thread writes chunks to the queue; the generator yields them to the response.
- **Queue bound:** 32 MB (33,554,432 bytes). Derived from the 4096-char input cap at 48 kHz s16le with a floor of 15 chars per second of audio. The engine thread blocks on `put()` when the queue is full, so a stalled client pauses generation but the engine never writes unboundedly.
- The engine lock is held for the duration of `generate_streaming` (or `generate` for Reverb), not for the full stream lifetime.

### 6.2 Concurrency guarantees

- The single `tts_lock` serialises all engine inference (S3 of the VoxCPM2 spec). A streaming request holds it during generation exactly like a non-streaming one.
- The idle-unload timer (`_voxcpm2_unload_tick`) acquires `tts_lock` with `blocking=False`; it never unloads under a running request.
- A clone deleted mid-request: the engine already holds a reference to the clone file path at synthesis start. The delete races the read; if the file vanishes, the engine raises `FileNotFoundError`, which the route catches and surfaces as the mid-stream error (S7).

### 6.3 Race table

| Interleaving | Outcome | Test |
|-------------|---------|------|
| Streaming client + concurrent `/health/ready` | `/health/ready` answers 200 within 2 s | `test_health_responds_during_stream` |
| Streaming client disconnects mid-stream + next speech request | Next request succeeds; no 500 | `test_disconnect_mid_stream_no_500` |
| Engine fails after two chunks | SSE: error event + `[DONE]`. Raw: truncated body. Log has one error line. | `test_engine_failure_mid_stream_clean_close` |
| Clone deleted while a stream uses it | Mid-stream error event (SSE) or truncated body (raw). No 500. | `test_clone_deleted_mid_stream_no_500` |
| Idle-unload tick fires during stream | Unload skipped (lock not free). Stream completes normally. | `test_idle_unload_skipped_during_stream` |
| Two concurrent streaming requests | Serialised by `tts_lock`; second waits. Both complete. | `test_two_concurrent_streams_serialised` |

## 7. Equivalence contract (S4)

**Promise:** the audio of a stream is the take the non-streaming route would return for the same request and seed, to within float rounding: same sample count, largest sample difference at most 1e-5 (F10 measured 1.2e-6, which is 8× smaller). Byte identity is promised by neither path (F11).

**Test with fakes:** `test_stream_and_non_stream_same_take` — a fake VoxCPM2 that records its output returns identical samples from both paths. The test asserts `np.array_equal(streamed, non_streamed)`.

**Measure on real engine at landing:** the orchestrator runs `ref/nds/stream_eq.py` with `max_diff_over_rms <= 1e-5` as the pass criterion.

## 8. Statistics (S5)

| Response form | `X-Speakeasy-Engine` | `-Elapsed` | `-Audio-Duration` | `-RTF` |
|--------------|---------------------|------------|-------------------|--------|
| Non-streaming (today) | Header | Header | Header | Header |
| Raw chunked (`stream_format: "audio"`) | Header | Absent | Absent | Absent |
| SSE (`stream_format: "sse"`) | In `speech.audio.done` `usage.engine` | In `speech.audio.done` `usage.elapsed` | In `speech.audio.done` `usage.audio_duration` | In `speech.audio.done` `usage.rtf` |

On a raw chunked stream the three absent values are not known before the first byte and are not sent as trailers (S5). The README documents this.

## 9. The page (S9)

### 9.1 Behaviour per engine and format

| Engine | Streaming | Page behaviour |
|--------|-----------|----------------|
| `voxcpm2` | Yes | Play audio as chunks arrive via `fetch` + `ReadableStream` reader + `MediaSource` or `Blob` assembly. Show elapsed time growing. |
| `reverb-tts` | No (finished take) | Today's behaviour: wait for complete response, set `<audio>.src` to blob URL. |

### 9.2 What the page shows

- **While receiving:** an elapsed counter (seconds since first chunk). The download link is hidden or disabled.
- **At completion:** the final statistics (duration, elapsed, RTF) shown in `speak-meta`. The download link appears on the complete audio.
- **Error:** the existing error display in `speak-error`. No additional UI element.

### 9.3 Fallback

If the browser does not support `ReadableStream` (the `fetch` body reader), the page falls back to today's `response.blob()` path. Detection: `if (!response.body || !response.body.getReader)` → fallback.

### 9.4 Content-Security-Policy

The CSP at `server.py:369-372` must not change. The page's streaming code must not need `unsafe-eval`, `unsafe-inline`, or a new `connect-src` entry. The existing `connect-src 'self'` already covers `fetch` to the same origin.

## 10. Security requirements (SEC-5 to SEC-9)

Each with the test that fails when it is broken. Source: brief §6.

| # | Requirement | Test |
|---|-------------|------|
| SEC-5 | A stream carries only audio and the fixed JSON of its events: no exception text, no path, no user text beyond the bounded echoes. | `test_stream_leaks_no_exception_text` — a fake engine that raises `RuntimeError("secret /path/to/weights")` after two chunks; the SSE error event body contains only the fixed message, and the log has one error line. |
| SEC-6 | The per-request buffer is bounded by the stated cap (32 MB). | `test_stream_buffer_bounded` — a fake engine that yields more than 32 MB; the generation thread raises `BufferError` or the queue's `put` blocks and the test times out cleanly. |
| SEC-7 | An SSE response is marked `Cache-Control: no-store` and is authenticated. | `test_sse_stream_cache_control` — asserts `Cache-Control: no-store` header. `test_sse_stream_401_without_key` — a request without a bearer token to an API-key-protected server returns 401 before any event. |
| SEC-8 | The diff of every branch touches none of `pyproject.toml`, `uv.lock`, `Dockerfile.cuda`, `Dockerfile.rocm`. | Not a test; the reviews check it and so does the orchestrator. |
| SEC-9 | The page adds no external URL, no inline handler, and no CSP change. | The existing `test_csp_unchanged` (from `test_ui.py`) stays green. `test_page_no_external_urls` — greps `index.html` and `app.js` for `http:` / `https:` URLs; only the connect-src-covered origin is allowed. |

## 11. Docs

### 11.1 README

A new section after the existing VoxCPM2 section in `README.md`. It includes:

- A short description of the `stream_format` parameter.
- A table of supported formats per engine (from §5.2 above).
- An SSE example with `curl --no-buffer`:
  ```bash
  curl -s -N http://localhost:8765/v1/audio/speech \
    -H "Authorization: Bearer $SPEAKEASY_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"voxcpm2","input":"Hello world","voice":"default","response_format":"wav","stream_format":"sse"}'
  ```
- A raw chunked audio example with `curl --no-buffer` and `--output -`.
- A note that on raw chunked streams, only `X-Speakeasy-Engine` is sent as a header; the other three statistics are absent.

### 11.2 Round-trip script

`scripts/tts_roundtrip.py` gains a `--stream` flag that sends `stream_format: "sse"`, reads the SSE events, decodes the base64 audio, verifies at least one `speech.audio.delta` and one `speech.audio.done` event, and checks that `usage.audio_duration` matches the decoded audio length to within 0.01 s.

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
**Then** the response is 200 with `Content-Type: audio/mpeg` (default format) and four `X-Speakeasy-*` headers. The body is a complete MP3. The engine's `synthesize` was called, not `generate_streaming`.
**Test:** `test_non_streaming_unchanged`

### Scenario 2: SSE stream with VoxCPM2

**Given** a fake VoxCPM2 engine provides three 0.16 s chunks via `generate_streaming`.
**When** a `POST /v1/audio/speech` with `model: "voxcpm2"` and `stream_format: "sse"` is sent.
**Then** the response is 200 with `Content-Type: text/event-stream` and `Cache-Control: no-store`. The body contains three `data: {"type":"speech.audio.delta","audio":"..."}` lines, one `data: {"type":"speech.audio.done","usage":{...}}` line, and `data: [DONE]`. The `usage` object carries `engine`, `elapsed`, `audio_duration` and `rtf`.
**Test:** `test_sse_stream_voxcpm2`

### Scenario 3: Raw chunked stream

**Given** a fake VoxCPM2 engine provides chunks via `generate_streaming`.
**When** a `POST /v1/audio/speech` with `model: "voxcpm2"`, `response_format: "pcm"` and `stream_format: "audio"` is sent.
**Then** the response is 200 with `Content-Type: audio/pcm` and `Transfer-Encoding: chunked`. Only `X-Speakeasy-Engine` is present in the headers. The body is the concatenated raw PCM chunks.
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
**Then** the response contains two `speech.audio.delta` events, one error event `{"type":"error","error":{"message":"<fixed message>"}}`, and a `data: [DONE]` line. No exception text appears in any event.
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

## 13. Slices

In order; each merged by its coder against the tip of the completed slice before it.

| Slice | Branch | Files | Tests |
|-------|--------|-------|-------|
| c2a — engines | `feat/tts-stream-engine` | `src/speakeasy/engines.py` (add `generate_streaming` and `synthesize_streaming` to `VoxCPM2Engine`; add `synthesize_streaming` to `ReverbEngine` that wraps `synthesize`), `tests/test_tts_streaming.py` (fake engine stream tests: scenarios 1, 2, 4, 10) | `pytest tests/test_tts_streaming.py -q` |
| c2b — route | `feat/tts-stream-route` | `src/speakeasy/server.py` (add `stream_format` to `SpeechRequest`, streaming response builders, the probe for S2, SSE event encoding), `tests/test_tts_streaming.py` (route tests: scenarios 3, 5, 6, 7, 8, 9) | `pytest tests/test_tts_streaming.py -q` + full suite |
| c2c — page | `feat/tts-stream-ui` | `src/speakeasy/static/app.js` (streaming read path, fallback, elapsed display), `tests/test_ui.py` (no new JS tests; existing tests stay green) | `pytest tests/test_ui.py -q` — every existing test green |
| c2d — docs | `docs/tts-streaming` | `README.md` (streaming section + examples), `scripts/tts_roundtrip.py` (streaming path with `--stream`), `docs/specs/tts-voice-cloning-ui.md` (one cross-ref line), `docs/specs/tts-voxcpm2-engine.md` (one cross-ref line) | `pytest tests/ -q` (everything green), manual README review |

## 14. Risks and open questions

Carried from the brief §10. Each with who settles it and the planner's recommendation.

| # | Risk/Question | Settled by | Recommendation |
|---|---------------|------------|----------------|
| 1 | **OpenAI's streaming interface unverified at spec time (F20).** | Planner | **SETTLED.** The `stream_format` parameter and SSE event schema are confirmed in openai-python types and the upstream OpenAPI spec (§4, O1-O3). |
| 2 | **Whether `wav`, `mp3` or `flac` can be encoded incrementally.** | Coder c2b, by the probe in S2 | `pcm` streams always. For the other three, the probe decides. Recommend the probe run first in c2b, and the spec table updated with its result before the route is built. |
| 3 | **The browser mechanism for play-as-it-arrives.** | Planner + orchestrator at landing | Recommend `fetch` with `ReadableStream` reader + `Blob` assembly: zero dependencies, works across engines, keep `connect-src 'self'` unchanged. Proven only by the landing browser run. |
