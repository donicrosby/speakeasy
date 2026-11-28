# Speakeasy: transcription has no default model

> Spec v1.2. Validated against `main` `301cf43`. Changes the contract of `POST /v1/audio/transcriptions`. Does not touch TTS routes.
> v1.2: the transcription times in section 13 are now measured figures, and references to the planning brief are removed so the spec stands alone. No ruling changed.

## 1. Goals and non-goals

### Goals

**Transcription has no default model** (owner decision D2). The `model` field on `POST /v1/audio/transcriptions` is required, must be a listed transcription model id, and no request is ever routed to an engine it did not name. The same rule the owner gave for speech in the previous lane.

### Non-goals

Changing what an engine transcribes or its accuracy. New engines or models. New dependencies. Any change to a TTS route or a TTS spec. Changing `response_format` or `temperature` semantics. A model selector in the page. Making the page choose differently than ruling T7.

## 2. Owner's decision (D2)

His message of 2026-11-27, verbatim:

> Fix the README, yes no default model, yes to streaming endpoint, 10 step default is probably better.

Row D2 — "yes no default model" — answers the question *"Does 'no default model' extend to transcription?"* It does. Transcription gets the rule speech already has. This is settled; the spec restates it and does not reopen it.

## 3. Rulings T1 to T9 — restated as requirements

Each ruling carries the check that proves it. No ruling is open.

### T1. Model is required (422)

A missing or empty `model` value on `POST /v1/audio/transcriptions` is a 422. The body is `{"detail": [{...}]}` — FastAPI's standard validation error for a required `Form(min_length=1)` field. A request body that lacks the `model` form part and a body that sets `model` to the empty string both receive 422. **Check:** a request with no `model` part and a request with `model=` both answer 422 in `tests/test_server.py`.

### T2. Listed ids only (404)

A `model` value that is not a listed transcription model id is a 404. The body shape is the speech route's `_unknown_model_404` (server.py:446-460 VERIFIED): `{"error": {"message": "unknown model '<echo>'. Known: <ids>"}}`, where `<ids>` names the listed transcription ids separated by `, `. The echoed value is capped at 64 characters followed by `...` when truncated. The check runs **before any engine state is read, before any lock, and before the upload is written to a temporary file.** **Check:** a typo'd value on each engine availability → 404 with the exact known-id list, in `tests/test_server.py`.

### T3. What is listed

`GET /v1/models` returns these transcription ids, listed by configuration, never by load state:

- `whisper-1` — always listed. This is OpenAI's name; it selects Whisper.
- The configured `SPEAKEASY_WHISPER_MODEL` when it differs from `whisper-1` (default: `large-v3-turbo`). Either `large-v3-turbo` or `whisper-1` selects Whisper.
- `moonshine-base` — listed when `SPEAKEASY_MOONSHINE_ENABLED` is true (default: true).

TTS engine ids (`reverb-tts`, `voxcpm2`) are still returned, unchanged and in their present order (after transcription ids). The exact id list a default deployment returns after the change, in order: `["whisper-1", "large-v3-turbo", "moonshine-base", "reverb-tts", "voxcpm2"]`. **Check:** assert the json at key `data[*].id` equals that list in `tests/test_server.py`.

### T4. No silent redirect (400)

Three former silent-redirect paths become explicit rejections:

- **(a)** `fast=true` is an explicit request for Moonshine: with `model=moonshine-base` it is redundant and fine; with a Whisper id (`whisper-1` or the configured name) it is a **400** with the fixed message `"fast=true requires model=moonshine-base"`, never a quiet override.
- **(b)** `model=moonshine-base` with `language` set to anything but `en` → **400**, fixed message: `"moonshine supports only English (language unset or 'en')"`.
- **(c)** A Whisper id with `language` unset keeps auto-detecting (unchanged).

A `model` value that is neither a listed transcription id nor a case covered by (a) or (b) is a 404 (T2). **Check:** T4(a) and T4(b) each get their own test in `tests/test_server.py`, asserting status 400 and the exact message text.

### T5. Duration auto-route goes, with what served it

The only trigger of the duration auto-route at server.py:1102-1106 (`not model and duration is not None and duration ≤ settings.fast_path_max_seconds`) was the empty `model`. The `probe_duration` call at server.py:1097 is removed from the transcription route because the engines report duration themselves (both result objects carry `duration`: engines.py:97 and engines.py:144). `probe_duration` (`engines.py:321`) has no other caller at `main` `301cf43` — its only occurrences are the definition, the import at server.py:52 and the call at server.py:1097 (F11) — so the import is also removed and the definition is deleted. `fast_path_max_seconds` is removed from `config.py:22`, from `README.md` (the routing text at line 296 and the env-table row at line 363), and from `README.docker.md` (the env-table row at line 146). **Check:** the setting is absent from the config test's field-enumeration assertion in `tests/test_config.py`.

### T6. Availability mirrors the TTS rule

An engine that is **configured but failed to load** stays listed and answers **503**. One that is **not configured** is unlisted and answers **404** (T2).

Today a Moonshine load failure at server.py:138-141 sets `moonshine = None` with no reason string (VERIFIED). To record a reason, a module-level `moonshine_error: str | None = None` is set to `f"moonshine failed to load: {e}"` on failure. A request naming `moonshine-base` when Moonshine is configured but failed answers 503 with `"error": {"message": "<reason>"}`. A request naming Whisper when Whisper failed does the same with its existing reason string (server.py:84 VERIFIED). **A request naming Moonshine must not fail because Whisper failed, and the reverse. Check:** one test per engine, in `tests/test_server.py`.

### T7. The page names its model and keeps today's split

`runTranscription` (app.js:303-323) sends `moonshine-base` when the recording is ≤10 s and `whisper-1` otherwise. Ten seconds is a page constant. No selector. The page already knows the recording's duration before sending (app.js:331 VERIFIED: `wav.duration` is logged). For a 3 s recording it sends `model=moonshine-base`; for a 20 s recording it sends `model=whisper-1`. **Check:** pinned in `tests/test_ui.py` as source-text assertions on the `runTranscription` function body and on its microphone call site in `init`, which must pass `captured.duration`.

### T8. Docs and scripts

The README's transcription table (§API row `model`), its routing section (§Engine routing) and its examples state the new contract. A short migration note names what breaks and the fix for each: a request with no `model` → add it; a name that is not listed → use a listed id; `fast=true` with a Whisper model → send `moonshine-base` instead; `moonshine-base` with another language → unset `language`. `scripts/tts_roundtrip.py` sends a listed id (the call at line 401 adds `model=whisper-1`). **Check:** the README diff and the round-trip script's transcription call visible in the commit.

### T9. Spec of record

A new file, `docs/specs/transcription-no-default-model.md` (this file). The two TTS specs (`docs/specs/tts-voice-cloning-ui.md`, `docs/specs/tts-voxcpm2-engine.md`) are not edited by this workstream.

## 4. Verified facts

Each fact is marked `VERIFIED` with its source at `main` `301cf43`.

- **F1** `VERIFIED` — `main` is `301cf43`; 310 tests pass in both environments (tested in a fresh worktree).
- **F3** `VERIFIED` — `POST /v1/audio/transcriptions` signature: server.py:1060-1074. Takes `file`, `model: str = Form(default="")`, `language`, `response_format`, `temperature`, and query flags `fast`, `vad`, `word_timestamps`.
- **F4** `VERIFIED` — Routing at server.py:1095-1107. Moonshine used when eligible (`language` unset or `en`) and any of: `fast` is true; `model == "moonshine-base"`; or `model` is empty and duration ≤ `settings.fast_path_max_seconds`. Otherwise Whisper.
- **F5** `VERIFIED` — Two silent redirects from F4: `fast=true` overrides a `model` that names Whisper (1099-1100); `model=moonshine-base` with non-`en` language quietly runs Whisper (1098-1101). A third: empty `model` with a short clip → Moonshine (1102-1106); any other `model` value → Whisper (1107).
- **F6** `VERIFIED` — `GET /v1/models` at server.py:410-440: returns `_model_ids()`: configured `whisper_model` while Whisper loaded (423-424), `"moonshine-base"` while Moonshine loaded (425-426), then `_listed_tts_engines()` (432-440) listing by configuration. Configured name is `"large-v3-turbo"` (config.py:15); `whisper-1` not listed.
- **F7** `VERIFIED` — Startup at server.py:119-141. Whisper load failure: `startup_error = "whisper failed to load: …"` (127), `whisper` stays `None`, route answers 503 (1075-1078). Moonshine load failure: `moonshine = None` (141), no reason recorded.
- **F8** `VERIFIED` — Speech route's 404 at server.py:446-460: `_unknown_model_404` echoes at most 64 characters + `...`, lists known ids from `_listed_tts_engines()`.
- **F9** `VERIFIED` — Page listen: `runTranscription` (app.js:303-323) sends only the file (311), depends on empty default. `wireListen` knows recording duration (app.js:331).
- **F10** `VERIFIED` — README documents old contract: `model` row at line 49 (default `""`, "empty string auto-routes"); Engine routing at 286-299 (auto-route at 295). `scripts/tts_roundtrip.py` posts to the route near line 401; it currently sends no `model` field.
- **F11** ASSUMED — `probe_duration` at engines.py:321 has no caller other than server.py:1097 (grep at main 301cf43). Confirmed by the coder before removing the import.

## 5. HTTP contract: `POST /v1/audio/transcriptions`

### Request fields

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `file` | file | *(required)* | — |
| `model` | string | *(required)* | `min_length=1`; must be a listed transcription id (T2) |
| `language` | string | `None` | ISO 639-1 code; only `en` is valid with `moonshine-base` (T4b) |
| `response_format` | string | `json` | One of `json`, `verbose_json`, `text`, `srt`, `vtt` |
| `temperature` | float | `0.0` | Accepted for API compatibility |
| `fast` | bool (query) | `false` | Only valid with `model=moonshine-base` (T4a) |
| `vad` | bool (query) | `true` | Whisper only |
| `word_timestamps` | bool (query) | `false` | Whisper only |

### Response headers (unchanged)

- `X-Speakeasy-Engine` — `whisper` or `moonshine`
- `X-Speakeasy-Elapsed` — transcription time in seconds
- `X-Speakeasy-Duration` — audio duration in seconds
- `X-Speakeasy-RTF` — real-time factor

### Error bodies

**422 — Missing or empty `model` (T1):**
```json
{"detail": [{"type": "missing", "loc": ["body", "model"], "msg": "Field required", "input": null}]}
```
When `model` is present but empty, the same body comes back (measured): on this stack an empty form value reports `type: "missing"` exactly like an absent one.

**404 — Unlisted model (T2):**
```json
{"error": {"message": "unknown model 'typo'd-id'. Known: whisper-1, large-v3-turbo, moonshine-base"}}
```
The known-id list names only transcription ids; TTS ids are not repeated. The echoed value is at most 64 characters with `...` when truncated.

**400 — Fast with wrong model (T4a):**
```json
{"error": {"message": "fast=true requires model=moonshine-base"}}
```

**400 — Moonshine with non-English (T4b):**
```json
{"error": {"message": "moonshine supports only English (language unset or 'en')"}}
```

**400 — Bad `response_format`:**
```json
{"error": {"message": "Unsupported response_format 'xml'. Supported: json, verbose_json, text, srt, vtt"}}
```

**503 — Engine configured but failed:**
```json
{"error": {"message": "<startup_error or moonshine_error>"}}
```

### `GET /v1/models` change (T3)

Before the change (current): `["large-v3-turbo", "moonshine-base", "reverb-tts", "voxcpm2"]` (when both STT engines loaded and both TTS engines enabled; TTS ids listed by configuration, STT ids by load state).

After the change: `["whisper-1", "large-v3-turbo", "moonshine-base", "reverb-tts", "voxcpm2"]` — transcription ids listed by configuration (always), TTS ids listed by configuration (unchanged). When `SPEAKEASY_MOONSHINE_ENABLED=false`, `moonshine-base` is absent. When `SPEAKEASY_WHISPER_MODEL=whisper-1`, the list is `["whisper-1", "moonshine-base", "reverb-tts", "voxcpm2"]` — `moonshine-base` stays listed while `SPEAKEASY_MOONSHINE_ENABLED` is true.

## 6. State table

Each listed transcription id against each engine availability state, against each request kind.

**Ids**: `whisper-1` / configured name (both select Whisper; the listing check accepts either), `moonshine-base`.

**Engine states:**
- **Whisper loaded:** `whisper is not None`, `startup_error is None`
- **Whisper failed:** `whisper is None`, `startup_error` is set
- **Moonshine enabled + loaded:** `settings.moonshine_enabled is True`, `moonshine is not None`
- **Moonshine enabled + failed:** `settings.moonshine_enabled is True`, `moonshine is None`, `moonshine_error` is set
- **Moonshine disabled:** `settings.moonshine_enabled is False`

| Request | Whisper loaded, Moonshine loaded | Whisper loaded, Moonshine failed | Whisper loaded, Moonshine disabled | Whisper failed, Moonshine loaded | Whisper failed, Moonshine failed | Whisper failed, Moonshine disabled |
|---------|----------------------------------|----------------------------------|------------------------------------|----------------------------------|----------------------------------|------------------------------------|
| `model=whisper-1` | **200** — Whisper runs; temp file written | **200** — Whisper runs | **200** — Whisper runs | **503** — `startup_error`; no temp file, no lock | **503** — `startup_error` | **503** — `startup_error` |
| `model=large-v3-turbo` | **200** — Whisper runs | **200** — Whisper runs | **200** — Whisper runs | **503** — `startup_error` | **503** — `startup_error` | **503** — `startup_error` |
| `model=moonshine-base` | **200** — Moonshine runs; temp file written | **503** — `moonshine_error`; no temp file | **404** — unlisted (T2); no temp file | **200** — Moonshine runs | **503** — `moonshine_error` | **404** — unlisted (T2) |
| missing/empty | **422** (T1) — no temp file | **422** (T1) | **422** (T1) | **422** (T1) | **422** (T1) | **422** (T1) |
| typo'd value | **404** (T2) — no temp file | **404** (T2) | **404** (T2) | **404** (T2) | **404** (T2) | **404** (T2) |
| `fast=true` + whisper id | **400** (T4a) — no temp file | **400** (T4a) | **400** (T4a) | **400** (T4a) | **400** (T4a) | **400** (T4a) |
| `moonshine-base` + `language=de` | **400** (T4b) — no temp file | **400** (T4b) | n/a (already 404) | **400** (T4b) | **400** (T4b) | n/a (already 404) |

**Side effects**: A temp file is written only when an engine runs (200 rows). The transcription route has no engine lock; the engine's own internal synchronisation is unchanged. The 422, 404 and 400 rows read no engine state at all — the listing they are checked against is configuration only (T3). The 503 rows read only the named engine's failure state (its `None` flag and reason string), never the other engine's. No 422, 404, 400 or 503 row writes a file or takes a lock.

**Before/after — the blanket 503 when Whisper is missing:**
- **Before** (server.py:1075-1078): every request, whatever the `model`, answers 503 when `whisper is None`.
- **After** (T6): only requests naming a Whisper id answer 503; requests naming Moonshine succeed if Moonshine is loaded, or answer their own 503/404.

## 7. No-silent-redirect census

Every place the route or a helper it calls picks an engine from anything other than the named id, at `main` `301cf43`.

| File:Line | What it does | Disposition |
|-----------|-------------|-------------|
| server.py:1099-1100 | `fast=true` forces Moonshine regardless of `model` value | → T4(a): 400 when `model` names Whisper |
| server.py:1101 | `model == "moonshine-base"` selects Moonshine | → kept as explicit selection |
| server.py:1102-1106 | Empty `model` + short clip → Moonshine | → T1: 422 (model required) |
| server.py:1107 (else) | Everything else → Whisper, including typos and `whisper-1` | → T2: 404 for unlisted values; `whisper-1` explicitly routed to Whisper |
| server.py:1098 | `language not in (None, "en")` disqualifies Moonshine; formerly fell through to Whisper | → T4(b): 400 when `model` explicitly names Moonshine |

No other path picks an engine implicitly. After the change every 200 request reaches the engine named by its `model` field and no other.

## 8. The page

### What `runTranscription` sends (T7)

- For a recording of **3 s** (≤10 s): sends `FormData` with `file` and `model=moonshine-base`.
- For a recording of **20 s** (>10 s): sends `FormData` with `file` and `model=whisper-1`.

The split threshold is a page constant, `LISTEN_MOONSHINE_MAX_SECONDS = 10`. The duration is read from `wav.duration` before the `FormData` is built (app.js:331 VERIFIED); no second probe is needed. The microphone path passes the duration `recordWav` already returned (`captured.duration`); the file path uses `wav.duration`.

### Error display

When a non-200 response comes back, the page reads the error body and shows it in `listen-error`:
- **422:** `model` field is missing or empty. The body is FastAPI's validation `detail` array, which the page's shared error path does not parse; the failure status is shown.
- **404:** model id is not listed. The page shows the error message.
- **400:** `fast=true` with wrong model, or `moonshine-base` with wrong language. The page shows the fixed message. A 503 shows the named engine's reason string the way the page does today.

The page adds no new UI and no new error parsing for these cases; the server's fixed messages surface through the existing `listen-error` element.

## 9. Security requirements

Each requirement names the test that fails when it is broken.

- **SEC-1.** The transcription 422 and 404 run before any engine state is read, any lock is taken, or any file is written. **Test:** `test_422_404_before_engine` in `tests/test_server.py` — a fake engine that asserts it is never called; after the request, the temp directory contains no new file.
- **SEC-2.** The transcription 404 echoes at most 64 characters of the value. **Test:** `test_404_echo_bound` in `tests/test_server.py` — a 200-character `model` value; the 404 body contains the first 64 characters followed by `...`.
- **SEC-3.** No transcription request reaches an engine it did not name. **Tests:** `test_fast_with_whisper_id_is_400` (formerly a redirect; now 400), `test_moonshine_non_english_is_400` (formerly a redirect; now 400), `test_typo_model_is_404_on_all_states` (typo on every engine availability → 404).
- **SEC-4.** Removing `fast_path_max_seconds` changes no other setting. **Test:** `test_config_field_names` in `tests/test_config.py` asserts the exact set of Settings field names; the removed field is absent and every other field stays named and at its existing default.
- **SEC-8.** The diff touches none of `pyproject.toml`, `uv.lock`, `Dockerfile.cuda`, `Dockerfile.rocm`. **Check:** the orchestrator and reviewers verify.
- **SEC-9.** The page adds no external URL, no inline handler and no content-security-policy change. **Check:** the existing page test that pins the policy header in `tests/test_ui.py` stays green, unchanged.

## 10. Docs

### README changes

**Transcription table (§API):** the `model` row changes from `default ""` + "empty string auto-routes" to `*(required)*` + "a listed transcription id from `GET /v1/models`". The `fast` query-param row adds: "only valid with `model=moonshine-base`; otherwise 400".

**Engine routing section (lines 289-303):** replaced with a statement that there is no routing — the `model` field names the engine. The three rules that remain: `model=whisper-1` or the configured name → Whisper; `model=moonshine-base` → Moonshine; any other value → 404.

**Examples:** each transcription example gains `model` as a required field.

**Migration note:**
- A request with no `model` → add `"model": "whisper-1"` (or whatever id you mean).
- A request with a name that is not listed → use one from `GET /v1/models`.
- `fast=true` with a Whisper model → send `"model": "moonshine-base"` instead.
- `model=moonshine-base` with a language other than `en` → unset `language`.

**Removed setting:** `SPEAKEASY_FAST_PATH_MAX_SECONDS` is removed (routing text and env-table row); a sentence says it is no longer read. The env-table row in `README.docker.md` (line 146) is removed with it — that file's transcription example already sends `model` and needs no change.

### Round-trip script

`scripts/tts_roundtrip.py` (the `_multipart` call at lines 395-398, the POST at 399-406): the form fields gain `"model": "whisper-1"`.

## 11. Acceptance scenarios

Every scenario runs without a GPU. Each names the test file and test function.

### Scenario: missing model → 422
**Given** a running server. **When** `POST /v1/audio/transcriptions` with a file but no `model` field. **Then** response is 422. **Test:** `test_missing_model_422` (`tests/test_server.py`).

### Scenario: empty model → 422
**Given** a running server. **When** `POST /v1/audio/transcriptions` with `model=""`. **Then** response is 422. **Test:** `test_empty_model_422` (`tests/test_server.py`).

### Scenario: whisper-1 → Whisper
**Given** Whisper and Moonshine both loaded. **When** `POST /v1/audio/transcriptions` with `model=whisper-1`. **Then** response is 200, `X-Speakeasy-Engine: whisper`. **Test:** `test_whisper_1_routes_to_whisper` (`tests/test_server.py`).

### Scenario: configured name → Whisper
**Given** Whisper loaded. **When** `POST /v1/audio/transcriptions` with `model=large-v3-turbo`. **Then** response is 200, `X-Speakeasy-Engine: whisper`. **Test:** `test_configured_name_routes_to_whisper` (`tests/test_server.py`).

### Scenario: moonshine-base → Moonshine
**Given** Moonshine loaded. **When** `POST /v1/audio/transcriptions` with `model=moonshine-base`. **Then** response is 200, `X-Speakeasy-Engine: moonshine`. **Test:** `test_moonshine_name_routes_to_moonshine` (`tests/test_server.py`).

### Scenario: unlisted model → 404
**Given** both engines loaded. **When** `POST /v1/audio/transcriptions` with `model=whisper-2`. **Then** response is 404 with the known-id list naming `whisper-1, large-v3-turbo, moonshine-base`. **Test:** `test_unlisted_model_404` (`tests/test_server.py`).

### Scenario: 404 echo bound → ≤64 chars + ellipsis
**Given** both engines loaded. **When** `POST /v1/audio/transcriptions` with a 200-character `model`. **Then** response is 404; body contains the first 64 characters followed by `...`. **Test:** `test_404_echo_bound` (`tests/test_server.py`).

### Scenario: fast=true with whisper id → 400
**Given** both engines loaded. **When** `POST /v1/audio/transcriptions?fast=true` with `model=whisper-1`. **Then** response is 400 with message `fast=true requires model=moonshine-base`. **Test:** `test_fast_with_whisper_id_is_400` (`tests/test_server.py`).

### Scenario: moonshine-base with non-English → 400
**Given** Moonshine loaded. **When** `POST /v1/audio/transcriptions` with `model=moonshine-base` and `language=de`. **Then** response is 400 with message `moonshine supports only English (language unset or 'en')`. **Test:** `test_moonshine_non_english_is_400` (`tests/test_server.py`).

### Scenario: 422/404 before any engine state, lock or temp file
**Given** a fake environment where both engines are patched to assert they are never called. **When** a 422 (missing model) and a 404 (unlisted model) request are sent. **Then** neither engine's `transcribe` is called, and the temp directory contains no new files. **Test:** `test_422_404_before_engine` (`tests/test_server.py`).

### Scenario: Whisper 503 when Whisper failed
**Given** Whisper is `None`, `startup_error = "whisper failed to load: boom"`, Moonshine is loaded. **When** `POST /v1/audio/transcriptions` with `model=whisper-1`. **Then** response is 503 with `"error": {"message": "whisper failed to load: boom"}`. **Test:** `test_whisper_unavailable_503` (`tests/test_server.py`).

### Scenario: Moonshine 503 when Moonshine failed
**Given** Moonshine is `None`, `moonshine_error = "moonshine failed to load: OOM"`. **When** `POST /v1/audio/transcriptions` with `model=moonshine-base`. **Then** response is 503 with `"error": {"message": "moonshine failed to load: OOM"}`. **Test:** `test_moonshine_unavailable_503` (`tests/test_server.py`).

### Scenario: Moonshine succeeds when Whisper failed
**Given** Whisper is `None`, Moonshine is loaded. **When** `POST /v1/audio/transcriptions` with `model=moonshine-base`. **Then** response is 200, `X-Speakeasy-Engine: moonshine`. **Test:** `test_moonshine_still_works_when_whisper_failed` (`tests/test_server.py`).

### Scenario: model listing by configuration
**Given** a default deployment (Whisper loaded, Moonshine loaded, both TTS engines enabled). **When** `GET /v1/models`. **Then** the `data[*].id` list equals `["whisper-1", "large-v3-turbo", "moonshine-base", "reverb-tts", "voxcpm2"]` in that order. **Test:** `test_models_list_has_all_ids` (`tests/test_server.py`).

### Scenario: model listing without Moonshine
**Given** `SPEAKEASY_MOONSHINE_ENABLED=false`. **When** `GET /v1/models`. **Then** `moonshine-base` is absent. **Test:** `test_models_list_without_moonshine` (`tests/test_server.py`).

### Scenario: config field set unchanged except removed key
**Given** the `Settings` class. **When** its field names are enumerated. **Then** `fast_path_max_seconds` is absent; all other fields from `test_config.py` are present with their existing defaults. **Test:** `test_config_field_names` (`tests/test_config.py`).

### Scenario: page sends model for short recording
**Given** the shipped `app.js` source. **When** `runTranscription` builds its `FormData`. **Then** for a ≤10 s recording it includes `model=moonshine-base`. **Test:** `test_listen_sends_moonshine_for_short` (`tests/test_ui.py`).

### Scenario: page sends model for long recording
**Given** the shipped `app.js` source. **When** `runTranscription` builds its `FormData`. **Then** for a >10 s recording it includes `model=whisper-1`. **Test:** `test_listen_sends_whisper_for_long` (`tests/test_ui.py`).

### Scenario: microphone path passes the recording's duration
**Given** the shipped `app.js` source. **When** the Listen panel's microphone callback runs. **Then** it calls `runTranscription(captured.blob, captured.duration)`. **Test:** `test_listen_recording_passes_its_duration` (`tests/test_ui.py`).

## 12. Slices

One coder slice: `feat/transcription-no-default-model`.

**Files:**
- `src/speakeasy/server.py` — transcription route and model listing
- `src/speakeasy/config.py` — removed `fast_path_max_seconds`
- `src/speakeasy/static/app.js` — `runTranscription` sends `model`
- `README.md` — transcription table, routing section, examples, migration note, removed setting
- `README.docker.md` — the removed setting's env-table row
- `scripts/tts_roundtrip.py` — transcription call adds `model=whisper-1`
- `tests/test_server.py` — all server-side acceptance scenarios
- `tests/test_config.py` — field-name assertion
- `tests/test_ui.py` — page source-text assertions

## 13. Risks and open questions

Each was settled while the spec was designed. The owner may overrule any of them.

1. **`whisper-1` and the configured name both listed (T3).** Settled while the spec was designed. The owner may overrule. Not open for the coder.

2. **The page's ten-second split (T7).** The owner may prefer always Whisper. The cost depends on whether the host has a GPU for Whisper. Measured on 2026-11-28, per call, time reported by the server:

| Host | 2.8 s clip, `whisper-1` | 2.8 s clip, `moonshine-base` | 16.9 s clip, `whisper-1` | 16.9 s clip, `moonshine-base` |
|---|---|---|---|---|
| `rocm` image on an AMD gfx1151 iGPU (Whisper on the GPU, Moonshine on the CPU), host shared with other GPU workloads | 1.4 to 2.6 s (median 2.4 s) | 1.1 to 2.2 s (median 1.6 s) | 3.1 to 5.1 s (median 4.0 s) | 3.5 to 5.7 s (median 5.5 s) |
| `cuda` image with no GPU, CPU only, load average 14 to 16 | 10.4 to 11.9 s (median 10.8 s) | 0.48 to 0.54 s (median 0.51 s) | 11.4 to 12.3 s (median 11.6 s) | 1.8 to 2.0 s (median 1.9 s) |

These are figures from one 32-core host, not a benchmark; they will differ elsewhere. The first call after start also pays the model load (about 10 s for Moonshine). On CPU, Whisper takes nearly the same time for both lengths, which is consistent with its encoder always processing a 30-second window. On the shared host, single calls occasionally took about 10 s: four of about thirty Whisper calls on the GPU, in which both of its GPU stages ran six to nine times slower for about thirty seconds and then returned to normal, and one of about twenty-five warm Moonshine calls. The cause was not established. The cost of always Moonshine on short clips: no punctuation. The current split preserves today's behaviour exactly — the server's former default of 10 s is now a page constant. **Settled by ruling T7**; the owner may overrule and prefer always Whisper; the coder builds the split as ruled.

3. **Callers that relied on the empty default.** Unknown outside this repository. The README's migration note is the mitigation. **Settled by the owner** — he knows his callers and issued ruling D2 with this consequence known.
