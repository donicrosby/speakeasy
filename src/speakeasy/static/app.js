"use strict";

// speakeasy static UI — three panels, no framework, no build step.
//
// The API key lives in sessionStorage only: it is read here, sent as an
// Authorization header, and never placed in a URL or a cookie. All dynamic
// text is written with textContent; server data is never rendered as markup.

const $ = (id) => document.getElementById(id);

const KEY_STORAGE = "speakeasy_key";
const key = () => sessionStorage.getItem(KEY_STORAGE) || "";

// Fixed messages. Kept literal so every state is greppable.
const MSG = {
  unauthorized: "Wrong or missing API key — enter it above.",
  loading:
    "TTS is still loading (first start downloads ~1 GB of weights) — check back in a minute.",
  disabled: "TTS is not available on this server.",
  ttsError: "TTS encountered an error and is unavailable — check the server logs.",
  modelsLoading: "The speech models are still loading — check the status line.",
  network: "Could not reach the server — check the connection.",
  mic: "mic needs HTTPS or localhost — upload a file instead",
  working: "working…",
  // The page's own table for the VoxCPM2 engine, chosen from /health/ready.
  voxcpm2Disabled: "VoxCPM2 is disabled — enable it in the server config and restart.",
  voxcpm2Loading:
    "VoxCPM2 is still loading (first start downloads ~4.7 GB of weights) — check back in a minute.",
  voxcpm2Error:
    "VoxCPM2 encountered an error and is unavailable — check the server logs.",
  voxcpm2BadTake:
    "VoxCPM2 could not generate a good take for this input — try shorter text or a different seed.",
};

// The server's two engine ids.
const ENGINE_REVERB = "reverb-tts";
const ENGINE_VOXCPM2 = "voxcpm2";

// While VoxCPM2 loads, readiness is re-read at this interval.
const LOAD_POLL_MS = 5000;

const TRANSCRIPT_REQUIRED = "Transcript";
const TRANSCRIPT_OPTIONAL = "Transcript (optional for VoxCPM2)";

// The server's default preset voice, and the language the page opens on. A
// rebuilt <select> lands on its first option and "de" sorts first, so both are
// set explicitly or an English speaker gets a German voice by accident.
const DEFAULT_VOICE = "en_male_narrator_1";
const DEFAULT_LANGUAGE = "en";

// The most recent recording, shared with the Clone panel.
let lastRecording = null;
let voicesLoaded = false;
// Every voice object from the last listing, before engine filtering.
let voices = [];
// A request made while VoxCPM2 was loading, re-sent once the engine is ready.
let pendingRequest = null;
// The single readiness poll loop; null when no loop is running.
let loadPollTimer = null;

// --- transport ---------------------------------------------------------------

// 503s come from two different loaders: the TTS endpoints (Reverb) and the
// transcription endpoint (Whisper). Only the TTS paths may map a 503 to a TTS
// message — a Whisper-side 503 must never read as "TTS encountered an error".
const TTS_PATHS = ["/v1/audio/speech", "/v1/audio/voices"];

function isTtsPath(path) {
  return TTS_PATHS.some((p) => path === p || path.startsWith(p + "/"));
}

async function errorText(response) {
  try {
    const body = await response.json();
    return (body && body.error && body.error.message) || "";
  } catch (e) {
    return "";
  }
}

async function ttsReason(response) {
  const text = await errorText(response);
  if (text.includes("loading")) return MSG.loading;
  if (text.includes("disabled")) return MSG.disabled;
  return MSG.ttsError;
}

async function sttReason(response) {
  // The transcription endpoint's 503 is Whisper's (failed to load, or still
  // loading), never TTS's: surface its reason, else a neutral line.
  return (await errorText(response)) || MSG.modelsLoading;
}

// The engine's state from readiness. The page's own 503 messages are chosen
// from this value, never from the API's response body.
async function voxcpm2State() {
  try {
    const response = await fetch("/health/ready");
    const body = await response.json();
    return body.voxcpm2 || "";
  } catch (e) {
    return "";
  }
}

function voxcpm2Message(state) {
  if (state === "disabled") return MSG.voxcpm2Disabled;
  if (state === "error") return MSG.voxcpm2Error;
  // A ready engine can still reject a take: that is the bad-take refusal.
  if (state === "ready") return MSG.voxcpm2BadTake;
  return MSG.voxcpm2Loading;
}

async function api(path, options) {
  const opts = options || {};
  const init = { headers: { Authorization: "Bearer " + key() } };
  if (opts.json) {
    init.method = "POST";
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.json);
  }
  if (opts.form) {
    init.method = "POST";
    init.body = opts.form;
  }
  if (opts.method) init.method = opts.method;
  let response;
  try {
    response = await fetch(path, init);
  } catch (e) {
    throw new Error(MSG.network);
  }
  if (response.status === 401) throw new Error(MSG.unauthorized);
  if (response.status === 503) {
    if (opts.engine === ENGINE_VOXCPM2) {
      // The page's own per-state message; the state rides the error so the
      // caller can hold the request for the loading path.
      const state = await voxcpm2State();
      const failure = new Error(voxcpm2Message(state));
      failure.voxcpm2State = state;
      throw failure;
    }
    const reason = isTtsPath(path)
      ? await ttsReason(response)
      : await sttReason(response);
    throw new Error(reason);
  }
  if (!response.ok) {
    let message = response.statusText || "HTTP " + response.status;
    try {
      const body = await response.json();
      if (body && body.error && body.error.message) message = body.error.message;
    } catch (e) {
      // Body was not JSON; keep the status text.
    }
    throw new Error(message);
  }
  return opts.raw ? response : response.json();
}

// --- errors ------------------------------------------------------------------

function showError(el, err) {
  el.textContent = err && err.message ? err.message : String(err);
}

function clearError(el) {
  el.textContent = "";
}

async function guard(el, fn) {
  clearError(el);
  try {
    await fn();
  } catch (err) {
    showError(el, err);
  }
}

// --- microphone: record -> 16 kHz mono 16-bit WAV ----------------------------

function micAvailable() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function recordWav(stopBtn) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const chunks = [];
  const recorder = new MediaRecorder(stream);
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size) chunks.push(event.data);
  });
  const stopped = new Promise((resolve) =>
    recorder.addEventListener("stop", resolve),
  );
  recorder.start();
  await new Promise((resolve) =>
    stopBtn.addEventListener("click", resolve, { once: true }),
  );
  recorder.stop();
  await stopped;
  for (const track of stream.getTracks()) track.stop();
  return decodeToWav(new Blob(chunks, { type: recorder.mimeType }));
}

async function decodeToWav(blob) {
  const bytes = await blob.arrayBuffer();
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  try {
    const source = await ctx.decodeAudioData(bytes.slice(0));
    const frames = Math.ceil(source.duration * 16000) || 1;
    const offline = new OfflineAudioContext(1, frames, 16000);
    const node = offline.createBufferSource();
    node.buffer = source;
    node.connect(offline.destination);
    node.start();
    const rendered = await offline.startRendering();
    const mono = rendered.getChannelData(0);
    return { blob: encodeWav(mono, 16000), duration: mono.length / 16000 };
  } finally {
    ctx.close(); // also on a decode/render failure, or the context leaks
  }
}

function encodeWav(samples, rate) {
  // Canonical 44-byte PCM WAV header: mono, 16-bit, `rate` samples/second.
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM header size
  view.setUint16(20, 1, true); // format = PCM
  view.setUint16(22, 1, true); // channels = mono
  view.setUint32(24, rate, true); // sample rate
  view.setUint32(28, rate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function toWav(blob) {
  try {
    return await decodeToWav(blob);
  } catch (e) {
    // Not decodable by WebAudio: hand the raw bytes to the server, which
    // decodes with PyAV.
    return { blob: blob, duration: 0 };
  }
}

async function recordInto(recBtn, stopBtn, stateEl) {
  recBtn.disabled = true;
  stopBtn.disabled = false;
  stateEl.textContent = "recording…";
  try {
    const captured = await recordWav(stopBtn);
    stateEl.textContent = "captured " + captured.duration.toFixed(1) + " s";
    lastRecording = captured;
    return captured;
  } finally {
    recBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

function wireRecording(panel, onCapture) {
  const recBtn = $(panel + "-record");
  const stopBtn = $(panel + "-stop");
  const stateEl = $(panel + "-state");
  const errEl = $(panel + "-error");
  if (!micAvailable()) {
    const note = document.createElement("span");
    note.className = "note";
    note.textContent = MSG.mic;
    recBtn.replaceWith(note);
    stopBtn.disabled = true;
    return;
  }
  recBtn.addEventListener("click", () => {
    guard(errEl, async () => {
      const captured = await recordInto(recBtn, stopBtn, stateEl);
      if (onCapture) await onCapture(captured);
    });
  });
}

// --- Listen ------------------------------------------------------------------

// Recordings at or under this length use the fast Moonshine model; longer ones
// use Whisper. The server has no default model, so the page names one.
const LISTEN_MOONSHINE_MAX_SECONDS = 10;

async function runTranscription(blob, duration) {
  const errEl = $("listen-error");
  clearError(errEl);
  const stateEl = $("listen-state");
  stateEl.textContent = MSG.working;
  const model =
    duration > 0 && duration <= LISTEN_MOONSHINE_MAX_SECONDS
      ? "moonshine-base"
      : "whisper-1";
  const form = new FormData();
  form.append("file", blob, "audio.wav");
  form.append("model", model);
  try {
    const response = await api("/v1/audio/transcriptions", { form: form, raw: true });
    const engine = response.headers.get("X-Speakeasy-Engine") || "?";
    const duration = response.headers.get("X-Speakeasy-Duration") || "?";
    const elapsed = response.headers.get("X-Speakeasy-Elapsed") || "?";
    const rtf = response.headers.get("X-Speakeasy-RTF") || "?";
    const body = await response.json();
    $("listen-text").value = body.text || "";
    $("listen-meta").textContent =
      "engine " + engine + " · " + duration + " s · " + elapsed + " s · " + rtf + "x real time";
  } finally {
    stateEl.textContent = "idle";
  }
}

function wireListen() {
  const fileInput = $("listen-file");
  fileInput.addEventListener("change", () => {
    if (!fileInput.files || !fileInput.files[0]) return;
    const file = fileInput.files[0];
    guard($("listen-error"), async () => {
      const wav = await toWav(file);
      lastRecording = wav;
      $("listen-state").textContent = "captured " + wav.duration.toFixed(1) + " s";
      await runTranscription(wav.blob, wav.duration);
    });
  });
}

// --- Speak -------------------------------------------------------------------

// The last Speak blob URL, revoked before the next one so the page does not
// leak decoded audio for the lifetime of the tab.
let speakUrl = null;
// The one AudioContext a streaming take plays through; the next Speak replaces
// it so contexts do not accumulate.
let speakContext = null;

function fillLanguages(languages) {
  for (const id of ["speak-language", "clone-language"]) {
    const select = $(id);
    const current = select.value; // read before the rebuild resets it
    select.textContent = "";
    for (const code of languages) {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = code;
      select.appendChild(option);
    }
    if (languages.includes(current)) {
      select.value = current; // a reload must never drop the user's choice
    } else if (languages.includes(DEFAULT_LANGUAGE)) {
      select.value = DEFAULT_LANGUAGE;
    } else {
      select.value = languages[0] || "";
    }
  }
}

function addGroup(select, label, voices) {
  const group = document.createElement("optgroup");
  group.label = label;
  for (const voice of voices) {
    const option = document.createElement("option");
    option.value = voice.id;
    option.textContent = voice.id;
    option.dataset.language = voice.language || "en";
    group.appendChild(option);
  }
  select.appendChild(group);
}

function fillVoices(items) {
  voices = items;
  const engine = $("speak-engine").value;
  const select = $("speak-voice");
  const current = select.value; // read before the rebuild resets it
  select.textContent = "";
  if (engine === ENGINE_VOXCPM2) {
    // VoxCPM2's own voice is synthetic: the model's default, no reference.
    const option = document.createElement("option");
    option.value = "default";
    option.textContent = "default";
    select.appendChild(option);
    addGroup(
      select,
      "Saved",
      items.filter((v) => v.type === "clone" && v.engine === ENGINE_VOXCPM2),
    );
  } else {
    addGroup(select, "Presets", items.filter((v) => v.type === "preset"));
    addGroup(
      select,
      "Saved",
      items.filter((v) => v.type === "clone" && v.engine !== ENGINE_VOXCPM2),
    );
  }
  const ids = Array.from(select.options).map((option) => option.value);
  if (ids.includes(current)) {
    select.value = current; // a reload must never drop the user's voice
  } else if (ids.includes(DEFAULT_VOICE)) {
    select.value = DEFAULT_VOICE;
  } else {
    select.value = ids[0] || "";
  }
  syncSpeakLanguage();
  syncSpeakDesignStyle();
}

function syncSpeakLanguage() {
  if ($("speak-engine").value === ENGINE_VOXCPM2) return;
  const select = $("speak-voice");
  const option = select.selectedOptions[0];
  const language = option && option.dataset.language;
  if (language) $("speak-language").value = language;
}

function applySpeakEngine() {
  const engine = $("speak-engine").value;
  const voxcpm2 = engine === ENGINE_VOXCPM2;
  // VoxCPM2 needs no language tag: hide the select, keep its content.
  $("speak-language-label").hidden = voxcpm2;
  $("speak-voxcpm2-controls").hidden = !voxcpm2;
  // Rebuild the list for the chosen engine, then re-apply the field rules.
  fillVoices(voices);
  syncCfgDisplay();
}

function syncCfgDisplay() {
  $("speak-cfg-value").textContent = Number($("speak-cfg").value).toFixed(1);
}

// An engine the server does not list is never offered: while `/health/ready`
// reports `voxcpm2` as `disabled`, its option is removed from both selectors.
function applyEngineAvailability(state) {
  const disabled = state === "disabled";
  for (const selectId of ["speak-engine", "clone-engine"]) {
    const select = $(selectId);
    const option = Array.from(select.options).find(
      (item) => item.value === ENGINE_VOXCPM2,
    );
    if (!option) continue;
    option.disabled = disabled;
    option.hidden = disabled;
    if (disabled && select.value === ENGINE_VOXCPM2) {
      select.value = ENGINE_REVERB;
    }
  }
  if (disabled) {
    applySpeakEngine();
    applyCloneEngine();
  }
}

function syncSpeakDesignStyle() {
  if ($("speak-engine").value !== ENGINE_VOXCPM2) return;
  const design = $("speak-design");
  const styleText = $("speak-style");
  const isDefault = $("speak-voice").value === "default";
  // Design only creates a voice, so it belongs to the default voice only; and
  // a request never carries both.
  design.disabled = !isDefault || styleText.value.trim() !== "";
  styleText.disabled = isDefault && design.value.trim() !== "";
}

// The speech body, per engine. `cfg`, `design` and `style` are VoxCPM2's
// alone; `speed` and `language` are Reverb's, so neither is sent for VoxCPM2.
function speechPayload(engine, text) {
  const voice = $("speak-voice").value;
  if (engine === ENGINE_VOXCPM2) {
    const payload = {
      model: ENGINE_VOXCPM2,
      input: text,
      voice: voice,
      response_format: "wav",
      cfg: Number($("speak-cfg").value),
    };
    if (voice === "default") {
      const design = $("speak-design").value.trim();
      if (design) payload.design = design;
    }
    const styleText = $("speak-style").value.trim();
    if (styleText) payload.style = styleText;
    return payload;
  }
  return {
    model: ENGINE_REVERB,
    input: text,
    voice: voice,
    language: $("speak-language").value || null,
    response_format: "wav",
  };
}

// --- Speak: stream the take as events, or ask for it whole -------------------

// The speech stream's two events, and the header carrying the take's rate.
const SSE_DELTA = "speech.audio.delta";
const SSE_DONE = "speech.audio.done";
const SAMPLE_RATE_HEADER = "X-Speakeasy-Sample-Rate";

// The browser can play the take as it arrives only if it can read a fetch body
// as it arrives, decode text, and run an AudioContext. Without all three the
// page asks for the whole take, as it always has.
function streamingAvailable() {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  return (
    typeof ReadableStream !== "undefined" &&
    typeof TextDecoder !== "undefined" &&
    typeof AudioCtx === "function"
  );
}

// Today's body, plus the two fields that ask for the take as pcm events.
function streamingPayload(payload) {
  return Object.assign({}, payload, {
    response_format: "pcm",
    stream_format: "sse",
  });
}

// One take's playback. The take starts in buffering: every delta is decoded and
// kept and nothing is scheduled, until 1.5 s of audio is held. Then the page
// decides once whether the engine keeps up with playback — the audio received
// after the first chunk, over the seconds since the first chunk arrived, at
// least 1. If it keeps up, everything held is scheduled and the take is live; if
// not, the take stays in buffering until the stream ends. A live delta that
// arrives after its start has passed switches the take to refill. Whatever is
// unscheduled at speech.audio.done plays as one block. A take is thus played
// whole, or live without gaps, or live until one pause and then whole again; it
// never stutters. Every sample is kept so the finished take can be written as a
// WAV Blob for the download link.
const LIVE_LEAD_SECONDS = 1.5;

function newStreamPlayer(rate) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (speakContext) speakContext.close();
  speakContext = new AudioCtx({ sampleRate: rate });
  // Resuming needs the click that started the request; a rejected promise only
  // means the take stays silent, so it is ignored.
  if (speakContext.resume) speakContext.resume();
  return {
    ctx: speakContext,
    rate: rate,
    samples: [],   // every chunk, in order, for the WAV Blob
    pending: [],   // chunks decoded but not yet scheduled
    end: 0,        // when the last scheduled chunk ends, on the ctx clock
    state: "buffering",
    tested: false, // the keep-up test is made once
    firstAt: 0,    // when the first chunk arrived, on the wall clock
  };
}

// s16le bytes, base64, as the float samples Web Audio and the WAV writer take.
function decodePcm(base64) {
  const bytes = atob(base64);
  const floats = new Float32Array(bytes.length / 2);
  for (let i = 0; i < floats.length; i++) {
    const low = bytes.charCodeAt(i * 2);
    const high = bytes.charCodeAt(i * 2 + 1);
    const sample = (high << 8) | low;
    floats[i] = (sample >= 0x8000 ? sample - 0x10000 : sample) / 32768.0;
  }
  return floats;
}

// Put one chunk on the timeline at `at`, and remember where it ends.
function scheduleChunk(player, floats, at) {
  const buffer = player.ctx.createBuffer(1, floats.length, player.rate);
  buffer.copyToChannel(floats, 0);
  const source = player.ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(player.ctx.destination);
  source.start(at);
  player.end = at + buffer.duration;
}

// Seconds of audio in `chunks`, from index `from`.
function audioSeconds(player, chunks, from) {
  let seconds = 0;
  for (let i = from || 0; i < chunks.length; i++) {
    seconds += chunks[i].length / player.rate;
  }
  return seconds;
}

// One delta, as it arrives. In buffering it is kept and nothing is scheduled;
// the first time 1.5 s of audio is held, the engine is judged once. In live it
// is scheduled where the previous chunk ended, unless that moment has passed —
// the queue ran dry — which switches the take to refill. In refill it is kept,
// unscheduled, until the stream ends.
function receiveChunk(player, floats) {
  player.samples.push(floats);
  if (player.state === "live") {
    if (player.end < player.ctx.currentTime) player.state = "refill";
    else return scheduleChunk(player, floats, player.end);
  }
  player.pending.push(floats);
  if (player.state === "buffering" && !player.tested &&
      audioSeconds(player, player.pending) >= LIVE_LEAD_SECONDS) {
    decideKeepUp(player);
  }
}

// The one keep-up decision, taken when 1.5 s of audio is held. The first chunk
// is left out of the rate: its own latency says nothing about the engine's. No
// elapsed time (every chunk at the same instant) counts as keeping up. When it
// keeps up, everything held is scheduled from now and the take is live.
function decideKeepUp(player) {
  player.tested = true;
  const elapsed = (Date.now() - player.firstAt) / 1000;
  const kept = audioSeconds(player, player.samples, 1);
  if (elapsed && kept / elapsed < 1) return;
  player.state = "live";
  let at = player.ctx.currentTime;
  for (const chunk of player.pending) {
    scheduleChunk(player, chunk, at);
    at = player.end;
  }
  player.pending = [];
}

// At speech.audio.done, play whatever is still unscheduled as one block, from
// the later of the last chunk's end and the present.
function finishStream(player) {
  if (!player.pending.length) return;
  const at = Math.max(player.end, player.ctx.currentTime);
  scheduleChunk(player, concatSamples(player.pending), at);
  player.pending = [];
}

function concatSamples(chunks) {
  let length = 0;
  for (const chunk of chunks) length += chunk.length;
  const all = new Float32Array(length);
  let at = 0;
  for (const chunk of chunks) {
    all.set(chunk, at);
    at += chunk.length;
  }
  return all;
}

// Read `data:` events off a fetch body as they arrive, one JSON object each.
async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    pending += decoder.decode(value, { stream: true });
    let split = pending.indexOf("\n\n");
    while (split !== -1) {
      const block = pending.slice(0, split).trim();
      pending = pending.slice(split + 2);
      if (block.startsWith("data:")) onEvent(JSON.parse(block.slice(5).trim()));
      split = pending.indexOf("\n\n");
    }
  }
}

// Play the take as its events arrive, then hand the finished WAV to the player
// and the download link and show the statistics the done event carried.
async function runSpeechStreaming(payload) {
  const response = await api("/v1/audio/speech", {
    json: streamingPayload(payload),
    raw: true,
    engine: payload.model,
  });
  if (!response.body || !response.body.getReader) {
    throw new Error(MSG.network);
  }
  const rate = Number(response.headers.get(SAMPLE_RATE_HEADER)) || 24000;
  const player = newStreamPlayer(rate);
  const download = $("speak-download");
  download.hidden = true;
  let firstAt = 0;
  let ticker = null;
  let usage = null;
  const onEvent = (event) => {
    if (event.type === SSE_DELTA) {
      if (!firstAt) {
        firstAt = Date.now();
        player.firstAt = firstAt;
        ticker = window.setInterval(() => {
          $("speak-meta").textContent =
            "receiving… " + ((Date.now() - firstAt) / 1000).toFixed(1) + " s";
        }, 250);
      }
      receiveChunk(player, decodePcm(event.audio));
    } else if (event.type === SSE_DONE) {
      usage = event.usage;
      finishStream(player);
    } else if (event.type === "error") {
      throw new Error((event.error && event.error.message) || MSG.ttsError);
    }
  };
  try {
    await readEvents(response, onEvent);
  } finally {
    if (ticker) window.clearInterval(ticker);
  }
  if (!usage) throw new Error(MSG.ttsError);
  const blob = encodeWav(concatSamples(player.samples), player.rate);
  const previous = speakUrl;
  speakUrl = URL.createObjectURL(blob);
  $("speak-audio").src = speakUrl;
  download.href = speakUrl;
  download.hidden = false;
  if (previous) URL.revokeObjectURL(previous);
  $("speak-meta").textContent =
    usage.audio_duration +
    " s · " +
    usage.elapsed +
    " s · " +
    usage.rtf +
    "x real time";
}

// Today's request, unchanged: the whole take in one response.
async function runSpeechFallback(payload) {
  const response = await api("/v1/audio/speech", {
    json: payload,
    raw: true,
    engine: payload.model,
  });
  const duration = response.headers.get("X-Speakeasy-Audio-Duration") || "?";
  const elapsed = response.headers.get("X-Speakeasy-Elapsed") || "?";
  const rtf = response.headers.get("X-Speakeasy-RTF") || "?";
  const blob = await response.blob();
  const previous = speakUrl;
  speakUrl = URL.createObjectURL(blob);
  $("speak-audio").src = speakUrl;
  const download = $("speak-download");
  download.href = speakUrl;
  download.hidden = false;
  if (previous) URL.revokeObjectURL(previous);
  $("speak-meta").textContent =
    duration + " s · " + elapsed + " s · " + rtf + "x real time";
}

// Stream the take where the browser can, and ask for it whole otherwise.
async function runSpeech(payload) {
  if (streamingAvailable()) return runSpeechStreaming(payload);
  return runSpeechFallback(payload);
}

async function speak() {
  clearError($("speak-error"));
  const text = $("speak-text").value.trim();
  if (!text) throw new Error("type something to speak");
  const payload = speechPayload($("speak-engine").value, text);
  const btn = $("speak-go");
  btn.disabled = true;
  btn.textContent = MSG.working;
  try {
    await runSpeech(payload);
  } catch (err) {
    if (err.voxcpm2State === "loading" || err.voxcpm2State === "not_loaded") {
      holdPending({ kind: "speech", payload: payload });
    }
    throw err;
  } finally {
    btn.disabled = false;
    btn.textContent = "Speak";
  }
}

function wireSpeak() {
  $("speak-voice").addEventListener("change", () => {
    syncSpeakLanguage();
    syncSpeakDesignStyle();
  });
  $("speak-engine").addEventListener("change", applySpeakEngine);
  $("speak-cfg").addEventListener("input", syncCfgDisplay);
  $("speak-design").addEventListener("input", syncSpeakDesignStyle);
  $("speak-style").addEventListener("input", syncSpeakDesignStyle);
  $("speak-go").addEventListener("click", () => {
    guard($("speak-error"), speak);
  });
}

// --- Clone -------------------------------------------------------------------

function renderClones(items) {
  const list = $("clone-list");
  list.textContent = "";
  for (const voice of items.filter((v) => v.type === "clone")) {
    const item = document.createElement("li");
    const label = document.createElement("span");
    // The engine is part of the name: ids are unique across both engines.
    label.textContent = voice.id + " · " + voice.engine;
    const del = document.createElement("button");
    del.type = "button";
    del.textContent = "Delete";
    del.addEventListener("click", () => {
      guard($("clone-error"), () => deleteVoice(voice.id));
    });
    item.appendChild(label);
    item.appendChild(del);
    list.appendChild(item);
  }
}

async function deleteVoice(id) {
  if (!window.confirm("Delete saved voice " + id + "?")) return;
  await api("/v1/audio/voices/" + encodeURIComponent(id), { method: "DELETE" });
  await loadVoices();
}

function applyCloneEngine() {
  const voxcpm2 = $("clone-engine").value === ENGINE_VOXCPM2;
  // The stored language still passes through to metadata; it is not a choice.
  $("clone-language-label").hidden = voxcpm2;
  $("clone-text-label").textContent = voxcpm2
    ? TRANSCRIPT_OPTIONAL
    : TRANSCRIPT_REQUIRED;
  $("clone-text").placeholder = voxcpm2
    ? "optional transcript of the recording"
    : "exact transcript of the recording";
}

function cloneForm(fields) {
  const form = new FormData();
  form.append("file", fields.blob, "reference.wav");
  if (fields.text) form.append("text", fields.text);
  form.append("language", fields.language || "en");
  form.append("model", fields.model);
  if (fields.voiceId) form.append("voice_id", fields.voiceId);
  return form;
}

async function runCloneSave(fields) {
  const btn = $("clone-save");
  btn.disabled = true;
  btn.textContent = MSG.working;
  try {
    const created = await api("/v1/audio/voices", {
      form: cloneForm(fields),
      engine: fields.model,
    });
    $("clone-meta").textContent = "saved " + created.id;
    await loadVoices();
  } finally {
    btn.disabled = false;
    btn.textContent = "Save";
  }
}

async function saveClone() {
  clearError($("clone-error"));
  const engine = $("clone-engine").value;
  const text = $("clone-text").value.trim();
  // VoxCPM2 takes a reference with or without its transcript.
  if (engine !== ENGINE_VOXCPM2 && !text) {
    throw new Error("enter the recording's exact transcript");
  }
  if (!lastRecording) throw new Error("record or upload a reference clip first");
  const seconds = lastRecording.duration;
  if (seconds && (seconds < 1 || seconds > 30)) {
    throw new Error("reference must be 1-30 s (3-10 s is best)");
  }
  const fields = {
    blob: lastRecording.blob,
    text: text,
    language: $("clone-language").value || "en",
    voiceId: $("clone-name").value.trim(),
    model: engine,
  };
  try {
    await runCloneSave(fields);
  } catch (err) {
    if (err.voxcpm2State === "loading" || err.voxcpm2State === "not_loaded") {
      holdPending({ kind: "clone", payload: fields });
    }
    throw err;
  }
}

function wireClone() {
  const fileInput = $("clone-file");
  fileInput.addEventListener("change", () => {
    if (!fileInput.files || !fileInput.files[0]) return;
    const file = fileInput.files[0];
    guard($("clone-error"), async () => {
      lastRecording = await toWav(file);
      $("clone-state").textContent =
        "captured " + lastRecording.duration.toFixed(1) + " s";
    });
  });
  $("clone-use-last").addEventListener("click", () => {
    $("clone-text").value = $("listen-text").value;
  });
  $("clone-engine").addEventListener("change", applyCloneEngine);
  $("clone-save").addEventListener("click", () => {
    guard($("clone-error"), saveClone);
  });
}

// --- first-use loading -------------------------------------------------------

// Hold the request the user just made and start (or join) the readiness poll.
function holdPending(request) {
  pendingRequest = request;
  pollWhileLoading();
}

function sendPending(pending) {
  if (pending.kind === "speech") {
    guard($("speak-error"), () => runSpeech(pending.payload));
  } else {
    guard($("clone-error"), () => runCloneSave(pending.payload));
  }
}

function pollWhileLoading() {
  if (loadPollTimer !== null) return; // one loop only, however many requests
  const tick = async () => {
    // The token stays held while the probe is in flight: a request held in
    // that window joins this loop instead of starting a second one.
    const state = await voxcpm2State();
    if (state === "loading") {
      loadPollTimer = window.setTimeout(tick, LOAD_POLL_MS);
      return;
    }
    // Any other status ends the loop: release the token, take the request
    // once, then decide. A resend that reports loading again can restart it.
    loadPollTimer = null;
    const pending = pendingRequest;
    pendingRequest = null;
    if (state !== "ready") return;
    await refreshStatus();
    clearError($("speak-error"));
    clearError($("clone-error"));
    loadVoices().catch((err) => showError($("speak-error"), err));
    if (pending) sendPending(pending);
  };
  loadPollTimer = window.setTimeout(tick, LOAD_POLL_MS);
}

// --- voices, status, boot ----------------------------------------------------

async function loadVoices() {
  const data = await api("/v1/audio/voices");
  const items = data.data || [];
  // `languages` comes only from a ready Reverb engine; a listing without it
  // must leave the language select's content alone.
  if (data.languages) fillLanguages(data.languages);
  fillVoices(items);
  renderClones(items);
  voicesLoaded = true;
}

async function refreshStatus() {
  const el = $("status");
  try {
    const response = await fetch("/health/ready");
    const body = await response.json();
    const models = (body.models || []).join(", ") || "none";
    el.textContent =
      (body.device || "?") +
      " · " +
      models +
      " · tts: " +
      (body.tts || "?") +
      ", voxcpm2: " +
      (body.voxcpm2 || "?");
    // A disabled engine is absent from the server's listing, so it is absent
    // from both selectors too.
    applyEngineAvailability(body.voxcpm2 || "");
    // The listing answers whenever either engine can supply voices.
    const canList =
      body.tts === "ready" ||
      (body.voxcpm2 && body.voxcpm2 !== "disabled");
    if (canList && !voicesLoaded) {
      loadVoices().catch((err) => showError($("speak-error"), err));
    }
  } catch (e) {
    el.textContent = MSG.network;
  }
}

function init() {
  const keyInput = $("apiKey");
  keyInput.value = key();
  keyInput.addEventListener("change", () => {
    sessionStorage.setItem(KEY_STORAGE, keyInput.value.trim());
    loadVoices().catch((err) => showError($("speak-error"), err));
  });

  wireListen();
  wireSpeak();
  wireClone();
  wireRecording("listen", (captured) =>
    runTranscription(captured.blob, captured.duration),
  );
  wireRecording("clone", null);

  refreshStatus();
  window.setInterval(refreshStatus, 30000);
  loadVoices().catch((err) => showError($("speak-error"), err));
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
