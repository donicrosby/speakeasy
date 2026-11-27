"use strict";

// speakeasy static UI — three panels, no framework, no build step.
//
// The API key lives in sessionStorage only: it is read here, sent as an
// Authorization header, and never placed in a URL or a cookie. All dynamic
// text is written with textContent; server data is never rendered as markup.

const $ = (id) => document.getElementById(id);

const KEY_STORAGE = "speakeasy_key";
const key = () => sessionStorage.getItem(KEY_STORAGE) || "";

// Fixed messages (spec §7.5). Kept literal so every state is greppable.
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
};

// The most recent recording, shared with the Clone panel.
let lastRecording = null;
let voicesLoaded = false;

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

async function runTranscription(blob) {
  const errEl = $("listen-error");
  clearError(errEl);
  const stateEl = $("listen-state");
  stateEl.textContent = MSG.working;
  const form = new FormData();
  form.append("file", blob, "audio.wav");
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
      await runTranscription(wav.blob);
    });
  });
}

// --- Speak -------------------------------------------------------------------

function fillLanguages(languages) {
  for (const id of ["speak-language", "clone-language"]) {
    const select = $(id);
    select.textContent = "";
    for (const code of languages) {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = code;
      select.appendChild(option);
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
  const select = $("speak-voice");
  select.textContent = "";
  addGroup(select, "Presets", items.filter((v) => v.type === "preset"));
  addGroup(select, "Saved", items.filter((v) => v.type === "clone"));
  syncSpeakLanguage();
}

function syncSpeakLanguage() {
  const select = $("speak-voice");
  const option = select.selectedOptions[0];
  const language = option && option.dataset.language;
  if (language) $("speak-language").value = language;
}

async function speak() {
  clearError($("speak-error"));
  const text = $("speak-text").value.trim();
  if (!text) throw new Error("type something to speak");
  const btn = $("speak-go");
  btn.disabled = true;
  btn.textContent = MSG.working;
  try {
    const response = await api("/v1/audio/speech", {
      json: {
        input: text,
        voice: $("speak-voice").value,
        language: $("speak-language").value || null,
        response_format: "wav",
      },
      raw: true,
    });
    const duration = response.headers.get("X-Speakeasy-Audio-Duration") || "?";
    const elapsed = response.headers.get("X-Speakeasy-Elapsed") || "?";
    const rtf = response.headers.get("X-Speakeasy-RTF") || "?";
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    $("speak-audio").src = url;
    $("speak-download").href = url;
    $("speak-meta").textContent =
      duration + " s · " + elapsed + " s · " + rtf + "x real time";
  } finally {
    btn.disabled = false;
    btn.textContent = "Speak";
  }
}

function wireSpeak() {
  $("speak-voice").addEventListener("change", syncSpeakLanguage);
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
    label.textContent = voice.id;
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

async function saveClone() {
  clearError($("clone-error"));
  const text = $("clone-text").value.trim();
  if (!text) throw new Error("enter the recording's exact transcript");
  if (!lastRecording) throw new Error("record or upload a reference clip first");
  const seconds = lastRecording.duration;
  if (seconds && (seconds < 1 || seconds > 30)) {
    throw new Error("reference must be 1-30 s (3-10 s is best)");
  }
  const form = new FormData();
  form.append("file", lastRecording.blob, "reference.wav");
  form.append("text", text);
  form.append("language", $("clone-language").value || "en");
  const name = $("clone-name").value.trim();
  if (name) form.append("voice_id", name);
  const btn = $("clone-save");
  btn.disabled = true;
  btn.textContent = MSG.working;
  try {
    const created = await api("/v1/audio/voices", { form: form });
    $("clone-meta").textContent = "saved " + created.id;
    await loadVoices();
  } finally {
    btn.disabled = false;
    btn.textContent = "Save";
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
  $("clone-save").addEventListener("click", () => {
    guard($("clone-error"), saveClone);
  });
}

// --- voices, status, boot ----------------------------------------------------

async function loadVoices() {
  const data = await api("/v1/audio/voices");
  const items = data.data || [];
  fillLanguages(data.languages || []);
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
      (body.device || "?") + " · " + models + " · tts: " + (body.tts || "?");
    if (body.tts === "ready" && !voicesLoaded) {
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
  wireRecording("listen", (captured) => runTranscription(captured.blob));
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
