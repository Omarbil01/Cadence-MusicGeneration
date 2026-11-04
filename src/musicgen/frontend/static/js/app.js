"use strict";

const config = JSON.parse(document.getElementById("app-config").textContent);
const form = document.getElementById("generation-form");
const durationInput = document.getElementById("duration");
const durationRange = document.getElementById("duration-range");
const generateButton = document.getElementById("generate-button");
const generateLabel = document.getElementById("generate-label");
const availabilityMessage = document.getElementById("availability-message");
const connectionDot = document.getElementById("connection-dot");
const connectionLabel = document.getElementById("connection-label");
const resultCard = document.getElementById("result-card");
const resultState = document.getElementById("result-state");
const resultHeading = document.getElementById("result-heading");
const resultMessage = document.getElementById("result-message");
const downloadButton = document.getElementById("download-button");
const elapsed = document.getElementById("elapsed-time");
const labels = {classical: "Classical", romantic: "Romantic", rock_pop: "Rock & Pop"};
const storageKey = "cadence.activeJob";
let available = false;
let busy = false;
let activeJob = null;
let timer = null;
let maxDuration = config.maxDuration || 240;

function apiUrl(path) {
  return config.apiBaseUrl.replace(/\/$/, "") + path;
}

function timeLabel(seconds) {
  seconds = Math.max(0, Math.round(Number(seconds) || 0));
  return String(Math.floor(seconds / 60)) + ":" + String(seconds % 60).padStart(2, "0");
}

function storeActive(job) {
  try {
    if (job) localStorage.setItem(storageKey, JSON.stringify({id: job.id}));
    else localStorage.removeItem(storageKey);
  } catch (_) { /* The page also works with browser storage disabled. */ }
}

function selectedGenre() {
  return form.querySelector('input[name="genre"]:checked').value;
}

function setDuration(value) {
  const duration = Number(value);
  if (Number.isInteger(duration) && duration >= 5 && duration <= maxDuration) {
    durationInput.value = duration;
    durationRange.value = duration;
    durationRange.setAttribute("aria-valuetext", String(duration) + " seconds");
    document.getElementById("summary-duration").textContent = timeLabel(duration);
    document.querySelectorAll("[data-duration]").forEach(function (button) {
      button.classList.toggle("selected", Number(button.dataset.duration) === duration);
    });
  }
}

function updateControls() {
  generateButton.disabled = busy || !available;
  form.querySelectorAll("input, button[data-duration]").forEach(function (control) {
    control.disabled = busy;
  });
  generateLabel.textContent = busy ? "Creating your MIDI…" : "Generate MIDI";
}

async function request(path, options) {
  const controller = new AbortController();
  const timeout = setTimeout(function () { controller.abort(); }, 8000);
  try {
    const response = await fetch(apiUrl(path), Object.assign({}, options || {}, {
      signal: controller.signal,
      cache: "no-store"
    }));
    let data;
    try { data = await response.json(); } catch (_) { data = {}; }
    return {response: response, data: data};
  } finally {
    clearTimeout(timeout);
  }
}

async function checkAvailability() {
  try {
    const result = await request("/api/health/ready");
    available = result.response.ok && result.data.ready === true;
    connectionLabel.textContent = available ? "Studio available" : "Generator offline";
    connectionDot.className = "connection-dot " + (available ? "available" : "offline");
    if (!busy) {
      availabilityMessage.textContent = available
        ? "Ready to create. Generation may take a little time."
        : "Music generation is currently unavailable. Please try again shortly.";
      availabilityMessage.classList.remove("error");
    }
  } catch (_) {
    available = false;
    connectionLabel.textContent = "Reconnecting";
    connectionDot.className = "connection-dot offline";
    if (!busy) availabilityMessage.textContent = "The studio is temporarily unreachable. Reconnecting…";
  }
  updateControls();
}

function showWorking(job) {
  resultCard.classList.add("is-working");
  resultState.textContent = job.status === "queued" ? "IN THE QUEUE" : "CREATING YOUR COMPOSITION";
  resultHeading.textContent = job.status === "queued" ? "Your idea is next." : "A little music in the making.";
  resultMessage.textContent = labels[job.genre] + " · " + timeLabel(job.duration_seconds) + " · MIDI";
  elapsed.hidden = false;
  downloadButton.hidden = true;
  if (timer) clearInterval(timer);
  const started = job.created_at ? job.created_at * 1000 : Date.now();
  function updateElapsed() {
    elapsed.textContent = "Elapsed " + timeLabel((Date.now() - started) / 1000);
  }
  updateElapsed();
  timer = setInterval(updateElapsed, 1000);
}

function finish(job) {
  busy = false;
  if (timer) clearInterval(timer);
  timer = null;
  elapsed.hidden = true;
  resultCard.classList.remove("is-working");
  storeActive(null);
  if (job.status === "completed") {
    resultState.textContent = "READY TO TAKE WITH YOU";
    resultHeading.textContent = "Your MIDI is ready.";
    resultMessage.textContent = labels[job.genre] + " · " +
      timeLabel(job.actual_duration_seconds || job.duration_seconds) + " · MIDI file";
    downloadButton.href = apiUrl("/api/generations/" + job.id + "/download");
    downloadButton.hidden = false;
    availabilityMessage.textContent = "Download your composition, or create another.";
    availabilityMessage.classList.remove("error");
  } else {
    resultState.textContent = "LET'S TRY AGAIN";
    resultHeading.textContent = "This one didn't finish.";
    resultMessage.textContent = job.error || "Please try a new composition.";
    downloadButton.hidden = true;
    availabilityMessage.textContent = job.error || "Generation failed. Please try again.";
    availabilityMessage.classList.add("error");
  }
  activeJob = null;
  updateControls();
}

async function poll(jobId) {
  let failures = 0;
  while (activeJob && activeJob.id === jobId) {
    try {
      const result = await request("/api/generations/" + jobId);
      if (result.response.status === 404) {
        finish({status: "failed", error: "This generation is no longer available. Start a new one."});
        return;
      }
      if (!result.response.ok) throw new Error("Status unavailable");
      const job = result.data;
      failures = 0;
      if (job.status === "completed" || job.status === "failed") {
        finish(job);
        return;
      }
      activeJob = job;
      showWorking(job);
      availabilityMessage.textContent = job.status === "queued"
        ? "Your composition is queued. You can leave this page open."
        : "Creating your composition. You can leave this page open.";
    } catch (_) {
      failures += 1;
      availabilityMessage.textContent = "Connection interrupted. Checking your generation again…";
    }
    await new Promise(function (resolve) {
      setTimeout(resolve, Math.min(10000, 1500 * (1 + failures)));
    });
  }
}

form.addEventListener("submit", async function (event) {
  event.preventDefault();
  if (busy || !form.reportValidity()) return;
  const payload = {genre: selectedGenre(), duration_seconds: Number(durationInput.value)};
  busy = true;
  updateControls();
  availabilityMessage.classList.remove("error");
  availabilityMessage.textContent = "Sending your composition request…";
  try {
    const result = await request("/api/generations", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    if (!result.response.ok) {
      const detail = typeof result.data.detail === "string" ? result.data.detail
        : "Please check your style and duration, then try again.";
      throw new Error(detail);
    }
    if (!/^[0-9a-f]{32}$/.test(result.data.id)) throw new Error("The studio returned an invalid job.");
    activeJob = result.data;
    storeActive(activeJob);
    showWorking(activeJob);
    void poll(activeJob.id);
  } catch (error) {
    busy = false;
    availabilityMessage.textContent = error.name === "AbortError"
      ? "The request timed out. Please check the connection before trying again."
      : error.message || "The request could not be sent.";
    availabilityMessage.classList.add("error");
    updateControls();
  }
});

durationInput.addEventListener("input", function () { setDuration(durationInput.value); });
durationRange.addEventListener("input", function () { setDuration(durationRange.value); });
document.querySelectorAll("[data-duration]").forEach(function (button) {
  button.addEventListener("click", function () { setDuration(button.dataset.duration); });
});
form.querySelectorAll('input[name="genre"]').forEach(function (input) {
  input.addEventListener("change", function () {
    document.getElementById("summary-genre").textContent = labels[selectedGenre()];
  });
});

async function initialize() {
  setDuration(config.defaultDuration || 45);
  try {
    const result = await request("/api/config");
    if (result.response.ok) {
      maxDuration = result.data.max_duration_seconds;
      durationInput.max = String(maxDuration);
      durationRange.max = String(maxDuration);
      document.getElementById("max-duration-label").textContent = timeLabel(maxDuration);
      document.querySelectorAll("[data-duration]").forEach(function (button) {
        button.hidden = Number(button.dataset.duration) > maxDuration;
      });
      setDuration(result.data.default_duration_seconds);
    }
  } catch (_) { /* The availability message reports connection trouble. */ }
  await checkAvailability();
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
    if (saved && /^[0-9a-f]{32}$/.test(saved.id)) {
      activeJob = saved;
      busy = true;
      updateControls();
      void poll(saved.id);
    }
  } catch (_) { /* No saved generation to restore. */ }
  setInterval(checkAvailability, 10000);
}

void initialize();
