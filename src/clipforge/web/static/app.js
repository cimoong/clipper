"use strict";

const form = document.getElementById("job-form");
const urlInput = document.getElementById("url");
const formMsg = document.getElementById("form-msg");
const tbody = document.querySelector("#jobs tbody");
const refreshBtn = document.getElementById("refresh");

// Job ids we already have a live SSE stream for, so we don't open duplicates.
const streaming = new Set();

function setMsg(text, kind) {
  formMsg.textContent = text;
  formMsg.className = "msg" + (kind ? " " + kind : "");
}

async function loadJobs() {
  const res = await fetch("/api/jobs");
  if (!res.ok) {
    setMsg("Could not load jobs.", "error");
    return;
  }
  const jobs = await res.json();
  tbody.innerHTML = "";
  for (const job of jobs) {
    tbody.appendChild(renderRow(job));
    if (job.status !== "DONE" && job.status !== "FAILED") {
      watch(job.id);
    }
  }
}

function renderRow(job) {
  const tr = document.createElement("tr");
  tr.dataset.jobId = job.id;
  tr.innerHTML = `
    <td class="job-id">${job.id}</td>
    <td>${escapeHtml(job.title || job.source_url || "")}</td>
    <td><span class="badge ${job.status}">${job.status}</span></td>
    <td><div class="bar"><span style="width:${job.progress || 0}%"></span></div></td>
  `;
  return tr;
}

function updateRow(jobId, status, progress) {
  const tr = tbody.querySelector(`tr[data-job-id="${jobId}"]`);
  if (!tr) return;
  const badge = tr.querySelector(".badge");
  badge.textContent = status;
  badge.className = "badge " + status;
  tr.querySelector(".bar > span").style.width = (progress || 0) + "%";
}

function watch(jobId) {
  if (streaming.has(jobId)) return;
  streaming.add(jobId);
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  es.addEventListener("progress", (ev) => {
    const data = JSON.parse(ev.data);
    updateRow(jobId, data.status, data.progress);
    if (data.status === "DONE" || data.status === "FAILED") {
      es.close();
      streaming.delete(jobId);
    }
  });
  es.addEventListener("error", () => {
    es.close();
    streaming.delete(jobId);
  });
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const url = urlInput.value.trim();
  if (!url) return;
  setMsg("Enqueuing…");
  const res = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) {
    setMsg("Failed to enqueue job.", "error");
    return;
  }
  const data = await res.json();
  setMsg(`Queued job ${data.job_id}`, "ok");
  urlInput.value = "";
  loadJobs();
});

refreshBtn.addEventListener("click", loadJobs);

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

loadJobs();
