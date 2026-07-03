"use strict";

// The only hand-written JS on the results page: a <dialog>-based preview modal.
// Everything else (job submit, live status, caption edit, re-render) is htmx.
const modal = document.getElementById("preview-modal");
const video = document.getElementById("preview-video");
const titleEl = document.getElementById("preview-title");

// Event delegation so cards swapped in by htmx keep working.
document.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".preview");
  if (btn && !btn.disabled) {
    video.src = btn.dataset.src;
    titleEl.textContent = btn.dataset.title || "Preview";
    modal.showModal();
    video.play().catch(() => {});
    return;
  }
  // Close on the ✕ button or a click on the backdrop (which targets <dialog>).
  if (ev.target.id === "preview-close" || ev.target === modal) {
    modal.close();
  }
});

// Stop playback whenever the dialog closes (button, Esc, or backdrop).
modal.addEventListener("close", () => {
  video.pause();
  video.removeAttribute("src");
  video.load();
});
