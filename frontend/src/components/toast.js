/**
 * components/toast.js
 * ====================
 * Lightweight, auto-dismissing notifications for transient
 * confirmations/errors that don't need a full inline alert taking up
 * permanent page space (e.g. "Settings saved", "Couldn't start a new
 * chat"). Reserve inline `.alert` elements (see modal.js callers and
 * each page) for errors that need to stay visible in context — a
 * failed document upload should stay next to the file it failed for,
 * not disappear after 4 seconds.
 *
 * One `<div>` region is created lazily on first use and reused for
 * every toast afterwards; it's an `aria-live="polite"` region so
 * screen readers announce new toasts without interrupting whatever
 * the user is currently doing.
 */

let regionEl = null;

function getRegion() {
  if (regionEl) return regionEl;
  regionEl = document.createElement("div");
  regionEl.className = "toast-region";
  regionEl.setAttribute("role", "status");
  regionEl.setAttribute("aria-live", "polite");
  document.body.appendChild(regionEl);
  return regionEl;
}

/**
 * @param {string} message
 * @param {{type?: 'info'|'error', duration?: number}} [options]
 */
export function showToast(message, { type = "info", duration = 3500 } = {}) {
  const region = getRegion();

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  region.appendChild(toast);

  // Force layout before adding the visible class so the CSS
  // transition actually plays instead of the toast appearing already
  // in its final state.
  requestAnimationFrame(() => toast.classList.add("toast-visible"));

  const remove = () => {
    toast.classList.remove("toast-visible");
    toast.addEventListener("transitionend", () => toast.remove(), { once: true });
    // Safety net in case the transitionend event never fires (e.g.
    // prefers-reduced-motion sets transition-duration to ~0).
    setTimeout(() => toast.remove(), 300);
  };

  toast.addEventListener("click", remove);
  setTimeout(remove, duration);
}
