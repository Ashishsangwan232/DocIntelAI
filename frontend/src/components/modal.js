/**
 * components/modal.js
 * ====================
 * A plain, one-shot modal: `openModal()` creates a fresh overlay+card
 * in the DOM and returns a `close()` that fully tears it down —
 * listeners included.
 *
 * This is deliberately NOT the Streamlit `st.dialog` pattern this app
 * used to have (a persistent "is this open" flag in session state,
 * re-checked on every rerun). That pattern was the root cause of a
 * real bug fixed earlier in this migration: a dialog dismissed any
 * way other than its own Close button left the flag stuck "open",
 * so it kept reappearing on unrelated pages. A modal created here
 * only exists because something just called `openModal()`, and
 * `close()` unconditionally removes it and its listeners — there is
 * no stale flag left behind for a later, unrelated action to
 * accidentally reactivate.
 */

export function openModal({ title, width = "default", onClose } = {}) {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";

  const modal = document.createElement("div");
  modal.className = width === "large" ? "modal modal-large" : "modal";
  modal.setAttribute("role", "dialog");
  modal.setAttribute("aria-modal", "true");
  if (title) modal.setAttribute("aria-label", title);

  modal.innerHTML = `
    <div class="modal-header">
      <h3>${title ?? ""}</h3>
      <button class="modal-close-x" aria-label="Close" type="button">&times;</button>
    </div>
    <div class="modal-body"></div>
  `;
  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  let closed = false;
  function close() {
    if (closed) return;
    closed = true;
    document.removeEventListener("keydown", onKeydown);
    overlay.remove();
    onClose?.();
  }

  function onKeydown(event) {
    if (event.key === "Escape") close();
  }

  document.addEventListener("keydown", onKeydown);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) close();
  });
  modal.querySelector(".modal-close-x").addEventListener("click", close);

  // Focus the modal itself so Escape/Tab work immediately without
  // requiring a click first — matches Streamlit's dialog behavior of
  // grabbing focus on open.
  modal.tabIndex = -1;
  modal.focus();

  return { overlay, modal, body: modal.querySelector(".modal-body"), close };
}
