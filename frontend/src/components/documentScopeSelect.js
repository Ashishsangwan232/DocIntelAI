/**
 * components/documentScopeSelect.js
 * ===================================
 * A closed-by-default dropdown for scoping Chat/Search to a subset of
 * `ready` documents — empty selection means "search everything".
 *
 * This is deliberately NOT a native `<select multiple>`: browsers
 * render a multi-select as an always-expanded listbox (every option
 * visible at once, no open/close affordance), which is why the old
 * implementation looked like a plain list of files rather than a
 * dropdown. This version is a button that opens a checkbox panel on
 * click, closes on outside click/Escape, and shows a short summary
 * ("All documents", "3 documents selected") when collapsed.
 */

import { documents as documentsApi } from "../api/resources.js";
import { escapeHtml } from "../utils/format.js";

export function createDocumentScopeSelect({ onChange } = {}) {
  const root = document.createElement("div");
  root.className = "doc-scope-dropdown";

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "doc-scope-toggle";
  toggle.setAttribute("aria-haspopup", "listbox");
  toggle.setAttribute("aria-expanded", "false");
  toggle.disabled = true;
  toggle.innerHTML = `
    <span class="doc-scope-toggle-label">Loading documents...</span>
    <span class="doc-scope-toggle-caret" aria-hidden="true">▾</span>
  `;

  const panel = document.createElement("div");
  panel.className = "doc-scope-panel";
  panel.setAttribute("role", "listbox");
  panel.setAttribute("aria-multiselectable", "true");
  panel.hidden = true;

  root.append(toggle, panel);

  let allDocuments = [];
  let selectedIds = new Set();

  function summaryLabel() {
    if (allDocuments.length === 0) return "No documents";
    if (selectedIds.size === 0) return "All documents";
    if (selectedIds.size === 1) {
      const doc = allDocuments.find((d) => d.id === [...selectedIds][0]);
      return doc ? doc.filename : "1 document selected";
    }
    return `${selectedIds.size} documents selected`;
  }

  function updateLabel() {
    toggle.querySelector(".doc-scope-toggle-label").textContent = summaryLabel();
  }

  function openPanel() {
    if (toggle.disabled) return;
    panel.hidden = false;
    toggle.setAttribute("aria-expanded", "true");
  }

  function closePanel() {
    panel.hidden = true;
    toggle.setAttribute("aria-expanded", "false");
  }

  toggle.addEventListener("click", () => {
    if (panel.hidden) openPanel();
    else closePanel();
  });

  document.addEventListener("click", (event) => {
    if (!panel.hidden && !root.contains(event.target)) closePanel();
  });

  root.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !panel.hidden) {
      closePanel();
      toggle.focus();
    }
  });

  function renderPanel() {
    panel.innerHTML = "";
    if (allDocuments.length === 0) {
      panel.innerHTML = `<p class="doc-scope-empty text-muted">No processed documents yet.</p>`;
      return;
    }
    for (const doc of allDocuments) {
      const optionId = `doc-scope-opt-${doc.id}`;
      const option = document.createElement("label");
      option.className = "doc-scope-option";
      option.htmlFor = optionId;
      option.innerHTML = `
        <input type="checkbox" id="${optionId}" value="${doc.id}" />
        <span>${escapeHtml(doc.filename)}</span>
      `;
      const checkbox = option.querySelector("input");
      checkbox.checked = selectedIds.has(doc.id);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) selectedIds.add(doc.id);
        else selectedIds.delete(doc.id);
        updateLabel();
        onChange?.([...selectedIds]);
      });
      panel.appendChild(option);
    }
  }

  async function load() {
    try {
      allDocuments = (await documentsApi.list({ status: "ready" })) ?? [];
      if (allDocuments.length === 0) {
        toggle.disabled = true;
        toggle.title = "No processed documents yet — upload one to get started.";
        updateLabel();
        return;
      }
      toggle.disabled = false;
      toggle.title = "Filter which documents to search. None selected = all documents.";
      renderPanel();
      updateLabel();
    } catch {
      toggle.disabled = true;
      toggle.title = "Couldn't load documents.";
      updateLabel();
    }
  }

  root.clearSelection = () => {
    selectedIds = new Set();
    renderPanel();
    updateLabel();
  };

  load();
  return root;
}