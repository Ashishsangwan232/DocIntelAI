/**
 * components/documentScopeSelect.js
 * ===================================
 * A `<select multiple>` of `ready` documents, used to scope both Chat
 * and Search to a subset of the library — empty selection means
 * "search everything". Extracted here once a second page (Search)
 * needed the identical selector Chat already had.
 */

import { documents as documentsApi } from "../api/resources.js";
import { escapeHtml } from "../utils/format.js";

export function createDocumentScopeSelect({ onChange } = {}) {
  const select = document.createElement("select");
  select.className = "select";
  select.multiple = true;
  select.size = 1;

  select.addEventListener("change", () => {
    const selectedIds = Array.from(select.selectedOptions).map((option) => option.value);
    onChange?.(selectedIds);
  });

  async function load() {
    try {
      const readyDocuments = (await documentsApi.list({ status: "ready" })) ?? [];
      if (readyDocuments.length === 0) {
        select.innerHTML = "";
        select.disabled = true;
        select.title = "No processed documents yet — upload one to get started.";
        return;
      }
      select.disabled = false;
      select.size = Math.min(readyDocuments.length, 4);
      select.title = "Ctrl/Cmd-click to select multiple. None selected = all documents.";
      select.innerHTML = readyDocuments
        .map((doc) => `<option value="${doc.id}">${escapeHtml(doc.filename)}</option>`)
        .join("");
    } catch {
      select.disabled = true;
    }
  }

  select.clearSelection = () => {
    Array.from(select.options).forEach((option) => {
      option.selected = false;
    });
  };

  load();
  return select;
}
