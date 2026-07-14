/**
 * pages/documents.js
 * ===================
 * Full Documents page — feature parity with `src/ui/upload.py`:
 * drag-and-drop upload with per-file outcomes, a searchable/sortable
 * library, and preview/delete/summary modals.
 *
 * Search and sort are client-side (filter/sort the already-fetched
 * list), same rationale as the Streamlit version: a library of a few
 * hundred documents doesn't need a server round-trip per keystroke.
 */

import { documents as documentsApi } from "../api/resources.js";
import { openModal } from "../components/modal.js";
import { escapeHtml, formatBytes, formatUploadedAt } from "../utils/format.js";

const SORT_OPTIONS = {
  "Newest first": (list) =>
    [...list].sort((a, b) => new Date(b.uploaded_at) - new Date(a.uploaded_at)),
  "Oldest first": (list) =>
    [...list].sort((a, b) => new Date(a.uploaded_at) - new Date(b.uploaded_at)),
  "Name (A-Z)": (list) => [...list].sort((a, b) => a.filename.localeCompare(b.filename)),
  "Size (largest first)": (list) => [...list].sort((a, b) => b.file_size_bytes - a.file_size_bytes),
};

const STATUS_BADGES = {
  ready: "🟢 Ready",
  processing: "🟡 Processing",
  failed: "🔴 Failed",
};

export function renderDocumentsPage(container) {
  let allDocuments = [];
  let searchQuery = "";
  let sortChoice = "Newest first";
  let disposed = false;

  container.innerHTML = `
    <div class="page-content stack">
      <div class="page-header">
        <h1>📁 Documents</h1>
        <p class="text-muted">Upload, preview, and manage your document library.</p>
      </div>

      <div class="card stack">
        <div>
          <h3>Upload Documents</h3>
          <p class="text-muted">
            Drag and drop files here, or click to browse. Supported formats: PDF, DOCX, TXT, Markdown.
          </p>
        </div>
        <div id="dropzone" class="dropzone" tabindex="0" role="button" aria-label="Upload documents">
          <input type="file" id="file-input" multiple accept=".pdf,.docx,.txt,.md" hidden />
          <div class="dropzone-icon" aria-hidden="true">⬆️</div>
          <p>Drop files here or click to browse</p>
        </div>
        <div id="upload-status" class="stack"></div>
      </div>

      <div class="card stack">
        <h3>Document Library</h3>
        <div class="row">
          <input
            type="text" id="search-input" class="input"
            placeholder="Search by filename..." style="flex: 1"
          />
          <select id="sort-select" class="select" style="max-width: 220px">
            ${Object.keys(SORT_OPTIONS)
              .map((label) => `<option value="${label}">${label}</option>`)
              .join("")}
          </select>
        </div>
        <div id="library-list" class="stack"></div>
      </div>
    </div>
  `;

  const dropzone = container.querySelector("#dropzone");
  const fileInput = container.querySelector("#file-input");
  const uploadStatus = container.querySelector("#upload-status");
  const searchInput = container.querySelector("#search-input");
  const sortSelect = container.querySelector("#sort-select");
  const libraryList = container.querySelector("#library-list");

  // --- Upload wiring -------------------------------------------------
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dropzone--active");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dropzone--active"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dropzone--active");
    handleFiles(event.dataTransfer.files);
  });
  fileInput.addEventListener("change", () => handleFiles(fileInput.files));

  async function handleFiles(fileList) {
    const files = Array.from(fileList);
    fileInput.value = ""; // allow re-selecting the same file next time
    if (files.length === 0) return;

    uploadStatus.innerHTML = `
      <div class="row"><span class="spinner"></span><span>Uploading ${files.length} file(s)...</span></div>
    `;

    try {
      const { results } = await documentsApi.upload(files);
      renderUploadResults(results);
      await loadDocuments();
    } catch (err) {
      uploadStatus.innerHTML = `<div class="alert alert-error">Upload failed: ${escapeHtml(err.message)}</div>`;
    }
  }

  function renderUploadResults(results) {
    const successes = results.filter((r) => r.success);
    const failures = results.filter((r) => !r.success);
    const parts = [];
    if (successes.length > 0) {
      parts.push(
        `<div class="alert alert-info">✅ ${successes.length} file(s) uploaded and processed successfully.</div>`
      );
    }
    for (const failure of failures) {
      parts.push(
        `<div class="alert alert-error"><strong>${escapeHtml(failure.filename)}</strong>: ${escapeHtml(failure.message)}</div>`
      );
    }
    uploadStatus.innerHTML = parts.join("");
  }

  // --- Library: load, search, sort ------------------------------------
  async function loadDocuments() {
    libraryList.innerHTML = `<div class="row"><span class="spinner"></span><span>Loading documents...</span></div>`;
    try {
      allDocuments = await documentsApi.list();
      renderLibrary();
    } catch (err) {
      libraryList.innerHTML = `<div class="alert alert-error">Couldn't load the document library: ${escapeHtml(err.message)}</div>`;
    }
  }

  searchInput.addEventListener("input", () => {
    searchQuery = searchInput.value;
    renderLibrary();
  });
  sortSelect.addEventListener("change", () => {
    sortChoice = sortSelect.value;
    renderLibrary();
  });

  function renderLibrary() {
    if (disposed) return;

    if (allDocuments.length === 0) {
      libraryList.innerHTML = `<div class="empty-state">No documents uploaded yet. Upload a file above to get started.</div>`;
      return;
    }

    const needle = searchQuery.trim().toLowerCase();
    const filtered = needle
      ? allDocuments.filter((d) => d.filename.toLowerCase().includes(needle))
      : allDocuments;
    const sorted = SORT_OPTIONS[sortChoice](filtered);

    if (sorted.length === 0) {
      libraryList.innerHTML = `<div class="empty-state">No documents match "${escapeHtml(searchQuery)}".</div>`;
      return;
    }

    libraryList.innerHTML = "";
    for (const doc of sorted) {
      libraryList.appendChild(renderDocumentRow(doc));
    }
  }

  function renderDocumentRow(doc) {
    const row = document.createElement("div");
    row.className = "card doc-row";

    const statusLabel = STATUS_BADGES[doc.status] ?? doc.status;
    const typeLabel = doc.file_type.toUpperCase();
    const sizeLabel = formatBytes(doc.file_size_bytes);
    const metaLine =
      doc.status === "ready" && doc.chunk_count != null
        ? `${typeLabel} · ${sizeLabel} · ${doc.chunk_count} chunks`
        : `${typeLabel} · ${sizeLabel}`;
    const showSummarize = doc.status === "ready";

    row.innerHTML = `
      <div class="doc-row-info">
        <p class="doc-row-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</p>
        <p class="text-muted doc-row-date">${formatUploadedAt(doc.uploaded_at)}</p>
      </div>
      <div class="doc-row-meta">
        <p>${statusLabel}</p>
        <p class="text-muted">${metaLine}</p>
      </div>
      <div class="doc-row-actions row">
        <button class="btn btn-icon" data-action="preview" title="Preview" aria-label="Preview ${escapeHtml(doc.filename)}">👁</button>
        ${showSummarize ? `<button class="btn btn-icon" data-action="summarize" title="Generate AI Summary" aria-label="Generate AI summary for ${escapeHtml(doc.filename)}">📝</button>` : ""}
        <button class="btn btn-icon btn-danger" data-action="delete" title="Delete" aria-label="Delete ${escapeHtml(doc.filename)}">🗑</button>
      </div>
    `;

    row.querySelector('[data-action="preview"]').addEventListener("click", () => openPreview(doc));
    row.querySelector('[data-action="delete"]').addEventListener("click", () => openDeleteConfirm(doc));
    row.querySelector('[data-action="summarize"]')?.addEventListener("click", () => openSummary(doc));

    return row;
  }

  // --- Preview modal ---------------------------------------------------
  async function openPreview(doc) {
    const { body } = openModal({ title: "Document preview", width: "large" });
    body.innerHTML = `
      <p><strong>${escapeHtml(doc.filename)}</strong></p>
      <div class="row"><span class="spinner"></span><span>Loading preview...</span></div>
    `;
    try {
      const { preview_text: previewText } = await documentsApi.preview(doc.id);
      body.innerHTML = `
        <p><strong>${escapeHtml(doc.filename)}</strong></p>
        <pre class="preview-text">${escapeHtml(previewText)}</pre>
      `;
    } catch (err) {
      body.innerHTML = `
        <p><strong>${escapeHtml(doc.filename)}</strong></p>
        <div class="alert alert-error">Preview unavailable: ${escapeHtml(err.message)}</div>
      `;
    }
  }

  // --- Delete confirmation modal -----------------------------------------
  function openDeleteConfirm(doc) {
    const { body, close } = openModal({ title: "Delete document" });
    body.innerHTML = `
      <p>Delete <strong>${escapeHtml(doc.filename)}</strong>? This will remove its extracted
      text, embeddings, and the original file. This cannot be undone.</p>
      <div class="row modal-footer">
        <button class="btn" data-action="cancel" style="flex: 1">Cancel</button>
        <button class="btn btn-danger" data-action="confirm" style="flex: 1">Delete</button>
      </div>
    `;
    body.querySelector('[data-action="cancel"]').addEventListener("click", close);
    body.querySelector('[data-action="confirm"]').addEventListener("click", async () => {
      const confirmButton = body.querySelector('[data-action="confirm"]');
      confirmButton.disabled = true;
      confirmButton.textContent = "Deleting...";
      try {
        await documentsApi.remove(doc.id);
        close();
        await loadDocuments();
      } catch (err) {
        confirmButton.disabled = false;
        confirmButton.textContent = "Delete";
        body.insertAdjacentHTML(
          "beforeend",
          `<div class="alert alert-error">Couldn't delete '${escapeHtml(doc.filename)}': ${escapeHtml(err.message)}</div>`
        );
      }
    });
  }

  // --- Summary modal -----------------------------------------------------
  function openSummary(doc) {
    const { body, close } = openModal({ title: "AI Summary", width: "large" });
    renderSummaryLoading(body, doc);
    loadSummary(body, doc, close);
  }

  function renderSummaryLoading(body, doc) {
    body.innerHTML = `
      <p><strong>${escapeHtml(doc.filename)}</strong></p>
      <div class="row"><span class="spinner"></span><span>Generating summary...</span></div>
    `;
  }

  async function loadSummary(body, doc, close) {
    try {
      const summary = await documentsApi.summarize(doc.id);
      body.innerHTML = `
        <p><strong>${escapeHtml(doc.filename)}</strong></p>
        <h4>Executive Summary</h4>
        <p>${escapeHtml(summary.executive_summary)}</p>
        ${
          summary.key_insights.length
            ? `<h4>Key Insights</h4><ul>${summary.key_insights.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>`
            : ""
        }
        ${summary.topics.length ? `<h4>Topics</h4><p>${escapeHtml(summary.topics.join(", "))}</p>` : ""}
        <div class="row modal-footer">
          <button class="btn" data-action="regenerate" style="flex: 1">🔄 Regenerate</button>
          <button class="btn" data-action="close" style="flex: 1">Close</button>
        </div>
      `;
      body.querySelector('[data-action="regenerate"]').addEventListener("click", () => {
        renderSummaryLoading(body, doc);
        loadSummary(body, doc, close);
      });
      body.querySelector('[data-action="close"]').addEventListener("click", close);
    } catch (err) {
      body.innerHTML = `
        <p><strong>${escapeHtml(doc.filename)}</strong></p>
        <div class="alert alert-error">Couldn't generate a summary: ${escapeHtml(err.message)}</div>
        <div class="row modal-footer">
          <button class="btn" data-action="close" style="flex: 1">Close</button>
        </div>
      `;
      body.querySelector('[data-action="close"]').addEventListener("click", close);
    }
  }

  loadDocuments();

  return function dispose() {
    disposed = true;
  };
}
