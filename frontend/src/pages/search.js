/**
 * pages/search.js
 * ===============
 * Full semantic search page — feature parity with `src/ui/search.py`:
 * retrieval only, no LLM call, ranked by similarity with highlighted
 * matching terms.
 *
 * Search is debounced (250ms) and guarded against out-of-order
 * responses: if the user types faster than the network responds, an
 * older request resolving after a newer one must not overwrite the
 * newer results. Streamlit's rerun model made this a non-issue (each
 * rerun blocks until its own search call returns); a hand-rolled async
 * page has to guard for it explicitly.
 */

import { documents as documentsApi, search as searchApi } from "../api/resources.js";
import { createDocumentScopeSelect } from "../components/documentScopeSelect.js";
import { escapeHtml, renderHighlightedSnippet, renderMatchDial } from "../utils/format.js";

const DEBOUNCE_MS = 250;

export function renderSearchPage(container) {
  let documentScope = [];
  let lastQuery = null;
  let requestToken = 0;
  let debounceTimer = null;

  container.innerHTML = `
    <div class="page-content stack">
      <div class="page-header">
        <h1>🔍 Semantic Search</h1>
        <p class="text-muted">
          Search your documents by meaning, not just keywords. Results are ranked by
          similarity — no AI-generated answer, just the source material.
        </p>
      </div>

      <div class="card stack">
        <div id="scope-container"></div>
        <input
          type="text" id="search-input" class="input"
          placeholder="Search your documents..." autocomplete="off"
        />
      </div>

      <div id="search-results" class="stack"></div>
    </div>
  `;

  const scopeContainer = container.querySelector("#scope-container");
  const searchInput = container.querySelector("#search-input");
  const resultsEl = container.querySelector("#search-results");

  const scopeSelect = createDocumentScopeSelect({
    onChange: (ids) => {
      documentScope = ids;
      // Scope changed — re-run the current query (if any) against the new scope.
      if (searchInput.value.trim()) runSearch(searchInput.value.trim());
    },
  });
  scopeContainer.appendChild(scopeSelect);

  documentsApi.list({ status: "ready" }).then((readyDocs) => {
    if ((readyDocs ?? []).length === 0) {
      resultsEl.innerHTML = `<div class="empty-state">No processed documents yet — upload one to start searching.</div>`;
    }
  }).catch(() => {});

  searchInput.addEventListener("input", () => {
    const value = searchInput.value.trim();
    clearTimeout(debounceTimer);

    if (!value) {
      lastQuery = null;
      resultsEl.innerHTML = "";
      return;
    }

    debounceTimer = setTimeout(() => runSearch(value), DEBOUNCE_MS);
  });

  async function runSearch(query) {
    if (query === lastQuery) return;
    lastQuery = query;

    const thisRequest = ++requestToken;
    resultsEl.innerHTML = `<div class="row"><span class="spinner"></span><span>Searching...</span></div>`;

    let hits;
    try {
      const response = await searchApi.run(query, {
        documentIds: documentScope.length ? documentScope : undefined,
      });
      hits = response.hits;
    } catch (err) {
      if (thisRequest !== requestToken) return; // a newer search superseded this one
      resultsEl.innerHTML = `<div class="alert alert-error">Search failed: ${escapeHtml(err.message)}</div>`;
      return;
    }

    if (thisRequest !== requestToken) return; // stale response, a newer search is in flight

    renderResults(hits, query);
  }

  function renderResults(hits, query) {
    if (hits.length === 0) {
      resultsEl.innerHTML = `<div class="empty-state">No results found for '${escapeHtml(query)}'. Try rephrasing your search.</div>`;
      return;
    }

    const count = hits.length;
    resultsEl.innerHTML = `<p class="text-muted">${count} result${count !== 1 ? "s" : ""} for '${escapeHtml(query)}'</p>`;
    for (const hit of hits) {
      resultsEl.appendChild(renderHit(hit));
    }
  }

  function renderHit(hit) {
    const card = document.createElement("div");
    card.className = "card search-hit";
    const pageSuffix = hit.page_number ? `, page ${hit.page_number}` : "";

    card.innerHTML = `
      <div class="search-hit-body">
        <p class="search-hit-title"><strong>${escapeHtml(hit.filename)}${pageSuffix}</strong></p>
        <p class="search-hit-snippet">${renderHighlightedSnippet(hit.snippet)}</p>
      </div>
      <div class="search-hit-score">
        ${renderMatchDial(hit.similarity_score, "")}
        <p class="text-muted" style="margin: 4px 0 0 0">${Math.round(hit.similarity_score * 100)}% match</p>
      </div>
    `;
    return card;
  }

  return function dispose() {
    clearTimeout(debounceTimer);
  };
}
