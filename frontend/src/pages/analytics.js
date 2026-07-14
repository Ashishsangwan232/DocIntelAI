/**
 * pages/analytics.js
 * ===================
 * Full Analytics + Settings page — feature parity with
 * `src/ui/analytics.py`: usage dashboard (8 metrics, upload-activity
 * chart, most-queried ranking) and a Settings form (chunking,
 * retrieval, LLM generation parameters).
 *
 * Presented as two sub-tabs on one page, same rationale as the
 * Streamlit version: both are small, both read from the same
 * services, and a dedicated route for Settings would be more
 * ceremony than value.
 */

import { analytics as analyticsApi, settings as settingsApi } from "../api/resources.js";
import { showToast } from "../components/toast.js";
import { escapeHtml } from "../utils/format.js";

export function renderAnalyticsPage(container) {
  let activeTab = "analytics";
  let disposed = false;

  container.innerHTML = `
    <div class="page-content stack">
      <div class="page-header">
        <h1>📈 Analytics & Settings</h1>
        <p class="text-muted">Usage dashboard and retrieval/generation preferences.</p>
      </div>
      <div class="subtabs">
        <button class="subtab-btn active" data-tab="analytics">📊 Analytics</button>
        <button class="subtab-btn" data-tab="settings">⚙️ Settings</button>
      </div>
      <div id="tab-content"></div>
    </div>
  `;

  const tabButtons = container.querySelectorAll(".subtab-btn");
  const tabContent = container.querySelector("#tab-content");

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.tab === activeTab) return;
      activeTab = button.dataset.tab;
      tabButtons.forEach((b) => b.classList.toggle("active", b === button));
      renderActiveTab();
    });
  });

  function renderActiveTab() {
    if (disposed) return;
    if (activeTab === "analytics") renderAnalyticsTab(tabContent);
    else renderSettingsTab(tabContent);
  }

  renderActiveTab();

  return function dispose() {
    disposed = true;
  };
}

// ----------------------------------------------------------------------
// Analytics tab
// ----------------------------------------------------------------------
async function renderAnalyticsTab(content) {
  content.innerHTML = `<div class="row"><span class="spinner"></span><span>Loading analytics...</span></div>`;

  let summary;
  try {
    summary = await analyticsApi.getSummary();
  } catch (err) {
    content.innerHTML = `<div class="alert alert-error">Couldn't load analytics: ${escapeHtml(err.message)}</div>`;
    return;
  }

  const mostQueriedName = summary.most_queried_documents[0]?.filename ?? "N/A";

  content.innerHTML = `
    <div class="stack">
      <div class="grid-metrics">
        ${metricCard("Total Documents", summary.total_documents)}
        ${metricCard("Total Chunks", summary.total_chunks)}
        ${metricCard("Total Embeddings", summary.total_embeddings)}
        ${metricCard("Storage Used", summary.total_storage_display)}
      </div>
      <div class="grid-metrics">
        ${metricCard("Chat Sessions", summary.total_chat_sessions)}
        ${metricCard("Total Queries", summary.total_queries)}
        ${metricCard("Avg. Response Time", summary.average_response_time_display)}
        ${metricCard("Most Queried", mostQueriedName)}
      </div>

      <div class="row" style="align-items: stretch; flex-wrap: wrap">
        <div class="card" style="flex: 1; min-width: 280px">
          <h3>Upload Activity (last 14 days)</h3>
          <div id="upload-chart"></div>
        </div>
        <div class="card" style="flex: 1; min-width: 280px">
          <h3>Most Queried Documents</h3>
          <div id="most-queried-list"></div>
        </div>
      </div>
    </div>
  `;

  content.querySelector("#upload-chart").appendChild(renderUploadActivityChart(summary.uploads_per_day));
  content.querySelector("#most-queried-list").appendChild(
    renderMostQueriedList(summary.most_queried_documents)
  );
}

function metricCard(label, value) {
  return `
    <div class="metric">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(String(value))}</div>
    </div>
  `;
}

function renderMostQueriedList(entries) {
  const wrapper = document.createElement("div");
  if (entries.length === 0) {
    wrapper.innerHTML = `<p class="text-muted">No queries logged yet.</p>`;
    return wrapper;
  }
  wrapper.className = "stack";
  wrapper.innerHTML = entries
    .map(
      (entry) => `
        <div>
          <p style="margin: 0"><strong>${escapeHtml(entry.filename)}</strong></p>
          <p class="text-muted" style="margin: 0">
            ${entry.query_count} quer${entry.query_count === 1 ? "y" : "ies"}
          </p>
        </div>
      `
    )
    .join("");
  return wrapper;
}

/** Plain inline SVG bar chart — no charting library for a handful of daily counts. */
function renderUploadActivityChart(uploadsPerDay) {
  const wrapper = document.createElement("div");

  if (!uploadsPerDay || uploadsPerDay.length === 0) {
    wrapper.innerHTML = `<p class="text-muted">No uploads yet in this period.</p>`;
    return wrapper;
  }

  const width = 280;
  const height = 140;
  const padding = { top: 8, bottom: 22, left: 4, right: 4 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const maxCount = Math.max(...uploadsPerDay.map(([, count]) => count), 1);
  const barGap = 4;
  const barWidth = plotWidth / uploadsPerDay.length - barGap;

  const bars = uploadsPerDay
    .map(([day, count], index) => {
      const barHeight = (count / maxCount) * plotHeight;
      const x = padding.left + index * (barWidth + barGap);
      const y = padding.top + (plotHeight - barHeight);
      const label = day.slice(5); // "MM-DD"
      return `
        <rect class="bar-chart-bar" x="${x}" y="${y}" width="${barWidth}" height="${Math.max(barHeight, 1)}" rx="2">
          <title>${escapeHtml(day)}: ${count} upload${count !== 1 ? "s" : ""}</title>
        </rect>
        ${
          count > 0
            ? `<text class="bar-chart-value" x="${x + barWidth / 2}" y="${y - 3}" text-anchor="middle">${count}</text>`
            : ""
        }
        <text class="bar-chart-label" x="${x + barWidth / 2}" y="${height - 6}" text-anchor="middle">${label}</text>
      `;
    })
    .join("");

  wrapper.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto" role="img" aria-label="Uploads per day, last 14 days">
      ${bars}
    </svg>
  `;
  return wrapper;
}

// ----------------------------------------------------------------------
// Settings tab
// ----------------------------------------------------------------------
async function renderSettingsTab(content) {
  content.innerHTML = `<div class="row"><span class="spinner"></span><span>Loading settings...</span></div>`;

  let current;
  let models;
  try {
    [current, models] = await Promise.all([
      settingsApi.get(),
      settingsApi.listModels().then((r) => r.models),
    ]);
  } catch (err) {
    content.innerHTML = `<div class="alert alert-error">Couldn't load current settings: ${escapeHtml(err.message)}</div>`;
    return;
  }

  content.innerHTML = `
    <div class="card stack">
      <div>
        <p class="text-muted">
          Changes to Top-K, Temperature, Max Tokens, and Model apply to your next chat
          message. Chunk Size/Overlap apply to documents uploaded after saving —
          already-processed documents are unaffected.
        </p>
      </div>

      <div>
        <h3>Document Chunking</h3>
        <div class="settings-grid">
          ${sliderField("chunk-size", "Chunk Size (characters)", 100, 4000, 50, current.chunk_size)}
          ${sliderField("chunk-overlap", "Chunk Overlap (characters)", 0, current.chunk_size - 1, 10, current.chunk_overlap)}
        </div>
      </div>

      <div>
        <h3>Retrieval</h3>
        ${sliderField("top-k", "Top-K Retrieval", 1, 20, 1, current.top_k)}
      </div>

      <div>
        <h3>LLM Generation</h3>
        <div class="settings-grid">
          ${sliderField("temperature", "Temperature", 0, 2, 0.1, current.temperature)}
          ${sliderField("max-tokens", "Maximum Tokens", 64, 8192, 64, current.max_tokens)}
        </div>
        <div class="field">
          <label class="field-label" for="model-select">Model</label>
          <select id="model-select" class="select">
            ${models.map((m) => `<option value="${escapeHtml(m)}" ${m === current.model ? "selected" : ""}>${escapeHtml(m)}</option>`).join("")}
          </select>
        </div>
      </div>

      <div class="row">
        <button class="btn btn-primary" id="save-settings-btn" style="flex: 1">Save Settings</button>
        <button class="btn" id="reset-settings-btn" style="flex: 1">Reset to Defaults</button>
      </div>
      <div id="settings-error"></div>
    </div>
  `;

  wireSlider(content, "chunk-size");
  wireSlider(content, "chunk-overlap");
  wireSlider(content, "top-k");
  wireSlider(content, "temperature");
  wireSlider(content, "max-tokens");

  // Chunk overlap's valid range depends on chunk size — re-clamp live,
  // mirroring the Streamlit slider's dynamic `max_value=chunk_size - 1`.
  const chunkSizeInput = content.querySelector("#chunk-size-input");
  const chunkOverlapInput = content.querySelector("#chunk-overlap-input");
  chunkSizeInput.addEventListener("input", () => {
    const newMax = Math.max(Number(chunkSizeInput.value) - 1, 1);
    chunkOverlapInput.max = newMax;
    if (Number(chunkOverlapInput.value) > newMax) {
      chunkOverlapInput.value = newMax;
      chunkOverlapInput.dispatchEvent(new Event("input"));
    }
  });

  const errorBox = content.querySelector("#settings-error");
  const saveBtn = content.querySelector("#save-settings-btn");
  const resetBtn = content.querySelector("#reset-settings-btn");

  saveBtn.addEventListener("click", async () => {
    saveBtn.disabled = true;
    errorBox.innerHTML = "";
    try {
      await settingsApi.update({
        chunk_size: Number(content.querySelector("#chunk-size-input").value),
        chunk_overlap: Number(content.querySelector("#chunk-overlap-input").value),
        top_k: Number(content.querySelector("#top-k-input").value),
        temperature: Number(content.querySelector("#temperature-input").value),
        max_tokens: Number(content.querySelector("#max-tokens-input").value),
        model: content.querySelector("#model-select").value,
      });
      showToast("Settings saved.");
    } catch (err) {
      errorBox.innerHTML = `<div class="alert alert-error">Couldn't save settings: ${escapeHtml(err.message)}</div>`;
    } finally {
      saveBtn.disabled = false;
    }
  });

  resetBtn.addEventListener("click", async () => {
    resetBtn.disabled = true;
    try {
      await settingsApi.reset();
      showToast("Settings reset to defaults.");
      await renderSettingsTab(content); // reload with fresh defaults
    } catch (err) {
      errorBox.innerHTML = `<div class="alert alert-error">Couldn't reset settings: ${escapeHtml(err.message)}</div>`;
      resetBtn.disabled = false;
    }
  });
}

function sliderField(id, label, min, max, step, value) {
  return `
    <div class="slider-field">
      <div class="slider-field-header">
        <label for="${id}-input">${escapeHtml(label)}</label>
        <span class="slider-field-value" id="${id}-value">${value}</span>
      </div>
      <input type="range" id="${id}-input" min="${min}" max="${max}" step="${step}" value="${value}" />
    </div>
  `;
}

function wireSlider(content, id) {
  const input = content.querySelector(`#${id}-input`);
  const valueLabel = content.querySelector(`#${id}-value`);
  input.addEventListener("input", () => {
    valueLabel.textContent = input.value;
  });
}
