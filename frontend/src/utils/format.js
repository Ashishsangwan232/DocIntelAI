/**
 * utils/format.js
 * ================
 * Ported from src/utils/helpers.py so both frontends render the
 * exact same strings for the same values.
 */

/** Mirrors `format_bytes()` in src/utils/helpers.py exactly. */
export function formatBytes(sizeBytes) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = sizeBytes;
  for (const unit of units) {
    if (size < 1024) {
      return unit === "B" ? `${Math.trunc(size)} ${unit}` : `${size.toFixed(1)} ${unit}`;
    }
    size /= 1024;
  }
  return `${size.toFixed(1)} PB`;
}

/** Mirrors `_format_uploaded_at()` in src/ui/upload.py: "Uploaded Mar 4, 2026 at 14:30". */
export function formatUploadedAt(isoString) {
  const date = new Date(isoString);
  const datePart = date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  const timePart = date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  return `Uploaded ${datePart} at ${timePart}`;
}

/** Escapes text for safe insertion into innerHTML — every dynamic string in the app goes through this. */
export function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

/**
 * Renders a search snippet safely: the raw text is escaped first
 * (neutralizing any literal HTML in the source document), then the
 * `**term**` markers `SearchService._build_snippet()` wraps matches in
 * are converted to `<mark>` — never the other way around, so there's
 * no way for document content to inject real HTML.
 */
export function renderHighlightedSnippet(snippet) {
  const escaped = escapeHtml(snippet);
  return escaped.replace(/\*\*(.+?)\*\*/g, "<mark>$1</mark>");
}

/**
 * Mirrors `render_match_dial()` in src/ui/theme.py exactly: a small
 * conic-gradient ring filled to the similarity percentage, paired
 * with a text label (never color-only — the percentage is always
 * spelled out too, for accessibility).
 */
export function renderMatchDial(similarityScore, label) {
  const percent = Math.round(Math.max(0, Math.min(1, similarityScore)) * 100);
  const text = label ?? `${percent}% match`;
  return (
    `<span class="match-dial-wrap">` +
    `<span class="match-dial" style="--pct:${percent};" role="img" aria-label="${percent} percent match"></span>` +
    `<span>${escapeHtml(text)}</span>` +
    `</span>`
  );
}
