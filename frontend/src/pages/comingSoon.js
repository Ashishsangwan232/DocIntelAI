/**
 * pages/comingSoon.js
 * ====================
 * Placeholder for a page whose real implementation lands in a later
 * migration phase. Each real page (Phases F-I) replaces its call site
 * in main.js's route table — nothing else changes.
 */

export function renderComingSoon(container, { title, phase, description }) {
  container.innerHTML = `
    <div class="page-content">
      <div class="empty-state card">
        <h2>${title}</h2>
        <p>${description}</p>
        <p class="text-muted">Arriving in ${phase} of the Streamlit → React migration.</p>
      </div>
    </div>
  `;
}
