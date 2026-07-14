/**
 * components/sidebar.js
 * ======================
 * Same 4 destinations, same order, as `src/ui/sidebar.py`'s Streamlit
 * nav — Chat, Search, Documents, Analytics & Settings — so the two
 * frontends stay a 1:1 mental map during the migration.
 */

const NAV_ITEMS = [
  { path: "/", label: "Chat", icon: "💬" },
  { path: "/search", label: "Search", icon: "🔍" },
  { path: "/documents", label: "Documents", icon: "📁" },
  { path: "/analytics", label: "Analytics & Settings", icon: "📈" },
];

export function renderSidebar() {
  const aside = document.createElement("aside");
  aside.className = "sidebar";
  aside.innerHTML = `
    <p class="sidebar-brand-title">🧠 DocIntel AI</p>
    <p class="sidebar-brand-subtitle">AI-Powered Document Intelligence</p>
    ${NAV_ITEMS.map(
      (item) => `
      <a class="nav-link" href="${item.path}" data-link data-nav-link>
        <span aria-hidden="true">${item.icon}</span>
        <span>${item.label}</span>
      </a>`
    ).join("")}
  `;
  return aside;
}
