/**
 * app.js
 * ======
 * The persistent shell: sidebar + a `<main id="page-outlet">` the
 * router swaps content into on every navigation. This never
 * re-renders itself after mount — only `page-outlet`'s children
 * change, which is what makes the sidebar's connection status and
 * active-link highlighting stable across navigations.
 */

import { renderSidebar } from "./components/sidebar.js";

export function mountApp(root) {
  const skipLink = document.createElement("a");
  skipLink.className = "skip-link";
  skipLink.href = "#page-outlet";
  skipLink.textContent = "Skip to content";
  root.appendChild(skipLink);

  const sidebar = renderSidebar();
  sidebar.setAttribute("role", "navigation");
  sidebar.setAttribute("aria-label", "Main");
  root.appendChild(sidebar);

  const main = document.createElement("main");
  main.id = "page-outlet";
  main.className = "page-outlet";
  main.tabIndex = -1;
  root.appendChild(main);
}
