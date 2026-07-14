/**
 * router.js
 * =========
 * A deliberately small History-API router — four pages, no nesting,
 * no third-party dependency earns its weight here.
 *
 * Contract: register a path -> `render(container)` function. `render`
 * receives the empty `<main>` outlet and is responsible for its own
 * content and its own cleanup (returning an optional `dispose()`
 * function, called before the next navigation swaps it out — pages
 * with a live SSE connection or polling interval use this to avoid
 * leaking one every time the user clicks away mid-stream).
 */

const routes = new Map();
let currentDispose = null;

export function registerRoute(path, render) {
  routes.set(path, render);
}

export function navigate(path, { replace = false } = {}) {
  if (window.location.pathname === path) return;
  if (replace) window.history.replaceState({}, "", path);
  else window.history.pushState({}, "", path);
  renderCurrentRoute();
}

function renderCurrentRoute() {
  const path = window.location.pathname;
  const render = routes.get(path) ?? routes.get("/");

  if (typeof currentDispose === "function") {
    currentDispose();
    currentDispose = null;
  }

  const outlet = document.getElementById("page-outlet");
  outlet.innerHTML = "";
  outlet.scrollTop = 0;
  currentDispose = render(outlet) ?? null;

  // Retrigger the fade transition: outlet.innerHTML was replaced, but
  // the outlet element itself was never removed/re-inserted, so a CSS
  // animation class only plays if we force it — remove, reflow, add.
  outlet.classList.remove("page-transition");
  void outlet.offsetWidth;
  outlet.classList.add("page-transition");

  // Move focus to the new page's heading so screen readers announce
  // the navigation — the router swaps content without a real page
  // load, so nothing does this automatically the way it would for a
  // full browser navigation.
  const heading = outlet.querySelector("h1");
  if (heading) {
    heading.setAttribute("tabindex", "-1");
    heading.focus({ preventScroll: true });
  }

  document.querySelectorAll("[data-nav-link]").forEach((el) => {
    el.classList.toggle("active", el.getAttribute("href") === path);
  });
}

export function initRouter() {
  document.addEventListener("click", (event) => {
    const link = event.target.closest("[data-link]");
    if (!link) return;
    event.preventDefault();
    navigate(link.getAttribute("href"));
  });

  window.addEventListener("popstate", renderCurrentRoute);
  renderCurrentRoute();
}
