# DocIntel AI — Frontend (Vite + vanilla JS)

The in-progress React-migration frontend, replacing the Streamlit UI
in `../src/ui/` page by page. Talks exclusively to the FastAPI backend
in `../api/`.

## Stack

Vite, vanilla JavaScript (ES modules), vanilla CSS. No framework, no
TypeScript, no CSS/component library — a hand-rolled History-API
router and a small `fetch`-based API client are the only "framework"
code here, both in `src/`.

## Structure

```
src/
├── main.js              # entry point — wires shell + routes + router
├── app.js                # persistent shell (sidebar + page outlet)
├── router.js              # History API router
├── api/
│   ├── client.js          # fetch wrapper: JSON, uploads, SSE streaming, downloads
│   └── resources.js        # one named function per backend endpoint
├── components/
│   └── sidebar.js
├── pages/
│   ├── chat.js             # Phase G
│   ├── search.js           # Phase H
│   ├── documents.js        # Phase F
│   ├── analytics.js        # Phase I
│   └── comingSoon.js       # shared placeholder for not-yet-built pages
└── styles/
    ├── tokens.css           # design tokens, ported 1:1 from ../assets/styles.css
    ├── base.css             # reset, typography, scrollbar, focus ring
    ├── components.css       # cards, buttons, inputs, chat bubbles, match-dial, sidebar nav
    └── main.css              # imports the three above, in order
```

## Running locally

You need the FastAPI backend running first (from the project root):

```bash
uvicorn api.main:app --reload --port 8000
```

Then, in this directory:

```bash
npm install
npm run dev
```

Open the printed `http://localhost:5173` URL. The dev server proxies
`/api/*` to `http://localhost:8000` (see `vite.config.js`), so the
browser only ever talks to Vite's own origin — no CORS involved in
local development at all.

## Building for production

```bash
npm run build
```

Outputs to `dist/`. In production (Phase K), FastAPI serves this
`dist/` directory as static files alongside the API, from one process
and one origin.

## Adding a page

1. Write `src/pages/<name>.js` exporting `render(container)`, which
   receives the empty `<main>` outlet.
2. If the page needs to clean up on navigation-away (an open SSE
   connection, a polling interval), have `render` return a `dispose`
   function — the router calls it automatically before swapping in
   the next page.
3. Replace the corresponding `renderComingSoon(...)` call with your
   new render function in `src/main.js`. Nothing else changes — the
   router, sidebar, and CSS are already wired up.
4. Talk to the backend only through `src/api/resources.js` — add a
   function there if the endpoint you need isn't already wrapped.
