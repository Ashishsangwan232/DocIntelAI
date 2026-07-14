import { defineConfig } from "vite";

// Dev-server proxy means the frontend never needs CORS at all in local
// development: the browser only ever talks to Vite's own origin
// (http://localhost:5173), and Vite forwards /api/* to FastAPI
// server-side. In production, this same relative "/api/v1" path works
// unchanged because FastAPI serves the built `dist/` and the API from
// one process/origin (see docs/DEPLOYMENT.md, Phase K).
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
