# Deploying DocIntel AI to Render

This guide covers the app's one deployable stack: **FastAPI + React** — one Render service serves both the REST/SSE API (`api/`) and the built frontend (`frontend/`) from a single process and origin, defined by `render.yaml`'s `docintel-ai-web` service.

---

## 0. Prerequisites

- A [Render](https://render.com) account
- This repository pushed to GitHub (or GitLab/Bitbucket)
- An Ollama Cloud API key ([ollama.com](https://ollama.com))

---

## 1. FastAPI + React (one service)

### Quick deployment

1. **Push this repository to GitHub.**

2. **In Render, click New → Web Service** and connect your repository.

3. **Configure the service:**

   | Setting | Value |
   |---|---|
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt && cd frontend && npm install && npm run build && cd ..` |
   | Start Command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
   | Instance Type | Free (or Starter for better performance) |

4. **Set environment variables** (Render dashboard → Environment tab):

   ```
   OLLAMA_CLOUD_API_KEY=your_real_key_here
   OLLAMA_CLOUD_BASE_URL=https://ollama.com
   LLM_MODEL=gpt-oss:120b-cloud
   APP_ENV=production
   DEBUG=false
   ```

   All other variables in `.env.example` have sensible defaults baked into `config.py` and don't need to be set unless you want to override them.

5. **Deploy.** The build step installs Python deps, then builds the React frontend into `frontend/dist/`, which `api/static.py` serves directly — there's no separate frontend hosting step, and no CORS configuration needed in production since the browser only ever talks to this one origin. First load will be slower (downloads the `BAAI/bge-base-en-v1.5` embedding model, ~430MB, on first use).

6. **Verify:** visit the deployed URL — you should see the React app. Visiting `<url>/docs` shows the interactive API documentation, and `<url>/api/v1/health` should return `{"status": "ok", ...}`.

**Important limitation:** Render's default filesystem is ephemeral — every redeploy or restart wipes `uploads/`, `database/`, and `vectorstore/`. Uploaded documents and chat history will not persist. This is fine for a live demo link where you upload a sample document each session, but not for real usage — see section 2 below for the persistent-disk setup.

### Local development

You need two processes running together — see `frontend/README.md` for the full walkthrough:

```bash
# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
cd frontend && npm install && npm run dev
```

Vite's dev server (`http://localhost:5173`) proxies `/api/*` to the FastAPI process, so there's no CORS involved locally either.

---

## 2. Production Deployment (Persistent Disk)

In addition to the quick-deployment steps above:

1. In the Render service settings, go to **Disks** and click **Add Disk**.

2. Configure:

   | Setting | Value |
   |---|---|
   | Name | `docintel-data` |
   | Mount Path | `/opt/render/project/src/data` |
   | Size | 1 GB (increase as your document library grows) |

3. **Set additional environment variables** to redirect storage onto the mounted disk:

   ```
   SQLITE_DB_NAME=/opt/render/project/src/data/docintel.db
   ```

   Then edit `config.py`'s `PathSettings` (or override via env vars — the cleanest approach is adding `UPLOAD_DIR`, `VECTORSTORE_DIR`, `DATABASE_DIR` env var overrides if you want this fully configurable; as shipped, the simplest change is pointing `BASE_DIR`-relative paths at the mounted disk by setting a `DATA_DIR` environment variable and adjusting `config.py`'s `PathSettings` to read from it). At minimum, `database/`, `vectorstore/`, and `uploads/` all need to live under the mounted disk path for persistence to actually work — a symlink from each expected path to the disk mount is the fastest way to retrofit this without code changes:

   ```bash
   # As a Render "Pre-Deploy Command" or one-time shell command:
   mkdir -p /opt/render/project/src/data/{database,vectorstore,uploads,logs}
   ln -sfn /opt/render/project/src/data/database /opt/render/project/src/database
   ln -sfn /opt/render/project/src/data/vectorstore /opt/render/project/src/vectorstore
   ln -sfn /opt/render/project/src/data/uploads /opt/render/project/src/uploads
   ln -sfn /opt/render/project/src/data/logs /opt/render/project/src/logs
   ```

4. Redeploy. Data now survives restarts and redeploys.

---

## 3. Using render.yaml (Infrastructure as Code)

This repo includes a `render.yaml` at the project root defining the `docintel-ai-web` service. In Render, choose **New → Blueprint**, point it at this repository, and Render will provision it from that file automatically. You'll still need to set `OLLAMA_CLOUD_API_KEY` manually in the dashboard (secrets are never committed to `render.yaml`).

---

## 4. Post-Deployment Checklist

- [ ] Visit the deployed URL and confirm the app loads without the "OLLAMA_CLOUD_API_KEY is not set" warning (check `<url>/api/v1/health/config`)
- [ ] Upload a small test document and confirm it processes to `READY`
- [ ] Ask a question in Chat and confirm you get a cited answer
- [ ] Check Render's logs tab if anything fails — `src/utils/logger.py` writes structured logs to both console and `logs/docintel.log`
- [ ] If using a persistent disk, restart the service and confirm your test document is still there

---

## 5. Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| "OLLAMA_CLOUD_API_KEY is not configured" | Env var not set or misspelled | Check Render → Environment tab |
| App works, then loses all documents after a while | No persistent disk | Follow section 2 |
| First request very slow | Embedding model downloading on cold start | Expected once per deploy; consider a paid instance type to avoid cold starts entirely |
| `chromadb` install fails on build | Some Render base images lack build tools for native deps | Try a Starter (or higher) instance type, which uses a build image with more complete tooling |
| Root URL returns a 404 | `frontend/dist/` wasn't built — check the build command actually ran `npm run build` and didn't fail silently | Check Render's build logs for `npm` errors; `api/static.py` logs "frontend/dist not found" on startup if the mount was skipped |
| Page works, but a hard refresh on `/documents` (or any non-root path) 404s | `api/static.py`'s SPA fallback isn't registered — usually means `frontend/dist/index.html` is missing | Same fix as above: confirm the frontend build actually produced `dist/` |
