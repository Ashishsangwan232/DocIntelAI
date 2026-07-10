# Deploying DocIntel AI to Render

This guide covers two deployment paths:

- **Quick / demo deployment** — no persistent disk, free-tier friendly. Uploaded documents and chat history are lost on redeploy or restart. Good for a portfolio demo link.
- **Production deployment** — adds a Render Persistent Disk so SQLite, ChromaDB, and uploaded files survive restarts.

Both use the same codebase — the only difference is one Render setting.

---

## 1. Prerequisites

- A [Render](https://render.com) account
- This repository pushed to GitHub (or GitLab/Bitbucket)
- An Ollama Cloud API key ([ollama.com](https://ollama.com))

---

## 2. Quick Deployment (No Persistent Disk)

1. **Push this repository to GitHub.**

2. **In Render, click New → Web Service** and connect your repository.

3. **Configure the service:**

   | Setting | Value |
   |---|---|
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true` |
   | Instance Type | Free (or Starter for better performance) |

4. **Set environment variables** (Render dashboard → Environment tab). At minimum:

   ```
   OLLAMA_CLOUD_API_KEY=your_real_key_here
   LLM_MODEL=gpt-oss:120b
   APP_ENV=production
   DEBUG=false
   ```

   All other variables in `.env.example` have sensible defaults baked into `config.py` and don't need to be set unless you want to override them.

5. **Deploy.** Render will build and start the app. First load will be slower (downloads the `BAAI/bge-base-en-v1.5` embedding model, ~430MB, on first use).

**Important limitation:** Render's default filesystem is ephemeral — every redeploy or restart wipes `uploads/`, `database/`, and `vectorstore/`. Uploaded documents and chat history will not persist. This is fine for a live demo link where you upload a sample document each session, but not for real usage.

---

## 3. Production Deployment (Persistent Disk)

Same as above, plus:

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

## 4. Using render.yaml (Infrastructure as Code)

This repo includes a `render.yaml` at the project root. In Render, choose **New → Blueprint**, point it at this repository, and Render will provision the service from that file automatically — you'll still need to set `OLLAMA_CLOUD_API_KEY` manually in the dashboard (secrets are never committed to `render.yaml`).

---

## 5. Docker Deployment (Alternative)

A `docker/Dockerfile` is included if you'd rather deploy as a container (Render, Fly.io, Railway, or any container host all work identically):

```bash
docker build -f docker/Dockerfile -t docintel-ai .
docker run -p 8501:8501 --env-file .env docintel-ai
```

On Render specifically: choose **New → Web Service**, select **Docker** as the runtime, and point it at `docker/Dockerfile`. Mount a persistent disk the same way as the non-Docker path above if you need data to survive restarts.

---

## 6. Post-Deployment Checklist

- [ ] Visit the deployed URL and confirm the app loads without the "OLLAMA_CLOUD_API_KEY is not set" warning
- [ ] Upload a small test document and confirm it processes to `READY`
- [ ] Ask a question in Chat and confirm you get a cited answer
- [ ] Check Render's logs tab if anything fails — `src/utils/logger.py` writes structured logs to both console and `logs/docintel.log`
- [ ] If using a persistent disk, restart the service and confirm your test document is still there

---

## 7. Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| "OLLAMA_CLOUD_API_KEY is not configured" | Env var not set or misspelled | Check Render → Environment tab |
| App works, then loses all documents after a while | No persistent disk | Follow section 3 |
| First request very slow | Embedding model downloading on cold start | Expected once per deploy; consider a paid instance type to avoid cold starts entirely |
| `chromadb` install fails on build | Some Render base images lack build tools for native deps | Use the Docker deployment path instead, which pins a compatible base image |
