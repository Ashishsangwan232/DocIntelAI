# DocIntel AI — Container image
# Build from the project root: docker build -f docker/Dockerfile -t docintel-ai .

FROM python:3.12-slim

# System dependencies for PyMuPDF (PDF parsing) and chromadb's native components.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime directories (also created automatically by config.py at
# startup, but declared here so volume mounts have somewhere to land).
RUN mkdir -p uploads database vectorstore logs

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
