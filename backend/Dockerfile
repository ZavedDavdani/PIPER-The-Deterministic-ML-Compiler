# PIPER backend image (Batch 4).
#
# Build context is the REPO ROOT (see docker-compose.yml's
# `build.context: .` for the backend service), not backend/ — the only
# requirements.txt in this project lives at the repo root (shared by
# every M1-M5 milestone's tests), and duplicating it under backend/
# would risk the two drifting apart. Build from the root with:
#
#   docker build -f backend/Dockerfile -t piper-backend .
#
# (docker-compose.yml does this automatically.)
#
# No Ollama here — PIPER_OLLAMA_HOST is configured (via
# docker-compose.yml) to reach the HOST machine's already-running
# Ollama installation through Docker Desktop's host.docker.internal,
# never bundled into this image. See CLAUDE.md's Dockerization section.

FROM python:3.11-slim

WORKDIR /srv/piper

# --- Python dependencies (cached separately from app code) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code only — no tests, no data/, no .venv ---
# (see backend/.dockerignore for the exact exclusions)
COPY backend/app ./app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Pure-Python healthcheck (no curl/wget dependency added to the image
# just for this) against the real GET /health endpoint app/main.py
# already exposes.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
