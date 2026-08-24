"""
FastAPI application entry point (M5).

Wires the EXISTING core (DatasetStore/SplitStore/InMemoryModelStore/
InMemoryRunStore, build_graph(), stream_with_tracing(), OllamaProvider)
into HTTP endpoints — no agent/ML logic lives here or in app/api/.
Every store and the LLM provider are created once, at process startup,
and shared for the process's lifetime via app.state (in-memory, no
persistence across restarts — matches every store's own documented
behavior; this is a local/demo deployment target, not a multi-worker
production one).

Run locally with:

    uvicorn app.main:app --reload

The real OllamaProvider is the default llm_provider — same
environment-variable-driven configuration used everywhere else in this
codebase (PIPER_OLLAMA_HOST / PIPER_LLM_MODEL /
PIPER_OLLAM_TIMEOUT_SECONDS), never hardcoded here. Tests override
app.state.llm_provider via FastAPI's dependency_overrides (see
tests/test_api_runs.py) rather than talking to a real Ollama server —
consistent with the rest of the suite's "the normal suite is never
Ollama-dependent" rule.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.datasets import router as datasets_router
from app.api.routers.learning import router as learning_router
from app.api.routers.predict import router as predict_router
from app.api.routers.runs import router as runs_router
from app.api.routers.settings import router as settings_router
from app.llm.ollama_provider import OllamaProvider
from app.storage import (
    InMemoryDatasetStore,
    InMemoryExplorationStore,
    InMemoryModelStore,
    InMemorySplitStore,
    create_run_store,
)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
"""
Vite's default dev-server origin (Batch 3 frontend). The frontend and
backend run on different origins/ports in local dev (and, later,
Docker), so the browser's fetch()/EventSource calls need CORS to be
allowed here — without this, every request from the frontend would be
blocked by the browser regardless of what the backend itself returns.
Overridable via PIPER_CORS_ORIGINS (comma-separated), same
environment-variable-driven pattern used everywhere else in this
codebase (e.g. PIPER_OLLAMA_HOST) — never hardcoded as the only option.
"""


def _cors_origins() -> list[str]:
    raw = os.environ.get("PIPER_CORS_ORIGINS")
    if raw is None:
        return DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.dataset_store = InMemoryDatasetStore()
    app.state.split_store = InMemorySplitStore()
    app.state.model_store = InMemoryModelStore()
    app.state.run_store = create_run_store()
    app.state.exploration_store = InMemoryExplorationStore()
    app.state.llm_provider = OllamaProvider()
    artifact_root = Path(os.environ.get("PIPER_ARTIFACT_DIR", "artifacts"))
    artifact_root.mkdir(parents=True, exist_ok=True)
    app.state.artifact_dir = artifact_root
    yield


app = FastAPI(
    title="PIPER",
    description="Autonomous ML Pipeline Intelligence Engine — API layer over the deterministic agent core.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router)
app.include_router(runs_router)
app.include_router(predict_router)
app.include_router(learning_router)
app.include_router(settings_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
