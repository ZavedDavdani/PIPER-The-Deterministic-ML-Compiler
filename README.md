# PIPER — Autonomous ML Pipeline Intelligence Engine

PIPER takes a messy tabular dataset and a prediction target, then
autonomously profiles the data, plans and executes cleaning and feature
engineering, trains and compares models, validates its own results
against deterministic guardrails, self-corrects when validation fails,
and streams the whole process live through an API and web UI.

It also exports verified standalone inference bundles, generates full
governance model/data cards, powers an interactive educational learning
environment (Student Mode), and lets you run controlled, one-variable
What-If experiments against its own results.

---

## The core design principle

**The LLM never controls routing.**

This is the single idea the entire codebase is built around. A local LLM
(via Ollama) *proposes* a plan. Deterministic Python code *validates,
executes, and decides*.

```
LLM  ──proposes──►  ProposedPlan (untrusted)
                          │
                          ▼
              validate_proposed_plan()      ← fixed 5-tool allowlist
                          │
                    [rejected?] ──► structured failure, zero execution
                          │
                          ▼
                    deterministic execution
                          │
                          ▼
                  validate_pipeline()       ← guardrails are the sole authority
                          │
              ┌───────────┴───────────┐
            PASS                    FAIL
              │                       │
            REPORT            [retries left?] ──► REPLAN / REPORT
```

Concretely:

- The LLM can only ever return a list of tool-name + argument proposals.
  It never executes code, never mutates state, never picks a branch.
- `validate_proposed_plan()` enforces a fixed 5-tool allowlist
  (`drop_column`, `convert_column_type`, `impute_missing_values`,
  `encode_categorical_features`, `scale_features`). Anything outside it,
  or malformed, is rejected *before* any execution.
- Guardrails (leakage, class imbalance, constant features, high
  cardinality, suspicious metrics, baseline gate) are the only authority
  on whether a pipeline passes. Routing reads `validation.valid` and
  `retry_count` — never plan content, never LLM output.
- Model selection is F1-max, fixed in code, not LLM-choosable.
- Explanations and governance reports are deterministic templates filled with real run values —
  never LLM-generated prose.

---

## Architecture & Product Modules

### Execution graph (LangGraph)

```
VALIDATE_INPUT → PROFILE → SANITIZE → PLAN_ENTRY → PLAN
  → CLEAN → FEATURE_ENGINEER → SPLIT → REPRODUCIBILITY
  → TRAIN → EVALUATE → COMPARE → BASELINE → VALIDATE
  → REPORT
```

`PLAN_ENTRY` is the graph's only loop-back target, reached from two
back-edges: a failed guardrail check (`VALIDATE → PLAN_ENTRY`) and a
retryable planning failure (`PLAN → PLAN_ENTRY`). Both independently
check `retry_count < max_retries` first. Every other edge is
straight-line.

A PIPER-owned `MAX_EXECUTION_STEPS` ceiling guarantees the graph always
returns a clean terminal state rather than raising a recursion error,
independent of how `max_retries` is configured.

### Product Modules (V1.2)

1. **Decision Trace & Human Intervention** (`app/agent/productization.py`):
   9-stage productized trace (LLM Proposed → Validated → Adequacy → Replan → Execution → Training → Evaluation → Guardrails → Final Verdict). Deterministic verdict and structured intervention package for failed runs.
2. **Local Run Store & Replay** (`app/storage/sqlite_run_store.py`):
   SQLite persistence for run metadata, events, and evidence. Deterministic run replay without invoking the LLM.
3. **Artifact Bundles & Parity Gate** (`app/artifacts/`):
   Exports self-contained pipeline bundles (`pipeline.joblib`, `pipeline.py`, `training_reproduction.ipynb`, `manifest.json`, `evidence.json`, `hashes.json`). Verification gate guarantees exact holdout prediction parity (`np.array_equal`) between the in-memory fitted pipeline and reloaded joblib artifact before publishing.
4. **Model Governance & Data Cards** (`app/governance/`):
   Deterministic model cards, dataset cards, SHA-256 fingerprinting, feature importance, and on-demand demographic subgroup fairness analysis.
5. **Deployment & Test Flight** (`app/deployment/`):
   Zero-LLM standalone `/predict` endpoint, deployment readiness verification, deployment package generator (`Dockerfile`, `inference.py`, `requirements.txt`), and interactive CSV Test Flight scoring.
6. **Student Mode & ML Education** (`app/learning/`):
   Evidence-grounded 14-stage learning journey, ML pipeline flowchart visualizer, deterministic "Why?" inspector, model/metric concept explainers, and safe single-variable What-If experiments.

### Stack

| Layer | Technology |
|---|---|
| Agent core | Python 3.11, LangGraph, Pydantic |
| ML & Data | pandas, scikit-learn, joblib |
| LLM | Ollama (`qwen3:4b` by default), stdlib `urllib` client only |
| Storage | SQLite (runs & events), in-memory stores for datasets/splits/models |
| API | FastAPI + Server-Sent Events (SSE) |
| Frontend | React 19, Vite, TypeScript, Tailwind CSS v4, Lucide Icons |
| Deployment | Docker Compose (2 services) |

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 22+ (frontend only)
- [Ollama](https://ollama.com) running on the host with a model pulled:
  ```bash
  ollama pull qwen3:4b
  ```

Ollama always runs **outside** Docker, directly on the host.

### Run with Docker

```bash
docker compose up --build
```

- Frontend → http://localhost:5173
- Backend → http://localhost:8000 (API docs at `/docs`)

### Run locally without Docker

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

### Configuration

All optional, overridable via environment variables or a root `.env` file:

| Variable | Default |
|---|---|
| `PIPER_OLLAMA_HOST` | `http://localhost:11434` (Docker: `http://host.docker.internal:11434`) |
| `PIPER_LLM_MODEL` | `qwen3:4b` |
| `PIPER_OLLAMA_TIMEOUT_SECONDS` | `600.0` |
| `PIPER_RUN_STORE` | `sqlite` (tests default to `memory`) |
| `PIPER_SQLITE_PATH` | `data/piper_runs.sqlite` |
| `PIPER_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` |
| `PIPER_BACKEND_PORT` / `PIPER_FRONTEND_PORT` | `8000` / `5173` |

---

## API Reference

### Datasets & Ingestion

| Endpoint | Method | Description |
|---|---|---|
| `/datasets` | `POST` | Upload dataset (CSV, TSV, Excel, JSON, .ipynb, Parquet; 100MB cap) |
| `/datasets/{id}` | `GET` | Retrieve dataset profile and column types |
| `/datasets` | `GET` | List registered dataset IDs |

### Runs & Lifecycle

| Endpoint | Method | Description |
|---|---|---|
| `/runs` | `POST` | Launch an autonomous run (returns `202 Accepted`) |
| `/runs` | `GET` | List run history (newest updated first) |
| `/runs/{id}` | `GET` | Current run status, active node, and attempt count |
| `/runs/{id}/events` | `GET` | Real-time SSE stream of execution events |
| `/runs/{id}/result` | `GET` | Full terminal state, comparison, validation, and reproducibility metadata |
| `/runs/{id}/decision-trace` | `GET` | 9-stage structured decision trace (active mid-run) |
| `/runs/{id}/verdict` | `GET` | Final deterministic verdict (`ACCEPTED`, `REJECTED`, `HUMAN_INTERVENTION_REQUIRED`) |
| `/runs/{id}/intervention` | `GET` | Structured human review package for failed runs |
| `/runs/{id}/evidence` | `GET` | Canonical JSON evidence export (`piper.evidence.v1`) |
| `/runs/{id}/replay` | `GET` | Reconstruct decision trace & evidence from store without LLM |

### Artifacts & Deployment

| Endpoint | Method | Description |
|---|---|---|
| `/runs/{id}/artifacts` | `POST` | Build and verify standalone artifact bundle |
| `/runs/{id}/artifacts` | `GET` | Artifact bundle metadata, verification status, file manifest |
| `/runs/{id}/artifacts/download/{file}` | `GET` | Download bundle file (`pipeline.joblib`, notebook, manifest, etc.) |
| `/runs/{id}/deployment` | `GET` | Deployment readiness check (5 criteria) |
| `/runs/{id}/deployment/predict` | `POST` | Standalone single/batch JSON prediction |
| `/runs/{id}/deployment/test-flight` | `POST` | Score unseen test CSV against verified artifact |
| `/runs/{id}/deployment/package` | `GET` | Download standalone deployment zip (`Dockerfile`, `inference.py`) |

### Governance & Fairness

| Endpoint | Method | Description |
|---|---|---|
| `/runs/{id}/governance` | `GET` | Full governance bundle (model card, dataset card, fingerprints) |
| `/runs/{id}/governance/subgroups` | `POST` | Operator-specified demographic subgroup fairness analysis |
| `/runs/{id}/governance/download/{doc}` | `GET` | Download markdown governance reports (`model_card.md`, etc.) |

### Student Mode & Explorations

| Endpoint | Method | Description |
|---|---|---|
| `/runs/{id}/learn/explanation` | `GET` | Level-aware deterministic run explanation (`beginner`/`intermediate`/`advanced`) |
| `/runs/{id}/learn/journey` | `GET` | 14-stage evidence-grounded learning journey |
| `/runs/{id}/learn/pipeline` | `GET` | Pipeline flowchart nodes, directed edges, and metrics |
| `/runs/{id}/learn/why` | `GET` | "Why did PIPER do this?" inspector for specific actions |
| `/learn/concepts` | `GET` | ML concept dictionary (imputation, scaling, encoding, leakage, etc.) |
| `/learn/actions` | `GET` | Explanation registry for all PIPER data cleaning actions |
| `/learn/models` | `GET` | Model architecture guidance (Logistic Regression, Random Forest) |
| `/learn/metrics` | `GET` | Evaluation metric formulas and guidance |
| `/runs/{id}/explore` | `POST` | Run safe single-variable What-If experiment |
| `/runs/{id}/explore` | `GET` | List all What-If explorations for a run |

---

## Limitations & Truth in Advertising

- **Binary & multiclass tabular classification only.** No regression, no time series, no computer vision or NLP.
- **Two candidate algorithms** (`LogisticRegression`, `RandomForestClassifier`), fixed in code.
- **Local Small-Model Planner Hit Rate:** With local small models (e.g. `qwen3:4b`), end-to-end single-shot completion is approximately **2/10** on complex real-world datasets due to strict schema adherence requirements. When planning fails, PIPER's state machine safely intercepts the failure via deterministic REPLAN, duplicate detection, and structured intervention packages without executing invalid code.
- **Single-variable What-If scope:** Explorations intentionally allow modifying only 1 variable at a time on the exact same split to preserve causal comparability.
- **Inference requires verified artifacts:** Standalone `/predict` and Test Flight refuse to score against unverified or failing runs.

---

## Test Suite & Verification

```bash
# Backend test suite (1003 passing, 5 skipped)
pytest backend/tests/ -q

# Frontend test suite (39 passing)
cd frontend && npm test -- --run

# Frontend production build
cd frontend && npm run build
```

The 5 skipped tests are live-Ollama integration tests gated behind `PIPER_RUN_OLLAMA_TESTS=1`.

---

## License

MIT License. Designed and built as an autonomous ML engineering platform and educational environment.
