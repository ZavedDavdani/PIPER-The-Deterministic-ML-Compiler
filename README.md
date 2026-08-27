# PIPER — A Deterministic ML Compiler Powered by LLM Planners

> Upload a tabular dataset. Describe the prediction target. PIPER plans, executes, validates, and delivers a verified ML pipeline — deterministically, reliably, with strict execution guardrails.

PIPER is an autonomous ML pipeline engine that combines an LLM planner (supporting Google Gemini, OpenAI, or local Ollama) with a fully deterministic execution, validation, and artifact layer. The LLM **proposes** a plan; rigid Python rules **validate, execute, and decide** — the planner can never bypass guards, pick models, or generate governance reports.

---

## Architecture

```mermaid
flowchart TD
    User([User]) -->|upload dataset + target| Ingest[Multi-format Ingestion]
    Ingest --> Sanitize[Prompt Sanitization]
    Sanitize --> LLM([LLM Planner: Gemini / OpenAI / Ollama\npropose only])
    LLM --> Validate{validate_proposed_plan\n5-tool allowlist}
    Validate -->|rejected| Fail1[Structured Failure\nno execution]
    Validate -->|valid| Execute[Deterministic Execution\nclean · engineer · split]
    Execute --> ML[Train & Evaluate\nLogisticRegression / RandomForest]
    ML --> Guards{Guardrails\nleakage · imbalance · cardinality\nconstant features · baseline gate}
    Guards -->|fail + retries left| LLM
    Guards -->|fail + exhausted| HI[Human Intervention Package]
    Guards -->|pass| Report[Run Report]
    Report --> Artifact[Verified Artifact Bundle\nnp.array_equal parity gate]
    Artifact --> Inference[Standalone Inference / Test Flight\nno LLM · no retrain]
    Report --> Governance[Governance: Model + Data Cards]
    Report --> StudentMode[Student Mode\n14-stage journey · What-If sandbox]
```

---

## What PIPER Does

| Phase | What happens |
|---|---|
| **Ingestion** | CSV, TSV, Excel, JSON, Parquet, Jupyter Notebook — detected and profiled automatically |
| **Profiling** | Column types, null missingness, cardinality, class distribution |
| **Sanitization** | Sample values scanned for prompt-injection patterns before reaching the LLM |
| **Multi-Provider Planning** | Flexible planner support for **Google Gemini**, **OpenAI**, or **Ollama**, with configurable model selection |
| **Deterministic Validation** | `validate_proposed_plan()` enforces a rigid 5-tool allowlist — anything else is rejected before execution |
| **Plan Adequacy** | Pre-execution verification that the plan addresses all required dataset cleaning needs |
| **Execution** | Drop columns, impute nulls, convert types, one-hot encode categoricals, scale numerics |
| **Training** | Logistic Regression and Random Forest — fitted as a standard scikit-learn `Pipeline` on the training split only |
| **Evaluation & Selection** | Accuracy, Precision, Recall, F1, ROC-AUC; deterministic F1-score model selection (fixed in code) |
| **Safety Guardrails** | Data leakage, class imbalance, constant features, high cardinality, suspicious metrics, baseline gate |
| **Autonomous REPLAN** | On guardrail/adequacy failure: targeted replan prompt with previous failure evidence and plan diff |
| **Duplicate Detection** | Plan hashes detect identical retry proposals as `DUPLICATE_PLAN` |
| **Artifact Generation** | `pipeline.joblib`, `pipeline.py`, `training_reproduction.ipynb`, `manifest.json`, `evidence.json`, `requirements.txt`, `hashes.json` — verified via `np.array_equal` parity gate |
| **Governance Evidence** | Deterministic Model Cards, Dataset Cards, Cryptographic Fingerprints, Feature Importance |
| **Test Flight** | Standalone inference interface scoring unseen batch CSV / JSON records against verified artifacts without retraining |
| **Controlled What-If** | Single-variable experiments in an isolated sandbox (`exp_...`) with client/server validation, preserving base run |
| **Student Mode** | 14-stage guided learning journey, interactive pipeline flowchart, level-aware "Why did PIPER do this?" explanation |

---

## Screenshots

### 1. Dataset Upload / Configuration Dashboard
![Dataset Upload / Configuration Dashboard](docs/screenshots/dataset-upload.png)

### 2. Student Mode — 14-Stage ML Learning Journey
![Student Mode — 14-Stage ML Learning Journey](docs/screenshots/student-mode.png)

### 3. End-to-End Pipeline Visualization
![End-to-End Pipeline Visualization](docs/screenshots/pipeline-visualization.png)

### 4. Model Comparison
![Model Comparison](docs/screenshots/model-comparison.png)

### 5. Decision Trace + Planning & Execution
![Decision Trace + Planning & Execution](docs/screenshots/decision-trace.png)

### 6. Baseline Gate
![Baseline Gate](docs/screenshots/baseline-gate.png)

### 7. Deterministic Guardrails
![Deterministic Guardrails](docs/screenshots/guardrails.png)

### 8. Reproducibility
![Reproducibility](docs/screenshots/reproducibility.png)

### 9. Verified Artifacts
![Verified Artifacts](docs/screenshots/verified-artifacts.png)

### 10. Test Flight / Inference
![Test Flight / Inference](docs/screenshots/test-flight.png)

---

## Design Constraints (Non-Negotiable)

1. **The LLM never controls routing.** Routing reads `validation.valid` and `retry_count` — never plan content, never LLM output.
2. **No automatic plan repair.** Invalid plans are rejected wholesale, not partially patched.
3. **Explanations are never LLM-generated.** Every explanation is a deterministic template filled with recorded run values.
4. **Artifact parity is exact.** `np.array_equal` — not `np.allclose`. Approximate agreement is rejected.
5. **Inference never retrains.** The standalone predict endpoint and Test Flight load the verified `pipeline.joblib` only.
6. **What-If experiments are fully isolated.** Running a What-If experiment creates an isolated `exp_` record and never mutates the base run or base verified artifact.

---

## Prerequisites

- **Python 3.11+**
- **Node.js 20+** with npm
- **LLM Provider** (any of the following):
  - **Google Gemini API Key** (`GEMINI_API_KEY`; model configurable via `GEMINI_MODEL`)
  - **OpenAI API Key** (`OPENAI_API_KEY`)
  - **Ollama** running locally:
    ```bash
    ollama pull qwen3:4b
    ```

---

## Setup

### Linux / macOS

```bash
git clone <repo-url> piper && cd piper
bash setup.sh
```

### Windows (PowerShell)

```powershell
git clone <repo-url> piper; cd piper
.\setup.ps1
```

Both scripts:
- Check Python 3.11+ and Node.js 20+
- Create a `.venv` virtual environment
- Install pinned Python dependencies (`requirements.txt`)
- Install frontend dependencies (`npm install`)
- Create `data/` and `artifacts/` directories
- Check Ollama connectivity and model availability

### Copy the environment template (optional)

```bash
cp .env.example .env   # edit to override any defaults
```

---

## System Check

```bash
python check.py
```

Expected output:

```
PIPER SYSTEM CHECK
────────────────────────────────────────────────
  ✓  Python version             3.11.x
  ✓  Python dependencies        all required packages found
  ✓  Directory data/            .../data
  ✓  Directory artifacts/       .../artifacts
  ✓  Ollama server              http://localhost:11434
  ✓  Planner model              qwen3:4b
  ✓  SQLite database            data/piper_runs.sqlite

PIPER READY
```

---

## Start

### Linux / macOS

```bash
bash run.sh
```

### Windows (PowerShell)

```powershell
.\run.ps1
```

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000 (Swagger docs at `/docs`)

### Or start each process manually

**Linux / macOS:**

```bash
# Terminal 1 — backend
cd backend
../.venv/bin/uvicorn app.main:app --reload

# Terminal 2 — frontend
cd frontend
npm run dev
```

**Windows (PowerShell):**

```powershell
# Terminal 1 — backend
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload

# Terminal 2 — frontend
cd frontend
npm run dev
```
---

## Sample Workflow

1. Open **http://localhost:5173**
2. Upload **`data/raw/telco_customer_churn.csv`** (included in the repo)
3. Set target column: **`Churn`**
4. Click **Start Run** — watch the live SSE event feed
5. When complete:
   - **Engineer Mode**: Decision trace, model comparison chart, guardrail checks, governance cards, artifact export
   - **Student Mode**: Learning journey, "Why did PIPER do this?", What-If experiments

### Try failure handling deliberately

Add a column that duplicates the target. The leakage guardrail detects it, PIPER replans, and — when the retry budget is exhausted — returns a structured `HUMAN_INTERVENTION_REQUIRED` package with exact evidence and recommended actions.

---

## API Quick Reference

```bash
# Upload dataset
curl -X POST http://localhost:8000/datasets \
  -F "file=@data/raw/telco_customer_churn.csv"

# Start run
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"<id>","target_column":"Churn"}'

# Live events
curl -N http://localhost:8000/runs/<run_id>/events

# Student Mode explanation
curl http://localhost:8000/runs/<run_id>/learn/explanation

# One-variable What-If experiment
curl -X POST http://localhost:8000/runs/<run_id>/explore \
  -H "Content-Type: application/json" \
  -d '{"variable": "algorithm", "value": "random_forest"}'
```

Full API reference: `/docs` when the backend is running.

---

## Configuration

All variables are optional. Defaults work out of the box with a local Ollama install.

| Variable | Default | Description |
|---|---|---|
| `PIPER_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `PIPER_LLM_MODEL` | `qwen3:4b` | Planner model (any Ollama-compatible model) |
| `PIPER_OLLAMA_TIMEOUT_SECONDS` | `600.0` | Per-socket timeout for Ollama calls |
| `PIPER_OLLAMA_TOTAL_DEADLINE_SECONDS` | `900.0` | Hard wall-clock deadline per planning call |
| `PIPER_OLLAMA_KEEP_ALIVE` | `10m` | Model keep-alive to avoid REPLAN cold-reload penalty |
| `PIPER_RUN_STORE` | `sqlite` | `sqlite` (persistent) or `memory` |
| `PIPER_SQLITE_PATH` | `data/piper_runs.sqlite` | SQLite database path |
| `PIPER_ARTIFACT_DIR` | `artifacts` | Directory for exported ML artifact bundles |
| `PIPER_CORS_ORIGINS` | `http://localhost:5173,...` | CORS origins for the API |

See `.env.example` for the full annotated template.

---

## Planner Model & Hardware Notes

PIPER's deterministic safety guarantees are **independent of the planner model**.
The model only proposes plans; PIPER's validator, guardrails, and routing make all decisions.

| Model | Typical latency | Notes |
|---|---|---|
| `qwen3:4b` | 143–418s per call (CPU, Telco dataset) | Documented baseline; ~2/10 single-shot success |
| `qwen3:8b` | ~400–460s per call (CPU) | Higher adherence to the planning schema |
| `qwen3:14b`+ | — | Better schema compliance; requires more RAM |

**Documented benchmark result:** `qwen3:4b`, 10 independent real Telco runs, CPU inference:
- 2/10 complete end-to-end success
- 8/10 trigger at least one REPLAN or exhaust the retry budget
- All failures produce structured, bounded, explainable outcomes

When planning fails, PIPER safely intercepts via deterministic REPLAN, duplicate detection, and structured intervention packages — no uncaught exceptions, no silent corruption.

---

## Tests & Verification

```bash
# Backend test suite (272 passing unit/integration tests)
cd backend && pytest tests/ -q

# Frontend test suite (51 passing unit/component/integration tests)
cd frontend && npm test -- --run

# Frontend production build
cd frontend && npm run build

# End-to-end Live Gemini Titanic verification
cd backend && python verify_titanic_demo.py
```

---

## Project Layout

```
PIPER/
├── backend/
│   ├── app/                # LangGraph graph, plan nodes, validation, tools
│   │   ├── agent/          # Agent graph and deterministic execution tools
│   │   ├── artifacts/      # Bundle export, SHA-256 hashes, parity gate
│   │   ├── deployment/     # Standalone inference, Test Flight, readiness
│   │   ├── governance/     # Model cards, data cards, fairness analysis
│   │   ├── learning/       # Student Mode: explanations, concept registry
│   │   ├── llm/            # OllamaProvider protocol + client
│   │   ├── schemas/        # Pydantic contracts (every structured type)
│   │   ├── storage/        # SQLite run store, in-memory stores
│   │   └── api/            # FastAPI routers
│   ├── tests/              # 1003-test behavioral suite
│   ├── Dockerfile          # Backend container image definition
│   ├── Dockerfile.dockerignore
│   └── pytest.ini
├── frontend/
│   ├── src/
│   │   ├── features/       # Runs, artifacts, governance, student, settings
│   │   ├── pages/          # HomePage, RunPage, HistoryPage
│   │   └── lib/            # API client, SSE hooks, types
│   ├── Dockerfile          # Frontend container image definition
│   ├── nginx.conf
│   └── package.json
├── data/raw/               # Reference dataset (Telco Customer Churn)
├── benchmark_data/         # Test fixture dataset (Titanic)
│   └── train.csv
├── benchmark_results/      # Recorded empirical benchmark evidence
├── check.py                # PIPER system readiness check
├── setup.sh / setup.ps1    # One-command setup (Linux/macOS / Windows)
├── run.sh / run.ps1        # One-command launcher
├── .env.example            # Configuration template
├── docker-compose.yml      # Docker Compose (backend + frontend)
├── LICENSE                 # MIT License
└── requirements.txt        # Pinned Python dependencies
```

---

## Limitations

- **Binary and multiclass tabular classification only.** No regression, time-series, text, images.
- **Two candidate algorithms** (Logistic Regression, Random Forest), fixed in code. The LLM cannot introduce new model families.
- **Local small models have low single-shot planning success rates.** This is a documented, bounded failure mode — not a silent one.
- **`valid=True` means no implemented guardrail found a violation**, not that the pipeline is provably optimal.
- **In-process storage** for datasets, splits, and models. SQLite for run history. Nothing persists across restarts except run history and artifact bundles on disk.
- **Single-variable What-If scope.** Experiments change exactly one thing to preserve comparability.
- **No auth, multi-tenancy, or horizontal scaling.** Local/demo target only.

---

## Docker

```bash
# Requires Ollama already running on the host
docker compose up --build
```

Frontend: http://localhost:5173 · Backend: http://localhost:8000

Override any configuration in `.env` (docker-compose reads it automatically).

---

## License

MIT
