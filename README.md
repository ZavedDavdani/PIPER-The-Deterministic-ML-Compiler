# PIPER — Autonomous ML Pipeline Intelligence Engine

PIPER takes a messy tabular dataset and a prediction target, then
autonomously profiles the data, plans and executes cleaning and feature
engineering, trains and compares models, validates its own results
against deterministic guardrails, self-corrects when validation fails,
and streams the whole process live through an API and web UI.

It also explains what it did — and lets you run controlled, one-variable
experiments against its own results.

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
- Explanations are deterministic templates filled with real run values —
  never LLM-generated prose.

---

## Architecture

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

### Stack

| Layer | Technology |
|---|---|
| Agent core | Python 3.11, LangGraph, Pydantic |
| ML | pandas, scikit-learn |
| LLM | Ollama (`qwen3:4b` by default), stdlib `urllib` client only |
| API | FastAPI + Server-Sent Events |
| Frontend | React, Vite, TypeScript, Tailwind CSS v4, Recharts |
| Deployment | Docker Compose (2 services) |

### Key invariants

- No raw DataFrames or fitted sklearn objects ever enter agent state —
  only references (`dataset_id`, `model_id`, `split_id`) and structured
  results.
- Preprocessing and the classifier are fit as **one** sklearn `Pipeline`
  on the training split only, making test-set contamination structurally
  impossible rather than merely avoided by convention.
- The LLM only ever sees a **sanitized** dataset view — sample values are
  scanned and neutralized for injection patterns before they reach a
  prompt. The original dataset is never mutated by sanitization.
- Every run executes against a private, run-scoped clone of the uploaded
  dataset, so concurrent or repeated runs can't corrupt each other.

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

### Run with Docker (recommended)

```bash
docker compose up --build
```

- Frontend → http://localhost:5173
- Backend → http://localhost:8000 (API docs at `/docs`)

### Run locally without Docker

```bash
pip install -r requirements.txt
cd backend && uvicorn app.main:app --reload
```

```bash
cd frontend && npm install && npm run dev
```

### Configuration

All optional, overridable via a root `.env` file:

| Variable | Default |
|---|---|
| `PIPER_OLLAMA_HOST` | `http://localhost:11434` (Docker: `http://host.docker.internal:11434`) |
| `PIPER_LLM_MODEL` | `qwen3:4b` |
| `PIPER_OLLAMA_TIMEOUT_SECONDS` | `600.0` |
| `PIPER_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` |
| `PIPER_BACKEND_PORT` / `PIPER_FRONTEND_PORT` | `8000` / `5173` |

---

## Usage

### Web UI

1. Open http://localhost:5173
2. Upload a dataset — CSV, TSV, Excel, JSON, Jupyter notebook, or
   Parquet (the reference dataset is at
   `data/raw/telco_customer_churn.csv`). PIPER shows the detected
   format and dimensions before you start the run.
3. Pick the target column and start a run
4. Watch the live SSE feed — events are grouped by attempt, with a
   REPLAN badge when the agent self-corrects
5. Inspect results: model comparison chart, baseline gate, every
   guardrail check, reproducibility metadata, and full failure detail

### API

```bash
# Upload (any supported format — detected from the extension)
curl -X POST http://localhost:8000/datasets \
  -F "file=@data/raw/telco_customer_churn.csv"

# Excel with an explicit worksheet
curl -X POST http://localhost:8000/datasets \
  -F "file=@book.xlsx" -F "sheet_name=Customers"

# Start a run
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"<id>","target_column":"Churn"}'

# Live progress
curl -N http://localhost:8000/runs/<run_id>/events
```

| Endpoint | Purpose |
|---|---|
| `POST /datasets` | Upload a dataset — CSV, TSV, Excel, JSON, .ipynb, or Parquet (100MB cap) |
| `GET /datasets/{id}` | Real dataset profile |
| `POST /runs` | Start a run (returns `202` immediately) |
| `GET /runs/{id}` | Live status |
| `GET /runs/{id}/events` | SSE progress stream |
| `GET /runs/{id}/result` | Full terminal result |
| `GET /runs/{id}/summary` | Aggregated run summary |
| `GET /runs/{id}/timeline` | Phase timeline (works mid-run) |
| `GET /runs/{id}/learn/explanation` | Deterministic explanation of this run |
| `GET /learn/formulas` | Static ML formula library |
| `GET /learn/comprehension-checks` | Static "check your understanding" content |
| `POST /runs/{id}/explore` | Run a one-variable experiment |
| `GET /runs/{id}/explore` | List experiments for a run |

---

## Capabilities

### Multi-format ingestion

Upload **CSV, TSV, Excel (.xlsx/.xlsm/.xls), JSON, Jupyter notebooks
(.ipynb), or Parquet**. The format is detected automatically and
normalized into a single internal representation, so the entire agent
and ML pipeline behaves identically no matter what you upload — there
is no separate code path per format.

- **Excel** — multi-sheet workbooks list every worksheet; the first
  non-empty sheet is used by default and the choice is reported, or you
  can name a sheet explicitly.
- **JSON** — records, columnar, pandas `split` orient, wrapper keys
  (`data`/`records`/`rows`), and JSON Lines. Non-tabular JSON is
  rejected with an explanation rather than force-flattened.
- **Parquet** — column types come from the file itself, so datetimes,
  booleans, and sized integers survive intact (CSV loses all three).
- **Jupyter notebooks** — **notebook code is never executed.** PIPER
  reads only already-saved cell outputs. If a displayed DataFrame was
  truncated by Jupyter, that's reported rather than silently ingested
  as partial data; if the notebook loads its data from an external
  file, that filename is named so you can upload it directly.

### Self-correction

When a guardrail fails, the graph decides whether to replan — bounded by
`max_retries`. On a REPLAN the LLM receives structured evidence: the
previous attempt's `FailureInfo` and a canonical diff of the plan it
already tried. Plans are hashed (excluding LLM rationale), so proposing
an executably identical plan is caught as `DUPLICATE_PLAN` rather than
wasting a retry.

### PIPER Learn — Explain

A read-only layer that explains a finished run in beginner-friendly
terms, grounded entirely in that run's own evidence: why each column was
dropped or imputed, why a model was selected, what each metric value
means, what each guardrail checks, and what a failure category implies.

Every explanation is a reviewed static template with real run values
plugged in. There is no LLM in this path at all — and no code path in the
graph that even knows the learning layer exists, so it is structurally
incapable of influencing a run.

### PIPER Learn — Explore

Controlled experimentation against a finished run: change **exactly one**
variable — either the algorithm, or one hyperparameter within its
existing locked bounds — and see the effect.

It reuses the original run's split (no new randomness, so the comparison
is genuinely apples-to-apples), reuses the same training and comparison
code, and stores results in an isolated `experiment_id` namespace. The
original run is never modified.

### Reproducibility

Every run records environment versions, a content fingerprint of the
actual split dataset, and the split/model random states. Identical inputs
produce identical model selection and metrics.

---

## Limitations

Stated plainly, because knowing the boundaries matters more than
overclaiming:

- **Binary/multiclass tabular classification only.** No regression, no
  time series, no text/image/multimodal data.
- **Two candidate models** (Logistic Regression, Random Forest), fixed in
  code. The LLM cannot introduce new algorithms.
- **`valid=True` means "no implemented guardrail found a violation"** — it
  is not proof the pipeline is free of every possible problem. Guardrails
  cover feature-level leakage indicators; pipeline-level contamination is
  prevented structurally by the architecture, not re-verified at runtime.
- **In-memory storage only.** Nothing persists across a process restart.
  This is a local/demo deployment target, single-process, no auth.
- **Small local models struggle with strict schemas.** With `qwen3:4b`, a
  run frequently needs one or more REPLANs before producing a plan that
  passes validation, and can exhaust its retry budget. That is the
  guardrail system working as designed — the failure is bounded,
  structured, and explainable — but it means a clean first-attempt run is
  not guaranteed. A larger model improves the hit rate.
- **Real-LLM latency is significant.** Planning against the full Telco
  dataset on CPU inference measured 143–418s per call (5 samples), which
  is why the default timeout is 600s.
- **Context budgeting is character-based**, not tokenizer-based — a
  deliberate choice to avoid a tokenizer dependency. It reduces only
  per-column sample values, never column names, types, target info,
  missingness, or summary statistics.
- **No plan-diff across REPLAN attempts in explanations.** Only plan
  hashes are retained across attempts, so there is nothing left to diff
  against once a later attempt begins.

---

## Testing

```bash
cd backend && pytest -q          # 654 passed, 5 skipped
cd frontend && npm run test:run  # 22 passed
```

The 5 skips are real-Ollama integration tests, gated behind an
environment variable so the normal suite never requires a live LLM
server:

```bash
PIPER_RUN_OLLAMA_TESTS=1 pytest -q tests/test_ollama_integration.py
```

Tests are behavioral where it matters — for example, "evaluation never
refits anything" is proven by monkeypatching `fit()` to raise, not by
reading the code and assuming.

---

## Project layout

```
backend/
  app/
    agent/          # graph, nodes, state, planning, tools
      tools/        # deterministic tools (profiling → guardrails)
    learning/       # PIPER Learn: explanations, formulas, checks
    llm/            # provider protocol, Ollama client, test double
    schemas/        # every structured contract
    storage/        # in-memory dataset/split/model/run/exploration stores
    api/            # FastAPI routers
  tests/
frontend/src/       # React app (pages, components, API client, SSE hook)
data/raw/           # reference Telco Customer Churn dataset
```

---

## Demo script

1. `docker compose up --build` (Ollama already running on the host)
2. Upload `data/raw/telco_customer_churn.csv`, target `Churn`
3. Watch the live SSE feed — point out attempt grouping and the REPLAN
   badge when the agent self-corrects
4. On completion: show the model comparison chart, then the deterministic
   selection justification ("logistic_regression selected: F1=0.4969 vs.
   0.4848 for random_forest")
5. Open the guardrail panel — show *all* checks, passed and failed, not
   just violations
6. Call `GET /runs/{id}/learn/explanation` to show the grounded,
   template-based explanation
7. `POST /runs/{id}/explore` swapping the algorithm — show the isolated
   result and that the original run is untouched

To demo failure handling deliberately, add a column that duplicates the
target. The leakage guardrail catches it, the agent replans, and — when
the retry budget runs out — reports a structured, bounded failure with
`human_intervention_required: true`.
