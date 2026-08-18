# PIPER — Autonomous ML Pipeline Intelligence Engine

Authoritative reference for continuing work on PIPER in Claude Code. This
file is kept concise and factual — it describes what actually exists,
verified against the real codebase, not aspirational or in-progress work.
Update it whenever a batch/milestone completes or a genuine finding is
discovered; don't let it drift from reality.

## Project identity

- **Root:** `C:\dev\PIPER`
- **Backend path:** `C:\dev\PIPER\backend` (Python 3.11+, FastAPI + LangGraph)
- **Frontend path:** `C:\dev\PIPER\frontend` (React + Vite + TypeScript + Tailwind CSS v4)
- **Purpose:** solo portfolio project — an autonomous agent that takes a
  messy tabular dataset and a prediction objective, profiles the data,
  plans and executes cleaning/feature engineering, trains and compares
  models, validates its own results against deterministic guardrails,
  self-corrects via REPLAN when validation fails, and exposes the whole
  process live through an API and web UI.
- **Scope (V1):** tabular **binary/multiclass classification** only.
  Models: Logistic Regression, Random Forest (fixed candidate set, not
  LLM-choosable). Reference dataset: Telco Customer Churn.

## V1 ARCHITECTURE FROZEN (frozen 2026-08-15; independently re-audited and re-verified 2026-08-17)

PIPER V1 is **COMPLETE AND FROZEN**. The architecture below has been
implemented, tested, and independently re-audited. Do not reopen,
redesign, or undo any of it.

**Read this before trusting any reliability claim in this file.** The
*architecture* is sound and verified — 846 passing tests, deterministic
guardrails intact, and no trust-boundary defect found in the 2026-08-17
audit. The *planner model* is not reliable enough for unattended
autonomous operation: qwen3:4b completes a real end-to-end Titanic run
**2 times out of 10**. One genuine non-trust-boundary defect IS open (the
600s timeout does not bound total wall time). Those are three separate
claims and this file keeps them separate on purpose. See "V1 Model Status
& Reliability Baseline" and the two limitation sections below.

### V1 Architecture Summary

All of the following are implemented, tested, and locked:

1. **Planner tool contract** — `TOOL_ARGUMENT_SCHEMAS` rendered into every
   planning prompt; the LLM sees the exact argument contract it must follow.
   Result: qwen3:4b 0% → 100% valid-plan rate (3/3) after fix.

2. **Effective-Feature Adequacy** — `evaluate_plan_adequacy()` gates
   missing-value severity on `ColumnTransformer(remainder="drop")` membership:
   - Column IS in effective feature set (named in encode/scale step) + unaddressed
     missing → **material** (blocks, triggers REPLAN)
   - Column NOT in effective feature set + unaddressed missing → **advisory**
     (recorded, never blocks)
   - 7 false-positive material findings removed from historical Titanic replay.

3. **State-Preserving REPLAN** — `classify_plan_steps()` splits plan into
   `valid_steps` / `implicated_steps`; `build_replan_prompt()` injects a
   `=== VALID OPERATIONS (preserve these) ===` section with exact production
   JSON tool-call syntax. Result: whack-a-mole regression 83% → 0%
   (v1: 1/6 executable, v2: 3/6 executable).

4. **Parse/Transport State Preservation** — `_carried_forward_preserved_steps()`
   carries previously-validated `valid_steps` / `implicated_steps` through
   provider/parse/transport failures. Re-validates before injecting. First-attempt
   failures with no prior state produce no fabricated steps.

5. **Duplicate-plan detection** — covers both VALID and REJECTED plans.
   Prevents burning the full retry budget on a repeated identical failure.

6. **Retry budget** — `max_retries=2` default, `PLAN_ENTRY` loop-back,
   `MAX_EXECUTION_STEPS` hard ceiling. Unchanged.

7. **Trust boundary** — `validate_proposed_plan()` is the sole, unweakened
   authority. No LLM output ever executes without passing it. No auto-correction,
   no plan mutation, no silent repair.

### V1 Test Baseline (pinned .venv)

| Suite | Result | Environment |
|---|---|---|
| Focused: parse-state + adequacy + contract + provider + validation (5 files) | **171 passed** | .venv, 2026-08-17 |
| Historical effective-feature replay (6 recorded plans) | **ALL HARD ASSERTIONS PASSED** (0 regressions) | offline, 2026-08-17 |
| REPLAN prompt snapshot (`verify_replan_prompt_snapshot.py`) | **ALL CHECKS PASSED** (9 checks) | offline, 2026-08-17 |
| **Full backend regression** | **928 passed, 5 skipped, 0 failures** (23m48s) | .venv, 2026-08-17 |
| Frontend (Vitest) | **28 passed** (6 files) | 2026-08-17 |
| Frontend production build | clean² | 2026-08-17 |

Regression history: 824 → 838 (+14, parse-failure state preservation) →
846 (+8, the `EXACT TOOL ARGUMENT CONTRACTS` tests) → **928** (+82, the
final hardening pass: 19 `test_adequacy_dtype_compatibility.py` + 13
`test_total_planning_deadline.py` + 4 `test_completed_run_reporting.py` +
46 `test_v1_matrix.py`). Each delta reconciles exactly against the tests
added; no test was weakened or removed.

**`tests/test_v1_matrix.py` is the compact V1 safety matrix** — 24
numbered scenarios covering structural validation (valid plan, invalid
tool, wrong key/type, invalid enum, `drop_column` array misuse,
multi-column ops), adequacy (numeric/categorical missingness, invalid
categorical strategy, advisory vs material), REPLAN and state preservation
across parse/provider/timeout failures, duplicate-plan detection,
oscillation termination, target protection, the retry ceiling, the
planning deadline, execution failure, and a successful end-to-end run.
All fake-provider driven; no Ollama.

The 5 skips are the real-Ollama integration tests (`tests/test_ollama_integration.py`),
gated behind `PIPER_RUN_OLLAMA_TESTS=1` — intentional.

¹ The 1 warning is `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead`. This is a third-party library compatibility notice, not a test failure and not a PIPER code issue. It does not affect functionality or test validity.

² The build emits a "chunks larger than 500 kB" advisory from Vite. That is a
bundle-size suggestion, not an error — the build succeeds. Code-splitting is
POST-V1.

### V1 Environment

The correct pinned environment is **`c:\dev\PIPER\.venv`** (Python 3.11),
NOT the system-wide Python. Always run tests with:

```
c:\dev\PIPER\.venv\Scripts\python.exe -m pytest ...
```

All 12 pinned packages in `requirements.txt` are present and version-matched
in `.venv`:

| Package | Pinned | Status |
|---|---|---|
| langgraph | 1.2.10 | ✓ |
| openpyxl | 3.1.5 | ✓ |
| xlrd | 2.0.2 | ✓ |
| pyarrow | 25.0.1 | ✓ |
| pandas | 3.0.2 | ✓ |
| scikit-learn | 1.8.0 | ✓ |
| pydantic | 2.13.4 | ✓ |
| pytest | 9.1.1 | ✓ |
| fastapi | 0.141.1 | ✓ |
| uvicorn | 0.52.1 | ✓ |
| httpx | 0.28.1 | ✓ |
| python-multipart | 0.0.32 | ✓ |

The system Python has mismatched versions (older langgraph, etc.) and must
NOT be used for running tests.

### V1 Model Status & Reliability Baseline

- **`qwen3:4b`** = current development reliability baseline (Empirical 10-run reliability: **2/10 = 20% end-to-end success**).
- **First-Attempt Structural Validity**: **9/10 (90%)** after exact tool argument contract hardening.
- **First-Attempt Adequacy**: **0/10 (0%)** — `qwen3:4b` consistently overlooks low-percentage categorical missingness (`Embarked` at 0.22%) on first attempts.
- **Final Executable Plan Rate**: **9/10 (90%)**.
- **Dominant Failure Mode**: `ADEQUACY_FAILURE` (40%) from multi-turn REPLAN state oscillation/regression and CPU inference timeouts (20%).
- **`qwen3:8b`** = screened and **REJECTED** — see "qwen3:8b screening" below.
- **`qwen3.5:4b`** = experimental candidate (faster, but erratic plan completeness at n=3).
- **keep_alive: 10m** — implemented and verified; do NOT change.
- **Temperature: the model's own default — PIPER never sets it.**
  Verified directly in `OllamaProvider.generate_plan()`: the request payload
  is exactly `{model, prompt, stream, format, keep_alive}` — there is **no
  `options` key**, and `OllamaProvider.__init__` accepts no temperature
  argument. PIPER therefore cannot and does not override temperature; each
  model runs on whatever it ships (qwen3:4b = 0.6). Any benchmark artifact
  reporting `"temperature": 0.0` is a **hardcoded label in that harness's
  own result record, not a measured or applied value** — see the correction
  note under "qwen3:8b screening". Do NOT add an `options` field to "fix"
  this; the absence is deliberate and production-realistic.
- **Ollama timeout: `DEFAULT_TIMEOUT_SECONDS = 600.0s`** — do NOT change.
  **Known limitation: this is NOT an enforced ceiling on total call
  duration** — see "Known limitation: the 600s timeout does not bound total
  planning wall time" below.

### V1 Demo Readiness & Reliability Classification

- [x] Dataset can be loaded (CSV, TSV, Excel, JSON, IPYNB, Parquet)
- [x] Dataset can be profiled
- [x] LLM produces a structured plan (90% first-attempt valid after contract fix)
- [x] Exact tool argument contracts hardened in prompt (`_format_exact_tool_contracts`)
- [x] Structural validation rejects malformed/invalid plans deterministically
- [x] Adequacy catches effective-feature problems
- [x] Advisory findings do not unnecessarily block execution
- [x] REPLAN preserves validated work (0% whack-a-mole regression when `valid_steps` present)
- [x] Parse failures carry forward previously validated state
- [x] Invalid plans cannot execute
- [x] Duplicate-plan detection terminates repetitive loops
- [x] Retry behavior terminates safely within `max_retries=2` and `MAX_EXECUTION_STEPS`
- [x] Successful plans reach training, evaluation, comparison, and reporting
- [x] Deterministic guardrails surround the LLM at every stage
- [x] Backend starts cleanly and serves `GET /health` → 200
- [x] Frontend builds (`npm run build`) and its 28 tests pass
- [x] One real end-to-end Titanic run completed, including a live REPLAN

**FINAL V1 VERDICT (2026-08-17): READY FOR DEMO — WITH DOCUMENTED LIMITATIONS.**

Ready, because: the deterministic architecture is verified sound (846
tests, 0 failures; the independent 2026-08-17 audit found **no defect in
the trust boundary** — validation, adequacy, REPLAN preservation, routing,
retry bounding, and duplicate detection all verified intact and
unmodified), the demo path starts and runs cleanly, and a real end-to-end
run completed including a genuine REPLAN recovery.

The audit DID find one real defect, and it is not in that layer: the
600s timeout does not bound total wall time (transport bounding — see
below). It is documented, not hidden, and deliberately left for POST-V1.

With limitations, because BOTH of these must be stated to anyone
demonstrating it:

1. **Run the demo attended, and budget ~11–15 minutes per run.** Measured
   end-to-end success is **2/10**. If a run fails, the correct narrative is
   that the deterministic layer *caught* a bad plan and terminated safely —
   that IS the system working, and it is the more interesting thing to
   show.
2. **Planning is now bounded, but it is not fast.** The unbounded-hang
   defect (5.8 h and 16.6 h observed) was FIXED in the final hardening pass
   — a call that reaches `DEFAULT_TOTAL_DEADLINE_SECONDS` (900s) now
   returns a structured `timeout` failure instead of hanging, so worst-case
   planning is `(max_retries + 1) × 900s`. Budget accordingly, and still
   have a completed run's `run_id` ready as a fallback.

Do NOT describe V1 as "reliable" or "production autonomous" — the evidence
does not support that, and the honest framing (deterministic guardrails
around an unreliable planner) is the stronger engineering story anyway.

**V1 RELIABILITY CLASSIFICATION (Empirical N=10): `NOT RELIABLE` (2/10 = 20%)**  
*Decision*: `V1 RELIABILITY NOT ACCEPTED` for **unattended** live demo. Architecture is frozen and sound, with no trust-boundary defect; the reliability bottleneck is the 4B model's ability to compose a COMPLETE cumulative plan across REPLAN turns, plus CPU-inference latency — not a defect in PIPER's deterministic layer.

### End-to-end smoke test — PASSED (2026-08-17, n=1)

One real run through the real API with the real production planner
(`qwen3:4b`), real Titanic CSV, real Ollama. Harness:
`backend/v1_e2e_smoke.py` (capped at 1500s precisely because the 600s
Ollama timeout does not bound wall time — see the limitation section).

```
run_c4e15325   t=   0s  initialized
               t=   5s  running      node=plan_entry   attempt=0
               t= 402s  replanning   node=plan_entry   attempt=1   <- real REPLAN
               t= 642s  running      node=reproducibility
               t= 647s  completed    node=report
```

| Stage | Result |
|---|---|
| Ingestion | 891 × 12, format `csv` detected |
| Profiling | real `profile_dataset()` output |
| Planning → structural validation → adequacy | attempt 0 rejected, **attempt 1 accepted** |
| REPLAN | `retry_count=1`, `replanned=true` — recovered within budget |
| Training | both candidates trained |
| Comparison | `random_forest` F1=**0.4091** vs `logistic_regression` F1=0.2526 → random_forest selected (F1-max, locked) |
| Guardrails | `valid=true`, 6 checks, **0 violations** |
| Baseline | accuracy 0.6145, majority class `"0"` — model beat the gate |
| Report | `status: completed`; timeline 27 phases, `replan_count=1` |

**Total: 647s (~11 min).** This exercised the full REPLAN recovery path in
production, end to end, and it worked.

**This is n=1 and does NOT contradict the 2/10 reliability figure** — it is
one draw from a distribution whose measured success rate is 20%. It proves
the demo path *can* complete; it does not make it *likely to*. Both facts
are true and both are recorded here deliberately.

**Post-hardening confirmatory run (2026-08-17, `run_622a3087`):** re-ran
the same smoke test against the hardened code to exercise it once more
before freeze. This attempt terminated at 880s with `DUPLICATE_PLAN` after
2 REPLANs — qwen3:4b proposed an executably-identical plan to one already
rejected. **This is not a defect**: it is the documented duplicate-plan
protection firing correctly, terminating within the exact 3-attempt budget
(no more, no fewer), with `human_intervention_required=true` and nothing
executed. It is additional evidence the hardened trust boundary still
behaves correctly under a real failure, not a contradiction of the
successful run above — both outcomes are consistent with the measured
2/10 reliability distribution, and neither was cherry-picked (this was the
next attempt run, reported as-is).

## V1 FINAL HARDENING PASS (2026-08-17) — dtype-aware adequacy, total planning deadline, clean completion reporting

Three genuine defects fixed, each with focused regression tests. Nothing
was auto-corrected, no validation was weakened, no retry/routing semantics
changed, and no model was switched.

### Fix 1 — adequacy dtype blind spot (found by the qwen3:8b run)

**Defect.** Adequacy marked a column's missingness ADDRESSED whenever the
column was *named* in an `impute_missing_values` step, without checking
that the strategy could run against that column's dtype. Observed live:
`impute_missing_values(column="Embarked", strategy="median")` passed
adequacy, then failed at execution
(`unsupported_dtype_strategy_combination`), leaving the missing values
unresolved.

**Why it was a V1 correctness issue, not cosmetics.** In the observed run
it was harmless only because `Embarked` never entered the effective
feature set. Had it been encoded or scaled, the NaN would have survived
into the training matrix and `LogisticRegression` would have raised at
`fit()` — a real crash path.

**The rule, verified against the real tool** (`type_conversion.py::impute_missing_values`),
not assumed:

| Strategy | Valid for |
|---|---|
| `mean`, `median` | numeric columns ONLY |
| `mode` | numeric OR categorical |

**Implementation** (`_evaluate_imputations()` in `plan_adequacy.py`): a
column counts as imputed only if the strategy is compatible with the dtype
that column will have **at that point in the plan**. A new
`imputation_strategy_compatibility` condition reports the incompatibility,
its severity following the SAME effective-feature rule as `missing_values`
so the two findings for one column can never disagree about blocking.

**ORDER IS LOAD-BEARING — and this is where the first attempt was wrong.**
`clean_node` executes steps sequentially, so a `convert_column_type` step
earlier in the plan changes what a later impute sees. A first version
judged dtype from the profile alone and **broke 24 existing tests**: the
canonical Telco plan converts `TotalCharges` (`str` at plan time, because
of blank strings) to numeric and only *then* imputes it with `median` —
valid at execution precisely because the convert ran first. The existing
suite caught the false positive; the fix now walks the plan in order and
tracks each column's effective dtype. Compatibility is also only evaluated
for columns that genuinely have missing values, since the condition is
about whether an impute RESOLVES missingness.

**Never auto-corrected:** `median` is never silently rewritten to `mode`.
The incompatible step is reported with the reason (and names `mode` as the
valid alternative); the existing REPLAN loop lets the model fix it.

### Fix 2 — total planning deadline (`DEFAULT_TOTAL_DEADLINE_SECONDS = 900.0`)

**Defect.** `urllib`'s `timeout` bounds each individual socket operation,
not total request duration — measured single calls of 20,923s (5.8 h) and
59,679s (16.6 h) against a configured 600s budget. The documented ceiling
was not enforceable.

**Implementation** (`OllamaProvider.generate_plan()`): the blocking read
runs on a daemon thread and is ABANDONED if `total_deadline_seconds`
passes (Python cannot safely kill a thread blocked in a socket read; the
daemon can never hold the process open and dies when its socket timeout
fires). A breach returns the SAME structured `timeout` ProviderError a
socket timeout already produced, so `plan_node_v2`'s existing
provider-failure branch handles it unchanged — **including carrying
previously-validated steps forward** via
`_carried_forward_preserved_steps()`.

900s is deliberately ABOVE the 600s socket timeout so it is a backstop,
not a new operating limit: every healthy call measured in this project
(~400–460s) completes well inside it. Configurable via
`PIPER_OLLAMA_TOTAL_DEADLINE_SECONDS`, same precedence as host/model/timeout.

**Total planning time for a run is now provably bounded** by
`(max_retries + 1) × total_deadline_seconds`. Transport-layer only: no
repair, no plan produced, nothing unvalidated can execute.

### Fix 3 — a completed run reported stale failure text

**Defect.** `report_node`'s success branch cleared `failure` (Pre-6A Polish
item 1) but not `error`. `tracing.py` builds both the REPORT `TraceEvent`
message and the terminal run message from `state.error`, so a run that
REPLANned and then SUCCEEDED emitted its superseded attempt's failure text
on its final event. Observed in the qwen3:8b run: `status: completed`,
`failure: null`, yet the report event read *"Plan is structurally valid but
inadequate… affecting ['Age']"*.

Cosmetic in the API (status/`validation.valid` were always correct) but
actively misleading in the live SSE feed and frontend timeline — which is
where a demo is actually watched. Fixed by also clearing `error`; a
genuinely failed run still reports its error (pinned by a control test).

### Assessed and deliberately NOT implemented (POST-V1)

- **REPLAN convergence / oscillation detection.** Verified by test that
  whack-a-mole oscillation already terminates safely: it is bounded by the
  retry budget, ends as a structured `PLAN_ADEQUACY` failure with
  `human_intervention_required=true`, and executes nothing. With a
  3-attempt budget, a dedicated convergence detector would add termination
  machinery for a case the budget already bounds — not worth new risk at
  freeze.
- **Failure-taxonomy expansion.** `EVALUATION_ERROR` deliberately covers
  schema/parse/provider/timeout failures, discriminated at the EVIDENCE
  level (`provider_error_code` vs `violations`/`rejected_steps`). The
  runtime CAN distinguish them and the information is machine-readable, so
  this is pinned by classification-stability tests rather than refactored
  into new categories (which would ripple into `RECOVERABLE_CATEGORIES`
  and Learn-Explain's parametrized coverage).
- Adequacy does not check `no_missing_values` or 100%-missing/`mode`
  edge cases — different tool preconditions, out of scope.

### Multi-agent provenance audit (2026-08-17)

Part of this V1 work was performed by a different agent (Antigravity)
after a context handoff. Because **this project has no git repository**,
provenance was reconstructed by file-modification-time audit against the
known timestamp of the 838-pass regression — a genuine limitation worth
stating: mtimes prove *when* a file changed, not *what* changed, so each
flagged file was then read and audited directly.

**Result: exactly ONE production application file changed after the
838-pass regression — `app/llm/prompts.py`.** Verified untouched:

| File | Last modified | Status |
|---|---|---|
| `app/agent/graph.py` (routing) | 2026-08-12 | untouched |
| `app/agent/state.py` | 2026-08-13 | untouched |
| `app/llm/ollama_provider.py` | 2026-08-14 13:36 | untouched (keep_alive work) |
| `app/agent/plan_validation.py` (validator) | 2026-08-14 01:29 | untouched |
| `app/agent/plan_adequacy.py` | 2026-08-14 19:13 | untouched |
| `app/agent/nodes/real_nodes.py` | 2026-08-14 23:27 | parse-failure fix only |
| `app/llm/prompts.py` | **2026-08-15 17:42** | **Antigravity change — audited, KEPT** |

**The audited change** — a new `_format_exact_tool_contracts()` rendering
an `=== EXACT TOOL ARGUMENT CONTRACTS ===` section with canonical
CORRECT/WRONG JSON examples per tool. **Assessment: correct, keep.** It is
prompt CONTENT only; it is additive and guarded (`if not tool_schemas:
return ""`, plus per-tool membership checks); it preserves the original
`_format_allowed_operations()` rendering rather than replacing it; and it
is backed by tests in `test_planner_contract_titanic.py` asserting that
the WRONG examples genuinely fail `validate_proposed_plan()` and the
CORRECT ones genuinely pass — so the prompt cannot silently drift from the
validator. It changes what the LLM is TOLD, never what is trusted or
executed.

**Consequence:** because a production file changed after the 838-pass run,
the full regression was re-run once this session (see the test baseline
table above). Nothing was reverted; nothing was course-corrected in
production code. The only corrections made were to **documentation and
artifact metadata** (the false `temperature: 0.0` claim), recorded below.

**Benchmark artifacts live in TWO directories** — a real inconsistency
introduced across agent sessions. Nothing was moved (scripts hardcode
these paths, and moving them would break reproducibility of the recorded
runs), but know where to look:

| Location | Contains |
|---|---|
| `benchmark_results/` (repo root) | `v1_reliability_10/`, `qwen3_8b_screening/` |
| `backend/benchmark_results/` | `post_contract/`, `adequacy_recovery/`, `adequacy_recovery_v2/` |

### qwen3:8b screening — REJECTED (screening terminated early, deliberately)

Controlled screening of `qwen3:8b` as a candidate V1 planner, run through
the **real production graph** (`build_graph()`) and the **real
`OllamaProvider`** against the real Titanic fixture — harness:
`backend/benchmark_qwen3_8b_screening_5.py`; results:
`benchmark_results/qwen3_8b_screening/qwen3_8b_5run.json`. Production
architecture, validator, adequacy, retry budget, and routing were NOT
modified for this screening.

**The screening is INCOMPLETE: 2 of the planned 5 trials were recorded**
before it was interrupted (the script supports resume via `start_trial`).
The artifact is preserved verbatim, including its incompleteness.

| Trial | Outcome | Wall time | Detail |
|---|---|---:|---|
| 1 (cold) | FAIL — `SCHEMA_FAILURE` | 506.9s | Invalid tool argument structure; 0 plan steps |
| 2 (warm) | FAIL — adequacy, then 2× timeout | **21,925.7s (6.1 h)** | Attempt 0 structurally valid but material `Embarked` finding; attempts 1–2 both timed out |

**Result: 0/2 end-to-end success → NOT PROMISING** under the locked
criteria (0–2/5 = NOT PROMISING).

**Decision: do NOT adopt qwen3:8b. `qwen3:4b` remains the V1 planner.**

**Why the remaining 3 trials were not run** (an explicit engineering call,
not an omission): the decisive evidence is **latency, and it is already
conclusive**. Trial 2 alone consumed 6.1 hours — a single planning attempt
took 20,923s — against a qwen3:4b median of 888s for a complete run. On
this CPU-only host (no discrete GPU; `ollama ps` reports 100% CPU), an 8B
model is categorically unusable as an interactive planner regardless of
plan quality, and even a best-case 3/3 on the remaining trials would only
reach BORDERLINE while costing ~18 further hours. Resuming would buy no
decision-relevant information.

**Correction to this artifact's metadata:** every record contains
`"temperature": 0.0`. That value is **hardcoded into the harness's result
dict** (`benchmark_qwen3_8b_screening_5.py`, in the `trial_result`
literal) and was never applied — the provider it constructs sends no
`options` field, so qwen3:8b ran at its own default. Do not cite 0.0 as a
measured configuration. The artifact was left unmodified so the historical
record stays intact; this note is the correction.

### Qwen3:8B Successful End-to-End Demonstration (2026-08-17)

**ONE successful end-to-end demonstration run.** This is a single genuine
run, not a reliability measurement — see the scope note at the end of this
section. It does **not** supersede or overwrite the 0/2 screening recorded
above, which remains the only systematic qwen3:8b evidence.

Nothing in PIPER was modified to obtain this. The model was selected
through the existing production mechanism (`PIPER_LLM_MODEL=qwen3:8b`,
read by `OllamaProvider.__init__`); prompts, validator, adequacy, retry
budget, routing, timeout and keep_alive are all untouched. Model identity
was confirmed from `ollama ps` ground truth mid-run: `qwen3:8b`, 5.9 GB,
**100% CPU**, context 4096. Evidence:
`benchmark_results/qwen3_8b_single_success/run_evidence.json` (full SSE
trace + result/summary/timeline/explanation). Harness:
`backend/capture_qwen3_8b_single_success.py`.

| Field | Value |
|---|---|
| Model | `qwen3:8b` (confirmed via `ollama ps`) |
| Dataset | `benchmark_data/train.csv` — Titanic, 891 × 12, no manual preprocessing |
| Target | `Survived` |
| run_id | `run_e13cf35f` |
| Planning attempts | **3** (attempt 0, 1, 2) |
| REPLAN | **YES** — twice; `retry_count=2`, `replanned=true` |
| Final status | **`completed`** |
| Total runtime | **1,184.9s (~19.7 min)** |

**Attempt-by-attempt (real adequacy evidence, not paraphrase):**

```
attempt 0  plan FAILURE  structurally valid but inadequate:
                         1 material finding, missing_values -> ['Embarked']
attempt 1  plan FAILURE  structurally valid but inadequate:
                         1 material finding, missing_values -> ['Age']
attempt 2  plan SUCCESS  -> clean -> feature_engineer -> split ->
                         reproducibility -> train -> evaluate -> compare ->
                         baseline -> validate -> report
```

All three attempts were **structurally valid** — every rejection was by the
adequacy layer, not the schema validator. Both REPLANs were driven by the
production state-preserving REPLAN path.

**Final executable plan (4 operations executed successfully):**

```
1. impute_missing_values({"column": "Age", "strategy": "median"})
   -> Imputed 177 missing value(s) in 'Age' using median (28.0000)
2. drop_column({"column": "Cabin"})   -> 12 -> 11 columns
3. drop_column({"column": "Name"})    -> 11 -> 10 columns
4. scale_features({"columns": ["Age", "Fare"]})
```

**Embarked handling — record it exactly as it happened, because it is not
a clean success.** The final plan DID contain an explicit fifth step,
`impute_missing_values({"column": "Embarked", "strategy": "median"})`.
Adequacy accepted it (a column named in an impute step counts as
ADDRESSED), so the plan passed. At execution that step **failed**:

```
[failure] cleaner/impute_missing_values:
  "Strategy 'median' requires a numeric column; 'Embarked' is not numeric."
```

The run still completed, because the plan contains **no
`encode_categorical_features` step at all** — so `Embarked` never entered
the effective feature set and was excluded by
`ColumnTransformer(remainder="drop")`, exactly as the effective-feature
semantics intend. Net result: **`Embarked` was resolved by EXCLUSION, not
by successful imputation**, even though the model attempted an
imputation. This matches the same
"resolved-by-exclusion" pattern recorded in adequacy-recovery v2.

**Training / evaluation / comparison / guardrails — all PASS:**

| Stage | Result |
|---|---|
| Training | both candidates trained on 712 rows, **2 features** (Age, Fare) |
| Evaluation | `random_forest` F1=**0.4091** (ROC-AUC 0.5854); `logistic_regression` F1=0.2526 (ROC-AUC 0.6507) |
| Comparison | `model_c4ca8268` (`random_forest`) selected by F1-max — "random_forest selected: F1=0.4091 vs. 0.2526 for logistic_regression." |
| Baseline gate | **passed** (majority class `0`, accuracy 0.6145; baseline F1 mathematically undefined) |
| Guardrails | **valid=true**, 6 checks, **0 violations** |
| Report | generated; `status: completed` |

**NEW FINDING — since FIXED, see "Fix 1" in the final hardening pass.**
Adequacy treated a column as ADDRESSED if it was merely *named* in an
`impute_missing_values` step, without checking that the chosen `strategy`
was compatible with the column's dtype. A semantically invalid impute
(`median` on a categorical) therefore satisfied adequacy and only failed
later, at execution. **The trust boundary was never compromised** — the
tool itself correctly rejected the invalid operation and nothing invalid
executed — but adequacy's precondition check had a genuine blind spot.
This run is what surfaced it, and it is now closed: `_evaluate_imputations()`
walks the plan in order and only counts an imputation as addressing a
column when the strategy suits that column's dtype at that point.

**Scope — read this before citing the run.** This is **one successful
end-to-end demonstration**. It is NOT evidence that qwen3:8b is reliable,
has a high success rate, or is production-ready. The systematic screening
above remains **0/2**, and this run took ~20 minutes on 100% CPU with the
host under heavy memory pressure (5.9 GB model against 5.66 GB free RAM).
`qwen3:4b` remains the V1 planner; no production model switch was made.

### Known limitation: the 600s timeout does not bound total planning wall time

`DEFAULT_TIMEOUT_SECONDS = 600.0` is passed to
`urllib.request.urlopen(request, timeout=...)`, which bounds **individual
socket operations**, not total request duration. Directly observed, not
theorized:

- qwen3:8b screening trial 2, attempt 2: **20,923s (5.8 h)** elapsed and
  *then* failed with `"Ollama did not respond within 600.0s."`
- qwen3:4b 10-run reliability: max planning latency **59,679s (16.6 h)**,
  mean 6,879s, against a median of 889s.

So a run can hang far beyond the configured budget. The deterministic
layer still behaves correctly when it eventually returns (structured
failure, bounded retries, no invalid execution) — this is a **transport
bounding defect, not a validation or routing defect**.

**RESOLVED in the final hardening pass** — see "Fix 2 — total planning
deadline" above. `OllamaProvider` now enforces a PIPER-owned total
deadline (`DEFAULT_TOTAL_DEADLINE_SECONDS = 900.0`) on top of the
socket-level timeout, so total planning is bounded by
`(max_retries + 1) × 900s`. The section above is retained as the record of
the defect and the measurements that proved it.

**Practical demo guidance (still applies):** run the demo attended. A call
that reaches the deadline now returns a structured `timeout` failure rather
than hanging, but planner reliability remains 2/10 — see the reliability
baseline.

### POST-V1 FUTURE WORK (do not start without explicit go-ahead)

- ~~Bound total planning wall time~~ — **DONE** in the final hardening
  pass (`DEFAULT_TOTAL_DEADLINE_SECONDS`). Remaining nuance: the abandoned
  daemon thread lives until its socket timeout fires, so a hung request
  still occupies one thread (bounded, harmless, never blocks process exit).
  A cancellable transport would remove even that.
- REPLAN convergence/oscillation detection — assessed and deliberately not
  implemented; see "Assessed and deliberately NOT implemented" above.
- Failure-taxonomy expansion (splitting `EVALUATION_ERROR` into
  SCHEMA/PARSE/PROVIDER/TIMEOUT) — assessed; currently discriminated by
  evidence and pinned by tests.
- Adequacy edge cases not covered: `no_missing_values`, and 100%-missing
  columns imputed with `mode`.
- Planner reliability: qwen3:4b is 2/10 end-to-end. The dominant, fully
  characterized failure is composing a COMPLETE cumulative plan across
  REPLAN turns (it fixes the reported `Embarked` finding but drops
  previously-valid work, or times out first). Candidate directions:
  constrained-grammar/structured sampling, or a larger model **once the
  hardware can run one** — `qwen3:8b` was screened on this host and
  rejected on latency (0/2, 6.1 h/trial), so this is gated on hardware, not
  on trying another download.
- Real-model benchmark of parse-failure state preservation (whether it
  closes the qwen3.5:4b gap — both budget-exhaustion trials followed
  exactly the parse-failure path)
- qwen3.5:4b re-benchmark: 5 more trials at default temperature to resolve
  whether the 1-step plan was an outlier
- Model decision: qwen3:4b vs qwen3.5:4b (needs more trials; note both run
  at their own shipped defaults, since PIPER never sets temperature)
- Frontend UI for Pre-6A Polish / Batch 6A / Batch 6B endpoints
- Persistent storage (all stores are currently in-memory)
- Broadened scope beyond binary/multiclass tabular classification

---

## Core design principle (non-negotiable)

**The LLM never controls routing.** It *proposes* a plan; deterministic
PIPER code *validates, executes, and decides*. Every design choice in this
codebase must survive "why did you design it this way?" in an interview.
Concretely:

- The LLM (via `OllamaProvider`) only ever returns a `ProposedPlan` — a
  list of tool-name/argument proposals. It never executes code, never
  mutates state, never chooses graph routing.
- `validate_proposed_plan()` (`app/agent/plan_validation.py`) is a fixed,
  5-tool allowlist (`drop_column`, `convert_column_type`,
  `impute_missing_values`, `encode_categorical_features`,
  `scale_features`) — an LLM proposal outside this set, or with malformed
  arguments, is rejected before any execution.
- Deterministic guardrails (`validate_pipeline()`) — leakage, imbalance,
  constant features, high cardinality, baseline gate — are the sole
  authority on whether a pipeline passes. The graph's routing functions
  (`_route_after_validate`, etc., in `app/agent/graph.py`) decide
  PASS/REPLAN/FAIL based only on `state.validation`/`state.retry_count`,
  never on plan content or LLM output.
- Model selection (`compare_models()`) is F1-max, fixed, never
  LLM-choosable.

## Architecture overview

### Graph flow (`app/agent/graph.py`)

```
VALIDATE_INPUT -> PROFILE -> SANITIZE -> PLAN_ENTRY -> PLAN
  -> [PLAN failed, retryable, retries left? -> PLAN_ENTRY (REPLAN)]
  -> [PLAN failed otherwise (non-retryable, or retries exhausted)? -> REPORT]
  -> CLEAN -> FEATURE_ENGINEER -> SPLIT
  -> REPRODUCIBILITY -> TRAIN -> EVALUATE -> COMPARE -> BASELINE -> VALIDATE
  -> [valid? -> REPORT (completed)]
  -> [invalid, retries left? -> PLAN_ENTRY (REPLAN)]
  -> [invalid, retries exhausted? -> REPORT (failed)]
  -> [VALIDATE never ran (an earlier node failed this attempt)?
      -> retryable failure + budget left ? PLAN_ENTRY : REPORT]   (Batch 7)
```

TRAIN/EVALUATE/COMPARE/BASELINE have no routing checks between them —
deliberately, since VALIDATE is the graph's single guardrail decision
point. A failure at any of them therefore carries forward to VALIDATE.
As of Batch 7 each of those downstream nodes passes an already-structured
upstream failure through UNCHANGED (`_upstream_already_failed()`) instead
of overwriting it, so the terminal result names the real root cause
rather than the last symptom in the chain.

`PLAN_ENTRY` (`_increment_retry_if_replanning`) is the graph's only
loop-back TARGET, reached via two back-edges as of Batch 5:
`VALIDATE -> PLAN_ENTRY` (a failed guardrail check) and
`PLAN -> PLAN_ENTRY` (a retryable PLAN-node failure — LLM provider
error, or a proposed plan that failed `validate_proposed_plan()`; see
`_route_after_plan`). Both back-edges independently verify
`retry_count < max_retries` before ever routing back here; `PLAN_ENTRY`
increments `retry_count` exactly once per genuine REPLAN and is also
the sole enforcement point for `MAX_EXECUTION_STEPS` (see M4). Every
other edge is a straight-line, one-directional flow through
`real_nodes.py`.

### Planning flow (`plan_node_v2`, `app/agent/nodes/real_nodes.py`)

```
build_sanitized_llm_context()   -- never raw dataset content
    v
llm_provider.generate_plan()    -- untrusted proposal (OllamaProvider or FakeLLMProvider)
    v
[provider failure?] -> structured FailureInfo, zero execution
    v
validate_proposed_plan()        -- deterministic allowlist/argument check
    v
[invalid?] -> structured FailureInfo, zero execution
    v
list[PlanStep] construction     -- ONLY from a plan that passed validation
    v
canonicalize_plan() + plan_hash() -- rationale excluded from identity
    v
duplicate check against plan_history -- DUPLICATE_PLAN if repeated
    v
state.plan
```

On REPLAN, `plan_node_v2` builds `failure_context` (from `state.failure`)
and `previous_plan_summary` (via `diff_plans()`) so the LLM has structured
evidence of what to change, not just "try again."

### Key backend modules

| Module | Role |
|---|---|
| `app/agent/graph.py` | Graph construction (`build_graph()`), all routing logic, `MAX_EXECUTION_STEPS` |
| `app/agent/nodes/real_nodes.py` | Every real node: `plan_node_v2`, `train_node_v2`, `evaluate_node_v2`, `compare_node`, `baseline_node`, `validate_node_v2`, etc. |
| `app/agent/state.py` | `AgentState` (the LangGraph state schema), `PlanStep` |
| `app/agent/plan_canonical.py` / `plan_diff.py` | Plan identity (hash) and structured diffing, excluding LLM rationale |
| `app/agent/plan_validation.py` | `validate_proposed_plan()`, the 5-tool `ALLOWED_TOOL_NAMES` allowlist, `TOOL_ARGUMENT_SCHEMAS` (LLM-facing argument contract, rendered into the prompt — see "Planner-contract fix") |
| `app/agent/plan_adequacy.py` | `evaluate_plan_adequacy()` — deterministic, read-only plan-COMPLETENESS layer (distinct from validity); see "Plan Adequacy" |
| `app/schemas/adequacy.py` | `PlanAdequacyResult`, `AdequacyFinding`, statuses/severities |
| `app/agent/tools/guardrails.py` | `validate_pipeline()` — leakage/imbalance/constant-features/high-cardinality + baseline gate |
| `app/agent/tools/sanitized_llm_context.py` | `build_sanitized_llm_context()` — the only dataset view an LLM ever sees |
| `app/agent/tracing.py` | `run_with_tracing()` (post-hoc) and `stream_with_tracing()` (live, per-node) — connect `TraceEvent`/`InMemoryRunStore` to real graph execution |
| `app/agent/run_summary.py` | `build_run_summary()` (Pre-6A Polish) — pure, read-only aggregation of a terminal run's state into `RunSummary` |
| `app/agent/timeline.py` | `build_execution_timeline()` (Pre-6A Polish) — pure derivation of a high-level phase timeline from a run's `TraceEvent` stream |
| `app/llm/provider.py` / `ollama_provider.py` | `LLMProvider` protocol, `FakeLLMProvider` (tests), `OllamaProvider` (real, stdlib `urllib` only) |
| `app/schemas/failure.py` | `FailureInfo`, the failure taxonomy (terminal vs. recoverable categories) |
| `app/schemas/guardrails.py` | `PipelineValidationResult` and all guardrail report schemas |
| `app/schemas/run_summary.py` / `execution_timeline.py` | `RunSummary`, `ExecutionTimeline`/`TimelinePhase` (Pre-6A Polish) |
| `app/learning/explain.py` | `build_run_explanation()` (Batch 6A) — pure, deterministic/template-based, read-only explanation layer over a terminal run's state |
| `app/learning/formulas.py` / `comprehension.py` | `FORMULA_LIBRARY`, `COMPREHENSION_CHECKS` (Batch 6A) — static, curated, generic content |
| `app/schemas/learning.py` | `RunExplanation` and its sub-schemas, `FormulaEntry`, `ComprehensionCheck` (Batch 6A) |
| `app/agent/tools/ingestion.py` | `ingest_dataset()`/`detect_format()` — the ONLY format-aware code in PIPER; normalizes CSV/TSV/Excel/JSON/IPYNB/Parquet into one DataFrame |
| `app/schemas/ingestion.py` | `DatasetFormat`, `FORMAT_EXTENSIONS`, `IngestionResult`, `SheetInfo` |
| `app/agent/tools/exploration.py` | `explore_alternative()` (Batch 6B) — reuses `train_model()`/`evaluate_model()`/`compare_models()`; enforces exactly-one-variable-changed |
| `app/schemas/exploration.py` | `ExplorationResult`, `ExplorationVariable` (Batch 6B) |
| `app/storage/exploration_store.py` | `ExplorationStore`/`InMemoryExplorationStore` (Batch 6B) — isolated `experiment_id` namespace, never merged into `RunStore` |
| `app/storage/*` | `DatasetStore`, `SplitStore`, `ModelStore`, `RunStore`, `ExplorationStore` — all in-memory, all with an abstract interface + `InMemory*` implementation |
| `app/main.py` | FastAPI app, CORS, lifespan-wired stores + `OllamaProvider` |
| `app/api/routers/datasets.py`, `runs.py`, `learning.py` | REST + SSE endpoints (see below) |

## Milestone status

| Milestone | Status |
|---|---|
| M1 — Deterministic foundation (tools, schemas) | Complete |
| M2 — LangGraph skeleton | Complete |
| M3 — Real LLM planning (Ollama) | Complete (Phases 1-7, incl. gated real-Ollama integration tests) |
| M4 — Guardrails and self-correction | Complete for the one concrete gap found (deterministic execution-step budget); guardrails/failure-taxonomy/REPLAN-context were already solid from M1-M3 |
| M5 — FastAPI backend + SSE ("Batch 2") | Complete |
| M6 — Frontend ("Batch 3") | Complete |
| M7 — Docker ("Batch 4") | Complete |
| Batch 5 — Production hardening | Complete |
| Pre-6A Polish | Complete |
| Batch 6A — PIPER Learn: Learn-Explain | Complete |
| Batch 6B — PIPER Learn: Learn-Explore | Complete |
| Batch 7 — Final integration, context-budgeting, + README | Complete |

### Current test baseline

**838 passed, 5 skipped** as of Parse-failure state preservation (see
that section below): 824 -> 838 (+14), exactly the new tests in
`tests/test_parse_failure_state_preservation.py`. Verified with one full
`pytest -q` run (838 passed, 5 skipped, 34m15s, zero failures);
production code changed (`real_nodes.py`), so the full suite was run per
the "unless infrastructure-only" policy. A second confirmatory run was
not performed (the change is additive evidence propagation only — no
validation, routing, retry, duplicate-plan, or execution logic modified
— and a 251-test planner/adequacy/graph sweep passed standalone
beforehand).

**824 passed, 5 skipped** as of Effective-Feature Adequacy +
State-Preserving REPLAN (see that section above): 807 -> 824 (+17),
exactly the net new tests in `tests/test_plan_adequacy.py` (32 -> 49)
after updating six tests to the corrected severity semantics and adding
`classify_plan_steps()` / REPLAN-prompt coverage. Verified with one full
`pytest -q` run (824 passed, 5 skipped, 27m26s, zero failures);
production code changed (`plan_adequacy.py`, `real_nodes.py`,
`prompts.py`), so the full suite was run per the
"unless infrastructure-only" policy. A second confirmatory run was not
performed (no validation-authority, routing, retry, or execution logic
changed — only severity classification and additive prompt content —
and a 211-test planner/validation/graph sweep passed standalone
beforehand).

**807 passed, 5 skipped** as of the Plan Adequacy layer (see "Plan
Adequacy" above): 774 -> 807 (+33). The delta reconciles exactly: +32
from the new `tests/test_plan_adequacy.py`, and **+1 automatically**
from `tests/test_learning.py`'s existing parametrization over
`FailureCategory.__args__`, which grew 11 -> 12 with `PLAN_ADEQUACY` —
the anti-drift mechanism documented in Batch 6A working as designed
(a new taxonomy member cannot be added without Learn-Explain covering
it). Verified with one full `pytest -q` run (807 passed, 5 skipped,
39m04s, zero failures); production code changed, so the full suite was
run per the "unless infrastructure-only" policy. A second confirmatory
run was not performed (the change is additive — no existing validation,
routing, retry, or execution logic was modified — and the 206-test
planner/graph/validation/learning sweep passed standalone beforehand).

**774 passed, 5 skipped** as of the `keep_alive` configuration change
(engineering-hardening Phase 2A — see above): 770 (planner-contract-fix
baseline, below) -> 774 (+4: `TestOllamaProviderConfiguration`'s
`test_defaults_to_documented_keep_alive`/
`test_reads_keep_alive_from_environment_variable`/
`test_explicit_keep_alive_constructor_arg_overrides_environment`/
`test_keep_alive_never_disabled_by_default` in `test_llm_provider.py`).
Verified with one full `pytest -q` run (774 passed, 5 skipped, 29m32s,
zero failures) — production code changed (`OllamaProvider`'s request
payload), so the full suite was run per the "unless infrastructure-only"
policy; a second confirmatory run was not performed (narrow, additive
config change — no validation/routing/execution logic touched — and
the directly relevant test set was also run standalone beforehand with
198/198 passing).

**770 passed, 5 skipped** as of the planner-contract fix (post-4-model-
benchmark investigation — see "Planner-contract fix" above): 741 (REPLAN
duplicate-invalid-plan fix baseline, below) -> 770 (+29: the new
`TestToolArgumentSchemasMatchValidator`/
`TestRealWorldBenchmarkFailurePatternsRejected` in
`test_plan_validation.py`, `TestToolSchemaRenderedIntoPrompt` in
`test_llm_provider.py`, and the new `test_planner_contract_titanic.py`).
Verified with one full `pytest -q` run (770 passed, 5 skipped, 45m58s,
zero failures) — production code changed (prompt construction,
`LLMPlanningContext`, `plan_node_v2`'s context-building call), so the
full suite was run per the "unless infrastructure-only" policy; a second
confirmatory run was not performed (the change is additive/
documentation-only — no validation, routing, or execution logic
touched — and every directly-relevant test file was also run
standalone beforehand with 194/194 passing).

Growth across the last eight checkpoints:
553 (Batch 5) -> 574 (Pre-6A Polish, +21) -> 615 (Batch 6A, +41) -> 642
(Batch 6B, +27) -> 659 (Batch 7, +17) -> 733 (multi-format ingestion,
+74: 65 in the new `tests/test_ingestion.py` covering all 6 formats/
malformed files/unsupported formats/IPYNB edge cases, plus 9 added to
`tests/test_api_datasets.py`, including a full real-agent-run proof
driven from a Parquet upload) -> 741 (REPLAN duplicate-invalid-plan
fix, +8: the new `tests/test_replan_duplicate_invalid_plan.py` — see
"Known finding: REPLAN could repeat an already-rejected invalid plan
forever" below). Each addition verified with a single full run (no
failures each time) — a second confirmatory run was skipped for the
ingestion change (scoped to the ingestion/API layer, no core graph/
state/routing/ML execution touched) and for the REPLAN fix (explicitly
requested as a single-run verification), per this project's "second run
only if core-affecting or the first run shows instability" policy —
the REPLAN fix DOES touch core routing-adjacent code
(`plan_node_v2`/`plan_history`), so treat this one number as slightly
less battle-tested than a doubly-confirmed baseline; re-run `pytest -q`
yourself before relying on it further.

The 5 skips are the real-Ollama integration tests
(`tests/test_ollama_integration.py`), gated behind
`PIPER_RUN_OLLAMA_TESTS=1` — intentional; the normal `pytest -q` suite
must never require a live Ollama server.

Full-suite runtime varies widely with machine load — observed between
~13 minutes and ~2 hours across this project. The wide upper end is
contention, not test growth: several suites drive real end-to-end graph
runs against the real 7,043-row Telco CSV, so a full run competing with
a Docker build or a live Ollama session slows dramatically. Run it on an
otherwise-quiet machine for a representative figure.

**Re-verify this number yourself before trusting it further** — don't
assume it's still accurate without running `pytest -q` on the actual
checkout, per this project's standing sync-discipline rule (see
"Development rules" below).

Frontend: 28 tests passing (Vitest + React Testing Library), including a
full-app MSW-driven integration test covering both a successful run and a
guardrail-rejected run, and (post-multi-format-ingestion) coverage of
uploading each supported format and surfacing the detected format/
dimensions panel.

## FastAPI + SSE architecture (M5)

Two stores/provider are wired once at process startup (`app/main.py`
lifespan) onto `app.state`: `DatasetStore`, `SplitStore`, `ModelStore`,
`RunStore` (all in-memory), and `llm_provider` (`OllamaProvider()` by
default, env-var configured — see below).

**Endpoints:**
- `POST /datasets` — upload a dataset in ANY supported format (CSV, TSV,
  Excel, JSON, `.ipynb`, Parquet — see "Multi-format ingestion" below);
  returns `dataset_id` plus the detected format and dimensions. Optional
  `sheet_name` form field selects an Excel worksheet.
- `GET /datasets` — list all `dataset_id`s.
- `GET /datasets/{id}` — real `profile_dataset()` output (columns, dtypes,
  missing/unique %, etc.).
- `POST /runs` — validates the dataset exists, builds `AgentState`, kicks
  off execution via `stream_with_tracing()` in a FastAPI `BackgroundTasks`
  call (non-blocking — Starlette offloads it to a worker thread). Returns
  `202` with `run_id` immediately.
- `GET /runs/{id}` — live status (`status`, `current_node`, `attempt`,
  `plan_history`) from `RunStore`.
- `GET /runs/{id}/result` — `409` until the run reaches a terminal status,
  then the real `validation`/`comparison`/`baseline`/`failure`/
  `reproducibility`/`model_results`/`evaluation_results`.
- `GET /runs/{id}/summary` (Pre-6A Polish) — `409` until terminal, same as
  `/result`; returns the real `RunSummary` (`build_run_summary()`).
- `GET /runs/{id}/timeline` (Pre-6A Polish) — **not** gated on terminal
  status, unlike `/result`/`/summary`; returns the real `ExecutionTimeline`
  (`build_execution_timeline()`) derived from whatever `TraceEvent`s exist
  so far, so it's meaningful mid-run too, like the SSE feed it reads from.
- `GET /runs/{id}/learn/explanation` (Batch 6A) — `409` until terminal,
  same as `/result`/`/summary`; returns the real `RunExplanation`
  (`build_run_explanation()`) — a read-only, deterministic/template-based
  explanation grounded in this run's own evidence (see the dedicated
  Batch 6A section below).
- `GET /learn/formulas` (Batch 6A) — the static, curated formula library
  (`FORMULA_LIBRARY`); no `run_id`, no `RunStore` dependency.
- `GET /learn/comprehension-checks` (Batch 6A) — the static "check your
  understanding" content (`COMPREHENSION_CHECKS`); no `run_id`, no
  `RunStore` dependency.
- `POST /runs/{id}/explore` (Batch 6B) — `409` until terminal, same as
  `/result`/`/summary`; body is `CreateExplorationRequest`
  (`base_model_id` + exactly one of `new_algorithm` or
  `hyperparameter_name`+`hyperparameter_value`). Synchronous (a single
  sklearn fit, no `BackgroundTasks`); `201` with the real
  `ExplorationResult` (`explore_alternative()`) on success, `400` for a
  structured `explore_alternative()` rejection (e.g. more than one
  variable changed, `base_model_id` not from this run, disallowed/
  out-of-bounds hyperparameter), `404` if `base_model_id` itself doesn't
  exist.
- `GET /runs/{id}/explore` (Batch 6B) — lists every `ExplorationResult`
  recorded for this run (`ExplorationStore.list_for_run()`).
- `GET /runs/{id}/explore/{experiment_id}` (Batch 6B) — one specific
  exploration by its own isolated `experiment_id`.
- `GET /runs/{id}/events` — Server-Sent Events. Polls `RunStore` for new
  `TraceEvent`s and streams them as `data: <json>\n\n` lines; closes one
  grace-period poll after the run goes terminal so trailing events aren't
  truncated.

**Live progress mechanism:** `stream_with_tracing()`
(`app/agent/tracing.py`) drives the graph via
`graph.stream(..., stream_mode="updates")` instead of `.invoke()`, so a
`TraceEvent` (and a live `RunStore.update()`) is appended immediately
after each node finishes, while the run is still executing — not just
reconstructed after the fact. `run_with_tracing()` (post-hoc,
`.invoke()`-based) still exists and is unchanged; both share the same
tool-call-level event derivation from `tool_trace`.

**CORS:** `app/main.py` adds `CORSMiddleware`, origins from
`PIPER_CORS_ORIGINS` (comma-separated env var), defaulting to
`http://localhost:5173` / `http://127.0.0.1:5173` (the Vite dev-server
and Docker-frontend origin).

## Multi-format dataset ingestion (post-Batch-7)

PIPER ingests **CSV, TSV, Excel (.xlsx/.xlsm/.xls), JSON, Jupyter
notebooks (.ipynb), and Parquet**. All of it lives in ONE place —
`app/agent/tools/ingestion.py` — and its entire job is to normalize
every format into the same pandas DataFrame that CSV always produced.

**The load-bearing constraint: format-awareness stops at ingestion.**
Once a DataFrame is in `DatasetStore`, nothing downstream
(profiling/cleaning/feature engineering/split/train/evaluate/guardrails)
can tell — or is allowed to be able to tell — which format it came from.
There is deliberately no per-format pipeline, no format flag on
`AgentState`, and no format branch anywhere in `graph.py`/`real_nodes.py`.
`TestFormatsAreEquivalentDownstream` pins this directly: the same
logical table uploaded in all six formats must produce equivalent stored
DataFrames.

**CSV behavior is unchanged.** The CSV branch is still literally
`pd.read_csv(io.BytesIO(raw))`, with the same zero-column/zero-row
rejections and the same HTTP status codes (400 unparseable / 422
parsed-but-empty). A dedicated test asserts the ingested frame is
identical to calling `pd.read_csv` directly on the same bytes. CSV
gained format *detection* around it, not new parsing behavior.

**Detection** is extension-driven against the closed
`FORMAT_EXTENSIONS` allowlist — an unsupported extension is a
structured `unsupported_format` error naming exactly what IS supported,
never a content-sniffing guess. A file whose contents don't match its
extension fails inside that format's reader with a clear parse error,
which is correct: PIPER never silently overrides what the filename claims.

**Per-format notes:**
- **Excel** — reads every worksheet so the user is always told what the
  workbook contained (`available_sheets`). Selection is explicit and
  reported, never silent: an explicitly requested `sheet_name` wins,
  otherwise the first NON-EMPTY sheet is used and the decision is
  surfaced in `notes`. An all-empty workbook is a clear error, not an
  empty DataFrame handed downstream.
- **JSON** — supports records (`[{...}]`), columnar (`{"a":[...]}`),
  pandas `split` orient, a records list under a `data`/`records`/`rows`/
  `items`/`results` wrapper key, and JSON Lines. Anything else
  (scalars, list-of-scalars, ragged arrays, nested non-tabular objects)
  is rejected with `unsupported_json_structure` describing what IS
  supported — never a best-effort flatten that would hand a misleading
  table downstream.
- **Parquet** — `pd.read_parquet` via pyarrow. Parquet stores dtypes in
  the file, so types are restored directly rather than re-inferred from
  text; a test asserts datetime64/bool/int32 survive Parquet but are
  genuinely lost through CSV.
- **IPYNB** — **never executes notebook code** (a notebook is untrusted
  input; executing it would be arbitrary code execution). Only
  already-saved cell OUTPUTS are read: pandas DataFrame HTML tables
  first, then `application/json` outputs; the largest recovered table
  wins. Two edge cases get explicit, honest errors rather than silent
  bad data:
  - *Truncated display* — Jupyter elides the middle of a large
    DataFrame. Detected via pandas' own `N rows × M columns` footer vs.
    the rows actually present, and rejected (`ipynb_output_truncated`)
    rather than ingesting a partial dataset.
  - *Missing external source* — a notebook with no saved table output
    that loads data via `pd.read_csv("...")` reports the referenced
    filename explicitly (`ipynb_external_source_missing`) so the user
    knows to upload that file instead.

  The notebook HTML table parser is **stdlib-only** (`html.parser`),
  not `pandas.read_html`, which would require lxml/bs4/html5lib. It
  targets one highly regular, machine-generated structure (pandas'
  own `to_html`), so a focused parser is sufficient and avoids a
  heavyweight dependency for a single feature — the same discipline
  as the stdlib-`urllib`-only `OllamaProvider` and the character-based
  context budgeter. Anything it can't confidently parse is an error,
  never a guess.

**New dependencies** (pinned in `requirements.txt`, verified working
against the existing pins): `openpyxl==3.1.5` (.xlsx), `xlrd==2.0.2`
(legacy .xls), `pyarrow==25.0.1` (Parquet). CSV/TSV/JSON/IPYNB need
none — pandas' own readers and the stdlib `json` module cover them.

**Error-code -> HTTP mapping** (`_INGESTION_STATUS_CODES` in
`app/api/routers/datasets.py`) preserves the original CSV contract:
unparseable -> 400, parsed-but-unusable-content -> 422.

**Frontend:** `DatasetUpload.tsx`'s `accept` list and extension check
now mirror the backend allowlist (client-side is UX convenience only;
the backend re-validates and remains the authority), and a panel below
the dropzone shows the **detected format and row × column dimensions**
after upload — plus the chosen Excel worksheet and any ingestion notes
— satisfying "show detected format and dataset dimensions before
running PIPER".

## Frontend (M6)

React + Vite + TypeScript + Tailwind CSS v4 + hand-rolled shadcn/ui-style
components (Radix primitives + `cva` + `tailwind-merge` — not the shadcn
CLI) + Recharts (model-comparison chart) + `react-router-dom`
(`BrowserRouter`, two routes: `/` and `/runs/:runId`).

- **Real API only** — every screen reads from the real FastAPI backend via
  `src/lib/api.ts` (typed REST client, `VITE_API_BASE_URL` env-configured,
  defaults to `http://localhost:8000`) and `src/lib/useRunEvents.ts` (real
  browser `EventSource` against `GET /runs/{id}/events`). No mock data in
  the shipped app — mocks exist only under `src/test/` (MSW handlers +
  fixtures matching the real backend contract), used exclusively by tests.
- **Pages:** `HomePage` (upload, dataset list/preview, run configuration),
  `RunPage` (live status, SSE event feed grouped by attempt with a REPLAN
  badge, terminal result view).
- **Results view (tabbed):** `ModelComparisonChart` (Recharts bar chart —
  F1/accuracy/precision/recall/ROC-AUC per candidate), `BaselinePanel`,
  `ValidationChecksPanel` (all guardrail checks, not just violations),
  `ReproducibilityPanel`, `FailurePanel` (full `FailureInfo`).
- **Dark mode** (post-Batch-7 addition) — `index.css` already defined a
  complete `.dark` OKLCH palette (`@custom-variant dark`); it wasn't wired
  to anything until now. `src/lib/useTheme.ts` toggles the `dark` class on
  `<html>` and persists the choice to `localStorage` (`piper-theme`,
  falling back to `prefers-color-scheme` on first visit);
  `src/components/ui/theme-toggle.tsx` is a sun/moon icon `Button` wired
  into both `HomePage` and `RunPage` headers. `index.html` carries a
  matching inline script that applies the stored/system theme before
  React mounts, so there's no flash of the wrong theme on load.
  `getInitialTheme()` guards against jsdom's missing `window.matchMedia`
  (real browsers always have it; the test environment doesn't). Verified
  live in the Dockerized frontend: toggle switches instantly on both
  pages, persists across a reload with no flash, and the frontend test
  suite (22/22) passes unchanged.
- **Select is a real Radix popover, not a native `<select>`** (fixed
  post-dark-mode: the target-column dropdown's native OS popup rendered
  white-on-white in dark mode). Setting CSS `color-scheme` was tried
  first and doesn't reliably work — Windows/Chrome still draws native
  `<select>` popups with the OS theme regardless. `src/components/ui/select.tsx`
  now wraps `@radix-ui/react-select` (added as a dependency; every
  other Radix-based component already followed this pattern, `Select`
  was the one holdout), so the popup is real HTML/CSS styled with the
  same `--card`/`--card-foreground` tokens as everything else — themed
  correctly on every OS/browser. `RunCreateForm.tsx` is the only
  consumer, updated to the composed `Select`/`SelectTrigger`/
  `SelectValue`/`SelectContent`/`SelectItem` API.
  `src/test/setup.ts` gained no-op polyfills for `hasPointerCapture`/
  `setPointerCapture`/`releasePointerCapture`/`scrollIntoView` — jsdom
  doesn't implement these, and Radix Select's open/select interactions
  depend on them; real browsers already have them. Verified: dropdown
  popup's computed background/text color match the dark palette exactly
  (not OS-native), all 21 real Telco columns render as selectable
  options, selection round-trips correctly, and the frontend suite
  (22/22, including the two integration tests exercising this exact
  form) passes with the interaction pattern updated from
  `userEvent.selectOptions()` to `click combobox -> click option`.

## Docker architecture (M7)

`docker-compose.yml` (repo root) defines exactly **two services** —
`backend` and `frontend`. No Ollama container, no database container.

- **`backend`** — `python:3.11-slim`. Build context is the **repo root**
  (not `backend/`), because `requirements.txt` lives there. Dockerfile at
  `backend/Dockerfile`; its ignore rules live at
  `backend/Dockerfile.dockerignore` (Docker's per-Dockerfile ignore-file
  convention, required specifically because the build context isn't the
  Dockerfile's own directory). Healthcheck: pure-Python `urllib` request
  against the real `GET /health`.
- **`frontend`** — multi-stage: `node:22-alpine` builds the static bundle
  (`VITE_API_BASE_URL` passed as a build ARG — Vite bakes it in at build
  time, must be a URL the **browser** can reach, e.g.
  `http://localhost:8000`, never a Docker-internal service name), then
  `nginx:1.27-alpine` serves it. `frontend/nginx.conf` adds SPA fallback
  (`try_files ... /index.html`) since the app uses `BrowserRouter`.
  Healthcheck: `wget --spider http://127.0.0.1:80/` — **must use
  `127.0.0.1`, not `localhost`** (see finding below).
- **Ports:** backend `8000:8000`, frontend `5173:80` (chosen to match the
  existing local-dev convention and avoid conflicts with other Docker
  projects on this machine using 5432/6333-6334/8080).
- **Environment** (overridable via a root `.env` file, all with sane
  defaults): `PIPER_OLLAMA_HOST`, `PIPER_LLM_MODEL`,
  `PIPER_OLLAMA_TIMEOUT_SECONDS`, `PIPER_CORS_ORIGINS`,
  `PIPER_PUBLIC_BACKEND_URL`, `PIPER_BACKEND_PORT`, `PIPER_FRONTEND_PORT`.

Run with `docker compose up --build`.

## Ollama configuration

Ollama runs **outside Docker**, directly on the host (Windows), exactly as
in local (non-Docker) development. Nothing in this project ever bundles
Ollama into a container.

- **Local dev (no Docker):** backend connects to `http://localhost:11434`.
- **Docker:** backend connects to `http://host.docker.internal:11434`
  (Docker Desktop resolves this automatically on Windows; `extra_hosts:
  host.docker.internal:host-gateway` is present in `docker-compose.yml`
  for portability to Linux Docker engines, where it's required).
- **Model:** `qwen3:4b` by default (`PIPER_LLM_MODEL`).
- **Timeout:** `PIPER_OLLAMA_TIMEOUT_SECONDS`, default **600.0s**
  (`app/llm/ollama_provider.py`'s `DEFAULT_TIMEOUT_SECONDS`).

### Timeout finding — resolved in Batch 5

The previous 150s default was set from a 5-run latency distribution (min
53.45s / median 64.51s / mean 74.72s / max 123.88s) measured against a
**small, 4-column synthetic dataset's** planning prompt. During Batch 4's
Docker verification, a run against the **full 21-column, 7,043-row real
Telco CSV** hit a genuine `provider_error_code: timeout` — one observation,
not itself a distribution, but a real signal.

Batch 5 collected a proper 5-run distribution against the same real Telco
CSV, through the actual production code path (`build_sanitized_llm_context`
-> `LLMPlanningContext` -> `OllamaProvider.generate_plan()`), real local
Ollama 0.32.6 + qwen3:4b, CPU inference:

    min:    143.56s
    median: 215.93s
    mean:   247.34s
    max:    418.24s
    stdev:  103.06s
    (5/5 successful)

The old 150s default sat **below the min** of this distribution — every
one of these 5 real calls but one would have timed out against a
realistically sized dataset, confirming this was never a rare edge case at
production scale. Fixed: `DEFAULT_TIMEOUT_SECONDS` raised to **600.0s**
(covers the observed max with ~44% margin), in both
`app/llm/ollama_provider.py` and `docker-compose.yml`'s
`PIPER_OLLAMA_TIMEOUT_SECONDS` default. Verified live: the real-Ollama
integration suite (5/5 tests, including a real `graph.invoke()` end to
end) and a manual Dockerized run against the full Telco CSV (see Batch 5
summary below) both completed real Ollama calls well within the new
budget with no timeouts.

## Planning model benchmark (Titanic workload) — MODEL DECISION: OPEN

Investigation-only session (post-Batch-7): benchmarked the current
production planning model, `qwen3:4b`, against the real Titanic dataset
(`benchmark_data/train.csv`, 891 rows × 12 columns, target `Survived`)
using an isolated, non-production script
(`backend/benchmark_planning_models.py`, never imported by `app/`) that
reuses the real `build_sanitized_llm_context()` /
`apply_context_budget()` / `build_planning_prompt()` /
`validate_proposed_plan()` / `canonicalize_plan()` functions so the
measurement is provably the same logic `plan_node_v2` runs in
production. Full report: `backend/benchmark_report.md`; raw data:
`backend/benchmark_results.json`.

**No production change was made** — model, `DEFAULT_TIMEOUT_SECONDS`
(600s), prompts, graph routing, retry logic, deterministic validators,
and the tool allowlist are all unchanged.

**Candidate availability** (`ollama list`): only `qwen3:4b` is
installed. `qwen3.5:4b`, `llama3.2:3b`, `gemma3:4b` are unavailable —
not auto-downloaded, per instruction. **No comparative model ranking
is possible from this session** — this is a single-candidate
measurement, not a benchmark bake-off.

**qwen3:4b results, 5 real Ollama calls (3 first-attempt + 2 REPLAN
follow-ups, same context every time):**
- Deterministic-validation-passed (valid plan) rate: **0/5 (0%)**.
- Structured-plan-produced rate: 4/5 (80%) — 1 call hit the current
  600s timeout with no response at all.
- Every completed attempt made the same mistake: `drop_column` called
  with `arguments.columns: [list]` (the shape `encode_categorical_features`/
  `scale_features` actually take) instead of the required singular
  `arguments.column: str` — a consistent argument-shape confusion, not
  random noise.
- Repeated-identical-invalid-plan rate on REPLAN (given the real
  rejected `tool_name`/`arguments` as evidence): **2/2 (100%)** — the
  model reproduced the byte-identical invalid plan both times. Had
  this gone through the real graph, the existing `DUPLICATE_PLAN`
  mechanism (see "REPLAN could repeat an already-rejected invalid plan
  forever" below) would have terminated it after the second rejection,
  confirming that safety net is still doing real work here.
- Avg latency (4 completed calls): 138.9s (Ollama `total_duration`),
  381 generated tokens, ~3.2 tokens/sec effective decode rate.

**Latency bottleneck, measured stage-by-stage:** context construction
(27.5ms) + prompt building (<2ms) are collectively ~4 orders of
magnitude smaller than the Ollama call itself. `eval_duration` (model
generation) is 69-99% of total latency in every completed call;
`prompt_eval_duration` is small on a first attempt (0.19-9.7s) but
grows to 30% of total on REPLAN (larger prompt from failure evidence).
**Root cause is CPU-bound autoregressive generation of `qwen3:4b`'s
thinking-mode reasoning trace, not context/prompt size or
structured-output formatting** — this reproduces the same mechanism
CLAUDE.md already documented for the 21-column Telco dataset, now
confirmed on an unrelated, smaller (12-column) dataset too, which rules
out dataset width as the primary driver.

**Recommendation:** the evidence justifies investigating an
alternative model (0% valid-plan rate + 100% repeated-invalid-plan
rate + one outright timeout, across two different real datasets this
session), but **no substitute can be recommended without a real
head-to-head comparison** — none of the three candidate alternatives
are installed. Next step (not started, needs explicit go-ahead):
install one candidate and re-run `benchmark_planning_models.py`
unmodified for a real comparison. Do not switch the production model
on this evidence alone.

**Verification:** relevant deterministic test files (`test_llm_provider.py`,
`test_llm_graph_integration.py`, `test_context_budget.py`,
`test_duplicate_plan_prevention.py`, `test_plan_canonical.py`,
`test_plan_diff.py`, `test_plan_validation.py`,
`test_replan_duplicate_invalid_plan.py`, `test_sanitized_llm_context.py`)
re-run under the project's `.venv` (`langgraph==1.2.10`, matching
`requirements.txt`) — **165 passed**, 0 failed. No production code was
touched, so this confirms no regression, not a fix. Full regression
suite intentionally not run (no production code changed).

### Update: qwen3.5:4b benchmarked against the qwen3:4b baseline — still OPEN

Controlled, single-candidate follow-up (same session policy: only one
new model per round, per explicit instruction). `qwen3.5:4b` was not
installed; checked disk space (346GB free), pulled **only**
`qwen3.5:4b` (3.4GB — nothing else downloaded or removed), then
re-ran the identical benchmark methodology (same Titanic fixture, same
context/prompt-building functions, same 3 first-attempt + up to 2
REPLAN-follow-up call budget) via `backend/benchmark_run_qwen35.py`.
Full comparison table: `backend/benchmark_report.md`.

**qwen3.5:4b, 5 real Ollama calls:** 0/5 (0%) passed deterministic
validation (same as baseline — still 0%), but 5/5 (100%) produced a
structured plan with zero timeouts (vs. baseline's 4/5, one timeout).
Violations per completed attempt averaged **3.00** (1–6 range) vs.
baseline's consistent **1.00** every time — qwen3.5:4b's failures are
broader and more erratic (a different hallucinated field-naming scheme
nearly every call — `method`, `columns_to_encode`, `column_names` —
none of which appear in the `ALLOWED OPERATIONS` the prompt actually
gives it), vs. qwen3:4b's single, narrow, structurally consistent
mistake. qwen3.5:4b did NOT repeat an identical invalid plan on either
REPLAN follow-up (0/2, vs. baseline's 2/2) and was ~14% faster on
average wall time (125.4s vs. 146.4s) with a ~23% faster decode rate
(3.97 vs. 3.23 tokens/sec).

**Applying the locked decision-criteria priority order** (1. correct
plan, 2. valid tool arguments, 3. no repeated malformed plans, 4.
reliable structured output, 5. latency, 6. resources): both models tie
on criterion 1 (neither ever produces a validator-passing plan), which
makes criterion 2 the deciding one — and qwen3.5:4b is clearly *worse*
there (3x more violations/attempt, less predictable, weaker adherence
to the given tool schema). Its real wins on criteria 3–5
(REPLAN-diversity, reliability, latency) don't outweigh a worse
showing on the higher-priority argument-correctness criterion, per the
explicit "do not choose it just because it's faster" rule.

**Verdict: Qwen3.5:4B did not establish a clear advantage; next
candidate should be Llama 3.2 3B.** No further models were benchmarked
this session (per instruction, no automatic cascade). No production
change was made. 165 relevant tests re-verified passing (268.41s)
after this comparison — no production code touched.

### Update: llama3.2:3b benchmarked against both Qwen candidates — still OPEN

Controlled, single-candidate follow-up. `llama3.2:3b` was not
installed; checked disk space (343GB free), pulled **only**
`llama3.2:3b` (2.0GB — `qwen3:4b`/`qwen3.5:4b`/everything else
untouched), then re-ran the identical benchmark methodology via
`backend/benchmark_run_llama32.py`. Full three-way comparison table:
`backend/benchmark_report.md`.

**llama3.2:3b, 5 real Ollama calls:** 0/5 (0%) passed deterministic
validation (still a three-way tie at 0%). Structured-plan-produced
rate 4/5 (80%, same as qwen3:4b) — but the one failure was a genuine
**HTTP 500 from Ollama** after 317s of wall time, not a clean timeout.
Violations per completed attempt averaged **2.50** (between qwen3:4b's
1.00 and qwen3.5:4b's 3.00), but one REPLAN attempt hallucinated 3
tool names that don't exist anywhere in `ALLOWED_TOOL_NAMES`
(`identify_categorical_columns`, `select_columns`, `onehot_encode`) —
a categorically more severe schema violation than either Qwen
candidate ever produced (both always at least named a real tool).
Repeated-identical-invalid-plan rate on REPLAN: 0/2 (matches
qwen3.5:4b, better than qwen3:4b's 2/2). Clearly the fastest of the
three (82.3s avg vs. 125.4s / 146.4s) and highest decode rate
(4.90 tok/s), consistent with being the smallest (2.0GB), non-
thinking-mode model of the three.

**Applying the locked decision-criteria priority order** (1. correct
plan, 2. correct tool arguments/schema, 3. reliability, 4. low
repeated-invalid-plan rate, 5. latency, 6. resources): all three tie
on criterion 1. Criterion 2 — the deciding one — still favors
**qwen3:4b**: its failure mode remains the narrowest and most
predictable of the three; llama3.2:3b's numerically-middling violation
count is undercut by introducing a new, more severe failure class
(fabricated tool names outside the allowlist) neither Qwen model
exhibited. llama3.2:3b's genuine wins on criteria 4-6 don't overturn a
worse showing on the higher-priority criterion 2, per the "do not
select solely because it's faster" rule.

**Verdict: Llama 3.2 3B does not clearly outperform both Qwen
candidates.** Per instruction, Gemma 3 4B was not automatically
benchmarked. **Next candidate, pending explicit go-ahead: Gemma 3 4B.**
No production change was made.

### Update: gemma3:4b benchmarked against all three prior candidates — still OPEN, all 4 locked candidates now tested

Controlled follow-up, explicit go-ahead given. `gemma3:4b` was not
installed; checked disk space (339GB free), pulled **only** `gemma3:4b`
(3.3GB — everything else untouched), then re-ran the identical
methodology via `backend/benchmark_run_gemma3.py`. Full four-way
comparison: `backend/benchmark_report.md`.

**gemma3:4b, 5 real Ollama calls:** 0/5 (0%) passed deterministic
validation (four-way tie at 0%). Violations per completed attempt
averaged **5.60** (5–6 range) — the **worst of all four candidates**
tested this session (vs. qwen3:4b's 1.00, llama3.2:3b's 2.50,
qwen3.5:4b's 3.00) — nearly every proposed step in every attempt
failed validation, consistently using a wrong field name
(`column_names`) across almost all tool calls. Repeated-identical-
invalid-plan rate: 1/2 (50%). Latency was by far the worst of any
candidate: mean wall time **472.5s** (vs. 82.3–146.4s for the other
three), with **two individual calls (909.4s, 786.2s) exceeding the
current 600s production timeout in wall-clock terms** — yet neither
registered as a `timeout` error.

**New finding, not fixed (out of this investigation's scope):**
Python's `urllib.request.urlopen(..., timeout=600.0)` — the exact
mechanism `OllamaProvider.generate_plan()` uses — bounds how long a
single blocking read can wait for *more* data, not total request
duration; if the connection keeps receiving any bytes before the full
body is assembled, the countdown resets rather than accumulating.
Both of gemma3:4b's over-600s calls apparently exhibited this, which
means **PIPER's documented 600s timeout may not be the hard ceiling on
total planning latency it's assumed to be**. This was never observed
before because gemma3:4b is the first candidate slow enough to expose
it. Needs its own separate investigation and explicit go-ahead before
any fix — not addressed here per this session's investigation-only
scope. Also observed: one call's `load_duration` was 500.1s (vs. 2.5–
2.9s everywhere else this session) — consistent with host-level memory
pressure from four different multi-GB models now resident from this
session's sequential benchmarks, not necessarily a gemma3:4b-specific
defect; not investigated further.

**Applying the locked decision-criteria priority order:** all four tie
on criterion 1. Criterion 2 (the deciding one) is not just unfavorable
to gemma3:4b — it's the worst result of any candidate tested this
session, and gemma3:4b also loses decisively on latency (criterion 5).
No dimension favors gemma3:4b over qwen3:4b at a priority level that
matters under the locked order.

**Verdict: Gemma 3 4B does not establish an advantage over qwen3:4b or
any other candidate.** This completes the originally-scoped
four-candidate list. **Across all four, and 20 real Ollama calls this
session, no candidate has ever produced a plan that passes
`validate_proposed_plan()`.** `qwen3:4b` remains the best performer on
the deciding criterion throughout, despite its own real problems (100%
repeated-identical-invalid-plan rate, one timeout). No production
change was made.

**Model decision remains OPEN.** No further models were benchmarked
without an explicit go-ahead.

## Planner-contract fix: the prompt never described tool arguments — resolved

Root-cause investigation triggered by the 4-model benchmark's
cross-cutting result (0/20 real calls ever produced a
`validate_proposed_plan()`-passing plan, across two model families and
two generations). Before concluding this was a model-capability
problem, the planner ↔ tool-schema contract itself was inspected end
to end: `ollama_provider.py`, `plan_node_v2`, `prompts.py`,
`sanitized_llm_context.py`, `plan_validation.py`, `plan_canonical.py`,
and the existing test suite.

**Root cause found:** the prompt's `=== ALLOWED OPERATIONS ===` section
rendered `context.allowed_operations` as nothing but a bare JSON array
of tool_name strings —

```
[
  "convert_column_type",
  "drop_column",
  "encode_categorical_features",
  "impute_missing_values",
  "scale_features"
]
```

— and `=== DETERMINISTIC CONSTRAINTS ===` only said arguments "must
match that operation's required shape" **without ever stating what
that shape is anywhere in the prompt**. No argument names, types,
required/optional status, enum values, or worked examples were ever
communicated to the LLM, for any of the five tools, at any point.
Confirmed by literally re-rendering the exact production prompt and
inspecting the section byte-for-byte — this is not an inference, it's
what every one of the 20 real Ollama calls this session actually
received. Every observed failure pattern is explained directly by this
gap: `drop_column` given a `columns` list (indistinguishable from
`encode_categorical_features`/`scale_features`'s real plural shape,
absent any documentation saying otherwise), and invented field names
(`method`, `column_names`, `columns_to_encode`, `columns_to_drop`) that
a model has no way to know are wrong when the real names were never
stated. Four independently-trained models (two families, two Qwen
generations) converging on the same class of guessing failure is
strong cross-model evidence this was a genuine contract gap, not
individual model weakness.

**Ruled out** (see `backend/benchmark_report.md` for the full
per-question walkthrough of all 11 investigation angles): Ollama's
`format`-constrained JSON decoding and `_extract_content()`/
`_strip_markdown_fences()`/`ProposedPlan.model_validate()` parsing all
worked correctly in every completed call this session
(`structured_plan_produced` was true for all but 3 of 20 calls, and
those 3 failed at the transport level — timeout/HTTP 500 — not
parsing); thinking-mode reasoning never leaked into the structured
JSON output; the parser never altered a model's actual field
names/values. `PLAN_JSON_SCHEMA` (Ollama's `format` field) only
constrains the outer envelope shape — `arguments` is declared as an
unconstrained `{"type": "object"}` — so it cannot enforce per-tool
argument correctness at the decoding level even in principle; making
it a discriminated, tool_name-conditional schema was considered but
deferred (uncertain, inconsistently-supported Ollama/llama.cpp grammar
feature across backends — a bigger, riskier change than the "smallest
correction necessary" scope called for this round).

**Fix — additive, documentation-only, zero validation-logic changes:**

- `app/agent/plan_validation.py` — new `TOOL_ARGUMENT_SCHEMAS: dict`
  constant: a declarative, LLM-facing description (argument name,
  type, required-ness, enum values, a short note, one worked example)
  for each of the five tools, hand-verified against
  `_validate_step_arguments()`'s real logic. `validate_proposed_plan()`
  itself is completely unchanged.
- `app/llm/provider.py` — `LLMPlanningContext` gained one new,
  additive field: `tool_schemas: dict = Field(default_factory=dict)`.
  Every existing caller that only sets `allowed_operations` (every
  pre-existing test fixture) is unaffected — default empty dict,
  `extra="forbid"` only blocks *unknown* fields, not new declared ones.
- `app/llm/prompts.py` — new `_format_allowed_operations()` helper:
  renders the full per-tool contract with examples when `tool_schemas`
  is populated; falls back to the original bare-list rendering
  (byte-for-byte, pinned by a regression test) when it's empty. Used
  by both `build_planning_prompt()` and `build_replan_prompt()`.
- `app/agent/nodes/real_nodes.py` — `plan_node_v2` now passes
  `tool_schemas=TOOL_ARGUMENT_SCHEMAS` into `LLMPlanningContext` (one
  import, one line) — the only call site wired to the real contract;
  every other caller keeps getting the plain list unless it opts in.

**Why this is architecturally correct, not a workaround:** it changes
*what the LLM is told*, never *what is trusted or executed*.
`validate_proposed_plan()` remains the sole, unweakened authority — an
invalid plan is still rejected exactly as before (pinned by new tests
reproducing the literal real-world failures and asserting they're
still rejected). The tool allowlist, canonicalization, duplicate-plan
detection, retry bounds, guardrails, downstream execution, the 600s
timeout, and the production model are all completely untouched.
`TOOL_ARGUMENT_SCHEMAS` is prompt content only, sourced from — and
tested against — the same module that already owned the real contract;
it introduces no new source of truth and cannot silently drift from
the validator (see the anti-drift test below).

**New test coverage (29 tests, all new):**
- `tests/test_plan_validation.py` — `TestToolArgumentSchemasMatchValidator`
  (11 tests): schema keys exactly match `ALLOWED_TOOL_NAMES`; every
  documented example is round-tripped through the real
  `validate_proposed_plan()` and asserted valid (parametrized over all
  5 tools); documented enums asserted equal to the validator's own
  `_VALID_IMPUTE_STRATEGIES`/`_VALID_CONVERT_TARGET_TYPES` constants;
  `drop_column`'s singular-string contract and
  `encode_categorical_features`/`scale_features`'s list contract
  explicitly pinned. `TestRealWorldBenchmarkFailurePatternsRejected`
  (10 tests): named regression tests reproducing the *literal* observed
  failures from all four benchmarked models (plural `columns` for
  `drop_column`, empty `column`, `columns_to_encode`/`columns_to_drop`/
  `method`/`column_names` invented fields, hallucinated tool names) —
  proving they were already, and remain, correctly rejected.
- `tests/test_llm_provider.py` — `TestToolSchemaRenderedIntoPrompt`
  (4 tests): `tool_schemas` defaults to `{}`; the no-schema fallback
  renders byte-identically to the pre-fix bare list; the with-schema
  path renders every tool name, argument, and example; REPLAN prompts
  carry the schema too.
- `tests/test_planner_contract_titanic.py` (new file, 4 tests): built
  from the real `benchmark_data/train.csv` fixture (891×12, target
  `Survived`) via the same `build_sanitized_llm_context()`/
  `apply_context_budget()` path `plan_node_v2` uses. Proves the real
  production prompt for this exact dataset now documents the real
  argument contract; a hand-built plan using ONLY what the documented
  schema says passes `validate_proposed_plan()` cleanly (the
  deterministic boundary is genuinely satisfiable, not just
  theoretically); and the literal invalid plan qwen3:4b produced
  against this exact dataset during the real benchmark is still
  correctly rejected.

**Test results:** targeted (96/96), the broader relevant planner/
Ollama-provider set (194/194), then one full `pytest -q` run (changed
real production code in the core planning path, so the standing "full
suite unless infrastructure-only" rule applied) — **770 passed, 5
skipped** (up from the documented 741 baseline by exactly +29, matching
the new tests added; identical 5 skips, the Ollama-gated integration
suite), zero failures, 45m58s. No regressions.

**Is the planner contract ready for another real-model benchmark?**
Yes, structurally — every model now receives the actual argument
contract it needs instead of guessing from bare tool names. **This was
not re-verified against a real model this session** (no model was
re-benchmarked, per explicit instruction) — whether it measurably
raises the valid-plan rate above 0% for any of the four already-tested
candidates is an open empirical question for the next benchmark round,
not something this investigation can claim in advance. The model
decision remains **OPEN**.

**Explicitly not changed:** deterministic validation logic, the tool
allowlist, canonicalization/duplicate-plan detection, retry/REPLAN
routing, guardrails, downstream ML execution, the Ollama timeout, and
the production model are all untouched. `PLAN_JSON_SCHEMA`'s
unconstrained `arguments` shape was investigated but deliberately left
alone this round (see "ruled out" above).

### AFTER-fix re-benchmark: qwen3:4b, 0% → 100% valid-plan rate

Controlled, single-candidate, real-Ollama re-benchmark of `qwen3:4b`
only (qwen3.5:4b/llama3.2:3b/gemma3:4b intentionally not re-tested this
round). Same Titanic fixture, same call budget (3 first-attempt +
conditional REPLAN follow-ups), same deterministic validator — the only
intentional difference: the prompt now includes `tool_schemas`, because
that's the fix under test and what `plan_node_v2` now actually sends.
`backend/benchmark_planning_models.py` was updated (both the
first-attempt and REPLAN-follow-up context construction) to include
`tool_schemas=TOOL_ARGUMENT_SCHEMAS`; a new isolated runner,
`backend/benchmark_run_qwen3_after_fix.py`, wrote results to a
**separate** file (`benchmark_results_after_fix.json`) so the original
BEFORE baseline was never overwritten (confirmed intact: still 5
trials, 1 violation each). Full detail:
`backend/benchmark_report.md`.

**Result: 3/3 real calls (100%) passed deterministic validation** — up
from the original baseline's 0/5 (0%). Zero REPLAN follow-ups were
needed (they only trigger on an invalid attempt, and none occurred —
itself part of the result). Every `drop_column` call across all three
trials used the correct singular `{"column": "..."}` shape — never
once the plural `columns` list that was qwen3:4b's single, 100%-
consistent mistake before the fix. The proposed plans were also
qualitatively sensible for Titanic (impute the genuinely-missing `Age`
column, drop high-cardinality/identifier columns, encode the real
categorical predictors, scale the real numeric ones) — a genuine
improvement in plan quality, not merely schema compliance. Mean
violations per attempt: 1.00 -> **0.00**.

**Honest caveats:** a 3-call sample from a model with already-
documented high latency/output variance; latency did not improve (one
call again exceeded the 600s timeout in wall-clock terms via the same
already-documented `urllib` read-timeout nuance — not a new
regression, and not something this round attempted to fix, since
latency optimization was explicitly out of scope); whether 100% holds
over more calls, and whether the fix also helps the other three
candidates, are open questions for a future benchmark round.

**Decision (per the locked branches): valid-plan reliability improved
substantially (0% -> 100%), so the planner-contract fix is documented
as having resolved the primary failure mode. `qwen3:4b` remains the
current development baseline. No model switch was made.**

**Test results:** no production code changed this round (only the
benchmark harness). The directly relevant planner-contract test files
were re-run as a confirmatory check anyway —
`test_plan_validation.py`, `test_llm_provider.py`,
`test_planner_contract_titanic.py`, `test_llm_graph_integration.py`:
**117 passed**, 0 failed. Full regression suite not re-run (no
production code changed).

**Model decision remains OPEN** for the other three candidates
(qwen3.5:4b/llama3.2:3b/gemma3:4b were not re-benchmarked against the
fixed contract) — but the specific, cross-model failure mode that
motivated this whole investigation is now resolved for the current
production model.

## Engineering hardening — Phase 1: planner latency measurement (investigation only, no code changed)

Instrumented the real planner/Ollama path stage-by-stage against the
real Titanic fixture, using the current, post-contract-fix production
prompt (`tool_schemas` included). No production code was changed —
measurement only, via a new isolated script,
`backend/benchmark_measure_stages.py`. Combined with the raw
per-stage data already captured across the BEFORE/AFTER benchmark
rounds (8 real Ollama calls total with full `ollama_stats`), the
following is directly measured, not inferred:

**GPU vs CPU (verified, not assumed):** `ollama ps` during a real
inference call reports `PROCESSOR: 100% CPU`. The only GPU on this
machine is an integrated Intel UHD Graphics adapter — no discrete
NVIDIA/AMD GPU is present. Ollama is running qwen3:4b entirely on CPU;
there is no GPU acceleration to enable on this hardware.

**Where PIPER spends time (uncontested, every data point agrees):**
dataset load + `build_sanitized_llm_context()` + `apply_context_budget()`
+ `LLMPlanningContext` construction + `build_planning_prompt()`
together cost **~20ms**, and response parsing + `validate_proposed_plan()`
cost **under 1ms** — both consistently **<0.01% of total call time**.
PIPER's own code is not a meaningful latency contributor at this
dataset scale. Effectively 100% of wall time is the Ollama HTTP call
itself (confirmed: script-measured wall time and Ollama's own
`total_duration` agree within ~2s every time — negligible transport
overhead for a localhost call).

**Within the Ollama call, two components matter — not just
generation:**

| Component | Observed range (8 real calls) | Share of total when high |
|---|---:|---:|
| `load_duration` (model load) | 0.65s (warm) – 14.12s (cold) | up to ~5% |
| `prompt_eval_duration` (processing the prompt) | 0.19s – 157.75s | **up to 53% — sometimes the LARGEST single component** |
| `eval_duration` (generating the output) | 97.4s – 521.3s | typically 40–95%, but not always dominant |

**Prompt processing is not reliably cheap** — this revises the
Batch-5-era assumption that generation alone dominates. The clearest
pattern in the data: **prompt_eval_duration correlates with how much of
the prompt is NEW versus a repeat of the immediately-preceding call's
prompt to the same loaded model**, not simply with prompt size.
Two consecutive BYTE-IDENTICAL prompts (AFTER-fix run,
`first_attempt_1` -> `first_attempt_2`, ~2247 tokens both times) show
`prompt_eval_duration` of 0.21s and 0.22s — near-instant. A REPLAN
prompt sharing a long prefix but appending new failure evidence (BEFORE
run, ~2387-2395 tokens) costs 45.6-47.9s. A prompt sent right after a
fresh model (re)load costs 123.6-157.8s for a similar (~2247-token)
prompt. This is consistent with Ollama/llama.cpp's prompt-prefix
KV-cache reuse for a repeated prompt against an already-loaded model —
plausible given the pattern, but not independently confirmed via
Ollama's own diagnostics, and this is a shared dev machine so some of
the variance could also be ordinary CPU contention noise. Both cold
model-load (`load_duration`) and cold prompt-eval consistently co-occur
on the first call after the model has been idle/unloaded.

**Reasoning vs. final-answer token split cannot be cleanly measured via
Ollama's API.** One fresh capture inspected the raw response body's
`thinking`/`response` fields directly: `response` was empty and the
*entire* 1,464-character/161-word completion (confirmed to contain a
valid, successfully-parsed 5-step plan) landed in `thinking`. Ollama
reports only a single combined `eval_count` (360 tokens here) — it does
not separately count reasoning vs. answer tokens, and this model/setup
does not reliably separate them into the two fields the way the field
names imply. A rough content-length estimate (a 4-7-step JSON plan
serializes to a few hundred characters, well under half of 1,464)
suggests a majority of generated tokens are reasoning/preamble PIPER's
deterministic system never consumes — but this is an estimate, not a
precise measurement.

**Token counts:** prompt tokens grew from ~1,842 (pre-contract-fix) to
~2,247 (post-fix, +22%) — the deliberate, evidence-justified cost of
`TOOL_ARGUMENT_SCHEMAS` (this is what raised the valid-plan rate from
0% to 100%, not bloat). Output tokens are roughly flat across every
measured call regardless of the fix (305-475 range, no clear
before/after shift) — the fix changed plan *correctness*, not
generation *length*.

**Is the prompt larger than necessary?** No evidence of unnecessary
bloat found. The sanitized dataset context for this 12-column dataset
is 2,780 chars — comfortably under the 8,000-char budget (budgeting
never activates), and every field in it (dtype, missing/unique %,
capped sample values, numeric min/max/mean) is plausibly
planning-relevant, not obviously redundant. No changes proposed to
context/prompt content without evidence, per the standing "don't remove
information merely because it's large" rule.

**Proposed optimization plan for Phase 2 (NOT implemented — reported
for review; ranked by expected-impact/risk):**
1. **Tune Ollama's `keep_alive` request parameter** (currently unset in
   `OllamaProvider`, defaulting to Ollama's standard ~5-minute unload).
   Directly targets the two clearly-identified, highest-cost, avoidable
   components (cold `load_duration` + cold `prompt_eval_duration| — in
   the worst observed case together larger than the entire generation
   phase). Single-field, zero-architecture-risk change confined to
   request payload construction — touches no validation/routing/retry/
   timeout logic. Must be measured before/after, since current evidence
   is observational, not a controlled A/B test.
2. **Investigate whether Ollama/qwen3 exposes a reasoning-budget or
   `think` control** for this model/Ollama version, to reduce
   reasoning-token volume without disabling thinking mode outright (the
   fix must preserve — not compromise — the 100% valid-plan rate just
   established; would need explicit before/after re-verification against
   the real benchmark before being trusted).
3. Prompt/context minimization — **not currently justified by evidence**
   at this dataset scale (context-side cost is already ~20ms; no
   demonstrated redundancy found). Would need a wider dataset
   (budgeting-triggering) to even be measurable.
4. Planning cache/reuse across REPLAN attempts — a separate axis from
   raw per-call latency; would help REPLAN-heavy runs specifically, not
   the base/first-call cost.

No code was changed in this phase. Model, timeout, prompts, graph
routing, retry logic, and deterministic validation remain untouched.
Awaiting explicit go-ahead before starting Phase 2 implementation.

## Engineering hardening — Phase 2A: `keep_alive` — measured and implemented

Controlled real-Ollama experiment testing exactly one variable
(`keep_alive`) against the current production prompt (post-contract-fix,
byte-identical across every trial). Full detail:
`backend/benchmark_report.md`. Script:
`backend/benchmark_keep_alive_experiment.py`.

**Method:** 6 real calls. Group A (no explicit `keep_alive`, Ollama's
5-minute default applies): forced-cold call -> immediate follow-up ->
forced-cold again -> **330s wait** (past the 5-minute default) ->
call after the gap. Group B (explicit `keep_alive="30m"`): forced-cold
call -> the same 330s wait -> call after the gap. `ollama ps` was
polled immediately before the critical calls for **ground-truth**
residency confirmation, not inferred from latency.

**Result — the cleanest comparison (identical 330s gap, identical
prompt, only `keep_alive` differs):** under the default, the gap
evicted the model (`ollama ps` confirmed NOT resident) and the next
call cost **690.4s**; with explicit `keep_alive=30m`, the model
survived the identical gap (`ollama ps` confirmed resident, "24
minutes from now") and the next call cost **186.0s** — a **73.1%
wall-time reduction**, directly attributable to residency, not noise.
Warm `prompt_eval_duration` was consistently ~0.2-0.3s across every
confirmed-resident trial; cold `prompt_eval_duration` ranged
159.58s-1010.00s — highly variable (plausibly shared-machine page-cache
contention from this session's many model loads/unloads), which if
anything strengthens the case for avoiding cold starts altogether:
the cost is not just high, it is unpredictable enough to occasionally
be extreme. All 6 calls produced valid plans (9/9 real calls valid
since the contract fix).

**Decision: substantial, confirmed improvement — implemented.**

| File | Change |
|---|---|
| `app/llm/ollama_provider.py` | New `DEFAULT_KEEP_ALIVE = "10m"`; `OllamaProvider.__init__` gained `keep_alive` (env `PIPER_OLLAMA_KEEP_ALIVE`, same override precedence as `host`/`model`/`timeout_seconds`); `generate_plan()`'s payload now sends `"keep_alive": self.keep_alive`. |
| `docker-compose.yml` | `PIPER_OLLAMA_KEEP_ALIVE: ${PIPER_OLLAMA_KEEP_ALIVE:-10m}` added. |

10 minutes (not the 30m tested) matches `DEFAULT_TIMEOUT_SECONDS` —
covers a realistic REPLAN gap without keeping the model resident
indefinitely for no reason. No shutdown-hook changes needed (Ollama
manages residency independently of PIPER's process). Deterministic
validation, routing, retries, and execution are all untouched.

**New test coverage:** 4 new tests in `test_llm_provider.py`
(`TestOllamaProviderConfiguration`) covering the default/env-var/
constructor-arg precedence and the "never silently disabled" invariant,
plus the existing wire-request test extended to assert `keep_alive`
reaches the real payload, not just the instance attribute.

**Test results:** `test_llm_provider.py` alone 46/46; broader relevant
set 198/198; full `pytest -q` run once (production code changed in the
core planner transport path): **774 passed, 5 skipped** (770 baseline
+4, exactly matching the new tests), 0 failures, 29m32s. No
regressions.

**Next highest-impact candidate (not started):** generation time
itself (`eval_duration`, ~110-235s, now the dominant remaining cost)
— investigate whether Ollama/qwen3 exposes a reasoning-budget/`think`
control to reduce non-essential reasoning-token volume without
disabling thinking mode or compromising the 100% valid-plan rate.
Needs its own explicit go-ahead.

## Engineering hardening — Phase 2B: `think` parameter — NEGATIVE RESULT, not adopted

Investigated whether Ollama 0.32.9 / qwen3:4b exposes a usable
generation-reduction control. `ollama show qwen3:4b` confirms
`thinking` as a first-class capability, and the `think` request field
IS honored — but it does not do what the latency hypothesis needed.

**Verified controls that exist:** `think` (bool), `num_predict`,
`stop`, and standard sampling params (`temperature`/`top_k`/`top_p`).
qwen3:4b ships defaults `temperature=0.6, top_k=20, top_p=0.95,
repeat_penalty=1` plus `<|im_start|>`/`<|im_end|>` stop tokens. No
"reasoning budget" parameter exists for this model/version.

**Controlled A/B (4 real calls, real Titanic prompt, only `think`
varied — `backend/benchmark_generation_control_experiment.py`):**

| Metric | Baseline (default) | `think=false` | Δ |
|---|---:|---:|---|
| Generation (`eval_duration`) | 139.9s | 150.5s | +10.6s (no gain) |
| Output tokens | 382.5 | 411.5 | +29 (more) |
| Tokens/sec | 2.73 | 2.73 | unchanged |
| Valid plans | 2/2 | 2/2 | preserved |
| Reasoning text location | `thinking` (1414–1883 ch) | `response` (1798–1684 ch) | **relocated** |

**Root finding: `think=false` relocates reasoning, it does not
eliminate it.** The model emits the same volume of reasoning prose
either way; only the response-envelope field changes. Generation was
marginally slower with more tokens — within this model's documented
variance, so the honest reading is "no benefit," not "actively worse."

**Conclusion: not adopted. Production sends no `think` field
(unchanged).** Generation cost is intrinsic to CPU-bound decoding for
this model, not toggleable overhead — so the remaining latency lever is
model choice or hardware, not a reasoning switch. A separate, untested
option is capping `num_predict`, but that risks truncating a valid plan
mid-JSON and was not attempted.

## Post-contract model comparison: qwen3:4b vs qwen3.5:4b — INCONCLUSIVE

The original four-model ranking ran BEFORE the planner-contract fix, so
it could not separate model capability from the underspecified prompt.
This re-runs the two Qwen candidates against the corrected contract.
Harness: `backend/benchmark_post_contract_comparison.py`; analysis:
`backend/benchmark_post_contract_analyze.py`; results in the isolated
`backend/benchmark_results/post_contract/` namespace (the original
pre-contract results are untouched).

**Method:** 3 trials/model, 6 real calls. Each trial runs the full
production PLAN REPLAN loop (real `build_replan_prompt()`, real
`FailureInfo` with `rejected_steps` evidence, real
`canonicalize_plan()`/`plan_history`/`DUPLICATE_PLAN` semantics,
bounded by `max_retries=2`). Symmetric cold/warm protocol per model:
evict all → force cold → verify via `ollama ps` → Trial 1 (cold) →
Trials 2–3 (warm, no artificial delay). Residency is recorded from
`ollama ps` ground truth at every trial, not inferred. Byte-identical
prompt across both models (SHA-256 pinned and re-verified after each
block). Harness REPLAN logic was smoke-tested against a fake local
server for all four outcomes (valid-first, invalid→valid,
repeated-invalid→`DUPLICATE_PLAN`, varied-invalid→budget exhaustion)
before spending real Ollama time.

**Documented confound (user-approved):** the two models ship DIFFERENT
sampling defaults — qwen3:4b `temperature=0.6`, qwen3.5:4b
`temperature=1.0` (plus `presence_penalty=1.5` and no stop tokens).
PIPER sends no `options` field, so each model runs on its own defaults.
This is production-realistic (it measures what PIPER would actually
get), but means any difference is attributable to
**model + its shipped sampling config jointly**, not model capability
in isolation. Captured in `comparison.json`'s fingerprint.

**Results — validity is a TIE, latency favors qwen3.5:4b:**

| Metric | qwen3:4b | qwen3.5:4b |
|---|---:|---:|
| First-attempt valid | **3/3** | **3/3** |
| Final valid | **3/3** | **3/3** |
| REPLAN rate | 0/3 | 0/3 |
| Validation violations | 0 | 0 |
| Timeouts / technical failures | 0/3, 0/3 | 0/3, 0/3 |
| Mean time-to-valid-plan | 234.5s | **124.7s** |
| Median time-to-valid-plan | 271.6s | **105.4s** |
| Mean generation latency | 184.6s | **60.4s** |
| Mean prompt-processing latency | 44.0s | 56.0s |
| Mean tokens/sec | 2.63 | **3.88** |
| Mean output tokens | 445 | 231 |
| Cold / warm / warm | 302.9s / 271.6s / 128.9s | 250.7s / 105.4s / **18.1s** |

**Plan completeness — the decisive counterweight.** PIPER has no
built-in plan-quality scorer, so the analysis script defines coverage
checks from objectively measured Titanic properties (Age 19.9% missing;
PassengerId/Name ≥90% unique; Sex/Embarked low-card categoricals;
Name/Ticket/Cabin high-cardinality text). These are descriptive
coverage facts, **not** a validated quality metric:

| Check | qwen3:4b | qwen3.5:4b |
|---|---:|---:|
| Target never touched (safety) | 3/3 | 3/3 |
| Imputes `Age` (19.9% missing) | **3/3** | 2/3 |
| Encodes `Sex`+`Embarked` | 3/3 | 3/3 |
| Applies scaling | **3/3** | 1/3 |
| Drops `Name` | **3/3** | 0/3 |
| Drops `Ticket` | **3/3** | 1/3 |
| Drops `PassengerId` | 0/3 | **2/3** |
| Drops `Cabin` (77% missing) | 0/3 | 0/3 |
| Steps per plan | **[5, 5, 5]** | [3, 5, 1] |

**qwen3.5:4b's speed advantage is causally entangled with doing less
work, not doing the same work faster** — it emits roughly half the
output tokens (231 vs 445) and its plans vary from 5 steps down to 1.
Its fastest trial (18.1s) produced a single-step plan that never
imputes `Age`. That plan is fully deterministic-validation-VALID —
which is the important architectural lesson: **PIPER's validator
enforces well-formedness, not completeness.** A valid plan can still be
a substantively poor one, and no current guardrail catches that.

**Shared blind spot (model-independent, useful for PIPER regardless of
model choice):** neither model ever handled `Cabin` (77% missing, 147
unique) in any of the 6 trials, and each leaves at least one
identifier-like column in — with complementary gaps (qwen3:4b always
drops Name+Ticket but never PassengerId; qwen3.5:4b usually drops
PassengerId but rarely Name/Ticket).

**Verdict: INCONCLUSIVE — no model switch.** Criterion 1 (final
valid-plan rate) is a tie, so it does not separate the candidates.
qwen3.5:4b wins every latency criterion, but that win is not free: its
plan completeness is erratic ([3,5,1] vs a perfectly consistent
[5,5,5]), including one substantively deficient plan, and the
`temperature=1.0` vs `0.6` confound plausibly explains that variance
but is unresolved. With n=3 this is an engineering bake-off, not a
statistically significant result. **`qwen3:4b` remains the development
baseline.**

**Minimum next experiment (not started, needs go-ahead):** 5 more
qwen3.5:4b trials at its own defaults (n=8 total) to establish whether
the plan-completeness variance is real or a small-sample artifact. If
it persists → keep qwen3:4b. If the next 5 are all complete → the
single 1-step trial was an outlier and qwen3.5:4b becomes a strong
candidate, at which point a temperature-matched arm (0.6) would isolate
the sampling confound. No production code was changed by this
comparison.

## Plan Adequacy — deterministic plan-completeness layer (COMPLETE)

Closes the structural gap the post-contract benchmark exposed:
**PIPER's validator proved a plan was well-formed, never that it was
sufficient.** A real recorded plan — qwen3.5:4b's single-step
`encode_categorical_features(["Sex","Embarked"])` — passed every
existing check while leaving `Age` (19.87% missing) entirely
unaddressed.

    LLM plan
        v
    validate_proposed_plan()   VALIDITY  (unchanged, still sole authority)
        v
    duplicate-plan detection   IDENTITY  (unchanged)
        v
    evaluate_plan_adequacy()   ADEQUACY  (new)
        v
    execution

### Verified V1 execution contract (measured, not assumed)

Empirically verified against this checkout's real sklearn (1.5.2) and
`train_model()`'s actual pipeline shape:

| Fact | Verified behavior |
|---|---|
| Feature-set membership | A column is a training feature **iff** it appears in `encode_categorical_features` or `scale_features`. `_feature_intent_from_plan()` reads exactly those two tools; `train_model()` uses `remainder="drop"`, so unlisted columns never reach the model. |
| `StandardScaler` + NaN | **Preserves NaN** → NaN enters the training matrix. |
| `LogisticRegression` + NaN | **Raises** `ValueError: Input X contains NaN`. |
| `RandomForestClassifier` + NaN | **Tolerates** NaN natively on sklearn 1.5.2. |
| `OneHotEncoder` + NaN | Treats NaN as **its own category** (`categories_` contains `nan`); emits a NaN-free 0/1 matrix — no crash. |
| `train_model()` error handling | No `try/except` around `pipeline.fit()`, so the ValueError escapes as an unhandled exception. |

**Correction to the premise this task was specified under:** it is NOT
true that both candidates reject NaN. Random Forest accepts it, and
OneHotEncoder absorbs categorical NaN as a category. The genuine crash
condition is narrow: **numeric + scaled + LogisticRegression**.

**The implemented rule is deliberately stricter than that crash
condition**, and that is an explicit design choice, not an oversight:

- A rule keyed to estimator NaN-tolerance would be *model-aware* — the
  graph would need estimator internals to judge a plan, inverting
  PIPER's core principle that deterministic code decides and a
  swappable component's quirks never leak into planning.
- The categorical case is not "safe", it is **silent**: it converts
  "value was missing" into a predictive one-hot category. Legitimate
  when chosen deliberately; never acceptable as an accident of not
  mentioning the column.
- Estimator-keyed rules silently change meaning when sklearn changes
  NaN support or a third candidate is added.

**V1 missing-value invariant (implemented):** every non-target column
with `missing_percentage > 0.0` must be explicitly **imputed or
dropped**. `missing_percentage` is read verbatim from the SAME
`SanitizedLLMContext` the planner was shown, never recomputed, so the
evaluator and the LLM can never disagree about the evidence. Scaling or
encoding a column does **not** address its missingness. No thresholds
were invented; the one threshold used (identifier uniqueness) is
imported from `guardrails.py`.

**Target invariant:** the target must not be dropped, imputed,
type-converted, encoded, or scaled. `validate_proposed_plan()` already
rejects target-as-encode/scale-feature, so adequacy's target check is a
**superset covering the gap** (drop/impute/convert of the target, which
previously failed only later, inside the tool at execution time). Both
reject; they cannot contradict.

### Design

| Aspect | Decision |
|---|---|
| Conditions | `missing_values` (material), `target_protection` (material), `identifier_like_column` (advisory) |
| Statuses | `ADDRESSED` / `NOT_ADDRESSED` / `NOT_APPLICABLE` — deliberately **no `UNKNOWN`**; every condition is decidable from evidence PIPER already has |
| Severity | `material` blocks and routes to REPLAN; `advisory` is recorded as evidence only and never blocks |
| Mutation | **None.** Read-only by construction — imports no store, mutates nothing, returns findings only. Pinned by a test asserting neither argument changes and by a source-level test that no store/LLM symbol is imported |
| New retry budget | **None.** No `adequacy_retry_count`, no second loop, no new graph branch |

### REPLAN + duplicate-plan integration

Adequacy runs **after** the existing duplicate check, and its failure
return appends `candidate_hash` to `plan_history`. That ordering is what
makes the required behavior fall out of existing machinery:

```
attempt N   : novel plan -> duplicate check passes -> adequacy FAILS
              -> retryable PLAN_ADEQUACY -> existing REPLAN, hash recorded
attempt N+1 : identical plan -> existing duplicate check fires FIRST
              -> DUPLICATE_PLAN (terminal)
```

Verified end-to-end: exactly **2 LLM calls**, not the full retry budget.
`_route_after_plan` was not touched — it already handles a retryable
PLAN-node failure. Adequacy evidence reaches the model through the
existing `FailureInfo` → `build_replan_prompt()` path with no prompt
changes (`failure_context` already renders structured evidence), so the
planner receives the real condition/column/percentage, never a generic
"try again".

### Historical replay — all 6 recorded plans (no Ollama calls)

Replayed via `backend/replay_adequacy_on_recorded_plans.py` against the
recorded post-contract plans; results in
`benchmark_results/post_contract/adequacy_replay.json`. Recorded
benchmark data was not modified.

| Trial | Steps | Schema-valid | Adequacy | Material columns |
|---|---:|---|---|---|
| qwen3:4b trial1 | 5 | valid | **FAIL** | Cabin, Embarked |
| qwen3:4b trial2 | 5 | valid | **FAIL** | Cabin, Embarked |
| qwen3:4b trial3 | 5 | valid | **FAIL** | Cabin, Embarked |
| qwen3.5:4b trial1 | 3 | valid | **FAIL** | Cabin, Embarked |
| qwen3.5:4b trial2 | 5 | valid | **FAIL** | Cabin, Embarked |
| qwen3.5:4b trial3 | **1** | valid | **FAIL** | **Age**, Cabin, Embarked |

**The 1-step plan is correctly and uniquely detected** — the only plan
flagged for `Age`, with 3 material findings vs. 2 for every other plan.
It also carries an advisory `Name` (100% unique) finding, as do both
other qwen3.5 plans; qwen3:4b's plans drop `Name` and so do not.

**New finding — all six plans miss the same two columns.** Both models,
in every trial, encoded `Embarked` (0.22% missing) without handling its
missing values, and never addressed `Cabin` (77.10% missing) at all.
This was invisible to every previous layer, including the manual
plan-coverage analysis in the model comparison above, which checked
Age/Cabin/identifiers but not Embarked.

**Behavioral consequence (OPEN — see below):** under this rule, no
recorded Titanic plan passes adequacy on the first attempt.

### Test results

- New `tests/test_plan_adequacy.py`: **32 passed** — all 18 required
  scenarios plus controls (scaling/encoding do not address missingness,
  numeric high-uniqueness is not identifier-like, malformed arguments
  cannot satisfy a condition, read-only proof, determinism proof).
- Related suites (planner/graph/validation/learning/hardening/context):
  **206 passed**, zero regressions.
- Full `pytest -q`: see "Current test baseline".
- **No Ollama calls were made in this task.**

### Why existing tests were unaffected

The reference Telco dataset has **zero** plan-time missingness — its
`TotalCharges` blanks are `" "` strings that only become NaN during
CLEAN, *after* planning. The missing-value rule is therefore a genuine
no-op for every existing Telco-driven test. Verified before integrating.

### OPEN FINDING — first-attempt adequacy on Titanic is unproven

Under the implemented rule, all six recorded plans fail on `Cabin` and
`Embarked`. Whether a real model can satisfy adequacy on a REPLAN was
**not tested**, because this task forbade Ollama calls. Two outcomes are
possible and only a real benchmark can distinguish them:

1. The REPLAN evidence is specific enough (exact column + exact
   percentage + required action) that the planner fixes it in one
   attempt — the intended behavior.
2. The planner keeps re-proposing plans that ignore `Cabin`/`Embarked`,
   in which case runs terminate via `DUPLICATE_PLAN` and Titanic becomes
   unrunnable end-to-end until either the planner improves or the
   severity policy is revisited.

If (2) occurs, the smallest defensible adjustment — **not implemented,
and deliberately not chosen pre-emptively** — is to make
`missing_values` material only for columns the plan actually routes into
the feature set (verified reachable via `remainder="drop"`), demoting
non-feature columns to advisory. That is a severity-policy change only;
the evaluator already computes feature-set membership and states it in
each finding's `reason`. **This requires an explicit go-ahead and a real
benchmark round — it must not be adopted on speculation.**

## Adequacy-recovery benchmark — CAN models recover from adequacy failure? (measurement only)

Real-model measurement of the Plan Adequacy REPLAN loop. 6 trials
(qwen3:4b ×3, qwen3.5:4b ×3), 18 real Ollama calls, production defaults
throughout. Harness: `backend/benchmark_adequacy_recovery.py`; results
in the isolated `benchmark_results/adequacy_recovery/`. **No production
code was changed.** The harness mirrors `plan_node_v2`'s pipeline in
exact order, reusing production code at every stage, with real
`build_replan_prompt()` + real `FailureInfo` carrying real adequacy
findings — no simplified retry prompt, no manual explanation, no
patching of model output. Smoke-tested against a fake server for all
four outcome classes before spending real Ollama time.

**Premise correction recorded before running:** the task framing assumed
the evaluator gates severity on effective-feature-set membership. It does
not. Verified in code: feature-set membership only changes a finding's
*explanation text*; **every** non-target column with
`missing_percentage > 0` is material. `Cabin` is therefore reported as a
material failure even though it is never in the feature set. Run as-is
per "do not modify adequacy rules".

### Results

| Metric | qwen3:4b | qwen3.5:4b |
|---|---|---|
| First-attempt structurally valid | **3/3** | **3/3** |
| First-attempt adequate | 0/3 | 0/3 |
| Final adequate | 0/3 | **1/3** |
| Outcomes | 3× BUDGET_EXHAUSTION | 1× SUCCESSFUL_PATCH, 2× BUDGET_EXHAUSTION |
| Mean total planning latency | 1601.8s | 838.3s |
| Mean generation latency | 1118.2s | 207.0s |
| Attempt-0 mean wall | 808.5s | 114.3s |
| REPLAN tax (total) | 2379.9s | 2171.9s |

Aggregate: 18/18 attempts structurally valid, 1/18 adequacy-PASS,
**0 duplicate-plan walls**, 0 parse/transport failures, 0 new
invalidations, 5 attempts exceeding 600s wall clock (the known `urllib`
read-timeout nuance). Only successful `time_to_executable_plan`: 581.4s.
Calls per successful plan: 18.

### The dominant failure mode: whack-a-mole regression

**10/12 REPLAN attempts fixed exactly the columns that were reported.**
**10/12 simultaneously REGRESSED a column that the previous attempt had
already addressed.** qwen3:4b oscillated identically in all three trials:

```
a0: Age imputed            -> reported: Cabin, Embarked
a1: Cabin+Embarked imputed -> Age dropped   -> reported: Age
a2: Age+Cabin addressed    -> Embarked dropped -> reported: Embarked
    (budget exhausted)
```

The models **understand the evidence and act on it precisely** — they
are not confused and not ignoring it. The problem is that REPLAN
evidence reports only *what is currently broken*, never *what must be
preserved*, so each REPLAN is treated as a fresh narrow task rather than
an incremental edit. `previous_plan_summary` is additionally an
empty-vs-empty diff on a PLAN-node failure (`state.plan` is still `[]`),
so the only trace of the prior plan is `evidence.proposed_steps`, which
is evidently not a strong enough "keep these" signal.

**Duplicate-plan protection never fired (0/6)** — models always varied
their plans, so the oscillation is not caught by plan identity; it
simply burns the retry budget. This is a genuine gap between two
mechanisms that each work correctly on their own.

### Severity-policy evidence (reported, NOT acted on)

| Measure | Value |
|---|---:|
| Material missing-value findings for columns IN the feature set | 8 |
| Material missing-value findings for columns NOT in the feature set | 17 |
| Attempts that materially failed | 17 |
| ...blocked **only** by non-feature columns | **9 (53%)** |
| ...blocked by a genuine feature-set column | 8 |

Read-only counterfactual over the recorded data: had `missing_values`
been material only for columns in the effective feature set, **9 of 17
failing attempts would have passed** — including qwen3.5:4b's attempt 0
in 2 of its 3 trials. The effect is asymmetric: qwen3.5:4b produces
smaller plans with fewer encode/scale steps, so more of its columns fall
outside the feature set and are flagged for conditions that
`remainder="drop"` guarantees can never reach the training matrix.

This is direct empirical support for the OPEN FINDING recorded when the
layer was built. **The severity policy was NOT changed by this
benchmark**, per its measurement-only scope.

### Architecture assessment

The chain **LLM proposal → structural validation → adequacy validation →
evidence-based REPLAN → duplicate protection → bounded termination**
held end to end with no unbounded loops, no crashes, and no bypasses.
Every trial terminated within the existing budget; no separate adequacy
budget exists. Two observed weaknesses, both at the *evidence* level
rather than the enforcement level: (1) REPLAN evidence does not state
what to preserve, producing the regression oscillation above; (2) under
the current severity policy the loop can be driven by columns that
cannot affect training.

**Conclusion (n=3/model — characterizing behavior, not statistically
significant): material adequacy failures are recoverable in principle
(1 real SUCCESSFUL_PATCH observed) but not reliably recoverable as
currently evidenced. Severity policy: QUESTION.**

## Effective-Feature Adequacy + State-Preserving REPLAN (COMPLETE)

Acts on both weaknesses the adequacy-recovery benchmark identified.
**Production code changed:** `app/agent/plan_adequacy.py`,
`app/agent/nodes/real_nodes.py`, `app/llm/prompts.py`. **Zero Ollama
calls were made** implementing or verifying this; the real-model
re-benchmark is deliberately a separate task.

### 1. Effective-feature-set adequacy semantics

Verified independently against the real training code, not assumed:
`train_model()` computes
`all_feature_columns = feature_intent.categorical_columns + numeric_columns_to_scale`,
builds its `ColumnTransformer` with `remainder="drop"`, and then selects
`X_train = train_df[all_feature_columns]`. A column becomes an effective
training feature **iff** it is named in an `encode_categorical_features`
or `scale_features` step; anything else is excluded twice over (by the
column selection AND by `remainder="drop"`) and cannot reach either
estimator.

Missing-value severity is therefore gated on that membership:

| Condition | Severity | Rationale |
|---|---|---|
| `missing_% > 0`, IS an effective feature, unaddressed | **material** (blocks) | NaN genuinely enters the training matrix |
| `missing_% > 0`, NOT an effective feature, unaddressed | **advisory** (never blocks) | `remainder="drop"` guarantees it cannot reach training |
| imputed or dropped | ADDRESSED | resolved deterministically before TRAIN |

Still estimator-independent — gated on plan structure, never on which of
the two candidates is running — and still uses **no invented
thresholds** (90%-missing and 0.22%-missing are treated identically; the
only threshold anywhere is identifier uniqueness, imported unchanged
from `guardrails.py`).

### 2. Strict advisory invariant

`material` findings contribute to FAIL; `advisory` findings never do.
Advisory does NOT mean "globally safe" — it means "does not block under
the current policy". A material finding still fails the plan even when
advisory findings are present alongside it
(`test_mixed_feature_and_non_feature_missing_columns` pins exactly this).
Target-protection violations remain material and unchanged.

### 3. Multi-column classification behavior (documented + tested)

`classify_plan_steps()` (new, pure, read-only) splits a plan into
`valid_steps` (preservable) and `implicated_steps` (must be revised).
**A multi-column step is implicated in FULL if ANY of its columns has a
material finding.** This was chosen after checking the real tool
contract rather than assumed: `scale_features`/`encode_categorical_features`
each take exactly one `columns` list and PIPER has no
"scale_features but skip column X" variant, so partial preservation is
not expressible. `test_multi_column_step_is_implicated_in_full` asserts
the tool contract itself (`TOOL_ARGUMENT_SCHEMAS[...]["arguments"] == {"columns"}`)
so the rationale cannot silently rot.

### 4. State-preserving REPLAN

`plan_node_v2`'s adequacy-failure branch now adds `valid_steps` /
`implicated_steps` to the existing `FailureInfo.evidence` (no new
failure type, no new retry loop, no new budget, no second duplicate
detector). `build_replan_prompt()` renders a new
`=== VALID OPERATIONS (preserve these) ===` section **only** when
`evidence.valid_steps` is present and non-empty — every other failure
type (EVALUATION_ERROR, DUPLICATE_PLAN, ...) renders byte-identically to
before. The instruction frames the task as PATCH-not-regenerate while
still requiring the COMPLETE revised plan, and explicitly permits
changing a preserved step when necessary (preserve ≠ freeze).

### 5. Exact JSON serialization

Preserved operations are emitted VERBATIM as
`{"tool_name": ..., "arguments": {...}}` — the same production
representation the model must produce. No prose rendering, no renamed
arguments, no second schema. `verify_replan_prompt_snapshot.py`
(offline, no Ollama) asserts the rendered section round-trips exactly to
`classify_plan_steps()`'s output, contains only allowlisted tool names
and real schema argument names, has no invented keys, and — the
strongest check — that the preserved steps **themselves re-pass
`validate_proposed_plan()`**, so the prompt can never instruct the model
to reproduce something the validator would reject. All 9 checks pass.

### 6. Historical replay — all 6 recorded plans (no Ollama)

`replay_adequacy_on_recorded_plans.py` now carries HARD ASSERTIONS
(old-behavior baseline, new advisory behavior, unchanged material
behavior, and a no-PASS→FAIL regression invariant). Results in
`benchmark_results/post_contract/adequacy_replay_effective_feature.json`;
the original recorded benchmark data was not modified.

| Trial | Effective feature set | OLD | NEW | Material now | Advisory now |
|---|---|---|---|---|---|
| qwen3:4b ×3 | Age, Embarked, Fare, Sex (+SibSp/Parch in t2) | FAIL | FAIL | Embarked | Cabin |
| qwen3.5:4b t1 | Embarked, Sex | FAIL | FAIL | Embarked | Cabin, Name |
| qwen3.5:4b t2 | Age, Embarked, Fare, Sex | FAIL | FAIL | Embarked | Cabin, Name |
| qwen3.5:4b t3 (1-step) | Embarked, Sex | FAIL | FAIL | Embarked | **Age**, Cabin, Name |

- OLD material finding count: **13** → NEW: **6**
- **FALSE-POSITIVE FINDINGS REMOVED: 7**
- Plans changed status: **0/6**; no PASS→FAIL regression (none previously passed)

**Honest caveat — the correction is real but did not unblock Titanic.**
`Cabin` (77.10% missing) is correctly advisory in all 6 plans, and in
the 1-step plan `Age` is now correctly advisory too (it genuinely is not
a feature there). But **every** recorded plan encodes `Embarked` while
never imputing it, and `Embarked` (0.22% missing) IS an effective
feature in all 6 — so all six still FAIL, now for a single genuine
reason instead of a mix of real and spurious ones. The layer became more
precise, not more permissive. Whether a real model can now clear that
one remaining finding is exactly what the next benchmark must answer.

### 7. Tests

`tests/test_plan_adequacy.py`: **49 passed** (32 → 49). Six tests
written against the prior uniform-severity policy were updated to the
new semantics by preserving their real invariants rather than loosening
them — e.g. test 1 now asserts the non-feature column is `advisory` and
non-blocking, and gained `test_1b`/`test_1c` counterparts proving the
identical column IS material once scaled/encoded. New coverage:
effective-feature material/advisory split, mixed feature/non-feature,
all-advisory-passes, `classify_plan_steps()` (including the multi-column
rule and a no-mutation proof), and 6 REPLAN-prompt tests covering
section presence/omission, exact-JSON round-trip, and validator
re-acceptance.

Broader relevant sweep (adequacy, provider, validation, planner
contract, graph integration, duplicate-plan, graph routing):
**211 passed**.

Full `pytest -q`, run ONCE (production code changed): **824 passed, 5
skipped**, 0 failures, 27m26s. 807 → 824 = +17, exactly the net new
adequacy tests (32 → 49). No regressions anywhere in the suite.

### 8. Remaining benchmark work

`backend/benchmark_adequacy_recovery.py` still builds the OLD evidence
shape (no `valid_steps`/`implicated_steps`), so it no longer mirrors
production REPLAN evidence. **It must be updated before the next
real-model run**, otherwise the benchmark would measure a REPLAN prompt
production no longer sends. The recorded adequacy-recovery results above
remain valid as the PRE-change baseline.

**Not claimed:** that model recovery is solved. The whack-a-mole
regression (10/12 REPLANs) was measured against the old evidence shape;
whether state-preserving REPLAN actually reduces it is unverified until
a real-model benchmark runs.

## Adequacy-recovery benchmark v2 — state-preserving REPLAN measured (measurement only)

Real-model re-run after harness synchronization. 6 trials
(qwen3:4b ×3, qwen3.5:4b ×3), **14 real Ollama calls**, production
defaults. Harness: `backend/benchmark_adequacy_recovery.py` (synced);
Phase-1 gate: `backend/smoke_adequacy_recovery.py`. Results in the
isolated `benchmark_results/adequacy_recovery_v2/`; v1 untouched.
**No production code was changed.**

**Harness production-parity was VERIFIED, not assumed.** The smoke test
runs the REAL graph offline with a `FakeLLMProvider`, captures the actual
`FailureInfo.evidence` `plan_node_v2` emits, and asserts the harness's
evidence is identical field-by-field (all 6 keys, incl. `valid_steps`/
`implicated_steps`). The harness imports the production
`classify_plan_steps()` rather than reimplementing it, and renders via
the production `build_replan_prompt()`. 11 prompt properties + 4 outcome
classifications also verified. All passed before any real call.

### Results

| Metric | qwen3:4b | qwen3.5:4b |
|---|---|---|
| First-attempt structurally valid | 2/3 | 3/3 |
| First-attempt adequate | 0/3 | 1/3 |
| **Final executable plan** | **2/3** | **1/3** |
| Outcomes | 2× SUCCESSFUL_PATCH, 1× BUDGET_EXHAUSTION | 1× FIRST_ATTEMPT_PASS, 2× BUDGET_EXHAUSTION |
| time-to-executable | 490.6s, 323.4s | 67.8s |
| REPLAN tax | 947.9s (52% of total) | 512.3s (42%) |
| Mean generation | 143.0s | 53.0s |

Aggregate: **3/6 trials produced an executable plan (v1: 1/6)**.
11/11 parseable attempts structurally valid, 0 duplicate-plan walls,
0 NEW_INVALIDATION, 1 attempt over 600s wall clock.

### State preservation — the headline result

| | v1 (old evidence) | v2 (state-preserving) |
|---|---|---|
| REPLANs regressing a previously-addressed condition | **10/12 (83%)** | **0/6 (0%)** |

Every REPLAN that actually RECEIVED `valid_steps` preserved **all** of
them (3/3). The clearest case — qwen3:4b trial2: attempt 0 proposed 5
steps, `Embarked` was material, `valid_steps` listed the other 4, and
attempt 1 returned exactly those 4 with the implicated encode step
removed → PASS. The whack-a-mole oscillation documented in v1 did not
recur in any trial.

### CONFIRMED PROBLEM (recorded, deliberately NOT fixed)

**A parse/transport failure breaks the preservation chain.** 3 of 14
calls failed at parsing (1 timeout, 2 malformed JSON). Those produce an
`EVALUATION_ERROR`, which carries **no `valid_steps`** — so the next
REPLAN gets no preservation evidence and starts from scratch. All 3
"chain broken" REPLANs came from this path, and both qwen3.5:4b
budget-exhaustion trials followed exactly this shape
(adequacy → parse failure → regressed plan). State preservation
currently exists only on the PLAN_ADEQUACY path.

### Embarked — resolved, but by exclusion

All 3 passing plans resolved `Embarked` by **removing it from the
feature set** (dropping/omitting the encode step), never by imputing it.
That is legitimate under effective-feature semantics — an unused column
cannot carry NaN into training — but it means the model discards a
usable predictor rather than repairing it. Adequacy is satisfied; plan
*quality* is not what improved.

### Verdict (n=3/model — characterizing behavior, not significant)

Executable-plan reliability: qwen3:4b 2/3 vs qwen3.5:4b 1/3 — too close
at this sample size to separate. qwen3.5:4b was far faster when it
worked (67.8s vs 323–491s) but lost both other trials to malformed JSON.
**Model decision: INCONCLUSIVE.** State-preserving REPLAN itself is
validated: 0% regression vs 83%, and 1/6 → 3/6 executable plans.

**Status key for this section:** effective-feature adequacy and
state-preserving REPLAN are **COMPLETE** (implemented, tested, and now
real-model-verified). The parse-failure-breaks-preservation gap above is
an **OPEN FINDING** — confirmed and recorded, deliberately not fixed
this round. The qwen3:4b-vs-qwen3.5:4b model choice remains **OPEN**.

**Files touched this round** (harness/test/doc only — no production
application-code changes): `backend/benchmark_adequacy_recovery.py`
(synchronized to production evidence shape), `backend/smoke_adequacy_recovery.py`
(new — Phase-1 parity gate), `backend/benchmark_results/adequacy_recovery_v2/*`
(new, isolated), `CLAUDE.md`. Verified via file-modification-time audit
(no git repository in this project) that `app/llm/ollama_provider.py`,
`docker-compose.yml`, and every prior `benchmark_results*` namespace were
untouched during this benchmark.

**Next step (not started):** fix the confirmed gap — thread `valid_steps`
through the `EVALUATION_ERROR` (parse/transport-failure) path too, so a
mid-trial parse failure doesn't discard preservation state — then
re-benchmark to see whether that closes the qwen3.5:4b gap specifically
(both of its failures followed exactly this path). Needs its own
explicit go-ahead; not attempted here per this benchmark's
measurement-only scope.

**RESOLVED** — see the next section. The re-benchmark it calls for is
still outstanding.

## Parse-failure state preservation (COMPLETE — implementation/tests only, NOT re-benchmarked)

Fixes the OPEN FINDING recorded by adequacy-recovery benchmark v2: state
preservation existed **only** on the `PLAN_ADEQUACY` path. Attempt N's
adequacy failure emits `valid_steps`/`implicated_steps` so attempt N+1
can PATCH rather than regenerate — but when attempt N+1's own Ollama call
failed at the transport/parse level (timeout, malformed JSON),
`plan_node_v2` returned an `EVALUATION_ERROR` carrying no classification
at all, so attempt N+2 planned from scratch. All 3 observed "chain
broken" REPLANs in v2 came from this path, and both qwen3.5:4b
budget-exhaustion trials followed exactly this shape.

**Fix — additive evidence propagation, two files:**

| File | Change |
|---|---|
| `app/agent/nodes/real_nodes.py` | New `_carried_forward_preserved_steps(state)`; its result is spread into the provider-failure branch's `FailureInfo.evidence`. |
| `app/llm/prompts.py` | Docstring correction only — **no logic change**. `_valid_steps_from_failure_context()` already keyed on the EVIDENCE rather than the failure category, which is precisely what lets an `EVALUATION_ERROR` carry the classification without the renderer knowing. |

The helper carries an EARLIER attempt's `valid_steps`/`implicated_steps`
forward **verbatim**, in the same production
`{"tool_name": ..., "arguments": {...}}` shape, and only after re-running
them through `validate_proposed_plan()` — duck-typed on
`.tool_name`/`.arguments`, so a `SimpleNamespace` suffices and no
provider type is reconstructed. If ANY carried step fails revalidation,
it carries **nothing**: partially-valid preservation evidence would be
worse than none. Returns `{}` when there is nothing to carry, so a
first-attempt provider failure's evidence is byte-identical to before
(keys omitted entirely, never emitted empty).

**Why this is not "trusting malformed output":** a parse/transport
failure produced no plan at all — there is no model output on this path
to trust, merge, or auto-correct. The steps carried forward are ones an
earlier attempt already put through `validate_proposed_plan()`
successfully, and they are re-validated again before being re-stated.
Validator authority, retry accounting, routing, duplicate-plan
semantics, and `plan_history` are all untouched: **this only changes what
the next prompt is TOLD.** A parse failure still writes no plan and
records no plan identity.

**Tests:** new `tests/test_parse_failure_state_preservation.py`, **14
tests** — helper unit coverage (first attempt carries nothing; verbatim
carry; previous failure without classification; empty `valid_steps`;
revalidation dropping an invalid carried step; revalidation rejecting a
target-as-feature carried step; malformed evidence shapes; a
no-mutation proof), `plan_node_v2`-level coverage (carry after an
adequacy failure, unchanged first-attempt evidence, survival across TWO
consecutive parse failures, and no plan/`plan_history` written), and
prompt-level coverage (the preserve section renders for a parse failure,
with a control proving a plain provider failure still renders none).

**Test results:** focused 14/14; broader relevant sweep (adequacy,
provider, validation, planner contract, graph integration,
duplicate-plan, graph routing, batch-5 hardening, context budget)
**251 passed**; full `pytest -q` run ONCE (production code changed):
**838 passed, 5 skipped**, 0 failures, 34m15s — 824 → 838 = +14, exactly
the new tests. No regressions.

**NOT done, deliberately:** no Ollama calls were made. Whether this
closes the qwen3.5:4b gap specifically (both of its v2
budget-exhaustion trials followed this exact path) is **unverified** and
needs its own real-model benchmark round with an explicit go-ahead.
`benchmark_adequacy_recovery.py` would need reviewing first — it
simulates the REPLAN chain itself, so it must reproduce the new
carry-forward behavior before it can measure it.

## Planner Contract Hardening — Canonical Tool Argument Contracts (COMPLETE)

Following live end-to-end testing where Qwen3 occasionally hallucinated multi-argument structures for singular operations (e.g. `drop_column` with `columns: [...]` instead of `column: "..."` or `impute_missing_values` with `column_name`/`imputation_strategy`), the planner prompt contract in `app/llm/prompts.py` was hardened with explicit, canonical schemas:

- `_format_exact_tool_contracts()` renders an authoritative `=== EXACT TOOL ARGUMENT CONTRACTS ===` block into both `build_planning_prompt()` and `build_replan_prompt()`.
- Every tool is explicitly documented with its exact required keys, types, and constraints:
  - `drop_column`: requires singular `"column": "<column_name>"` (explicit negative constraint against `"columns"` array or multiple keys).
  - `convert_column_type`: requires `"column": "<column_name>"` and `"target_type": "<type>"`.
  - `impute_missing_values`: requires `"column": "<column_name>"` and `"strategy": "<mean|median|mode|constant>"`.
  - `encode_categorical_features`: requires list `"columns": ["<col1>", "<col2>"]`.
  - `scale_features`: requires list `"columns": ["<col1>", "<col2>"]`.
- Added explicit negative examples and warnings against inventing parameter names or combining multiple operations into one step.
- Zero changes to deterministic validator authority (`validate_proposed_plan()` remains authoritative), zero plan mutation, and zero automatic repair.
- Pinned by 62 unit and integration tests across `tests/test_planner_contract_titanic.py` and `tests/test_plan_validation.py`.

## PIPER V1 — 10-Run Empirical Reliability Benchmark (qwen3:4b Baseline — COMPLETE)

A controlled, sequential 10-trial measurement benchmark was executed against the real Titanic dataset (`benchmark_data/train.csv`, 891 rows, 12 columns, target = `Survived`) using the frozen PIPER V1 pipeline with `qwen3:4b` (`timeout=600s`, `keep_alive=10m`, `temperature=0.0`).

Results persisted in [`benchmark_results/v1_reliability_10/v1_reliability_10.json`](file:///c:/dev/PIPER/benchmark_results/v1_reliability_10/v1_reliability_10.json) and [`benchmark_results/v1_reliability_10/v1_reliability_10_summary.json`](file:///c:/dev/PIPER/benchmark_results/v1_reliability_10/v1_reliability_10_summary.json).

### Empirical Reliability Summary (N=10)

| Trial | Cold/Warm | First Valid | First Adequate | REPLANS | Final Executable | End-to-End Success | Primary Terminal Cause / Winner |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | Cold | Yes | No | 2 | Yes (5 steps) | No | `ADEQUACY_FAILURE` (Unimputed material feature `Embarked`) |
| **2** | Warm | Yes | No | 1 | Yes (7 steps) | **Yes** | **SUCCESS** (`random_forest`, $F_1=0.7353$) |
| **3** | Warm | Yes | No | 2 | Yes (3 steps) | No | `ADEQUACY_FAILURE` (Attempt 2 regressed adequacy) |
| **4** | Warm | Yes | No | 2 | Yes (4 steps) | No | `TIMEOUT` (Attempt 2 exceeded 600s inference limit) |
| **5** | Warm | No (Timeout) | No | 2 | No | No | `TIMEOUT` (Consecutive Ollama timeouts $\ge 600$s) |
| **6** | Warm | Yes | No | 2 | Yes (2 steps) | No | `EXECUTION_FAILURE` (Attempt 2 plan dropped required features) |
| **7** | Warm | Yes | No | 1 | Yes (6 steps) | **Yes** | **SUCCESS** (`random_forest`, $F_1=0.4091$) |
| **8** | Warm | Yes | No | 2 | Yes (2 steps) | No | `ADEQUACY_FAILURE` (Unimputed material feature `Embarked`) |
| **9** | Warm | Yes | No | 2 | Yes (5 steps) | No | `DUPLICATE_PLAN` (Attempt 2 proposed duplicate of Attempt 0) |
| **10** | Warm | Yes | No | 2 | Yes (4 steps) | No | `ADEQUACY_FAILURE` (Attempt 2 regressed adequacy) |

### Key Findings & Metrics
- **End-to-End Success Rate**: **2 / 10 (20.0%)**
- **First-Attempt Structural Validity**: **9 / 10 (90.0%)** — confirms planner contract hardening eliminated schema syntax errors.
- **First-Attempt Adequacy**: **0 / 10 (0.0%)** — `qwen3:4b` consistently failed to impute `Embarked` (0.22% missingness) before encoding.
- **Final Executable Plan Rate**: **9 / 10 (90.0%)**
- **REPLAN Engagement Rate**: **10 / 10 (100.0%)**
- **REPLAN Recovery Rate**: **2 / 10 (20.0%)**
- **Duplicate-Plan Termination Rate**: **1 / 10 (10.0%)** — correctly intercepted repetitive plan proposals.
- **Dominant Failure Modes**: `ADEQUACY_FAILURE` (40%) and local CPU inference `TIMEOUT` (20%).
- **Reliability Decision**: **`NOT RELIABLE` (< 5/10)** $\rightarrow$ **`V1 RELIABILITY NOT ACCEPTED`** for unattended live demo.

## Model Screening Benchmark (qwen3:8b — COMPLETE)

Screening benchmark conducted on the larger parameter-class candidate `qwen3:8b` in `benchmark_results/qwen3_8b_screening/` to evaluate whether model capacity improvements resolve the multi-turn REPLAN state oscillation observed in the 4B model class without modifying frozen V1 architecture.

## Known finding: PLAN-node failures never get a REPLAN chance — resolved in Batch 5

`FailureInfo.retryable` is set to `True` for a PLAN-node failure (LLM
provider error, or a proposed plan that fails `validate_proposed_plan()`)
— see `plan_node_v2` in `app/agent/nodes/real_nodes.py`. But
`_route_after_plan` in `app/agent/graph.py` used to unconditionally route
**any** `status == "failed"` at the PLAN node straight to `REPORT`, never
checking `retry_count < max_retries`. In practice this meant a PLAN-node
failure got **zero retries**, regardless of `max_retries`, despite being
marked `retryable=True` — unlike a VALIDATE-node/guardrail failure, which
genuinely did get a REPLAN chance via `_route_after_validate`. Observed
live during Batch 3's manual verification (qwen3:4b proposed a plan with
empty/malformed tool arguments and the run failed immediately, attempt 0,
no REPLAN attempted).

**Fixed in Batch 5** — confirmed genuine (not a deliberate design choice)
after tracing every existing test through the change and confirming none
of them relied on the old zero-retry behavior. `_route_after_plan` now
checks `state.failure.retryable and state.retry_count < state.max_retries`
before deciding REPORT vs. routing back to `PLAN_ENTRY`, mirroring
`_route_after_validate`'s existing bounded-retry check exactly (see the
updated graph-flow diagram above). Two follow-on gaps this exposed and
fixed in the same change:
- `plan_node_v2`'s `is_replan` flag required `len(state.plan) > 0`, which
  silently dropped `failure_context`/`previous_plan_summary` from the LLM
  on a PLAN-triggered retry (state.plan is still `[]` at that point, since
  plan construction never got that far on the failed attempt). Now keys
  off `state.failure is not None` alone — equivalent for every
  VALIDATE-triggered case, additionally correct for the new PLAN-triggered
  case.
- `report_node` only flipped `failure.human_intervention_required` to
  `True` on retry-exhaustion for the VALIDATE path (which arrives at
  `REPORT` with `status != "failed"`); a PLAN-node failure arrives with
  `status == "failed"` already and hit the early `return {}`, so it never
  got the same flip. Now checked and flipped for both paths.

Verified live in Docker against the real Telco CSV + real qwen3:4b (not
just unit tests): a real run genuinely replanned twice (attempt 0 -> 1 ->
2) after qwen3:4b proposed plans with `validate_proposed_plan()`
violations, then correctly reported a terminal failure with
`retryable=true` and `human_intervention_required=true` once the retry
budget was exhausted — visible end to end through the SSE live feed and
the frontend's `FailurePanel`.

## Known finding: REPLAN could repeat an already-rejected invalid plan forever — resolved post-multi-format-ingestion

**Problem, observed live** (run `run_dfcbae97`, real qwen3:4b, Telco
acceptance dataset): the LLM proposed `drop_column` with an empty
`column` argument. `validate_proposed_plan()` correctly rejected it —
but on REPLAN, the LLM proposed the **exact same invalid step again**,
on both attempt 1 and attempt 2, burning the entire retry budget (and
several real minutes of Ollama latency) on content PIPER already had
complete evidence was invalid.

**Root cause:** `canonicalize_plan()`/`plan_hash()`/`plan_history` — the
existing mechanism that stops an LLM from re-proposing an identical
plan — only ever ran on a plan that had **already passed**
`validate_proposed_plan()` (see the duplicate-plan block in
`plan_node_v2`). A REJECTED (invalid) proposal was constructed into
real `PlanStep` objects and hashed only *after* the validity check; on
the failure path, execution returned immediately, so a rejected
proposal was never given any executable identity at all. Nothing
deterministic could ever tell "this is the exact same invalid plan I
already rejected" — the retry loop had no memory of REJECTED content,
only of successfully-validated content.

**Fix** (`plan_node_v2`, `app/agent/nodes/real_nodes.py`): when
`validate_proposed_plan()` rejects a proposal, it is now ALSO
canonicalized (via the same throwaway-`PlanStep` + `canonicalize_plan()`
machinery already used for valid plans — `PlanStep` never itself
validates `tool_name`/`arguments`, so this works structurally regardless
of validity) and checked against `state.plan_history`:
- **First occurrence** of a given invalid proposal: unchanged behavior
  — a normal, retryable `EVALUATION_ERROR`, REPLAN gets its chance
  exactly as before. Its hash is now recorded in `plan_history`.
- **Repeat occurrence** (executably identical to an already-rejected
  proposal): a terminal, non-retryable `DUPLICATE_PLAN` failure —
  exactly the same category, and exactly the same routing consequence
  (`_route_after_plan` already routes `retryable=False` straight to
  REPORT), as the existing post-validation duplicate check. No graph
  routing code was touched; this reuses the existing mechanism one
  stage earlier.

`plan_history`'s field description (`app/agent/state.py`) was corrected
to say what it always claimed to mean — "every plan **attempted**",
not silently "every plan that passed validation" — closing the gap
between the field's stated and actual purpose, not changing its
public shape (`GET /runs/{id}` still returns the same `list[str]`).

**Also fixed in the same change:** the FailureInfo `evidence` for a
plan-validation failure now includes the actual rejected
`tool_name`/`arguments` per step (`evidence.rejected_steps`), not just
the abstract field name and reason (`evidence.violations`, unchanged) —
so REPLAN sees the concrete invalid value it submitted, not only a
description of the rule it broke.

**What was deliberately NOT changed:** `validate_proposed_plan()` itself
— still the sole, unweakened authority on validity; no argument is ever
auto-corrected or silently patched. `_route_after_plan`/
`_route_after_validate`/`_increment_retry_if_replanning` — zero graph
routing changes; retry/termination semantics are identical to before
except that a genuinely-repeated invalid proposal now terminates
*earlier* (bounded by "propose the same invalid content twice", not by
`max_retries`), never later and never unconditionally.

**Regression tests:** `tests/test_replan_duplicate_invalid_plan.py` (8
tests) — reproduces the exact observed scenario end-to-end through the
real graph (a provider that always proposes the identical invalid
`drop_column` call terminates after exactly 2 LLM calls, not the full
retry budget), a control proving the first occurrence is still
retryable, a control proving a genuinely *different* invalid proposal
is never mistaken for a duplicate, an evidence-content assertion, a
control proving a real recovering plan still succeeds, an edge case at
`max_retries=0`, and two `plan_node_v2()`-level unit tests. One
pre-existing test (`test_batch5_hardening.py::TestPlanNodeRetryRouting
::test_retries_are_bounded_by_max_retries_not_unconditional`) was
inadvertently exercising this exact bug — its `_AlwaysInvalidPlanProvider`
proposed the byte-identical invalid plan on every call, so post-fix it
correctly terminated after 2 calls instead of the full 3-call budget.
Fixed by varying the provider's invalid content per call (not by
loosening the assertion — the test's real invariant, "the retry budget
bounds a genuinely-never-repeating failure," is unchanged and still
verified against a provider that now actually varies).

**Verification performed:** targeted (`test_replan_duplicate_invalid_plan.py`,
8/8) then a broader plan-node-adjacent sweep (132 tests across
`test_llm_graph_integration.py`, `test_batch5_hardening.py`,
`test_graph.py`, `test_context_budget.py`,
`test_duplicate_plan_prevention.py`, `test_execution_budget.py`,
`test_multi_model_integration.py`, `test_e2e_matrix.py`,
`test_failure_taxonomy_integration.py`), then one full `pytest -q`
(**741 passed, 5 skipped**, no failures — see "Current test baseline"
above). Per explicit instruction, only one full regression run (not
two) was performed for this change.

**Separate, unresolved:** the underlying LATENCY issue (a REPLAN cycle
against real qwen3:4b costing several real minutes per attempt) is a
property of the model's own planning latency on CPU inference, not of
this bug — this fix reduces WASTED latency (fewer pointless repeat
attempts) but does not change per-attempt Ollama call latency itself.
See "Ollama configuration" above for the existing measured latency
distribution (143–418s/call against the real Telco dataset); no new
work was done on that here, per instruction to report it separately.

## Batch 5 — Production hardening (complete)

Full-system inspection across security, reliability, error handling,
state isolation, concurrency, configuration, observability/logging, API
quality, validation, reproducibility, and performance. No architecture
changes, no new features — only genuine, verified issues were fixed.
Both documented open findings (PLAN-node retry routing; Ollama timeout on
a realistic dataset) were investigated with real measurements before
touching anything, per the standing "measure, don't guess" rule.

**Genuine issues found and fixed** (each with a regression test in
`backend/tests/test_batch5_hardening.py` unless noted):

1. **PLAN-node retry routing** — see the dedicated section above.
2. **Cross-run dataset corruption (state isolation).** `clean_node`
   mutates the dataset stored under `state.dataset_id` in place (a
   documented, intentional invariant within a single run's REPLAN loop).
   `POST /runs` (`app/api/routers/runs.py`) passed the *uploaded*
   `dataset_id` straight into `AgentState` unchanged — so a second run
   against the same uploaded dataset (a normal, UI-supported workflow:
   datasets persist in `DatasetStore` and stay re-selectable in the
   frontend's dataset list) would silently execute against whatever the
   FIRST run's cleaning already mutated it into, not the original upload;
   two concurrent runs against the same uploaded `dataset_id` would also
   race on the same mutable rows. Fixed: every run now clones the dataset
   into a private, run-scoped `dataset_id` (`{run_id}_data`) before
   execution; `GET /runs/{id}` still reports the originally uploaded
   `dataset_id` back to the caller via a new `display_dataset_id` param on
   `RunStore.create()`. `InMemoryRunStore.create()` was also made
   idempotent (a second `create()` call for an already-known `run_id` is a
   no-op) — needed because `stream_with_tracing()`/`run_with_tracing()`
   already call `create()` themselves at their own start (load-bearing for
   every caller that invokes them directly without a prior `create()`,
   which is most of the test suite), and without idempotency that second
   call silently clobbered the API layer's `display_dataset_id` override.
   Verified live in Docker: a real run's `profiler` trace event showed it
   operating on `run_<id>_data`, while `GET /runs/{id}` and
   `GET /datasets/{original_id}` both confirmed the original upload was
   never touched.
3. **No exception safety net around graph execution.** Nothing caught an
   unexpected exception (e.g. an uncaught `sklearn` `fit()` failure —
   `train_model()` in `app/agent/tools/training.py` has no try/except
   around `pipeline.fit()`) inside `stream_with_tracing()`
   (`app/agent/tracing.py`) — the function `POST /runs` actually uses via
   `BackgroundTasks`. Such an exception would leave the run stuck at
   `"running"` in `RunStore` forever: `GET /runs/{id}/result` would 409
   indefinitely, and `GET /runs/{id}/events` (SSE) would loop forever
   waiting for a terminal status that would never arrive. Fixed: both
   `stream_with_tracing()` and `run_with_tracing()` now catch `Exception`
   around graph execution, log it (`logging.getLogger(__name__)` — this
   was the only logging call anywhere in the backend before this fix), and
   produce a genuine terminal `"failed"` `RunStore` result instead.
   Purely additive — never changes the outcome of a run that doesn't hit
   this path (confirmed by the full regression suite passing unchanged).
4. **Invalid `FailureCategory` literal.** `plan_node_v2`'s
   sanitized-context-build failure branch constructed
   `FailureInfo(category="DATA_QUALITY_ERROR", ...)` — not a member of
   `FailureCategory`'s `Literal` in `app/schemas/failure.py`. Since
   `FailureInfo` is a strict Pydantic model, this would have raised
   `pydantic.ValidationError` the instant this defensive branch was ever
   actually exercised (e.g. the dataset vanishing from `DatasetStore`
   between PROFILE and PLAN), turning a should-be-graceful structured
   failure into an unhandled crash. Untested before this batch. Fixed to
   `"DATA_ERROR"` (the existing category for "the dataset/environment is
   broken for this run"), which was already what the branch's
   `retryable=False` argument assumed.
5. **Unbounded CSV upload size.** `POST /datasets` read the entire
   uploaded file into memory (`await file.read()`) before any validation
   ran — a memory-exhaustion risk with no cap, in a single-process,
   no-auth, in-memory-store local/demo deployment. Fixed: a 100MB cap
   (`MAX_UPLOAD_BYTES` in `app/api/routers/datasets.py`), checked via
   `UploadFile.size` where available and again after reading as a
   backstop, returning `413`.
6. **Ollama timeout on a realistic dataset** — see the dedicated section
   above.

**Verification performed:** targeted tests for each fix; full `pytest -q`
regression suite run twice post-fix (553 passed, 5 skipped both times,
identical); the real-Ollama integration suite (5/5 passed,
`PIPER_RUN_OLLAMA_TESTS=1`); the frontend suite (22/22 passed, unchanged —
no frontend code was touched this batch); a full Dockerized system
verification (`docker compose up --build`, both services healthy) driving
**two** independent real runs against the real Telco CSV through the real
frontend, real FastAPI + SSE, and real Ollama:
- Run 1 (`max_retries=2`): genuinely REPLANned twice (qwen3:4b proposed
  plans with `validate_proposed_plan()` violations both times), then
  correctly reported a terminal, bounded, structured failure
  (`retryable=true`, `human_intervention_required=true`) once the retry
  budget was exhausted — visible end to end in the SSE feed and the
  frontend's `FailurePanel`.
- Run 2 (`max_retries=3`): REPLANned three times on the same kind of
  validation violation, then genuinely **succeeded on attempt 3** —
  trained both candidates, passed every guardrail (leakage, imbalance,
  constant features, high cardinality, suspicious metric, baseline gate),
  selected logistic regression (F1=0.4969) over random forest
  (F1=0.4848), reached `status: "completed"`. Confirms the retry bound
  correctly follows the *configured* `max_retries` value, not a
  hardcoded one.

Both runs, plus direct `GET /datasets/{original_id}` calls, confirmed the
originally uploaded dataset was never mutated by either run's cleaning
steps — only the private, run-scoped clone was.

**Explicitly not changed:** no architecture changes, no new
features/endpoints, no changes to `PIPER Learn` (not started, out of
scope for this batch), no relaxation of any existing guardrail or
validation rule.

**New finding, observed but not fixed (out of this batch's closed
scope):** `report_node` sets `status: "completed"` once
`state.validation.valid` is `True`, but never clears a stale
`AgentState.failure` object left over from an earlier, superseded REPLAN
attempt — confirmed live via Run 2 above: its `GET /runs/{id}/result`
response has `status: "completed"` and `validation.valid: true` (both
correct) but `failure` still shows attempt 2's validation violation, not
`null`. Cosmetic, not a correctness bug — `status`/`validation.valid` are
authoritative and correct — but a client naively checking
`failure !== null` on a completed run would be misled. Pre-existing (not
introduced by this batch's routing fix — the same gap applies to a
VALIDATE-triggered REPLAN that eventually succeeds), just never observed
before because it requires a live multi-attempt run that ultimately
succeeds. Needs an explicit go-ahead before touching `report_node`'s
success path.

## Pre-6A Polish (complete)

Four locked, scoped items, backend-only (no frontend UI was built for any
of the new fields/endpoints below — none of the four items' locked spec
called for frontend integration, unlike Batch 6A's explicit "API-first,
frontend once ready" posture, which this deliberately mirrors). No
architecture changes, no new execution paths, no changes to routing logic
beyond item 1's single-field fix.

1. **Stale failure cleanup** (`app/agent/graph.py`'s `report_node`) — the
   fix for the Batch 5 open finding documented above. The
   `state.validation.valid` success branch now returns
   `{"status": "completed", "failure": None}` instead of just
   `{"status": "completed"}`. Scoped to exactly that one branch — the
   failed/retries-exhausted branch (which flips
   `human_intervention_required`) and the already-`"failed"` passthrough
   branch are both untouched, confirmed by a regression test for each.
   Verified live via the real graph: a run driven through a genuine
   PLAN-triggered REPLAN (attempt 0 fails with a structured `EVALUATION_ERROR`,
   attempt 1 recovers and passes every guardrail) now reports
   `failure: null` at `status: "completed"`, not attempt 0's stale
   `FailureInfo`.
2. **Model-selection transparency** — `ModelComparison` (`app/schemas/evaluation.py`)
   gained a `justification: str` field; `compare_models()`
   (`app/agent/tools/evaluation.py`) computes it via a new
   `_build_selection_justification()` helper, built only from the
   already-computed `ModelComparisonEntry.f1`/`algorithm` values already in
   `entries` — e.g. `"logistic_regression selected: F1=0.4969 vs. 0.4848
   for random_forest."` No LLM involvement; a test proves the same inputs
   produce byte-identical justification text across repeated calls (a
   guarantee an LLM-generated string could never make). Flows through to
   the API automatically via `RunResultResponse.comparison` and the new
   `RunSummary.selection_justification` below, since both reference
   `ModelComparison` directly rather than redefining its shape.
3. **Structured `RunSummary`** — new `app/schemas/run_summary.py`
   (`RunSummary`) + `app/agent/run_summary.py` (`build_run_summary()`, a
   pure function: `run_id` + a terminal state in, a `RunSummary` out,
   nothing mutated). Aggregates `retry_count`/`replanned`,
   `state.comparison.models` (candidate scores), the winning
   `model_id`/`algorithm` + its justification (item 2, read by reference,
   not recomputed), `state.cleaning_log + state.feature_log` (operations
   executed), and `state.validation`'s `valid`/`checks`/`violations`/
   `warnings` (guardrail status) — every field either a direct reference
   or a trivial derivation, never a new source of truth. Exposed via a new
   `GET /runs/{id}/summary` endpoint, gated on terminal status exactly
   like `/result` (both read `record.final_state`). This required one
   small, necessary plumbing addition: `app/agent/tracing.py`'s
   `_RunResultState` shim (what `record.final_state` actually is on the
   API path) didn't carry `cleaning_log`/`feature_log` at all before this
   batch — harmless while nothing read them, but `RunSummary.operations_executed`
   needs them, so both were added to the shim's `__init__`.
4. **Structured execution timeline** — new
   `app/schemas/execution_timeline.py` (`ExecutionTimeline`,
   `TimelinePhase`) + `app/agent/timeline.py` (`build_execution_timeline()`,
   pure: `run_id` + a `list[TraceEvent]` in, an `ExecutionTimeline` out).
   Collapses consecutive same-phase-same-attempt `TraceEvent`s into one
   `TimelinePhase` (marked `"failure"` if any collapsed event failed);
   `replan_count` is the highest `attempt` observed. A `_PHASE_LABELS`
   table normalizes both event-name vocabularies already present in a
   real run's event stream onto the same human-readable labels — the live
   per-node events `stream_with_tracing()` emits (`"train"`, `"validate"`,
   ...) and the post-hoc tool-call-level events both tracing functions
   derive from `tool_trace` (`"trainer"`, `"validator"`, ...) — so the
   timeline reads the same regardless of which tracing function produced
   the run. Exposed via a new `GET /runs/{id}/timeline` endpoint —
   deliberately **not** gated on terminal status (unlike `/result` and
   `/summary`): it reads directly off `run_store.get_events(run_id)`,
   which accumulates progressively during a live run, so it's meaningful
   mid-run too, consistent with the existing live SSE `/events` feed it's
   derived from the same way.

**Verification performed:** targeted tests for each item (44 tests: the
new `tests/test_pre6a_polish.py`, 3 added to `tests/test_evaluation.py`,
6 added to `tests/test_api_runs.py`), all passing before the full-suite
run; full `pytest -q` regression suite run twice (**574 passed, 5
skipped**, identical both times — see "Current test baseline" above); the
frontend suite (22/22 passed, unchanged — no frontend code was touched
this batch, confirming no regression). The real-Ollama integration suite
was not re-run (nothing in this batch touches `OllamaProvider` or prompt
construction).

**Explicitly not changed:** no architecture changes, no new execution
paths, no changes to `AgentState`'s own fields (both new features read
existing state, never add to it), no relaxation of any guardrail or
routing rule beyond item 1's single documented fix, no `PIPER Learn`
work (Batch 6A/6B, not started, out of scope), no frontend UI changes.

**New findings this batch:** none beyond the one already-documented
Batch 5 finding that item 1 fixes.

## Batch 6A — PIPER Learn: Learn-Explain (complete)

A read-only explanation layer over existing PIPER execution state — new
`app/learning/` package (`__init__.py`, `explain.py`, `formulas.py`,
`comprehension.py`), new `app/schemas/learning.py`, and three new
read-only endpoints. No architecture changes, no new execution paths,
zero coupling into `AgentState`/the graph.

**Design decision (deliberate): no `learning_mode` flag anywhere.**
Rather than adding a flag to `CreateRunRequest`/`AgentState` that a run
could be started "with" or "without," Learn-Explain has no execution-time
presence at all — it only ever reads an already-terminal run's state,
strictly after the fact, via its own endpoint. This trivially satisfies
the locked constraint ("off by default, never affects a normal run,"
"structurally incapable of influencing a run") by construction — there
is no code path in `graph.py`/`real_nodes.py` that even knows
Learn-Explain exists. "Learning Mode" is simply whichever client chooses
to call the `/learn/*` endpoints or not.

**What was built:**
- `app/schemas/learning.py` — `RunExplanation` (the per-run bundle) plus
  five sub-schemas, each tied to the ONE real schema it explains rather
  than a generic "evidence" abstraction: `OperationExplanation`
  (`OperationRecord`), `ModelSelectionExplanation` (`ModelComparison`,
  reusing Pre-6A Polish's `justification`), `EvaluationExplanation` +
  `MetricExplanation` (`EvaluationResult`, `BaselineComparisonResult`),
  `GuardrailCheckExplanation` (`ValidationCheck`), `FailureExplanation`
  (`FailureInfo`). Plus two run-independent, purely static schemas:
  `FormulaEntry`, `ComprehensionCheck`.
- `app/learning/explain.py` — pure functions
  (`explain_operation()`/`explain_model_selection()`/
  `explain_evaluation()`/`explain_guardrail_check()`/`explain_failure()`/
  `build_run_explanation()`). Every "meaning" string is a static,
  reviewed template with the real per-run value plugged in (exactly the
  pattern already used for `_build_selection_justification()`, Pre-6A
  Polish item 2) — never LLM-generated, never fabricated.
  `_GUARDRAIL_MEANINGS` and `_FAILURE_CATEGORY_MEANINGS` are static
  lookup tables covering, respectively, all 6 real guardrail check names
  (`app/agent/tools/guardrails.py`) and all 11 real `FailureCategory`
  values (`app/schemas/failure.py`) — both parametrized in the test
  suite so a taxonomy addition that's missed here fails loudly (falls
  back to a generic placeholder) instead of silently.
- `app/learning/formulas.py` — `FORMULA_LIBRARY`: 8 static, curated
  entries (Accuracy, Precision, Recall, F1 Score, ROC-AUC,
  Standardization, One-Hot Encoding, Median/Mean Imputation), generic
  and reviewed, never generated per-run or per-dataset.
- `app/learning/comprehension.py` — `COMPREHENSION_CHECKS`: 7 static
  question/explanation pairs (no grading, no scoring, no per-user state
  — presented content only).
- Three new endpoints (`app/api/routers/runs.py` for the per-run one,
  new `app/api/routers/learning.py` for the two static ones — see
  "FastAPI + SSE architecture" above for the exact contract of each):
  `GET /runs/{id}/learn/explanation`, `GET /learn/formulas`,
  `GET /learn/comprehension-checks`.

**What was deliberately left out (real architectural limitation, not an
oversight):** no `PlanDiff`-based "what changed between REPLAN attempts"
explanation. `AgentState.plan_history` only retains plan HASHES across
attempts (see `plan_canonical.py`), never the full prior `PlanStep` list
— so there is nothing left to diff against once a later attempt starts.
Adding that would mean retaining full historical plans in `AgentState`,
which the locked constraint explicitly forbids modifying. A REPLAN'd
run's explanation still surfaces real, grounded evidence for what
actually happened (via `FailureExplanation`, whichever attempt's
evidence survived into the terminal state) — it just can't show a
step-by-step plan diff across attempts. The locked spec listed
`PlanDiff` as one of several allowed evidence sources ("may cite"), not
a mandatory field, so this is in-scope-as-written, not a shortfall.

**Verification performed:** targeted tests (41 tests: the new
`tests/test_learning.py` — formula-library/comprehension-check content,
every `explain_*()` function checked against the exact real value it was
built from, full parametrized coverage of all 6 guardrail check names
and all 11 failure categories, and three dedicated zero-effect-on-
execution tests — plus 6 added to `tests/test_api_runs.py`), all passing
before the full-suite run; full `pytest -q` regression suite run twice
(**615 passed, 5 skipped** at the time of that batch, identical both
times; the current project-wide figure is higher — see "Current test
baseline" above); the frontend suite (22/22 passed, unchanged — no
frontend code was touched this batch, matching Pre-6A Polish's own
backend-only posture, since neither item's locked spec called for
frontend integration). The real-Ollama integration suite was not re-run
(nothing in this batch touches `OllamaProvider` or prompt construction).

The "zero effect on execution" proof specifically: (1) `build_run_explanation()`
never mutates the state instance it's given (verified via attribute
snapshot before/after); (2) calling it twice on the same state produces
byte-identical output (deterministic, not time-based); (3) two
independent runs against the identical input dataset — one with
`build_run_explanation()` invoked afterward, one without — produce the
same deterministic outcome (status, validation, comparison metrics,
operations executed), compared via a projection that excludes only the
randomly-generated identifiers (`run_id`, cloned `dataset_id`,
`model_id`, `split_id`, `operation_id`) that legitimately differ between
any two independent runs regardless of Learn-Explain.

**Explicitly not changed:** no architecture changes, no new execution
paths, no changes to `AgentState`'s own fields or shape, no relaxation
of any guardrail or routing rule, no `Learn-Explore` work (Batch 6B, not
started — depends on this batch, out of scope until its own go-ahead),
no frontend UI changes (backend/API-first, matching the locked spec's
own "if frontend integration isn't ready when this batch starts"
posture).

**New findings this batch:** none.

## Batch 6B — PIPER Learn: Learn-Explore (complete)

Controlled, single-variable experimentation against an already-terminal
run. New `app/agent/tools/exploration.py`, `app/schemas/exploration.py`,
`app/storage/exploration_store.py`, and three new endpoints. No new
training logic, no changes to the graph, no changes to `AgentState`.

**What was built:**
- `explore_alternative()` — a thin orchestration wrapper over the SAME
  `train_model()`/`evaluate_model()`/`compare_models()` the real graph
  uses. Enforces, before anything is fit: exactly one variable changed
  (either `new_algorithm` OR `hyperparameter_name`+`hyperparameter_value`,
  never both, never neither); the new algorithm must differ from the
  base model's and be one of the two already-supported V1 algorithms;
  the hyperparameter must already be inside `training.py`'s locked
  allowlist for that algorithm (bounds enforcement is delegated to
  `train_model()`'s existing `_validate_params()`, never duplicated);
  and `base_model_id` must genuinely belong to the run being explored.
- **Same-split reuse:** the split_id is read from the base model's own
  `ModelStore` metadata (mirroring `evaluate_model()`'s existing
  "read split_id from the model, never accept it as an argument"
  pattern), so no new splitting and no new randomness can affect
  comparability.
- **Run-scoping boundary:** `ModelStore` is a shared, global, in-memory
  store across every run, so a bare `model_id` string alone proves
  nothing about which run trained it. `explore_alternative()` therefore
  takes the original run's own `state.model_results` model_ids as an
  explicit scoping argument and rejects anything outside it
  (`model_not_from_this_run`).
- **Isolation:** results live in `ExplorationStore` keyed by their own
  `experiment_id`, never merged into `RunStore`. New models are
  additive `ModelStore` entries — the base model is never overwritten.
  Nothing in this path ever calls `run_store.update()`.
- **Learn-Explain integration:** reuses Batch 6A's `explain_evaluation()`
  and `explain_model_selection()` directly rather than reimplementing
  explanation text.

**Verification performed:** 27 new tests (18 in the new
`tests/test_exploration.py` — exactly-one-variable enforcement,
same-split reuse, base-model run-scoping, Learn-Explain integration,
and explicit before/after isolation proofs for both the original
`RunRecord` and the base model's `ModelStore` metadata — plus 9 added to
`tests/test_api_runs.py`); full `pytest -q` twice (**642 passed, 5
skipped** at the time of that batch, identical both times, 34m44s /
16m49s); frontend 22/22
unchanged.

**Explicitly not changed:** no graph/routing changes, no `AgentState`
changes, no new training logic, no frontend UI.

**New findings this batch:** none.

## Batch 7 — Final integration & context-budgeting hardening (complete)

Deterministic LLM context-budgeting, full end-to-end verification of the
whole system, the real README, and two genuine reliability findings
fixed (both surfaced *by* the end-to-end verification, which is exactly
what this batch exists for).

### Context-budgeting (`app/agent/tools/context_budget.py`)

`plan_node_v2` now calls `apply_context_budget()` between
`build_sanitized_llm_context()` and `llm_provider.generate_plan()`.

- Size is estimated as the character count of the context's JSON
  serialization — a deliberate, dependency-free proxy (no tokenizer
  dependency added; `OllamaProvider` is stdlib-`urllib`-only and this
  follows the same discipline).
- `DEFAULT_MAX_CONTEXT_CHARS = 8000`, grounded in a **real measurement**
  per this project's standing "measure, don't guess" rule: the real
  Telco dataset's context is **4,569 chars** (so budgeting is a genuine
  no-op there — proven by a test asserting the budgeted context is
  `==` the raw one), while a synthetic 81-column dataset is **18,539
  chars** and genuinely triggers reduction.
- Reduction is **staged and deterministic** (sample-value cap 5 → 2 → 1
  → 0, stopping at the first stage that fits). **Only `sample_values`
  ever shrinks.** Column names, dtypes, `target_column`,
  missing/unique percentages, and numeric min/max/mean are never
  touched — asserted field-by-field at an impossible budget
  (`max_chars=1`) that forces the floor.

### Finding 1 — unbudgeted retry loop (fixed)

**Problem:** a real Dockerized run made 4 real Ollama calls while
`retry_count` stayed at 1 the whole time.
**Cause:** `_route_after_validate` decided REPLAN vs. REPORT purely from
`state.validation`, ignoring the case where VALIDATE never actually ran
this attempt (an earlier node failed, so `validate_node_v2` hit its "no
evaluation result" early return and left `validation` None). It fell
through to the unconditional `retry_count < max_retries -> PLAN` branch;
`_increment_retry_if_replanning` then correctly declined to increment
(neither of its conditions held), but the graph looped back to PLAN
anyway — an unbudgeted retry bounded only by `MAX_EXECUTION_STEPS`.
**Fix:** mirror `_route_after_plan`'s already-correct pattern — when
`status == "failed"` with no fresh validation, only REPLAN if
`state.failure` is genuinely retryable AND budget remains.
**Regression test:** `TestValidateRoutingDoesNotLoopWithoutBudgetedRetries`
(`tests/test_graph.py`), asserting the invariant as a relationship
(`provider.calls == retry_count + 1`, `<= max_retries + 1`) rather than
a magic number, plus a control proving a genuinely retryable failure
still gets its REPLAN chance.

### Finding 2 — cascade overwrote the failure root cause (fixed)

**Problem:** the same Dockerized run reported
`"baseline_node reached with no state.comparison"` as its terminal
failure, when the real root cause was that the plan produced an empty
feature set and `train_model()` failed.
**Cause:** TRAIN → EVALUATE → COMPARE → BASELINE → VALIDATE are
unconditional edges (correct — VALIDATE is the graph's one guardrail
decision point). But a TRAIN failure cascaded through the rest, and each
downstream node **overwrote** `state.failure` with its own "reached with
no X" symptom. `train_node_v2` also returned only a bare `error` string,
never a structured `FailureInfo`. The last symptom in the chain
(baseline's, marked `retryable=False`) additionally denied the genuinely
fixable root cause its remaining budgeted retry.
**Fix:** `train_node_v2`/`evaluate_node_v2` now emit a structured
`FailureInfo` (`TRAINING_ERROR`/`EVALUATION_ERROR`, `retryable=True`),
and `evaluate_node_v2`/`compare_node`/`baseline_node`/`validate_node_v2`
each pass an already-structured upstream failure through **unchanged**
via a new `_upstream_already_failed()` guard — the exact pattern
`plan_node_v2` already used for `EXECUTION_BUDGET_EXCEEDED`. **No graph
edges or routing functions were changed**; state still arrives at
VALIDATE with `status="failed"` and `validation=None`, so
`_route_after_validate` makes the same decision it always did. The guard
keys off `status == "failed"` (not merely `failure is not None`) so a
stale failure from a superseded attempt can never trigger it.
**Regression test:** `TestUpstreamFailureRootCauseIsPreserved`
(`tests/test_context_budget.py`) — an end-to-end run whose terminal
failure must be `TRAINING_ERROR`/`node="train"`, a unit-level
pass-through proof for all four downstream nodes, and a control proving
the guard does not trigger on a healthy running state carrying a stale
failure.

This fix also corrected the `TestValidateRoutingDoesNotLoopWithoutBudgetedRetries`
expectations: that scenario now terminates after the **full** budgeted
retry allowance (3 LLM calls, `retry_count == max_retries == 2`) citing
the real root cause, instead of terminating one attempt early citing a
misleading symptom. The test's actual invariant — every LLM call is
budgeted — is unchanged and now asserted more explicitly.

### Verification performed

- Full `pytest -q`: **659 passed, 5 skipped**, run twice — identical
  both times (1h57m43s and 56m44s; the spread is machine contention,
  see the test-baseline note above). 642 before this batch; +17 = 15 in the new
  `tests/test_context_budget.py` (12 context-budgeting + 3 cascade
  root-cause regression tests) and 2 routing tests in
  `tests/test_graph.py`.
- Real-Ollama integration suite (`PIPER_RUN_OLLAMA_TESTS=1`) against
  real local Ollama + `qwen3:4b`, confirming the budgeted context still
  produces valid plans against a real model. **Note on a flaky
  observation:** one run of this suite reported 4 passed / 1 timeout at
  the 600s limit — that run was executing concurrently with a full
  pytest suite AND a Docker build on the same machine. Re-run in
  isolation, the same test passed in **354.57s**. The failure was
  machine contention, not a context-budgeting regression.
- Frontend suite: 22/22, unchanged (no frontend code touched).
- Dockerized end-to-end (`docker compose up --build`): both services
  built and healthy, real Telco CSV uploaded through the real stack, and
  a real run driven by real host Ollama + `qwen3:4b`. Observed exactly
  the designed behavior: qwen3:4b proposed a plan that failed
  `validate_proposed_plan()` on all three attempts (0 -> 1 -> 2), each
  rejected BEFORE any execution, REPLANned within the bounded budget,
  then terminated with a structured `EVALUATION_ERROR`
  (`retryable: true`, `human_intervention_required: true`,
  `attempt: 2`). Endpoints verified live against that run:
  `/result`, `/summary` (`retry_count: 2`, `replanned: true`),
  `/timeline` (`replan_count: 2`, real phase list), and
  `/learn/explanation` (grounded template text). `/health` and the nginx
  frontend both 200; `/learn/formulas` and `/learn/comprehension-checks`
  both returned real static content.

  **Not demonstrated live in Docker:** the `/explore` endpoints (Batch
  6B), because they require a run that actually trained models and
  qwen3:4b never produced a plan that passed validation in these
  sessions. They are covered by 27 automated tests (18 unit + 9 API)
  against real trained models, so this is a demo-coverage gap, not a
  verification gap.

  **Environment note (not a PIPER defect):** Docker Desktop's
  `host.docker.internal` route to the host Ollama dropped intermittently
  on this machine mid-session — a container-side
  `[Errno 101] Network is unreachable` while the host's own
  `localhost:11434` stayed healthy (HTTP 200). `docker compose restart
  backend` restored it each time. Worth knowing when demoing: if a run
  fails with "Could not reach Ollama", restart the backend container
  rather than assuming a code problem. PIPER itself handled the outage
  correctly — a structured, retryable, bounded terminal failure rather
  than a hang or crash.
- Real README written (architecture, setup, usage, capabilities,
  honest limitations, demo script).

**Explicitly not changed:** no new features beyond the locked
context-budgeting item, no graph edge/routing-function changes beyond
Finding 1's single documented fix, no relaxation of any guardrail, no
frontend UI changes.

## Roadmap — all locked batches complete

**Every locked batch through Batch 7 is now Complete.** The locked
specs are retained below for reference (they document what each batch
was *required* to do); the dedicated "(complete)" sections earlier in
this file document what was *actually built and verified*.

```
M1-M7  ->  Batch 5  ->  Pre-6A Polish  ->  Batch 6A  ->  Batch 6B  ->  Batch 7
                                                                     (all Complete)
```

| Batch | What was actually built (see its own section above) |
|---|---|
| Pre-6A Polish | `report_node` stale-failure fix, `ModelComparison.justification`, `RunSummary` + `/summary`, `ExecutionTimeline` + `/timeline` |
| Batch 6A | `app/learning/`, `RunExplanation` + `/learn/explanation`, static formula library + comprehension checks |
| Batch 6B | `explore_alternative()`, `ExplorationResult`, `ExplorationStore`, three `/explore` endpoints |
| Batch 7 | `context_budget.py` wired into `plan_node_v2`, two reliability findings fixed, real README, full end-to-end verification |

**Status key:** *Complete* = implemented, tested, verified live.
*Open finding* = a real, confirmed gap observed and documented but
intentionally left unfixed pending an explicit go-ahead. There are
currently **no open findings** — both findings raised during Batch 7's
end-to-end verification were fixed within that batch's own locked
"fix only genuine integration/reliability issues found" scope, each
with a regression test (see the Batch 7 section above).

### Possible future work (NOT locked, NOT started — do not begin without an explicit go-ahead)

These are ideas, not commitments. Nothing below has a locked spec:

- Broaden scope beyond binary/multiclass tabular classification
  (regression, more model families) — would require revisiting the
  locked V1 scope and the fixed candidate set.
- Persistent storage (every store is currently in-memory by design).
- Frontend UI for the Pre-6A Polish / Batch 6A / Batch 6B endpoints —
  all four batches were deliberately backend/API-first, and the
  frontend still shows only the M6 feature set.
- Tokenizer-based context budgeting (the current estimator is
  character-based by deliberate choice, to avoid a new dependency).

### Batch 6A — PIPER Learn: Learn-Explain (complete — locked spec retained below for reference)

- Optional Learning Mode — off by default, never affects a normal run.
- A **read-only** explanation layer over existing PIPER execution state
  only. No new execution paths, no new agent behavior.
- Explanations are **deterministic/template-based only** — **no
  LLM-generated explanations**, matching PIPER's core "LLM never
  controls outcomes" principle extended to the learning layer.
- Every explanation must cite real evidence from `FailureInfo`,
  `PlanDiff`, `OperationRecord`, `TraceEvent`, real metrics, or other
  existing pipeline state — never fabricated or generic filler text.
- Includes "Why did PIPER choose this?" explanations (e.g. why a
  particular model was selected, why a column was dropped, why a
  guardrail failed).
- Includes beginner-friendly explanations of preprocessing, feature
  engineering, model selection, evaluation, and failures.
- Includes static "Check your understanding" content (no grading, no
  scoring — see constraints below).
- Includes a curated, reviewed, static formula library for the relevant
  ML formulas (Accuracy/Precision/Recall/F1/ROC-AUC/standardization/
  etc.) — formula examples must be generic, reviewed, and static, never
  generated per-run or per-dataset.
- **Explicitly excluded:** quizzes, grading, or experimentation of any
  kind (that is Batch 6B's job, and only in the constrained form
  described below).
- **Must never modify** `AgentState`, plans, validation, retry counts,
  model selection, or execution — a read path only, structurally
  incapable of influencing a run.
- API-first: expose via read-only API endpoints if frontend integration
  isn't ready when this batch starts.
- Tests (once implemented) must verify: every explanation is grounded in
  real evidence (not fabricated), the formula library is used correctly,
  and Learning Mode has zero effect on execution (a run with Learning
  Mode invoked produces byte-identical `AgentState`/results to one
  without).

### Batch 6B — PIPER Learn: Learn-Explore (complete — locked spec retained below for reference)

- Controlled student exploration of **exactly one** alternative ML
  decision per exploration — not general experimentation.
- The one variable that may change: either an already-supported model
  (from the fixed Logistic Regression / Random Forest candidate set) OR
  one already-supported hyperparameter within its existing locked
  allowlist/bounds (see `app/agent/tools/training.py`) — never both,
  never anything outside what's already supported.
- Reuses the existing training/model-comparison machinery
  (`train_model()`, `compare_models()`, etc.) — no new training logic.
- Reuses the **same dataset split** as the original run — no new
  splitting, no new randomness affecting comparability.
- **Never modifies the original PIPER run** in any way.
- Exploration results are stored in an **isolated `run_id` +
  `experiment_id` namespace** — structurally separate from the original
  run's state, never merged into it.
- Must never modify the original run's plan, `AgentState`, validation,
  retry count, model selection, or results.
- Integrates with Learn-Explain and the frontend (once both exist).
- **Explicitly excluded:** a general experimentation engine, arbitrary
  pipeline modifications, or multi-plan optimization of any kind.
- Tests (once implemented) must verify: exactly one variable changed per
  exploration, the same split was reused, the exploration's results are
  fully isolated from the original run's namespace, and the original
  run's plan/`AgentState`/validation/retry count/model
  selection/results are byte-identical before and after an exploration
  runs.

### Batch 7 — Final Integration & Context-Budgeting Hardening (complete — locked spec retained below for reference)

- Verify the complete PIPER system end-to-end: agent, ML pipeline,
  FastAPI, SSE, frontend, Ollama, Docker, Learn-Explain, and
  Learn-Explore, all together.
- Final UX and demo polish.
- Fix only genuine integration/reliability issues found — no unrelated
  features.
- **Deterministic LLM context-budgeting for larger datasets:**
  - Estimate the planning context's size before calling
    `llm_provider.generate_plan()`.
  - Apply deterministic context reduction when the estimate exceeds a
    defined budget — never arbitrary/lossy truncation of important
    information.
  - Preserve, at minimum: column names/types, target information,
    missingness statistics, essential summary statistics, and a limited
    set of representative sample values per column.
  - Re-run the real Ollama integration suite after context-budgeting
    changes land, to confirm the reduced context still produces valid
    plans against a real model.
- Verification once implementation is stable: full regression suite run
  **twice**, the real-Ollama integration suite, and the Dockerized
  application verified end-to-end.
- Only after the implementation is verified stable: finalize
  documentation and write the real README (architecture, setup, usage,
  capabilities, limitations, demo instructions).

Pre-6A Polish, Batch 6A, Batch 6B, and Batch 7 are ALL now Complete —
see their dedicated "(complete)" sections earlier in this file for what
was actually built and verified in each. The specs above are retained
only as a record of what each batch was required to deliver.

There is no locked next batch. Anything further is unscoped future work
(see "Possible future work" above) and must not be started without an
explicit go-ahead and a locked spec of its own.

## Development and testing rules

1. **Inspect actual code before modifying anything.** Never guess at
   contracts or file contents; never assume a prior session's claims are
   still true without verifying on the actual checkout.
2. **Minimal, scoped changes.** Don't touch unrelated architecture.
3. **Never weaken or delete a test to make it pass.** Find the test's real
   invariant and preserve it.
4. **Run targeted tests first, then the full regression suite twice**
   before declaring any phase/checkpoint complete — unless the change is
   infrastructure-only (e.g. Docker files) and provably cannot affect test
   execution, in which case running once and stating that reasoning is
   sufficient.
5. **Report exact files changed and exact test results.**
6. **If ambiguous, stop and inspect the literal source** rather than
   guessing.
7. Report genuine implementation bugs found along the way using
   `Problem: / Cause: / Fix: / Regression test:` — don't report expected
   behavior as a bug, and don't fix a genuine finding without being asked
   unless it's directly blocking the current task.
8. **Do not silently expand scope.** Stop at the requested checkpoint and
   wait for explicit approval before continuing to the next batch/phase.
9. This project runs directly against the user's real working directory
   (Claude Code, not a sandbox) — "verified" and "synced" are the same
   thing here. Be explicit and precise about what was actually run.

## Architectural invariants (carried through every milestone)

- No raw DataFrames or fitted sklearn model objects ever enter
  `AgentState` — only references (`dataset_id`, `model_id`, `split_id`)
  and structured results.
- `max_retries` defaults to 2; `retry_count` starts at 0.
- Invalid/unresolvable `task_type` (non-binary/multiclass target) is
  always terminal — never replanned.
- `MAX_EXECUTION_STEPS` (`app/agent/graph.py`, M4) is a hard,
  PIPER-owned execution-step ceiling, independent of `max_retries` and of
  any externally supplied LangGraph `recursion_limit` — guarantees
  `graph.invoke()`/`.stream()` always returns a clean terminal state
  rather than raising `GraphRecursionError`.
- `clean_node` mutates the working dataset in place under the same
  `dataset_id` — never reuse one `DatasetStore`/`dataset_id` across
  multiple separate `graph.invoke()` calls expecting the same input twice.
  This invariant is unchanged at the `graph.invoke()`/`AgentState` level —
  direct callers (tests, scripts) must still supply a fresh `dataset_id`
  per invocation. `POST /runs` (the API layer) protects against this for
  real users by cloning into a run-scoped `dataset_id` before building
  `AgentState` (Batch 5 fix — see below).
- `train_node_v2`/`evaluate_node_v2` **reset** `model_results`/
  `evaluation_results` each REPLAN cycle — they do not accumulate across
  attempts.
- pandas 3.0.2 uses a native string dtype, not legacy `object`, for text
  columns — use `is_numeric_dtype()` from
  `app/agent/tools/_profiling_helpers.py`, never a raw `dtype == object`
  check.
- `qwen3:4b`'s Ollama response content is not guaranteed to land in the
  `response` field — `OllamaProvider._extract_content()` checks
  `response` -> `thinking` -> `message.content`, first non-empty wins.
- **The learning layer is read-only and has no execution-time presence.**
  `app/learning/` and `app/agent/tools/exploration.py` only ever consume
  already-terminal state. Nothing in `graph.py`/`real_nodes.py` imports
  them or knows they exist, and there is deliberately no `learning_mode`
  flag on `AgentState`/`CreateRunRequest` — "Learning Mode" is simply
  whether a client calls the `/learn/*` endpoints.
- **Exploration never mutates the run it explores.** `explore_alternative()`
  writes only new, additive `ModelStore` entries plus its own
  `ExplorationStore` record keyed by `experiment_id`; it never calls
  `run_store.update()`. It also reuses the base model's own `split_id`
  (read from `ModelStore` metadata), so no new randomness can affect
  comparability.
- **A downstream node must never overwrite an upstream node's structured
  failure** (Batch 7). `evaluate_node_v2`/`compare_node`/`baseline_node`/
  `validate_node_v2` each begin with `_upstream_already_failed()` and
  pass an existing failure through unchanged — otherwise the terminal
  result names the last cascade symptom instead of the real root cause.
- **Context budgeting may only ever shrink `sample_values`.** Column
  names, dtypes, `target_column`, missing/unique percentages, and
  numeric min/max/mean are locked minimums that survive even an
  impossible budget.
