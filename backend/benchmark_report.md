# Planning-model benchmark — qwen3:4b vs. candidates (Titanic workload)

**Status: investigation only. Production model, timeout, prompts, graph
routing, retry logic, deterministic validators, and tool allowlist were
NOT changed. Model decision is OPEN — see "Recommendation" below.**

Date: 2026-08-13. Isolated benchmark script:
[`backend/benchmark_planning_models.py`](benchmark_planning_models.py)
(never imported by `app/`, never wired into the graph). Raw results:
[`backend/benchmark_results.json`](benchmark_results.json).

## Method

- Real Titanic dataset (`benchmark_data/train.csv`, provided — not
  modified, not uploaded through the frontend): **891 rows × 12
  columns**, target `Survived`. Loaded with the identical call the
  production CSV-ingestion path uses (`pd.read_csv(io.BytesIO(raw))`).
- The benchmark reuses the real production functions rather than
  reimplementing them: `build_sanitized_llm_context()`,
  `apply_context_budget()`, `build_planning_prompt()` /
  `build_replan_prompt()`, `OllamaProvider`'s own `PLAN_JSON_SCHEMA` /
  `_extract_content()` / `_strip_markdown_fences()`, `ProposedPlan`
  schema validation, `validate_proposed_plan()`, and
  `canonicalize_plan()`/`diff_plans()` for REPLAN evidence — so what's
  measured here is provably the same logic `plan_node_v2` runs in
  production. The one addition: the script makes the Ollama HTTP call
  itself (identical request shape to `OllamaProvider.generate_plan()`)
  so it can capture Ollama's own `prompt_eval_count`/`eval_count`/
  `*_duration` fields, which the production provider discards after
  extracting the plan.
- `PIPER_OLLAMA_TIMEOUT_SECONDS`/`DEFAULT_TIMEOUT_SECONDS` (600s,
  unchanged) used for every call.
- 3 independent first-attempt trials. For any trial whose first
  attempt failed deterministic validation, exactly one REPLAN
  follow-up call was made (capped at 2 total), built the same way
  `plan_node_v2` builds a real REPLAN prompt (`failure_context` +
  `previous_plan_summary`), to directly measure repeated-invalid-plan
  behavior.

## Candidate availability

`ollama list` output:

```
qwen3:4b                   2.5 GB
nomic-embed-text:latest    274 MB
```

Only `qwen3:4b` (the current baseline) is installed. `qwen3.5:4b`,
`llama3.2:3b`, and `gemma3:4b` are **unavailable** — not installed, and
per instructions not auto-downloaded. **No comparative ranking across
models is possible from this run.**

## Results — qwen3:4b (5 real Ollama calls: 3 first-attempt + 2 REPLAN follow-ups)

| Trial | Outcome | Wall time | Ollama total_duration | prompt_eval | eval (generation) | prompt tokens | output tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| first_attempt_0 | **timeout** (no response) | 1572.5s (>600s budget) | — | — | — | — | — |
| first_attempt_1 | structured, **invalid** | 125.4s | 123.33s | 0.19s | 122.29s | 1842 | 430 |
| replan_after_1 | structured, **invalid**, **repeated identical** | 165.8s | 163.78s | 47.88s | 114.94s | 2395 | 349 |
| first_attempt_2 | structured, **invalid** | 143.4s | 141.32s | 9.66s | 130.21s | 1842 | 429 |
| replan_after_2 | structured, **invalid**, **repeated identical** | 151.1s | 149.07s | 45.62s | 102.72s | 2387 | 315 |

| Metric | Value |
|---|---|
| Structured-plan-produced rate | 4/5 = 80% (1 hard timeout at the current 600s budget) |
| **Deterministic-validation-passed (valid plan) rate** | **0/5 = 0%** |
| Validation violations per (completed) attempt | 1, identical, every time |
| Invalid tool/argument pattern | `drop_column` called with `arguments.columns: [list]` instead of the required `arguments.column: str` — every single completed attempt |
| Repeated-identical-invalid-plan rate on REPLAN (given full failure evidence) | **2/2 = 100%** |
| Avg total latency (4 completed calls) | 138.9s (Ollama `total_duration`); 146.4s wall-clock |
| Avg output (generated) tokens | 381 |
| Avg prompt tokens | 2117 (1842 first-attempt / 2391 REPLAN) |
| Effective generation rate | ~3.2 tokens/sec |

## Latency bottleneck analysis

Per-stage instrumentation (dataset load → sanitize → budget → prompt
construction → Ollama call → parse → validate):

| Stage | Time | Share of total |
|---|---:|---:|
| Sanitized-context construction | 27.5ms | ~0.02% |
| Context budgeting | 0.4ms | ~0.0003% (no-op — 2,780 chars, well under the 8,000-char budget; budgeting never activates for a 12-column dataset) |
| Prompt string construction | <2ms | ~0.001% |
| **Ollama prompt processing (`prompt_eval_duration`)** | 0.19–47.9s | 0.1%–30.6% (small on first attempt; grows substantially on REPLAN because `failure_context`/`previous_plan_summary` roughly doubles the prompt) |
| Model load (`load_duration`) | 0.69–1.34s | <1% (model stayed warm across calls) |
| **Model generation (`eval_duration`)** | 102.7–130.2s | **69%–99% of total latency — the dominant cost in every completed call** |
| Response parsing + deterministic validation | <1ms | negligible |

**Conclusion: the observed 2–3 minute (and occasionally >600s) latency
is overwhelmingly caused by autoregressive token generation on CPU, not
by context/prompt construction or by structured-output formatting.**
Context/prompt-building stages are collectively under 30ms — three to
four orders of magnitude smaller than the Ollama call itself. Ollama's
own `format`-constrained JSON generation worked correctly in every
completed call (valid JSON matching `ProposedPlan`'s schema every
time) — the JSON-schema constraint is not itself adding meaningful
overhead or causing failures.

`eval_count` (300–430 tokens generated) is far larger than a 4-step
JSON plan needs (well under 100 tokens serialized) — consistent with
`qwen3:4b` being a thinking-mode model whose hidden reasoning trace,
not the final JSON answer, accounts for most generated tokens and most
of the latency. This matches CLAUDE.md's existing documented finding
for the Telco dataset (Batch 5's 143–418s distribution) — this
benchmark reproduces the same mechanism on a completely different,
smaller (12 vs. 21 column) dataset, which rules out "large/wide
dataset" as the primary latency driver: a 12-column dataset with a
2,780-char sanitized context still took 123–164s per completed call,
and one attempt exceeded the full 600s budget.

One important nuance: `prompt_eval_duration` for `first_attempt_1`
(0.19s for 1,842 tokens) is implausibly fast compared to
`first_attempt_2`'s 9.66s for the identical prompt/token count — almost
certainly a KV-cache hit from `first_attempt_0`'s prompt (same
prefix), which Ollama can reuse across consecutive calls to the same
loaded model. This is a real Ollama behavior worth knowing about but
does not change the core conclusion (generation dominates regardless).

## Planning correctness/reliability — the actual finding

Every one of the 4 completed calls proposed a **structurally
consistent** mistake, not random noise: it wanted to drop two columns
(`Name`, `Ticket`) in a single step and used the multi-column list
shape (`arguments.columns: [...]`) that `encode_categorical_features`/
`scale_features` actually take, instead of `drop_column`'s real,
singular `arguments.column: str` contract. `validate_proposed_plan()`
correctly rejected this every time (exactly as designed — no
partially-invalid step ever reached execution). On REPLAN, given the
exact violation AND the exact rejected `tool_name`/`arguments` as
evidence, the model reproduced the **byte-identical** invalid plan
both times (`repeated_identical_invalid_plan: true`) — this is the
same "REPLAN could repeat an already-rejected invalid plan forever"
failure mode CLAUDE.md already documents as fixed at the deterministic
layer (the `DUPLICATE_PLAN` mechanism, confirmed still structurally
correct here: had this run gone through the real graph, it would have
terminated as `DUPLICATE_PLAN` after the second identical rejection
rather than burning the full retry budget).

This is a different, more specific defect than the "empty `column`
argument" failure originally reported against the Telco dataset — same
symptom class (an invalid `drop_column` call, repeated verbatim on
REPLAN), different concrete argument-shape confusion. Across this
session's evidence (Telco, reported; Titanic, benchmarked here),
`qwen3:4b` has never been observed to produce a `validate_proposed_plan()`
-passing plan against a real, non-trivial dataset — 0/5 here, and the
duplicate-rejected-plan pattern is consistent across both datasets.

## Recommendation

**Fastest model:** qwen3:4b (only candidate benchmarked; N/A as a
comparison).

**Most reliable model:** N/A — 0% valid-plan rate, no alternative
available to compare against.

**Best overall candidate for PIPER: cannot be determined from this
run.** Only one of the four candidate models is currently installed;
per instructions, none were auto-downloaded. A model decision requires
at least one working alternative to benchmark against the same
evidence bar.

**Is changing models justified?** The evidence justifies **investigating
alternatives** — `qwen3:4b` has now shown a 0% deterministic-validation-pass
rate on two different real datasets (Telco, Titanic) in this session, plus
a 100% repeated-identical-invalid-plan rate on REPLAN, plus one outright
timeout at the current 600s budget. That is a real reliability signal, not
just a latency complaint. But per the decision rule in scope ("do not
choose a model solely because it's faster" and "priority: valid plans >
low invalid-tool rate > low repeated-plan rate > latency > resources"),
no substitute can be recommended without comparable head-to-head evidence
against at least one other model.

**Is the bottleneck model-specific or architecture/context-specific?**
Architecture/context-specific factors (context size, prompt
construction, JSON-schema formatting) are ruled out as the primary
latency driver — all are negligible (<30ms) relative to the 100+ second
Ollama calls, and this holds on a dataset less than half the size of
Telco. The bottleneck is specific to this model's generation behavior
on this CPU (thinking-mode reasoning trace + ~3.2 tokens/sec decode
rate) — a different, smaller/faster/non-thinking-mode model could
plausibly change this, but that is exactly the untested comparison
above.

## ⚠️ MODEL DECISION: OPEN — no production change made

Do not treat this report as a recommendation to switch. `qwen3:4b`
remains the production default (`DEFAULT_LLM_MODEL`, unchanged), the
600s timeout is unchanged, and no prompts/graph/validators were
touched. Next step (not started): install at least one of
`qwen3.5:4b`/`llama3.2:3b`/`gemma3:4b` and re-run
`benchmark_planning_models.py` unmodified against the same Titanic
fixture for a real head-to-head comparison.

## Update: qwen3.5:4b vs. qwen3:4b (controlled comparison)

**Status: investigation only, same as above. Production model, timeout,
prompts, graph routing, retry logic, and deterministic validation are
still unchanged.**

`qwen3.5:4b` was not installed. Checked disk space first (346GB free —
ample). Pulled **only** `qwen3.5:4b` (3.4GB; nothing else was
downloaded, removed, or modified — `qwen3:4b`, other Docker
images/volumes, and other projects untouched). Benchmarked with the
exact same methodology, dataset, context, prompts, tool definitions,
and call budget (3 first-attempt trials + up to 2 REPLAN follow-ups,
same 891×12 Titanic fixture, same `build_sanitized_llm_context()` /
`apply_context_budget()` / `build_planning_prompt()` /
`validate_proposed_plan()` functions) via
[`backend/benchmark_run_qwen35.py`](benchmark_run_qwen35.py), which
calls `benchmark_planning_models.py`'s functions unmodified. Merged
results in `benchmark_results.json`.

### Results — qwen3.5:4b (5 real Ollama calls)

| Trial | Outcome | Wall time | total_duration | prompt_eval | eval (gen) | prompt tok | output tok | violations |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| first_attempt_0 | structured, invalid | 173.4s | 171.3s | 129.1s | 30.2s | 2051 | 119 | 3 |
| replan_after_0 | structured, invalid, **not** repeated | 147.5s | 145.5s | 70.4s | 73.7s | 2660 | 273 | 3 |
| first_attempt_1 | structured, invalid | 74.3s | 72.0s | 33.8s | 36.8s | 2051 | 144 | 2 |
| replan_after_1 | structured, invalid, **not** repeated | 98.0s | 95.9s | 73.0s | 21.1s | 2636 | 90 | 1 |
| first_attempt_2 | structured, invalid | 133.8s | 131.6s | 30.1s | 100.2s | 2051 | 402 | 6 |

### Side-by-side (5 calls each)

| Metric | qwen3:4b (baseline) | qwen3.5:4b | Winner |
|---|---:|---:|---|
| Deterministic-validation-passed (valid plan) rate | 0/5 (0%) | 0/5 (0%) | **tie — both fail** |
| Structured-plan-produced rate (no timeout) | 4/5 (80%) | **5/5 (100%)** | qwen3.5:4b |
| Violations per completed attempt | **1, 1, 1, 1 (mean 1.00)** | 3, 3, 2, 1, 6 (mean 3.00) | **qwen3:4b — 3x fewer, single consistent root cause** |
| Argument-shape pattern | one consistent mistake every time (`drop_column` list-vs-str) | a *different* hallucinated field-naming scheme almost every call (`method`, `columns_to_encode`, `columns_to_drop`, `column_names`...) | **qwen3:4b — narrower, more predictable** |
| Repeated-identical-invalid-plan rate on REPLAN | **2/2 (100%)** | **0/2 (0%)** | qwen3.5:4b |
| Timeouts | 1/5 | 0/5 | qwen3.5:4b |
| Avg wall time (completed calls) | 146.4s | **125.4s** | qwen3.5:4b (~14% faster) |
| Avg generated tokens | 380.8 | 205.6 | qwen3.5:4b (less reasoning overhead) |
| Avg generation rate | 3.23 tok/s | **3.97 tok/s** | qwen3.5:4b (~23% faster decode) |
| Model size on disk | 2.5 GB | 3.4 GB | qwen3:4b (smaller) |

### Applying the decision criteria, in priority order

1. **Correct PIPER plan** — tie: neither model produced one (0/5 vs 0/5).
2. **Valid tool arguments** — **qwen3:4b wins clearly.** Its failure is
   a single, narrow, structurally consistent mistake (one wrong
   argument shape on one tool, every time — 1 violation/attempt, 4/4
   completed calls). qwen3.5:4b's failures are broader and erratic (1–6
   violations/attempt, mean 3x higher, and it invents a *different*
   made-up field-naming convention almost every call —
   `columns_to_encode`, `column_names`, `method` — none of which appear
   anywhere in the `ALLOWED OPERATIONS` section of the prompt it was
   given). This reads as materially weaker adherence to the tool
   schema it was explicitly shown, not just "different bugs."
3. No repeated malformed plans — qwen3.5:4b wins (0/2 vs 2/2).
4. Reliable structured output — qwen3.5:4b wins (100% vs 80%, no timeout).
5. Latency — qwen3.5:4b wins (~14% faster wall time, ~23% faster decode).
6. Resources — qwen3:4b wins (smaller: 2.5GB vs 3.4GB).

Criterion 2 is explicitly prioritized above criteria 3–6. Both models
tie on criterion 1 (the actual top priority — neither ever produces a
plan PIPER's own validator accepts), which pushes the deciding
criterion to #2 — where qwen3.5:4b is clearly *worse*, not better.
qwen3.5:4b's real, measured wins on REPLAN-diversity, reliability, and
latency (criteria 3–5) do not outweigh a worse showing on the
higher-priority argument-correctness criterion (#2), per the explicit
"do not choose it just because it's faster" decision rule.

### Verdict

**Qwen3.5:4B did not establish a clear advantage; next candidate should
be Llama 3.2 3B.**

No production change made. Per instructions, no further models were
benchmarked automatically this session.

## Update: llama3.2:3b vs. both Qwen candidates (controlled comparison)

**Status: investigation only, same as above. Production model, timeout,
prompts, graph routing, retry logic, and deterministic validation are
still unchanged.**

`llama3.2:3b` was not installed. Checked disk space first (343GB free
— ample). Pulled **only** `llama3.2:3b` (2.0GB; `qwen3:4b`,
`qwen3.5:4b`, and everything else — Docker images/volumes, other
projects — untouched). Benchmarked with the exact same methodology,
dataset, context, prompts, tool definitions, and call budget (3
first-attempt trials + up to 2 REPLAN follow-ups) via
[`backend/benchmark_run_llama32.py`](benchmark_run_llama32.py), which
calls `benchmark_planning_models.py`'s functions unmodified. Merged
results in `benchmark_results.json`.

### Results — llama3.2:3b (5 real Ollama calls)

| Trial | Outcome | Wall time | total_duration | eval (gen) | output tok | violations |
|---|---|---:|---:|---:|---:|---:|
| first_attempt_0 | **HTTP 500 from Ollama** (no plan; not a timeout) | 317.2s | — | — | — | — |
| first_attempt_1 | structured, invalid | 105.6s | 98.6s | 29.7s | 165 | 2 |
| replan_after_1 | structured, invalid, not repeated, **hallucinated 3 non-existent tool names** | 101.7s | 99.6s | 69.4s | 326 | 4 |
| first_attempt_2 | structured, invalid | 39.0s | 36.9s | 33.0s | 173 | 2 |
| replan_after_2 | structured, invalid, not repeated | 82.9s | 80.8s | 52.6s | 217 | 2 |

`first_attempt_0` is a distinct, new failure mode not seen in either
Qwen candidate: Ollama returned `HTTP 500 Internal Server Error` after
317s of wall time (not the current 600s timeout, and not a parsing/
schema failure — the server itself errored, and burned over 5 minutes
before doing so). `first_attempt_1`'s `load_duration` of 7.9s (vs.
<1s for every other completed call) is consistent with the model
having been reloaded after that failure. Not investigated further here
— out of scope for a model-selection benchmark — but worth knowing if
this recurs.

`replan_after_first_attempt_1` is qualitatively the most severe
violation seen across all three models benchmarked so far: 3 of its 4
proposed steps used tool names that don't exist anywhere in
`ALLOWED_TOOL_NAMES` at all — `identify_categorical_columns`,
`select_columns`, `onehot_encode` — correctly rejected by
`validate_proposed_plan()`'s tool-name allowlist check, exactly as
designed. Neither Qwen model ever proposed a nonexistent tool name;
both always used real tool names with wrong argument shapes.

### Three-way comparison (5 calls each)

| Metric | qwen3:4b | qwen3.5:4b | llama3.2:3b |
|---|---:|---:|---:|
| Deterministic-validation-passed (valid plan) rate | 0/5 | 0/5 | 0/5 |
| Structured-plan-produced rate | 4/5 (80%) | **5/5 (100%)** | 4/5 (80%) |
| Failure mode on the non-structured call | timeout (600s+) | — | **HTTP 500 (after 317s)** |
| Violations per completed attempt (mean) | **1.00** | 3.00 | 2.50 |
| Worst violation kind observed | wrong arg shape on a real tool | wrong/hallucinated field names on real tools | **hallucinated tool names outside the allowlist entirely** |
| Repeated-identical-invalid-plan rate on REPLAN | 2/2 (100%) | 0/2 (0%) | 0/2 (0%) |
| Avg wall time (completed calls) | 146.4s | 125.4s | **82.3s** |
| Avg generation rate | 3.23 tok/s | 3.97 tok/s | **4.90 tok/s** |
| Avg output tokens | 380.8 | 205.6 | 220.2 |
| Model size on disk | 2.5 GB | 3.4 GB | **2.0 GB** |

### Applying the decision criteria, in priority order

1. **Correct executable PIPER plan** — three-way tie: 0/5 for every model.
2. **Correct tool arguments/schema** — **qwen3:4b still wins.** Its
   mean violation count (1.00) is the lowest, and its single failure
   mode is the narrowest and most consistent of the three. llama3.2:3b's
   mean (2.50) sits between the two Qwen candidates numerically, but
   its worst observed violation — fabricating tool names
   (`identify_categorical_columns`, `select_columns`, `onehot_encode`)
   that don't exist anywhere in the given `ALLOWED OPERATIONS` list —
   is a categorically more severe failure than either Qwen model
   produced (both always at least named a real tool). On schema
   fidelity specifically, llama3.2:3b is not an improvement over the
   qwen3:4b baseline.
3. **Reliability across repeated calls** — qwen3.5:4b wins outright
   (100%, no failures of any kind). qwen3:4b and llama3.2:3b tie
   numerically (4/5, 80%), but llama3.2:3b's failure (a genuine
   HTTP 500 after 317s of wall time) is arguably a worse signal than a
   clean timeout — it indicates the Ollama server itself errored, not
   merely that generation ran long.
4. **Low repeated-invalid-plan rate** — llama3.2:3b ties qwen3.5:4b
   (0/2 vs. qwen3:4b's 2/2) — a genuine win over the baseline here.
5. **Latency** — llama3.2:3b is clearly fastest on completed calls
   (82.3s avg vs. 125.4s / 146.4s) and has the highest decode rate
   (4.90 tok/s) — consistent with being the smallest, non-thinking-mode
   model of the three (lowest average output-token count alongside
   qwen3.5:4b, at roughly half of qwen3:4b's).
6. **Resources** — llama3.2:3b wins (smallest: 2.0GB).

Criterion 1 is a three-way tie. Criterion 2 — the deciding one — still
favors **qwen3:4b**, the original baseline, not llama3.2:3b: the
baseline's failure mode remains the narrowest, most singular, and most
predictable of the three, while llama3.2:3b introduces a new and more
severe class of schema violation (nonexistent tool names) even though
its raw violation count average sits below qwen3.5:4b's. llama3.2:3b's
real wins on criteria 4–6 (repeated-plan avoidance, latency, resource
footprint) do not overturn a worse (or at best equal-but-different-
in-kind) result on the higher-priority argument/schema-correctness
criterion, per the explicit "do not select a model solely because it's
faster" rule.

### Verdict

**Llama 3.2 3B does not clearly outperform both Qwen candidates** — it
is faster and avoids qwen3:4b's repeated-invalid-plan problem, but it
does not improve on argument/schema correctness (the higher-priority
criterion) and introduces a new failure mode (a genuine HTTP 500, and
tool-name-level hallucination) that neither Qwen model exhibited.

Per instructions, Gemma 3 4B was **not** automatically benchmarked.
**Next candidate, pending explicit go-ahead: Gemma 3 4B.**

No production change was made.

## Update: gemma3:4b vs. all three prior candidates (controlled comparison)

**Status: investigation only, same as above. Production model, timeout,
prompts, graph routing, retry logic, and deterministic validation are
still unchanged.**

`gemma3:4b` was not installed. Checked disk space first (339GB free —
ample). Pulled **only** `gemma3:4b` (3.3GB; `qwen3:4b`/`qwen3.5:4b`/
`llama3.2:3b`/everything else untouched). Benchmarked with the exact
same methodology, dataset, context, prompts, tool definitions, and
call budget via
[`backend/benchmark_run_gemma3.py`](benchmark_run_gemma3.py), which
calls `benchmark_planning_models.py`'s functions unmodified. Merged
results in `benchmark_results.json`.

### Results — gemma3:4b (5 real Ollama calls)

| Trial | Outcome | Wall time | total_duration | load_duration | eval (gen) | output tok | violations |
|---|---|---:|---:|---:|---:|---:|---:|
| first_attempt_0 | structured, invalid | **909.4s** | 907.4s | 12.9s | 785.1s | 498 | 6 |
| replan_after_0 | structured, invalid, **repeated identical** | 205.3s | 203.2s | 2.8s | 103.6s | 469 | 6 |
| first_attempt_1 | structured, invalid | 263.4s | 260.4s | 2.6s | 227.8s | 494 | 6 |
| replan_after_1 | structured, invalid, not repeated | **786.2s** | 783.9s | **500.1s** | 86.6s | 376 | 5 |
| first_attempt_2 | structured, invalid | 198.4s | 196.3s | 2.5s | 89.9s | 463 | 5 |

Two important, distinct observations from this run, both **out of
scope to fix here** (investigation only) but worth recording:

1. **Two of five calls (909.4s, 786.2s) exceeded the current 600s
   production timeout in wall-clock terms, yet neither registered as a
   `timeout` error.** The benchmark's `_ollama_call()` mirrors
   `OllamaProvider.generate_plan()`'s exact request construction,
   including `urllib.request.urlopen(request, timeout=600.0)`. Python's
   socket `timeout` parameter bounds how long a single blocking I/O
   operation can wait for *more* data, not the total request duration —
   if the connection keeps receiving any bytes (even slowly, even
   partial TCP segments) before the full JSON body is assembled, the
   countdown resets on each read rather than accumulating toward a hard
   600s ceiling. Both of these gemma3:4b calls apparently exhibited
   exactly that pattern. **This means PIPER's documented 600s timeout
   may not be the hard ceiling on total planning latency it's assumed
   to be** — a real, previously-unobserved finding, surfaced here for
   the first time because gemma3:4b is the first candidate slow enough
   to expose it. Not investigated further or fixed, per this session's
   explicit "investigation only" scope — flagging for separate
   attention.
2. **`replan_after_first_attempt_1`'s `load_duration` was 500.1
   seconds** — vs. 2.5–2.9s for every other completed call in this
   entire benchmark session (all four models). This strongly suggests
   host-level memory pressure (four different multi-GB models now
   resident on disk, likely contending for RAM as Ollama swaps between
   them across this session's sequential benchmark runs), not a
   `gemma3:4b`-specific defect. Also not investigated further here —
   noted as a possible confound on this one data point's latency
   figures.

`gemma3:4b`'s violations follow the same systemic pattern observed in
`llama3.2:3b` (a wrong-field-name convention, here consistently
`column_names` instead of the real `column`/`columns` fields) but
applied far more comprehensively: **every single proposed step in
every one of gemma3:4b's five plans failed validation** (5–6
violations per attempt against typically 4–5 proposed steps) — the
highest violation density of any of the four models benchmarked this
session. Unlike llama3.2:3b, gemma3:4b never hallucinated a tool name
outside the allowlist — every violation was an argument-shape/field-
naming mistake on a real tool.

### Four-way comparison (5 calls each)

| Metric | qwen3:4b | qwen3.5:4b | llama3.2:3b | gemma3:4b |
|---|---:|---:|---:|---:|
| Deterministic-validation-passed rate | 0/5 | 0/5 | 0/5 | 0/5 |
| Structured-plan-produced rate | 4/5 (80%) | 5/5 (100%) | 4/5 (80%) | 5/5 (100%)* |
| Violations per completed attempt (mean) | **1.00** | 3.00 | 2.50 | **5.60 (worst)** |
| Repeated-identical-invalid-plan rate | 2/2 (100%) | 0/2 (0%) | 0/2 (0%) | 1/2 (50%) |
| Avg wall time (completed calls) | 146.4s | 125.4s | **82.3s (fastest)** | **472.5s (worst, 2 calls >600s)** |
| Avg generation rate | 3.23 tok/s | 3.97 tok/s | **4.90 tok/s** | 3.37 tok/s** |
| Model size on disk | 2.5 GB | 3.4 GB | **2.0 GB (smallest)** | 3.3 GB |

\* Structured-response rate looks perfect, but 2/5 calls took longer
than the current 600s timeout in wall-clock terms — see finding #1
above. Treated as a strict "got a usable response within budget"
measure, gemma3:4b's real reliability is worse than this number alone
suggests.
\** One call's generation rate (0.63 tok/s) is a clear outlier
consistent with the load-time stall above; excluding it, the other
four calls average ~4.05 tok/s — still unremarkable, not a
particular strength.

### Applying the decision criteria, in priority order

1. **Correct executable PIPER plan** — four-way tie: 0/5 for every model.
2. **Correct tool arguments/schema** — **qwen3:4b remains the clear
   winner**, and gemma3:4b is now the clear **worst** of all four
   candidates: nearly every proposed step across all five attempts
   failed validation (mean 5.60 violations/attempt — almost 6x
   qwen3:4b's baseline, and nearly double qwen3.5:4b's).
3. **Reliability across repeated calls** — nominally tied with
   qwen3.5:4b (100%), but qualified by finding #1 above: 2 of 5 calls
   ran longer than the current production timeout.
4. **Low repeated-invalid-plan rate** — gemma3:4b: 1/2 (50%), worse
   than qwen3.5:4b/llama3.2:3b (0/2 each), better than qwen3:4b (2/2).
5. **Latency** — gemma3:4b is clearly the **worst** of all four: mean
   wall time more than 3x llama3.2:3b's and roughly double the other
   two Qwen candidates, with two individual calls exceeding the current
   600s timeout in wall-clock terms.
6. **Resources** — 3.3GB, roughly tied with qwen3.5:4b (3.4GB),
   larger than llama3.2:3b (2.0GB) and qwen3:4b (2.5GB).

Criterion 1 is a four-way tie. Criterion 2 — the deciding one — is not
just unfavorable to gemma3:4b, it is the **worst result of any
candidate tested this session**, and gemma3:4b also loses decisively
on latency (criterion 5). There is no dimension on which gemma3:4b
outperforms qwen3:4b at a priority level that matters under the locked
decision order.

### Verdict

**Gemma 3 4B does not establish an advantage over qwen3:4b or any
other candidate tested this session — it is the weakest performer of
the four on the deciding criterion (argument/schema correctness) and
by far the slowest.**

This completes the originally scoped four-candidate list
(`qwen3:4b`, `qwen3.5:4b`, `llama3.2:3b`, `gemma3:4b`). **Across all
four, and across 20 real Ollama calls total this session, no
candidate has ever produced a plan that passes PIPER's deterministic
`validate_proposed_plan()`.** qwen3:4b remains the best performer on
the highest-discriminating criterion (argument/schema correctness)
throughout, despite its own real problems (100% repeated-identical-
invalid-plan rate, one outright timeout). No production change was
made.

## Update: root-cause investigation and planner-contract fix

**Status: not a benchmark. Root-cause investigation into WHY all four
candidates scored 0/20 valid plans, followed by a scoped, additive fix
to the planner prompt. Production model, timeout, prompts' section
structure, graph routing, retry logic, and deterministic validation
logic itself are unchanged — see exactly what changed below.**

### Investigation

All 11 candidate questions posed were checked against the actual code
and the real benchmark evidence (not assumed):

| # | Question | Finding |
|---|---|---|
| 1 | Does the prompt clearly describe allowed tool NAMES? | Yes — just names, nothing else |
| 2 | Is every tool ARGUMENT explicitly described? | **No — not one argument, anywhere, for any tool** |
| 3 | Does the model see the same schema validation expects? | No — validation's real contract was never rendered into the prompt at all |
| 4 | Are Pydantic/JSON-schema tool definitions passed to Ollama correctly? | Partially — `PLAN_JSON_SCHEMA` constrains the outer envelope correctly, but `arguments` is declared as an unconstrained `{"type": "object"}`, so per-tool argument shape is never enforced at the decoding level even in principle |
| 5 | Is field naming ambiguous anywhere? | Yes, unavoidably so without documentation: `drop_column` (singular `column`) vs. `encode_categorical_features`/`scale_features` (plural `columns`) — a legitimate, correct distinction at the tool-implementation level, but never explained |
| 6 | Is singular/plural naming inconsistent? | Same as #5 — intentional and correct, but undocumented |
| 7 | Do prompt examples contradict the schema? | No examples existed at all — absence, not contradiction |
| 8 | Is the planner asked for unnecessary reasoning/alternate formats? | No — `REQUIRED_OUTPUT_FORMAT` is clear and unambiguous |
| 9 | Does qwen3's thinking mode interfere with structured extraction? | No — `structured_plan_produced` was true for 17/20 real calls; the 3 failures were transport-level (1 timeout, 1 HTTP 500, 1 borderline) never parsing failures |
| 10 | Is the parser transforming responses incorrectly? | No — `_extract_content()`/`_strip_markdown_fences()`/`ProposedPlan.model_validate()` are minimal and faithful; violations reported exactly what each model actually proposed |
| 11 | Does the benchmark expose a real gap normal tests don't cover? | **Yes** — `FakeLLMProvider`/`heuristic_llm_provider()` always produce already-correct arguments by construction; nothing in the pre-existing suite ever tested whether the prompt ALONE gives a real model enough information to guess correctly |

### Root cause

The `=== ALLOWED OPERATIONS ===` prompt section — confirmed by
rendering the actual production prompt and reading it byte-for-byte —
rendered nothing but a bare JSON array of tool_name strings:

```json
[
  "convert_column_type",
  "drop_column",
  "encode_categorical_features",
  "impute_missing_values",
  "scale_features"
]
```

`=== DETERMINISTIC CONSTRAINTS ===` said arguments "must match that
operation's required shape" without ever stating what that shape is,
anywhere in the prompt. This is exactly what all 20 real calls this
session received — no argument names, types, required/optional status,
enum values, or examples, for any tool, ever. Every observed violation
maps directly onto this gap:

- `drop_column` given a `columns` list — indistinguishable from the
  real plural shape `encode_categorical_features`/`scale_features`
  actually use, with nothing telling a model these differ.
- Invented field names (`method`, `column_names`, `columns_to_encode`,
  `columns_to_drop`) — a model has no way to know these are wrong when
  the real names were never stated.

Four independently-trained models (Qwen3, Qwen3.5, Llama 3.2, Gemma 3)
converging on the same class of guessing failure is strong evidence
this was a genuine contract gap, not a per-model capability limit.

### Fix — additive, documentation-only

| File | Change |
|---|---|
| `app/agent/plan_validation.py` | New `TOOL_ARGUMENT_SCHEMAS: dict` — declarative, LLM-facing per-tool contract (argument name/type/required/enum/note/example), hand-verified against the real `_validate_step_arguments()` logic. `validate_proposed_plan()` itself unchanged. |
| `app/llm/provider.py` | `LLMPlanningContext` gained one additive field: `tool_schemas: dict = Field(default_factory=dict)`. Every existing caller is unaffected (default empty). |
| `app/llm/prompts.py` | New `_format_allowed_operations()`: renders the full per-tool contract + example when `tool_schemas` is populated; falls back to the original bare list (pinned byte-for-byte by a regression test) when empty. Used by both prompt builders. |
| `app/agent/nodes/real_nodes.py` | `plan_node_v2` now passes `tool_schemas=TOOL_ARGUMENT_SCHEMAS` — the only production call site wired to the real contract. |

**Why this is architecturally correct:** it changes what the LLM is
*told*, never what is *trusted or executed*. `validate_proposed_plan()`
remains the sole, unweakened authority (proven by new tests showing
the literal real-world invalid plans are still rejected exactly as
before). Nothing about the allowlist, canonicalization, duplicate-plan
detection, retries, guardrails, execution, the timeout, or the
production model changed. `TOOL_ARGUMENT_SCHEMAS` is prompt content
only, derived from — and tested against — the same module that already
owned the real contract, so it cannot silently drift from what
validation actually enforces (see the anti-drift test below).

**Deliberately not done this round:** making `PLAN_JSON_SCHEMA`
tool_name-conditional (a discriminated union so Ollama's grammar-based
decoding could enforce per-tool argument shape, not just the outer
envelope) — investigated, but Ollama/llama.cpp's conditional
JSON-schema support varies by backend/version and this is a larger,
less certain change than the "smallest correction necessary" scope
called for. Worth a dedicated, separately-scoped investigation later.

### New test coverage (29 tests)

- `tests/test_plan_validation.py::TestToolArgumentSchemasMatchValidator`
  (11 tests) — schema keys match `ALLOWED_TOOL_NAMES` exactly; every
  documented example passes the real `validate_proposed_plan()`
  (parametrized over all 5 tools); documented enums match the
  validator's own constants; the singular/plural distinction is
  explicitly pinned.
- `tests/test_plan_validation.py::TestRealWorldBenchmarkFailurePatternsRejected`
  (10 tests) — named regressions reproducing the *literal* failures
  observed from all four benchmarked models this session, proving they
  were (and remain) correctly rejected.
- `tests/test_llm_provider.py::TestToolSchemaRenderedIntoPrompt`
  (4 tests) — additive-field default, byte-identical fallback
  rendering, full-contract rendering, REPLAN-prompt carry-through.
- `tests/test_planner_contract_titanic.py` (new file, 4 tests) — built
  from the real `benchmark_data/train.csv` fixture via the exact
  production context-building path. Proves the real prompt for this
  real dataset now documents the real contract; a hand-built,
  schema-conformant plan passes `validate_proposed_plan()` cleanly; and
  the literal qwen3:4b failure observed against this exact dataset is
  still correctly rejected.

### Test results

- Targeted (new tests only): 96/96 passed.
- Broader relevant set (planner/validation/Ollama-provider files):
  194/194 passed.
- Full `pytest -q` (production code changed — core planning path):
  **770 passed, 5 skipped** (741 baseline +29, exactly matching the new
  tests added), 0 failures, 45m58s. No regressions.

### Readiness for another real-model benchmark

Structurally ready — every model now receives the actual argument
contract instead of guessing from bare tool names. **Not re-verified
against a real model this session** (none was re-benchmarked, per
explicit instruction) — whether this measurably raises the valid-plan
rate above 0% for any of the four already-tested candidates is an open
empirical question for the next benchmark round. Model decision
remains **OPEN**.

## Update: AFTER-fix re-benchmark — qwen3:4b only

**Objective: determine whether the planner-contract fix actually
raises the valid-plan rate.** Controlled, single-candidate, real-Ollama
re-benchmark of `qwen3:4b` only (qwen3.5:4b/llama3.2:3b/gemma3:4b were
NOT re-benchmarked, per explicit instruction), same Titanic fixture,
same call budget (3 first-attempt trials + up to 2 REPLAN follow-ups
—triggered only if an attempt is invalid), same target/objective/tool
allowlist/deterministic validator. The only intentional difference from
the BEFORE run: the prompt now includes `tool_schemas` — because that
is the fix being tested, and because that's what `plan_node_v2` now
actually sends in production. `benchmark_planning_models.py`'s context
construction was updated (both the first-attempt and REPLAN-follow-up
paths) to include `tool_schemas=TOOL_ARGUMENT_SCHEMAS`, matching
`plan_node_v2` exactly — without this the benchmark harness would have
silently kept testing the OLD, pre-fix prompt. No prompt wording was
tuned or iterated on for this run; the schema content is exactly what
`app/agent/plan_validation.py` already declares.

Results written to a **separate** file,
`backend/benchmark_results_after_fix.json`, deliberately not merged
into `benchmark_results.json` — preserves the original BEFORE baseline
untouched for a clean diff (confirmed intact: still 5 trials, 1
violation each, same as originally recorded).

### AFTER-fix results (qwen3:4b, 3 real Ollama calls)

| Trial | Valid? | Violations | Wall time | eval (gen) | output tok | Proposed plan |
|---|---|---:|---:|---:|---:|---|
| first_attempt_0 | **True** | 0 | 661.2s* | 521.3s | 375 | impute Age(median), drop Name, drop Ticket, encode [Sex, Embarked] |
| first_attempt_1 | **True** | 0 | 143.3s | 130.3s | 470 | impute Age(median), encode [Sex, Embarked], drop Name, drop Ticket, scale [Age, Fare] |
| first_attempt_2 | **True** | 0 | 100.4s | 97.4s | 304 | impute Age(median), drop Cabin, encode [Sex, Embarked], scale [Age, Fare] |

\* Exceeded the current 600s timeout in wall-clock terms without
raising a `timeout` error — the same `urllib` socket-read-timeout
nuance already documented from the gemma3:4b run (bounds gaps between
reads, not total duration). Recurred here too; still not investigated
or fixed, per this session's scope. This call's generation was also
genuinely slower per-token (0.72 tok/s vs. 3.1–3.6 tok/s for the other
two) — consistent with the already-documented high latency variance
for this model on this hardware (CLAUDE.md's Batch 5 distribution:
143–418s), not something newly introduced by the fix.

**No REPLAN follow-ups occurred** — by design: they only trigger when
an attempt is invalid, and all three first attempts were valid. This is
the same methodology as the BEFORE run (which needed 2 follow-ups
precisely because its first attempts kept failing); needing zero this
time is itself part of the result, not a deviation from it.

Crucially, every `drop_column` call across all three trials used the
correct singular shape — `{"column": "Name"}`, `{"column": "Ticket"}`,
`{"column": "Cabin"}` — never once the plural `columns` list that was
qwen3:4b's single, 100%-consistent mistake in the BEFORE run. The
proposed plans are also qualitatively sensible for Titanic (impute the
one genuinely missing numeric column, drop high-cardinality/identifier
columns, encode the two real categorical predictors, scale the two
real numeric ones) — a real improvement in plan quality, not merely
schema compliance.

### BEFORE vs. AFTER (qwen3:4b only)

| Metric | BEFORE (original baseline) | AFTER (contract fix) |
|---|---:|---:|
| Deterministic-validation-passed rate | **0/5 (0%)** | **3/3 (100%)** |
| Structured-plan-produced rate | 4/5 (80%, 1 timeout) | 3/3 (100%) |
| Violations per completed attempt (mean) | 1.00 (every attempt) | **0.00 (every attempt)** |
| Failure pattern | `drop_column` given `columns` list (100% of attempts) | none observed |
| Repeated-identical-invalid-plan rate | 2/2 (100%) | N/A — no invalid plans to repeat |
| Avg output tokens | 380.8 | 383.0 (essentially unchanged) |
| Avg prompt tokens | 2116.5 (first-attempt: 1842) | 2247 (first-attempt only; +22% vs. before, the cost of the added schema) |
| Avg wall time (completed calls) | 146.4s | 301.6s (121.9s excl. the one slow outlier call) |
| Timeouts (wall-clock >600s) | 1/5 | 1/3 (same read-timeout nuance, not a new regression) |

### Decision

**Valid-plan reliability improved substantially: 0% → 100%.** Per the
locked decision branches: **the planner-contract fix is documented as
having resolved the primary failure mode** (the singular/plural
argument-shape confusion that accounted for 100% of qwen3:4b's
violations in the original benchmark). `qwen3:4b` remains the current
development baseline. No model switch was made.

**Caveats, stated plainly rather than overclaimed:** this is a 3-call
sample (n=3 valid, following the same conditional-REPLAN methodology
as the original 5-call baseline) — real, not simulated, but still a
small sample from a model with already-documented high latency/output
variance. Latency did not improve (one call exceeded 600s in wall-clock
terms, same as before) and was not the fix's goal — the fix targets
plan *correctness*, not speed; no latency optimization was attempted or
should be inferred from these numbers. Whether 100% validation-pass
holds up over more calls, and whether the fix also helps
qwen3.5:4b/llama3.2:3b/gemma3:4b (not re-tested this round, per
instruction), remain open questions for a future, explicitly-requested
benchmark round.

### Files changed this round (benchmark harness only, no production code)

- `backend/benchmark_planning_models.py` — `main()`'s
  `first_attempt_context` and `benchmark_model()`'s `replan_context`
  both now include `tool_schemas=TOOL_ARGUMENT_SCHEMAS`, matching
  `plan_node_v2`'s real post-fix behavior. Without this the benchmark
  harness would have kept testing the old, pre-fix prompt.
- `backend/benchmark_run_qwen3_after_fix.py` (new) — isolated runner
  scoped to `qwen3:4b` only, writing to a separate results file so the
  BEFORE baseline is never overwritten.

### Test results

No production code was changed this round (only the benchmark harness
scripts above). The directly relevant planner-contract test files were
re-run as a confirmatory check anyway: `test_plan_validation.py`,
`test_llm_provider.py`, `test_planner_contract_titanic.py`,
`test_llm_graph_integration.py` — **117 passed**, 0 failed. Full
regression suite not re-run (no production code changed, consistent
with the standing policy).

## Update: Phase 2A — controlled `keep_alive` experiment

**Status: measurement-only investigation, followed by a small,
evidence-justified production change. Model, prompts, tool schemas,
planner logic, timeout, graph routing, and retry behavior all
unchanged — see exactly what changed below.**

### Method

Isolated script ([`backend/benchmark_keep_alive_experiment.py`](benchmark_keep_alive_experiment.py))
built the current production prompt (post-contract-fix,
`tool_schemas` included) **once** and reused it byte-identical across
every trial — prompt content was never a variable. `ollama stop
qwen3:4b` forced genuine COLD state before each cold trial; `ollama ps`
was polled immediately before the critical calls for **ground-truth**
residency confirmation, not inference from latency alone. 6 real Ollama
calls total:

- **Group A (current/default — no `keep_alive` sent):** A1 (forced
  cold) -> A2 (immediate follow-up, same prompt) -> [reset] -> A3
  (forced cold) -> **330s wait** (past Ollama's documented 5-minute
  default) -> A4 (after the gap).
- **Group B (explicit `keep_alive="30m"`):** [reset] -> B1 (forced
  cold) -> **330s wait** (identical gap duration) -> B2 (after the
  gap).

### Results

| Trial | keep_alive | Wall | load | prompt_eval | eval (gen) | Valid? |
|---|---|---:|---:|---:|---:|---|
| A1_cold_default | none | 407.2s | 10.33s | 159.58s | 235.15s | True |
| A2_immediate_default | none | 133.4s | 11.67s | **0.22s** | 119.31s | True |
| A3_cold_default | none | 397.7s | 10.25s | 259.54s | 125.64s | True |
| A4_after_gap_default | none | **690.4s** | 9.89s | 550.74s | 127.57s | True |
| B1_cold_explicit | 30m | 1131.2s | 9.03s | 1010.00s | 109.96s | True |
| B2_after_gap_explicit | 30m | **186.0s** | 69.60s | **0.28s** | 113.95s | True |

**Ground-truth residency** (`ollama ps`, not inferred):
- Before A2 (no wait): **resident** — Ollama's own 5-minute default
  does cover an immediate back-to-back call.
- Before A4 (330s gap, default): **not resident** — confirmed evicted.
- Before B2 (330s gap, explicit `keep_alive=30m`): **resident**,
  reporting "24 minutes from now" — exactly consistent with a 30-minute
  timer started at B1 with ~6 minutes elapsed.

All 6 calls produced valid plans (100%, consistent with the
post-contract-fix result — now 9/9 real calls valid since that fix).

### The headline comparison: A4 vs. B2

**Identical scenario in every respect except `keep_alive`** — same
330-second gap, same byte-identical prompt, same model: under the
current default, the gap evicts the model and the next call pays the
full cold cost (**690.4s**); with an explicit `keep_alive=30m`, the
model survives the identical gap and the next call is fast
(**186.0s**). **73.1% wall-time reduction**, directly attributable to
residency (confirmed by `ollama ps`, not just latency numbers).

**Honest caveat on the COLD-state numbers themselves:** they were far
more variable than expected — `prompt_eval_duration` for cold calls
ranged from 159.58s to a startling **1010.00s** (B1), more than 6x
spread across supposedly-equivalent cold starts on the same prompt.
This is plausibly OS-level page-cache/memory contention on this shared
dev machine (this session has loaded/unloaded 4 different multi-GB
models many times) rather than a stable, precisely-reproducible "cold
cost." If anything this strengthens the case for keeping the model
resident: cold-start cost is not just *high*, it is **unpredictable
enough to occasionally be catastrophic** (B1 alone took nearly 17
minutes), and warm calls were uniformly fast and consistent
(0.22s/0.28s prompt_eval, both confirmed resident) regardless of that
variance.

`load_duration` was NOT a reliable warm/cold signal on its own (stayed
9-12s across most calls regardless of residency, and B2 — confirmed
warm — showed 69.60s) — `prompt_eval_duration` cross-validated against
`ollama ps` was the reliable signal throughout.

### Decision

**Substantial, confirmed improvement — implementing the smallest
production change.** Added a configurable `keep_alive` to
`OllamaProvider`:

| File | Change |
|---|---|
| `app/llm/ollama_provider.py` | New `DEFAULT_KEEP_ALIVE = "10m"` constant (evidence-documented); `OllamaProvider.__init__` gained a `keep_alive` parameter reading `PIPER_OLLAMA_KEEP_ALIVE` (same override precedence as `host`/`model`/`timeout_seconds`); `generate_plan()`'s payload now includes `"keep_alive": self.keep_alive`. |
| `docker-compose.yml` | `PIPER_OLLAMA_KEEP_ALIVE: ${PIPER_OLLAMA_KEEP_ALIVE:-10m}` added alongside the existing `PIPER_OLLAMA_*` env vars. |

**10 minutes** was chosen (not the 30m tested) to match
`DEFAULT_TIMEOUT_SECONDS` — long enough to survive a realistic
PLAN-to-REPLAN gap (comfortably covering even a single call's own
worst-case duration) without keeping the model resident indefinitely
for no reason, satisfying "not create an indefinitely running model
without reason." No shutdown-hook changes were needed — Ollama manages
model residency independently of PIPER's process lifecycle; the model
unloads on its own timer regardless of whether PIPER is still running.
Nothing about deterministic validation, routing, retries, or execution
was touched.

**New test coverage (4 tests, `tests/test_llm_provider.py`):**
`test_defaults_to_documented_keep_alive`, `test_reads_keep_alive_from_environment_variable`,
`test_explicit_keep_alive_constructor_arg_overrides_environment`,
`test_keep_alive_never_disabled_by_default` — plus the existing
`test_request_includes_model_prompt_and_json_schema_format` was
extended to assert `keep_alive` actually reaches the real wire
request, not just the instance attribute.

### Test results

- `test_llm_provider.py` alone: 46/46 passed.
- Broader relevant set (planner/validation/Ollama-provider files):
  198/198 passed.
- Full `pytest -q` (production code changed in the core planner
  transport path): **774 passed, 5 skipped** (770 baseline +4, exactly
  matching the new tests added), 0 failures, 29m32s. No regressions.

### Answering the required report items

1. **Cold latency:** highly variable, 159.58s-1010.00s `prompt_eval_duration`
   (mean ~495s, median ~405s) on top of ~10s load and ~110-235s
   generation — total wall time 397.7s-1131.2s across 4 cold trials.
2. **Warm latency:** consistently ~0.2-0.3s `prompt_eval_duration`
   (both warm trials), total wall time 133.4s/186.0s (generation-bound).
3. **Improvement:** 73.1% wall-time reduction in the cleanest
   apples-to-apples comparison (A4 vs. B2, identical gap/prompt).
4. **Did the model actually remain resident?** Yes — confirmed via
   `ollama ps` ground truth, not inferred: resident before B2 (explicit
   keep_alive survived the 330s gap), not resident before A4 (default
   evicted across the identical gap).
5. **Worth keeping?** Yes — implemented as described above.
6. **Next highest-impact candidate:** generation time itself
   (`eval_duration`, ~110-235s, now the dominant remaining cost once
   prompt-eval is eliminated) — per Phase 1's finding, investigate
   whether Ollama/qwen3 exposes a reasoning-budget/`think` control to
   reduce non-essential reasoning-token volume, without disabling
   thinking mode or compromising the 100% valid-plan rate. Not started;
   needs its own explicit go-ahead and before/after measurement.

## Update: Phase 2B (`think`) + post-contract Qwen comparison

Two investigations, no production code changed by either.

### Phase 2B — `think` parameter: NEGATIVE RESULT

Ollama 0.32.9 / qwen3:4b exposes `think` (bool), `num_predict`, `stop`,
and standard sampling params. No reasoning-budget parameter exists.
Controlled 4-call A/B on the real Titanic prompt, only `think` varied
([`benchmark_generation_control_experiment.py`](benchmark_generation_control_experiment.py)):

| Metric | Baseline | `think=false` | Δ |
|---|---:|---:|---|
| `eval_duration` | 139.9s | 150.5s | +10.6s |
| Output tokens | 382.5 | 411.5 | +29 |
| Tokens/sec | 2.73 | 2.73 | — |
| Valid plans | 2/2 | 2/2 | preserved |
| Reasoning location | `thinking` 1414–1883 ch | `response` 1798–1684 ch | relocated |

`think=false` **relocates** reasoning into the `response` field rather
than eliminating it. No latency gain. Not adopted; production sends no
`think` field.

### Post-contract qwen3:4b vs qwen3.5:4b — INCONCLUSIVE

3 trials/model, 6 real calls, full production REPLAN loop per trial,
symmetric cold/warm protocol with `ollama ps` ground-truth residency,
byte-identical SHA-256-pinned prompt. Isolated results namespace:
`benchmark_results/post_contract/` (pre-contract results untouched).

**Confound (user-approved):** qwen3:4b ships `temperature=0.6`,
qwen3.5:4b ships `temperature=1.0`. PIPER sends no `options`, so each
runs on its own defaults — production-realistic, but differences reflect
model + shipped sampling config jointly.

| Metric | qwen3:4b | qwen3.5:4b |
|---|---:|---:|
| First-attempt valid / Final valid | 3/3 / 3/3 | 3/3 / 3/3 |
| REPLAN rate / violations | 0/3 / 0 | 0/3 / 0 |
| Timeouts / technical failures | 0/3 / 0/3 | 0/3 / 0/3 |
| Mean / median time-to-valid-plan | 234.5s / 271.6s | **124.7s / 105.4s** |
| Mean generation latency | 184.6s | **60.4s** |
| Mean prompt-processing latency | 44.0s | 56.0s |
| Mean tokens/sec / output tokens | 2.63 / 445 | **3.88** / 231 |
| Cold / warm / warm | 302.9 / 271.6 / 128.9s | 250.7 / 105.4 / **18.1s** |
| Steps per plan | **[5,5,5]** | [3,5,1] |
| Imputes `Age` / scaling | **3/3 / 3/3** | 2/3 / 1/3 |
| Drops Name / Ticket / PassengerId | 3/3 / 3/3 / 0/3 | 0/3 / 1/3 / **2/3** |

Validity ties at 3/3. qwen3.5:4b wins every latency metric — but by
emitting ~half the output tokens, with erratic plan completeness; its
18.1s trial produced a 1-step plan that never imputes `Age`, yet was
fully VALID. **PIPER's validator enforces well-formedness, not
completeness.** Neither model ever handled `Cabin` (77% missing).

**Verdict: INCONCLUSIVE, no switch. qwen3:4b remains baseline.**
Minimum next experiment: +5 qwen3.5:4b trials at defaults (n=8) to test
whether the completeness variance is real; if it disappears, follow with
a temperature-matched (0.6) arm to isolate the confound.

## Verification performed

- `benchmark_planning_models.py` syntax-checked and smoke-tested
  against a local mock HTTP server (both a valid-plan shape and the
  real invalid-`drop_column` shape, including the REPLAN-repeat
  detection path) before any real Ollama call was made.
- Relevant deterministic test files re-run under the project's actual
  `.venv` (`langgraph==1.2.10`, matching `requirements.txt`):
  `test_llm_provider.py`, `test_llm_graph_integration.py`,
  `test_context_budget.py`, `test_duplicate_plan_prevention.py`,
  `test_plan_canonical.py`, `test_plan_diff.py`,
  `test_plan_validation.py`, `test_replan_duplicate_invalid_plan.py`,
  `test_sanitized_llm_context.py` — **165 passed**, 0 failed
  (295.41s). No production code was modified this session, so this
  confirms no regression, not a fix.
  - Note: an initial run using this machine's global Python (which has
    a stray, unrelated `langgraph==0.2.34` on its path, not this
    project's `.venv`) failed 33/165 tests with
    `ValueError: 'profile' is already being used as a state key` —
    reproduced with a direct `build_graph()` call outside pytest too.
    This is a pre-existing environment/interpreter mismatch on this
    machine (global Python vs. the project's pinned `.venv`), not a
    PIPER defect — confirmed by the identical suite passing cleanly
    once run through `.venv`'s correctly pinned `langgraph==1.2.10`.
    Worth knowing for next time: always invoke the project's `.venv`
    Python explicitly, since `python`/`pytest` on this machine's PATH
    resolve to the wrong interpreter.
- The real-Ollama-gated integration suite (`test_ollama_integration.py`,
  `PIPER_RUN_OLLAMA_TESTS=1`) was not additionally re-run — this
  benchmark itself already exercised the real `OllamaProvider`-equivalent
  call path 5 times against real Ollama, a strictly larger and more
  targeted sample than that suite's own real-Ollama calls, and no
  production code changed that could affect it.
- Full regression suite (`pytest -q`, 741 baseline) was **not** run —
  correctly out of scope, since no production code was modified.
- After the qwen3.5:4b comparison, the same relevant test files were
  re-run under the project's `.venv` — **165 passed**, 0 failed
  (268.41s). No production code was modified this session (only new,
  isolated benchmark scripts and this report), so this confirms no
  regression.
- After the llama3.2:3b comparison, the same relevant test files were
  re-run again under the project's `.venv` — **165 passed**, 0 failed
  (332.81s). No production code was modified this session (only new,
  isolated benchmark scripts and this report), so this confirms no
  regression.
- After the gemma3:4b comparison, the same relevant test files were
  re-run again under the project's `.venv` — **165 passed**, 0 failed
  (232.88s). No production code was modified this session (only new,
  isolated benchmark scripts and this report), so this confirms no
  regression.
