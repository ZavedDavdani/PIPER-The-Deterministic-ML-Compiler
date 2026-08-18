"""
BENCHMARK-ONLY SCRIPT — isolated from production code.

Not imported by app/ or tests/, not wired into the graph, not part of
any request path. Its only job is to measure real Ollama planning
latency/reliability for candidate models against the SAME real PIPER
planning workload, so a model-choice decision can be made from
evidence instead of guesswork (per CLAUDE.md's standing "measure,
don't guess" rule).

It deliberately REUSES the real production functions/classes rather
than reimplementing them, so what's measured here is provably the same
logic plan_node_v2 actually runs in production:
    - app.agent.tools.sanitized_llm_context.build_sanitized_llm_context
    - app.agent.tools.context_budget.apply_context_budget
    - app.llm.prompts.build_planning_prompt / build_replan_prompt
    - app.llm.ollama_provider.PLAN_JSON_SCHEMA / _extract_content /
      _strip_markdown_fences (the exact response-parsing logic
      OllamaProvider.generate_plan() uses internally)
    - app.llm.provider.ProposedPlan (schema validation)
    - app.agent.plan_validation.validate_proposed_plan (deterministic
      tool/argument allowlist check)
    - app.agent.plan_canonical.canonicalize_plan /
      app.agent.plan_diff.diff_plans (plan identity + REPLAN evidence,
      including the post-multi-format-ingestion REPLAN-duplicate-
      invalid-plan mechanism)

The ONE thing this script does that production code does not: it makes
the raw HTTP call to Ollama itself (mirroring OllamaProvider's exact
request construction) so it can capture Ollama's own reported
prompt_eval_count/eval_count/*_duration fields, which
OllamaProvider.generate_plan() deliberately discards after extracting
the plan. This is purely additive instrumentation — no production
request/response contract is changed.

Does NOT change: the production model, the timeout, prompts, graph
routing, retry logic, deterministic validators, or the tool allowlist.
Does NOT auto-download any Ollama model — a candidate not already
present in `ollama list` is recorded as unavailable and skipped.

Usage:
    python benchmark_planning_models.py
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from app.agent.plan_canonical import canonicalize_plan
from app.agent.plan_diff import diff_plans
from app.agent.plan_validation import ALLOWED_TOOL_NAMES, TOOL_ARGUMENT_SCHEMAS, validate_proposed_plan
from app.agent.state import PlanStep
from app.agent.tools.context_budget import apply_context_budget
from app.agent.tools.sanitized_llm_context import build_sanitized_llm_context
from app.llm.ollama_provider import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_TIMEOUT_SECONDS,
    PLAN_JSON_SCHEMA,
    _extract_content,
    _strip_markdown_fences,
)
from app.llm.prompts import build_planning_prompt, build_replan_prompt
from app.llm.provider import LLMPlanningContext, ProposedPlan
from app.schemas.failure import FailureInfo
from app.storage import InMemoryDatasetStore
from pydantic import ValidationError

REPO_ROOT = Path(__file__).parent.parent
DATASET_PATH = REPO_ROOT / "benchmark_data" / "train.csv"
TARGET_COLUMN = "Survived"
DATASET_ID = "dataset_titanic_benchmark"

CANDIDATE_MODELS = ["qwen3:4b", "qwen3.5:4b", "llama3.2:3b", "gemma3:4b"]
N_FIRST_ATTEMPT_TRIALS = 3
MAX_REPLAN_FOLLOWUPS = 2

RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"


def _installed_models() -> set[str]:
    proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30)
    names = set()
    for line in proc.stdout.splitlines()[1:]:  # skip header
        line = line.strip()
        if not line:
            continue
        names.add(line.split()[0])
    return names


def _load_dataset() -> InMemoryDatasetStore:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {DATASET_PATH}")
    raw = DATASET_PATH.read_bytes()
    df = pd.read_csv(io.BytesIO(raw))  # identical call to ingestion.py's CSV branch
    store = InMemoryDatasetStore()
    store.save(DATASET_ID, df)
    return store, df


def _ollama_call(model: str, prompt: str, host: str, timeout_seconds: float) -> dict:
    """
    Mirrors OllamaProvider.generate_plan()'s request construction
    exactly (same URL, same payload shape, same JSON schema, same
    stream=False), but returns the FULL raw decoded response body
    instead of just the extracted plan, plus timing. Never raises for
    ordinary failure modes — returns a dict with an "error" key
    instead, same contract style as the real provider.
    """
    payload = {"model": model, "prompt": prompt, "stream": False, "format": PLAN_JSON_SCHEMA}
    request_body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{host.rstrip('/')}/api/generate",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_bytes = response.read()
    except urllib.error.HTTPError as e:
        return {"error": f"http_error: HTTP {e.code}: {e.reason}", "wall_seconds": time.perf_counter() - t_start}
    except TimeoutError:
        return {"error": f"timeout: no response within {timeout_seconds}s", "wall_seconds": time.perf_counter() - t_start}
    except urllib.error.URLError as e:
        return {"error": f"provider_unavailable: {e.reason}", "wall_seconds": time.perf_counter() - t_start}
    t_end = time.perf_counter()

    try:
        body = json.loads(raw_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return {"error": f"malformed_response: envelope not valid JSON: {e}", "wall_seconds": t_end - t_start}

    if not isinstance(body, dict):
        return {"error": "malformed_response: envelope not a JSON object", "wall_seconds": t_end - t_start}

    body["_wall_seconds"] = t_end - t_start
    return body


def _parse_and_validate(body: dict, target_column: str) -> dict:
    """
    Reuses the exact same content-extraction/parsing/schema-validation
    functions OllamaProvider.generate_plan() uses internally, then runs
    the real validate_proposed_plan(). Returns a structured record of
    every stage's outcome.
    """
    record: dict = {
        "structured_plan_produced": False,
        "provider_error": None,
        "deterministic_validation_passed": None,
        "violations": [],
        "rejected_steps": [],
        "proposed_steps": [],
        "step_count": None,
    }

    if "error" in body:
        record["provider_error"] = body["error"]
        return record

    content = _extract_content(body)
    if content is None:
        record["provider_error"] = "malformed_response: no content in known fields"
        return record

    cleaned = _strip_markdown_fences(content)

    try:
        plan_json = json.loads(cleaned)
    except json.JSONDecodeError as e:
        record["provider_error"] = f"malformed_response: generated content not valid JSON: {e}"
        record["raw_content_excerpt"] = cleaned[:500]
        return record

    try:
        plan = ProposedPlan.model_validate(plan_json)
    except ValidationError as e:
        record["provider_error"] = f"invalid_plan_schema: {e}"
        record["raw_content_excerpt"] = cleaned[:500]
        return record

    record["structured_plan_produced"] = True
    record["step_count"] = len(plan.steps)
    record["proposed_steps"] = [
        {"tool_name": s.tool_name, "arguments": s.arguments} for s in plan.steps
    ]

    validation_result = validate_proposed_plan(plan.steps, target_column)
    record["deterministic_validation_passed"] = validation_result.valid
    record["violations"] = [v.model_dump(mode="json") for v in validation_result.violations]

    if not validation_result.valid:
        record["rejected_steps"] = [
            {"step_index": i, "tool_name": s.tool_name, "arguments": s.arguments}
            for i, s in enumerate(plan.steps)
        ]

    return record


def _ollama_stats(body: dict) -> dict:
    """Ollama's own reported duration/token counters, converted ns -> s. Empty dict if absent (e.g. on a transport error)."""
    if "error" in body:
        return {}
    ns_fields = ["total_duration", "load_duration", "prompt_eval_duration", "eval_duration"]
    stats = {f: (body[f] / 1e9 if f in body and isinstance(body[f], (int, float)) else None) for f in ns_fields}
    stats["prompt_eval_count"] = body.get("prompt_eval_count")
    stats["eval_count"] = body.get("eval_count")
    stats["done_reason"] = body.get("done_reason")
    return stats


def run_trial(model: str, host: str, context: LLMPlanningContext, label: str) -> dict:
    t_prompt_start = time.perf_counter()
    prompt = (
        build_replan_prompt(context) if context.failure_context is not None else build_planning_prompt(context)
    )
    t_prompt_end = time.perf_counter()

    body = _ollama_call(model, prompt, host, DEFAULT_TIMEOUT_SECONDS)

    t_parse_start = time.perf_counter()
    parsed = _parse_and_validate(body, TARGET_COLUMN)
    t_parse_end = time.perf_counter()

    trial = {
        "label": label,
        "prompt_chars": len(prompt),
        "prompt_construction_seconds": t_prompt_end - t_prompt_start,
        "ollama_wall_seconds": body.get("_wall_seconds"),
        "parse_and_validate_seconds": t_parse_end - t_parse_start,
        "ollama_stats": _ollama_stats(body),
        **parsed,
    }
    return trial


def benchmark_model(model: str, host: str, first_attempt_context: LLMPlanningContext) -> dict:
    print(f"\n=== Benchmarking {model} ===", flush=True)
    trials = []
    replan_followups_used = 0

    for i in range(N_FIRST_ATTEMPT_TRIALS):
        label = f"first_attempt_{i}"
        print(f"  [{model}] running {label} ...", flush=True)
        t0 = time.perf_counter()
        trial = run_trial(model, host, first_attempt_context, label)
        print(
            f"    -> {time.perf_counter() - t0:.1f}s wall, "
            f"structured_plan={trial['structured_plan_produced']}, "
            f"valid={trial['deterministic_validation_passed']}",
            flush=True,
        )
        trials.append(trial)

        if (
            trial["structured_plan_produced"]
            and trial["deterministic_validation_passed"] is False
            and replan_followups_used < MAX_REPLAN_FOLLOWUPS
        ):
            replan_followups_used += 1
            rejected_plan = [
                PlanStep(
                    step_id=f"step_{j + 1:02d}",
                    action="benchmark_rejected_step",
                    tool_name=s["tool_name"],
                    arguments=s["arguments"],
                    reasoning="benchmark",
                )
                for j, s in enumerate(trial["proposed_steps"])
            ]
            rejected_hash = canonicalize_plan(rejected_plan, TARGET_COLUMN).plan_hash()

            failure = FailureInfo(
                category="EVALUATION_ERROR",
                message=(
                    f"LLM-proposed plan failed deterministic tool/argument validation "
                    f"({len(trial['violations'])} violation(s))."
                ),
                evidence={
                    "violations": trial["violations"],
                    "rejected_steps": trial["rejected_steps"],
                },
                node="plan",
                attempt=0,
                retryable=True,
                human_intervention_required=False,
            )
            empty_canonical = canonicalize_plan([], TARGET_COLUMN)
            previous_canonical = canonicalize_plan([], TARGET_COLUMN)  # state.plan is still [] at this point in production, see plan_node_v2
            previous_plan_summary = diff_plans(empty_canonical, previous_canonical).model_dump(mode="json")

            replan_context = LLMPlanningContext(
                objective=first_attempt_context.objective,
                dataset_context=first_attempt_context.dataset_context,
                allowed_operations=first_attempt_context.allowed_operations,
                tool_schemas=first_attempt_context.tool_schemas,
                failure_context=failure.model_dump(mode="json"),
                previous_plan_summary=previous_plan_summary,
            )

            replan_label = f"replan_after_{label}"
            print(f"  [{model}] running {replan_label} ...", flush=True)
            t0 = time.perf_counter()
            replan_trial = run_trial(model, host, replan_context, replan_label)
            print(
                f"    -> {time.perf_counter() - t0:.1f}s wall, "
                f"structured_plan={replan_trial['structured_plan_produced']}, "
                f"valid={replan_trial['deterministic_validation_passed']}",
                flush=True,
            )

            if replan_trial["structured_plan_produced"]:
                new_plan_steps = [
                    PlanStep(
                        step_id=f"step_{j + 1:02d}",
                        action="benchmark_replan_step",
                        tool_name=s["tool_name"],
                        arguments=s["arguments"],
                        reasoning="benchmark",
                    )
                    for j, s in enumerate(replan_trial["proposed_steps"])
                ]
                new_hash = canonicalize_plan(new_plan_steps, TARGET_COLUMN).plan_hash()
                replan_trial["repeated_identical_invalid_plan"] = new_hash == rejected_hash
            else:
                replan_trial["repeated_identical_invalid_plan"] = None

            trials.append(replan_trial)

    return {"model": model, "trials": trials}


def main():
    print(f"Loading benchmark dataset from {DATASET_PATH}")
    store, df = _load_dataset()
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns, target='{TARGET_COLUMN}'")
    assert df.shape == (891, 12), f"Expected 891x12, got {df.shape}"

    t0 = time.perf_counter()
    sanitized_result = build_sanitized_llm_context(DATASET_ID, TARGET_COLUMN, store)
    t_sanitize = time.perf_counter() - t0
    assert sanitized_result.success, sanitized_result.error

    t0 = time.perf_counter()
    budgeted_context, budget_report = apply_context_budget(sanitized_result.data)
    t_budget = time.perf_counter() - t0

    print(f"Context prep: sanitize={t_sanitize * 1000:.1f}ms, budget={t_budget * 1000:.1f}ms")
    print(f"Context budgeting: {budget_report.model_dump()}")

    first_attempt_context = LLMPlanningContext(
        objective=f"Predict '{TARGET_COLUMN}' from the remaining columns (binary/multiclass classification).",
        dataset_context=budgeted_context.model_dump(mode="json"),
        allowed_operations=sorted(ALLOWED_TOOL_NAMES),
        tool_schemas=TOOL_ARGUMENT_SCHEMAS,
    )
    context_prep_report = {
        "sanitize_seconds": t_sanitize,
        "budget_seconds": t_budget,
        "budget_report": budget_report.model_dump(),
        "dataset_context_chars": len(json.dumps(first_attempt_context.dataset_context)),
    }

    installed = _installed_models()
    print(f"\nInstalled Ollama models: {sorted(installed)}")

    host = DEFAULT_OLLAMA_HOST
    all_results = {"context_prep": context_prep_report, "models": []}

    for model in CANDIDATE_MODELS:
        if model not in installed:
            print(f"\n=== {model}: UNAVAILABLE (not installed, not auto-downloading) ===")
            all_results["models"].append({"model": model, "status": "unavailable"})
            continue
        result = benchmark_model(model, host, first_attempt_context)
        result["status"] = "benchmarked"
        all_results["models"].append(result)
        RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
