"""
MEASUREMENT-ONLY SCRIPT — isolated from production code.

Post-contract-fix qwen3:4b vs. qwen3.5:4b comparison. The original
4-model benchmark (documented in CLAUDE.md / benchmark_report.md) ran
BEFORE the planner-contract fix (TOOL_ARGUMENT_SCHEMAS), so it cannot
be treated as a definitive model-capability ranking. This establishes
a clean post-contract comparison between the two Qwen candidates only.

Reuses production code directly wherever it exists — never a
reimplementation:
    - app.agent.tools.sanitized_llm_context.build_sanitized_llm_context
    - app.agent.tools.context_budget.apply_context_budget
    - app.llm.prompts.build_planning_prompt / build_replan_prompt
    - app.agent.plan_validation.TOOL_ARGUMENT_SCHEMAS / ALLOWED_TOOL_NAMES
      / validate_proposed_plan
    - app.agent.plan_canonical.canonicalize_plan
    - app.agent.plan_diff.diff_plans
    - app.schemas.failure.FailureInfo
    - app.agent.state.PlanStep
    - app.llm.ollama_provider.PLAN_JSON_SCHEMA / _extract_content /
      _strip_markdown_fences / DEFAULT_KEEP_ALIVE / DEFAULT_TIMEOUT_SECONDS
    - app.llm.provider.LLMPlanningContext / ProposedPlan

The one thing this script does that production doesn't: it makes the
raw HTTP call itself (identical shape to OllamaProvider.generate_plan())
so it can capture full Ollama stats and enforce the cold/warm
experimental protocol below.

EACH TRIAL runs the FULL production PLAN-node REPLAN loop faithfully —
attempt 0, and on failure (provider error OR validation rejection),
build a real REPLAN context exactly as plan_node_v2 does (FailureInfo,
rejected_steps evidence, canonicalize_plan/diff_plans, plan_history/
DUPLICATE_PLAN detection) — bounded by max_retries=2 (AgentState's
real default), i.e. up to 3 attempts per trial. This is intentionally
scoped to the PLAN node's own REPLAN behavior in isolation (no
execution/guardrails run) — consistent with every benchmark this
session.

Cold/warm protocol per model block: evict everything, force the target
model cold (`ollama stop`), verify via `ollama ps`, Trial 1, then
Trial 2/3 immediately after with no artificial delay. Actual residency
is recorded via `ollama ps` ground truth before every trial's attempt
0, regardless of intended labeling — a trial that runs long enough to
exceed keep_alive is recorded as cold, not mislabeled warm.

Results are written to backend/benchmark_results/post_contract/ — a
NEW, isolated namespace. The original benchmark_results.json and
benchmark_results_after_fix.json are never touched.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import benchmark_planning_models as b
from app.agent.plan_canonical import canonicalize_plan
from app.agent.plan_diff import diff_plans
from app.agent.state import PlanStep
from app.llm.ollama_provider import DEFAULT_KEEP_ALIVE
from app.schemas.failure import FailureInfo

MODELS = ["qwen3:4b", "qwen3.5:4b"]
N_TRIALS_PER_MODEL = 3
MAX_RETRIES = 2  # AgentState's real default (see app/agent/state.py)
BENCHMARK_VERSION = "post_contract_v1"

RESULTS_DIR = Path(__file__).parent / "benchmark_results" / "post_contract"


# --- Ollama residency control (verified against the installed 0.32.9 API) --


def _ollama_version() -> str:
    proc = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=30)
    return proc.stdout.strip() or proc.stderr.strip()


def _ollama_show_parameters(model: str) -> dict:
    """
    Captures a model's own embedded generation defaults (temperature,
    top_k/top_p, penalties, stop tokens) via `ollama show`.

    Recorded because these DIFFER between the two candidates and PIPER
    sends no `options` field, so each model runs on its own defaults —
    a deliberate, user-approved experimental choice (production-realistic:
    it measures what PIPER would actually get if it switched models),
    but a real confound that must travel with the raw results rather
    than live only in prose. Any difference observed between models is
    therefore attributable to model+its-shipped-sampling-config jointly,
    NOT to model capability in isolation.
    """
    proc = subprocess.run(["ollama", "show", model], capture_output=True, text=True, timeout=30)
    params: dict = {}
    in_params = False
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped == "Parameters":
            in_params = True
            continue
        if in_params:
            if not stripped or stripped in ("License", "Capabilities", "Model"):
                break
            parts = stripped.split(None, 1)
            if len(parts) == 2:
                key, value = parts[0], parts[1].strip()
                # `stop` legitimately appears multiple times — keep all.
                if key in params:
                    existing = params[key]
                    params[key] = existing + [value] if isinstance(existing, list) else [existing, value]
                else:
                    params[key] = value
    return params


def _ollama_ps() -> list[dict]:
    proc = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30)
    lines = proc.stdout.strip().splitlines()
    resident = []
    for line in lines[1:]:
        if line.strip():
            parts = line.split()
            resident.append({"name": parts[0], "raw": line.strip()})
    return resident


def _ollama_stop(model: str):
    subprocess.run(["ollama", "stop", model], capture_output=True, text=True, timeout=30)
    time.sleep(1)


def _evict_all():
    for entry in _ollama_ps():
        _ollama_stop(entry["name"])


def _is_resident(model: str) -> bool:
    return any(entry["name"] == model for entry in _ollama_ps())


# --- Fingerprinting ----------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256(text.encode("utf-8"))


# --- One LLM call (production request shape, full raw capture) ---------


def _call(model: str, prompt: str) -> dict:
    import urllib.request

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": b.PLAN_JSON_SCHEMA,
        "keep_alive": DEFAULT_KEEP_ALIVE,
    }
    request_body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{b.DEFAULT_OLLAMA_HOST.rstrip('/')}/api/generate",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    request_start = time.perf_counter()
    error = None
    body: dict = {}
    try:
        with urllib.request.urlopen(request, timeout=b.DEFAULT_TIMEOUT_SECONDS) as response:
            raw_bytes = response.read()
        body = json.loads(raw_bytes.decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - benchmark must record, never crash mid-run
        error = f"{type(e).__name__}: {e}"
    request_end = time.perf_counter()
    request_total_ms = (request_end - request_start) * 1000

    return {
        "request_start": request_start,
        "request_end": request_end,
        "request_total_ms": request_total_ms,
        "wall_clock_over_600s": (request_end - request_start) > 600.0,
        "transport_error": error,
        "body": body,
    }


def _parse_and_validate(body: dict, target_column: str) -> dict:
    t_parse_start = time.perf_counter()
    content = b._extract_content(body) if body else None
    cleaned = b._strip_markdown_fences(content) if content else None
    plan = None
    parse_error = None
    if cleaned is not None:
        try:
            plan_json = json.loads(cleaned)
            plan = b.ProposedPlan.model_validate(plan_json)
        except Exception as e:  # noqa: BLE001
            parse_error = f"{type(e).__name__}: {e}"
    elif content is None and body:
        parse_error = "no content in any known response field"
    t_parse_end = time.perf_counter()

    t_validate_start = time.perf_counter()
    validation = b.validate_proposed_plan(plan.steps, target_column) if plan else None
    t_validate_end = time.perf_counter()

    return {
        "parse_duration_ms": (t_parse_end - t_parse_start) * 1000,
        "validation_duration_ms": (t_validate_end - t_validate_start) * 1000,
        "plan": plan,
        "parse_error": parse_error,
        "validation": validation,
    }


def _ollama_stats(body: dict) -> dict:
    if not body:
        return {}
    return {
        "total_duration_s": body.get("total_duration", 0) / 1e9,
        "load_duration_s": body.get("load_duration", 0) / 1e9,
        "prompt_eval_duration_s": body.get("prompt_eval_duration", 0) / 1e9,
        "eval_duration_s": body.get("eval_duration", 0) / 1e9,
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
        "done_reason": body.get("done_reason"),
    }


# --- One trial: full production REPLAN loop -----------------------------


def _run_trial(model: str, trial_id: str, first_attempt_context, target_column: str) -> dict:
    attempts = []
    plan_history: list[str] = []  # canonical plan hashes attempted THIS TRIAL only
    final_plan_valid = False
    time_to_valid_plan = None
    cumulative_latency = 0.0
    duplicate_plan_triggered = False

    context = first_attempt_context
    attempt_number = 0

    residency_before = _ollama_ps()
    cold_or_warm = "warm" if _is_resident(model) else "cold"

    while attempt_number <= MAX_RETRIES:
        is_replan = attempt_number > 0

        t_prompt_start = time.perf_counter()
        prompt = b.build_replan_prompt(context) if is_replan else b.build_planning_prompt(context)
        t_prompt_end = time.perf_counter()

        call = _call(model, prompt)
        cumulative_latency += call["request_total_ms"] / 1000.0
        parsed = _parse_and_validate(call["body"], target_column)
        stats = _ollama_stats(call["body"])

        plan = parsed["plan"]
        validation = parsed["validation"]
        is_valid = bool(validation and validation.valid)

        attempt_record = {
            "model": model,
            "trial_id": trial_id,
            "attempt_number": attempt_number,
            "is_replan": is_replan,
            "cold_or_warm": cold_or_warm if attempt_number == 0 else "n/a (mid-trial, model stays resident)",
            "prompt_build_ms": (t_prompt_end - t_prompt_start) * 1000,
            "request_total_ms": call["request_total_ms"],
            "wall_clock_over_600s": call["wall_clock_over_600s"],
            "transport_error": call["transport_error"],
            "parse_duration_ms": parsed["parse_duration_ms"],
            "validation_duration_ms": parsed["validation_duration_ms"],
            "parse_error": parsed["parse_error"],
            "ollama_stats": stats,
            "output_tokens": stats.get("eval_count"),
            "tokens_per_sec": (stats.get("eval_count") / stats["eval_duration_s"])
            if stats.get("eval_count") and stats.get("eval_duration_s") else None,
            "structured_plan_produced": plan is not None,
            "tool_names": [s.tool_name for s in plan.steps] if plan else [],
            "tool_arguments": [{"tool_name": s.tool_name, "arguments": s.arguments} for s in plan.steps] if plan else [],
            "validation_passed": is_valid,
            "validation_violation_count": len(validation.violations) if validation else None,
            "validation_errors": [v.model_dump(mode="json") for v in validation.violations] if validation else [],
            "failure_category": None,
            "failure_details": None,
        }

        if is_valid:
            attempts.append(attempt_record)
            final_plan_valid = True
            time_to_valid_plan = cumulative_latency
            break

        # --- Not valid: build the REAL production REPLAN evidence -------
        if plan is None:
            # Provider-level failure (transport error, malformed JSON,
            # schema-invalid response) — mirrors plan_node_v2's
            # `if not provider_result.success` branch exactly. No
            # rejected-plan canonicalization/plan_history entry here,
            # matching production (there is no parseable proposal to
            # canonicalize).
            error_code = "timeout" if call["transport_error"] and "timeout" in call["transport_error"].lower() else (
                "transport_error" if call["transport_error"] else "malformed_response"
            )
            failure = FailureInfo(
                category="EVALUATION_ERROR",
                message=f"LLM planner failed: {parsed['parse_error'] or call['transport_error']}",
                evidence={"provider_error_code": error_code, "raw_error": parsed["parse_error"] or call["transport_error"]},
                node="plan",
                attempt=attempt_number,
                retryable=True,
                human_intervention_required=False,
            )
            attempt_record["failure_category"] = failure.category
            attempt_record["failure_details"] = failure.model_dump(mode="json")
            attempts.append(attempt_record)

            if attempt_number >= MAX_RETRIES:
                break
            attempt_number += 1
            empty_canonical = canonicalize_plan([], target_column)
            previous_plan_summary = diff_plans(empty_canonical, empty_canonical).model_dump(mode="json")
            context = b.LLMPlanningContext(
                objective=first_attempt_context.objective,
                dataset_context=first_attempt_context.dataset_context,
                allowed_operations=first_attempt_context.allowed_operations,
                tool_schemas=first_attempt_context.tool_schemas,
                failure_context=failure.model_dump(mode="json"),
                previous_plan_summary=previous_plan_summary,
            )
            continue

        # --- Structured plan produced, but rejected by validate_proposed_plan() ---
        rejected_plan = [
            PlanStep(step_id=f"step_{i+1:02d}", action="benchmark", tool_name=s.tool_name,
                      arguments=s.arguments, reasoning="benchmark")
            for i, s in enumerate(plan.steps)
        ]
        rejected_hash = canonicalize_plan(rejected_plan, target_column).plan_hash()

        evidence = {
            "violations": [v.model_dump(mode="json") for v in validation.violations],
            "rejected_steps": [{"step_index": i, "tool_name": s.tool_name, "arguments": s.arguments}
                                for i, s in enumerate(plan.steps)],
        }

        if rejected_hash in plan_history:
            failure = FailureInfo(
                category="DUPLICATE_PLAN",
                message="Executably identical to a plan already rejected earlier this trial.",
                evidence={**evidence, "duplicate_rejected_plan_hash": rejected_hash},
                node="plan",
                attempt=attempt_number,
                retryable=False,
                human_intervention_required=True,
            )
            attempt_record["failure_category"] = failure.category
            attempt_record["failure_details"] = failure.model_dump(mode="json")
            attempts.append(attempt_record)
            duplicate_plan_triggered = True
            break

        plan_history.append(rejected_hash)
        failure = FailureInfo(
            category="EVALUATION_ERROR",
            message=f"LLM-proposed plan failed deterministic validation ({len(validation.violations)} violation(s)).",
            evidence=evidence,
            node="plan",
            attempt=attempt_number,
            retryable=True,
            human_intervention_required=False,
        )
        attempt_record["failure_category"] = failure.category
        attempt_record["failure_details"] = failure.model_dump(mode="json")
        attempts.append(attempt_record)

        if attempt_number >= MAX_RETRIES:
            break
        attempt_number += 1
        empty_canonical = canonicalize_plan([], target_column)
        previous_plan_summary = diff_plans(empty_canonical, empty_canonical).model_dump(mode="json")
        context = b.LLMPlanningContext(
            objective=first_attempt_context.objective,
            dataset_context=first_attempt_context.dataset_context,
            allowed_operations=first_attempt_context.allowed_operations,
            tool_schemas=first_attempt_context.tool_schemas,
            failure_context=failure.model_dump(mode="json"),
            previous_plan_summary=previous_plan_summary,
        )

    residency_after = _ollama_ps()

    total_planning_latency = sum(a["request_total_ms"] for a in attempts) / 1000.0
    total_generation_latency = sum(a["ollama_stats"].get("eval_duration_s", 0) for a in attempts)
    total_prompt_processing_latency = sum(a["ollama_stats"].get("prompt_eval_duration_s", 0) for a in attempts)

    return {
        "model": model,
        "trial_id": trial_id,
        "cold_or_warm": cold_or_warm,
        "residency_before": residency_before,
        "residency_after": residency_after,
        "attempts": attempts,
        "first_attempt_valid": attempts[0]["validation_passed"] if attempts else None,
        "final_plan_valid": final_plan_valid,
        "replan_count": max(0, len(attempts) - 1),
        "duplicate_plan_triggered": duplicate_plan_triggered,
        "time_to_valid_plan": time_to_valid_plan,
        "total_planning_latency": total_planning_latency,
        "total_generation_latency": total_generation_latency,
        "total_prompt_processing_latency": total_prompt_processing_latency,
    }


def _run_block(model: str, first_attempt_context) -> list[dict]:
    print(f"\n{'='*70}\nBLOCK: {model}\n{'='*70}", flush=True)

    print("-- evicting all resident models --", flush=True)
    _evict_all()
    print(f"ollama ps after evict-all: {_ollama_ps()}", flush=True)

    print(f"-- forcing {model} cold --", flush=True)
    _ollama_stop(model)
    resident = _is_resident(model)
    print(f"ollama ps before Trial 1: resident={resident} -> {_ollama_ps()}", flush=True)
    if resident:
        print(f"WARNING: {model} still shows resident after ollama stop — cold state could not be verified.", flush=True)

    trials = []
    for i in range(1, N_TRIALS_PER_MODEL + 1):
        trial_id = f"{model}_trial{i}"
        print(f"\n-- Trial {i} ({'intended COLD' if i == 1 else 'intended WARM'}) --", flush=True)
        t0 = time.perf_counter()
        trial = _run_trial(model, trial_id, first_attempt_context, b.TARGET_COLUMN)
        elapsed = time.perf_counter() - t0
        print(
            f"   -> actual_state={trial['cold_or_warm']} attempts={len(trial['attempts'])} "
            f"first_attempt_valid={trial['first_attempt_valid']} final_valid={trial['final_plan_valid']} "
            f"time_to_valid={trial['time_to_valid_plan']} wall={elapsed:.1f}s",
            flush=True,
        )
        trials.append(trial)

    return trials


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Checklist: verify dataset ---
    assert b.DATASET_PATH.exists(), f"Dataset not found: {b.DATASET_PATH}"
    dataset_bytes = b.DATASET_PATH.read_bytes()
    store, df = b._load_dataset()
    assert df.shape == (891, 12), f"Expected 891x12, got {df.shape}"
    assert b.TARGET_COLUMN == "Survived"
    dataset_sha256 = _sha256(dataset_bytes)
    print(f"Dataset verified: {df.shape}, target={b.TARGET_COLUMN}, sha256={dataset_sha256[:16]}...")

    # --- Checklist: verify both models installed ---
    installed = b._installed_models()
    for m in MODELS:
        assert m in installed, f"{m} not installed: {installed}"
    print(f"Both models confirmed installed: {MODELS}")

    # --- Build the ONE shared context/prompt (identical for both models) ---
    sanitized_result = b.build_sanitized_llm_context(b.DATASET_ID, b.TARGET_COLUMN, store)
    assert sanitized_result.success
    budgeted_context, _ = b.apply_context_budget(sanitized_result.data)
    context = b.LLMPlanningContext(
        objective=f"Predict '{b.TARGET_COLUMN}' from the remaining columns (binary/multiclass classification).",
        dataset_context=budgeted_context.model_dump(mode="json"),
        allowed_operations=sorted(b.ALLOWED_TOOL_NAMES),
        tool_schemas=b.TOOL_ARGUMENT_SCHEMAS,
    )
    prompt_text = b.build_planning_prompt(context)
    assert context.tool_schemas, "tool_schemas must be populated (planner-contract fix must be active)"
    prompt_sha256 = _sha256_text(prompt_text)
    schema_sha256 = _sha256_text(json.dumps(b.TOOL_ARGUMENT_SCHEMAS, sort_keys=True))
    print(f"Shared prompt verified: {len(prompt_text)} chars, sha256={prompt_sha256[:16]}...")
    print(f"Tool-schema sha256={schema_sha256[:16]}...")

    ollama_version = _ollama_version()
    print(f"Ollama version: {ollama_version}")
    print(f"keep_alive={DEFAULT_KEEP_ALIVE!r} timeout={b.DEFAULT_TIMEOUT_SECONDS}s max_retries={MAX_RETRIES}")

    fingerprint = {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset_sha256": dataset_sha256,
        "prompt_sha256": prompt_sha256,
        "tool_schema_sha256": schema_sha256,
        "ollama_version": ollama_version,
        "keep_alive": DEFAULT_KEEP_ALIVE,
        "timeout_seconds": b.DEFAULT_TIMEOUT_SECONDS,
        "max_retries": MAX_RETRIES,
        "n_trials_per_model": N_TRIALS_PER_MODEL,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": MODELS,
        "execution_order": MODELS,
        "sampling_config_policy": (
            "Each model runs on its OWN embedded defaults; PIPER sends no `options` "
            "field, so this is production-realistic. These defaults DIFFER between "
            "candidates (see model_default_parameters) — a documented confound: any "
            "observed difference is attributable to model+shipped-sampling-config "
            "jointly, not to model capability in isolation."
        ),
        "model_default_parameters": {m: _ollama_show_parameters(m) for m in MODELS},
    }
    print(json.dumps(fingerprint, indent=2))

    all_results = {"fingerprint": fingerprint, "blocks": {}}

    for model in MODELS:
        trials = _run_block(model, context)
        all_results["blocks"][model] = trials

        # Verify fingerprint consistency wasn't silently broken mid-run
        recheck_prompt = b.build_planning_prompt(context)
        assert _sha256_text(recheck_prompt) == prompt_sha256, "STOP: prompt fingerprint changed mid-benchmark"

        model_path = RESULTS_DIR / f"{model.replace(':', '_')}.json"
        model_path.write_text(json.dumps({"model": model, "trials": trials}, indent=2, default=str))
        print(f"Wrote {model_path}")

    raw_path = RESULTS_DIR / "raw_results.json"
    raw_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nWrote {raw_path}")


if __name__ == "__main__":
    main()
