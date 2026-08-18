"""
MEASUREMENT-ONLY HARNESS — isolated from production code.

Measures whether a real model can RECOVER from a deterministic Plan
Adequacy failure when handed structured REPLAN evidence.

Mirrors plan_node_v2's attempt pipeline EXACTLY, in the same order,
reusing production code at every stage (never a reimplementation):

    LLM call
      v  OllamaProvider._extract_content / _strip_markdown_fences
    response extraction
      v  ProposedPlan.model_validate
    plan parsing
      v  validate_proposed_plan()
    structural validation
      v  canonicalize_plan() / plan_hash() vs plan_history
    duplicate-plan detection
      v  evaluate_plan_adequacy()
    Plan Adequacy
      v
    PASS / REPLAN / TERMINATE

REPLAN uses the real build_replan_prompt() fed by a real FailureInfo
carrying the real adequacy findings — no simplified retry prompt, no
manual explanation, no patching of model output.

Bounded by max_retries=2 (AgentState's real default) => at most 3
attempts per trial. No new budget is introduced.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import benchmark_planning_models as b
from app.agent.plan_adequacy import classify_plan_steps, evaluate_plan_adequacy
from app.agent.plan_canonical import canonicalize_plan
from app.agent.plan_diff import diff_plans
from app.agent.state import PlanStep
from app.llm.ollama_provider import DEFAULT_KEEP_ALIVE, DEFAULT_TIMEOUT_SECONDS
from app.schemas.failure import FailureInfo

MODELS = ["qwen3:4b", "qwen3.5:4b"]
N_TRIALS = 3
MAX_RETRIES = 2
EXPERIMENT = "adequacy_recovery_v2"
RESULTS_DIR = Path(__file__).parent / "benchmark_results" / "adequacy_recovery_v2"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ollama_ps() -> list[dict]:
    p = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30)
    return [{"name": l.split()[0], "raw": l.strip()} for l in p.stdout.strip().splitlines()[1:] if l.strip()]


def _ollama_show_parameters(model: str) -> dict:
    """
    Captures a model's OWN shipped generation defaults via `ollama show`.
    PIPER sends no `options` field, so each model runs on its own defaults —
    and those DIFFER between candidates (qwen3:4b temperature=0.6 vs
    qwen3.5:4b temperature=1.0). Recorded in the fingerprint so this known
    confound stays visible in the results rather than living only in prose.
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
                if key in params:
                    existing = params[key]
                    params[key] = existing + [value] if isinstance(existing, list) else [existing, value]
                else:
                    params[key] = value
    return params


def _evict_all():
    for e in _ollama_ps():
        subprocess.run(["ollama", "stop", e["name"]], capture_output=True, text=True, timeout=30)
    time.sleep(1)


def _is_resident(model: str) -> bool:
    return any(e["name"] == model for e in _ollama_ps())


def _call(model: str, prompt: str) -> dict:
    """Identical request shape to OllamaProvider.generate_plan()."""
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "format": b.PLAN_JSON_SCHEMA, "keep_alive": DEFAULT_KEEP_ALIVE,
    }
    req = urllib.request.Request(
        url=f"{b.DEFAULT_OLLAMA_HOST.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    error, body = None, {}
    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SECONDS) as r:
            body = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 — must record, never crash the run
        error = f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    return {"wall_seconds": wall, "over_timeout": wall > DEFAULT_TIMEOUT_SECONDS,
            "transport_error": error, "body": body}


def _stats(body: dict) -> dict:
    if not body:
        return {}
    ev_s = body.get("eval_duration", 0) / 1e9
    return {
        "total_duration_s": body.get("total_duration", 0) / 1e9,
        "load_duration_s": body.get("load_duration", 0) / 1e9,
        "prompt_eval_duration_s": body.get("prompt_eval_duration", 0) / 1e9,
        "eval_duration_s": ev_s,
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
        "tokens_per_sec": (body.get("eval_count") / ev_s) if body.get("eval_count") and ev_s else None,
    }


def _run_trial(model: str, trial_id: str, base_ctx, sanitized_ctx, target: str) -> dict:
    attempts: list[dict] = []
    plan_history: list[str] = []
    context = base_ctx
    cumulative = 0.0
    time_to_executable = None
    outcome = None
    terminal_category = None

    cold_or_warm = "warm" if _is_resident(model) else "cold"
    residency_before = _ollama_ps()

    for attempt_no in range(MAX_RETRIES + 1):
        is_replan = attempt_no > 0
        prompt = b.build_replan_prompt(context) if is_replan else b.build_planning_prompt(context)

        call = _call(model, prompt)
        cumulative += call["wall_seconds"]

        rec: dict = {
            "attempt_number": attempt_no,
            "is_replan": is_replan,
            "cold_or_warm": cold_or_warm if attempt_no == 0 else "warm(mid-trial)",
            "prompt_chars": len(prompt),
            "wall_seconds": call["wall_seconds"],
            "wall_clock_over_timeout": call["over_timeout"],
            "transport_error": call["transport_error"],
            "ollama_stats": _stats(call["body"]),
            "stage_reached": None,
            "structural_valid": None,
            "structural_violations": [],
            "rejected_steps": [],
            "duplicate_detected": False,
            "adequacy_status": None,
            "adequacy_material_failure": None,
            "adequacy_findings": [],
            "valid_steps": [],
            "implicated_steps": [],
            "proposed_steps": [],
            "failure_category": None,
        }

        # --- response extraction + parsing (production path) ---------
        content = b._extract_content(call["body"]) if call["body"] else None
        cleaned = b._strip_markdown_fences(content) if content else None
        plan_obj, parse_error = None, None
        if cleaned is not None:
            try:
                plan_obj = b.ProposedPlan.model_validate(json.loads(cleaned))
            except Exception as e:  # noqa: BLE001
                parse_error = f"{type(e).__name__}: {e}"
        rec["parse_error"] = parse_error

        if plan_obj is None:
            rec["stage_reached"] = "parsing"
            rec["failure_category"] = "EVALUATION_ERROR"
            failure = FailureInfo(
                category="EVALUATION_ERROR",
                message=f"LLM planner failed: {parse_error or call['transport_error']}",
                evidence={"provider_error_code": "transport_or_parse",
                          "raw_error": parse_error or call["transport_error"]},
                node="plan", attempt=attempt_no, retryable=True, human_intervention_required=False,
            )
            attempts.append(rec)
            if attempt_no >= MAX_RETRIES:
                outcome, terminal_category = "BUDGET_EXHAUSTION", "EVALUATION_ERROR"
                break
            context = _replan_context(base_ctx, failure, target)
            continue

        rec["proposed_steps"] = [{"tool_name": s.tool_name, "arguments": s.arguments} for s in plan_obj.steps]
        rec["step_count"] = len(plan_obj.steps)

        # --- structural validation (production) ----------------------
        validation = b.validate_proposed_plan(plan_obj.steps, target)
        rec["structural_valid"] = validation.valid
        rec["structural_violations"] = [v.model_dump(mode="json") for v in validation.violations]

        # Canonical identity (same machinery for valid AND rejected plans)
        as_steps = [
            PlanStep(step_id=f"step_{i+1:02d}", action="a", tool_name=s.tool_name,
                     arguments=s.arguments, reasoning="r")
            for i, s in enumerate(plan_obj.steps)
        ]
        plan_hash = canonicalize_plan(as_steps, target).plan_hash()
        rec["plan_hash"] = plan_hash

        if not validation.valid:
            rec["stage_reached"] = "structural_validation"
            rec["rejected_steps"] = [
                {"step_index": i, "tool_name": s.tool_name, "arguments": s.arguments}
                for i, s in enumerate(plan_obj.steps)
            ]
            evidence = {"violations": rec["structural_violations"], "rejected_steps": rec["rejected_steps"]}

            if plan_hash in plan_history:
                rec["duplicate_detected"] = True
                rec["failure_category"] = "DUPLICATE_PLAN"
                attempts.append(rec)
                outcome, terminal_category = "DUPLICATE_PLAN_WALL", "DUPLICATE_PLAN"
                break

            plan_history.append(plan_hash)
            rec["failure_category"] = "EVALUATION_ERROR"
            failure = FailureInfo(
                category="EVALUATION_ERROR",
                message=f"LLM-proposed plan failed deterministic tool/argument validation ({len(validation.violations)} violation(s)).",
                evidence=evidence, node="plan", attempt=attempt_no,
                retryable=True, human_intervention_required=False,
            )
            attempts.append(rec)
            if attempt_no >= MAX_RETRIES:
                outcome, terminal_category = "BUDGET_EXHAUSTION", "EVALUATION_ERROR"
                break
            context = _replan_context(base_ctx, failure, target)
            continue

        # --- duplicate-plan detection (production ordering) ----------
        if plan_hash in plan_history:
            rec["stage_reached"] = "duplicate_detection"
            rec["duplicate_detected"] = True
            rec["failure_category"] = "DUPLICATE_PLAN"
            attempts.append(rec)
            outcome, terminal_category = "DUPLICATE_PLAN_WALL", "DUPLICATE_PLAN"
            break

        # --- Plan Adequacy (production) -------------------------------
        adequacy = evaluate_plan_adequacy(sanitized_ctx, plan_obj.steps, target)
        rec["stage_reached"] = "adequacy"
        rec["adequacy_status"] = adequacy.status
        rec["adequacy_material_failure"] = adequacy.material_failure
        rec["adequacy_findings"] = [f.model_dump(mode="json") for f in adequacy.findings]

        if adequacy.material_failure:
            plan_history.append(plan_hash)
            rec["failure_category"] = "PLAN_ADEQUACY"
            # PRODUCTION PARITY: mirrors plan_node_v2's adequacy-failure
            # evidence construction exactly (real_nodes.py), reusing the real
            # classify_plan_steps() rather than reimplementing it.
            step_classification = classify_plan_steps(adequacy.findings, plan_obj.steps)
            rec["valid_steps"] = step_classification["valid_steps"]
            rec["implicated_steps"] = step_classification["implicated_steps"]
            failure = FailureInfo(
                category="PLAN_ADEQUACY",
                message=adequacy.summary,
                evidence={
                    "adequacy_status": adequacy.status,
                    "findings": [f.model_dump(mode="json") for f in adequacy.material_findings],
                    "advisory_findings": [
                        f.model_dump(mode="json") for f in adequacy.findings
                        if f.severity == "advisory" and f.status == "NOT_ADDRESSED"
                    ],
                    "proposed_steps": [
                        {"tool_name": s.tool_name, "arguments": s.arguments}
                        for s in plan_obj.steps
                    ],
                    "valid_steps": step_classification["valid_steps"],
                    "implicated_steps": step_classification["implicated_steps"],
                },
                node="plan", attempt=attempt_no, retryable=True, human_intervention_required=False,
            )
            attempts.append(rec)
            if attempt_no >= MAX_RETRIES:
                outcome, terminal_category = "BUDGET_EXHAUSTION", "PLAN_ADEQUACY"
                break
            context = _replan_context(base_ctx, failure, target)
            continue

        # --- PASS ------------------------------------------------------
        rec["stage_reached"] = "pass"
        plan_history.append(plan_hash)
        attempts.append(rec)
        time_to_executable = cumulative
        outcome = "SUCCESSFUL_PATCH" if attempt_no > 0 else "FIRST_ATTEMPT_PASS"
        break

    if outcome is None:
        outcome, terminal_category = "BUDGET_EXHAUSTION", attempts[-1].get("failure_category")

    # NEW_INVALIDATION: adequacy failed first, then a LATER attempt broke
    # structural validity, and the trial never recovered.
    if outcome not in ("SUCCESSFUL_PATCH", "FIRST_ATTEMPT_PASS"):
        had_adequacy_fail = any(a["failure_category"] == "PLAN_ADEQUACY" for a in attempts)
        later_structural = any(
            a["structural_valid"] is False and a["attempt_number"] > 0 for a in attempts
        )
        if had_adequacy_fail and later_structural:
            outcome = "NEW_INVALIDATION"

    first = attempts[0]
    return {
        "trial_id": trial_id,
        "model": model,
        "cold_or_warm": cold_or_warm,
        "residency_before": residency_before,
        "residency_after": _ollama_ps(),
        "attempts": attempts,
        "llm_calls": len(attempts),
        "replan_calls": max(0, len(attempts) - 1),
        "adequacy_failures": sum(1 for a in attempts if a["failure_category"] == "PLAN_ADEQUACY"),
        "structural_failures": sum(1 for a in attempts if a["structural_valid"] is False),
        "duplicate_terminations": sum(1 for a in attempts if a["duplicate_detected"]),
        "first_attempt_structurally_valid": first["structural_valid"],
        "first_attempt_adequate": bool(first["structural_valid"]) and first.get("adequacy_material_failure") is False,
        "final_executable": time_to_executable is not None,
        "time_to_executable_plan": time_to_executable,
        "total_planning_latency": cumulative,
        "total_generation_latency": sum(a["ollama_stats"].get("eval_duration_s", 0) for a in attempts),
        "total_prompt_processing_latency": sum(a["ollama_stats"].get("prompt_eval_duration_s", 0) for a in attempts),
        "outcome": outcome,
        "terminal_category": terminal_category,
    }


def _replan_context(base_ctx, failure: FailureInfo, target: str):
    """
    Mirrors plan_node_v2's REPLAN context construction EXACTLY, including
    the fact that state.plan is still [] after a PLAN-node failure (so the
    diff is empty-vs-empty). The concrete rejected/proposed steps travel in
    failure.evidence, exactly as production sends them.
    """
    empty = canonicalize_plan([], target)
    ctx = b.LLMPlanningContext(
        objective=base_ctx.objective,
        dataset_context=base_ctx.dataset_context,
        allowed_operations=base_ctx.allowed_operations,
        tool_schemas=base_ctx.tool_schemas,
        failure_context=failure.model_dump(mode="json"),
        previous_plan_summary=diff_plans(empty, empty).model_dump(mode="json"),
    )
    return ctx


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = b.DATASET_PATH.read_bytes()
    store, df = b._load_dataset()
    assert df.shape == (891, 12)
    dataset_sha = hashlib.sha256(raw).hexdigest()

    installed = b._installed_models()
    for m in MODELS:
        assert m in installed, f"{m} not installed"

    sanitized = b.build_sanitized_llm_context(b.DATASET_ID, b.TARGET_COLUMN, store)
    assert sanitized.success
    budgeted, _ = b.apply_context_budget(sanitized.data)

    base_ctx = b.LLMPlanningContext(
        objective=f"Predict '{b.TARGET_COLUMN}' from the remaining columns (binary/multiclass classification).",
        dataset_context=budgeted.model_dump(mode="json"),
        allowed_operations=sorted(b.ALLOWED_TOOL_NAMES),
        tool_schemas=b.TOOL_ARGUMENT_SCHEMAS,
    )
    prompt_sha = _sha(b.build_planning_prompt(base_ctx))
    schema_sha = _sha(json.dumps(b.TOOL_ARGUMENT_SCHEMAS, sort_keys=True))

    fingerprint = {
        "experiment": EXPERIMENT,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_sha,
        "dataset_shape": list(df.shape),
        "prompt_sha256": prompt_sha,
        "tool_schema_sha256": schema_sha,
        "keep_alive": DEFAULT_KEEP_ALIVE,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "max_retries": MAX_RETRIES,
        "models": MODELS,
        "execution_order": MODELS,
        "n_trials_per_model": N_TRIALS,
        "note": "Each model runs on its OWN shipped sampling defaults; PIPER sends no options field.",
        "model_default_parameters": {m: _ollama_show_parameters(m) for m in MODELS},
    }
    print(json.dumps(fingerprint, indent=2), flush=True)

    results = {"fingerprint": fingerprint, "blocks": {}}

    for model in MODELS:
        print(f"\n{'='*72}\nBLOCK: {model}\n{'='*72}", flush=True)
        _evict_all()
        print(f"evicted; ollama ps = {_ollama_ps()}", flush=True)

        trials = []
        for i in range(1, N_TRIALS + 1):
            tid = f"{model}_trial{i}"
            print(f"\n-- {tid} ({'COLD' if i == 1 else 'warm'}) --", flush=True)
            t = _run_trial(model, tid, base_ctx, budgeted, b.TARGET_COLUMN)
            print(
                f"   state={t['cold_or_warm']} calls={t['llm_calls']} "
                f"first_adequate={t['first_attempt_adequate']} final={t['final_executable']} "
                f"outcome={t['outcome']} ttx={t['time_to_executable_plan']}",
                flush=True,
            )
            for a in t["attempts"]:
                print(f"      attempt{a['attempt_number']}: stage={a['stage_reached']} "
                      f"struct_valid={a['structural_valid']} adequacy={a['adequacy_status']} "
                      f"cat={a['failure_category']} steps={a.get('step_count')}", flush=True)
            trials.append(t)

        results["blocks"][model] = trials
        (RESULTS_DIR / f"{model.replace(':','_')}.json").write_text(
            json.dumps({"model": model, "trials": trials}, indent=2, default=str)
        )
        assert _sha(b.build_planning_prompt(base_ctx)) == prompt_sha, "STOP: prompt changed mid-benchmark"

    (RESULTS_DIR / "raw_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {RESULTS_DIR / 'raw_results.json'}")


if __name__ == "__main__":
    main()
