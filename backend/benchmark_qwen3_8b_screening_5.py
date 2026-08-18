"""
PIPER — QWEN3:8B 5-RUN SCREENING BENCHMARK RUNNER (Measurement Only)
Authoritative sequential execution of 5 Titanic screening trials on frozen V1 with qwen3:8b.
"""

import os
import sys
import time
import json
import statistics
import pandas as pd
from types import SimpleNamespace

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore, InMemoryRunStore
from app.agent.graph import build_graph
from app.agent.state import AgentState, PlanStep
from app.agent.plan_validation import validate_proposed_plan, ALLOWED_TOOL_NAMES, TOOL_ARGUMENT_SCHEMAS
from app.agent.plan_adequacy import evaluate_plan_adequacy
from app.agent.plan_canonical import canonicalize_plan
from app.agent.tracing import run_with_tracing
from app.agent.run_summary import build_run_summary
from app.agent.timeline import build_execution_timeline
from app.learning.explain import build_run_explanation
from app.agent.tools.profiling import profile_dataset
from app.agent.tools import build_sanitized_llm_context
from app.agent.tools.context_budget import apply_context_budget
from app.llm.ollama_provider import OllamaProvider


class InstrumentedOllamaProvider:
    """
    Measurement wrapper around OllamaProvider.
    Captures raw proposed plans and per-call latency without modifying
    production provider behavior or state logic.
    """
    def __init__(self, inner_provider: OllamaProvider):
        self.inner = inner_provider
        self.call_logs = []

    def generate_plan(self, context):
        t0 = time.time()
        result = self.inner.generate_plan(context)
        latency = time.time() - t0

        raw_steps = None
        if result.success and result.plan and hasattr(result.plan, "steps"):
            raw_steps = [
                {
                    "action": getattr(s, "action", ""),
                    "tool_name": getattr(s, "tool_name", ""),
                    "arguments": getattr(s, "arguments", {}),
                    "reasoning": getattr(s, "reasoning", "")
                }
                for s in result.plan.steps
            ]

        valid_steps_received = []
        if context.failure_context and isinstance(context.failure_context, dict):
            # Check for valid_steps in failure_context or details
            if "valid_steps" in context.failure_context:
                valid_steps_received = context.failure_context["valid_steps"]
            elif "details" in context.failure_context and isinstance(context.failure_context["details"], dict):
                valid_steps_received = context.failure_context["details"].get("valid_steps", [])

        self.call_logs.append({
            "timestamp": time.time(),
            "latency_seconds": latency,
            "success": result.success,
            "error": result.error.message if result.error else None,
            "error_code": result.error.code if result.error else None,
            "raw_steps": raw_steps,
            "is_replan": context.failure_context is not None,
            "valid_steps_received": valid_steps_received
        })
        return result


def analyze_embarked_handling(plan_steps) -> str:
    """
    Classify how Embarked was handled in the plan:
    A: Imputed before encoding
    B: Excluded from effective feature set (or dropped)
    C: Unhandled (in effective feature set but not imputed/dropped)
    """
    if not plan_steps:
        return "NO_PLAN"

    imputed = False
    dropped = False
    encoded = False

    for s in plan_steps:
        tool = s.get("tool_name") if isinstance(s, dict) else getattr(s, "tool_name", "")
        args = s.get("arguments", {}) if isinstance(s, dict) else getattr(s, "arguments", {})

        if tool == "impute_missing_values" and args.get("column") == "Embarked":
            imputed = True
        elif tool == "drop_column" and args.get("column") == "Embarked":
            dropped = True
        elif tool == "encode_categorical_features" and "Embarked" in args.get("columns", []):
            encoded = True

    if imputed:
        return "A (imputed before encoding)" if encoded else "A (imputed)"
    elif dropped:
        return "B (dropped / excluded)"
    elif not encoded:
        return "B (excluded from effective features)"
    else:
        return "C (unhandled in effective features)"


def run_single_trial(trial_idx: int, is_cold: bool, dataset_path: str, target_column: str = "Survived") -> dict:
    print(f"\n========================================================")
    print(f"  STARTING TRIAL {trial_idx}/5 ({'COLD' if is_cold else 'WARM'}) [qwen3:8b]")
    print(f"========================================================")

    # 1. Fresh stores
    dataset_store = InMemoryDatasetStore()
    split_store = InMemorySplitStore()
    model_store = InMemoryModelStore()
    run_store = InMemoryRunStore()

    # 2. Ingest
    df = pd.read_csv(dataset_path)
    dataset_id = f"titanic_qwen8b_trial_{trial_idx}"
    dataset_store.save(dataset_id, df)

    # Pre-compute budgeted context for offline verification of attempt plans
    profile = profile_dataset(dataset_id, dataset_store)
    sanitized_res = build_sanitized_llm_context(dataset_id, target_column, dataset_store)
    budgeted_ctx, _ = apply_context_budget(sanitized_res.data)

    # 3. Provider with instrumentation (qwen3:8b)
    base_provider = OllamaProvider(
        model="qwen3:8b",
        host="http://localhost:11434",
        timeout_seconds=600.0,
        keep_alive="10m"
    )
    instrumented = InstrumentedOllamaProvider(base_provider)

    # 4. Build LangGraph
    graph = build_graph(
        dataset_store=dataset_store,
        split_store=split_store,
        model_store=model_store,
        llm_provider=instrumented
    )

    run_id = f"qwen8b_scr_run_{trial_idx}"
    initial_state = AgentState(
        run_id=run_id,
        dataset_id=dataset_id,
        target_column=target_column,
        task_type="binary_classification",
        max_retries=2
    )

    # 5. Run with tracing
    t0_wall = time.time()
    final_state = run_with_tracing(graph, initial_state, run_store)
    t1_wall = time.time()
    total_wall_time = t1_wall - t0_wall

    status = final_state.get("status")
    retry_count = final_state.get("retry_count", 0)
    failure = final_state.get("failure")
    plan = final_state.get("plan", [])
    model_results = final_state.get("model_results", [])
    comparison = final_state.get("comparison")

    # 6. Analyze attempts
    attempts_detail = []
    first_attempt_structural_validity = False
    first_attempt_adequacy = None
    final_structural_validity = False
    final_adequacy = False
    total_schema_violations = 0
    tool_argument_violations = []
    adequacy_violations = []

    duplicate_plan = (failure.category == "DUPLICATE_PLAN") if failure else False
    parse_failure = False
    provider_failure = False
    timeout_occurred = False
    budget_exhaustion = (failure.category == "RETRY_BUDGET_EXCEEDED" or failure.category == "EXECUTION_BUDGET_EXCEEDED") if failure else False

    for idx, log in enumerate(instrumented.call_logs):
        raw_steps = log.get("raw_steps")
        if not log["success"]:
            if "timeout" in (log.get("error") or "").lower() or log.get("error_code") == "timeout":
                timeout_occurred = True
            elif "parse" in (log.get("error") or "").lower() or log.get("error_code") in ("malformed_response", "invalid_plan_schema"):
                parse_failure = True
            else:
                provider_failure = True

            attempts_detail.append({
                "attempt_index": idx,
                "latency_seconds": log["latency_seconds"],
                "success": False,
                "error": log["error"],
                "raw_steps": None,
                "structurally_valid": False,
                "violations": [log["error"]],
                "adequate": None,
                "adequacy_findings": [],
                "valid_steps_received": log.get("valid_steps_received", [])
            })
            continue

        # Evaluate structural validation
        steps_objs = [
            SimpleNamespace(tool_name=s["tool_name"], arguments=s["arguments"])
            for s in raw_steps
        ]
        val_res = validate_proposed_plan(steps_objs, target_column)
        struct_valid = val_res.valid
        violations_str = [f"{v.field}: {v.reason}" for v in val_res.violations]

        if not struct_valid:
            total_schema_violations += len(val_res.violations)
            tool_argument_violations.extend(violations_str)

        # Evaluate adequacy
        adequacy_res = evaluate_plan_adequacy(budgeted_ctx, steps_objs, target_column)
        is_adequate = (adequacy_res.status == "PASS")
        adeq_findings = [f"{f.condition} ({f.severity}): {', '.join(f.columns)}" for f in adequacy_res.material_findings]
        if not is_adequate:
            adequacy_violations.extend(adeq_findings)

        if idx == 0:
            first_attempt_structural_validity = struct_valid
            first_attempt_adequacy = is_adequate if struct_valid else False

        if idx == len(instrumented.call_logs) - 1:
            final_structural_validity = struct_valid
            final_adequacy = is_adequate

        attempts_detail.append({
            "attempt_index": idx,
            "latency_seconds": log["latency_seconds"],
            "success": True,
            "error": None,
            "raw_steps": raw_steps,
            "structurally_valid": struct_valid,
            "violations": violations_str,
            "adequate": is_adequate,
            "adequacy_findings": adeq_findings,
            "valid_steps_received": log.get("valid_steps_received", [])
        })

    # 7. Check final executable plan
    final_executable_plan = None
    if plan and len(plan) > 0:
        final_executable_plan = [
            {
                "tool_name": step.tool_name,
                "arguments": step.arguments,
                "action": step.action,
                "reasoning": step.reasoning
            }
            for step in plan
        ]

    # 8. Check end-to-end criteria
    training_completed = len(model_results) >= 2
    model_comparison_completed = comparison is not None and comparison.recommended_model_id is not None
    
    # Try generating final report artifacts
    final_report_generated = False
    rec = run_store.get(run_id)
    if rec and rec.final_state and status == "completed":
        try:
            _summary = build_run_summary(run_id, rec.final_state)
            _timeline = build_execution_timeline(run_id, rec.events)
            _explanation = build_run_explanation(run_id, rec.final_state)
            final_report_generated = True
        except Exception as e:
            print(f"Report generation error: {e}")
            final_report_generated = False

    completed_successfully = (
        status == "completed" and
        final_structural_validity and
        final_adequacy and
        training_completed and
        model_comparison_completed and
        final_report_generated
    )

    # 9. Classify failure category
    failure_category = None
    failure_reason = None
    if not completed_successfully:
        if duplicate_plan:
            failure_category = "DUPLICATE_PLAN"
            failure_reason = failure.message if failure else "Duplicate plan detected"
        elif not final_structural_validity or total_schema_violations > 0:
            failure_category = "SCHEMA_FAILURE"
            failure_reason = "; ".join(tool_argument_violations) if tool_argument_violations else "Invalid tool argument structure"
        elif not final_adequacy:
            failure_category = "ADEQUACY_FAILURE"
            failure_reason = "; ".join(adequacy_violations) if adequacy_violations else "Material adequacy conditions unaddressed"
        elif parse_failure:
            failure_category = "PARSE_FAILURE"
            failure_reason = "Failed to parse JSON response from LLM"
        elif timeout_occurred:
            failure_category = "TIMEOUT"
            failure_reason = "LLM inference timeout"
        elif provider_failure:
            failure_category = "PROVIDER_FAILURE"
            failure_reason = "Ollama provider error"
        elif not training_completed or not model_comparison_completed:
            failure_category = "EXECUTION_FAILURE"
            failure_reason = "Pipeline execution failed during training/evaluation"
        else:
            failure_category = "OTHER"
            failure_reason = failure.message if failure else "Unknown failure"

    # 10. Qualitative plan inspections
    qualitative = {
        "plan_steps_count": len(plan),
        "age_addressed": any(s.tool_name in ("impute_missing_values", "drop_column") and s.arguments.get("column") == "Age" for s in plan),
        "embarked_addressed": any(
            (s.tool_name in ("impute_missing_values", "drop_column") and s.arguments.get("column") == "Embarked") or
            (s.tool_name == "encode_categorical_features" and "Embarked" in s.arguments.get("columns", []))
            for s in plan
        ),
        "target_survived_untouched": not any(
            s.arguments.get("column") == "Survived" or "Survived" in s.arguments.get("columns", [])
            for s in plan
        ),
        "identifiers_handled": any(
            s.tool_name == "drop_column" and s.arguments.get("column") in ("PassengerId", "Name", "Ticket", "Cabin")
            for s in plan
        ),
        "categorical_encoding_valid": any(
            s.tool_name == "encode_categorical_features" and isinstance(s.arguments.get("columns"), list)
            for s in plan
        ),
        "numeric_scaling_valid": any(
            s.tool_name == "scale_features" and isinstance(s.arguments.get("columns"), list)
            for s in plan
        ),
        "embarked_strategy": analyze_embarked_handling(plan) if plan else (
            analyze_embarked_handling(attempts_detail[0]["raw_steps"]) if attempts_detail and attempts_detail[0]["raw_steps"] else "NO_PLAN"
        )
    }

    generation_time = sum(a["latency_seconds"] for a in attempts_detail)

    trial_result = {
        "trial_id": trial_idx,
        "cold_or_warm": "cold" if is_cold else "warm",
        "model": "qwen3:8b",
        "temperature": 0.0,
        "keep_alive": "10m",
        "timeout": 600.0,
        "first_attempt_structural_validity": first_attempt_structural_validity,
        "first_attempt_adequacy": first_attempt_adequacy,
        "replan_count": retry_count,
        "final_structural_validity": final_structural_validity,
        "final_adequacy": final_adequacy,
        "final_executable_plan": final_executable_plan,
        "end_to_end_success": completed_successfully,
        "training_completed": training_completed,
        "model_comparison_completed": model_comparison_completed,
        "final_report_generated": final_report_generated,
        "duplicate_plan": duplicate_plan,
        "schema_failure": failure_category == "SCHEMA_FAILURE",
        "parse_failure": parse_failure,
        "provider_failure": provider_failure,
        "timeout_failure": timeout_occurred,
        "adequacy_failure": failure_category == "ADEQUACY_FAILURE",
        "execution_failure": failure_category == "EXECUTION_FAILURE",
        "attempt_count": len(instrumented.call_logs),
        "total_wall_time": total_wall_time,
        "generation_time": generation_time,
        "failure_category": failure_category,
        "failure_reason": failure_reason,
        "winner_model": comparison.recommended_model_id if comparison else None,
        "winner_algorithm": comparison.justification if comparison else None,
        "qualitative_checks": qualitative,
        "attempts": attempts_detail
    }

    print(f"Trial {trial_idx} Finished in {total_wall_time:.2f}s (Gen: {generation_time:.2f}s):")
    print(f"  Status: {status} | Success: {completed_successfully} | Attempts: {len(attempts_detail)} | Retries: {retry_count}")
    print(f"  Embarked Handling: {qualitative['embarked_strategy']}")
    if failure_category:
        print(f"  Failure: {failure_category} -> {failure_reason}")
    if completed_successfully and comparison:
        print(f"  Winner: {comparison.recommended_model_id} | {comparison.justification}")
    return trial_result


def main():
    print("=================================================================")
    print("  PIPER — QWEN3:8B 5-RUN SCREENING BENCHMARK (TITANIC / qwen3:8b)")
    print("=================================================================")
    
    dataset_path = "benchmark_data/train.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "../benchmark_data/train.csv"
    assert os.path.exists(dataset_path), f"Dataset not found at {dataset_path}"

    results_dir = os.path.join("benchmark_results", "qwen3_8b_screening")
    os.makedirs(results_dir, exist_ok=True)
    raw_results_file = os.path.join(results_dir, "qwen3_8b_5run.json")
    summary_file = os.path.join(results_dir, "qwen3_8b_5run_summary.json")

    trials_data = []
    if os.path.exists(raw_results_file):
        try:
            with open(raw_results_file, "r") as f:
                trials_data = json.load(f)
            print(f"Resuming benchmark: loaded {len(trials_data)} existing trials from {raw_results_file}")
        except Exception as e:
            print(f"Could not load existing results ({e}), starting fresh.")
            trials_data = []

    start_trial = len(trials_data) + 1
    for trial_idx in range(start_trial, 6):
        is_cold = (trial_idx == 1)
        trial_res = run_single_trial(trial_idx, is_cold, dataset_path, target_column="Survived")
        trials_data.append(trial_res)

        # Save intermediate results
        with open(raw_results_file, "w") as f:
            json.dump(trials_data, f, indent=2)

    # Compute aggregate metrics
    n = len(trials_data)
    successful_trials = sum(1 for t in trials_data if t["end_to_end_success"])
    first_valid = sum(1 for t in trials_data if t["first_attempt_structural_validity"])
    first_adequate = sum(1 for t in trials_data if t["first_attempt_adequacy"])
    final_executable = sum(1 for t in trials_data if t["final_executable_plan"] is not None and len(t["final_executable_plan"]) > 0)
    replan_trials = sum(1 for t in trials_data if t["replan_count"] > 0)
    replan_successful = sum(1 for t in trials_data if t["replan_count"] > 0 and t["end_to_end_success"])
    replan_recovery_rate = (replan_successful / replan_trials) if replan_trials > 0 else 1.0

    duplicate_plan_trials = sum(1 for t in trials_data if t["duplicate_plan"])
    schema_failure_trials = sum(1 for t in trials_data if t["schema_failure"])
    adequacy_failure_trials = sum(1 for t in trials_data if t["adequacy_failure"])
    parse_failure_trials = sum(1 for t in trials_data if t["parse_failure"])
    timeout_failure_trials = sum(1 for t in trials_data if t["timeout_failure"])
    execution_failure_trials = sum(1 for t in trials_data if t["execution_failure"])
    total_attempts = sum(t["attempt_count"] for t in trials_data)

    planning_latencies = [t["generation_time"] for t in trials_data]
    wall_latencies = [t["total_wall_time"] for t in trials_data]

    mean_planning_latency = statistics.mean(planning_latencies) if planning_latencies else 0.0
    median_planning_latency = statistics.median(planning_latencies) if planning_latencies else 0.0
    min_planning_latency = min(planning_latencies) if planning_latencies else 0.0
    max_planning_latency = max(planning_latencies) if planning_latencies else 0.0

    # Decision rule
    classification = ""
    if successful_trials >= 4:
        classification = "PROMISING"
        recommendation = "Qwen3:8B merits a controlled 10-run confirmation benchmark."
    elif successful_trials == 3:
        classification = "BORDERLINE"
        recommendation = "Do not proceed directly to n=10; analyze failure modes and latency first."
    else:
        classification = "NOT PROMISING"
        recommendation = "Qwen3:8B does not sufficiently improve PIPER reliability under the current configuration."

    summary = {
        "n_trials": n,
        "successful_trials": successful_trials,
        "overall_reliability": successful_trials / n,
        "first_attempt_structural_validity_rate": first_valid / n,
        "first_attempt_adequacy_rate": first_adequate / n,
        "final_executable_plan_rate": final_executable / n,
        "replan_rate": replan_trials / n,
        "replan_recovery_rate": replan_recovery_rate,
        "duplicate_plan_rate": duplicate_plan_trials / n,
        "schema_failure_rate": schema_failure_trials / n,
        "adequacy_failure_rate": adequacy_failure_trials / n,
        "parse_failure_rate": parse_failure_trials / n,
        "timeout_rate": timeout_failure_trials / n,
        "execution_failure_rate": execution_failure_trials / n,
        "end_to_end_completion_rate": successful_trials / n,
        "average_attempts_per_trial": total_attempts / n,
        "planning_latency": {
            "mean_seconds": mean_planning_latency,
            "median_seconds": median_planning_latency,
            "min_seconds": min_planning_latency,
            "max_seconds": max_planning_latency
        },
        "wall_time": {
            "mean_seconds": statistics.mean(wall_latencies),
            "median_seconds": statistics.median(wall_latencies),
            "min_seconds": min(wall_latencies),
            "max_seconds": max(wall_latencies)
        },
        "screening_decision": classification,
        "recommendation": recommendation
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=================================================================")
    print("  QWEN3:8B 5-RUN SCREENING BENCHMARK COMPLETE")
    print("=================================================================")
    print(f"Successful: {successful_trials}/{n} ({summary['overall_reliability']*100:.1f}%)")
    print(f"First-attempt Valid: {first_valid}/{n}")
    print(f"First-attempt Adequate: {first_adequate}/{n}")
    print(f"Final Executable: {final_executable}/{n}")
    print(f"REPLAN Rate: {replan_trials}/{n}")
    print(f"REPLAN Recovery: {replan_successful}/{replan_trials if replan_trials > 0 else 1}")
    print(f"Duplicate Plan Rate: {duplicate_plan_trials}/{n}")
    print(f"Schema Failure Rate: {schema_failure_trials}/{n}")
    print(f"Adequacy Failure Rate: {adequacy_failure_trials}/{n}")
    print(f"Timeout Rate: {timeout_failure_trials}/{n}")
    print(f"Execution Failure Rate: {execution_failure_trials}/{n}")
    print(f"Average Attempts: {summary['average_attempts_per_trial']:.2f}")
    print(f"Mean Planning Latency: {mean_planning_latency:.2f}s (Median: {median_planning_latency:.2f}s)")
    print(f"Screening Decision: {classification}")
    print(f"Recommendation: {recommendation}")
    print("=================================================================")


if __name__ == "__main__":
    main()
