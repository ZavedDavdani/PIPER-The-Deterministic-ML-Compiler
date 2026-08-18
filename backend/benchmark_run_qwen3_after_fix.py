"""
BENCHMARK-ONLY SCRIPT — isolated from production code (same status as
the other benchmark_run_*.py scripts).

AFTER-fix re-benchmark of ONLY qwen3:4b, using the exact same
methodology as the original baseline run (3 first-attempt trials + up
to 2 REPLAN follow-ups, same Titanic fixture, same call budget) via
benchmark_planning_models.py's now-updated functions (first_attempt_context
and replan_context both now carry tool_schemas=TOOL_ARGUMENT_SCHEMAS,
matching plan_node_v2's real production behavior post-fix).

Deliberately scoped to ONLY qwen3:4b, and deliberately writes to a
SEPARATE results file (benchmark_results_after_fix.json) rather than
merging into benchmark_results.json — qwen3.5:4b/llama3.2:3b/gemma3:4b
are all now installed on this machine from prior benchmark rounds, so
reusing the CANDIDATE_MODELS loop unmodified would silently re-benchmark
all four, which was explicitly not requested this round. Keeping a
separate output file also guarantees the original BEFORE baseline for
qwen3:4b in benchmark_results.json is never overwritten, so a clean
before/after diff remains possible.
"""

from __future__ import annotations

import json

import benchmark_planning_models as b

MODEL = "qwen3:4b"
AFTER_RESULTS_PATH = b.RESULTS_PATH.parent / "benchmark_results_after_fix.json"


def main():
    installed = b._installed_models()
    assert MODEL in installed, f"{MODEL} not installed: {installed}"

    store, df = b._load_dataset()
    assert df.shape == (891, 12)

    sanitized_result = b.build_sanitized_llm_context(b.DATASET_ID, b.TARGET_COLUMN, store)
    assert sanitized_result.success
    budgeted_context, budget_report = b.apply_context_budget(sanitized_result.data)
    print(f"Context budgeting: {budget_report.model_dump()}")

    first_attempt_context = b.LLMPlanningContext(
        objective=f"Predict '{b.TARGET_COLUMN}' from the remaining columns (binary/multiclass classification).",
        dataset_context=budgeted_context.model_dump(mode="json"),
        allowed_operations=sorted(b.ALLOWED_TOOL_NAMES),
        tool_schemas=b.TOOL_ARGUMENT_SCHEMAS,
    )
    assert first_attempt_context.tool_schemas, "tool_schemas must be populated for the AFTER-fix run"

    result = b.benchmark_model(MODEL, b.DEFAULT_OLLAMA_HOST, first_attempt_context)
    result["status"] = "benchmarked"
    result["note"] = "AFTER planner-contract fix (tool_schemas now included in prompt)"

    AFTER_RESULTS_PATH.write_text(json.dumps({"models": [result]}, indent=2, default=str))
    print(f"\nWrote AFTER-fix results to {AFTER_RESULTS_PATH}")


if __name__ == "__main__":
    main()
