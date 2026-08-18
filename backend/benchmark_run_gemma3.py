"""
BENCHMARK-ONLY SCRIPT — isolated from production code (same status as
the other benchmark_run_*.py scripts, which this mirrors exactly).

Runs the exact same methodology as the qwen3:4b/qwen3.5:4b/llama3.2:3b
benchmarks, scoped to ONLY gemma3:4b. No other model is re-benchmarked.
Merges the new result into the existing benchmark_results.json,
leaving every other entry untouched.
"""

from __future__ import annotations

import json

import benchmark_planning_models as b

MODEL = "gemma3:4b"


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
    )

    result = b.benchmark_model(MODEL, b.DEFAULT_OLLAMA_HOST, first_attempt_context)
    result["status"] = "benchmarked"

    existing = json.loads(b.RESULTS_PATH.read_text())
    for i, entry in enumerate(existing["models"]):
        if entry["model"] == MODEL:
            existing["models"][i] = result
            break
    else:
        existing["models"].append(result)

    b.RESULTS_PATH.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nMerged {MODEL} results into {b.RESULTS_PATH}")


if __name__ == "__main__":
    main()
