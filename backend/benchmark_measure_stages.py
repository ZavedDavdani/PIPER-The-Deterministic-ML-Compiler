"""
MEASUREMENT-ONLY SCRIPT — isolated from production code, Phase 1 of
the engineering-hardening investigation (see CLAUDE.md).

Makes exactly ONE real Ollama call against the current PRODUCTION
prompt construction (post planner-contract-fix: tool_schemas included,
matching plan_node_v2 exactly) against the real Titanic fixture, with
maximally fine-grained stage timing AND full raw response body capture
— including Ollama's `thinking` and `response` fields separately,
which the existing benchmark harness discards after extracting the
final plan. This is the one piece of data needed to estimate the
reasoning-vs-final-answer token split, since Ollama's non-streaming
/api/generate response reports only a single combined `eval_count`
(no separate thinking/response token counts).

Does not modify or call any production code path — reuses the same
functions benchmark_planning_models.py already reuses from app/.
"""

from __future__ import annotations

import json
import time
import urllib.request

import benchmark_planning_models as b

OUTPUT_PATH = "measurement_stage_capture.json"


def main():
    t_load_start = time.perf_counter()
    store, df = b._load_dataset()
    t_load_end = time.perf_counter()
    assert df.shape == (891, 12)

    t_sanitize_start = time.perf_counter()
    sanitized_result = b.build_sanitized_llm_context(b.DATASET_ID, b.TARGET_COLUMN, store)
    t_sanitize_end = time.perf_counter()
    assert sanitized_result.success

    t_budget_start = time.perf_counter()
    budgeted_context, budget_report = b.apply_context_budget(sanitized_result.data)
    t_budget_end = time.perf_counter()

    t_ctx_build_start = time.perf_counter()
    context = b.LLMPlanningContext(
        objective=f"Predict '{b.TARGET_COLUMN}' from the remaining columns (binary/multiclass classification).",
        dataset_context=budgeted_context.model_dump(mode="json"),
        allowed_operations=sorted(b.ALLOWED_TOOL_NAMES),
        tool_schemas=b.TOOL_ARGUMENT_SCHEMAS,
    )
    t_ctx_build_end = time.perf_counter()

    t_prompt_start = time.perf_counter()
    prompt = b.build_planning_prompt(context)
    t_prompt_end = time.perf_counter()

    payload = {"model": "qwen3:4b", "prompt": prompt, "stream": False, "format": b.PLAN_JSON_SCHEMA}
    request_body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{b.DEFAULT_OLLAMA_HOST.rstrip('/')}/api/generate",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"Prompt: {len(prompt)} chars. Sending request at {time.strftime('%H:%M:%S')}...")
    t_request_start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=b.DEFAULT_TIMEOUT_SECONDS) as response:
        raw_bytes = response.read()
    t_request_end = time.perf_counter()
    print(f"Response received after {t_request_end - t_request_start:.1f}s wall time.")

    body = json.loads(raw_bytes.decode("utf-8"))

    t_parse_start = time.perf_counter()
    content = b._extract_content(body)
    cleaned = b._strip_markdown_fences(content) if content else None
    plan_json = json.loads(cleaned) if cleaned else None
    plan = b.ProposedPlan.model_validate(plan_json) if plan_json else None
    t_parse_end = time.perf_counter()

    t_validate_start = time.perf_counter()
    validation_result = None
    if plan is not None:
        validation_result = b.validate_proposed_plan(plan.steps, b.TARGET_COLUMN)
    t_validate_end = time.perf_counter()

    thinking_text = body.get("thinking") or ""
    response_text = body.get("response") or ""

    result = {
        "stage_timings_seconds": {
            "dataset_load": t_load_end - t_load_start,
            "sanitize_context": t_sanitize_end - t_sanitize_start,
            "apply_budget": t_budget_end - t_budget_start,
            "build_llm_planning_context": t_ctx_build_end - t_ctx_build_start,
            "build_prompt_string": t_prompt_end - t_prompt_start,
            "ollama_http_wall": t_request_end - t_request_start,
            "response_parsing": t_parse_end - t_parse_start,
            "deterministic_validation": t_validate_end - t_validate_start,
        },
        "prompt_chars": len(prompt),
        "ollama_reported": {
            "total_duration_s": body.get("total_duration", 0) / 1e9,
            "load_duration_s": body.get("load_duration", 0) / 1e9,
            "prompt_eval_duration_s": body.get("prompt_eval_duration", 0) / 1e9,
            "eval_duration_s": body.get("eval_duration", 0) / 1e9,
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
            "done_reason": body.get("done_reason"),
        },
        "thinking_response_split": {
            "thinking_chars": len(thinking_text),
            "response_chars": len(response_text),
            "thinking_words": len(thinking_text.split()),
            "response_words": len(response_text.split()),
        },
        "plan_valid": validation_result.valid if validation_result else None,
        "violations": [v.model_dump(mode="json") for v in validation_result.violations] if validation_result else [],
        "proposed_step_count": len(plan.steps) if plan else 0,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(json.dumps(result, indent=2, default=str))
    print(f"\nFull result written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
