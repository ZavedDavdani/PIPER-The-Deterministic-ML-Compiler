"""
MEASUREMENT-ONLY SCRIPT — isolated from production code.
Engineering-hardening Phase 2B, Steps 2+3: establishes the current
generation-time baseline against the real Titanic fixture, then tests
ONE isolated generation-control candidate (`think`) — the only
verified, Ollama-honored mechanism found in Step 1b's probe — against
the REAL production request shape (grammar-constrained JSON `format`,
`tool_schemas`, `keep_alive`), since the cheap unconstrained probe
might not predict behavior under schema-constrained decoding.

Only ONE variable changes between baseline and candidate: the `think`
field. Everything else (prompt, schema, keep_alive, model) is
identical, matching the "do not change multiple variables
simultaneously" instruction.

  BASELINE (2 calls): current production default — no explicit
      `think` field sent (exactly what OllamaProvider.generate_plan()
      does today).
  CANDIDATE (2 calls): identical request, with `"think": false` added.
"""

from __future__ import annotations

import json
import time
import urllib.request

import benchmark_planning_models as b
from app.llm.ollama_provider import DEFAULT_KEEP_ALIVE

RESULTS_PATH = "generation_control_experiment_results.json"


def _call(prompt: str, think, label: str) -> dict:
    payload = {
        "model": "qwen3:4b",
        "prompt": prompt,
        "stream": False,
        "format": b.PLAN_JSON_SCHEMA,
        "keep_alive": DEFAULT_KEEP_ALIVE,
    }
    if think is not None:
        payload["think"] = think

    request = urllib.request.Request(
        url=f"{b.DEFAULT_OLLAMA_HOST.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"  [{label}] sending (think={think!r})...", flush=True)
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=b.DEFAULT_TIMEOUT_SECONDS) as response:
        raw = response.read()
    wall = time.perf_counter() - t0
    body = json.loads(raw.decode("utf-8"))

    content = b._extract_content(body)
    cleaned = b._strip_markdown_fences(content) if content else None
    plan = None
    parse_error = None
    try:
        plan_json = json.loads(cleaned) if cleaned else None
        plan = b.ProposedPlan.model_validate(plan_json) if plan_json else None
    except Exception as e:
        parse_error = str(e)

    validation = b.validate_proposed_plan(plan.steps, b.TARGET_COLUMN) if plan else None

    record = {
        "label": label,
        "think_sent": think,
        "wall_seconds": wall,
        "load_duration_s": body.get("load_duration", 0) / 1e9,
        "prompt_eval_duration_s": body.get("prompt_eval_duration", 0) / 1e9,
        "eval_duration_s": body.get("eval_duration", 0) / 1e9,
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
        "tokens_per_sec": (body.get("eval_count") or 0) / (body.get("eval_duration", 1) / 1e9)
        if body.get("eval_duration") else None,
        "thinking_field_chars": len(body.get("thinking") or ""),
        "response_field_chars": len(body.get("response") or ""),
        "structured_plan_produced": plan is not None,
        "parse_error": parse_error,
        "plan_valid": validation.valid if validation else None,
        "violations": [v.model_dump(mode="json") for v in validation.violations] if validation else [],
        "step_count": len(plan.steps) if plan else 0,
        "proposed_steps": [{"tool_name": s.tool_name, "arguments": s.arguments} for s in plan.steps] if plan else [],
    }
    print(
        f"    -> wall={wall:.1f}s eval={record['eval_duration_s']:.1f}s "
        f"tokens={record['eval_count']} valid={record['plan_valid']} "
        f"structured={record['structured_plan_produced']}",
        flush=True,
    )
    return record


def main():
    store, df = b._load_dataset()
    assert df.shape == (891, 12)
    sanitized_result = b.build_sanitized_llm_context(b.DATASET_ID, b.TARGET_COLUMN, store)
    assert sanitized_result.success
    budgeted_context, _ = b.apply_context_budget(sanitized_result.data)
    context = b.LLMPlanningContext(
        objective=f"Predict '{b.TARGET_COLUMN}' from the remaining columns (binary/multiclass classification).",
        dataset_context=budgeted_context.model_dump(mode="json"),
        allowed_operations=sorted(b.ALLOWED_TOOL_NAMES),
        tool_schemas=b.TOOL_ARGUMENT_SCHEMAS,
    )
    prompt = b.build_planning_prompt(context)
    print(f"Fixed prompt: {len(prompt)} chars, keep_alive={DEFAULT_KEEP_ALIVE!r}\n")

    results = {"prompt_chars": len(prompt), "trials": []}

    print("=== BASELINE: current production default (no explicit think field) ===")
    results["trials"].append(_call(prompt, None, "baseline_1_think_default"))
    results["trials"].append(_call(prompt, None, "baseline_2_think_default"))

    print("\n=== CANDIDATE: think=false (only variable changed) ===")
    results["trials"].append(_call(prompt, False, "candidate_1_think_false"))
    results["trials"].append(_call(prompt, False, "candidate_2_think_false"))

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
