"""
MEASUREMENT-ONLY SCRIPT — isolated from production code.
Engineering-hardening Phase 2B, Step 1b: cheap empirical verification
that Ollama 0.32.9's `think` request parameter is actually honored by
qwen3:4b, BEFORE spending real Titanic-benchmark-scale time on it.

Uses a trivial, cheap prompt (not the real planning prompt) — the goal
here is only to confirm the mechanism exists and measurably changes
behavior, not to measure planning quality (that's Step 2/3, against
the real fixture).
"""

from __future__ import annotations

import json
import time
import urllib.request

HOST = "http://localhost:11434"
MODEL = "qwen3:4b"
PROMPT = "What is 7 plus 5? Respond with only the final number, nothing else."


def _call(think, label: str) -> dict:
    payload = {"model": MODEL, "prompt": PROMPT, "stream": False}
    if think is not None:
        payload["think"] = think

    request = urllib.request.Request(
        url=f"{HOST}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"[{label}] sending (think={think!r})...", flush=True)
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read()
    wall = time.perf_counter() - t0
    body = json.loads(raw.decode("utf-8"))

    thinking = body.get("thinking") or ""
    answer = body.get("response") or ""
    record = {
        "label": label,
        "think_sent": think,
        "wall_seconds": wall,
        "eval_count": body.get("eval_count"),
        "eval_duration_s": body.get("eval_duration", 0) / 1e9,
        "thinking_field_present_and_nonempty": bool(thinking.strip()),
        "thinking_chars": len(thinking),
        "response_chars": len(answer),
        "response_text": answer.strip()[:200],
        "thinking_excerpt": thinking.strip()[:200],
    }
    print(json.dumps(record, indent=2))
    return record


def main():
    results = []
    results.append(_call(False, "think_false"))
    results.append(_call(True, "think_true"))
    results.append(_call(None, "think_omitted_default"))

    with open("think_probe_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nWritten to think_probe_results.json")


if __name__ == "__main__":
    main()
