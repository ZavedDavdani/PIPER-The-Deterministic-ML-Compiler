"""Small read-only helpers for recorded AgentState / store objects."""

from __future__ import annotations

from typing import Any, Optional


def dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return value


def field(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def winner_id(state: Any) -> Optional[str]:
    comparison = field(state, "comparison")
    return field(comparison, "recommended_model_id", "recommended_model_id")


def winner_training(state: Any) -> Optional[Any]:
    mid = winner_id(state)
    if not mid:
        return None
    for item in field(state, "model_results", "model_results", default=[]) or []:
        if field(item, "model_id") == mid:
            return item
    return None


def winner_evaluation(state: Any) -> Optional[Any]:
    mid = winner_id(state)
    if not mid:
        return None
    for item in field(state, "evaluation_results", "evaluation_results", default=[]) or []:
        if field(item, "model_id") == mid:
            return item
    return None


def operation_rows(state: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for log_name in ("cleaning_log", "cleaning_log", "feature_log", "feature_log"):
        for item in field(state, log_name, default=[]) or []:
            tool = field(item, "tool_name")
            if not tool:
                continue
            rows.append(
                {
                    "tool_name": str(tool),
                    "arguments": dict(field(item, "arguments", default={}) or {}),
                    "result_summary": field(item, "result_summary"),
                }
            )
    return rows


def preprocessing_lines(state: Any) -> list[str]:
    lines: list[str] = []
    for row in operation_rows(state):
        summary = row.get("result_summary")
        if summary:
            lines.append(f"{row['tool_name']}: {summary}")
        else:
            lines.append(row["tool_name"])
    return lines
