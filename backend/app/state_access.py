"""Read fields from AgentState objects or dict snapshots restored from SQLite."""

from __future__ import annotations

from typing import Any, Optional


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
    return field(comparison, "recommended_model_id")
