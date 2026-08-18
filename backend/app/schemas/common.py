"""
Common envelope shared by every deterministic tool.

Locked contract:
    ToolResult
    ├── success: bool
    ├── tool_name: str
    ├── message: str
    ├── data: structured payload (tool-specific, typed at call site)
    └── error: optional structured error

Tools never call the LLM and never decide what action to take — they
execute the action requested by the agent and report the result. This
envelope is what makes that reporting uniform across every tool, and
what the tool_trace[] entries in AgentState will wrap later.
"""

from __future__ import annotations

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

DataT = TypeVar("DataT", bound=BaseModel)


class ToolError(BaseModel):
    """Structured error — never a bare string exception message."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        description=(
            "Stable machine-readable error identifier, e.g. "
            "'dataset_not_found', 'column_not_found', "
            "'target_column_protected', 'unsupported_conversion'."
        ),
    )
    message: str = Field(..., description="Human-readable explanation.")
    details: dict = Field(
        default_factory=dict,
        description="Optional structured context (e.g. {'column': 'foo'}).",
    )


class ToolResult(BaseModel, Generic[DataT]):
    """
    Generic envelope every tool function returns.

    `data` is populated only when `success` is True.
    `error` is populated only when `success` is False.
    Exactly one of the two should be non-null — enforced by each tool's
    construction helper (see storage/schemas usage), not re-validated
    here, to keep this envelope reusable for every payload type.
    """

    model_config = ConfigDict(extra="forbid")

    success: bool
    tool_name: str
    message: str
    data: Optional[DataT] = None
    error: Optional[ToolError] = None
