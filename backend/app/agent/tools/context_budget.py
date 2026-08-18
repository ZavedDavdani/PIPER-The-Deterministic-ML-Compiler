"""
Deterministic LLM context-budgeting (Batch 7).

Estimates the size of a SanitizedLLMContext (the ONLY dataset view an
LLM ever sees — see sanitized_llm_context.py) before it's handed to
llm_provider.generate_plan(), and applies a deterministic, staged
reduction when that estimate exceeds a fixed budget — never an
arbitrary or lossy truncation of important information.

Locked minimums this module NEVER removes, regardless of how far over
budget the context is: column names, dtypes, target_column,
missing_percentage, unique_percentage, and (for numeric columns)
min/max/mean. The ONLY thing ever reduced is sample_values — first
capped smaller, then dropped to empty as the last resort — since
per-column sample values are the one field whose size scales with how
MANY representative examples are kept, not with how much essential
metadata is kept.

Size is estimated as a character count of the context's JSON
serialization — a deterministic, dependency-free proxy for LLM prompt
size (this project has no tokenizer dependency; OllamaProvider itself
is stdlib-only `urllib`, and this module follows the same "no new
dependency" discipline). ~4 characters per token is a commonly used
rough approximation for English text, so DEFAULT_MAX_CONTEXT_CHARS
corresponds to roughly 2000 tokens of dataset_context content.

DEFAULT_MAX_CONTEXT_CHARS is grounded in a real measurement, not a
guess (per this project's standing "measure, don't guess" rule — see
CLAUDE.md's Ollama timeout finding for the same discipline applied
elsewhere): the real reference Telco Customer Churn dataset (21
columns) produces a 4,569-character SanitizedLLMContext at the default
MAX_SAMPLE_VALUES_PER_COLUMN=5 — comfortably under budget, so this
change has zero effect on the reference dataset's planning context. A
synthetic 81-column dataset produces 18,539 characters — budgeting
genuinely activates there, which is the "larger datasets" case this
batch targets.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from app.agent.tools.sanitized_llm_context import SanitizedLLMContext

DEFAULT_MAX_CONTEXT_CHARS = 8000
"""
~75% above the real Telco dataset's own 4,569-character context (real
measurement, see module docstring) — enough headroom that a modestly
wider real dataset still passes through unbudgeted, while a genuinely
large/wide dataset (tens of columns) will exceed it and trigger
deterministic reduction.
"""

_SAMPLE_VALUE_REDUCTION_STAGES: tuple[int, ...] = (2, 1, 0)
"""
Staged caps tried in order once the context exceeds budget at its
starting cap (sanitized_llm_context.MAX_SAMPLE_VALUES_PER_COLUMN, 5).
Stops at the first stage that brings the estimate back under budget;
0 (no sample values at all, metadata only) is the deterministic floor.
"""


class ContextBudgetReport(BaseModel):
    """
    What apply_context_budget() actually did — returned alongside the
    (possibly reduced) context so callers/tests can verify budgeting
    behavior without re-deriving it from the context itself.
    """

    model_config = ConfigDict(extra="forbid")

    original_size_chars: int = Field(..., ge=0)
    final_size_chars: int = Field(..., ge=0)
    budget_chars: int = Field(..., ge=0)
    reduction_applied: bool
    sample_values_cap_used: int = Field(
        ..., ge=0, description="The per-column sample_values cap actually applied in the returned context."
    )


def estimate_context_size(context: SanitizedLLMContext) -> int:
    """Character count of the context's JSON serialization — see module docstring for the token-proxy rationale."""
    return len(json.dumps(context.model_dump(mode="json")))


def apply_context_budget(
    context: SanitizedLLMContext, max_chars: int = DEFAULT_MAX_CONTEXT_CHARS
) -> tuple[SanitizedLLMContext, ContextBudgetReport]:
    """
    Returns (possibly-reduced context, report). If the context is
    already within budget, returns it completely unchanged (same
    sample_values, same everything) — budgeting is a no-op for any
    dataset small enough not to need it, which covers every dataset
    this codebase currently tests against, including the real Telco
    CSV (see module docstring's real measurement).
    """
    original_size = estimate_context_size(context)
    starting_cap = max((len(col.sample_values) for col in context.column_contexts), default=0)

    if original_size <= max_chars:
        return context, ContextBudgetReport(
            original_size_chars=original_size,
            final_size_chars=original_size,
            budget_chars=max_chars,
            reduction_applied=False,
            sample_values_cap_used=starting_cap,
        )

    for cap in _SAMPLE_VALUE_REDUCTION_STAGES:
        reduced_columns = [
            col.model_copy(update={"sample_values": col.sample_values[:cap]})
            for col in context.column_contexts
        ]
        reduced_context = context.model_copy(update={"column_contexts": reduced_columns})
        size = estimate_context_size(reduced_context)

        if size <= max_chars or cap == _SAMPLE_VALUE_REDUCTION_STAGES[-1]:
            # Column names/dtypes/target_column/missing_percentage/
            # unique_percentage/min/max/mean are never touched above —
            # only sample_values shrinks. cap == 0 (the last stage) is
            # the deterministic floor: even if STILL over budget with
            # zero sample values (an extremely wide dataset), this is
            # returned anyway rather than ever dropping a column or any
            # other locked-minimum field.
            return reduced_context, ContextBudgetReport(
                original_size_chars=original_size,
                final_size_chars=size,
                budget_chars=max_chars,
                reduction_applied=True,
                sample_values_cap_used=cap,
            )

    # Unreachable (the loop's last stage always returns above), kept
    # only so this function has an explicit, typed fallback.
    return context, ContextBudgetReport(
        original_size_chars=original_size,
        final_size_chars=original_size,
        budget_chars=max_chars,
        reduction_applied=False,
        sample_values_cap_used=starting_cap,
    )
