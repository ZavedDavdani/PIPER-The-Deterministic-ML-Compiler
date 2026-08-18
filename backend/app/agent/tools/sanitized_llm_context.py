"""
Sanitized LLM-context representation (M3 Phase 3).

Closes the architectural gap identified during Phase 1 inspection:

    state.profile (DatasetProfile, from profile_dataset())
        -> column_profiles[].sample_values
        -> RAW dataset values, never filtered by sanitize_for_llm_context()

sanitize_for_llm_context() (Section 10) already scans every non-numeric
cell and computes a neutralized replacement value per finding (see
_neutralize() in app/agent/tools/sanitization.py) — but it DISCARDS
that replacement, keeping only an 80-char excerpt inside
SanitizationFinding for human-readable evidence, not for reuse as safe
LLM-facing content. This module is what actually builds the redacted,
LLM-safe view: it reuses (imports, never duplicates) the exact same
detection/classification/neutralization functions from
app.agent.tools.sanitization, applied to the dataset's SAMPLE VALUES
specifically (the one raw-content surface DatasetProfile exposes),
and combines that with the profile's already-safe-by-construction
fields (row/column counts, dtypes, numeric stats, missing/unique
percentages — none of which are free text, none of which need
scanning).

    Original DataFrame (DatasetStore, read via store.get(), never mutated)
        |
        +--> deterministic pipeline (unchanged, reads via DatasetStore directly)
        |
        +--> build_sanitized_llm_context()
                 |
                 v
             per-column sample-value scan (reuses sanitization.py's
             _scan_value/_neutralize/_classify_column)
                 |
                 v
             SanitizedLLMContext (safe sample_values, safe metadata,
             sanitization evidence carried alongside)
                 |
                 v
             LLM (via LLMPlanningContext.dataset_context, wired in a
             later phase)

CRITICAL INVARIANTS:
- The original DataFrame in DatasetStore is NEVER mutated. This module
  only ever calls store.get() (which itself returns a defensive copy,
  per InMemoryDatasetStore's existing contract) and reads from that
  copy.
- state.profile (DatasetProfile) is NEVER mutated by this module. If a
  deterministic component needs DatasetProfile's raw sample_values
  (none currently do — plan_node_v2 only reads dtype/unique_percentage/
  name, never sample_values), it continues to read profile_dataset()'s
  output directly; SanitizedLLMContext is an ADDITIONAL, separate
  representation, never a replacement for DatasetProfile.
- Numeric columns are never subjected to text scanning (matches
  sanitize_for_llm_context()'s own dtype-based skip) — their sample
  values pass through unchanged, since a float/bool cannot carry a
  string-based injection pattern by construction.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.agent.tools._profiling_helpers import is_numeric_dtype
from app.agent.tools.sanitization import _classify_column, _neutralize, _scan_value
from app.schemas import ToolError, ToolResult
from app.schemas.sanitization import SanitizationReport
from app.storage import DatasetNotFoundError, DatasetStore

MAX_SAMPLE_VALUES_PER_COLUMN = 5
"""
How many (safe, post-sanitization) sample values are carried into the
LLM context per column — deliberately smaller than DatasetProfile's
own up-to-10 sample_values, since this content is actually reaching an
LLM prompt (token cost matters) rather than just being available for
programmatic inspection.
"""


class SanitizedColumnContext(BaseModel):
    """
    One column's LLM-safe representation. Metadata fields (dtype,
    missing/unique percentages, numeric stats) are carried through
    unchanged from DatasetProfile — they are safe-by-construction
    (never free text, never user-authored content) and do not need
    scanning. sample_values is the ONE field that is genuinely
    rebuilt here, never copied raw from DatasetProfile.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    missing_percentage: float = Field(..., ge=0.0, le=100.0)
    unique_percentage: float = Field(..., ge=0.0, le=100.0)
    sample_values: list = Field(
        default_factory=list,
        description="Post-sanitization sample values only — never raw dataset content for non-numeric columns.",
    )
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None


class SanitizedLLMContext(BaseModel):
    """
    Output of build_sanitized_llm_context(). This — never
    DatasetProfile directly, never a raw DataFrame — is what should
    reach an LLM provider's dataset_context field (wiring happens in a
    later phase).
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    rows: int = Field(..., ge=0)
    columns: int = Field(..., ge=0)
    target_column: Optional[str] = None
    column_contexts: list[SanitizedColumnContext] = Field(default_factory=list)
    sanitization_findings_count: int = Field(
        ..., ge=0, description="How many sanitization findings were produced while building this context."
    )
    high_risk_columns: list[str] = Field(default_factory=list)
    scope_note: str = Field(
        default=(
            "sample_values in this context have already been scanned and, "
            "where necessary, neutralized/truncated/excluded per the "
            "Section 10 sanitization policy. This is defense-in-depth "
            "pattern detection, not a guarantee against every possible "
            "injection technique."
        ),
    )


def build_sanitized_llm_context(
    dataset_id: str,
    target_column: Optional[str],
    store: DatasetStore,
) -> ToolResult[SanitizedLLMContext]:
    """
    Builds the LLM-safe context view for a dataset. Reuses
    sanitization.py's _scan_value()/_neutralize()/_classify_column()
    directly (never a reimplementation) so detection/neutralization
    logic has exactly one source of truth in the codebase.

    Sample values are drawn from the dataset's NON-NULL values per
    column (mirrors DatasetProfile's own sample_values convention),
    capped at MAX_SAMPLE_VALUES_PER_COLUMN after sanitization — a
    value that gets excluded is simply not included in the sample
    (never replaced with a placeholder that itself counts toward the
    sample size), so the LLM sees fewer, genuinely safe examples
    rather than a sample padded with visible "[REDACTED]" placeholders.

    Errors:
    - dataset doesn't exist
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[SanitizedLLMContext](
            success=False,
            tool_name="build_sanitized_llm_context",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    total_rows = len(df)
    column_contexts: list[SanitizedColumnContext] = []
    findings_count = 0
    high_risk_columns: list[str] = []

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_percentage = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        unique_percentage = round((unique_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        col_min = col_max = col_mean = None

        if is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            # Numeric/boolean columns cannot carry a string-based
            # injection pattern by construction — matches
            # sanitize_for_llm_context()'s own dtype-based skip.
            # Sample values pass through unchanged; numeric stats are
            # computed since they're safe, useful planning evidence.
            non_null = series.dropna()
            sample_values = non_null.head(MAX_SAMPLE_VALUES_PER_COLUMN).tolist()
            if is_numeric_dtype(series) and len(non_null) > 0:
                col_min = round(float(non_null.min()), 4)
                col_max = round(float(non_null.max()), 4)
                col_mean = round(float(non_null.mean()), 4)
        else:
            classification = _classify_column(series, total_rows)
            if classification.risk_level == "high_risk_free_text":
                high_risk_columns.append(col)

            non_null = series.dropna()
            safe_values: list = []
            for _, value in non_null.items():
                if len(safe_values) >= MAX_SAMPLE_VALUES_PER_COLUMN:
                    break

                value_str = str(value)
                value_findings = _scan_value(value_str)

                if not value_findings:
                    safe_values.append(value_str)
                    continue

                finding_types = [f[0] for f in value_findings]
                sanitized_value, action = _neutralize(value_str, finding_types)
                findings_count += len(value_findings)

                if action == "excluded":
                    # Never add placeholder text to the sample — an
                    # excluded value simply contributes nothing to the
                    # sample, per this function's documented policy.
                    continue

                safe_values.append(sanitized_value)

            sample_values = safe_values

        column_contexts.append(
            SanitizedColumnContext(
                name=col,
                dtype=str(series.dtype),
                missing_percentage=missing_percentage,
                unique_percentage=unique_percentage,
                sample_values=sample_values,
                min=col_min,
                max=col_max,
                mean=col_mean,
            )
        )

    context = SanitizedLLMContext(
        dataset_id=dataset_id,
        rows=total_rows,
        columns=len(df.columns),
        target_column=target_column,
        column_contexts=column_contexts,
        sanitization_findings_count=findings_count,
        high_risk_columns=high_risk_columns,
    )

    return ToolResult[SanitizedLLMContext](
        success=True,
        tool_name="build_sanitized_llm_context",
        message=(
            f"Built sanitized LLM context for '{dataset_id}': {len(df.columns)} columns, "
            f"{findings_count} sanitization finding(s) applied, "
            f"{len(high_risk_columns)} high-risk column(s)."
        ),
        data=context,
    )
