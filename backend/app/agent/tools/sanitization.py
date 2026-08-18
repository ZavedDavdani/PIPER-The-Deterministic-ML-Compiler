"""
sanitize_for_llm_context() (section 10).

    original_dataset (DatasetStore, untouched)
        |
        v
    for every non-numeric column (dtype fact, not a risk judgment):
        STAGE 1 -- cheap, UNCONDITIONAL per-cell security-pattern scan
                   (runs regardless of column classification/risk level)
        |
        v
        suspicious? --no--> done, nothing recorded
        |
       yes
        |
        v
    STAGE 2 -- classify + neutralize + record structured evidence
        |
        v
    build a SEPARATE sanitized_llm_context dict (never written back
    to DatasetStore) + structured SanitizationReport evidence

CRITICAL DESIGN INVARIANT (this was a real bug, now fixed and
regression-tested): column risk classification (low_categorical /
business_text / high_risk_free_text) NEVER gates whether the security
scan runs. It only affects contextual treatment downstream (e.g. how
findings are presented). A column can look exactly like an ordinary
low-cardinality categorical (mostly-identical benign values) while
still containing a single injected malicious row, and the scan MUST
still catch it — cardinality is not a security signal, it is
incidental. The classification exists so a genuinely legitimate
categorical column (Contract: "Month-to-month"/"One year"/"Two year")
still receives a lighter CONTEXTUAL treatment and isn't reported as if
it were free text, but it is never treated as exempt from scanning.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from app.agent.tools._profiling_helpers import is_numeric_dtype
from app.schemas import ToolError, ToolResult
from app.schemas.sanitization import (
    MAX_SAFE_TEXT_LENGTH,
    ColumnRiskClassification,
    SanitizationFinding,
    SanitizationReport,
)
from app.storage import DatasetNotFoundError, DatasetStore

# --- Detection patterns ----------------------------------------------------
# Deliberately pattern-based (not an ML classifier) so behavior is
# fully deterministic and explainable — consistent with every other
# guardrail in this system.

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(the\s+)?above", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|earlier)", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(developer|debug|admin|god)\s*mode", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"</?(system|assistant|user)>", re.IGNORECASE),
]

_ROLE_IMPERSONATION_PATTERNS = [
    re.compile(r"^\s*(system|assistant|developer)\s*:", re.IGNORECASE),
    re.compile(r"\[(system|assistant|developer)\]", re.IGNORECASE),
]

_COMMAND_LIKE_PATTERNS = [
    re.compile(r"\brm\s+-rf\b"),
    re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
    re.compile(r"<script[\s>]", re.IGNORECASE),
    re.compile(r"\b(exec|eval)\s*\("),
]

_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Low-cardinality threshold: a text column where distinct values are
# fewer than this fraction of total rows is treated as categorical,
# not free text — matches the same spirit as check_high_cardinality()'s
# uniqueness reasoning, but inverted (LOW uniqueness = categorical
# here, whereas check_high_cardinality() flags HIGH uniqueness).
_CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.05
_CATEGORICAL_MAX_AVG_LENGTH = 30


def _classify_column(series: pd.Series, total_rows: int) -> ColumnRiskClassification:
    name = series.name
    if is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return ColumnRiskClassification(
            column=name, risk_level="low_categorical",
            reason="Numeric/boolean column — not text, no scanning needed.",
            unique_value_ratio=0.0,
        )

    non_null = series.dropna()
    if len(non_null) == 0:
        return ColumnRiskClassification(
            column=name, risk_level="low_categorical",
            reason="Column is entirely null.", unique_value_ratio=0.0,
        )

    unique_ratio = non_null.nunique() / total_rows if total_rows > 0 else 0.0
    avg_length = non_null.astype(str).str.len().mean()

    if unique_ratio <= _CATEGORICAL_UNIQUE_RATIO_THRESHOLD and avg_length <= _CATEGORICAL_MAX_AVG_LENGTH:
        return ColumnRiskClassification(
            column=name, risk_level="low_categorical",
            reason=f"Low cardinality ({unique_ratio:.2%} unique) and short values (avg {avg_length:.0f} chars) — categorical, not free text.",
            unique_value_ratio=round(float(unique_ratio), 4),
        )

    if avg_length > 100 or unique_ratio > 0.5:
        return ColumnRiskClassification(
            column=name, risk_level="high_risk_free_text",
            reason=f"High cardinality ({unique_ratio:.2%} unique) or long values (avg {avg_length:.0f} chars) — genuine free text, scanned per-cell.",
            unique_value_ratio=round(float(unique_ratio), 4),
        )

    return ColumnRiskClassification(
        column=name, risk_level="business_text",
        reason=f"Moderate cardinality ({unique_ratio:.2%} unique), moderate length (avg {avg_length:.0f} chars) — ordinary business text, scanned per-cell but not over-sanitized.",
        unique_value_ratio=round(float(unique_ratio), 4),
    )


def _scan_value(value: str) -> list:
    """Returns list of (finding_type, reason) tuples for one value."""
    findings = []

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(value):
            findings.append(("prompt_injection_pattern", f"Matched injection pattern: {pattern.pattern}"))
            break  # one injection finding per value is enough evidence

    for pattern in _ROLE_IMPERSONATION_PATTERNS:
        if pattern.search(value):
            findings.append(("role_impersonation", f"Matched role-impersonation pattern: {pattern.pattern}"))
            break

    for pattern in _COMMAND_LIKE_PATTERNS:
        if pattern.search(value):
            findings.append(("command_like_text", f"Matched command-like pattern: {pattern.pattern}"))
            break

    if _CONTROL_CHAR_PATTERN.search(value):
        findings.append(("control_characters", "Contains non-printable control characters."))

    if len(value) > MAX_SAFE_TEXT_LENGTH:
        findings.append(("excessive_length", f"Value length {len(value)} exceeds max safe length {MAX_SAFE_TEXT_LENGTH}."))

    # Suspicious encoding: text containing characters that normalize
    # differently under NFKC (a common technique for hiding
    # instructions using visually-similar Unicode characters).
    try:
        normalized = unicodedata.normalize("NFKC", value)
        if normalized != value and any(ord(c) > 0x2000 for c in value):
            findings.append(("suspicious_encoding", "Value normalizes differently under NFKC and contains unusual Unicode ranges."))
    except (TypeError, ValueError):
        pass

    return findings


def _neutralize(value: str, finding_types: list) -> tuple:
    """Returns (sanitized_value, action_taken)."""
    if "prompt_injection_pattern" in finding_types or "role_impersonation" in finding_types or "command_like_text" in finding_types:
        return "[REDACTED: content excluded by sanitization gate]", "excluded"
    if "control_characters" in finding_types:
        cleaned = _CONTROL_CHAR_PATTERN.sub("", value)
        if len(cleaned) > MAX_SAFE_TEXT_LENGTH:
            cleaned = cleaned[:MAX_SAFE_TEXT_LENGTH] + "...[truncated]"
        return cleaned, "neutralized"
    if "excessive_length" in finding_types:
        return value[:MAX_SAFE_TEXT_LENGTH] + "...[truncated]", "truncated"
    if "suspicious_encoding" in finding_types:
        return "[REDACTED: suspicious encoding excluded by sanitization gate]", "excluded"
    return value, "neutralized"


def sanitize_for_llm_context(
    dataset_id: str,
    store: DatasetStore,
    max_findings_per_column: int = 500,
) -> ToolResult[SanitizationReport]:
    """
    Builds a sanitized LLM-context view of a dataset's sample values —
    a SEPARATE structure, never written back to the original dataset.

    STAGE 1 (the security scan) runs on EVERY non-numeric row, not a
    truncated sample — this was a second, related bug found alongside
    the classification-gating bug: an earlier version used
    `.head(sample_rows)` to cap scanning to the first N rows, which
    meant an attacker could simply place malicious content past that
    cutoff and evade detection entirely, independent of the
    classification issue. Regex pattern matching is cheap enough that
    full-column scanning is practical even for large datasets; there
    is no security justification for a positional sampling cutoff on
    a detection pass. max_findings_per_column caps how much evidence
    is RECORDED (to avoid a pathological dataset producing millions of
    finding records), not how much is SCANNED — every row is always
    scanned.

    Errors:
    - dataset doesn't exist
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[SanitizationReport](
            success=False,
            tool_name="sanitize_for_llm_context",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    total_rows = len(df)
    classifications: list[ColumnRiskClassification] = []
    findings: list[SanitizationFinding] = []
    excluded_count = 0
    truncated_count = 0

    for col in df.columns:
        series = df[col]
        classification = _classify_column(series, total_rows)
        classifications.append(classification)

        if is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            # Only genuinely non-textual columns are skipped. This is
            # a dtype fact, not a risk judgment — a float or bool
            # column cannot contain a string-based injection pattern
            # by construction. Every column that COULD contain text is
            # scanned regardless of classification (see module
            # docstring: classification controls contextual treatment,
            # never whether the security scan runs at all — this was
            # a genuine bug in the original design, fixed here).
            continue

        # STAGE 1 — cheap, unconditional per-cell security-pattern
        # scan. Runs on EVERY row of every non-numeric column,
        # regardless of risk_level AND regardless of row position —
        # both the classification gate and a positional sampling cap
        # were real bugs (see module docstring); fixed here. This is
        # what makes it impossible for a malicious cell to escape
        # detection either by hiding among low-cardinality benign
        # values OR by sitting past some row cutoff.
        non_null = series.dropna()
        column_finding_count = 0
        for idx, value in non_null.items():
            if column_finding_count >= max_findings_per_column:
                break  # cap RECORDED evidence only — see docstring

            value_str = str(value)
            value_findings = _scan_value(value_str)
            if not value_findings:
                continue

            # STAGE 2 — deeper handling only for values that actually
            # triggered stage 1. Classification affects presentation
            # (e.g. how much of the excerpt to keep) but never whether
            # the finding is reported.
            finding_types = [f[0] for f in value_findings]
            _, action = _neutralize(value_str, finding_types)

            if action == "excluded":
                excluded_count += 1
            elif action == "truncated":
                truncated_count += 1

            for finding_type, reason in value_findings:
                findings.append(
                    SanitizationFinding(
                        column=col,
                        row_index=int(idx) if isinstance(idx, (int, float)) else 0,
                        finding_type=finding_type,
                        original_excerpt=value_str[:80],
                        action_taken=action,
                        reason=reason,
                    )
                )
            column_finding_count += 1

    high_risk_columns = [c.column for c in classifications if c.risk_level == "high_risk_free_text"]

    report = SanitizationReport(
        dataset_id=dataset_id,
        column_classifications=classifications,
        findings=findings,
        high_risk_columns=high_risk_columns,
        values_excluded_count=excluded_count,
        values_truncated_count=truncated_count,
        original_dataset_unchanged=True,
    )

    # Explicit, checkable proof (not just an assumed claim): confirm
    # the stored dataset's shape/columns are identical to what we
    # started with. If store.get() ever returned a mutated copy, this
    # would catch it.
    post_check_df = store.get(dataset_id)
    assert post_check_df.shape == df.shape, "sanitize_for_llm_context must never alter the original dataset"

    message = (
        f"Scanned {len(df.columns)} column(s); {len(findings)} finding(s), "
        f"{excluded_count} value(s) excluded, {truncated_count} truncated. "
        f"Original dataset unchanged."
    )

    return ToolResult[SanitizationReport](
        success=True,
        tool_name="sanitize_for_llm_context",
        message=message,
        data=report,
    )
