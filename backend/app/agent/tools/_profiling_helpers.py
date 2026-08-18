"""
Internal helpers shared by the profiling tools. Not part of the public
tool contract — nothing here is called directly by the agent.
"""

from __future__ import annotations

import pandas as pd

# Above this uniqueness ratio, a column is treated as an identifier
# rather than a genuine categorical feature. Used only for descriptive
# semantic_type labeling here — NOT the same as the >99% guardrail
# threshold that check_high_cardinality() will enforce later. This is
# purely informational for inspect_column().
IDENTIFIER_UNIQUENESS_THRESHOLD = 0.99


def is_numeric_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def infer_semantic_type(series: pd.Series, unique_percentage: float) -> str:
    """
    Best-effort descriptive label. This is observational only — it
    does not drive any automatic action (see profiling.py schema
    docstring: no recommended_action fields anywhere).
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series):
        return "numeric"
    if unique_percentage >= IDENTIFIER_UNIQUENESS_THRESHOLD * 100:
        return "identifier"
    # pandas can back text columns with either the legacy object dtype
    # or its newer native string dtype (str in 2.x+ with the string
    # backend enabled). is_string_dtype covers both, plus explicit
    # StringDtype; object is also checked directly since
    # is_string_dtype does not always catch generic object columns
    # containing non-string data.
    if (
        pd.api.types.is_string_dtype(series)
        or series.dtype == object
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return "categorical"
    return "unknown"


def numeric_stats(series: pd.Series) -> dict:
    """
    Returns min/max/mean/median/std for a numeric series, or an empty
    dict if the series is not numeric or has no non-null values.
    """
    if not is_numeric_dtype(series):
        return {}
    non_null = series.dropna()
    if non_null.empty:
        return {}
    return {
        "min": float(non_null.min()),
        "max": float(non_null.max()),
        "mean": float(non_null.mean()),
        "median": float(non_null.median()),
        "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
    }


def sample_values(series: pd.Series, n: int = 5) -> list:
    """Up to n non-null sample values, JSON-safe (no numpy scalar types)."""
    non_null = series.dropna()
    samples = non_null.head(n).tolist()
    # Convert numpy scalars (int64, float64, bool_) to native Python
    # types so these are always JSON-serializable without surprises.
    return [x.item() if hasattr(x, "item") else x for x in samples]
