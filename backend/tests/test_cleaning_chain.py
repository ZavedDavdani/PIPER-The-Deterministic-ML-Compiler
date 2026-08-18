"""
THE M1 ACCEPTANCE TEST.

    TotalCharges
        |
    11 blank strings
        |
    convert_column_type()
        |
    11 NaN
        |
    detect_missing_values()
        |
    11 missing
        |
    impute_missing_values()
        |
    0 missing

This is a single connected chain, not isolated tool calls — each step
operates on the *result* of the previous step via the shared
InMemoryDatasetStore, exactly the way the agent will use these tools
once LangGraph exists (M2+). If this test passes, M1's deterministic
foundation is proven to actually compose correctly, not just work in
isolation.
"""

from __future__ import annotations

import pandas as pd

from app.agent.tools import (
    convert_column_type,
    detect_missing_values,
    impute_missing_values,
)
from app.storage import InMemoryDatasetStore


def test_total_charges_conversion_missing_detection_imputation_chain(
    store: InMemoryDatasetStore, telco_df: pd.DataFrame
):
    store.save("dataset_chain", telco_df)

    # Step 0 — baseline: TotalCharges is text, blanks are not yet NaN.
    baseline = detect_missing_values("dataset_chain", store)
    assert baseline.success is True
    assert baseline.data.total_missing == 0

    # Step 1 — convert_column_type(TotalCharges, numeric).
    conversion = convert_column_type("dataset_chain", "TotalCharges", "numeric", store)
    assert conversion.success is True
    assert conversion.data.failed_conversion_count == 11
    assert conversion.data.converted_count == 7032
    assert conversion.data.applied is True  # 11/7043 ≈ 0.16% << 10% threshold

    # Step 2 — the 11 failed conversions must now be visible as real NaN.
    after_conversion = detect_missing_values("dataset_chain", store)
    assert after_conversion.success is True
    assert after_conversion.data.total_missing == 11
    tc_entry = next(
        e for e in after_conversion.data.columns_with_missing if e.column == "TotalCharges"
    )
    assert tc_entry.count == 11

    # Step 3 — impute_missing_values(TotalCharges, median).
    imputation = impute_missing_values("dataset_chain", "TotalCharges", "median", store)
    assert imputation.success is True
    assert imputation.data.missing_before == 11
    assert imputation.data.missing_after == 0

    # Step 4 — final state: zero missing anywhere, TotalCharges is
    # genuinely numeric now, not just NaN-free text.
    final_check = detect_missing_values("dataset_chain", store)
    assert final_check.data.total_missing == 0

    final_df = store.get("dataset_chain")
    assert pd.api.types.is_numeric_dtype(final_df["TotalCharges"])


def test_chain_is_order_dependent_imputation_before_conversion_fails_to_find_anything():
    """
    Sanity check on the chain's logic, not just its happy path: if you
    tried to impute TotalCharges *before* converting it, there would be
    nothing to impute, because the blanks are strings, not NaN. This
    confirms detect_missing_values() genuinely depends on the prior
    conversion step rather than coincidentally passing.
    """
    import pytest

    store = InMemoryDatasetStore()
    from pathlib import Path

    csv_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_customer_churn.csv"
    if not csv_path.exists():
        pytest.skip(f"Telco CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    store.save("dataset_order_check", df)

    # Attempt imputation BEFORE conversion — should fail with
    # no_missing_values, because pandas doesn't see the blank strings
    # as missing yet.
    premature = impute_missing_values("dataset_order_check", "TotalCharges", "median", store)
    assert premature.success is False
    assert premature.error.code == "no_missing_values"
