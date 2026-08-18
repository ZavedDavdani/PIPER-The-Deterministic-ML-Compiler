"""
Schema-layer tests: valid construction and strict-validation rejection.

These mirror the ad-hoc verification already done during schema
development, formalized as real pytest assertions.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    ColumnProfile,
    DatasetProfile,
    ImputationResult,
    MissingValueColumnEntry,
)


class TestValidConstruction:
    def test_dataset_profile_matches_telco_shape(self):
        profile = DatasetProfile(
            dataset_id="dataset_001",
            rows=7043,
            columns=21,
            duplicate_rows=0,
            memory_usage_bytes=970457,
            column_profiles=[
                ColumnProfile(
                    name="customerID",
                    dtype="object",
                    missing_count=0,
                    missing_percentage=0.0,
                    unique_count=7043,
                    unique_percentage=100.0,
                    sample_values=["7590-VHVEG"],
                ),
            ],
        )
        assert profile.rows == 7043
        assert profile.column_profiles[0].unique_percentage == 100.0

    def test_tool_result_is_json_serializable(self):
        from app.schemas import ToolResult

        profile = DatasetProfile(
            dataset_id="d1", rows=1, columns=1, duplicate_rows=0,
            memory_usage_bytes=1, column_profiles=[],
        )
        result = ToolResult[DatasetProfile](
            success=True, tool_name="profile_dataset", message="ok", data=profile,
        )
        # Must not raise.
        json_str = result.model_dump_json()
        assert "dataset_id" in json_str


class TestStrictValidationRejectsMalformedData:
    def test_negative_missing_count_rejected(self):
        with pytest.raises(ValidationError):
            ColumnProfile(
                name="x", dtype="int64",
                missing_count=-1, missing_percentage=0.0,
                unique_count=5, unique_percentage=50.0,
            )

    def test_percentage_over_100_rejected(self):
        with pytest.raises(ValidationError):
            MissingValueColumnEntry(column="x", count=5, percentage=150.0)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ImputationResult(
                dataset_id="d1", column="TotalCharges", strategy="median",
                missing_before=11, missing_after=0, replacement_value="70.35",
                made_up_field="not allowed",
            )

    def test_invalid_strategy_literal_rejected(self):
        with pytest.raises(ValidationError):
            ImputationResult(
                dataset_id="d1", column="TotalCharges", strategy="guess_randomly",
                missing_before=11, missing_after=0, replacement_value="70.35",
            )

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            DatasetProfile(dataset_id="d1", rows=100, columns=5)
