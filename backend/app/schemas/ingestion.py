"""
Multi-format dataset ingestion contract.

PIPER's downstream pipeline (profiling -> cleaning -> feature
engineering -> split -> train -> evaluate -> guardrails) operates on ONE
representation only: a pandas DataFrame behind a `dataset_id` in
DatasetStore. Ingestion's entire job is to normalize every supported
input format into exactly that — never to introduce a second, parallel
pipeline for "the Excel path" or "the JSON path".

IngestionResult is therefore deliberately thin: the DataFrame itself
does NOT live here (it goes straight into DatasetStore, exactly as the
CSV path always did). What lives here is only the *evidence* about how
that DataFrame was obtained — which format was detected, how, and what
had to be decided along the way (e.g. which Excel sheet was used when
the workbook had several). That evidence is what the API surfaces to
the user before a run starts, per the locked "show detected format and
dataset dimensions" requirement.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

DatasetFormat = Literal["csv", "tsv", "excel", "json", "ipynb", "parquet"]
"""
Every format PIPER can ingest. Deliberately a closed Literal — an
unknown extension is rejected with a structured error naming exactly
these, never silently guessed at.
"""

FORMAT_EXTENSIONS: dict[str, DatasetFormat] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    ".json": "json",
    ".ipynb": "ipynb",
    ".parquet": "parquet",
    ".pq": "parquet",
}
"""
Extension -> format. Detection is extension-driven FIRST (it is what the
user explicitly told us), with a content-based sanity check inside each
reader — a file named .csv that is actually a Parquet binary fails in
the CSV reader with a clear parse error, which is the correct outcome:
we never silently override what the filename claims.
"""


class SheetInfo(BaseModel):
    """One worksheet in a multi-sheet Excel workbook."""

    model_config = ConfigDict(extra="forbid")

    name: str
    rows: int = Field(..., ge=0)
    columns: int = Field(..., ge=0)


class IngestionResult(BaseModel):
    """
    Evidence about how a DataFrame was obtained from an uploaded file.
    Never carries the DataFrame itself — see the module docstring.
    """

    model_config = ConfigDict(extra="forbid")

    detected_format: DatasetFormat
    source_filename: str
    rows: int = Field(..., ge=0)
    columns: list[str]

    sheet_name: Optional[str] = Field(
        default=None,
        description="Excel only: which worksheet was actually ingested.",
    )
    available_sheets: list[SheetInfo] = Field(
        default_factory=list,
        description=(
            "Excel only: every worksheet found, so a user who got the "
            "wrong one can see what else was available and re-upload "
            "naming the sheet they wanted."
        ),
    )

    notes: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable decisions made during ingestion that the "
            "user should know about — e.g. 'workbook had 3 sheets; "
            "ingested the first non-empty one', or which notebook "
            "variable a DataFrame was recovered from."
        ),
    )
