"""
Dataset endpoints (M5).

Thin HTTP wrapper around the EXISTING DatasetStore/profile_dataset()
core. All actual file parsing lives in the ingestion tool
(app/agent/tools/ingestion.py) — this router's only jobs are reading
multipart bytes (an unavoidably HTTP-layer concern) and mapping
structured ToolError codes onto HTTP status codes.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.agent.tools.ingestion import detect_format, ingest_dataset
from app.agent.tools.profiling import profile_dataset
from app.api.dependencies import get_dataset_store
from app.api.schemas import DatasetListResponse, DatasetUploadResponse
from app.schemas.profiling import DatasetProfile
from app.storage import DatasetStore

router = APIRouter(prefix="/datasets", tags=["datasets"])

# profile_dataset()'s ToolError.code values that mean "the dataset_id
# itself doesn't exist" (404) vs. "the dataset exists but is malformed"
# (422) — see app/agent/tools/profiling.py's own documented error
# contract for the exhaustive set this tool can ever return.
_NOT_FOUND_ERROR_CODES = {"dataset_not_found"}

_INGESTION_STATUS_CODES: dict[str, int] = {
    # 400 — the file itself is unusable/unreadable as the type it claims.
    "unsupported_format": 400,
    "parse_error": 400,
    # 422 — the file parsed fine, but its CONTENT can't form a dataset.
    "zero_columns": 422,
    "empty_dataset": 422,
    "unsupported_json_structure": 422,
    "empty_workbook": 422,
    "sheet_not_found": 422,
    "ipynb_no_tabular_output": 422,
    "ipynb_external_source_missing": 422,
    "ipynb_output_truncated": 422,
}
"""
Preserves the ORIGINAL CSV status-code contract exactly: an unparseable
file is 400 (as a malformed CSV always was) and a parsed-but-empty file
is 422 (as an empty CSV always was). Every new format's error codes are
slotted into whichever of those two categories they genuinely belong to.
"""

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
"""
Batch 5 hardening: without a cap, an uploaded file is read entirely
into memory (`await file.read()`) before any validation runs — an
unbounded upload is a genuine memory-exhaustion risk for a
single-process, in-memory-store, no-auth local/demo deployment (see
CLAUDE.md's Docker/architecture notes). 100MB comfortably covers any
realistic tabular CSV for this project's scope (the reference Telco
dataset is under 1MB) while still bounding worst-case memory use.
"""


def _new_dataset_id() -> str:
    return f"dataset_{uuid.uuid4().hex[:8]}"


@router.post("", response_model=DatasetUploadResponse, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(
        default=None,
        description="Excel only: ingest this specific worksheet instead of the first non-empty one.",
    ),
    dataset_store: DatasetStore = Depends(get_dataset_store),
) -> DatasetUploadResponse:
    """
    Accepts CSV, TSV, Excel (.xlsx/.xlsm/.xls), JSON, Jupyter notebooks
    (.ipynb), and Parquet. Every format is normalized by the ingestion
    tool into the SAME DataFrame representation before being stored, so
    nothing downstream of this endpoint is format-aware.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required to detect the file format.")

    # Reject an unsupported extension BEFORE reading the body, so an
    # unusable upload never costs the memory (matching the spirit of
    # the MAX_UPLOAD_BYTES guard below).
    if detect_format(file.filename) is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{file.filename}'. Supported formats: "
                "CSV, TSV, Excel (.xlsx/.xlsm/.xls), JSON, Jupyter notebook (.ipynb), Parquet."
            ),
        )

    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum upload size ({MAX_UPLOAD_BYTES // (1024 * 1024)}MB).",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum upload size ({MAX_UPLOAD_BYTES // (1024 * 1024)}MB).",
        )

    dataset_id = _new_dataset_id()
    result = ingest_dataset(raw, file.filename, dataset_id, dataset_store, sheet_name=sheet_name)

    if not result.success:
        raise HTTPException(
            status_code=_INGESTION_STATUS_CODES.get(result.error.code, 400),
            detail=result.message,
        )

    ingestion = result.data
    return DatasetUploadResponse(
        dataset_id=dataset_id,
        filename=file.filename,
        rows=ingestion.rows,
        columns=ingestion.columns,
        detected_format=ingestion.detected_format,
        column_count=len(ingestion.columns),
        sheet_name=ingestion.sheet_name,
        available_sheets=ingestion.available_sheets,
        notes=ingestion.notes,
    )


@router.get("", response_model=DatasetListResponse)
def list_datasets(dataset_store: DatasetStore = Depends(get_dataset_store)) -> DatasetListResponse:
    return DatasetListResponse(dataset_ids=dataset_store.list_ids())


@router.get("/{dataset_id}", response_model=DatasetProfile)
def get_dataset(
    dataset_id: str, dataset_store: DatasetStore = Depends(get_dataset_store)
) -> DatasetProfile:
    """
    Returns the real DatasetProfile (profile_dataset(), the same tool
    PROFILE uses inside the agent graph) — a genuine preview, not a
    separate/duplicated summary shape.
    """
    result = profile_dataset(dataset_id, dataset_store)
    if not result.success:
        status_code = 404 if result.error.code in _NOT_FOUND_ERROR_CODES else 422
        raise HTTPException(status_code=status_code, detail=result.message)
    return result.data
