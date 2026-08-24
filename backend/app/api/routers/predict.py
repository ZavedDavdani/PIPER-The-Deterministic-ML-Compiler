"""POST /predict — score unseen rows with a VERIFIED PIPER artifact."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_artifact_dir, get_run_store
from app.deployment.errors import InferenceError
from app.deployment.predict import predict_unseen
from app.deployment.schema import rows_to_frame
from app.schemas.deployment import PredictRequest, PredictResponse
from app.storage.run_store import InMemoryRunStore

router = APIRouter(tags=["deployment"])

_STATUS = {
    "invalid_run_id": 404,
    "artifact_missing": 404,
    "missing_features": 422,
    "invalid_input": 422,
    "unsupported_file_type": 400,
    "file_too_large": 413,
}


def inference_http(exc: InferenceError) -> HTTPException:
    code = _STATUS.get(exc.code, 409)
    return HTTPException(
        status_code=code,
        detail={"code": exc.code, "message": exc.message, "details": exc.details},
    )


def require_known_run(run_id: str, run_store: InMemoryRunStore) -> None:
    if not run_store.exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")


def attach_sample(frame: pd.DataFrame, payload: dict, limit: int = 8) -> PredictResponse:
    sample = []
    for idx, pred in enumerate(payload["predictions"][:limit]):
        row = frame.iloc[idx].where(pd.notna(frame.iloc[idx]), None).to_dict()
        row["prediction"] = pred
        sample.append(row)
    return PredictResponse(**payload, sample=sample)


@router.post("/predict", response_model=PredictResponse)
def post_predict(
    body: PredictRequest,
    run_store: InMemoryRunStore = Depends(get_run_store),
    artifact_dir: Path = Depends(get_artifact_dir),
) -> PredictResponse:
    require_known_run(body.run_id, run_store)
    try:
        frame = rows_to_frame(body.rows)
        payload = predict_unseen(artifact_dir, body.run_id, frame)
        return attach_sample(frame, payload)
    except InferenceError as exc:
        raise inference_http(exc) from exc
