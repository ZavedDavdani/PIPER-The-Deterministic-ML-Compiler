"""
End-to-End Titanic Verification Script.
Executes a real run through the FastAPI application, verifies all stages,
artifacts, parity, governance, Student Mode, Engineer Mode, Test Flight, and Deployment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


def run_titanic_verification():
    print("=== Starting PIPER Titanic End-to-End Verification ===", flush=True)
    with TestClient(app) as client:
        # 1. Health check
        health_res = client.get("/health")
        print(f"1. Health check: status={health_res.status_code}, data={health_res.json()}", flush=True)
        assert health_res.status_code == 200

        # 2. Check provider settings
        prov_res = client.get("/settings/provider")
        print(f"2. Provider status: status={prov_res.status_code}, data={prov_res.json()}", flush=True)
        assert prov_res.status_code == 200

        # 3. Ingest Titanic dataset
        titanic_path = Path("../benchmark_data/train.csv")
        if not titanic_path.exists():
            titanic_path = Path("benchmark_data/train.csv")
        with open(titanic_path, "rb") as f:
            upload_res = client.post("/datasets", files={"file": ("titanic.csv", f, "text/csv")})
        print(f"3. Dataset Upload: status={upload_res.status_code}, data={upload_res.json()}", flush=True)
        assert upload_res.status_code == 201
        dataset_id = upload_res.json()["dataset_id"]

        # 4. Create Run
        run_req = {
            "dataset_id": dataset_id,
            "target_column": "Survived",
            "max_retries": 2,
        }
        create_res = client.post("/runs", json=run_req)
        print(f"4. Create Run: status={create_res.status_code}, data={create_res.json()}", flush=True)
        assert create_res.status_code == 202
        run_id = create_res.json()["run_id"]

        # 5. Poll run until completion
        print(f"5. Polling run '{run_id}' for completion...", flush=True)
        start_time = time.time()
        final_status = None
        while time.time() - start_time < 300:
            status_res = client.get(f"/runs/{run_id}")
            status_data = status_res.json()
            current_status = status_data.get("status")
            current_node = status_data.get("current_node")
            print(f"   [{int(time.time() - start_time)}s] status={current_status}, current_node={current_node}", flush=True)
            if current_status in ("completed", "failed"):
                final_status = status_data
                break
            time.sleep(3)

        assert final_status is not None, "Run timed out after 300s"
        print(f"   Final Status: {final_status['status']}", flush=True)
        if final_status["status"] == "failed":
            print(f"   Failure detail: {final_status.get('failure')}", flush=True)
            raise RuntimeError(f"Run failed: {final_status.get('failure')}")

        # 6. Publish / Verify Artifacts & Parity
        print("6. Publishing / Verifying ML Artifacts & Strict Parity...", flush=True)
        publish_res = client.post(f"/runs/{run_id}/artifacts")
        print(f"   Publish Artifacts: status={publish_res.status_code}", flush=True)
        if publish_res.status_code != 201:
            print(f"   Detail: {publish_res.text}", flush=True)
        assert publish_res.status_code == 201
        artifact_status = publish_res.json()
        print(f"   Artifact Status: eligible={artifact_status.get('eligible')}, parity={artifact_status.get('parity')}", flush=True)
        assert artifact_status.get("parity", {}).get("verified") is True, "Artifact parity verification failed"

        # 7. Verify Governance Bundle
        print("7. Verifying Governance Evidence...", flush=True)
        gov_res = client.get(f"/runs/{run_id}/governance")
        print(f"   Governance: status={gov_res.status_code}", flush=True)
        assert gov_res.status_code == 200
        gov_data = gov_res.json()
        print(f"   Model Card generated: {bool(gov_data.get('model_card'))}", flush=True)
        print(f"   Data Card generated: {bool(gov_data.get('data_card'))}", flush=True)
        print(f"   Fingerprints generated: {bool(gov_data.get('fingerprints'))}", flush=True)
        print(f"   Feature Importance generated: {bool(gov_data.get('feature_importance'))}", flush=True)

        # 8. Verify Test Flight Unseen Prediction
        print("8. Verifying Test Flight inference...", flush=True)
        test_rows = [
            {"PassengerId": 892, "Pclass": 3, "Name": "Kelly, Mr. James", "Sex": "male", "Age": 34.5, "SibSp": 0, "Parch": 0, "Ticket": "330911", "Fare": 7.8292, "Cabin": None, "Embarked": "Q"},
            {"PassengerId": 893, "Pclass": 1, "Name": "Wilkes, Mrs. James", "Sex": "female", "Age": 47.0, "SibSp": 1, "Parch": 0, "Ticket": "363272", "Fare": 7.0, "Cabin": None, "Embarked": "S"},
            {"PassengerId": 894, "Pclass": 2, "Name": "Myles, Mr. Thomas Francis", "Sex": "male", "Age": 62.0, "SibSp": 0, "Parch": 0, "Ticket": "240276", "Fare": 9.6875, "Cabin": None, "Embarked": "Q"},
        ]
        predict_res = client.post("/predict", json={"run_id": run_id, "rows": test_rows})
        print(f"   Predict: status={predict_res.status_code}, data={predict_res.json()}", flush=True)
        assert predict_res.status_code == 200
        predictions = predict_res.json().get("predictions")
        print(f"   Predictions on unseen rows: {predictions}", flush=True)
        assert len(predictions) == 3

        # 9. Verify Student Mode Endpoints
        print("9. Verifying Student Mode endpoints...", flush=True)
        journey_res = client.get(f"/runs/{run_id}/learn/journey")
        print(f"   Student Journey: status={journey_res.status_code}, stages={len(journey_res.json().get('stages', []))}", flush=True)
        assert journey_res.status_code == 200
        assert len(journey_res.json().get("stages", [])) >= 10

        explain_res = client.get(f"/runs/{run_id}/learn/explain?level=beginner")
        print(f"   Student Explain: status={explain_res.status_code}, summary length={len(explain_res.json().get('summary', ''))}", flush=True)
        assert explain_res.status_code == 200

        pipeline_res = client.get(f"/runs/{run_id}/learn/pipeline")
        print(f"   Student Pipeline: status={pipeline_res.status_code}, steps={len(pipeline_res.json().get('steps', []))}", flush=True)
        assert pipeline_res.status_code == 200

        # 10. Verify Engineer Mode Endpoints
        print("10. Verifying Engineer Mode endpoints...", flush=True)
        timeline_res = client.get(f"/runs/{run_id}/timeline")
        print(f"   Execution Timeline: status={timeline_res.status_code}, phases={len(timeline_res.json().get('phases', []))}", flush=True)
        assert timeline_res.status_code == 200

        verdict_res = client.get(f"/runs/{run_id}/verdict")
        print(f"   Verdict: status={verdict_res.status_code}, approved={verdict_res.json().get('approved')}", flush=True)
        assert verdict_res.status_code == 200

        evidence_res = client.get(f"/runs/{run_id}/evidence")
        print(f"   Evidence: status={evidence_res.status_code}, run_id={evidence_res.json().get('run_id')}", flush=True)
        assert evidence_res.status_code == 200

        # 11. Verify Deployment Readiness & Package
        print("11. Verifying Deployment endpoints...", flush=True)
        deploy_res = client.get(f"/runs/{run_id}/deployment")
        print(f"   Deployment Readiness: status={deploy_res.status_code}, readiness={deploy_res.json().get('status')}", flush=True)
        assert deploy_res.status_code == 200

        pkg_res = client.post(f"/runs/{run_id}/deployment/package")
        print(f"   Deployment Package: status={pkg_res.status_code}, files={pkg_res.json().get('files')}", flush=True)
        assert pkg_res.status_code == 201

        # 12. Run summary & result
        summary_res = client.get(f"/runs/{run_id}/summary")
        assert summary_res.status_code == 200
        summary_data = summary_res.json()

        result_res = client.get(f"/runs/{run_id}/result")
        assert result_res.status_code == 200
        result_data = result_res.json()

        summary_result = {
            "run_id": run_id,
            "status": final_status["status"],
            "dataset": "Titanic (train.csv)",
            "target": "Survived",
            "winning_model": summary_data.get("winning_model"),
            "candidate_models": summary_data.get("candidate_models"),
            "baseline": result_data.get("baseline"),
            "parity_verified": artifact_status.get("parity", {}).get("verified"),
            "parity_max_abs_diff": artifact_status.get("parity", {}).get("max_abs_diff"),
            "test_flight_predictions": predictions,
            "stages_count": len(journey_res.json().get("stages", [])),
            "timeline_phases": len(timeline_res.json().get("phases", [])),
            "deployment_status": deploy_res.json().get("status"),
            "deployment_files": pkg_res.json().get("files"),
            "verdict": verdict_res.json(),
        }

        out_file = Path("artifacts/titanic_demo_verified.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(summary_result, indent=2), encoding="utf-8")
        print(f"=== Verification Complete! Saved to {out_file} ===", flush=True)
        print(json.dumps(summary_result, indent=2), flush=True)


if __name__ == "__main__":
    run_titanic_verification()
