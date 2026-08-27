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

        # 2. Configure provider to Gemini
        prov_update = client.put("/settings/provider", json={"provider": "gemini", "model": "gemini-3.6-flash"})
        print(f"2. Provider configured: status={prov_update.status_code}, data={prov_update.json()}", flush=True)
        assert prov_update.status_code == 200
        assert prov_update.json()["provider"] == "gemini"
        assert prov_update.json()["model"] == "gemini-3.6-flash"
        assert prov_update.json()["reachable"] is True

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
        print(f"   Artifact Status: artifact_status={artifact_status.get('artifact_status')}, parity_status={artifact_status.get('parity_status')}", flush=True)
        assert artifact_status.get("artifact_status") == "VERIFIED", "Artifact status is not VERIFIED"
        assert artifact_status.get("parity_status") == "passed", "Parity status is not passed"

        # 6b. Verify Required Artifact Bundle Files
        files_res = client.get(f"/runs/{run_id}/artifacts/files")
        assert files_res.status_code == 200
        published_files = files_res.json().get("files", [])
        print(f"   Published Artifact Files: {published_files}", flush=True)
        for expected_file in (
            "pipeline.joblib",
            "pipeline.py",
            "training_reproduction.ipynb",
            "manifest.json",
            "evidence.json",
            "requirements.txt",
            "hashes.json",
        ):
            assert expected_file in published_files, f"Missing expected artifact file: {expected_file}"
            file_dl = client.get(f"/runs/{run_id}/artifacts/files/{expected_file}")
            assert file_dl.status_code == 200, f"Failed to download {expected_file}"
            assert len(file_dl.content) > 0, f"Empty artifact file: {expected_file}"

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

        # 9. Verify What-If Controlled Experiments Sandbox
        print("9. Verifying What-If Controlled Experiments Sandbox...", flush=True)
        result_before_whatif = client.get(f"/runs/{run_id}/result").json()
        artifact_before_whatif = client.get(f"/runs/{run_id}/artifacts").json()
        winning_model_id = artifact_status.get("winning_model_id") or result_before_whatif.get("comparison", {}).get("recommended_model_id")

        # 9a. Test invalid hyperparameter rejection
        invalid_whatif = client.post(
            f"/runs/{run_id}/explore",
            json={
                "base_model_id": winning_model_id,
                "hyperparameter_name": "n_estimators",
                "hyperparameter_value": 0.1,
            },
        )
        print(f"   Invalid What-If status (expected 400): {invalid_whatif.status_code}", flush=True)
        assert invalid_whatif.status_code == 400

        # 9b. Test valid What-If hyperparameter execution
        valid_whatif = client.post(
            f"/runs/{run_id}/explore",
            json={
                "base_model_id": winning_model_id,
                "hyperparameter_name": "n_estimators",
                "hyperparameter_value": 100,
            },
        )
        print(f"   Valid What-If status (expected 201): {valid_whatif.status_code}", flush=True)
        assert valid_whatif.status_code == 201
        whatif_data = valid_whatif.json()
        print(f"   What-If Experiment ID: {whatif_data.get('experiment_id')}", flush=True)
        print(f"   What-If F1: {whatif_data.get('evaluation', {}).get('f1')}", flush=True)
        assert whatif_data.get("experiment_id", "").startswith("exp_")

        # 9c. Verify Base Run & Verified Artifact Isolation
        result_after_whatif = client.get(f"/runs/{run_id}/result").json()
        artifact_after_whatif = client.get(f"/runs/{run_id}/artifacts").json()
        assert result_before_whatif == result_after_whatif, "Original run state was mutated by What-If experiment!"
        assert artifact_before_whatif == artifact_after_whatif, "Original artifact was mutated by What-If experiment!"
        print("   What-If isolation verified: base run and verified artifact remain 100% unchanged.", flush=True)

        # 10. Verify Student Mode Endpoints
        print("10. Verifying Student Mode endpoints...", flush=True)
        journey_res = client.get(f"/runs/{run_id}/learn/journey")
        print(f"   Student Journey: status={journey_res.status_code}, stages={len(journey_res.json().get('stages', []))}", flush=True)
        assert journey_res.status_code == 200
        assert len(journey_res.json().get("stages", [])) >= 10

        explain_res = client.get(f"/runs/{run_id}/learn/explanation?level=beginner")
        print(f"   Student Explain: status={explain_res.status_code}", flush=True)
        assert explain_res.status_code == 200

        pipeline_res = client.get(f"/runs/{run_id}/learn/pipeline")
        print(f"   Student Pipeline: status={pipeline_res.status_code}, steps={len(pipeline_res.json().get('nodes', []))}", flush=True)
        assert pipeline_res.status_code == 200

        # 11. Verify Engineer Mode Endpoints
        print("11. Verifying Engineer Mode endpoints...", flush=True)
        timeline_res = client.get(f"/runs/{run_id}/timeline")
        print(f"   Execution Timeline: status={timeline_res.status_code}, phases={len(timeline_res.json().get('phases', []))}", flush=True)
        assert timeline_res.status_code == 200

        verdict_res = client.get(f"/runs/{run_id}/verdict")
        print(f"   Verdict: status={verdict_res.status_code}, approved={verdict_res.json().get('approved')}", flush=True)
        assert verdict_res.status_code == 200

        evidence_res = client.get(f"/runs/{run_id}/evidence")
        print(f"   Evidence: status={evidence_res.status_code}, run_id={evidence_res.json().get('run_id')}", flush=True)
        assert evidence_res.status_code == 200

        # 12. Verify Deployment Readiness & Package
        print("12. Verifying Deployment endpoints...", flush=True)
        deploy_res = client.get(f"/runs/{run_id}/deployment")
        print(f"   Deployment Readiness: status={deploy_res.status_code}, readiness={deploy_res.json().get('status')}", flush=True)
        assert deploy_res.status_code == 200

        pkg_res = client.post(f"/runs/{run_id}/deployment/package")
        print(f"   Deployment Package: status={pkg_res.status_code}, files={pkg_res.json().get('files')}", flush=True)
        assert pkg_res.status_code == 201

        # 13. Run summary & result
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
            "artifact_status": artifact_status.get("artifact_status"),
            "parity_status": artifact_status.get("parity_status"),
            "artifact_files": published_files,
            "test_flight_predictions": predictions,
            "what_if_experiment": {
                "experiment_id": whatif_data.get("experiment_id"),
                "f1": whatif_data.get("evaluation", {}).get("f1"),
                "variable": whatif_data.get("variable_changed"),
            },
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
