"""
M5: API tests for the dataset endpoints (app/api/routers/datasets.py).

Uses the real FastAPI app (via the `api_client` fixture in conftest.py)
and the real Telco CSV — no mocked HTTP layer, no mocked DatasetStore.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TELCO_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_customer_churn.csv"


def _telco_csv_bytes() -> bytes:
    return TELCO_CSV_PATH.read_bytes()


class TestUploadDataset:
    def test_upload_returns_201_with_dataset_id(self, api_client):
        response = api_client.post(
            "/datasets", files={"file": ("telco.csv", _telco_csv_bytes(), "text/csv")}
        )

        assert response.status_code == 201
        body = response.json()
        assert body["dataset_id"].startswith("dataset_")
        assert body["filename"] == "telco.csv"
        assert body["rows"] == 7043
        assert "Churn" in body["columns"]

    def test_uploaded_dataset_is_immediately_retrievable(self, api_client):
        upload = api_client.post(
            "/datasets", files={"file": ("telco.csv", _telco_csv_bytes(), "text/csv")}
        )
        dataset_id = upload.json()["dataset_id"]

        response = api_client.get(f"/datasets/{dataset_id}")

        assert response.status_code == 200
        assert response.json()["dataset_id"] == dataset_id
        assert response.json()["rows"] == 7043

    def test_unsupported_extension_rejected(self, api_client):
        response = api_client.post(
            "/datasets", files={"file": ("telco.txt", _telco_csv_bytes(), "text/plain")}
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_csv_upload_reports_detected_format_and_dimensions(self, api_client):
        """Multi-format ingestion surfaces what PIPER detected and how big
        the dataset is, before any run is started."""
        response = api_client.post(
            "/datasets", files={"file": ("telco.csv", _telco_csv_bytes(), "text/csv")}
        )

        body = response.json()
        assert body["detected_format"] == "csv"
        assert body["rows"] == 7043
        assert body["column_count"] == 21

    def test_malformed_csv_content_rejected(self, api_client):
        response = api_client.post(
            "/datasets", files={"file": ("bad.csv", b"\x00\x01\x02not,a,csv\xff\xfe", "text/csv")}
        )
        assert response.status_code == 400

    def test_empty_csv_rejected(self, api_client):
        response = api_client.post(
            "/datasets", files={"file": ("empty.csv", b"col_a,col_b\n", "text/csv")}
        )
        assert response.status_code == 422

    def test_oversized_upload_rejected(self, api_client, monkeypatch):
        """
        Batch 5 hardening: without MAX_UPLOAD_BYTES, an unbounded
        upload is read entirely into memory before any validation
        runs. Patches the cap down (rather than uploading a real 100MB
        file) so this test stays fast.
        """
        import app.api.routers.datasets as datasets_module

        monkeypatch.setattr(datasets_module, "MAX_UPLOAD_BYTES", 100)
        oversized_csv = b"col_a,col_b\n" + b"1,2\n" * 50  # > 100 bytes

        response = api_client.post(
            "/datasets", files={"file": ("big.csv", oversized_csv, "text/csv")}
        )

        assert response.status_code == 413


class TestUploadMultiFormat:
    """
    Every supported format must reach the SAME downstream state: stored
    under a dataset_id and immediately profileable via GET /datasets/{id}
    — the API-level proof that ingestion is the only format-aware stage.
    """

    @staticmethod
    def _frame():
        import pandas as pd

        return pd.DataFrame(
            {"feature_a": [1, 2, 3, 4], "feature_b": [0.5, 1.5, 2.5, 3.5], "target": ["Yes", "No", "Yes", "No"]}
        )

    def _payloads(self) -> dict[str, bytes]:
        import io
        import json as _json

        frame = self._frame()

        xlsx = io.BytesIO()
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="Data", index=False)

        parquet = io.BytesIO()
        frame.to_parquet(parquet, index=False)

        notebook = _json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["df"],
                        "outputs": [
                            {
                                "output_type": "execute_result",
                                "data": {"text/html": frame.to_html()},
                                "metadata": {},
                                "execution_count": 1,
                            }
                        ],
                        "metadata": {},
                        "execution_count": 1,
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ).encode()

        return {
            "d.csv": frame.to_csv(index=False).encode(),
            "d.tsv": frame.to_csv(index=False, sep="\t").encode(),
            "d.xlsx": xlsx.getvalue(),
            "d.json": frame.to_json(orient="records").encode(),
            "d.parquet": parquet.getvalue(),
            "d.ipynb": notebook,
        }

    def test_every_format_uploads_and_is_profileable(self, api_client):
        expected_format = {
            "d.csv": "csv",
            "d.tsv": "tsv",
            "d.xlsx": "excel",
            "d.json": "json",
            "d.parquet": "parquet",
            "d.ipynb": "ipynb",
        }

        for filename, raw in self._payloads().items():
            upload = api_client.post("/datasets", files={"file": (filename, raw, "application/octet-stream")})
            assert upload.status_code == 201, f"{filename}: {upload.text}"

            body = upload.json()
            assert body["detected_format"] == expected_format[filename], filename
            assert body["rows"] == 4, filename
            assert body["column_count"] == 3, filename

            # The same downstream profiling endpoint works identically
            # regardless of which format it came from.
            profile = api_client.get(f"/datasets/{body['dataset_id']}")
            assert profile.status_code == 200, filename
            assert profile.json()["rows"] == 4, filename
            assert profile.json()["columns"] == 3, filename

    def test_excel_multi_sheet_surfaces_sheets_and_allows_choosing_one(self, api_client):
        import io

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            self._frame().to_excel(writer, sheet_name="Primary", index=False)
            pd.DataFrame({"other": [7, 8]}).to_excel(writer, sheet_name="Secondary", index=False)
        raw = buffer.getvalue()

        default_upload = api_client.post("/datasets", files={"file": ("book.xlsx", raw, "application/octet-stream")})
        body = default_upload.json()
        assert body["sheet_name"] == "Primary"
        assert [s["name"] for s in body["available_sheets"]] == ["Primary", "Secondary"]
        assert any("2 worksheets" in note for note in body["notes"])

        chosen = api_client.post(
            "/datasets",
            files={"file": ("book.xlsx", raw, "application/octet-stream")},
            data={"sheet_name": "Secondary"},
        )
        assert chosen.status_code == 201
        assert chosen.json()["sheet_name"] == "Secondary"
        assert chosen.json()["columns"] == ["other"]

    def test_unknown_sheet_returns_422(self, api_client):
        import io

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            self._frame().to_excel(writer, sheet_name="Primary", index=False)

        response = api_client.post(
            "/datasets",
            files={"file": ("book.xlsx", buffer.getvalue(), "application/octet-stream")},
            data={"sheet_name": "Nope"},
        )
        assert response.status_code == 422

    def test_non_tabular_json_returns_422(self, api_client):
        response = api_client.post(
            "/datasets", files={"file": ("d.json", b'{"a": {"b": 1}}', "application/json")}
        )
        assert response.status_code == 422

    def test_malformed_json_returns_400(self, api_client):
        response = api_client.post(
            "/datasets", files={"file": ("d.json", b"{not json", "application/json")}
        )
        assert response.status_code == 400

    def test_malformed_parquet_returns_400(self, api_client):
        response = api_client.post(
            "/datasets", files={"file": ("d.parquet", b"nope", "application/octet-stream")}
        )
        assert response.status_code == 400

    def test_a_non_csv_upload_drives_a_complete_agent_run(self, api_client):
        """
        The highest-level proof of the locked "no separate downstream
        pipeline per format" constraint: a Parquet upload of the real
        Telco dataset must drive the FULL agent graph (plan -> clean ->
        train -> guardrails) to a completed, validated run, using the
        exact same endpoints a CSV upload uses.
        """
        import io

        telco = pd.read_csv(TELCO_CSV_PATH)
        buffer = io.BytesIO()
        telco.to_parquet(buffer, index=False)

        upload = api_client.post(
            "/datasets", files={"file": ("telco.parquet", buffer.getvalue(), "application/octet-stream")}
        )
        assert upload.status_code == 201
        assert upload.json()["detected_format"] == "parquet"
        assert upload.json()["rows"] == 7043

        create = api_client.post(
            "/runs", json={"dataset_id": upload.json()["dataset_id"], "target_column": "Churn"}
        )
        run_id = create.json()["run_id"]

        result = api_client.get(f"/runs/{run_id}/result").json()
        assert result["status"] == "completed"
        assert result["validation"]["valid"] is True
        assert result["comparison"] is not None

    def test_notebook_missing_external_source_returns_422_naming_the_file(self, api_client):
        import json as _json

        notebook = _json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["df = pd.read_csv('customers.csv')"],
                        "outputs": [],
                        "metadata": {},
                        "execution_count": 1,
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ).encode()

        response = api_client.post(
            "/datasets", files={"file": ("a.ipynb", notebook, "application/octet-stream")}
        )

        assert response.status_code == 422
        assert "customers.csv" in response.json()["detail"]


class TestListDatasets:
    def test_empty_store_returns_empty_list(self, api_client):
        response = api_client.get("/datasets")
        assert response.status_code == 200
        assert response.json() == {"dataset_ids": []}

    def test_lists_uploaded_datasets(self, api_client):
        upload_a = api_client.post(
            "/datasets", files={"file": ("a.csv", _telco_csv_bytes(), "text/csv")}
        )
        upload_b = api_client.post(
            "/datasets", files={"file": ("b.csv", _telco_csv_bytes(), "text/csv")}
        )

        response = api_client.get("/datasets")

        ids = set(response.json()["dataset_ids"])
        assert upload_a.json()["dataset_id"] in ids
        assert upload_b.json()["dataset_id"] in ids


class TestGetDataset:
    def test_returns_real_profile_shape(self, api_client):
        upload = api_client.post(
            "/datasets", files={"file": ("telco.csv", _telco_csv_bytes(), "text/csv")}
        )
        dataset_id = upload.json()["dataset_id"]

        response = api_client.get(f"/datasets/{dataset_id}")

        body = response.json()
        assert body["columns"] == 21
        assert len(body["column_profiles"]) == 21
        assert body["duplicate_rows"] >= 0

    def test_missing_dataset_returns_404(self, api_client):
        response = api_client.get("/datasets/dataset_does_not_exist")
        assert response.status_code == 404
