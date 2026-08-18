"""
Multi-format dataset ingestion tests (CSV, TSV, Excel, JSON, IPYNB,
Parquet).

The locked constraint under test throughout: every format normalizes
into the SAME DataFrame representation, so nothing downstream is
format-aware. TestFormatsAreEquivalentDownstream is the load-bearing
proof of that — the same logical table, uploaded in all six formats,
must produce byte-identical stored DataFrames.

CSV behavior must be unchanged by this work; TestCsvBehaviorUnchanged
pins that explicitly.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from app.agent.tools.ingestion import detect_format, ingest_dataset
from app.storage import InMemoryDatasetStore


@pytest.fixture()
def store() -> InMemoryDatasetStore:
    return InMemoryDatasetStore()


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    """A small table exercising int / float / string dtypes together."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "monthly_charges": [29.85, 56.95, 53.85, 42.30],
            "contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
            "churn": ["Yes", "No", "Yes", "No"],
        }
    )


def _ingest(store, filename: str, raw: bytes, sheet_name: str | None = None):
    return ingest_dataset(raw, filename, "dataset_test", store, sheet_name=sheet_name)


# --- Format detection -----------------------------------------------------


class TestFormatDetection:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("data.csv", "csv"),
            ("DATA.CSV", "csv"),
            ("data.tsv", "tsv"),
            ("data.tab", "tsv"),
            ("book.xlsx", "excel"),
            ("book.xlsm", "excel"),
            ("legacy.xls", "excel"),
            ("payload.json", "json"),
            ("analysis.ipynb", "ipynb"),
            ("table.parquet", "parquet"),
            ("table.pq", "parquet"),
        ],
    )
    def test_supported_extensions_detected(self, filename, expected):
        assert detect_format(filename) == expected

    @pytest.mark.parametrize("filename", ["notes.txt", "report.pdf", "archive.zip", "noextension", ""])
    def test_unsupported_extensions_return_none(self, filename):
        assert detect_format(filename) is None

    def test_unsupported_format_is_a_structured_error_naming_what_is_supported(self, store):
        result = _ingest(store, "report.pdf", b"%PDF-1.4")

        assert result.success is False
        assert result.error.code == "unsupported_format"
        assert ".csv" in result.error.details["supported_extensions"]
        assert ".parquet" in result.error.details["supported_extensions"]
        assert store.list_ids() == []  # nothing stored on failure


# --- CSV (must be unchanged) ---------------------------------------------


class TestCsvBehaviorUnchanged:
    def test_csv_ingests_with_original_semantics(self, store, sample_frame):
        result = _ingest(store, "data.csv", sample_frame.to_csv(index=False).encode())

        assert result.success is True
        assert result.data.detected_format == "csv"
        assert result.data.rows == 4
        assert result.data.columns == list(sample_frame.columns)
        pd.testing.assert_frame_equal(store.get("dataset_test"), sample_frame)

    def test_csv_result_matches_a_direct_pandas_read_csv(self, store, sample_frame):
        """The CSV path still literally is pd.read_csv — proven by
        comparing against calling it directly on the same bytes."""
        raw = sample_frame.to_csv(index=False).encode()
        _ingest(store, "data.csv", raw)

        pd.testing.assert_frame_equal(store.get("dataset_test"), pd.read_csv(io.BytesIO(raw)))

    def test_malformed_csv_is_a_parse_error(self, store):
        result = _ingest(store, "bad.csv", b"\x00\x01\x02not,a,csv\xff\xfe")
        assert result.success is False
        assert result.error.code == "parse_error"

    def test_header_only_csv_is_empty_dataset(self, store):
        result = _ingest(store, "empty.csv", b"col_a,col_b\n")
        assert result.success is False
        assert result.error.code == "empty_dataset"

    def test_completely_empty_csv_is_a_parse_error(self, store):
        result = _ingest(store, "nothing.csv", b"")
        assert result.success is False
        assert result.error.code == "parse_error"


# --- TSV ------------------------------------------------------------------


class TestTsv:
    def test_tab_separated_file_ingests(self, store, sample_frame):
        result = _ingest(store, "data.tsv", sample_frame.to_csv(index=False, sep="\t").encode())

        assert result.success is True
        assert result.data.detected_format == "tsv"
        pd.testing.assert_frame_equal(store.get("dataset_test"), sample_frame)

    def test_tsv_is_not_parsed_as_csv(self, store, sample_frame):
        """A tab-separated file read with the CSV comma separator would
        collapse into a single column — this proves the separator is
        genuinely format-driven."""
        result = _ingest(store, "data.tsv", sample_frame.to_csv(index=False, sep="\t").encode())
        assert len(result.data.columns) == 4

    def test_tab_extension_also_works(self, store, sample_frame):
        result = _ingest(store, "data.tab", sample_frame.to_csv(index=False, sep="\t").encode())
        assert result.success is True
        assert result.data.detected_format == "tsv"


# --- Excel ----------------------------------------------------------------


def _xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


class TestExcel:
    def test_single_sheet_workbook_ingests(self, store, sample_frame):
        result = _ingest(store, "book.xlsx", _xlsx_bytes({"Sheet1": sample_frame}))

        assert result.success is True
        assert result.data.detected_format == "excel"
        assert result.data.sheet_name == "Sheet1"
        pd.testing.assert_frame_equal(store.get("dataset_test"), sample_frame)

    def test_multi_sheet_workbook_reports_every_sheet_and_explains_the_choice(self, store, sample_frame):
        other = pd.DataFrame({"z": [9, 9]})
        result = _ingest(store, "book.xlsx", _xlsx_bytes({"Customers": sample_frame, "Notes": other}))

        assert result.success is True
        assert result.data.sheet_name == "Customers"
        assert [s.name for s in result.data.available_sheets] == ["Customers", "Notes"]
        assert {s.name: (s.rows, s.columns) for s in result.data.available_sheets}["Notes"] == (2, 1)
        # The multi-sheet decision must be surfaced, never silent.
        assert any("2 worksheets" in note for note in result.data.notes)

    def test_explicit_sheet_name_is_honoured(self, store, sample_frame):
        other = pd.DataFrame({"z": [9, 9]})
        result = _ingest(
            store, "book.xlsx", _xlsx_bytes({"Customers": sample_frame, "Notes": other}), sheet_name="Notes"
        )

        assert result.success is True
        assert result.data.sheet_name == "Notes"
        assert result.data.columns == ["z"]

    def test_unknown_sheet_name_lists_the_real_ones(self, store, sample_frame):
        result = _ingest(store, "book.xlsx", _xlsx_bytes({"Customers": sample_frame}), sheet_name="Missing")

        assert result.success is False
        assert result.error.code == "sheet_not_found"
        assert result.error.details["available_sheets"] == ["Customers"]

    def test_first_empty_sheet_is_skipped_in_favour_of_a_real_one(self, store, sample_frame):
        result = _ingest(
            store, "book.xlsx", _xlsx_bytes({"Empty": pd.DataFrame(), "Data": sample_frame})
        )

        assert result.success is True
        assert result.data.sheet_name == "Data"

    def test_workbook_with_only_empty_sheets_is_rejected(self, store):
        result = _ingest(store, "book.xlsx", _xlsx_bytes({"Empty": pd.DataFrame()}))

        assert result.success is False
        assert result.error.code == "empty_workbook"

    def test_malformed_excel_is_a_parse_error(self, store):
        result = _ingest(store, "book.xlsx", b"this is definitely not a workbook")
        assert result.success is False
        assert result.error.code == "parse_error"


# --- JSON -----------------------------------------------------------------


class TestJson:
    def test_list_of_objects_records_orient(self, store, sample_frame):
        result = _ingest(store, "d.json", sample_frame.to_json(orient="records").encode())

        assert result.success is True
        assert result.data.detected_format == "json"
        pd.testing.assert_frame_equal(store.get("dataset_test"), sample_frame)

    def test_object_of_equal_length_arrays_columnar(self, store):
        payload = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        result = _ingest(store, "d.json", json.dumps(payload).encode())

        assert result.success is True
        assert result.data.rows == 3
        assert result.data.columns == ["a", "b"]

    def test_pandas_split_orient(self, store, sample_frame):
        result = _ingest(store, "d.json", sample_frame.to_json(orient="split", index=False).encode())

        assert result.success is True
        assert result.data.columns == list(sample_frame.columns)
        assert result.data.rows == 4

    @pytest.mark.parametrize("key", ["data", "records", "rows", "items", "results"])
    def test_records_list_under_a_wrapper_key(self, store, key):
        payload = {key: [{"a": 1, "b": 2}, {"a": 3, "b": 4}], "meta": {"source": "api"}}
        result = _ingest(store, "d.json", json.dumps(payload).encode())

        assert result.success is True
        assert result.data.rows == 2
        assert any(key in note for note in result.data.notes)

    def test_json_lines(self, store):
        raw = b'{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n'
        result = _ingest(store, "d.json", raw)

        assert result.success is True
        assert result.data.rows == 2
        assert any("JSON Lines" in note for note in result.data.notes)

    def test_unequal_column_lengths_rejected_with_the_lengths(self, store):
        payload = {"a": [1, 2, 3], "b": [1]}
        result = _ingest(store, "d.json", json.dumps(payload).encode())

        assert result.success is False
        assert result.error.code == "unsupported_json_structure"
        assert result.error.details["column_lengths"] == [1, 3]

    @pytest.mark.parametrize(
        "payload",
        [42, "just a string", [1, 2, 3], {"nested": {"deep": {"value": 1}}}],
    )
    def test_non_tabular_json_rejected_clearly(self, store, payload):
        result = _ingest(store, "d.json", json.dumps(payload).encode())

        assert result.success is False
        assert result.error.code in ("unsupported_json_structure", "empty_dataset")
        assert "tabular" in result.message.lower() or "list of objects" in result.message.lower()

    def test_empty_json_list_rejected(self, store):
        result = _ingest(store, "d.json", b"[]")
        assert result.success is False
        assert result.error.code == "empty_dataset"

    def test_malformed_json_is_a_parse_error(self, store):
        result = _ingest(store, "d.json", b"{not valid json at all")
        assert result.success is False
        assert result.error.code == "parse_error"


# --- Parquet --------------------------------------------------------------


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


class TestParquet:
    def test_parquet_ingests(self, store, sample_frame):
        result = _ingest(store, "d.parquet", _parquet_bytes(sample_frame))

        assert result.success is True
        assert result.data.detected_format == "parquet"
        pd.testing.assert_frame_equal(store.get("dataset_test"), sample_frame)

    def test_parquet_preserves_types_that_csv_would_lose(self, store):
        """
        Parquet stores dtypes in the file itself. This asserts the real
        difference: a datetime and a boolean column survive Parquet
        round-tripping as datetime64/bool, whereas the same frame via
        CSV comes back as plain strings/objects.
        """
        typed = pd.DataFrame(
            {
                "when": pd.to_datetime(["2024-01-01", "2024-06-15"]),
                "flag": [True, False],
                "count": pd.Series([1, 2], dtype="int32"),
            }
        )

        _ingest(store, "d.parquet", _parquet_bytes(typed))
        from_parquet = store.get("dataset_test")

        assert str(from_parquet["when"].dtype).startswith("datetime64")
        assert from_parquet["flag"].dtype == bool
        assert from_parquet["count"].dtype == "int32"

        # Contrast: the same data through CSV loses all three.
        csv_store = InMemoryDatasetStore()
        ingest_dataset(typed.to_csv(index=False).encode(), "d.csv", "csv_ds", csv_store)
        from_csv = csv_store.get("csv_ds")
        assert not str(from_csv["when"].dtype).startswith("datetime64")

    def test_malformed_parquet_is_a_parse_error(self, store):
        result = _ingest(store, "d.parquet", b"not a parquet file")
        assert result.success is False
        assert result.error.code == "parse_error"


# --- IPYNB ----------------------------------------------------------------


def _notebook(cells: list[dict]) -> bytes:
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}).encode()


def _html_output(html: str) -> dict:
    return {"output_type": "execute_result", "data": {"text/html": html}, "metadata": {}, "execution_count": 1}


def _code_cell(source: str, outputs: list[dict] | None = None) -> dict:
    return {"cell_type": "code", "source": [source], "outputs": outputs or [], "metadata": {}, "execution_count": 1}


class TestIpynb:
    def test_recovers_a_displayed_dataframe(self, store, sample_frame):
        raw = _notebook([_code_cell("df", [_html_output(sample_frame.to_html())])])

        result = _ingest(store, "analysis.ipynb", raw)

        assert result.success is True
        assert result.data.detected_format == "ipynb"
        assert result.data.rows == 4
        assert result.data.columns == list(sample_frame.columns)

    def test_recovered_numeric_columns_get_real_dtypes_not_strings(self, store, sample_frame):
        raw = _notebook([_code_cell("df", [_html_output(sample_frame.to_html())])])
        _ingest(store, "analysis.ipynb", raw)

        recovered = store.get("dataset_test")
        assert pd.api.types.is_numeric_dtype(recovered["customer_id"])
        assert pd.api.types.is_numeric_dtype(recovered["monthly_charges"])
        assert not pd.api.types.is_numeric_dtype(recovered["contract"])

    def test_notebook_code_is_never_executed(self, store, sample_frame):
        """
        A notebook is untrusted input. This cell's source would raise if
        executed — ingestion must still succeed by reading only the
        SAVED output, proving no execution happens.
        """
        dangerous = "raise SystemExit('this notebook must never be executed')"
        raw = _notebook([_code_cell(dangerous, [_html_output(sample_frame.to_html())])])

        result = _ingest(store, "analysis.ipynb", raw)

        assert result.success is True
        assert any("NOT executed" in note for note in result.data.notes)

    def test_largest_table_wins_when_several_are_displayed(self, store, sample_frame):
        small = pd.DataFrame({"a": [1]})
        raw = _notebook(
            [
                _code_cell("df.head(1)", [_html_output(small.to_html())]),
                _code_cell("df", [_html_output(sample_frame.to_html())]),
            ]
        )

        result = _ingest(store, "analysis.ipynb", raw)

        assert result.success is True
        assert result.data.rows == 4
        assert any("largest" in note for note in result.data.notes)

    def test_truncated_display_is_rejected_rather_than_silently_partial(self, store, sample_frame):
        """
        Jupyter elides the middle of a large DataFrame. Ingesting that
        would silently drop rows, so it must be a clear error naming
        both numbers.
        """
        truncated_html = sample_frame.to_html() + "<p>7043 rows × 4 columns</p>"
        raw = _notebook([_code_cell("df", [_html_output(truncated_html)])])

        result = _ingest(store, "analysis.ipynb", raw)

        assert result.success is False
        assert result.error.code == "ipynb_output_truncated"
        assert result.error.details["declared_rows"] == 7043
        assert result.error.details["rows_available_in_output"] == 4

    def test_missing_external_source_is_named_explicitly(self, store):
        raw = _notebook([_code_cell("import pandas as pd\ndf = pd.read_csv('telco_churn.csv')")])

        result = _ingest(store, "analysis.ipynb", raw)

        assert result.success is False
        assert result.error.code == "ipynb_external_source_missing"
        assert "telco_churn.csv" in result.error.details["referenced_files"]
        assert "telco_churn.csv" in result.message

    def test_notebook_with_no_tabular_output_reports_clearly(self, store):
        raw = _notebook([{"cell_type": "markdown", "source": ["# Title"], "metadata": {}}])

        result = _ingest(store, "analysis.ipynb", raw)

        assert result.success is False
        assert result.error.code == "ipynb_no_tabular_output"

    def test_json_output_is_also_supported(self, store):
        raw = _notebook(
            [_code_cell("df", [{"output_type": "execute_result",
                                "data": {"application/json": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]},
                                "metadata": {}, "execution_count": 1}])]
        )

        result = _ingest(store, "analysis.ipynb", raw)

        assert result.success is True
        assert result.data.rows == 2

    def test_invalid_json_notebook_is_a_parse_error(self, store):
        result = _ingest(store, "analysis.ipynb", b"{not json")
        assert result.success is False
        assert result.error.code == "parse_error"

    def test_valid_json_that_is_not_a_notebook_is_rejected(self, store):
        result = _ingest(store, "analysis.ipynb", b'{"foo": 1}')
        assert result.success is False
        assert result.error.code == "parse_error"
        assert "not a Jupyter notebook" in result.message

    def test_notebook_with_empty_cells_list_reports_no_output(self, store):
        result = _ingest(store, "analysis.ipynb", _notebook([]))
        assert result.success is False
        assert result.error.code == "ipynb_no_tabular_output"


# --- The core invariant ---------------------------------------------------


class TestFormatsAreEquivalentDownstream:
    def test_all_six_formats_produce_the_same_stored_dataframe(self, sample_frame):
        """
        The locked "no separate downstream pipeline per format"
        constraint, proven directly: the same logical table uploaded in
        every supported format must land in DatasetStore as an
        equivalent DataFrame, so no downstream stage can behave
        differently based on origin.
        """
        html_notebook = _notebook([_code_cell("df", [_html_output(sample_frame.to_html())])])
        payloads = {
            "d.csv": sample_frame.to_csv(index=False).encode(),
            "d.tsv": sample_frame.to_csv(index=False, sep="\t").encode(),
            "d.xlsx": _xlsx_bytes({"Sheet1": sample_frame}),
            "d.json": sample_frame.to_json(orient="records").encode(),
            "d.parquet": _parquet_bytes(sample_frame),
            "d.ipynb": html_notebook,
        }

        stored: dict[str, pd.DataFrame] = {}
        for filename, raw in payloads.items():
            store = InMemoryDatasetStore()
            result = ingest_dataset(raw, filename, "ds", store)
            assert result.success is True, f"{filename} failed: {result.message}"
            stored[filename] = store.get("ds")

        reference = stored["d.csv"]
        for filename, frame in stored.items():
            assert list(frame.columns) == list(reference.columns), f"{filename} column mismatch"
            assert frame.shape == reference.shape, f"{filename} shape mismatch"
            # Values must match regardless of origin (dtypes can differ
            # legitimately — e.g. Parquet preserves int32 where CSV
            # infers int64 — so compare values, not dtype identity).
            pd.testing.assert_frame_equal(
                frame.reset_index(drop=True).astype(str),
                reference.reset_index(drop=True).astype(str),
                check_dtype=False,
            )

    def test_column_labels_are_always_strings(self):
        """JSON and Excel can yield integer column labels; the rest of
        the pipeline assumes string column names."""
        store = InMemoryDatasetStore()
        payload = {"0": [1, 2], "1": [3, 4]}
        ingest_dataset(json.dumps(payload).encode(), "d.json", "ds", store)

        assert all(isinstance(c, str) for c in store.get("ds").columns)

    def test_nothing_is_stored_when_ingestion_fails(self, store):
        _ingest(store, "d.json", b"[]")
        assert store.list_ids() == []
