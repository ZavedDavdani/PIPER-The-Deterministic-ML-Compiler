"""
Multi-format dataset ingestion: CSV, TSV, Excel, JSON, IPYNB, Parquet.

    uploaded bytes + filename
        |
        v
    _detect_format()            -- extension-driven, closed allowlist
        |
        v
    per-format reader           -- the ONLY format-aware code in PIPER
        |
        v
    one pandas DataFrame        -- normalized representation
        |
        v
    DatasetStore.save(dataset_id, df)
        |
        v
    [everything downstream is byte-for-byte the same for every format]

The locked constraint this module exists to satisfy: **no separate
downstream pipeline per format.** Format-awareness stops here. Once a
DataFrame is in DatasetStore, profiling/cleaning/feature-engineering/
split/train/evaluate/guardrails cannot tell (and must never be able to
tell) whether it came from a CSV or a Parquet file.

CSV behavior is deliberately unchanged from the pre-existing
implementation: the same `pd.read_csv(io.BytesIO(raw))` call, the same
resulting DataFrame, the same empty/zero-column rejection. The CSV path
gained format *detection* around it, not new parsing behavior.

Errors are structured (IngestionError -> ToolError), never bare
exception strings, so the API layer can map them onto HTTP status codes
without string-matching.
"""

from __future__ import annotations

import io
import json
import re
from html.parser import HTMLParser
from typing import Optional

import pandas as pd

from app.schemas import ToolError, ToolResult
from app.schemas.ingestion import (
    FORMAT_EXTENSIONS,
    DatasetFormat,
    IngestionResult,
    SheetInfo,
)
from app.storage import DatasetStore

__all__ = ["ingest_dataset", "detect_format", "IngestionError"]


class IngestionError(Exception):
    """
    Internal, structured ingestion failure. Converted to a ToolError by
    ingest_dataset() — never allowed to escape this module as a raw
    exception, matching how every other tool group reports failure.
    """

    def __init__(self, code: str, message: str, details: Optional[dict] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)


def detect_format(filename: str) -> Optional[DatasetFormat]:
    """
    Extension-driven detection against the closed FORMAT_EXTENSIONS
    allowlist. Returns None for anything unsupported — the caller turns
    that into a structured `unsupported_format` error listing exactly
    what IS supported, rather than attempting to guess at content.
    """
    lowered = (filename or "").lower()
    for extension, fmt in FORMAT_EXTENSIONS.items():
        if lowered.endswith(extension):
            return fmt
    return None


# --- CSV / TSV ------------------------------------------------------------


def _read_delimited(raw: bytes, sep: str, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(raw), sep=sep)
    except Exception as e:
        raise IngestionError(
            code="parse_error",
            message=f"Could not parse the file as {label}: {e}",
            details={"format": label},
        ) from e


# --- Excel ----------------------------------------------------------------


def _read_excel(raw: bytes, requested_sheet: Optional[str]) -> tuple[pd.DataFrame, str, list[SheetInfo], list[str]]:
    """
    Reads every worksheet so the user can always be told what the
    workbook actually contained, then ingests exactly one of them.

    Sheet selection is explicit and reported, never silent:
      - an explicitly requested sheet is used (error if absent),
      - otherwise the FIRST NON-EMPTY sheet is used.
    A workbook whose sheets are all empty is a clear error, not an
    empty DataFrame handed downstream.
    """
    try:
        # sheet_name=None -> every sheet, as an ordered dict.
        # pandas picks the engine from the file's own magic bytes
        # (openpyxl for .xlsx/.xlsm, xlrd for legacy .xls).
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
    except Exception as e:
        raise IngestionError(
            code="parse_error",
            message=f"Could not parse the file as Excel: {e}",
            details={"format": "excel"},
        ) from e

    if not sheets:
        raise IngestionError(
            code="empty_workbook",
            message="The Excel workbook contains no worksheets.",
            details={"format": "excel"},
        )

    available = [
        SheetInfo(name=name, rows=int(frame.shape[0]), columns=int(frame.shape[1]))
        for name, frame in sheets.items()
    ]
    notes: list[str] = []

    if requested_sheet is not None:
        if requested_sheet not in sheets:
            raise IngestionError(
                code="sheet_not_found",
                message=(
                    f"Worksheet '{requested_sheet}' was not found. "
                    f"Available worksheets: {[s.name for s in available]}."
                ),
                details={"requested_sheet": requested_sheet, "available_sheets": [s.name for s in available]},
            )
        chosen_name = requested_sheet
        notes.append(f"Ingested the explicitly requested worksheet '{chosen_name}'.")
    else:
        non_empty = [name for name, frame in sheets.items() if frame.shape[0] > 0 and frame.shape[1] > 0]
        if not non_empty:
            raise IngestionError(
                code="empty_workbook",
                message=(
                    "Every worksheet in this workbook is empty "
                    f"({[s.name for s in available]})."
                ),
                details={"available_sheets": [s.name for s in available]},
            )
        chosen_name = non_empty[0]
        if len(sheets) > 1:
            notes.append(
                f"Workbook contains {len(sheets)} worksheets "
                f"({[s.name for s in available]}); ingested the first non-empty one, "
                f"'{chosen_name}'. Re-upload with an explicit sheet name to choose a different one."
            )

    return sheets[chosen_name], chosen_name, available, notes


# --- JSON -----------------------------------------------------------------

_JSON_RECORD_KEYS = ("data", "records", "rows", "items", "results")


def _read_json(raw: bytes) -> tuple[pd.DataFrame, list[str]]:
    """
    Supports the common tabular JSON shapes and rejects everything else
    with an explicit description of what IS supported — never a silent
    best-effort flatten that would hand a misleading table downstream.

    Supported:
      - list of objects (records):       [{"a": 1}, {"a": 2}]
      - object of arrays (columnar):     {"a": [1, 2], "b": [3, 4]}
      - a records list under a common wrapper key: {"data": [...]}
      - pandas 'split' orient:           {"columns": [...], "data": [[...]]}
      - newline-delimited JSON objects (JSON Lines)
    """
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        raise IngestionError(code="parse_error", message="The JSON file is empty.", details={"format": "json"})

    notes: list[str] = []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as first_error:
        # Might be JSON Lines (one JSON object per line) rather than a
        # single JSON document — a genuinely common tabular shape.
        records = _try_json_lines(text)
        if records is None:
            raise IngestionError(
                code="parse_error",
                message=f"Could not parse the file as JSON: {first_error}",
                details={"format": "json"},
            ) from first_error
        notes.append("Parsed as newline-delimited JSON (JSON Lines).")
        return pd.DataFrame(records), notes

    frame = _frame_from_json_payload(payload, notes)
    return frame, notes


def _try_json_lines(text: str) -> Optional[list[dict]]:
    records: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        records.append(parsed)
    return records or None


def _frame_from_json_payload(payload, notes: list[str]) -> pd.DataFrame:
    if isinstance(payload, list):
        if not payload:
            raise IngestionError(
                code="empty_dataset",
                message="The JSON file contains an empty list — no rows to ingest.",
                details={"format": "json"},
            )
        if all(isinstance(item, dict) for item in payload):
            return pd.DataFrame(payload)
        raise IngestionError(
            code="unsupported_json_structure",
            message=(
                "This JSON is a list, but not a list of objects. PIPER needs tabular JSON: "
                "a list of objects ([{...}, {...}]), an object of equal-length arrays, "
                "a records list under a 'data'/'records'/'rows' key, or JSON Lines."
            ),
            details={"format": "json"},
        )

    if isinstance(payload, dict):
        # pandas 'split' orient.
        if "columns" in payload and "data" in payload and isinstance(payload["data"], list):
            try:
                return pd.DataFrame(payload["data"], columns=payload["columns"])
            except Exception as e:
                raise IngestionError(
                    code="unsupported_json_structure",
                    message=f"This JSON looks like pandas 'split' orient but could not be assembled: {e}",
                    details={"format": "json"},
                ) from e

        # A records list nested under a common wrapper key.
        for key in _JSON_RECORD_KEYS:
            value = payload.get(key)
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                notes.append(f"Ingested the records list found under the '{key}' key.")
                return pd.DataFrame(value)

        # Columnar: object of equal-length arrays.
        if payload and all(isinstance(value, list) for value in payload.values()):
            lengths = {len(value) for value in payload.values()}
            if len(lengths) != 1:
                raise IngestionError(
                    code="unsupported_json_structure",
                    message=(
                        "This JSON is an object of arrays, but the arrays have different lengths "
                        f"({sorted(lengths)}) so they cannot form a table."
                    ),
                    details={"format": "json", "column_lengths": sorted(lengths)},
                )
            return pd.DataFrame(payload)

        raise IngestionError(
            code="unsupported_json_structure",
            message=(
                "This JSON object is not tabular. PIPER needs a list of objects "
                "([{...}, {...}]), an object of equal-length arrays, a records list under a "
                "'data'/'records'/'rows' key, pandas 'split' orient, or JSON Lines."
            ),
            details={"format": "json", "top_level_keys": list(payload)[:20]},
        )

    raise IngestionError(
        code="unsupported_json_structure",
        message=(
            f"This JSON is a bare {type(payload).__name__}, not tabular data. PIPER needs a list "
            "of objects, an object of equal-length arrays, or JSON Lines."
        ),
        details={"format": "json"},
    )


# --- Parquet --------------------------------------------------------------


def _read_parquet(raw: bytes) -> pd.DataFrame:
    """
    Parquet carries its own column types in the file itself, so
    pd.read_parquet() restores them directly (int64 stays int64, a
    timestamp column stays datetime64) — no inference, no round-tripping
    through text. This is why Parquet preserves types more faithfully
    than CSV, and why nothing here overrides the dtypes it returns.
    """
    try:
        return pd.read_parquet(io.BytesIO(raw))
    except Exception as e:
        raise IngestionError(
            code="parse_error",
            message=f"Could not parse the file as Parquet: {e}",
            details={"format": "parquet"},
        ) from e


# --- IPYNB ----------------------------------------------------------------


class _DataFrameTableParser(HTMLParser):
    """
    Extracts rows from pandas' own `DataFrame.to_html()` markup, which
    is what Jupyter stores in a cell's `text/html` output when a
    DataFrame is displayed.

    Deliberately stdlib-only rather than `pandas.read_html` (which
    requires lxml/bs4/html5lib). This parser targets ONE highly regular,
    machine-generated structure — not arbitrary web HTML — so a focused
    ~60-line parser is both sufficient and avoids adding a heavyweight
    HTML-parsing dependency for a single feature. Anything it cannot
    confidently parse is reported as an error, never guessed at.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.header: list[str] = []
        self._in_table = False
        self._in_head = False
        self._current: list[str] = []
        self._cell: Optional[list[str]] = None
        self._row_is_header = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._in_table = True
        elif not self._in_table:
            return
        elif tag == "thead":
            self._in_head = True
        elif tag == "tr":
            self._current = []
            self._row_is_header = False
        elif tag in ("td", "th"):
            self._cell = []
            if tag == "th":
                self._row_is_header = self._in_head

    def handle_endtag(self, tag):
        if not self._in_table:
            return
        if tag == "table":
            self._in_table = False
        elif tag == "thead":
            self._in_head = False
        elif tag in ("td", "th") and self._cell is not None:
            self._current.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr":
            if self._row_is_header and not self.header:
                self.header = self._current
            elif self._current:
                self.rows.append(self._current)
            self._current = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


_ROWS_COLS_FOOTER = re.compile(r"<p>\s*([\d,]+)\s*rows?\s*[×x]\s*([\d,]+)\s*columns?\s*</p>", re.IGNORECASE)


def _frame_from_dataframe_html(html: str) -> Optional[tuple[pd.DataFrame, int]]:
    """
    Returns (frame, declared_row_count) or None if this HTML doesn't
    look like a pandas DataFrame table. declared_row_count comes from
    pandas' own "N rows × M columns" footer when present, and is what
    lets the caller detect a TRUNCATED display (Jupyter elides the
    middle of a large DataFrame) rather than silently ingesting a
    partial dataset.
    """
    parser = _DataFrameTableParser()
    try:
        parser.feed(html)
    except Exception:
        return None

    if not parser.header or not parser.rows:
        return None

    header = list(parser.header)
    rows = [list(r) for r in parser.rows]

    # pandas emits a leading blank <th> for the index column; drop that
    # column from both the header and every body row so the resulting
    # frame's columns line up with the real data columns.
    if header and header[0] == "":
        header = header[1:]
        rows = [row[1:] if len(row) == len(header) + 1 else row for row in rows]

    rows = [row for row in rows if len(row) == len(header)]
    if not rows:
        return None

    declared_rows = -1
    match = _ROWS_COLS_FOOTER.search(html)
    if match:
        declared_rows = int(match.group(1).replace(",", ""))

    frame = pd.DataFrame(rows, columns=header)
    # Values arrive as strings from HTML; recover real numeric dtypes
    # where the whole column is genuinely numeric, leaving everything
    # else untouched (never coercing a mixed column into NaNs).
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().all():
            frame[column] = converted

    return frame, declared_rows


_READ_CALL = re.compile(r"""(?:pd|pandas)\s*\.\s*read_\w+\s*\(\s*[rbf]?['"]([^'"]+)['"]""")


def _read_ipynb(raw: bytes) -> tuple[pd.DataFrame, list[str]]:
    """
    Recovers tabular data from a notebook WITHOUT executing any of its
    code (a notebook is untrusted input; executing it would be arbitrary
    remote code execution). Only already-stored cell OUTPUTS are read.

    Strategy, in order:
      1. A cell output containing a pandas DataFrame HTML table — real,
         complete data that was actually displayed when the notebook ran.
      2. A cell output containing tabular JSON.
    Truncated DataFrame displays (Jupyter elides the middle of a large
    frame) are detected via pandas' own "N rows × M columns" footer and
    rejected rather than silently ingested as a partial dataset.

    If no usable output exists, any `pd.read_csv("...")`-style call in
    the notebook's source is surfaced by name, since that external file
    is what the user actually needs to upload instead.
    """
    try:
        notebook = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise IngestionError(
            code="parse_error",
            message=f"Could not parse the file as a Jupyter notebook (invalid JSON): {e}",
            details={"format": "ipynb"},
        ) from e

    if not isinstance(notebook, dict) or "cells" not in notebook:
        raise IngestionError(
            code="parse_error",
            message="This file is valid JSON but not a Jupyter notebook (no 'cells' key).",
            details={"format": "ipynb"},
        )

    cells = notebook.get("cells") or []
    if not isinstance(cells, list):
        raise IngestionError(
            code="parse_error",
            message="This notebook's 'cells' is not a list.",
            details={"format": "ipynb"},
        )

    candidates: list[tuple[pd.DataFrame, int, int]] = []  # (frame, declared_rows, cell_index)
    truncated_found: list[tuple[int, int]] = []  # (declared_rows, parsed_rows)
    referenced_files: list[str] = []

    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue

        source = cell.get("source")
        if isinstance(source, list):
            source_text = "".join(str(s) for s in source)
        else:
            source_text = str(source or "")
        for referenced in _READ_CALL.findall(source_text):
            if referenced not in referenced_files:
                referenced_files.append(referenced)

        for output in cell.get("outputs") or []:
            if not isinstance(output, dict):
                continue
            data = output.get("data") or {}
            if not isinstance(data, dict):
                continue

            html_payload = data.get("text/html")
            if html_payload is not None:
                html = "".join(html_payload) if isinstance(html_payload, list) else str(html_payload)
                parsed = _frame_from_dataframe_html(html)
                if parsed is not None:
                    frame, declared_rows = parsed
                    if declared_rows > 0 and declared_rows > len(frame):
                        # Jupyter elided the middle of this DataFrame —
                        # ingesting it would silently lose rows.
                        truncated_found.append((declared_rows, len(frame)))
                        continue
                    candidates.append((frame, declared_rows, index))
                    continue

            json_payload = data.get("application/json")
            if isinstance(json_payload, (list, dict)):
                try:
                    frame = _frame_from_json_payload(json_payload, [])
                except IngestionError:
                    continue
                if not frame.empty:
                    candidates.append((frame, len(frame), index))

    if candidates:
        # Largest recovered table wins — with several displayed frames,
        # the biggest is overwhelmingly the actual dataset rather than a
        # `.head()` preview or a small summary table.
        frame, _declared, cell_index = max(candidates, key=lambda c: (c[0].shape[0], c[0].shape[1]))
        notes = [
            f"Recovered a {frame.shape[0]}x{frame.shape[1]} table from the stored output of notebook cell "
            f"{cell_index + 1}. The notebook's code was NOT executed — only saved outputs were read."
        ]
        if len(candidates) > 1:
            notes.append(
                f"{len(candidates)} displayed tables were found; ingested the largest one."
            )
        return frame, notes

    if truncated_found:
        declared_rows, parsed_rows = truncated_found[0]
        raise IngestionError(
            code="ipynb_output_truncated",
            message=(
                f"This notebook displays a DataFrame of {declared_rows} rows, but Jupyter only saved "
                f"{parsed_rows} of them in the output (the middle rows are elided with '...'). "
                "Ingesting it would silently lose data. Export the full table from the notebook "
                "(e.g. df.to_csv('data.csv')) and upload that file instead."
            ),
            details={"declared_rows": declared_rows, "rows_available_in_output": parsed_rows},
        )

    if referenced_files:
        raise IngestionError(
            code="ipynb_external_source_missing",
            message=(
                "This notebook has no saved table output to read, and its code was not executed "
                f"(that would be unsafe). It loads its data from: {referenced_files}. "
                "Upload that data file directly instead."
            ),
            details={"referenced_files": referenced_files},
        )

    raise IngestionError(
        code="ipynb_no_tabular_output",
        message=(
            "No tabular data could be recovered from this notebook. PIPER reads only saved cell "
            "outputs (it never executes notebook code), so the notebook must contain a displayed "
            "DataFrame. Re-run the notebook with a cell that displays the full DataFrame, and save "
            "it — or upload the underlying data file directly."
        ),
        details={"format": "ipynb"},
    )


# --- Orchestration --------------------------------------------------------


def ingest_dataset(
    raw: bytes,
    filename: str,
    dataset_id: str,
    store: DatasetStore,
    sheet_name: Optional[str] = None,
) -> ToolResult[IngestionResult]:
    """
    Detects the format, normalizes the file into one DataFrame, and
    saves it under dataset_id — the single entry point the API layer
    uses for every supported format.

    Follows the same store-mutating, ToolResult-returning shape as every
    other tool group (drop_column, split_dataset, ...): on success the
    DataFrame is already in the store and `data` describes how it got
    there; on failure nothing is saved and `error` is structured.
    """
    detected = detect_format(filename)
    if detected is None:
        supported = sorted(set(FORMAT_EXTENSIONS))
        return ToolResult[IngestionResult](
            success=False,
            tool_name="ingest_dataset",
            message=f"Unsupported file type: '{filename}'. Supported extensions: {supported}.",
            error=ToolError(
                code="unsupported_format",
                message=f"Unsupported file type. Supported extensions: {supported}.",
                details={"filename": filename, "supported_extensions": supported},
            ),
        )

    notes: list[str] = []
    chosen_sheet: Optional[str] = None
    available_sheets: list[SheetInfo] = []

    try:
        if detected == "csv":
            frame = _read_delimited(raw, sep=",", label="CSV")
        elif detected == "tsv":
            frame = _read_delimited(raw, sep="\t", label="TSV")
        elif detected == "excel":
            frame, chosen_sheet, available_sheets, notes = _read_excel(raw, sheet_name)
        elif detected == "json":
            frame, notes = _read_json(raw)
        elif detected == "ipynb":
            frame, notes = _read_ipynb(raw)
        else:  # parquet
            frame = _read_parquet(raw)
    except IngestionError as e:
        return ToolResult[IngestionResult](
            success=False,
            tool_name="ingest_dataset",
            message=e.message,
            error=ToolError(code=e.code, message=e.message, details={**e.details, "detected_format": detected}),
        )

    # Identical emptiness contract to the original CSV-only path — the
    # same two rejections, now applied uniformly to every format.
    if frame.shape[1] == 0:
        return ToolResult[IngestionResult](
            success=False,
            tool_name="ingest_dataset",
            message=f"The uploaded {detected} file has zero columns.",
            error=ToolError(
                code="zero_columns",
                message=f"The uploaded {detected} file has zero columns.",
                details={"detected_format": detected},
            ),
        )
    if frame.shape[0] == 0:
        return ToolResult[IngestionResult](
            success=False,
            tool_name="ingest_dataset",
            message=f"The uploaded {detected} file is empty (zero rows).",
            error=ToolError(
                code="empty_dataset",
                message=f"The uploaded {detected} file is empty (zero rows).",
                details={"detected_format": detected},
            ),
        )

    # Column names must be strings for the rest of the pipeline (JSON
    # and Excel can both yield integer column labels); this is a
    # labeling normalization only — no values are touched.
    frame.columns = [str(c) for c in frame.columns]

    store.save(dataset_id, frame)

    result = IngestionResult(
        detected_format=detected,
        source_filename=filename,
        rows=int(frame.shape[0]),
        columns=list(frame.columns),
        sheet_name=chosen_sheet,
        available_sheets=available_sheets,
        notes=notes,
    )

    return ToolResult[IngestionResult](
        success=True,
        tool_name="ingest_dataset",
        message=(
            f"Ingested {filename} as {detected}: {frame.shape[0]} rows x {frame.shape[1]} columns."
        ),
        data=result,
    )
