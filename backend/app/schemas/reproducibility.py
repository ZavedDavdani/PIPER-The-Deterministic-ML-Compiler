"""
Reproducibility metadata: what environment, what dataset content, and
what random-state configuration actually produced a given run.

Answers, from a completed AgentState alone, without re-reading source:
    What environment produced this result?
    Which dataset (by CONTENT, not just by dataset_id reference) produced it?
    Which deterministic seed(s) were used?

Design mirrors app/agent/plan_canonical.py's canonicalization philosophy
(deterministic canonical representation -> sorted-key JSON -> SHA-256)
but is a DEDICATED implementation for DataFrames, not a reuse of
PlanStep-specific code — a DataFrame's canonical representation (column
names/order/dtypes/values) has nothing in common with a plan step's
(tool_name/arguments), so sharing code between them would force an
artificial abstraction. Only the underlying hashing PATTERN is shared.

random_state is recorded as two EXPLICIT, separate fields
(split_random_state, model_random_state) rather than one collapsed
value. These are two independently-defined constants in the codebase
(app/agent/tools/preparation.py's RANDOM_STATE and
app/agent/tools/training.py's RANDOM_STATE) that currently happen to
both equal 42 — collapsing them into a single field would misrepresent
the architecture and could go silently stale if either is ever changed
independently of the other.
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field


# --- Environment metadata --------------------------------------------------


class EnvironmentMetadata(BaseModel):
    """
    Actual runtime versions, read live from the running process — never
    hardcoded. NumPy is a transitive dependency (via pandas/scikit-learn,
    not listed directly in requirements.txt); recording its version adds
    no new dependency, since it must already be importable for pandas/
    sklearn to function at all.
    """

    model_config = ConfigDict(extra="forbid")

    python_version: str
    pandas_version: str
    numpy_version: str
    sklearn_version: str


def capture_environment_metadata() -> EnvironmentMetadata:
    """Reads live version info from the actual running environment."""
    return EnvironmentMetadata(
        python_version=sys.version,
        pandas_version=pd.__version__,
        numpy_version=numpy.__version__,
        sklearn_version=sklearn.__version__,
    )


# --- Dataset fingerprint -----------------------------------------------


def _canonicalize_value(value) -> object:
    """
    Normalizes a single cell value into something JSON-serializable and
    stable. NaN/NaT (which are NOT self-equal and serialize inconsistently
    across pandas versions) are collapsed to the literal string "__NaN__"
    so that missing-value semantics are captured deterministically
    without depending on how a specific pandas/NumPy version reprs them.
    """
    if pd.isna(value):
        return "__NaN__"
    if isinstance(value, (numpy.integer,)):
        return int(value)
    if isinstance(value, (numpy.floating,)):
        return float(value)
    if isinstance(value, (numpy.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    Deterministic SHA-256 fingerprint of a DataFrame's actual content.

    CANONICALIZATION DECISION (explicit, mirrors plan_canonical.py's
    documented order-sensitivity choices):

        column NAMES and ORDER  -> ORDER-SENSITIVE, included as-is.
            A dataset contract where 'A,B' and 'B,A' are silently
            treated as identical would hide genuine schema changes
            (e.g. a column-reordering bug upstream) from the
            fingerprint. Nothing in the existing dataset contract
            (DatasetStore, profiling, cleaning tools) treats column
            order as insignificant, so it is NOT normalized away here.

        row ORDER / INDEX       -> INCLUDED, in the DataFrame's given
            order. Nothing in the existing dataset contract (DatasetStore
            never reorders rows; split_dataset() depends on row identity
            being stable) treats row order as unordered/interchangeable,
            so rows are hashed in their given order, not sorted. The
            DataFrame's index values are included as their own column in
            the canonical representation, since a reset-vs-preserved
            index is a meaningful content difference this project's
            tools could produce (e.g. after drop_column/impute in place).

        VALUES and DTYPES       -> both included. Two columns with
            identical values but different dtypes (e.g. '1' as a string
            vs 1 as an int) are NOT the same dataset for reproducibility
            purposes, since downstream behavior (encoding, scaling,
            fitting) genuinely differs by dtype.

    Implementation: builds a canonical JSON payload of
    {columns: [...], dtypes: {...}, index: [...], rows: [[...], ...]}
    via pandas' own to_dict/tolist (never Python object identity, never
    memory addresses), serializes with sort_keys=True and no whitespace
    variance (same pattern as plan_canonical.canonical_json()), then
    SHA-256-hashes the UTF-8 bytes. Operates on df.copy() throughout —
    the original DataFrame is never mutated, and no copy is written back
    to any store.
    """
    working = df.copy()

    columns = [str(c) for c in working.columns]
    dtypes = {str(c): str(working[c].dtype) for c in working.columns}
    index_values = [_canonicalize_value(v) for v in working.index.tolist()]

    rows = [
        [_canonicalize_value(v) for v in row]
        for row in working.itertuples(index=False, name=None)
    ]

    payload = {
        "columns": columns,
        "dtypes": dtypes,
        "index": index_values,
        "rows": rows,
    }

    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# --- Reproducibility metadata (run-level) -----------------------------


class ReproducibilityMetadata(BaseModel):
    """
    Run-level reproducibility record. Populated once per run, at the
    point in the graph where the split dataset content and random-state
    configuration are both concretely fixed (after SPLIT — see
    real_nodes.reproducibility_node for the justification). Carried on
    AgentState.reproducibility through to the final result.

    split_random_state / model_random_state are recorded SEPARATELY
    (see module docstring) even though both are currently 42 — this is
    a deliberate accuracy choice, not an oversight.
    """

    model_config = ConfigDict(extra="forbid")

    environment: EnvironmentMetadata
    dataset_fingerprint: str
    split_random_state: int
    model_random_state: int
    pipeline_fingerprint: str | None = Field(
        default=None,
        description=(
            "Optional deterministic fingerprint of the executable "
            "pipeline configuration for this run (candidate models, "
            "feature-engineering intent, target, split config). None "
            "if not computed for this run."
        ),
    )
