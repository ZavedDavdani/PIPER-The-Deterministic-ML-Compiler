"""Deterministic training_reproduction.ipynb from recorded run metadata."""

from __future__ import annotations

import json
from typing import Any, Sequence


def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]}


def _code(source: str) -> dict:
    lines = [line + "\n" for line in source.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
        if not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def render_training_notebook(
    *,
    run_id: str,
    dataset_id: str,
    target_column: str,
    split_id: str,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    numeric_columns: Sequence[str],
    candidates: list[dict[str, Any]],
    winner: dict[str, Any],
    metrics: dict[str, Any] | None,
    executed_operations: list[str],
) -> str:
    """
    Notebook cells describe the ACTUAL executed configuration.
    Hyperparameters not present in TrainingResult.parameters are omitted.
    """
    ops_block = "\n".join(f"- `{op}`" for op in executed_operations) or "- (none recorded)"
    candidate_lines = []
    for item in candidates:
        params = item.get("parameters") or {}
        candidate_lines.append(
            f"- `{item.get('algorithm')}` (`{item.get('model_id')}`): `{params}`"
        )
    candidates_md = "\n".join(candidate_lines) or "- (none recorded)"
    winner_params = winner.get("parameters") or {}
    metrics_json = json.dumps(metrics or {}, indent=2, default=str)

    cells = [
        _md(
            f"# Training reproduction — `{run_id}`\n\n"
            "Generated deterministically from the **executed** PIPER run, "
            "not from the original LLM proposal.\n\n"
            f"- Dataset id: `{dataset_id}`\n"
            f"- Target: `{target_column}`\n"
            f"- Split id: `{split_id}`\n"
            f"- Winning model: `{winner.get('algorithm')}` (`{winner.get('model_id')}`)\n"
        ),
        _md("## Dataset loading\n\nReload the same tabular file you uploaded to PIPER, then assign the target."),
        _code(
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            "# Point this at the original dataset file used for the run.\n"
            "DATA_PATH = Path(\"dataset.csv\")\n"
            f"TARGET = {target_column!r}\n"
            "df = pd.read_csv(DATA_PATH)\n"
            "df.head()"
        ),
        _md("## Executed preprocessing operations\n\n" + ops_block),
        _md(
            "## Train/test split\n\n"
            "PIPER used `split_dataset()` with `random_state=42` and `test_size=0.2` "
            "(locked V1 defaults). The recorded split id is shown for audit; "
            "re-running this cell will not reconstruct that exact row partition "
            "unless you persist the split frames separately."
        ),
        _code(
            "from sklearn.model_selection import train_test_split\n\n"
            f"FEATURE_COLUMNS = {list(feature_columns)!r}\n"
            "X = df[FEATURE_COLUMNS]\n"
            "y = df[TARGET]\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y,\n"
            ")"
        ),
        _md(
            "## Preprocessing (as fitted inside the winning Pipeline)\n\n"
            f"- Categorical (OneHotEncoder): `{list(categorical_columns)}`\n"
            f"- Numeric (StandardScaler): `{list(numeric_columns)}`\n\n"
            "Remainder columns are dropped (`ColumnTransformer(remainder='drop')`), "
            "matching `train_model()`."
        ),
        _md("## Candidate models actually trained\n\n" + candidates_md),
        _md(
            "## Winning estimator configuration\n\n"
            f"Algorithm: `{winner.get('algorithm')}`\n\n"
            f"Recorded hyperparameters: `{winner_params}`\n\n"
            "Do not invent additional hyperparameters."
        ),
        _code(
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n\n"
            f"categorical_columns = {list(categorical_columns)!r}\n"
            f"numeric_columns = {list(numeric_columns)!r}\n"
            f"algorithm = {winner.get('algorithm')!r}\n"
            f"params = {winner_params!r}\n\n"
            "transformers = []\n"
            "if categorical_columns:\n"
            "    transformers.append(('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_columns))\n"
            "if numeric_columns:\n"
            "    transformers.append(('numeric', StandardScaler(), numeric_columns))\n"
            "preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')\n"
            "if algorithm == 'logistic_regression':\n"
            "    clf = LogisticRegression(\n"
            "        C=params.get('C', 1.0),\n"
            "        max_iter=int(params.get('max_iter', 1000)),\n"
            "        l1_ratio=0.0,\n"
            "        random_state=42,\n"
            "    )\n"
            "else:\n"
            "    clf = RandomForestClassifier(\n"
            "        n_estimators=int(params.get('n_estimators', 200)),\n"
            "        max_depth=params.get('max_depth', None),\n"
            "        min_samples_split=int(params.get('min_samples_split', 2)),\n"
            "        min_samples_leaf=int(params.get('min_samples_leaf', 1)),\n"
            "        random_state=42,\n"
            "    )\n"
            "pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])\n"
            "pipeline.fit(X_train, y_train)"
        ),
        _md("## Evaluation metrics recorded for the winner\n\n```json\n" + metrics_json + "\n```"),
        _md(
            "## Preferred path for exact inference\n\n"
            "For bit-identical predictions vs the verified run, load `pipeline.joblib` "
            "(see `pipeline.py`). The cells above reconstruct *equivalent* training "
            "code from recorded metadata; they are not a substitute for the parity-gated joblib."
        ),
        _code(
            "import joblib\n"
            "from pathlib import Path\n\n"
            "exported = joblib.load(Path('pipeline.joblib'))\n"
            "exported.predict(X_test.head())"
        ),
    ]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "piper": {"run_id": run_id, "generated": "deterministic"},
        },
        "cells": cells,
    }
    return json.dumps(notebook, indent=2)
