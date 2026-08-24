"""Optional standalone deployment package. Docker is optional. No cloud deps."""

from __future__ import annotations

import shutil
from pathlib import Path

from app.deployment.errors import InferenceError
from app.deployment.inference_script import render_inference_py
from app.deployment.loader import load_verified_bundle
from app.deployment.paths import PACKAGE_FILES, package_dir

_README = """# PIPER standalone inference package

This package scores **new unseen rows** with a VERIFIED `pipeline.joblib`.
It does **not** include PIPER, LangGraph, Ollama, or SQLite.

## Files

- `pipeline.joblib` — fitted sklearn Pipeline
- `inference.py` — schema check + `predict()`
- `requirements.txt` — pandas / scikit-learn / joblib
- `Dockerfile` — **optional** container build

## Local use

```
pip install -r requirements.txt
python inference.py new_data.csv -o predictions.csv
```

The input CSV is never modified. Predictions are written to a new file.

## Docker (optional)

```
docker build -t piper-inference .
docker run --rm -v /path/to/data:/data piper-inference /data/new_data.csv -o /data/predictions.csv
```

Do not treat this image as a multi-service platform. Redis, Celery,
PostgreSQL, and Kubernetes are not required and are not included.
"""

_DOCKERFILE = """\
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline.joblib inference.py ./
ENTRYPOINT ["python", "inference.py"]
"""


def _requirements_text(manifest: dict) -> str:
    versions = manifest.get("library_versions") or {}
    pandas_v = versions.get("pandas") or "3.0.2"
    sklearn_v = versions.get("scikit-learn") or "1.8.0"
    joblib_v = versions.get("joblib") or "1.4.2"
    lines = [
        f"pandas=={pandas_v}" if pandas_v != "unknown" else "pandas",
        f"scikit-learn=={sklearn_v}" if sklearn_v != "unknown" else "scikit-learn",
        f"joblib=={joblib_v}" if joblib_v != "unknown" else "joblib",
    ]
    return "\n".join(lines) + "\n"


def write_deployment_package(artifact_root: Path, run_id: str) -> dict:
    bundle = load_verified_bundle(artifact_root, run_id)
    dest = package_dir(artifact_root, run_id)
    dest.mkdir(parents=True, exist_ok=True)
    joblib_src = bundle["joblib_path"]
    shutil.copy2(joblib_src, dest / "pipeline.joblib")
    manifest = bundle["manifest"]
    target = str(manifest.get("target") or "")
    algorithm = str(bundle["algorithm"] or "unknown")
    (dest / "inference.py").write_text(
        render_inference_py(
            run_id=run_id,
            target_column=target,
            algorithm=algorithm,
            feature_columns=bundle["feature_columns"],
        ),
        encoding="utf-8",
    )
    (dest / "requirements.txt").write_text(_requirements_text(manifest), encoding="utf-8")
    (dest / "README.md").write_text(_README, encoding="utf-8")
    (dest / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8")
    files = [name for name in PACKAGE_FILES if (dest / name).is_file()]
    if "pipeline.joblib" not in files:
        raise InferenceError("package_failed", "pipeline.joblib was not copied into the package.")
    return {
        "run_id": run_id,
        "status": "READY",
        "directory": str(dest),
        "files": files,
        "docker_optional": True,
    }
