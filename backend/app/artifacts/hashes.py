"""SHA-256 hashes for finalized artifact files. Written last."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hashes_manifest(bundle_dir: Path, filenames: list[str]) -> dict[str, str]:
    hashes = {name: sha256_file(bundle_dir / name) for name in filenames}
    payload = {
        "algorithm": "sha256",
        "files": hashes,
    }
    (bundle_dir / "hashes.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashes
