"""SHA-256 fingerprints. Tamper-evident, not a claim of immutability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HASH_ALGORITHM = "sha256"

HASH_CAVEAT = (
    "These fingerprints are tamper-evident content hashes. They do not "
    "make an artifact immutable, and they are not a legal chain of custody."
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_canonical_json(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=True,
    )
    return sha256_text(serialized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
