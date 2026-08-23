"""Construct the process-lifetime RunStore from environment.

Default for a local PIPER process is SQLite. The pytest suite forces
PIPER_RUN_STORE=memory so existing tests stay isolated and do not
write a shared database file.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.storage.run_store import InMemoryRunStore
from app.storage.sqlite_run_store import SqliteRunStore

DEFAULT_SQLITE_PATH = Path("data") / "piper_runs.sqlite"


def create_run_store():
    backend = os.environ.get("PIPER_RUN_STORE", "sqlite").strip().lower()
    if backend in {"memory", "mem", "inmemory", "in-memory"}:
        return InMemoryRunStore()
    path = os.environ.get("PIPER_SQLITE_PATH", str(DEFAULT_SQLITE_PATH))
    return SqliteRunStore(path)
