"""
DatasetStore — the only storage component M1 actually needs.

Locked contract:
    DatasetStore
    ├── save()
    ├── get()
    └── delete()

Tools operate against a dataset_id, never a raw DataFrame passed
around by hand. This is what makes that possible: an abstract
interface now, a dict-backed implementation for M1–M2, and a
SQLite/PostgreSQL implementation later (M3+) behind the exact same
interface — the tools and agent never need to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from app.storage.exceptions import DatasetNotFoundError


class DatasetStore(ABC):
    """Abstract interface. Tools depend on this, never on a concrete impl."""

    @abstractmethod
    def save(self, dataset_id: str, df: pd.DataFrame) -> None:
        """Store (or overwrite) a DataFrame under dataset_id."""
        raise NotImplementedError

    @abstractmethod
    def get(self, dataset_id: str) -> pd.DataFrame:
        """
        Retrieve the DataFrame for dataset_id.

        Raises DatasetNotFoundError if it doesn't exist.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, dataset_id: str) -> None:
        """
        Remove dataset_id from storage.

        Raises DatasetNotFoundError if it doesn't exist.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, dataset_id: str) -> bool:
        """Non-raising existence check — tools use this before get() when
        they want to produce a clean ToolError instead of catching an
        exception."""
        raise NotImplementedError

    @abstractmethod
    def list_ids(self) -> list[str]:
        """
        All dataset_ids currently stored, in no particular guaranteed
        order (concrete implementations may document their own
        ordering). Added for the API layer's dataset-listing endpoint
        (M5) — no tool needed this before, since every tool already
        operates against a specific, known dataset_id.
        """
        raise NotImplementedError


class InMemoryDatasetStore(DatasetStore):
    """
    M1–M2 implementation: a plain dict keyed by dataset_id.

    Explicitly NOT thread-safe and NOT persistent across process
    restarts — both are acceptable for a local, single-user demo, and
    both are exactly the concerns a future SQLite/Postgres
    implementation would need to handle. Nothing here should ever be
    imported directly by a tool; tools receive a DatasetStore instance
    (this or a future implementation) via dependency injection.
    """

    def __init__(self) -> None:
        self._data: dict[str, pd.DataFrame] = {}

    def save(self, dataset_id: str, df: pd.DataFrame) -> None:
        # Store a copy — callers mutating their own reference to df
        # after saving must never silently corrupt what's in storage.
        self._data[dataset_id] = df.copy()

    def get(self, dataset_id: str) -> pd.DataFrame:
        if dataset_id not in self._data:
            raise DatasetNotFoundError(dataset_id)
        # Return a copy — tools must go through save() to persist any
        # change, never mutate the stored DataFrame in place via a
        # reference obtained from get(). This is what keeps
        # cleaning_log/tool_trace as the single source of truth for
        # "what actually changed," rather than silent in-place edits.
        return self._data[dataset_id].copy()

    def delete(self, dataset_id: str) -> None:
        if dataset_id not in self._data:
            raise DatasetNotFoundError(dataset_id)
        del self._data[dataset_id]

    def exists(self, dataset_id: str) -> bool:
        return dataset_id in self._data

    def list_ids(self) -> list[str]:
        # Insertion order — dict preserves it in Python 3.7+, and this
        # is a reasonable, stable-enough default for a listing
        # endpoint (most-recently-added last).
        return list(self._data.keys())
