"""
SplitStore — holds train/test DataFrame pairs, keyed by split_id.

Mirrors DatasetStore's shape and copy-isolation guarantees exactly,
per the decision to keep splits as their own store type (symmetric
with RunStore/ModelStore) rather than folding them into DatasetStore.
Implemented now (not just stubbed) because split_dataset() is being
built immediately and needs somewhere real to write to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from app.storage.exceptions import SplitNotFoundError


class SplitStore(ABC):
    """Abstract interface. Tools depend on this, never a concrete impl."""

    @abstractmethod
    def save(
        self, split_id: str, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, split_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (train_df, test_df). Raises SplitNotFoundError if missing."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, split_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self, split_id: str) -> bool:
        raise NotImplementedError


class InMemorySplitStore(SplitStore):
    """
    M1–M2 implementation: a plain dict keyed by split_id, each value a
    (train_df, test_df) tuple. Same copy-isolation guarantee as
    InMemoryDatasetStore: get() and save() never hand out or accept a
    live reference that could silently mutate what's stored.
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def save(
        self, split_id: str, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        self._data[split_id] = (train_df.copy(), test_df.copy())

    def get(self, split_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if split_id not in self._data:
            raise SplitNotFoundError(split_id)
        train_df, test_df = self._data[split_id]
        return train_df.copy(), test_df.copy()

    def delete(self, split_id: str) -> None:
        if split_id not in self._data:
            raise SplitNotFoundError(split_id)
        del self._data[split_id]

    def exists(self, split_id: str) -> bool:
        return split_id in self._data
