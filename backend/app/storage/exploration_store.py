"""
ExplorationStore (Batch 6B: Learn-Explore).

Holds every exploration's ExplorationResult, keyed by its own
experiment_id — a namespace structurally SEPARATE from RunStore's
per-run records (locked constraint: exploration results must never be
merged into the original run's own state). list_for_run() is the only
way an exploration is ever associated back to its original run_id, and
it's a read filter, never a write into RunStore.

In-memory only, mirroring every other store in this codebase (no
persistence across process restarts).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.exploration import ExplorationResult
from app.storage.exceptions import ExplorationNotFoundError


class ExplorationStore(ABC):
    @abstractmethod
    def save(self, result: ExplorationResult) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, experiment_id: str) -> ExplorationResult:
        raise NotImplementedError

    @abstractmethod
    def list_for_run(self, run_id: str) -> list[ExplorationResult]:
        raise NotImplementedError


class InMemoryExplorationStore(ExplorationStore):
    def __init__(self) -> None:
        self._data: dict[str, ExplorationResult] = {}

    def save(self, result: ExplorationResult) -> None:
        self._data[result.experiment_id] = result

    def get(self, experiment_id: str) -> ExplorationResult:
        if experiment_id not in self._data:
            raise ExplorationNotFoundError(experiment_id)
        return self._data[experiment_id]

    def list_for_run(self, run_id: str) -> list[ExplorationResult]:
        return [r for r in self._data.values() if r.run_id == run_id]
