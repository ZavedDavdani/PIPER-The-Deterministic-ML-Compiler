"""
RunStore and ModelStore — interfaces only.

Neither is implemented yet. RunStore becomes relevant once AgentState
exists (M2, LangGraph skeleton); ModelStore becomes relevant once
train_model() exists (later in M1, after cleaning tools). Declaring
the interfaces now costs nothing and keeps the repository-pattern
shape consistent across all three stores, but no concrete class is
written until the milestone that actually needs it — implementing
either now would be scope creep ahead of the locked build order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RunStore(ABC):
    """
    Will hold AgentState per run_id once M2 introduces AgentState.

    Locked shape (from the state contract):
        RunStore
        ├── create()
        ├── get()
        ├── update()
        └── append_trace()
    """

    @abstractmethod
    def create(self, run_id: str, initial_state: Any, *, display_dataset_id: str | None = None) -> None:
        """
        display_dataset_id (Batch 5): overrides the dataset_id reported
        by RunRecord/GET /runs/{run_id} when the caller executes the
        run against a different, internal dataset_id than the one the
        user actually selected (see app/api/routers/runs.py's per-run
        dataset clone) — defaults to reading dataset_id off
        initial_state, unchanged, when omitted.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, run_id: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def update(self, run_id: str, state: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_trace(self, run_id: str, trace_entry: Any) -> None:
        raise NotImplementedError


class ModelStore(ABC):
    """
    Will hold fitted scikit-learn model objects per model_id once
    train_model() exists. Locked principle: trained models never enter
    LangGraph state directly — only their model_id and metadata
    (ModelResult) do. The actual fitted object lives here.
    """

    @abstractmethod
    def save(self, model_id: str, model: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, model_id: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def delete(self, model_id: str) -> None:
        raise NotImplementedError
