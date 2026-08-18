"""
Run lifecycle, status, results, and live-progress endpoints (M5).

Every ML/agent decision still happens entirely inside build_graph()/
stream_with_tracing() (app/agent) — this router's job is orchestration
only: validate the request, kick off execution in the background (a
real run can take seconds to several minutes depending on the LLM
provider), and expose RunStore's state over HTTP/SSE.

Execution runs via FastAPI's BackgroundTasks, which Starlette offloads
to a worker thread for a synchronous callable (stream_with_tracing()
is fully synchronous, matching the rest of the agent core) — so it
never blocks the event loop, and concurrent GET/SSE requests for the
same run_id are served normally while it's in flight.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agent import AgentState, build_graph
from app.agent.run_summary import build_run_summary
from app.agent.timeline import build_execution_timeline
from app.agent.tools.exploration import explore_alternative
from app.agent.tracing import stream_with_tracing
from app.api.dependencies import (
    get_dataset_store,
    get_exploration_store,
    get_llm_provider,
    get_model_store,
    get_run_store,
    get_split_store,
)
from app.api.schemas import (
    CreateExplorationRequest,
    CreateRunRequest,
    CreateRunResponse,
    RunResultResponse,
    RunStatusResponse,
)
from app.learning.explain import build_run_explanation
from app.schemas.execution_timeline import ExecutionTimeline
from app.schemas.exploration import ExplorationResult
from app.schemas.learning import RunExplanation
from app.schemas.run_summary import RunSummary
from app.storage import (
    DatasetStore,
    InMemoryExplorationStore,
    InMemoryModelStore,
    InMemoryRunStore,
    RunNotFoundError,
    SplitStore,
)
from app.storage.exceptions import ExplorationNotFoundError

router = APIRouter(prefix="/runs", tags=["runs"])

_GRAPH_RECURSION_LIMIT = 50
"""
Matches the convention used at every other build_graph()/graph.invoke()
call site in this codebase (tests, tracing tests). PIPER's own
MAX_EXECUTION_STEPS (app/agent/graph.py, M4) is the real, unconditional
termination guarantee independent of this value — see
CreateRunRequest.max_retries's own docstring.
"""

_TERMINAL_STATUSES = {"completed", "failed"}


def _new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:8]}"


@router.post("", response_model=CreateRunResponse, status_code=202)
def create_run(
    body: CreateRunRequest,
    background_tasks: BackgroundTasks,
    dataset_store: DatasetStore = Depends(get_dataset_store),
    split_store: SplitStore = Depends(get_split_store),
    model_store: InMemoryModelStore = Depends(get_model_store),
    run_store: InMemoryRunStore = Depends(get_run_store),
    llm_provider=Depends(get_llm_provider),
) -> CreateRunResponse:
    if not dataset_store.exists(body.dataset_id):
        raise HTTPException(status_code=404, detail=f"Dataset '{body.dataset_id}' does not exist.")

    run_id = _new_run_id()

    # Batch 5 fix (state isolation): clean_node mutates the dataset
    # stored under state.dataset_id IN PLACE (see graph.py's documented
    # invariant) — every run therefore executes against a private,
    # run-scoped copy of the uploaded dataset, never the uploaded
    # dataset_id itself. Without this, a second run against the same
    # uploaded dataset_id (a normal, UI-supported workflow — datasets
    # persist and are re-selectable) would silently execute against
    # whatever the FIRST run's cleaning/feature-engineering already
    # mutated it into, not the original upload; concurrently in-flight
    # runs against the same uploaded dataset_id would also race on the
    # same mutable rows. DatasetStore.get()/save() already copy
    # internally, so this get()+save() round trip is a genuine,
    # independent clone. display_dataset_id keeps GET /runs/{run_id}
    # reporting the dataset the USER actually selected, not this
    # internal clone id.
    run_dataset_id = f"{run_id}_data"
    dataset_store.save(run_dataset_id, dataset_store.get(body.dataset_id))

    initial_state = AgentState(
        run_id=run_id,
        dataset_id=run_dataset_id,
        target_column=body.target_column,
        max_retries=body.max_retries,
    )
    # Created synchronously here (not inside the background task) so a
    # client polling GET /runs/{run_id} immediately after this 202
    # response can never race a not-yet-started background task.
    run_store.create(run_id, initial_state, display_dataset_id=body.dataset_id)

    graph = build_graph(dataset_store, split_store, model_store, llm_provider)
    background_tasks.add_task(
        stream_with_tracing, graph, initial_state, run_store,
        config={"recursion_limit": _GRAPH_RECURSION_LIMIT},
    )

    return CreateRunResponse(run_id=run_id, status="running")


@router.get("/{run_id}", response_model=RunStatusResponse)
def get_run_status(
    run_id: str, run_store: InMemoryRunStore = Depends(get_run_store)
) -> RunStatusResponse:
    try:
        record = run_store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")

    return RunStatusResponse(
        run_id=record.run_id,
        dataset_id=record.dataset_id,
        target_column=record.target_column,
        status=record.status,
        current_node=record.current_node,
        attempt=record.attempt,
        plan_history=record.plan_history,
    )


@router.get("/{run_id}/result", response_model=RunResultResponse)
def get_run_result(
    run_id: str, run_store: InMemoryRunStore = Depends(get_run_store)
) -> RunResultResponse:
    try:
        record = run_store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")

    if record.status not in _TERMINAL_STATUSES or record.final_state is None:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' is still '{record.status}' — no result yet.",
        )

    final_state = record.final_state
    return RunResultResponse(
        run_id=run_id,
        status=record.status,
        validation=final_state.validation,
        comparison=final_state.comparison,
        baseline=final_state.baseline,
        failure=final_state.failure,
        reproducibility=final_state.reproducibility,
        model_results=final_state.model_results,
        evaluation_results=final_state.evaluation_results,
        error=final_state.error,
    )


@router.get("/{run_id}/summary", response_model=RunSummary)
def get_run_summary(
    run_id: str, run_store: InMemoryRunStore = Depends(get_run_store)
) -> RunSummary:
    """
    Pre-6A Polish: a single, clean, top-level aggregation of state
    already computed elsewhere in this run (retry/REPLAN count, each
    candidate model's scores, the winning model + its deterministic
    selection justification, the operations actually executed, and
    guardrail status/checks) — see build_run_summary(). Gated on
    terminal status exactly like GET /runs/{run_id}/result, since it
    reads the same record.final_state.
    """
    try:
        record = run_store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")

    if record.status not in _TERMINAL_STATUSES or record.final_state is None:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' is still '{record.status}' — no summary yet.",
        )

    return build_run_summary(run_id, record.final_state)


@router.get("/{run_id}/timeline", response_model=ExecutionTimeline)
def get_run_timeline(
    run_id: str, run_store: InMemoryRunStore = Depends(get_run_store)
) -> ExecutionTimeline:
    """
    Pre-6A Polish: a high-level phase timeline derived entirely from
    this run's existing TraceEvent stream — see
    build_execution_timeline(). Unlike /result and /summary, this is
    available at any time (including mid-run): it reflects whatever
    phases have completed so far, exactly like the live SSE
    /runs/{run_id}/events feed it's built from.
    """
    if not run_store.exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")

    events = run_store.get_events(run_id)
    return build_execution_timeline(run_id, events)


@router.get("/{run_id}/learn/explanation", response_model=RunExplanation)
def get_run_learn_explanation(
    run_id: str, run_store: InMemoryRunStore = Depends(get_run_store)
) -> RunExplanation:
    """
    Batch 6A (PIPER Learn: Learn-Explain) — a read-only, deterministic,
    template-based explanation of this run, grounded entirely in real
    evidence already computed elsewhere in the run (see
    build_run_explanation()). Structurally incapable of influencing
    the run it explains: this endpoint only ever reads
    record.final_state, never writes to RunStore or AgentState. Gated
    on terminal status exactly like /result and /summary, since it
    reads the same record.final_state.
    """
    try:
        record = run_store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")

    if record.status not in _TERMINAL_STATUSES or record.final_state is None:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' is still '{record.status}' — no explanation yet.",
        )

    return build_run_explanation(run_id, record.final_state)


@router.post("/{run_id}/explore", response_model=ExplorationResult, status_code=201)
def create_exploration(
    run_id: str,
    body: CreateExplorationRequest,
    run_store: InMemoryRunStore = Depends(get_run_store),
    split_store: SplitStore = Depends(get_split_store),
    model_store: InMemoryModelStore = Depends(get_model_store),
    exploration_store: InMemoryExplorationStore = Depends(get_exploration_store),
) -> ExplorationResult:
    """
    Batch 6B (PIPER Learn: Learn-Explore) — controlled, single-variable
    exploration of an alternative to a model this run already trained
    (see explore_alternative()'s own docstring for the full contract).
    Synchronous: a single sklearn fit against an already-split dataset
    takes seconds, not minutes, so unlike POST /runs this needs no
    BackgroundTasks/polling. Never modifies the original run: only
    reads record.final_state, never calls run_store.update() for
    run_id — every new artifact (model, evaluation, comparison) is
    either a brand-new, additive ModelStore entry or this exploration's
    own ExplorationStore record, isolated by experiment_id.
    """
    try:
        record = run_store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")

    if record.status not in _TERMINAL_STATUSES or record.final_state is None:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' is still '{record.status}' — nothing trained to explore yet.",
        )

    run_model_ids = [m.model_id for m in record.final_state.model_results]

    result = explore_alternative(
        run_id, run_model_ids, body.base_model_id, split_store, model_store,
        new_algorithm=body.new_algorithm,
        hyperparameter_name=body.hyperparameter_name,
        hyperparameter_value=body.hyperparameter_value,
    )
    if not result.success:
        status_code = 404 if result.error.code == "model_not_found" else 400
        raise HTTPException(status_code=status_code, detail=result.message)

    exploration_store.save(result.data)
    return result.data


@router.get("/{run_id}/explore", response_model=list[ExplorationResult])
def list_explorations(
    run_id: str,
    run_store: InMemoryRunStore = Depends(get_run_store),
    exploration_store: InMemoryExplorationStore = Depends(get_exploration_store),
) -> list[ExplorationResult]:
    if not run_store.exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")
    return exploration_store.list_for_run(run_id)


@router.get("/{run_id}/explore/{experiment_id}", response_model=ExplorationResult)
def get_exploration(
    run_id: str,
    experiment_id: str,
    exploration_store: InMemoryExplorationStore = Depends(get_exploration_store),
) -> ExplorationResult:
    try:
        result = exploration_store.get(experiment_id)
    except ExplorationNotFoundError:
        raise HTTPException(status_code=404, detail=f"Exploration '{experiment_id}' does not exist.")
    if result.run_id != run_id:
        raise HTTPException(status_code=404, detail=f"Exploration '{experiment_id}' does not belong to run '{run_id}'.")
    return result


@router.get("/{run_id}/events")
async def stream_run_events(
    run_id: str, run_store: InMemoryRunStore = Depends(get_run_store)
) -> StreamingResponse:
    """
    Server-Sent Events: live TraceEvents as they're appended to
    run_store by the in-flight background execution (see
    stream_with_tracing()) — one `data: <TraceEvent JSON>\\n\\n` line
    per event, in arrival order. Closes once the run reaches a
    terminal status AND one additional grace-period poll has confirmed
    no further events are still landing (stream_with_tracing() appends
    a handful of detail/summary events in the moments immediately after
    the run_store status update that first goes terminal — see its own
    docstring — so closing on the very first terminal observation could
    truncate the very last few events).
    """
    if not run_store.exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")

    async def event_generator():
        sent = 0
        grace_done = False
        while True:
            events = run_store.get_events(run_id)
            for event in events[sent:]:
                yield f"data: {event.model_dump_json()}\n\n"
            sent = len(events)

            record = run_store.get(run_id)
            if record.status in _TERMINAL_STATUSES:
                if grace_done:
                    break
                grace_done = True
            await asyncio.sleep(0.2)

    return StreamingResponse(
        event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )
