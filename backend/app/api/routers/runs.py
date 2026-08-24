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
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.agent import AgentState, build_graph
from app.agent.productization import (
    build_decision_trace,
    build_evidence_export,
    build_intervention,
    build_verdict,
)
from app.agent.run_summary import build_run_summary
from app.agent.timeline import build_execution_timeline
from app.agent.tools.exploration import explore_alternative
from app.agent.tracing import stream_with_tracing
from app.governance import (
    GOVERNANCE_DOCUMENT_NAMES,
    assemble_governance_bundle,
    render_governance_document,
)
from app.artifacts.errors import ArtifactEligibilityError, ArtifactParityError
from app.artifacts.publisher import (
    DOWNLOADABLE_FILES,
    publish_run_artifacts,
    read_artifact_status,
)
from app.api.dependencies import (
    get_artifact_dir,
    get_dataset_store,
    get_exploration_store,
    get_llm_provider,
    get_model_store,
    get_run_store,
    get_split_store,
)
from app.api.schemas import (
    ArtifactFileListResponse,
    ArtifactStatusResponse,
    CreateExplorationRequest,
    CreateRunRequest,
    CreateRunResponse,
    RunListItem,
    RunListResponse,
    RunResultResponse,
    RunStatusResponse,
)
from app.learning.explain import build_run_explanation
from app.schemas.execution_timeline import ExecutionTimeline
from app.schemas.exploration import ExplorationResult
from app.schemas.learning import RunExplanation
from app.schemas.productization import (
    DecisionTrace,
    EvidenceExport,
    HumanInterventionPackage,
    PiperVerdict,
    ReplayResponse,
)
from app.schemas.governance import FairnessReport, GovernanceBundle
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


@router.get("", response_model=RunListResponse)
def list_runs(run_store: InMemoryRunStore = Depends(get_run_store)) -> RunListResponse:
    """Persisted run history. Does not invoke the LLM."""
    items = []
    for record in run_store.list():
        items.append(
            RunListItem(
                run_id=record.run_id,
                dataset_id=record.dataset_id,
                target_column=record.target_column,
                status=record.status,
                current_node=record.current_node,
                attempt=record.attempt,
                created_at=getattr(record, "created_at", None),
                updated_at=getattr(record, "updated_at", None),
            )
        )
    return RunListResponse(runs=items)


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


def _load_run_or_404(run_id: str, run_store: InMemoryRunStore):
    try:
        return run_store.get(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")


@router.get("/{run_id}/decision-trace", response_model=DecisionTrace)
def get_run_decision_trace(
    run_id: str, run_store: InMemoryRunStore = Depends(get_run_store)
) -> DecisionTrace:
    """
    V1.2 productization: the operator-facing decision stages for this
    run, derived from existing TraceEvents plus recorded planning
    attempts. Available mid-run (like /timeline). Never includes LLM
    reasoning. Does not affect execution.
    """
    record = _load_run_or_404(run_id, run_store)
    events = run_store.get_events(run_id)
    state = record.final_state
    target = record.target_column
    if state is not None:
        target = getattr(state, "target_column", target)
    return build_decision_trace(
        run_id, record.status, events, state, target_column=target,
    )


@router.get("/{run_id}/verdict", response_model=PiperVerdict)
def get_run_verdict(
    run_id: str, run_store: InMemoryRunStore = Depends(get_run_store)
) -> PiperVerdict:
    """Deterministic final PIPER verdict. Terminal runs only."""
    record = _load_run_or_404(run_id, run_store)
    if record.status not in _TERMINAL_STATUSES or record.final_state is None:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' is still '{record.status}' — no verdict yet.",
        )
    return build_verdict(run_id, record.status, record.final_state)


@router.get("/{run_id}/intervention", response_model=HumanInterventionPackage)
def get_run_intervention(
    run_id: str, run_store: InMemoryRunStore = Depends(get_run_store)
) -> HumanInterventionPackage:
    """Human-intervention package. Terminal runs only. No chain-of-thought."""
    record = _load_run_or_404(run_id, run_store)
    if record.status not in _TERMINAL_STATUSES or record.final_state is None:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' is still '{record.status}' — no intervention package yet.",
        )
    return build_intervention(run_id, record.status, record.final_state)


@router.get("/{run_id}/evidence", response_model=EvidenceExport)
def get_run_evidence(
    run_id: str, run_store: InMemoryRunStore = Depends(get_run_store)
) -> EvidenceExport:
    """JSON evidence export. Terminal runs only."""
    record = _load_run_or_404(run_id, run_store)
    if record.status not in _TERMINAL_STATUSES or record.final_state is None:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' is still '{record.status}' — no evidence export yet.",
        )
    events = run_store.get_events(run_id)
    state = record.final_state
    export = build_evidence_export(
        run_id,
        record.status,
        events,
        state,
        dataset_id=record.dataset_id,
        target_column=getattr(state, "target_column", record.target_column),
    )
    if hasattr(run_store, "save_evidence"):
        run_store.save_evidence(run_id, export.model_dump(mode="json"))
    return export


@router.get("/{run_id}/replay", response_model=ReplayResponse)
def replay_run(
    run_id: str, run_store: InMemoryRunStore = Depends(get_run_store)
) -> ReplayResponse:
    """
    Rebuild decision evidence from persisted events + terminal state.
    Does not call generate_plan() or any LLM provider.
    """
    record = _load_run_or_404(run_id, run_store)
    if record.status not in _TERMINAL_STATUSES or record.final_state is None:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' is still '{record.status}' — nothing to replay yet.",
        )
    events = run_store.get_events(run_id)
    state = record.final_state
    target = getattr(state, "target_column", record.target_column)
    evidence = build_evidence_export(
        run_id,
        record.status,
        events,
        state,
        dataset_id=record.dataset_id,
        target_column=target,
    )
    return ReplayResponse(
        run_id=run_id,
        llm_invoked=False,
        source="persisted_events_and_state",
        status=record.status,
        decision_trace=evidence.decision_trace,
        verdict=evidence.verdict,
        intervention=evidence.intervention,
        evidence=evidence,
    )


def _parse_subgroup_columns(column: list[str]) -> list[str]:
    names: list[str] = []
    for item in column:
        names.extend(part.strip() for part in item.split(",") if part.strip())
    return list(dict.fromkeys(names))


def _require_terminal_run(run_id: str, run_store: InMemoryRunStore):
    record = _load_run_or_404(run_id, run_store)
    if record.status not in _TERMINAL_STATUSES or record.final_state is None:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' is still '{record.status}' — no governance export yet.",
        )
    return record


def _governance_for_record(
    record,
    *,
    model_store: InMemoryModelStore,
    split_store: SplitStore,
    dataset_store: DatasetStore,
    artifact_dir: Path,
    subgroup_columns: list[str],
) -> GovernanceBundle:
    return assemble_governance_bundle(
        record.run_id,
        run_status=record.status,
        state=record.final_state,
        dataset_id=record.dataset_id,
        model_store=model_store,
        split_store=split_store,
        dataset_store=dataset_store,
        artifact_root=artifact_dir,
        subgroup_columns=subgroup_columns,
    )


@router.get("/{run_id}/governance", response_model=GovernanceBundle)
def get_run_governance(
    run_id: str,
    column: list[str] = Query(default=[]),
    run_store: InMemoryRunStore = Depends(get_run_store),
    model_store: InMemoryModelStore = Depends(get_model_store),
    split_store: SplitStore = Depends(get_split_store),
    dataset_store: DatasetStore = Depends(get_dataset_store),
    artifact_dir: Path = Depends(get_artifact_dir),
) -> GovernanceBundle:
    """
    Deterministic Model Card, Data Card, fingerprints, and optional
    subgroup analysis. Never calls an LLM.
    """
    record = _require_terminal_run(run_id, run_store)
    return _governance_for_record(
        record,
        model_store=model_store,
        split_store=split_store,
        dataset_store=dataset_store,
        artifact_dir=artifact_dir,
        subgroup_columns=_parse_subgroup_columns(column),
    )


@router.get("/{run_id}/governance/fairness", response_model=FairnessReport)
def get_run_fairness(
    run_id: str,
    column: list[str] = Query(default=[]),
    run_store: InMemoryRunStore = Depends(get_run_store),
    model_store: InMemoryModelStore = Depends(get_model_store),
    split_store: SplitStore = Depends(get_split_store),
    dataset_store: DatasetStore = Depends(get_dataset_store),
    artifact_dir: Path = Depends(get_artifact_dir),
) -> FairnessReport:
    record = _require_terminal_run(run_id, run_store)
    bundle = _governance_for_record(
        record,
        model_store=model_store,
        split_store=split_store,
        dataset_store=dataset_store,
        artifact_dir=artifact_dir,
        subgroup_columns=_parse_subgroup_columns(column),
    )
    return bundle.fairness


@router.get("/{run_id}/governance/documents/{filename}")
def download_governance_document(
    run_id: str,
    filename: str,
    column: list[str] = Query(default=[]),
    run_store: InMemoryRunStore = Depends(get_run_store),
    model_store: InMemoryModelStore = Depends(get_model_store),
    split_store: SplitStore = Depends(get_split_store),
    dataset_store: DatasetStore = Depends(get_dataset_store),
    artifact_dir: Path = Depends(get_artifact_dir),
) -> Response:
    if filename not in GOVERNANCE_DOCUMENT_NAMES:
        raise HTTPException(status_code=404, detail=f"Governance document '{filename}' is not published.")
    record = _require_terminal_run(run_id, run_store)
    bundle = _governance_for_record(
        record,
        model_store=model_store,
        split_store=split_store,
        dataset_store=dataset_store,
        artifact_dir=artifact_dir,
        subgroup_columns=_parse_subgroup_columns(column),
    )
    media, body = render_governance_document(bundle, filename)
    return Response(
        content=body,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _artifact_status_response(payload: dict) -> ArtifactStatusResponse:
    return ArtifactStatusResponse(
        run_id=payload["run_id"],
        artifact_status=payload["artifact_status"],
        parity_status=payload["parity_status"],
        winning_model_id=payload.get("winning_model_id"),
        algorithm=payload.get("algorithm"),
        files=list(payload.get("files") or []),
        error=payload.get("error"),
        created_at=payload.get("created_at"),
        parity=payload.get("parity"),
    )


@router.post("/{run_id}/artifacts", response_model=ArtifactStatusResponse, status_code=201)
def generate_run_artifacts(
    run_id: str,
    run_store: InMemoryRunStore = Depends(get_run_store),
    model_store: InMemoryModelStore = Depends(get_model_store),
    split_store: SplitStore = Depends(get_split_store),
    artifact_dir: Path = Depends(get_artifact_dir),
) -> ArtifactStatusResponse:
    """
    Compile a verified completed run into a portable ML artifact bundle.
    Does not invoke Ollama. Failed/unsafe runs are rejected.
    """
    if not run_store.exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")
    try:
        payload = publish_run_artifacts(
            run_id,
            run_store=run_store,
            model_store=model_store,
            split_store=split_store,
            artifact_root=artifact_dir,
        )
    except ArtifactEligibilityError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": exc.message, "details": exc.details},
        ) from exc
    except ArtifactParityError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": exc.message, "details": exc.details},
        ) from exc
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")
    return _artifact_status_response(payload)


@router.get("/{run_id}/artifacts", response_model=ArtifactStatusResponse)
def get_run_artifact_status(
    run_id: str,
    run_store: InMemoryRunStore = Depends(get_run_store),
    artifact_dir: Path = Depends(get_artifact_dir),
) -> ArtifactStatusResponse:
    if not run_store.exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")
    return _artifact_status_response(read_artifact_status(artifact_dir, run_id))


@router.get("/{run_id}/artifacts/files", response_model=ArtifactFileListResponse)
def list_run_artifact_files(
    run_id: str,
    run_store: InMemoryRunStore = Depends(get_run_store),
    artifact_dir: Path = Depends(get_artifact_dir),
) -> ArtifactFileListResponse:
    if not run_store.exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")
    payload = read_artifact_status(artifact_dir, run_id)
    return ArtifactFileListResponse(
        run_id=run_id,
        artifact_status=payload["artifact_status"],
        files=list(payload.get("files") or []),
    )


@router.get("/{run_id}/artifacts/files/{filename}")
def download_run_artifact_file(
    run_id: str,
    filename: str,
    run_store: InMemoryRunStore = Depends(get_run_store),
    artifact_dir: Path = Depends(get_artifact_dir),
) -> FileResponse:
    if not run_store.exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' does not exist.")
    if filename not in DOWNLOADABLE_FILES:
        raise HTTPException(status_code=404, detail=f"Artifact file '{filename}' is not published.")
    path = (artifact_dir / run_id / filename).resolve()
    root = (artifact_dir / run_id).resolve()
    if root not in path.parents and path != root / filename:
        raise HTTPException(status_code=404, detail=f"Artifact file '{filename}' is not published.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact file '{filename}' is not available.")
    media = "application/octet-stream"
    if filename.endswith(".json") or filename.endswith(".ipynb"):
        media = "application/json"
    elif filename.endswith(".py"):
        media = "text/x-python"
    return FileResponse(path, media_type=media, filename=filename)


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
