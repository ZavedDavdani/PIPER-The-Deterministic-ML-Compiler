import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { API_BASE_URL } from '@/lib/api'
import {
  fixtureCompletedResult,
  fixtureDatasetProfile,
  fixtureFailedResult,
} from './fixtures'
import type {
  CreateRunRequest,
  CreateRunResponse,
  DatasetUploadResponse,
  DecisionTrace,
  HumanInterventionPackage,
  PiperVerdict,
  RunStatusResponse,
  GovernanceBundle,
} from '@/lib/types'

/**
 * MSW handlers matching the REAL FastAPI contract (paths, methods,
 * status codes, JSON shapes) exactly — verified against
 * backend/app/api/routers/{datasets,runs}.py. Tests using these are
 * checking real request/response handling in the frontend, against a
 * simulated network layer, not hardcoded UI mock data.
 */

let runStatusOverride: RunStatusResponse['status'] = 'completed'
let resultOverride = fixtureCompletedResult

export function setNextRunOutcome(status: RunStatusResponse['status'], result = fixtureCompletedResult) {
  runStatusOverride = status
  resultOverride = result
}

export function resetRunOutcome() {
  runStatusOverride = 'completed'
  resultOverride = fixtureCompletedResult
}

const STAGE_IDS = [
  'LLM_PROPOSED',
  'VALIDATED',
  'ADEQUACY',
  'REPLAN',
  'EXECUTION',
  'TRAINING',
  'EVALUATION',
  'GUARDRAILS',
  'FINAL_VERDICT',
] as const

function fixtureDecisionTrace(runId: string, status: RunStatusResponse['status']): DecisionTrace {
  const terminal = status === 'completed' || status === 'failed'
  return {
    run_id: runId,
    run_status: status,
    stages: STAGE_IDS.map((id) => ({
      id,
      label: id.replaceAll('_', ' '),
      status: terminal ? (status === 'failed' && id === 'FINAL_VERDICT' ? 'failed' : 'passed') : 'pending',
      summary: terminal ? 'Recorded.' : 'Waiting.',
      attempt: 0,
      evidence: {},
    })),
    planning_attempts: [],
    plan_diffs: [],
  }
}

function fixtureVerdict(runId: string, status: RunStatusResponse['status']): PiperVerdict {
  const accepted = status === 'completed'
  return {
    run_id: runId,
    outcome: accepted ? 'ACCEPTED' : 'REJECTED',
    reason_code: accepted ? 'ACCEPTED_GUARDRAILS_PASSED' : 'REJECTED',
    summary: accepted ? 'PIPER accepted this run.' : 'PIPER rejected this run.',
    retry_count: 0,
    max_retries: 2,
    structurally_valid_plan: accepted,
    adequacy_passed: accepted,
    guardrails_passed: accepted,
    human_intervention_required: status === 'failed',
    executed: accepted,
  }
}

function fixtureIntervention(runId: string, status: RunStatusResponse['status']): HumanInterventionPackage {
  return {
    run_id: runId,
    required: status === 'failed',
    headline: status === 'failed' ? 'Human review is required.' : 'No human intervention required.',
    failure_category: status === 'failed' ? 'DUPLICATE_PLAN' : null,
    failure_message: status === 'failed' ? 'duplicate' : null,
    retry_count: 0,
    max_retries: 2,
    last_proposed_steps: [],
    structural_violations: [],
    material_adequacy_findings: [],
    advisory_adequacy_findings: [],
    preserved_valid_steps: [],
    implicated_steps: [],
    recommended_actions: status === 'failed' ? ['Read the structured failure.'] : [],
    blocked_invalid_execution: status === 'failed',
  }
}

export const handlers = [
  http.post(`${API_BASE_URL}/datasets`, async () => {
    const response: DatasetUploadResponse = {
      dataset_id: fixtureDatasetProfile.dataset_id,
      filename: 'telco.csv',
      rows: fixtureDatasetProfile.rows,
      columns: fixtureDatasetProfile.column_profiles.map((c) => c.name),
    }
    return HttpResponse.json(response, { status: 201 })
  }),

  http.get(`${API_BASE_URL}/datasets`, () => {
    return HttpResponse.json({ dataset_ids: [fixtureDatasetProfile.dataset_id] })
  }),

  http.get(`${API_BASE_URL}/datasets/:datasetId`, ({ params }) => {
    if (params.datasetId !== fixtureDatasetProfile.dataset_id) {
      return HttpResponse.json({ detail: 'not found' }, { status: 404 })
    }
    return HttpResponse.json(fixtureDatasetProfile)
  }),

  http.post(`${API_BASE_URL}/runs`, async ({ request }) => {
    const body = (await request.json()) as CreateRunRequest
    if (body.dataset_id !== fixtureDatasetProfile.dataset_id) {
      return HttpResponse.json({ detail: `Dataset '${body.dataset_id}' does not exist.` }, { status: 404 })
    }
    const response: CreateRunResponse = { run_id: resultOverride.run_id, status: 'running' }
    return HttpResponse.json(response, { status: 202 })
  }),

  http.get(`${API_BASE_URL}/runs`, () => {
    return HttpResponse.json({
      runs: [
        {
          run_id: resultOverride.run_id,
          dataset_id: fixtureDatasetProfile.dataset_id,
          target_column: 'Churn',
          status: runStatusOverride,
          current_node: null,
          attempt: 0,
          created_at: null,
          updated_at: null,
        },
      ],
    })
  }),

  http.get(`${API_BASE_URL}/settings/ollama`, () => {
    return HttpResponse.json({
      host: 'http://localhost:11434',
      model: 'qwen3:4b',
      keep_alive: '10m',
      reachable: false,
      models: [],
      error: 'unreachable',
    })
  }),

  http.put(`${API_BASE_URL}/settings/ollama`, async ({ request }) => {
    const body = (await request.json()) as { model?: string; host?: string }
    return HttpResponse.json({
      host: body.host ?? 'http://localhost:11434',
      model: body.model ?? 'qwen3:4b',
      keep_alive: '10m',
      reachable: false,
      models: [],
      error: 'unreachable',
    })
  }),

  http.get(`${API_BASE_URL}/runs/:runId`, ({ params }) => {
    const response: RunStatusResponse = {
      run_id: String(params.runId),
      dataset_id: fixtureDatasetProfile.dataset_id,
      target_column: 'Churn',
      status: runStatusOverride,
      current_node: runStatusOverride === 'completed' || runStatusOverride === 'failed' ? 'report' : 'train',
      attempt: 0,
      plan_history: [],
    }
    return HttpResponse.json(response)
  }),

  http.get(`${API_BASE_URL}/runs/:runId/result`, () => {
    if (runStatusOverride !== 'completed' && runStatusOverride !== 'failed') {
      return HttpResponse.json({ detail: 'still running' }, { status: 409 })
    }
    return HttpResponse.json(runStatusOverride === 'failed' ? fixtureFailedResult : resultOverride)
  }),

  http.get(`${API_BASE_URL}/runs/:runId/decision-trace`, ({ params }) => {
    return HttpResponse.json(fixtureDecisionTrace(String(params.runId), runStatusOverride))
  }),

  http.get(`${API_BASE_URL}/runs/:runId/verdict`, ({ params }) => {
    if (runStatusOverride !== 'completed' && runStatusOverride !== 'failed') {
      return HttpResponse.json({ detail: 'still running' }, { status: 409 })
    }
    return HttpResponse.json(fixtureVerdict(String(params.runId), runStatusOverride))
  }),

  http.get(`${API_BASE_URL}/runs/:runId/intervention`, ({ params }) => {
    if (runStatusOverride !== 'completed' && runStatusOverride !== 'failed') {
      return HttpResponse.json({ detail: 'still running' }, { status: 409 })
    }
    return HttpResponse.json(fixtureIntervention(String(params.runId), runStatusOverride))
  }),

  http.get(`${API_BASE_URL}/runs/:runId/evidence`, ({ params }) => {
    if (runStatusOverride !== 'completed' && runStatusOverride !== 'failed') {
      return HttpResponse.json({ detail: 'still running' }, { status: 409 })
    }
    const runId = String(params.runId)
    return HttpResponse.json({
      schema_version: 'piper.evidence.v1',
      run_id: runId,
      dataset_id: fixtureDatasetProfile.dataset_id,
      target_column: 'Churn',
      status: runStatusOverride,
      decision_trace: fixtureDecisionTrace(runId, runStatusOverride),
      verdict: fixtureVerdict(runId, runStatusOverride),
      intervention: fixtureIntervention(runId, runStatusOverride),
      notes: ['LLM chain-of-thought is intentionally omitted.'],
    })
  }),

  http.get(`${API_BASE_URL}/runs/:runId/replay`, ({ params }) => {
    if (runStatusOverride !== 'completed' && runStatusOverride !== 'failed') {
      return HttpResponse.json({ detail: 'still running' }, { status: 409 })
    }
    const runId = String(params.runId)
    const evidence = {
      schema_version: 'piper.evidence.v1',
      run_id: runId,
      dataset_id: fixtureDatasetProfile.dataset_id,
      target_column: 'Churn',
      status: runStatusOverride,
      decision_trace: fixtureDecisionTrace(runId, runStatusOverride),
      verdict: fixtureVerdict(runId, runStatusOverride),
      intervention: fixtureIntervention(runId, runStatusOverride),
      notes: ['LLM chain-of-thought is intentionally omitted.'],
    }
    return HttpResponse.json({
      run_id: runId,
      llm_invoked: false,
      source: 'persisted_events_and_state',
      status: runStatusOverride,
      decision_trace: evidence.decision_trace,
      verdict: evidence.verdict,
      intervention: evidence.intervention,
      evidence,
    })
  }),

  http.get(`${API_BASE_URL}/runs/:runId/governance`, ({ params }) => {
    if (runStatusOverride !== 'completed' && runStatusOverride !== 'failed') {
      return HttpResponse.json({ detail: 'still running' }, { status: 409 })
    }
    const runId = String(params.runId)
    const importance = {
      status: 'NOT_AVAILABLE' as const,
      method: 'NOT_AVAILABLE',
      algorithm: null,
      rows: [],
      disclaimer: 'These scores describe association, not causal effects.',
      reason: 'MSW fixture has no fitted pipeline.',
    }
    const body: GovernanceBundle = {
      schema_version: 'piper.governance.v1',
      run_id: runId,
      run_status: runStatusOverride,
      model_card: {
        status: runStatusOverride === 'completed' ? 'AVAILABLE' : 'NOT_AVAILABLE',
        run_id: runId,
        dataset_id: fixtureDatasetProfile.dataset_id,
        task_type: 'binary_classification',
        target: 'Churn',
        winning_model_id: runStatusOverride === 'completed' ? 'model_lr001' : null,
        winning_algorithm: runStatusOverride === 'completed' ? 'logistic_regression' : null,
        candidate_models: [],
        evaluation_metrics: runStatusOverride === 'completed' ? [{ name: 'f1', value: 0.5 }] : [],
        baseline_comparison: null,
        train_test_split: null,
        preprocessing_summary: [],
        guardrail_results: [],
        limitations: ['MSW fixture.'],
        artifact_information: { artifact_status: 'NOT_GENERATED' },
        feature_importance: importance,
        reason: null,
      },
      data_card: {
        status: 'AVAILABLE',
        run_id: runId,
        dataset_id: fixtureDatasetProfile.dataset_id,
        rows: fixtureDatasetProfile.rows,
        columns: fixtureDatasetProfile.columns,
        target: 'Churn',
        feature_list: [],
        column_summaries: [],
        numeric_features: [],
        categorical_features: [],
        missingness: [],
        preprocessing_operations: [],
        train_test: null,
        data_quality_findings: [],
        limitations: [],
        reason: null,
      },
      fingerprints: {
        run_id: runId,
        hash_algorithm: 'sha256',
        content_hashes: [],
        metadata: {},
        caveat: 'These fingerprints are tamper-evident content hashes.',
      },
      feature_importance: importance,
      fairness: {
        status: 'NOT_REQUESTED',
        requested_columns: [],
        minimum_group_size: 30,
        positive_class: null,
        reference_group_rule: 'largest-n group',
        groups: [],
        warnings: [],
        disclaimer: 'Statistical subgroup measurements, not a legal compliance decision.',
        reason: 'No subgroup columns were supplied.',
      },
      limitations: [],
      artifact_status: { artifact_status: 'NOT_GENERATED' },
      notes: ['LLM chain-of-thought is intentionally omitted.'],
    }
    return HttpResponse.json(body)
  }),

  http.get(`${API_BASE_URL}/runs/:runId/artifacts`, ({ params }) => {
    return HttpResponse.json({
      run_id: String(params.runId),
      artifact_status: 'NOT_GENERATED',
      parity_status: 'not_run',
      winning_model_id: null,
      algorithm: null,
      files: [],
      error: null,
      created_at: null,
    })
  }),

  http.post(`${API_BASE_URL}/runs/:runId/artifacts`, ({ params }) => {
    if (runStatusOverride !== 'completed') {
      return HttpResponse.json({ detail: { code: 'run_not_verified', message: 'Run is not eligible.' } }, { status: 409 })
    }
    return HttpResponse.json(
      {
        run_id: String(params.runId),
        artifact_status: 'VERIFIED',
        parity_status: 'passed',
        winning_model_id: 'model_lr001',
        algorithm: 'logistic_regression',
        files: [
          'pipeline.joblib',
          'pipeline.py',
          'training_reproduction.ipynb',
          'manifest.json',
          'evidence.json',
          'hashes.json',
        ],
        error: null,
        created_at: '2026-08-23T00:00:00Z',
      },
      { status: 201 },
    )
  }),

  http.get(`${API_BASE_URL}/runs/:runId/deployment`, ({ params }) => {
    return HttpResponse.json({
      run_id: String(params.runId),
      status: 'NOT_READY',
      artifact_status: 'NOT_GENERATED',
      winning_model_id: null,
      algorithm: null,
      required_columns: [],
      checks: [],
      reason: { code: 'artifact_missing', message: 'No artifact bundle exists for this run.' },
    })
  }),
]

export const server = setupServer(...handlers)
