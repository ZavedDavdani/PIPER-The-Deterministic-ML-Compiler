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
      notes: ['LLM reasoning / chain-of-thought is intentionally omitted.'],
    })
  }),
]

export const server = setupServer(...handlers)
