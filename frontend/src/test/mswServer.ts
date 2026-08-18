import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { API_BASE_URL } from '@/lib/api'
import {
  fixtureCompletedResult,
  fixtureDatasetProfile,
  fixtureFailedResult,
} from './fixtures'
import type { CreateRunRequest, CreateRunResponse, DatasetUploadResponse, RunStatusResponse } from '@/lib/types'

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
]

export const server = setupServer(...handlers)
