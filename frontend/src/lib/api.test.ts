import { describe, expect, it } from 'vitest'
import { http, HttpResponse } from 'msw'
import { API_BASE_URL, ApiError, createRun, getDataset, getRunResult, listDatasets, uploadDataset } from './api'
import { server } from '@/test/mswServer'
import { fixtureCompletedResult, fixtureDatasetProfile } from '@/test/fixtures'

describe('api client', () => {
  it('uploadDataset posts a multipart form and returns the real response shape', async () => {
    const file = new File(['a,b\n1,2\n'], 'data.csv', { type: 'text/csv' })
    const response = await uploadDataset(file)
    expect(response.dataset_id).toBe(fixtureDatasetProfile.dataset_id)
    expect(response.columns).toContain('Churn')
  })

  it('listDatasets returns dataset_ids', async () => {
    const response = await listDatasets()
    expect(response.dataset_ids).toContain(fixtureDatasetProfile.dataset_id)
  })

  it('getDataset returns the real DatasetProfile shape', async () => {
    const profile = await getDataset(fixtureDatasetProfile.dataset_id)
    expect(profile.rows).toBe(7043)
    expect(profile.column_profiles).toHaveLength(4)
  })

  it('getDataset throws ApiError with the backend detail message on 404', async () => {
    await expect(getDataset('dataset_missing')).rejects.toMatchObject({
      status: 404,
    })
  })

  it('createRun posts the request body and returns run_id/status', async () => {
    const response = await createRun({ dataset_id: fixtureDatasetProfile.dataset_id, target_column: 'Churn' })
    expect(response.status).toBe('running')
    expect(response.run_id).toBeTruthy()
  })

  it('createRun surfaces a 404 ApiError for an unknown dataset', async () => {
    await expect(createRun({ dataset_id: 'nope', target_column: 'Churn' })).rejects.toBeInstanceOf(ApiError)
  })

  it('getRunResult surfaces a 409 ApiError while the run is still in progress', async () => {
    server.use(
      http.get(`${API_BASE_URL}/runs/:runId/result`, () =>
        HttpResponse.json({ detail: "Run 'x' is still 'running'." }, { status: 409 }),
      ),
    )
    await expect(getRunResult('run_x')).rejects.toMatchObject({ status: 409 })
  })

  it('getRunResult returns the real result shape once terminal', async () => {
    server.use(
      http.get(`${API_BASE_URL}/runs/:runId/result`, () => HttpResponse.json(fixtureCompletedResult)),
    )
    const result = await getRunResult('run_test0001')
    expect(result.status).toBe('completed')
    expect(result.comparison?.models.length).toBeGreaterThan(0)
  })
})
