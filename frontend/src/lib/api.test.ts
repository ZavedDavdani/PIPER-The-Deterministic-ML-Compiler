import { describe, expect, it } from 'vitest'
import { http, HttpResponse } from 'msw'
import { API_BASE_URL, ApiError, createExploration, createRun, getDataset, getRunResult, listDatasets, uploadDataset } from './api'
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

  it('createExploration sends application/json and maps backend response fields', async () => {
    let capturedContentType: string | null = null
    let capturedBody: Record<string, unknown> | null = null

    server.use(
      http.post(`${API_BASE_URL}/runs/:runId/explore`, async ({ request }) => {
        capturedContentType = request.headers.get('Content-Type')
        capturedBody = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({
          experiment_id: 'exp_test_001',
          run_id: 'run_test_001',
          base_model_id: 'model_lr001',
          variable_changed: {
            kind: 'model',
            name: 'algorithm',
            old_value: 'logistic_regression',
            new_value: 'random_forest',
          },
          training: {
            model_id: 'model_rf_exp',
            algorithm: 'random_forest',
            parameters: {},
            split_id: 'split_001',
            training_rows: 5634,
            feature_count: 20,
            training_duration_seconds: 0.5,
          },
          evaluation: {
            model_id: 'model_rf_exp',
            split_id: 'split_001',
            accuracy: 0.84,
            precision: 0.82,
            recall: 0.81,
            f1: 0.815,
            roc_auc: 0.87,
            confusion_matrix: { tp: 120, tn: 500, fp: 50, fn: 80 },
            test_rows: 750,
          },
          comparison_vs_base: {
            models: [
              { model_id: 'model_lr001', f1: 0.81 },
              { model_id: 'model_rf_exp', f1: 0.815 },
            ],
            recommended_model_id: 'model_rf_exp',
            justification: 'random_forest selected: F1=0.8150 vs 0.8100.',
          },
        })
      }),
    )

    const result = await createExploration('run_test_001', {
      base_model_id: 'model_lr001',
      new_algorithm: 'random_forest',
    })

    expect(capturedContentType).toBe('application/json')
    expect(capturedBody).toEqual({
      base_model_id: 'model_lr001',
      new_algorithm: 'random_forest',
    })
    expect(result.variable.base_value).toBe('logistic_regression')
    expect(result.comparison.base_metric_value).toBe(0.81)
    expect(result.comparison.new_metric_value).toBe(0.815)
  })

  it('createExploration surfaces readable validation errors for malformed payloads', async () => {
    server.use(
      http.post(`${API_BASE_URL}/runs/:runId/explore`, () =>
        HttpResponse.json(
          {
            detail: [
              {
                type: 'model_attributes_type',
                loc: ['body'],
                msg: 'Input should be a valid dictionary or object to extract fields from',
                input: '{"base_model_id":"model_x"}',
              },
            ],
          },
          { status: 422 },
        ),
      ),
    )

    await expect(
      createExploration('run_test_001', {
        base_model_id: 'model_lr001',
        new_algorithm: 'random_forest',
      }),
    ).rejects.toMatchObject({
      status: 422,
      message: 'The request payload was malformed. Please try again.',
    })
  })
})
