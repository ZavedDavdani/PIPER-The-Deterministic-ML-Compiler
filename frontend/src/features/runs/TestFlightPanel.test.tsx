import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { TestFlightPanel } from './TestFlightPanel'
import { server } from '@/test/mswServer'
import { API_BASE_URL } from '@/lib/api'

describe('TestFlightPanel', () => {
  it('scores unseen CSV against a verified artifact and shows predictions', async () => {
    server.use(
      http.get(`${API_BASE_URL}/runs/:runId/artifacts`, () => {
        return HttpResponse.json({
          run_id: 'run_ok',
          artifact_status: 'VERIFIED',
          parity_status: 'passed',
          winning_model_id: 'model_1',
          algorithm: 'logistic_regression',
          files: ['pipeline.joblib'],
          error: null,
          created_at: '2026-08-24T00:00:00Z',
        })
      }),
      http.get(`${API_BASE_URL}/runs/:runId/deployment`, () => {
        return HttpResponse.json({
          run_id: 'run_ok',
          status: 'READY',
          artifact_status: 'VERIFIED',
          winning_model_id: 'model_1',
          algorithm: 'logistic_regression',
          required_columns: ['cat', 'num'],
          checks: [{ check: 'pipeline_loads', passed: true, detail: null }],
          reason: null,
        })
      }),
      http.post(`${API_BASE_URL}/runs/:runId/test-flight`, () => {
        return HttpResponse.json({
          run_id: 'run_ok',
          artifact_id: 'run_ok',
          winning_model_id: 'model_1',
          algorithm: 'logistic_regression',
          row_count: 2,
          predictions: ['yes', 'no'],
          schema_status: 'valid',
          required_columns: ['cat', 'num'],
          parity: { parity_status: 'passed', mismatched_rows: 0, row_count: 2 },
          data_kind: 'NEW_UNSEEN_DATA',
          sample: [{ cat: 'a', num: 1, prediction: 'yes' }, { cat: 'b', num: 2, prediction: 'no' }],
        })
      }),
    )

    const user = userEvent.setup()
    render(<TestFlightPanel runId="run_ok" runStatus="completed" />)

    expect(await screen.findByText('Test Flight')).toBeInTheDocument()
    expect(screen.getByText(/TRAINING DATA/)).toBeInTheDocument()
    expect(screen.getByText(/NEW UNSEEN DATA/)).toBeInTheDocument()
    expect(await screen.findByText(/Required columns: cat, num/)).toBeInTheDocument()

    const csv = new File(['cat,num\na,1\nb,2\n'], 'new_data.csv', { type: 'text/csv' })
    await user.upload(screen.getByLabelText(/unseen csv/i), csv)
    await user.click(screen.getByRole('button', { name: /run prediction/i }))
    expect(await screen.findByText('valid')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
    expect(screen.getByText('passed')).toBeInTheDocument()
    expect(screen.getByText(/yes/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /download prediction csv/i })).toBeInTheDocument()
  })

  it('does not offer scoring until a verified artifact exists', async () => {
    render(<TestFlightPanel runId="run_plain" runStatus="completed" />)
    expect(await screen.findByText(/generate a verified artifact first/i)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /run prediction/i })).not.toBeInTheDocument()
  })
})
