import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ArtifactPanel } from './ArtifactPanel'
import { http, HttpResponse } from 'msw'
import { server } from '@/test/mswServer'
import { API_BASE_URL } from '@/lib/api'

describe('ArtifactPanel', () => {
  it('shows VERIFIED ARTIFACT and download links after generation', async () => {
    server.use(
      http.get(`${API_BASE_URL}/runs/:runId/artifacts`, () => {
        return HttpResponse.json({
          run_id: 'run_ok',
          artifact_status: 'NOT_GENERATED',
          parity_status: 'not_run',
          winning_model_id: null,
          algorithm: null,
          files: [],
          error: null,
          created_at: null,
        })
      }),
      http.post(`${API_BASE_URL}/runs/:runId/artifacts`, () => {
        return HttpResponse.json(
          {
            run_id: 'run_ok',
            artifact_status: 'VERIFIED',
            parity_status: 'passed',
            winning_model_id: 'model_1',
            algorithm: 'logistic_regression',
            files: ['pipeline.joblib', 'pipeline.py', 'training_reproduction.ipynb', 'manifest.json', 'evidence.json', 'hashes.json'],
            error: null,
            created_at: '2026-08-23T00:00:00Z',
          },
          { status: 201 },
        )
      }),
    )

    render(<ArtifactPanel runId="run_ok" runStatus="completed" />)
    await userEvent.click(await screen.findByRole('button', { name: /generate artifacts/i }))
    expect(await screen.findByText(/verified artifact/i)).toBeInTheDocument()
    expect(screen.getByText(/logistic_regression/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /download fitted pipeline/i })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /reproduction notebook/i })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /evidence/i })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /manifest/i })).toBeInTheDocument()
  })

  it('shows ARTIFACT GENERATION FAILED distinctly', async () => {
    server.use(
      http.get(`${API_BASE_URL}/runs/:runId/artifacts`, () => {
        return HttpResponse.json({
          run_id: 'run_bad',
          artifact_status: 'FAILED',
          parity_status: 'failed',
          winning_model_id: 'model_1',
          algorithm: 'logistic_regression',
          files: [],
          error: { code: 'artifact_parity_failed', message: 'Predictions did not match.' },
          created_at: '2026-08-23T00:00:00Z',
        })
      }),
    )

    render(<ArtifactPanel runId="run_bad" runStatus="completed" />)
    expect(await screen.findByText(/artifact generation failed/i)).toBeInTheDocument()
    expect(screen.getByText(/predictions did not match/i)).toBeInTheDocument()
  })
})
