import { describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { RunCreateForm } from './RunCreateForm'
import { server } from '@/test/mswServer'
import { API_BASE_URL } from '@/lib/api'
import { fixtureDatasetProfile } from '@/test/fixtures'
import type { CreateRunRequest } from '@/lib/types'

describe('RunCreateForm', () => {
  it('submits the real request shape and reports the new run_id', async () => {
    let captured: CreateRunRequest | null = null
    server.use(
      http.post(`${API_BASE_URL}/runs`, async ({ request }) => {
        captured = (await request.json()) as CreateRunRequest
        return HttpResponse.json({ run_id: 'run_abc123', status: 'running' }, { status: 202 })
      }),
    )

    const onCreated = vi.fn()
    render(
      <RunCreateForm datasetId={fixtureDatasetProfile.dataset_id} columns={['Contract', 'Churn']} onCreated={onCreated} />,
    )

    await userEvent.click(screen.getByRole('combobox', { name: /objective/i }))
    await userEvent.click(await screen.findByRole('option', { name: 'Churn' }))
    await userEvent.click(screen.getByRole('button', { name: /start run/i }))

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith('run_abc123'))
    expect(captured).toEqual({
      dataset_id: fixtureDatasetProfile.dataset_id,
      target_column: 'Churn',
      max_retries: 2,
    })
  })

  it('shows a real backend error message when the dataset no longer exists', async () => {
    server.use(
      http.post(`${API_BASE_URL}/runs`, () =>
        HttpResponse.json({ detail: "Dataset 'x' does not exist." }, { status: 404 }),
      ),
    )

    const onCreated = vi.fn()
    render(<RunCreateForm datasetId="dataset_gone" columns={['Churn']} onCreated={onCreated} />)

    await userEvent.click(screen.getByRole('button', { name: /start run/i }))

    expect(await screen.findByText(/does not exist/i)).toBeInTheDocument()
    expect(onCreated).not.toHaveBeenCalled()
  })
})
