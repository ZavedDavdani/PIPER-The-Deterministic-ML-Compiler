import { describe, expect, it } from 'vitest'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import App from './App'
import { FakeEventSource } from '@/test/fakeEventSource'
import { resetRunOutcome, setNextRunOutcome } from '@/test/mswServer'
import { fixtureCompletedResult, fixtureDatasetProfile } from '@/test/fixtures'
import type { TraceEvent } from '@/lib/types'

/**
 * End-to-end (within the frontend) proof of the real workflow: upload
 * -> select dataset -> configure + start a run -> live SSE progress
 * -> terminal result — driven through the actual component tree and
 * router, against MSW handlers that mirror the real FastAPI contract
 * exactly (see src/test/mswServer.ts / fixtures.ts) plus a fake
 * EventSource standing in for the browser API jsdom doesn't provide
 * (see src/test/fakeEventSource.ts). The genuinely real, live-network
 * version of this same flow (actual backend, actual browser
 * EventSource) is verified separately, manually, against a running
 * uvicorn + Vite dev server.
 */

function liveEvent(overrides: Partial<TraceEvent>): TraceEvent {
  return {
    run_id: fixtureCompletedResult.run_id,
    step_id: `trace_${Math.random().toString(36).slice(2)}`,
    parent_span_id: null,
    attempt: 0,
    node: 'profile',
    event_type: 'node_completed',
    timestamp: new Date().toISOString(),
    status: 'success',
    severity: 'info',
    tool_name: null,
    guardrail_name: null,
    evidence: {},
    duration_ms: 0,
    message: '',
    ...overrides,
  }
}

describe('PIPER frontend — real API workflow', () => {
  it('takes a dataset upload all the way to a completed run result', async () => {
    resetRunOutcome()
    setNextRunOutcome('running')

    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>,
    )

    // 1. Upload a dataset (real POST /datasets via MSW).
    const file = new File(['a,b\n1,2\n'], 'telco.csv', { type: 'text/csv' })
    const input = screen.getByLabelText(/choose dataset file/i)
    await userEvent.upload(input, file)

    // 2. The uploaded dataset auto-selects; its real profile loads
    // (GET /datasets/:id), surfacing real column names.
    await screen.findAllByText(fixtureDatasetProfile.dataset_id)
    await userEvent.click(await screen.findByRole('combobox', { name: /objective/i }))
    await userEvent.click(await screen.findByRole('option', { name: 'Churn' }))

    // 3. Start the run (real POST /runs).
    await userEvent.click(screen.getByRole('button', { name: /start run/i }))

    // 4. Navigated to the run page; status polling begins (real GET /runs/:id).
    await screen.findByText(/running/i)

    // 5. Live SSE progress arrives via the fake EventSource.
    const source = FakeEventSource.latest()!
    expect(source.url).toContain(`/runs/${fixtureCompletedResult.run_id}/events`)
    act(() => source.simulateOpen())
    act(() => source.simulateMessage(liveEvent({ node: 'validate_input' })))
    act(() => source.simulateMessage(liveEvent({ node: 'profile' })))
    act(() =>
      source.simulateMessage(
        liveEvent({ node: 'clean', event_type: 'tool_called', tool_name: 'drop_column' }),
      ),
    )

    await screen.findByText('drop_column')

    // 6. The backend run reaches a terminal status; the run_completed
    // summary event arrives over SSE too.
    setNextRunOutcome('completed', fixtureCompletedResult)
    act(() => source.simulateMessage(liveEvent({ node: 'report', event_type: 'run_completed' })))

    // 7. Once terminal, the real GET /runs/:id/result result renders.
    await waitFor(
      () => expect(screen.getByText(/pipeline completed and passed validation/i)).toBeInTheDocument(),
      { timeout: 3000 },
    )

    const resultsCard = screen.getByText('Trained candidates').closest('[data-slot="card"]')!
    expect(within(resultsCard as HTMLElement).getByText('f1')).toBeInTheDocument()
  })

  it('shows a structured failure for a failed run', async () => {
    resetRunOutcome()
    setNextRunOutcome('failed')

    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>,
    )

    const file = new File(['a,b\n1,2\n'], 'telco.csv', { type: 'text/csv' })
    await userEvent.upload(screen.getByLabelText(/choose dataset file/i), file)
    await screen.findAllByText(fixtureDatasetProfile.dataset_id)
    await userEvent.click(await screen.findByRole('combobox', { name: /objective/i }))
    await userEvent.click(await screen.findByRole('option', { name: 'Churn' }))
    await userEvent.click(screen.getByRole('button', { name: /start run/i }))

    await screen.findAllByText(/failed|running/i)
    const source = FakeEventSource.latest()!
    act(() => source.simulateOpen())
    act(() => source.simulateMessage(liveEvent({ node: 'report', event_type: 'run_failed', status: 'failure' })))

    await waitFor(
      () => expect(screen.getByText(/duplicate plan/i)).toBeInTheDocument(),
      { timeout: 3000 },
    )
    expect(screen.getByText('terminal')).toBeInTheDocument()
    expect(screen.getByText('needs human review')).toBeInTheDocument()
  })
})
