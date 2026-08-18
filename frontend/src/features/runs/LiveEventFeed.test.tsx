import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { LiveEventFeed } from './LiveEventFeed'
import type { TraceEvent } from '@/lib/types'

function makeEvent(overrides: Partial<TraceEvent>): TraceEvent {
  return {
    run_id: 'run_x',
    step_id: `trace_${Math.random().toString(36).slice(2)}`,
    parent_span_id: null,
    attempt: 0,
    node: 'profile',
    event_type: 'node_completed',
    timestamp: '2026-01-01T00:00:00Z',
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

describe('LiveEventFeed', () => {
  it('shows a waiting message with no events yet', () => {
    render(<LiveEventFeed events={[]} />)
    expect(screen.getByText(/waiting for piper to start/i)).toBeInTheDocument()
  })

  it('groups events by attempt', () => {
    render(
      <LiveEventFeed
        events={[
          makeEvent({ node: 'profile', attempt: 0 }),
          makeEvent({ node: 'plan', attempt: 1 }),
        ]}
      />,
    )
    expect(screen.getByText('attempt 0')).toBeInTheDocument()
    expect(screen.getByText('attempt 1')).toBeInTheDocument()
  })

  it('badges a REPLAN transition distinctly from an ordinary node_completed event', () => {
    render(
      <LiveEventFeed
        events={[
          makeEvent({
            node: 'plan_entry',
            attempt: 1,
            event_type: 'node_completed',
            evidence: { updated_fields: ['retry_count', 'status'] },
          }),
        ]}
      />,
    )
    expect(screen.getByText(/replan → attempt 1/i)).toBeInTheDocument()
  })

  it('shows the failure message for a node_failed event', () => {
    render(
      <LiveEventFeed
        events={[
          makeEvent({
            node: 'validate',
            event_type: 'node_failed',
            status: 'failure',
            severity: 'error',
            message: 'Data leakage detected.',
          }),
        ]}
      />,
    )
    expect(screen.getByText('Data leakage detected.')).toBeInTheDocument()
  })

  it('shows the tool name for a tool_called event', () => {
    render(
      <LiveEventFeed
        events={[makeEvent({ node: 'cleaner', event_type: 'tool_called', tool_name: 'drop_column' })]}
      />,
    )
    expect(screen.getByText('drop_column')).toBeInTheDocument()
  })
})
