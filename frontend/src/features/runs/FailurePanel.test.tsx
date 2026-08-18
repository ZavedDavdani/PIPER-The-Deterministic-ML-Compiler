import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { FailurePanel } from './FailurePanel'
import type { FailureInfo } from '@/lib/types'

describe('FailurePanel', () => {
  it('renders the real failure category, message, and evidence keys — nothing invented', () => {
    const failure: FailureInfo = {
      category: 'LEAKAGE_ERROR',
      message: "Feature 'leaky_dup' is a near-exact duplicate of the target.",
      evidence: { correlation: 0.98, feature: 'leaky_dup' },
      node: 'validate',
      attempt: 0,
      retryable: true,
      human_intervention_required: false,
    }

    render(<FailurePanel failure={failure} />)

    expect(screen.getByText(/data leakage detected/i)).toBeInTheDocument()
    expect(screen.getByText(failure.message)).toBeInTheDocument()
    expect(screen.getByText('correlation')).toBeInTheDocument()
    expect(screen.getByText('0.98')).toBeInTheDocument()
    expect(screen.getByText('retryable')).toBeInTheDocument()
    expect(screen.queryByText('needs human review')).not.toBeInTheDocument()
  })

  it('flags a terminal, human-intervention-required failure distinctly', () => {
    const failure: FailureInfo = {
      category: 'EXECUTION_BUDGET_EXCEEDED',
      message: 'Execution step budget (80) was reached.',
      evidence: { steps_executed: 80 },
      node: 'plan_entry',
      attempt: 5,
      retryable: false,
      human_intervention_required: true,
    }

    render(<FailurePanel failure={failure} />)

    expect(screen.getByText('terminal')).toBeInTheDocument()
    expect(screen.getByText('needs human review')).toBeInTheDocument()
  })
})
