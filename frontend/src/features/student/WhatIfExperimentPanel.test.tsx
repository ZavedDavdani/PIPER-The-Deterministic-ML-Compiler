import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { WhatIfExperimentPanel } from './WhatIfExperimentPanel'

describe('WhatIfExperimentPanel', () => {
  it('rejects decimal n_estimators before making the API request', async () => {
    render(
      <WhatIfExperimentPanel
        runId="run_test_001"
        baseModelId="model_rf001"
        candidates={[
          {
            model_id: 'model_rf001',
            algorithm: 'random_forest',
            accuracy: 0.8,
            precision: 0.8,
            recall: 0.8,
            f1: 0.8,
            roc_auc: 0.8,
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByRole('combobox', { name: /Experiment Type/i }))
    fireEvent.click(screen.getByRole('option', { name: /Tweak Hyperparameter/i }))

    fireEvent.click(screen.getByRole('combobox', { name: /Hyperparameter/i }))
    fireEvent.click(screen.getByRole('option', { name: /Tree Count/i }))

    fireEvent.change(screen.getByLabelText(/New Value/i), { target: { value: '0.1' } })
    fireEvent.click(screen.getByRole('button', { name: /Run What-If/i }))

    await waitFor(() => {
      expect(screen.getByText('n_estimators must be a whole number.')).toBeInTheDocument()
    })
    expect(screen.queryByText(/Experiment Result/i)).not.toBeInTheDocument()
  })

  it('rejects non-numeric and out-of-bounds hyperparameter values before API request', async () => {
    render(
      <WhatIfExperimentPanel
        runId="run_test_001"
        baseModelId="model_rf001"
        candidates={[
          {
            model_id: 'model_rf001',
            algorithm: 'random_forest',
            accuracy: 0.8,
            precision: 0.8,
            recall: 0.8,
            f1: 0.8,
            roc_auc: 0.8,
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByRole('combobox', { name: /Experiment Type/i }))
    fireEvent.click(screen.getByRole('option', { name: /Tweak Hyperparameter/i }))

    // Test non-numeric
    fireEvent.change(screen.getByLabelText(/New Value/i), { target: { value: 'abc' } })
    fireEvent.click(screen.getByRole('button', { name: /Run What-If/i }))
    await waitFor(() => {
      expect(screen.getByText(/Please enter a valid value/i)).toBeInTheDocument()
    })

    // Test out-of-bounds integer
    fireEvent.change(screen.getByLabelText(/New Value/i), { target: { value: '10' } })
    fireEvent.click(screen.getByRole('button', { name: /Run What-If/i }))
    await waitFor(() => {
      expect(screen.getByText('n_estimators must be between 50 and 500.')).toBeInTheDocument()
    })
  })

  it('allows valid floating point values for C in logistic regression', async () => {
    render(
      <WhatIfExperimentPanel
        runId="run_test_001"
        baseModelId="model_lr001"
        candidates={[
          {
            model_id: 'model_lr001',
            algorithm: 'logistic_regression',
            accuracy: 0.8,
            precision: 0.8,
            recall: 0.8,
            f1: 0.8,
            roc_auc: 0.8,
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByRole('combobox', { name: /Experiment Type/i }))
    fireEvent.click(screen.getByRole('option', { name: /Tweak Hyperparameter/i }))

    fireEvent.change(screen.getByLabelText(/New Value/i), { target: { value: '0.05' } })
    fireEvent.click(screen.getByRole('button', { name: /Run What-If/i }))

    // No error for valid float 0.05
    await waitFor(() => {
      expect(screen.queryByText(/must be a whole number/i)).not.toBeInTheDocument()
    })
  })

  it('uses readable typography classes for form controls', () => {
    render(
      <WhatIfExperimentPanel
        runId="run_test_001"
        baseModelId="model_lr001"
        candidates={[
          {
            model_id: 'model_lr001',
            algorithm: 'logistic_regression',
            accuracy: 0.8,
            precision: 0.8,
            recall: 0.8,
            f1: 0.8,
            roc_auc: 0.8,
          },
        ]}
      />,
    )

    expect(screen.getByText('Experiment Type')).toHaveClass('text-sm')
    expect(screen.getByRole('button', { name: /Run What-If/i })).toHaveClass('text-sm')
    expect(screen.getByRole('combobox', { name: /Experiment Type/i })).toHaveClass('text-sm')
  })
})
