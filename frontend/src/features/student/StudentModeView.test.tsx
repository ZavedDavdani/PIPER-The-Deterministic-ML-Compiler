import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { StudentModeView } from './StudentModeView'
import { DecisionWhyCard } from './DecisionWhyCard'
import { FeatureImportanceEducation } from './FeatureImportanceEducation'
import { MetricExplainerCard } from './MetricExplainerCard'
import { WhatIfExperimentPanel } from './WhatIfExperimentPanel'

describe('StudentModeView', () => {
  it('renders student mode view with 14-stage journey, metrics, and why inspector', async () => {
    render(
      <StudentModeView
        runId="run_test_001"
        winningModelId="model_lr001"
        selectionJustification="logistic_regression selected: F1=0.81 vs 0.76."
        baselineAccuracy={0.73}
      />
    )

    // Heading
    expect(screen.getByText(/Student Mode — ML Education & Guided Walkthrough/i)).toBeInTheDocument()

    // Wait for MSW data to load
    await waitFor(() => {
      expect(screen.getByText(/14-Stage ML Learning Journey/i)).toBeInTheDocument()
      expect(screen.getByText(/End-to-End Pipeline Visualization/i)).toBeInTheDocument()
      expect(screen.getByText(/Model Architectures Explained/i)).toBeInTheDocument()
      expect(screen.getByText(/Evaluation Metrics Explained/i)).toBeInTheDocument()
    })

    // Disclaimer
    expect(screen.getByText(/Feature importance shows association with model predictions; it does not prove causation./i)).toBeInTheDocument()
  })

  it('allows switching explanation levels on DecisionWhyCard', () => {
    const mockOp = {
      operation_id: 'op_1',
      tool_name: 'scale_features',
      what_happened: 'PIPER scaled numeric features.',
      why: 'Numeric features had different scales.',
      level: 'beginner' as const,
      concept: 'Feature Standardization (Z-score)',
      alternative_consideration: 'Without scaling, large numbers dominate.',
    }

    render(<DecisionWhyCard operation={mockOp} />)

    expect(screen.getByText('scale_features')).toBeInTheDocument()
    expect(screen.getByText('Feature Standardization (Z-score)')).toBeInTheDocument()

    // Switch to advanced level
    const advButton = screen.getByRole('button', { name: /advanced/i })
    fireEvent.click(advButton)
    expect(advButton).toHaveClass('bg-background')
  })

  it('renders mandatory non-causal disclaimer in FeatureImportanceEducation', () => {
    const fiData = {
      available: true,
      method: 'Model-Derived Feature Importance',
      algorithm: 'logistic_regression',
      disclaimer: 'Feature importance shows association with model predictions; it does not prove causation.',
      features: [
        { feature: 'tenure', importance: 0.42 },
        { feature: 'MonthlyCharges', importance: 0.35 },
      ],
      educational_summary: 'Weights indicate feature influence on decision boundaries.',
    }

    render(<FeatureImportanceEducation featureImportance={fiData} />)

    expect(
      screen.getByText('"Feature importance shows association with model predictions; it does not prove causation."')
    ).toBeInTheDocument()
    expect(screen.getByText('tenure')).toBeInTheDocument()
    expect(screen.getByText('0.4200')).toBeInTheDocument()
  })

  it('renders metric cards with real run values and formulas', () => {
    const metrics = [
      {
        metric: 'accuracy',
        value: 0.85,
        meaning: 'Accuracy = 0.8500',
        formula: 'Accuracy = (TP + TN) / (TP + TN + FP + FN)',
        guidance: 'Best when balanced.',
      },
      {
        metric: 'f1',
        value: 0.82,
        meaning: 'F1 Score = 0.8200',
        formula: 'F1 = 2 * (Precision * Recall) / (Precision + Recall)',
        guidance: 'Balances precision and recall.',
      },
    ]

    render(<MetricExplainerCard metrics={metrics} baselineAccuracy={0.7} />)

    expect(screen.getByText('ACCURACY')).toBeInTheDocument()
    expect(screen.getByText('0.8500')).toBeInTheDocument()
    expect(screen.getByText('F1')).toBeInTheDocument()
    expect(screen.getByText('0.8200')).toBeInTheDocument()
    expect(screen.getByText('MAJORITY BASELINE')).toBeInTheDocument()
    expect(screen.getByText('0.7000')).toBeInTheDocument()
  })

  it('executes a safe What-If experiment and displays side-by-side metric comparison', async () => {
    render(
      <WhatIfExperimentPanel
        runId="run_test_001"
        baseModelId="model_lr001"
        candidates={[]}
      />
    )

    expect(screen.getByText(/Controlled What-If Experiments/i)).toBeInTheDocument()
    const runBtn = screen.getByRole('button', { name: /Run What-If/i })
    fireEvent.click(runBtn)

    await waitFor(() => {
      expect(screen.getByText(/Experiment Result \(exp_msw_001\)/i)).toBeInTheDocument()
      expect(screen.getByText('0.8100')).toBeInTheDocument()
      expect(screen.getByText('0.8150')).toBeInTheDocument()
      expect(screen.getByText('+0.0050')).toBeInTheDocument()
    })
  })
})
