import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { GovernancePanel } from './GovernancePanel'
import { http, HttpResponse } from 'msw'
import { server } from '@/test/mswServer'
import { API_BASE_URL } from '@/lib/api'
import type { GovernanceBundle } from '@/lib/types'

const importance = {
  status: 'AVAILABLE' as const,
  method: 'logistic_regression_coefficients',
  algorithm: 'logistic_regression',
  rows: [
    {
      feature: 'categorical__cat_a',
      transformed_feature: 'categorical__cat_a',
      importance: 0.42,
      direction: 'positive' as const,
      source_feature: 'cat',
    },
  ],
  disclaimer: 'These scores describe association, not causal effects.',
  reason: null,
}

function fixtureBundle(runId: string): GovernanceBundle {
  return {
    schema_version: 'piper.governance.v1',
    run_id: runId,
    run_status: 'completed',
    model_card: {
      status: 'AVAILABLE',
      run_id: runId,
      dataset_id: 'ds1',
      task_type: 'binary_classification',
      target: 'label',
      winning_model_id: 'model_1',
      winning_algorithm: 'logistic_regression',
      candidate_models: [],
      evaluation_metrics: [{ name: 'f1', value: 0.8 }],
      baseline_comparison: null,
      train_test_split: { training_rows: 64, test_rows: 16 },
      preprocessing_summary: ['scale_features'],
      guardrail_results: [],
      limitations: ['Feature importance is associative rather than causal.'],
      artifact_information: { artifact_status: 'NOT_GENERATED' },
      feature_importance: importance,
      reason: null,
    },
    data_card: {
      status: 'AVAILABLE',
      run_id: runId,
      dataset_id: 'ds1',
      rows: 80,
      columns: 3,
      target: 'label',
      feature_list: ['cat', 'num'],
      column_summaries: [],
      numeric_features: ['num'],
      categorical_features: ['cat'],
      missingness: [],
      preprocessing_operations: [],
      train_test: null,
      data_quality_findings: [],
      limitations: ['Sample cell values from profiling are omitted.'],
      reason: null,
    },
    fingerprints: {
      run_id: runId,
      hash_algorithm: 'sha256',
      content_hashes: [
        {
          name: 'executed_plan',
          kind: 'CONTENT_HASH',
          algorithm: 'sha256',
          digest: 'abc123def4567890',
          available: true,
          reason: null,
        },
      ],
      metadata: {},
      caveat: 'These fingerprints are tamper-evident content hashes.',
    },
    feature_importance: importance,
    fairness: {
      status: 'NOT_REQUESTED',
      requested_columns: [],
      minimum_group_size: 30,
      positive_class: null,
      reference_group_rule: 'largest-n group',
      groups: [],
      warnings: [],
      disclaimer: 'Statistical subgroup measurements, not a legal compliance decision.',
      reason: 'No subgroup columns were supplied.',
    },
    limitations: ['No language model is invoked while building this bundle.'],
    artifact_status: { artifact_status: 'NOT_GENERATED' },
    notes: [],
  }
}

describe('GovernancePanel', () => {
  it('renders model card, fingerprints, feature importance, and downloads', async () => {
    server.use(
      http.get(`${API_BASE_URL}/runs/:runId/governance`, () => {
        return HttpResponse.json(fixtureBundle('run_gov'))
      }),
    )

    render(<GovernancePanel runId="run_gov" runStatus="completed" />)

    expect(await screen.findByText('Governance')).toBeInTheDocument()
    expect(screen.getByText(/logistic_regression/)).toBeInTheDocument()
    expect(screen.getByText(/f1:/i)).toBeInTheDocument()
    expect(screen.getByText(/sha256/i)).toBeInTheDocument()
    expect(screen.getByText(/tamper-evident/i)).toBeInTheDocument()
    expect(screen.getByText(/categorical__cat_a/)).toBeInTheDocument()
    expect(screen.getByText(/not causal/i)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /download model card \(markdown\)/i })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /download data card \(json\)/i })).toBeInTheDocument()
  })

  it('runs subgroup analysis only for operator-specified columns', async () => {
    server.use(
      http.get(`${API_BASE_URL}/runs/:runId/governance`, () => {
        return HttpResponse.json(fixtureBundle('run_gov'))
      }),
      http.get(`${API_BASE_URL}/runs/:runId/governance/fairness`, ({ request }) => {
        const url = new URL(request.url)
        const columns = url.searchParams.getAll('column')
        return HttpResponse.json({
          status: 'INSUFFICIENT_DATA',
          requested_columns: columns,
          minimum_group_size: 30,
          positive_class: 'yes',
          reference_group_rule: 'largest-n group',
          groups: [
            {
              column: 'group',
              group: 'a',
              n: 4,
              accuracy: null,
              precision: null,
              recall: null,
              f1: null,
              selection_rate: null,
              disparate_impact_ratio: null,
              sufficient: false,
              warning: 'n=4 is below the minimum group size (30); rates are not reported.',
            },
          ],
          warnings: ["Column 'group' group 'a' has n=4 < 30."],
          disclaimer: 'Statistical subgroup measurements, not a legal compliance decision.',
          reason: 'Every requested subgroup is below the minimum sample size.',
        })
      }),
    )

    render(<GovernancePanel runId="run_gov" runStatus="completed" />)
    await screen.findByText('Governance')
    await userEvent.type(screen.getByLabelText(/subgroup columns/i), 'group')
    await userEvent.click(screen.getByRole('button', { name: /analyze subgroups/i }))
    expect(await screen.findByText(/INSUFFICIENT_DATA/)).toBeInTheDocument()
    expect(screen.getByText(/n=4 is below the minimum group size/i)).toBeInTheDocument()
  })
})
