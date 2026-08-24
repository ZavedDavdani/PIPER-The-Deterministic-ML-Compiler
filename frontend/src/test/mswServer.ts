import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { API_BASE_URL } from '@/lib/api'
import {
  fixtureCompletedResult,
  fixtureDatasetProfile,
  fixtureFailedResult,
} from './fixtures'
import type {
  CreateRunRequest,
  CreateRunResponse,
  DatasetUploadResponse,
  DecisionTrace,
  HumanInterventionPackage,
  PiperVerdict,
  RunStatusResponse,
  GovernanceBundle,
} from '@/lib/types'

/**
 * MSW handlers matching the REAL FastAPI contract (paths, methods,
 * status codes, JSON shapes) exactly — verified against
 * backend/app/api/routers/{datasets,runs}.py. Tests using these are
 * checking real request/response handling in the frontend, against a
 * simulated network layer, not hardcoded UI mock data.
 */

let runStatusOverride: RunStatusResponse['status'] = 'completed'
let resultOverride = fixtureCompletedResult

export function setNextRunOutcome(status: RunStatusResponse['status'], result = fixtureCompletedResult) {
  runStatusOverride = status
  resultOverride = result
}

export function resetRunOutcome() {
  runStatusOverride = 'completed'
  resultOverride = fixtureCompletedResult
}

const STAGE_IDS = [
  'LLM_PROPOSED',
  'VALIDATED',
  'ADEQUACY',
  'REPLAN',
  'EXECUTION',
  'TRAINING',
  'EVALUATION',
  'GUARDRAILS',
  'FINAL_VERDICT',
] as const

function fixtureDecisionTrace(runId: string, status: RunStatusResponse['status']): DecisionTrace {
  const terminal = status === 'completed' || status === 'failed'
  return {
    run_id: runId,
    run_status: status,
    stages: STAGE_IDS.map((id) => ({
      id,
      label: id.replaceAll('_', ' '),
      status: terminal ? (status === 'failed' && id === 'FINAL_VERDICT' ? 'failed' : 'passed') : 'pending',
      summary: terminal ? 'Recorded.' : 'Waiting.',
      attempt: 0,
      evidence: {},
    })),
    planning_attempts: [],
    plan_diffs: [],
  }
}

function fixtureVerdict(runId: string, status: RunStatusResponse['status']): PiperVerdict {
  const accepted = status === 'completed'
  return {
    run_id: runId,
    outcome: accepted ? 'ACCEPTED' : 'REJECTED',
    reason_code: accepted ? 'ACCEPTED_GUARDRAILS_PASSED' : 'REJECTED',
    summary: accepted ? 'PIPER accepted this run.' : 'PIPER rejected this run.',
    retry_count: 0,
    max_retries: 2,
    structurally_valid_plan: accepted,
    adequacy_passed: accepted,
    guardrails_passed: accepted,
    human_intervention_required: status === 'failed',
    executed: accepted,
  }
}

function fixtureIntervention(runId: string, status: RunStatusResponse['status']): HumanInterventionPackage {
  return {
    run_id: runId,
    required: status === 'failed',
    headline: status === 'failed' ? 'Human review is required.' : 'No human intervention required.',
    failure_category: status === 'failed' ? 'DUPLICATE_PLAN' : null,
    failure_message: status === 'failed' ? 'duplicate' : null,
    retry_count: 0,
    max_retries: 2,
    last_proposed_steps: [],
    structural_violations: [],
    material_adequacy_findings: [],
    advisory_adequacy_findings: [],
    preserved_valid_steps: [],
    implicated_steps: [],
    recommended_actions: status === 'failed' ? ['Read the structured failure.'] : [],
    blocked_invalid_execution: status === 'failed',
  }
}

export const handlers = [
  http.post(`${API_BASE_URL}/datasets`, async () => {
    const response: DatasetUploadResponse = {
      dataset_id: fixtureDatasetProfile.dataset_id,
      filename: 'telco.csv',
      rows: fixtureDatasetProfile.rows,
      columns: fixtureDatasetProfile.column_profiles.map((c) => c.name),
    }
    return HttpResponse.json(response, { status: 201 })
  }),

  http.get(`${API_BASE_URL}/datasets`, () => {
    return HttpResponse.json({ dataset_ids: [fixtureDatasetProfile.dataset_id] })
  }),

  http.get(`${API_BASE_URL}/datasets/:datasetId`, ({ params }) => {
    if (params.datasetId !== fixtureDatasetProfile.dataset_id) {
      return HttpResponse.json({ detail: 'not found' }, { status: 404 })
    }
    return HttpResponse.json(fixtureDatasetProfile)
  }),

  http.post(`${API_BASE_URL}/runs`, async ({ request }) => {
    const body = (await request.json()) as CreateRunRequest
    if (body.dataset_id !== fixtureDatasetProfile.dataset_id) {
      return HttpResponse.json({ detail: `Dataset '${body.dataset_id}' does not exist.` }, { status: 404 })
    }
    const response: CreateRunResponse = { run_id: resultOverride.run_id, status: 'running' }
    return HttpResponse.json(response, { status: 202 })
  }),

  http.get(`${API_BASE_URL}/runs`, () => {
    return HttpResponse.json({
      runs: [
        {
          run_id: resultOverride.run_id,
          dataset_id: fixtureDatasetProfile.dataset_id,
          target_column: 'Churn',
          status: runStatusOverride,
          current_node: null,
          attempt: 0,
          created_at: null,
          updated_at: null,
        },
      ],
    })
  }),

  http.get(`${API_BASE_URL}/settings/ollama`, () => {
    return HttpResponse.json({
      host: 'http://localhost:11434',
      model: 'qwen3:4b',
      keep_alive: '10m',
      reachable: false,
      models: [],
      error: 'unreachable',
    })
  }),

  http.put(`${API_BASE_URL}/settings/ollama`, async ({ request }) => {
    const body = (await request.json()) as { model?: string; host?: string }
    return HttpResponse.json({
      host: body.host ?? 'http://localhost:11434',
      model: body.model ?? 'qwen3:4b',
      keep_alive: '10m',
      reachable: false,
      models: [],
      error: 'unreachable',
    })
  }),

  http.get(`${API_BASE_URL}/runs/:runId`, ({ params }) => {
    const response: RunStatusResponse = {
      run_id: String(params.runId),
      dataset_id: fixtureDatasetProfile.dataset_id,
      target_column: 'Churn',
      status: runStatusOverride,
      current_node: runStatusOverride === 'completed' || runStatusOverride === 'failed' ? 'report' : 'train',
      attempt: 0,
      plan_history: [],
    }
    return HttpResponse.json(response)
  }),

  http.get(`${API_BASE_URL}/runs/:runId/result`, () => {
    if (runStatusOverride !== 'completed' && runStatusOverride !== 'failed') {
      return HttpResponse.json({ detail: 'still running' }, { status: 409 })
    }
    return HttpResponse.json(runStatusOverride === 'failed' ? fixtureFailedResult : resultOverride)
  }),

  http.get(`${API_BASE_URL}/runs/:runId/decision-trace`, ({ params }) => {
    return HttpResponse.json(fixtureDecisionTrace(String(params.runId), runStatusOverride))
  }),

  http.get(`${API_BASE_URL}/runs/:runId/verdict`, ({ params }) => {
    if (runStatusOverride !== 'completed' && runStatusOverride !== 'failed') {
      return HttpResponse.json({ detail: 'still running' }, { status: 409 })
    }
    return HttpResponse.json(fixtureVerdict(String(params.runId), runStatusOverride))
  }),

  http.get(`${API_BASE_URL}/runs/:runId/intervention`, ({ params }) => {
    if (runStatusOverride !== 'completed' && runStatusOverride !== 'failed') {
      return HttpResponse.json({ detail: 'still running' }, { status: 409 })
    }
    return HttpResponse.json(fixtureIntervention(String(params.runId), runStatusOverride))
  }),

  http.get(`${API_BASE_URL}/runs/:runId/evidence`, ({ params }) => {
    if (runStatusOverride !== 'completed' && runStatusOverride !== 'failed') {
      return HttpResponse.json({ detail: 'still running' }, { status: 409 })
    }
    const runId = String(params.runId)
    return HttpResponse.json({
      schema_version: 'piper.evidence.v1',
      run_id: runId,
      dataset_id: fixtureDatasetProfile.dataset_id,
      target_column: 'Churn',
      status: runStatusOverride,
      decision_trace: fixtureDecisionTrace(runId, runStatusOverride),
      verdict: fixtureVerdict(runId, runStatusOverride),
      intervention: fixtureIntervention(runId, runStatusOverride),
      notes: ['LLM chain-of-thought is intentionally omitted.'],
    })
  }),

  http.get(`${API_BASE_URL}/runs/:runId/replay`, ({ params }) => {
    if (runStatusOverride !== 'completed' && runStatusOverride !== 'failed') {
      return HttpResponse.json({ detail: 'still running' }, { status: 409 })
    }
    const runId = String(params.runId)
    const evidence = {
      schema_version: 'piper.evidence.v1',
      run_id: runId,
      dataset_id: fixtureDatasetProfile.dataset_id,
      target_column: 'Churn',
      status: runStatusOverride,
      decision_trace: fixtureDecisionTrace(runId, runStatusOverride),
      verdict: fixtureVerdict(runId, runStatusOverride),
      intervention: fixtureIntervention(runId, runStatusOverride),
      notes: ['LLM chain-of-thought is intentionally omitted.'],
    }
    return HttpResponse.json({
      run_id: runId,
      llm_invoked: false,
      source: 'persisted_events_and_state',
      status: runStatusOverride,
      decision_trace: evidence.decision_trace,
      verdict: evidence.verdict,
      intervention: evidence.intervention,
      evidence,
    })
  }),

  http.get(`${API_BASE_URL}/runs/:runId/governance`, ({ params }) => {
    if (runStatusOverride !== 'completed' && runStatusOverride !== 'failed') {
      return HttpResponse.json({ detail: 'still running' }, { status: 409 })
    }
    const runId = String(params.runId)
    const importance = {
      status: 'NOT_AVAILABLE' as const,
      method: 'NOT_AVAILABLE',
      algorithm: null,
      rows: [],
      disclaimer: 'These scores describe association, not causal effects.',
      reason: 'MSW fixture has no fitted pipeline.',
    }
    const body: GovernanceBundle = {
      schema_version: 'piper.governance.v1',
      run_id: runId,
      run_status: runStatusOverride,
      model_card: {
        status: runStatusOverride === 'completed' ? 'AVAILABLE' : 'NOT_AVAILABLE',
        run_id: runId,
        dataset_id: fixtureDatasetProfile.dataset_id,
        task_type: 'binary_classification',
        target: 'Churn',
        winning_model_id: runStatusOverride === 'completed' ? 'model_lr001' : null,
        winning_algorithm: runStatusOverride === 'completed' ? 'logistic_regression' : null,
        candidate_models: [],
        evaluation_metrics: runStatusOverride === 'completed' ? [{ name: 'f1', value: 0.5 }] : [],
        baseline_comparison: null,
        train_test_split: null,
        preprocessing_summary: [],
        guardrail_results: [],
        limitations: ['MSW fixture.'],
        artifact_information: { artifact_status: 'NOT_GENERATED' },
        feature_importance: importance,
        reason: null,
      },
      data_card: {
        status: 'AVAILABLE',
        run_id: runId,
        dataset_id: fixtureDatasetProfile.dataset_id,
        rows: fixtureDatasetProfile.rows,
        columns: fixtureDatasetProfile.columns,
        target: 'Churn',
        feature_list: [],
        column_summaries: [],
        numeric_features: [],
        categorical_features: [],
        missingness: [],
        preprocessing_operations: [],
        train_test: null,
        data_quality_findings: [],
        limitations: [],
        reason: null,
      },
      fingerprints: {
        run_id: runId,
        hash_algorithm: 'sha256',
        content_hashes: [],
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
      limitations: [],
      artifact_status: { artifact_status: 'NOT_GENERATED' },
      notes: ['LLM chain-of-thought is intentionally omitted.'],
    }
    return HttpResponse.json(body)
  }),

  http.get(`${API_BASE_URL}/runs/:runId/artifacts`, ({ params }) => {
    return HttpResponse.json({
      run_id: String(params.runId),
      artifact_status: 'NOT_GENERATED',
      parity_status: 'not_run',
      winning_model_id: null,
      algorithm: null,
      files: [],
      error: null,
      created_at: null,
    })
  }),

  http.post(`${API_BASE_URL}/runs/:runId/artifacts`, ({ params }) => {
    if (runStatusOverride !== 'completed') {
      return HttpResponse.json({ detail: { code: 'run_not_verified', message: 'Run is not eligible.' } }, { status: 409 })
    }
    return HttpResponse.json(
      {
        run_id: String(params.runId),
        artifact_status: 'VERIFIED',
        parity_status: 'passed',
        winning_model_id: 'model_lr001',
        algorithm: 'logistic_regression',
        files: [
          'pipeline.joblib',
          'pipeline.py',
          'training_reproduction.ipynb',
          'manifest.json',
          'evidence.json',
          'hashes.json',
        ],
        error: null,
        created_at: '2026-08-23T00:00:00Z',
      },
      { status: 201 },
    )
  }),

  http.get(`${API_BASE_URL}/runs/:runId/deployment`, ({ params }) => {
    return HttpResponse.json({
      run_id: String(params.runId),
      status: 'NOT_READY',
      artifact_status: 'NOT_GENERATED',
      winning_model_id: null,
      algorithm: null,
      required_columns: [],
      checks: [],
      reason: { code: 'artifact_missing', message: 'No artifact bundle exists for this run.' },
    })
  }),

  http.get(`${API_BASE_URL}/runs/:runId/learn/explanation`, ({ params }) => {
    return HttpResponse.json({
      run_id: String(params.runId),
      status: runStatusOverride,
      level: 'beginner',
      preprocessing: [
        {
          operation_id: 'op_1',
          tool_name: 'impute_missing_values',
          what_happened: 'PIPER replaced missing values in TotalCharges using the median.',
          why: 'TotalCharges is numeric and contained nulls.',
          level: 'beginner',
          concept: 'Missing Value Imputation',
          alternative_consideration: 'Dropping rows loses training data.',
        },
      ],
      feature_engineering: [
        {
          operation_id: 'op_2',
          tool_name: 'encode_categorical_features',
          what_happened: 'PIPER converted categories into binary columns.',
          why: 'Estimators require numeric input matrices.',
          level: 'beginner',
          concept: 'One-Hot Encoding',
          alternative_consideration: 'Label encoding creates fake numeric order.',
        },
      ],
      model_selection: {
        recommended_model_id: 'model_lr001',
        recommended_algorithm: 'logistic_regression',
        justification: 'logistic_regression selected: F1=0.81 vs 0.76.',
        candidates: [
          {
            model_id: 'model_lr001',
            algorithm: 'logistic_regression',
            accuracy: 0.82,
            precision: 0.8,
            recall: 0.82,
            f1: 0.81,
            roc_auc: 0.85,
          },
        ],
        concept: 'Model Selection & Metric Optimization',
      },
      evaluation: [
        {
          model_id: 'model_lr001',
          algorithm: 'logistic_regression',
          metrics: [
            {
              metric: 'accuracy',
              value: 0.82,
              meaning: 'Accuracy = 0.8200 — The proportion of total predictions that were correct.',
              formula: 'Accuracy = (TP + TN) / (TP + TN + FP + FN)',
              guidance: 'Best when dataset is balanced.',
            },
            {
              metric: 'f1',
              value: 0.81,
              meaning: 'F1 Score = 0.8100 — The harmonic mean of Precision and Recall.',
              formula: 'F1 = 2 * (Precision * Recall) / (Precision + Recall)',
              guidance: 'PIPER selection metric.',
            },
          ],
          confusion_matrix_meaning: 'Out of 200 rows: 140 TN, 24 TP, 16 FP, 20 FN.',
          baseline_comparison: 'Model F1=0.81 exceeds baseline.',
          model_concept: {
            algorithm: 'logistic_regression',
            name: 'Logistic Regression',
            concept: 'Linear classification algorithm mapping log-odds to class probabilities.',
            strengths: ['Interpretable', 'Fast'],
            tradeoffs: ['Linear decision boundary only'],
            how_piper_used_it: 'Trained with L2 regularization.',
            is_winner: true,
          },
        },
      ],
      guardrail_checks: [
        {
          check: 'data_leakage',
          passed: true,
          severity: 'error',
          meaning: 'Checks whether any feature is suspiciously predictive of target.',
          message: 'No leakage detected.',
        },
      ],
      failure: null,
      replan: {
        replan_occurred: false,
        total_attempts: 1,
        attempts_summary: [],
        plan_differences: [],
        educational_takeaway: 'Autonomous self-correction handles plan defects gracefully.',
      },
      feature_importance: {
        available: true,
        method: 'Model-Derived Feature Importance',
        algorithm: 'logistic_regression',
        disclaimer: 'Feature importance shows association with model predictions; it does not prove causation.',
        features: [
          { feature: 'tenure', importance: 0.42 },
          { feature: 'TotalCharges', importance: 0.31 },
        ],
        educational_summary: 'Features with higher weights had greater influence on model decisions.',
      },
      model_concepts: [
        {
          algorithm: 'logistic_regression',
          name: 'Logistic Regression',
          concept: 'Linear classification algorithm mapping log-odds to class probabilities.',
          strengths: ['Interpretable', 'Fast'],
          tradeoffs: ['Linear decision boundary only'],
          how_piper_used_it: 'Trained with L2 regularization.',
          is_winner: true,
        },
      ],
    })
  }),

  http.get(`${API_BASE_URL}/runs/:runId/learn/journey`, ({ params }) => {
    return HttpResponse.json({
      run_id: String(params.runId),
      status: runStatusOverride,
      current_stage_id: 1,
      stages: [
        {
          stage_id: 1,
          title: 'Understand the Dataset',
          description: 'Load raw dataset and inspect row/column counts.',
          status: 'completed',
          summary: 'Dataset profiling complete.',
          details: {},
          concept: 'Data Profiling',
        },
        {
          stage_id: 2,
          title: 'Identify the Target',
          description: 'Define the outcome variable.',
          status: 'completed',
          summary: 'Target Churn identified.',
          details: {},
          concept: 'Target Specification',
        },
        {
          stage_id: 14,
          title: 'Test Unseen Data',
          description: 'Score new unseen test data.',
          status: 'completed',
          summary: 'Test flight available.',
          details: {},
          concept: 'Test Flight',
        },
      ],
    })
  }),

  http.get(`${API_BASE_URL}/runs/:runId/learn/pipeline`, ({ params }) => {
    return HttpResponse.json({
      run_id: String(params.runId),
      nodes: [
        {
          id: 'dataset',
          name: 'Dataset',
          stage: 'Input',
          status: 'passed',
          summary: 'Raw dataset profile.',
          details: {},
        },
        {
          id: 'preprocessing',
          name: 'Preprocessing',
          stage: 'Data Preparation',
          status: 'passed',
          summary: 'Imputed missing values.',
          details: {},
        },
      ],
      edges: [{ from_node: 'dataset', to_node: 'preprocessing' }],
    })
  }),

  http.post(`${API_BASE_URL}/runs/:runId/explore`, ({ params }) => {
    return HttpResponse.json({
      experiment_id: 'exp_msw_001',
      run_id: String(params.runId),
      base_model_id: 'model_lr001',
      variable: {
        kind: 'algorithm',
        name: 'algorithm',
        base_value: 'logistic_regression',
        new_value: 'random_forest',
      },
      new_model: {
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
      comparison: {
        base_model_id: 'model_lr001',
        new_model_id: 'model_rf_exp',
        primary_metric: 'f1',
        base_metric_value: 0.81,
        new_metric_value: 0.815,
        delta: 0.005,
        winner_id: 'model_rf_exp',
        justification: 'random_forest selected: F1=0.8150 vs 0.8100.',
      },
    })
  }),
]

export const server = setupServer(...handlers)
