import { useEffect, useState } from 'react'
import { GraduationCap, AlertCircle, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { LearningJourneyStepper } from './LearningJourneyStepper'
import { PipelineVisualizer } from './PipelineVisualizer'
import { DecisionWhyCard } from './DecisionWhyCard'
import { MetricExplainerCard } from './MetricExplainerCard'
import { ModelExplainerCard } from './ModelExplainerCard'
import { ReplanEducationCard } from './ReplanEducationCard'
import { FeatureImportanceEducation } from './FeatureImportanceEducation'
import { WhatIfExperimentPanel } from './WhatIfExperimentPanel'
import {
  getRunExplanation,
  getLearningJourney,
  getPipelineVisualization,
} from '@/lib/api'
import type {
  RunExplanation,
  LearningJourney,
  PipelineVisualization,
  ModelComparisonEntry,
} from '@/lib/types'

interface StudentModeViewProps {
  runId: string
  candidates?: ModelComparisonEntry[]
  winningModelId?: string | null
  selectionJustification?: string | null
  baselineAccuracy?: number | null
}

export function StudentModeView({
  runId,
  candidates = [],
  winningModelId,
  selectionJustification,
  baselineAccuracy,
}: StudentModeViewProps) {
  const [explanation, setExplanation] = useState<RunExplanation | null>(null)
  const [journey, setJourney] = useState<LearningJourney | null>(null)
  const [pipeline, setPipeline] = useState<PipelineVisualization | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [expRes, journeyRes, pipeRes] = await Promise.all([
        getRunExplanation(runId, 'beginner').catch(() => null),
        getLearningJourney(runId).catch(() => null),
        getPipelineVisualization(runId).catch(() => null),
      ])
      setExplanation(expRes)
      setJourney(journeyRes)
      setPipeline(pipeRes)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load Student Mode data.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [runId])

  if (error) {
    return (
      <div className="p-6 rounded-xl border border-rose-300/40 bg-rose-500/10 space-y-3">
        <div className="flex items-center gap-2 text-rose-700 dark:text-rose-400 font-semibold text-sm">
          <AlertCircle className="w-5 h-5" />
          <span>Error loading Student Mode</span>
        </div>
        <p className="text-xs text-muted-foreground">{error}</p>
        <Button variant="outline" size="sm" onClick={loadData} className="gap-1.5 text-xs">
          <RefreshCw className="w-3.5 h-3.5" /> Retry
        </Button>
      </div>
    )
  }

  const allOperations = [
    ...(explanation?.preprocessing ?? []),
    ...(explanation?.feature_engineering ?? []),
  ]

  const metricsList = explanation?.evaluation?.[0]?.metrics ?? []
  const baseModelId = winningModelId ?? explanation?.model_selection?.recommended_model_id ?? candidates[0]?.model_id ?? ''

  return (
    <div className="space-y-6">
      {/* Student Mode Header Banner */}
      <div className="p-4 bg-primary/10 rounded-xl border border-primary/20 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-primary text-primary-foreground rounded-lg">
            <GraduationCap className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-foreground">
              Student Mode — ML Education & Guided Walkthrough
            </h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              Deterministic, evidence-grounded explanations of every decision PIPER made during this run.
            </p>
          </div>
        </div>
      </div>

      {/* 1. Learning Journey */}
      <LearningJourneyStepper journey={journey} isLoading={loading} />

      {/* 2. Pipeline Visualization */}
      <PipelineVisualizer pipeline={pipeline} isLoading={loading} />

      {/* 3. Model Architectures */}
      {explanation && explanation.model_concepts && explanation.model_concepts.length > 0 && (
        <ModelExplainerCard
          modelConcepts={explanation.model_concepts}
          winningModelId={winningModelId}
          selectionJustification={selectionJustification ?? explanation.model_selection?.justification}
        />
      )}

      {/* 4. Evaluation Metrics */}
      {metricsList.length > 0 && (
        <MetricExplainerCard metrics={metricsList} baselineAccuracy={baselineAccuracy} />
      )}

      {/* 5. REPLAN Education */}
      {explanation?.replan && <ReplanEducationCard replan={explanation.replan} />}

      {/* 6. Feature Importance Education */}
      {explanation?.feature_importance && (
        <FeatureImportanceEducation featureImportance={explanation.feature_importance} />
      )}

      {/* 7. Why did PIPER do this? Inspector */}
      {allOperations.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold uppercase tracking-wider text-foreground">
              Decisions & 'Why?' Inspector ({allOperations.length} Actions)
            </h3>
            <span className="text-xs text-muted-foreground">
              Inspect deterministic rules behind every preprocessing and feature engineering step
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {allOperations.map((op) => (
              <DecisionWhyCard key={op.operation_id} operation={op} />
            ))}
          </div>
        </div>
      )}

      {/* 8. Controlled What-If Experiments */}
      {baseModelId && (
        <WhatIfExperimentPanel
          runId={runId}
          baseModelId={baseModelId}
          candidates={candidates}
        />
      )}
    </div>
  )
}
