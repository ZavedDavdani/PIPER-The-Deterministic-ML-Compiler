import { useState } from 'react'
import { FlaskConical, Play, CheckCircle2, ShieldCheck, AlertCircle } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { createExploration } from '@/lib/api'
import type { ExplorationResult, ModelComparisonEntry } from '@/lib/types'

interface WhatIfExperimentPanelProps {
  runId: string
  baseModelId: string
  candidates?: ModelComparisonEntry[]
}

export function WhatIfExperimentPanel({
  runId,
  baseModelId,
}: WhatIfExperimentPanelProps) {
  const [experimentType, setExperimentType] = useState<'algorithm' | 'hyperparameter'>('algorithm')
  const [newAlgorithm, setNewAlgorithm] = useState<string>('random_forest')
  const [hyperparamName, setHyperparamName] = useState<string>('C')
  const [hyperparamValue, setHyperparamValue] = useState<string>('0.1')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ExplorationResult | null>(null)

  const handleRunExperiment = async () => {
    setLoading(true)
    setError(null)
    try {
      let params: {
        base_model_id: string
        new_algorithm?: string
        hyperparameter_name?: string
        hyperparameter_value?: unknown
      }

      if (experimentType === 'algorithm') {
        params = {
          base_model_id: baseModelId,
          new_algorithm: newAlgorithm,
        }
      } else {
        const val = parseFloat(hyperparamValue)
        params = {
          base_model_id: baseModelId,
          hyperparameter_name: hyperparamName,
          hyperparameter_value: isNaN(val) ? hyperparamValue : val,
        }
      }

      const res = await createExploration(runId, params)
      setResult(res)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'What-If experiment failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center gap-2">
              <FlaskConical className="w-5 h-5 text-primary" /> Controlled What-If Experiments
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              Explore single-variable changes on the exact same train/test split without mutating the base run.
            </p>
          </div>
          <Badge variant="outline" className="text-primary border-primary/30 text-xs flex items-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5" /> Isolated Sandbox
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Safe Sandbox Warning / Info */}
        <div className="p-3 bg-muted/20 rounded-xl border border-border/60 text-xs text-muted-foreground">
          What-If experiments fit a separate model in an isolated experiment namespace. The original run and its verified artifact are never modified.
        </div>

        {/* Experiment Controls */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 p-3.5 bg-background rounded-xl border border-border">
          <div>
            <label className="text-xs font-semibold text-foreground block mb-1">Experiment Type</label>
            <select
              value={experimentType}
              onChange={(e) => setExperimentType(e.target.value as 'algorithm' | 'hyperparameter')}
              className="w-full text-xs p-2 rounded-lg border border-border bg-card text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="algorithm">Swap Algorithm</option>
              <option value="hyperparameter">Tweak Hyperparameter</option>
            </select>
          </div>

          {experimentType === 'algorithm' ? (
            <div>
              <label className="text-xs font-semibold text-foreground block mb-1">Alternative Algorithm</label>
              <select
                value={newAlgorithm}
                onChange={(e) => setNewAlgorithm(e.target.value)}
                className="w-full text-xs p-2 rounded-lg border border-border bg-card text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              >
                <option value="random_forest">Random Forest Classifier</option>
                <option value="logistic_regression">Logistic Regression</option>
              </select>
            </div>
          ) : (
            <>
              <div>
                <label className="text-xs font-semibold text-foreground block mb-1">Hyperparameter</label>
                <select
                  value={hyperparamName}
                  onChange={(e) => setHyperparamName(e.target.value)}
                  className="w-full text-xs p-2 rounded-lg border border-border bg-card text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                >
                  <option value="C">Regularization Strength (C)</option>
                  <option value="n_estimators">Tree Count (n_estimators)</option>
                  <option value="max_depth">Max Depth (max_depth)</option>
                </select>
              </div>
              <div>
                <label className="text-xs font-semibold text-foreground block mb-1">New Value</label>
                <input
                  type="text"
                  value={hyperparamValue}
                  onChange={(e) => setHyperparamValue(e.target.value)}
                  className="w-full text-xs p-2 rounded-lg border border-border bg-card text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                  placeholder="e.g. 0.1, 50, 5"
                />
              </div>
            </>
          )}

          <div className="flex items-end">
            <Button
              onClick={handleRunExperiment}
              disabled={loading}
              className="w-full text-xs gap-1.5 h-8"
            >
              <Play className="w-3.5 h-3.5" />
              {loading ? 'Running Fit...' : 'Run What-If'}
            </Button>
          </div>
        </div>

        {error && (
          <div className="p-3 bg-rose-500/10 rounded-xl border border-rose-300/40 text-xs text-rose-600 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* Experiment Results Side-by-Side */}
        {result && (
          <div className="p-4 bg-primary/5 rounded-xl border border-primary/20 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                <span className="text-sm font-semibold text-foreground">
                  Experiment Result ({result.experiment_id})
                </span>
              </div>
              <Badge variant="outline" className="text-xs font-mono">
                {result.variable.name}: {String(result.variable.base_value)} → {String(result.variable.new_value)}
              </Badge>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              <div className="p-2.5 bg-background rounded-lg border border-border">
                <span className="text-muted-foreground block">Base Model F1</span>
                <span className="text-sm font-bold text-foreground font-mono">
                  {result.comparison.base_metric_value.toFixed(4)}
                </span>
              </div>
              <div className="p-2.5 bg-background rounded-lg border border-border">
                <span className="text-muted-foreground block">What-If Model F1</span>
                <span className="text-sm font-bold text-foreground font-mono">
                  {result.comparison.new_metric_value.toFixed(4)}
                </span>
              </div>
              <div className="p-2.5 bg-background rounded-lg border border-border">
                <span className="text-muted-foreground block">F1 Metric Delta</span>
                <span
                  className={`text-sm font-bold font-mono ${
                    result.comparison.delta > 0
                      ? 'text-emerald-600'
                      : result.comparison.delta < 0
                      ? 'text-rose-600'
                      : 'text-foreground'
                  }`}
                >
                  {result.comparison.delta > 0 ? '+' : ''}
                  {result.comparison.delta.toFixed(4)}
                </span>
              </div>
            </div>

            <p className="text-xs text-muted-foreground bg-background/60 p-2.5 rounded border border-border/40">
              <span className="font-semibold text-foreground">Comparison Justification: </span>
              {result.comparison.justification}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
