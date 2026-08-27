import { useEffect, useMemo, useState } from 'react'
import { FlaskConical, Play, CheckCircle2, ShieldCheck, AlertCircle } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { createExploration } from '@/lib/api'
import {
  defaultHyperparameterForAlgorithm,
  defaultValueForHyperparameter,
  hyperparametersForAlgorithm,
  validateHyperparameterValue,
  type AlgorithmName,
  type HyperparameterMeta,
} from '@/lib/hyperparameterValidation'
import type { CreateExplorationRequest, ExplorationResult, ModelComparisonEntry } from '@/lib/types'

interface WhatIfExperimentPanelProps {
  runId: string
  baseModelId: string
  candidates?: ModelComparisonEntry[]
}

function resolveBaseAlgorithm(
  baseModelId: string,
  candidates: ModelComparisonEntry[] | undefined,
): AlgorithmName {
  const match = candidates?.find((candidate) => candidate.model_id === baseModelId)
  if (match?.algorithm === 'random_forest' || match?.algorithm === 'logistic_regression') {
    return match.algorithm
  }
  return 'logistic_regression'
}

export function WhatIfExperimentPanel({
  runId,
  baseModelId,
  candidates = [],
}: WhatIfExperimentPanelProps) {
  const baseAlgorithm = useMemo(
    () => resolveBaseAlgorithm(baseModelId, candidates),
    [baseModelId, candidates],
  )
  const hyperparameterOptions = useMemo(
    () => hyperparametersForAlgorithm(baseAlgorithm),
    [baseAlgorithm],
  )

  const [experimentType, setExperimentType] = useState<'algorithm' | 'hyperparameter'>('algorithm')
  const [newAlgorithm, setNewAlgorithm] = useState<string>(
    baseAlgorithm === 'logistic_regression' ? 'random_forest' : 'logistic_regression',
  )
  const [hyperparamMeta, setHyperparamMeta] = useState<HyperparameterMeta>(
    () => defaultHyperparameterForAlgorithm(baseAlgorithm),
  )
  const [hyperparamValue, setHyperparamValue] = useState<string>(() =>
    defaultValueForHyperparameter(defaultHyperparameterForAlgorithm(baseAlgorithm)),
  )
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ExplorationResult | null>(null)

  useEffect(() => {
    const nextMeta = defaultHyperparameterForAlgorithm(baseAlgorithm)
    setHyperparamMeta(nextMeta)
    setHyperparamValue(defaultValueForHyperparameter(nextMeta))
    setNewAlgorithm(baseAlgorithm === 'logistic_regression' ? 'random_forest' : 'logistic_regression')
  }, [baseAlgorithm])

  const handleHyperparameterChange = (name: string) => {
    const nextMeta = hyperparameterOptions.find((item) => item.name === name) ?? hyperparameterOptions[0]
    setHyperparamMeta(nextMeta)
    setHyperparamValue(defaultValueForHyperparameter(nextMeta))
  }

  const handleRunExperiment = async () => {
    setLoading(true)
    setError(null)
    try {
      let params: CreateExplorationRequest

      if (experimentType === 'algorithm') {
        params = {
          base_model_id: baseModelId,
          new_algorithm: newAlgorithm as CreateExplorationRequest['new_algorithm'],
        }
      } else {
        const validation = validateHyperparameterValue(hyperparamMeta, hyperparamValue)
        if (!validation.valid) {
          setError(validation.message)
          return
        }
        params = {
          base_model_id: baseModelId,
          hyperparameter_name: hyperparamMeta.name,
          hyperparameter_value: validation.value,
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
            <CardTitle className="text-lg font-bold flex items-center gap-2 text-foreground">
              <FlaskConical className="w-5 h-5 text-primary" /> Controlled What-If Experiments
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1 leading-normal">
              Explore single-variable changes on the exact same train/test split without mutating the base run.
            </p>
          </div>
          <Badge variant="outline" className="text-primary border-primary/30 text-sm font-medium flex items-center gap-1.5 px-2.5 py-0.5">
            <ShieldCheck className="w-4 h-4" /> Isolated Sandbox
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="p-3.5 bg-muted/30 rounded-xl border border-border/60 text-sm text-foreground/80 leading-relaxed">
          What-If experiments fit a separate model in an isolated experiment namespace. The original run and its verified artifact are never modified.
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 bg-background rounded-xl border border-border shadow-xs">
          <div className="flex flex-col gap-2">
            <Label htmlFor="experiment-type" className="text-sm font-medium text-foreground">
              Experiment Type
            </Label>
            <Select
              value={experimentType}
              onValueChange={(value) => setExperimentType(value as 'algorithm' | 'hyperparameter')}
            >
              <SelectTrigger id="experiment-type" className="h-10 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="text-sm">
                <SelectItem value="algorithm" className="text-sm py-2">
                  Swap Algorithm
                </SelectItem>
                <SelectItem value="hyperparameter" className="text-sm py-2">
                  Tweak Hyperparameter
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {experimentType === 'algorithm' ? (
            <div className="flex flex-col gap-2">
              <Label htmlFor="alternative-algorithm" className="text-sm font-medium text-foreground">
                Alternative Algorithm
              </Label>
              <Select value={newAlgorithm} onValueChange={setNewAlgorithm}>
                <SelectTrigger id="alternative-algorithm" className="h-10 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="text-sm">
                  <SelectItem value="random_forest" className="text-sm py-2">
                    Random Forest Classifier
                  </SelectItem>
                  <SelectItem value="logistic_regression" className="text-sm py-2">
                    Logistic Regression
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          ) : (
            <>
              <div className="flex flex-col gap-2">
                <Label htmlFor="hyperparameter-name" className="text-sm font-medium text-foreground">
                  Hyperparameter
                </Label>
                <Select value={hyperparamMeta.name} onValueChange={handleHyperparameterChange}>
                  <SelectTrigger id="hyperparameter-name" className="h-10 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="text-sm">
                    {hyperparameterOptions.map((option) => (
                      <SelectItem key={option.name} value={option.name} className="text-sm py-2">
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="hyperparameter-value" className="text-sm font-medium text-foreground">
                  New Value
                </Label>
                <Input
                  id="hyperparameter-value"
                  type="text"
                  value={hyperparamValue}
                  onChange={(e) => setHyperparamValue(e.target.value)}
                  placeholder={hyperparamMeta.placeholder}
                  className="h-10 text-sm"
                />
              </div>
            </>
          )}

          <div className="flex items-end">
            <Button onClick={handleRunExperiment} disabled={loading} className="w-full h-10 text-sm font-medium gap-2">
              <Play className="w-4 h-4" />
              {loading ? 'Running Fit...' : 'Run What-If'}
            </Button>
          </div>
        </div>

        {error && (
          <div className="p-3.5 bg-destructive/10 dark:bg-destructive/20 rounded-xl border border-destructive/30 text-sm text-destructive font-medium flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span className="leading-snug">{error}</span>
          </div>
        )}

        {result && (
          <div className="p-4 bg-primary/5 rounded-xl border border-primary/20 space-y-3.5 shadow-xs">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                <span className="text-sm font-bold text-foreground">
                  Experiment Result ({result.experiment_id})
                </span>
              </div>
              <Badge variant="outline" className="text-sm font-mono px-2.5 py-0.5">
                {result.variable.name}: {String(result.variable.base_value)} → {String(result.variable.new_value)}
              </Badge>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
              <div className="p-3 bg-background rounded-lg border border-border">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block mb-1">
                  Base Model F1
                </span>
                <span className="text-base font-bold text-foreground font-mono">
                  {result.comparison.base_metric_value.toFixed(4)}
                </span>
              </div>
              <div className="p-3 bg-background rounded-lg border border-border">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block mb-1">
                  What-If Model F1
                </span>
                <span className="text-base font-bold text-foreground font-mono">
                  {result.comparison.new_metric_value.toFixed(4)}
                </span>
              </div>
              <div className="p-3 bg-background rounded-lg border border-border">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block mb-1">
                  F1 Metric Delta
                </span>
                <span
                  className={`text-base font-bold font-mono ${
                    result.comparison.delta > 0
                      ? 'text-emerald-600 dark:text-emerald-400'
                      : result.comparison.delta < 0
                      ? 'text-rose-600 dark:text-rose-400'
                      : 'text-foreground'
                  }`}
                >
                  {result.comparison.delta > 0 ? '+' : ''}
                  {result.comparison.delta.toFixed(4)}
                </span>
              </div>
            </div>

            <p className="text-sm text-foreground/90 bg-background/80 p-3 rounded-lg border border-border/50 leading-relaxed">
              <span className="font-semibold text-foreground">Comparison Justification: </span>
              {result.comparison.justification}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
