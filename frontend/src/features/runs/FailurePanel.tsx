import { Fragment } from 'react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import type { FailureInfo } from '@/lib/types'

const CATEGORY_LABELS: Record<string, string> = {
  DATA_ERROR: 'Data error',
  SCHEMA_ERROR: 'Schema error',
  TARGET_ERROR: 'Target column error',
  EXECUTION_BUDGET_EXCEEDED: 'Execution budget exceeded',
  LEAKAGE_ERROR: 'Data leakage detected',
  IMBALANCE_ERROR: 'Target imbalance',
  FEATURE_ERROR: 'Feature issue',
  TRAINING_ERROR: 'Training error',
  EVALUATION_ERROR: 'Evaluation error',
  DUPLICATE_PLAN: 'Duplicate plan',
  BASELINE_GATE_FAILED: 'Baseline gate failed',
}

interface FailurePanelProps {
  failure: FailureInfo
}

export function FailurePanel({ failure }: FailurePanelProps) {
  const evidenceEntries = Object.entries(failure.evidence ?? {})

  return (
    <Alert variant="destructive">
      <AlertTitle className="flex flex-wrap items-center gap-2">
        {CATEGORY_LABELS[failure.category] ?? failure.category}
        <Badge variant="outline" className="text-[10px]">
          node: {failure.node}
        </Badge>
        <Badge variant="outline" className="text-[10px]">
          attempt {failure.attempt}
        </Badge>
        {failure.retryable ? (
          <Badge variant="warning" className="text-[10px]">
            retryable
          </Badge>
        ) : (
          <Badge variant="destructive" className="text-[10px]">
            terminal
          </Badge>
        )}
        {failure.human_intervention_required && (
          <Badge variant="destructive" className="text-[10px]">
            needs human review
          </Badge>
        )}
      </AlertTitle>
      <AlertDescription>
        <p>{failure.message}</p>
        {evidenceEntries.length > 0 && (
          <dl className="mt-2 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-xs">
            {evidenceEntries.map(([key, value]) => (
              <Fragment key={key}>
                <dt className="text-muted-foreground font-mono">{key}</dt>
                <dd className="font-mono break-all">
                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </dd>
              </Fragment>
            ))}
          </dl>
        )}
      </AlertDescription>
    </Alert>
  )
}
