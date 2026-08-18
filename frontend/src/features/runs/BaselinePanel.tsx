import { Badge } from '@/components/ui/badge'
import type { BaselineComparisonResult } from '@/lib/types'

interface BaselinePanelProps {
  baseline: BaselineComparisonResult
}

function fmt(value: number | null): string {
  return value === null ? '—' : value.toFixed(3)
}

export function BaselinePanel({ baseline }: BaselinePanelProps) {
  return (
    <div className="flex flex-col gap-2 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={baseline.gate_passed ? 'success' : 'destructive'}>
          {baseline.gate_passed ? 'Beats trivial baseline' : 'Did not beat baseline'}
        </Badge>
        <span className="text-muted-foreground text-xs">
          Δ {baseline.primary_metric} = {fmt(baseline.delta)} (min required {baseline.minimum_required_delta})
        </span>
      </div>
      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-4">
        <div>
          <div className="text-muted-foreground">Model {baseline.primary_metric}</div>
          <div className="font-mono">{fmt(baseline.model_primary_metric_value)}</div>
        </div>
        <div>
          <div className="text-muted-foreground">Baseline {baseline.primary_metric}</div>
          <div className="font-mono">{fmt(baseline.baseline_primary_metric_value)}</div>
        </div>
        <div>
          <div className="text-muted-foreground">Majority class</div>
          <div className="font-mono">{baseline.baseline.majority_class}</div>
        </div>
        <div>
          <div className="text-muted-foreground">Baseline accuracy</div>
          <div className="font-mono">{fmt(baseline.baseline.accuracy)}</div>
        </div>
      </div>
      <p className="text-muted-foreground text-xs">{baseline.reason}</p>
    </div>
  )
}
