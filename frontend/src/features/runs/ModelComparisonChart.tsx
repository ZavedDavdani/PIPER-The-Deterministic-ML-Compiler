import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Badge } from '@/components/ui/badge'
import type { ModelComparison } from '@/lib/types'

interface ModelComparisonChartProps {
  comparison: ModelComparison
}

const METRIC_COLORS: Record<string, string> = {
  accuracy: 'var(--color-muted-foreground)',
  precision: 'var(--color-warning)',
  recall: 'var(--color-accent-foreground)',
  f1: 'var(--color-primary)',
  roc_auc: 'var(--color-success)',
}

export function ModelComparisonChart({ comparison }: ModelComparisonChartProps) {
  const data = comparison.models.map((model) => ({
    name: model.algorithm,
    model_id: model.model_id,
    accuracy: Number(model.accuracy.toFixed(3)),
    precision: Number(model.precision.toFixed(3)),
    recall: Number(model.recall.toFixed(3)),
    f1: Number(model.f1.toFixed(3)),
    roc_auc: Number(model.roc_auc.toFixed(3)),
  }))

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2 text-xs">
        <span className="text-muted-foreground">Selection metric:</span>
        <Badge variant="outline">{comparison.selection_metric}</Badge>
        <span className="text-muted-foreground">Recommended:</span>
        <Badge>{comparison.models.find((m) => m.model_id === comparison.recommended_model_id)?.algorithm ?? comparison.recommended_model_id}</Badge>
      </div>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-border)', borderRadius: 8, fontSize: 12 }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {Object.keys(METRIC_COLORS).map((metric) => (
              <Bar key={metric} dataKey={metric} fill={METRIC_COLORS[metric]} radius={[3, 3, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
