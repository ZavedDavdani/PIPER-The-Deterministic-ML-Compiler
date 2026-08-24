import { BarChart3, Calculator, Info } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { MetricExplanation } from '@/lib/types'

interface MetricExplainerCardProps {
  metrics: MetricExplanation[]
  baselineAccuracy?: number | null
}

export function MetricExplainerCard({ metrics, baselineAccuracy }: MetricExplainerCardProps) {
  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-primary" /> Evaluation Metrics Explained
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              Grounded in the test split scores computed during this run.
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {metrics.map((m) => (
            <div
              key={m.metric}
              className="p-3.5 rounded-xl border border-border bg-muted/10 space-y-2.5 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between gap-1">
                  <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    {m.metric.replace('_', '-').toUpperCase()}
                  </span>
                  <Badge variant="secondary" className="font-mono text-xs font-bold text-foreground">
                    {m.value.toFixed(4)}
                  </Badge>
                </div>
                <p className="text-xs text-foreground/90 mt-1.5 leading-relaxed">{m.meaning}</p>
              </div>

              <div className="space-y-1.5 pt-2 border-t border-border/40 text-xs">
                {m.formula && (
                  <div className="flex items-start gap-1.5 text-muted-foreground font-mono text-[11px] bg-background/80 p-1.5 rounded border border-border/40">
                    <Calculator className="w-3.5 h-3.5 text-primary flex-shrink-0 mt-0.5" />
                    <span>{m.formula}</span>
                  </div>
                )}
                {m.guidance && (
                  <div className="flex items-start gap-1.5 text-muted-foreground text-[11px] bg-primary/5 p-1.5 rounded border border-primary/20">
                    <Info className="w-3.5 h-3.5 text-primary flex-shrink-0 mt-0.5" />
                    <span>{m.guidance}</span>
                  </div>
                )}
              </div>
            </div>
          ))}

          {baselineAccuracy !== undefined && baselineAccuracy !== null && (
            <div className="p-3.5 rounded-xl border border-border bg-muted/10 space-y-2.5 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between gap-1">
                  <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    MAJORITY BASELINE
                  </span>
                  <Badge variant="outline" className="font-mono text-xs font-bold text-muted-foreground">
                    {baselineAccuracy.toFixed(4)}
                  </Badge>
                </div>
                <p className="text-xs text-foreground/90 mt-1.5 leading-relaxed">
                  The accuracy of a zero-intelligence model always predicting the most frequent label.
                </p>
              </div>
              <div className="space-y-1.5 pt-2 border-t border-border/40 text-xs">
                <div className="flex items-start gap-1.5 text-muted-foreground text-[11px] bg-primary/5 p-1.5 rounded border border-primary/20">
                  <Info className="w-3.5 h-3.5 text-primary flex-shrink-0 mt-0.5" />
                  <span>A trained model must meaningfully outperform this baseline to be useful.</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
