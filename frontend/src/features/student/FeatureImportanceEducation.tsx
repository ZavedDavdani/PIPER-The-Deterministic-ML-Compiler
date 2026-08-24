import { BarChart2, AlertCircle, Info } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { FeatureImportanceEducation as FeatureImportanceEducationType } from '@/lib/types'

interface FeatureImportanceEducationProps {
  featureImportance: FeatureImportanceEducationType | null
}

export function FeatureImportanceEducation({ featureImportance }: FeatureImportanceEducationProps) {
  if (!featureImportance) {
    return null
  }

  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-primary" /> Feature Importance & Model Signals
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              Which features had the greatest statistical weight on final predictions.
            </p>
          </div>
          {featureImportance.algorithm && (
            <Badge variant="secondary" className="font-mono text-xs">
              {featureImportance.algorithm}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* MANDATORY DISCLAIMER */}
        <div className="p-3 bg-amber-500/10 rounded-xl border border-amber-300/40 flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="text-xs">
            <span className="font-semibold text-foreground">Critical Educational Rule:</span>
            <p className="text-amber-800 dark:text-amber-300 font-medium mt-0.5">
              "{featureImportance.disclaimer}"
            </p>
          </div>
        </div>

        {/* Educational Summary */}
        <div className="p-3 bg-muted/20 rounded-xl border border-border/60 flex items-start gap-2.5 text-xs">
          <Info className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
          <p className="text-muted-foreground">{featureImportance.educational_summary}</p>
        </div>

        {/* Feature List if available */}
        {featureImportance.features.length > 0 ? (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-foreground">
              Top Ranked Features
            </h4>
            <div className="space-y-1.5">
              {featureImportance.features.map((f, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2.5 rounded-lg border border-border bg-card text-xs font-mono"
                >
                  <span className="font-semibold text-foreground">{String(f.feature ?? f.name ?? `Feature ${idx + 1}`)}</span>
                  <span className="text-primary font-bold">
                    {typeof f.importance === 'number' ? f.importance.toFixed(4) : String(f.importance ?? '')}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground italic bg-muted/10 p-2.5 rounded-lg text-center">
            Feature weights are derived mathematically from the fitted pipeline estimator without modifying test data.
          </div>
        )}
      </CardContent>
    </Card>
  )
}
