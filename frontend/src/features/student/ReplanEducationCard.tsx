import { RefreshCw, ArrowRight, ShieldAlert, CheckCircle2, Lightbulb } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { ReplanExplanation } from '@/lib/types'

interface ReplanEducationCardProps {
  replan: ReplanExplanation | null
}

export function ReplanEducationCard({ replan }: ReplanEducationCardProps) {
  if (!replan) {
    return null
  }

  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center gap-2">
              <RefreshCw className="w-5 h-5 text-primary" /> Autonomous REPLAN & Self-Correction
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              How PIPER's state machine diagnoses validation issues and safely recovers with a new plan.
            </p>
          </div>
          {replan.replan_occurred ? (
            <Badge className="bg-amber-500/10 text-amber-600 border-amber-300 text-xs">
              REPLAN Triggered ({replan.total_attempts} Attempts)
            </Badge>
          ) : (
            <Badge variant="outline" className="text-emerald-600 border-emerald-300 text-xs">
              First Attempt Passed (1 Attempt)
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Conceptual Diagram */}
        <div className="p-3 bg-muted/20 rounded-xl border border-border/60">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-2 bg-background p-2 rounded-lg border border-border/70">
              <span className="font-semibold text-foreground">Attempt 1: Proposed Plan</span>
            </div>
            <ArrowRight className="w-4 h-4 text-muted-foreground hidden sm:block" />
            <div className="flex items-center gap-1.5 bg-amber-500/10 text-amber-700 dark:text-amber-400 p-2 rounded-lg border border-amber-300/40">
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>Guardrails / Adequacy Check</span>
            </div>
            <ArrowRight className="w-4 h-4 text-muted-foreground hidden sm:block" />
            <div className="flex items-center gap-1.5 bg-primary/10 text-primary p-2 rounded-lg border border-primary/20">
              <RefreshCw className="w-3.5 h-3.5" />
              <span>REPLAN (Preserve Context)</span>
            </div>
            <ArrowRight className="w-4 h-4 text-muted-foreground hidden sm:block" />
            <div className="flex items-center gap-1.5 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 p-2 rounded-lg border border-emerald-300/40">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Attempt 2: Verified Execution</span>
            </div>
          </div>
        </div>

        {/* Educational Takeaway */}
        <div className="p-3 bg-primary/5 rounded-xl border border-primary/20 flex items-start gap-2.5">
          <Lightbulb className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />
          <div className="text-xs">
            <span className="font-semibold text-foreground">Why Self-Correction Matters:</span>
            <p className="text-muted-foreground mt-0.5">{replan.educational_takeaway}</p>
          </div>
        </div>

        {/* Plan Differences if multiple attempts */}
        {replan.plan_differences.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-foreground">
              Recorded Plan Differences
            </h4>
            <div className="space-y-1.5">
              {replan.plan_differences.map((d, idx) => (
                <div key={idx} className="p-2.5 rounded-lg border border-border bg-card text-xs">
                  <span className="font-semibold text-primary">{String(d.comparison)}:</span>{' '}
                  <span className="text-muted-foreground">{String(d.summary)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
