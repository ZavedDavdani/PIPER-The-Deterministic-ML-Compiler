import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { getEvidence, getIntervention, getVerdict } from '@/lib/api'
import type {
  DecisionStage,
  DecisionTrace,
  HumanInterventionPackage,
  PiperVerdict,
  StageStatus,
} from '@/lib/types'

const STATUS_CLASS: Record<StageStatus, string> = {
  passed: 'border-green-600/40 bg-green-600/10 text-green-800 dark:text-green-400',
  failed: 'border-destructive/40 bg-destructive/10 text-destructive',
  current: 'border-blue-600/40 bg-blue-600/10 text-blue-800 dark:text-blue-400',
  pending: 'border-border bg-muted/40 text-muted-foreground',
  skipped: 'border-border bg-muted/20 text-muted-foreground',
  not_reached: 'border-border bg-muted/20 text-muted-foreground',
}

function StageChip({ stage }: { stage: DecisionStage }) {
  return (
    <div className={`min-w-[7.5rem] flex-1 rounded-md border px-2 py-2 ${STATUS_CLASS[stage.status]}`}>
      <p className="text-[10px] font-semibold tracking-wide uppercase">{stage.label}</p>
      <p className="mt-1 text-[11px] leading-snug">{stage.summary}</p>
    </div>
  )
}

interface DecisionTracePanelProps {
  runId: string
  isTerminal: boolean
  trace: DecisionTrace | null
}

export function DecisionTracePanel({ runId, isTerminal, trace }: DecisionTracePanelProps) {
  const [verdict, setVerdict] = useState<PiperVerdict | null>(null)
  const [intervention, setIntervention] = useState<HumanInterventionPackage | null>(null)
  const [exportError, setExportError] = useState<string | null>(null)

  useEffect(() => {
    if (!isTerminal || !runId) return
    let cancelled = false
    void Promise.all([getVerdict(runId), getIntervention(runId)])
      .then(([nextVerdict, nextIntervention]) => {
        if (cancelled) return
        setVerdict(nextVerdict)
        setIntervention(nextIntervention)
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setExportError(err instanceof Error ? err.message : 'Could not load verdict.')
        }
      })
    return () => {
      cancelled = true
    }
  }, [isTerminal, runId])

  async function downloadEvidence() {
    setExportError(null)
    try {
      const payload = await getEvidence(runId)
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${runId}-evidence.json`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Could not export evidence.')
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <CardTitle>Decision trace</CardTitle>
        </CardHeader>
        <CardContent>
          {trace ? (
            <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap">
              {trace.stages.map((stage, index) => (
                <div key={stage.id} className="flex flex-1 items-stretch gap-1">
                  <StageChip stage={stage} />
                  {index < trace.stages.length - 1 && (
                    <span className="text-muted-foreground hidden self-center text-xs sm:inline" aria-hidden>
                      →
                    </span>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground text-sm">Waiting for planner evidence…</p>
          )}
          {trace && trace.plan_diffs.length > 0 && (
            <div className="mt-4">
              <p className="mb-1 text-sm font-medium">Plan diff (REPLAN)</p>
              <pre className="bg-muted max-h-48 overflow-auto rounded-md p-2 text-xs">
                {JSON.stringify(trace.plan_diffs, null, 2)}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>

      {verdict && (
        <Alert variant={verdict.outcome === 'ACCEPTED' ? 'success' : 'destructive'}>
          <AlertTitle>Final PIPER verdict: {verdict.outcome}</AlertTitle>
          <AlertDescription>{verdict.summary}</AlertDescription>
        </Alert>
      )}

      {intervention && intervention.required && (
        <Card>
          <CardHeader>
            <CardTitle>Human intervention</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-2 text-sm">
            <p>{intervention.headline}</p>
            {intervention.failure_category && (
              <p>
                <span className="font-medium">{intervention.failure_category}</span>
                {intervention.failure_message ? ` — ${intervention.failure_message}` : ''}
              </p>
            )}
            <ul className="list-disc pl-5">
              {intervention.recommended_actions.map((action) => (
                <li key={action}>{action}</li>
              ))}
            </ul>
            {intervention.blocked_invalid_execution && (
              <p className="text-muted-foreground">
                Invalid LLM plans were not executed. validate_proposed_plan() remains the authority.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {isTerminal && (
        <div>
          <Button type="button" variant="outline" onClick={() => void downloadEvidence()}>
            Download evidence JSON
          </Button>
          {exportError && <p className="text-destructive mt-2 text-sm">{exportError}</p>}
        </div>
      )}
    </div>
  )
}
