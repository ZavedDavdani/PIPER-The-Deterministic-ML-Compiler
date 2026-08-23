import { Link, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { RunStatusHeader } from '@/features/runs/RunStatusHeader'
import { LiveEventFeed } from '@/features/runs/LiveEventFeed'
import { ResultSummary } from '@/features/runs/ResultSummary'
import { DecisionTracePanel } from '@/features/runs/DecisionTracePanel'
import { useRunEvents } from '@/lib/useRunEvents'
import { useRunStatus } from '@/lib/useRunStatus'
import { useRunResult } from '@/lib/useRunResult'
import { useDecisionTrace } from '@/lib/useDecisionTrace'

export function RunPage() {
  const { runId = '' } = useParams<{ runId: string }>()
  const { status, error: statusError } = useRunStatus(runId || null)
  const { events, connectionState } = useRunEvents(runId || null, Boolean(runId))
  const isTerminal = status ? status.status === 'completed' || status.status === 'failed' : false
  const { result, error: resultError } = useRunResult(runId || null, isTerminal)
  const { trace } = useDecisionTrace(runId || null, isTerminal)

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-8">
      <div className="flex items-center justify-between gap-4">
        <Link to="/" className="text-muted-foreground hover:text-foreground flex items-center gap-1 text-sm">
          <ArrowLeft className="size-3.5" /> Back to datasets
        </Link>
        <Link to="/history" className="text-muted-foreground hover:text-foreground text-sm">
          History
        </Link>
        <ThemeToggle />
      </div>

      <Card>
        <CardHeader>
          <RunStatusHeader runId={runId} status={status} connectionState={connectionState} />
        </CardHeader>
      </Card>

      {statusError && <p className="text-destructive text-sm">{statusError}</p>}

      <DecisionTracePanel runId={runId} isTerminal={isTerminal} trace={trace} />

      <Card>
        <CardHeader>
          <CardTitle>Planning &amp; execution</CardTitle>
        </CardHeader>
        <CardContent>
          <LiveEventFeed events={events} />
        </CardContent>
      </Card>

      {isTerminal && (
        <>
          {resultError && <p className="text-destructive text-sm">{resultError}</p>}
          {result && <ResultSummary result={result} />}
        </>
      )}
    </div>
  )
}
