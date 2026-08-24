import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Code2, GraduationCap } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { RunStatusHeader } from '@/features/runs/RunStatusHeader'
import { LiveEventFeed } from '@/features/runs/LiveEventFeed'
import { ResultSummary } from '@/features/runs/ResultSummary'
import { DecisionTracePanel } from '@/features/runs/DecisionTracePanel'
import { ArtifactPanel } from '@/features/runs/ArtifactPanel'
import { GovernancePanel } from '@/features/runs/GovernancePanel'
import { TestFlightPanel } from '@/features/runs/TestFlightPanel'
import { StudentModeView } from '@/features/student/StudentModeView'
import { useRunEvents } from '@/lib/useRunEvents'
import { useRunStatus } from '@/lib/useRunStatus'
import { useRunResult } from '@/lib/useRunResult'
import { useDecisionTrace } from '@/lib/useDecisionTrace'

export function RunPage() {
  const { runId = '' } = useParams<{ runId: string }>()
  const [viewMode, setViewMode] = useState<'engineer' | 'student'>('engineer')
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
        <div className="flex items-center gap-3">
          <Link to="/history" className="text-muted-foreground hover:text-foreground text-sm">
            History
          </Link>
          <ThemeToggle />
        </div>
      </div>

      <Card>
        <CardHeader>
          <RunStatusHeader runId={runId} status={status} connectionState={connectionState} />
        </CardHeader>
      </Card>

      {/* Engineer vs Student Mode Toggle */}
      <div className="flex items-center justify-between bg-card p-2 rounded-xl border border-border">
        <div className="flex items-center gap-1 bg-muted p-1 rounded-lg border border-border/60">
          <button
            onClick={() => setViewMode('engineer')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-semibold uppercase tracking-wider transition-all ${
              viewMode === 'engineer'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <Code2 className="w-3.5 h-3.5" /> Engineer Mode
          </button>
          <button
            onClick={() => setViewMode('student')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-semibold uppercase tracking-wider transition-all ${
              viewMode === 'student'
                ? 'bg-primary text-primary-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <GraduationCap className="w-3.5 h-3.5" /> Student Mode
          </button>
        </div>
        <span className="text-xs text-muted-foreground hidden sm:block">
          {viewMode === 'engineer'
            ? 'Detailed technical logs, artifacts, and engineering outputs'
            : 'Interactive, evidence-grounded educational learning environment'}
        </span>
      </div>

      {statusError && <p className="text-destructive text-sm">{statusError}</p>}

      {viewMode === 'student' ? (
        <StudentModeView
          runId={runId}
          candidates={result?.comparison?.models ?? []}
          winningModelId={result?.comparison?.recommended_model_id}
          selectionJustification={result?.comparison?.selection_metric ? `Selected via ${result.comparison.selection_metric}` : undefined}
          baselineAccuracy={result?.baseline?.baseline?.accuracy}
        />
      ) : (
        <>
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
              <ArtifactPanel runId={runId} runStatus={status?.status ?? ''} />
              <GovernancePanel runId={runId} runStatus={status?.status ?? ''} />
              <TestFlightPanel runId={runId} runStatus={status?.status ?? ''} />
            </>
          )}
        </>
      )}
    </div>
  )
}

