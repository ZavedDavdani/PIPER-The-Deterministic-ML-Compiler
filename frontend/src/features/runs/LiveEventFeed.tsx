import { AlertTriangle, CheckCircle2, Circle, Hammer, RotateCcw, XCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import type { TraceEvent } from '@/lib/types'
import { cn } from '@/lib/utils'

interface LiveEventFeedProps {
  events: TraceEvent[]
}

function isReplanTransition(event: TraceEvent): boolean {
  const updatedFields = event.evidence?.updated_fields
  return (
    event.node === 'plan_entry' &&
    Array.isArray(updatedFields) &&
    updatedFields.includes('retry_count')
  )
}

function eventVisual(event: TraceEvent) {
  if (event.event_type === 'run_failed' || event.event_type === 'node_failed') {
    return { icon: XCircle, className: 'text-destructive' }
  }
  if (event.event_type === 'run_completed') {
    return { icon: CheckCircle2, className: 'text-success' }
  }
  if (isReplanTransition(event)) {
    return { icon: RotateCcw, className: 'text-warning' }
  }
  if (event.event_type === 'tool_called') {
    return { icon: Hammer, className: 'text-muted-foreground' }
  }
  if (event.severity === 'warning') {
    return { icon: AlertTriangle, className: 'text-warning' }
  }
  return { icon: Circle, className: 'text-muted-foreground' }
}

function formatTime(timestamp: string): string {
  const date = new Date(timestamp)
  if (Number.isNaN(date.getTime())) return timestamp
  return date.toLocaleTimeString(undefined, { hour12: false })
}

function groupByAttempt(events: TraceEvent[]): Map<number, TraceEvent[]> {
  const groups = new Map<number, TraceEvent[]>()
  for (const event of events) {
    const list = groups.get(event.attempt) ?? []
    list.push(event)
    groups.set(event.attempt, list)
  }
  return groups
}

export function LiveEventFeed({ events }: LiveEventFeedProps) {
  if (events.length === 0) {
    return <p className="text-muted-foreground py-6 text-center text-sm">Waiting for PIPER to start…</p>
  }

  const groups = groupByAttempt(events)

  return (
    <div className="flex max-h-[32rem] flex-col gap-4 overflow-y-auto pr-1">
      {[...groups.entries()].map(([attempt, attemptEvents]) => (
        <div key={attempt} className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <Badge variant="outline">attempt {attempt}</Badge>
            <Separator className="flex-1" />
          </div>
          <ol className="flex flex-col gap-1.5">
            {attemptEvents.map((event) => {
              const { icon: Icon, className } = eventVisual(event)
              const replanTransition = isReplanTransition(event)
              return (
                <li key={event.step_id} className="flex items-start gap-2.5 text-sm">
                  <Icon className={cn('mt-0.5 size-4 shrink-0', className)} />
                  <div className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-2">
                    <span className="text-muted-foreground font-mono text-[11px]">{formatTime(event.timestamp)}</span>
                    <span className="font-medium">{event.node}</span>
                    {event.tool_name && (
                      <Badge variant="secondary" className="text-[10px]">
                        {event.tool_name}
                      </Badge>
                    )}
                    {replanTransition && (
                      <Badge variant="warning" className="text-[10px]">
                        REPLAN → attempt {attempt}
                      </Badge>
                    )}
                    {event.message && <span className="text-muted-foreground truncate">{event.message}</span>}
                  </div>
                </li>
              )
            })}
          </ol>
        </div>
      ))}
    </div>
  )
}
