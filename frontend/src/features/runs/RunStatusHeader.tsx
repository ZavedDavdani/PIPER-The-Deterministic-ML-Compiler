import { CheckCircle2, Loader2, RotateCcw, XCircle, Circle, WifiOff } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import type { RunStatusResponse } from '@/lib/types'
import type { SseConnectionState } from '@/lib/useRunEvents'
import { cn } from '@/lib/utils'

interface RunStatusHeaderProps {
  runId: string
  status: RunStatusResponse | null
  connectionState: SseConnectionState
}

const STATUS_STYLES: Record<string, { variant: 'default' | 'success' | 'warning' | 'destructive' | 'secondary'; icon: typeof Circle; label: string }> = {
  initialized: { variant: 'secondary', icon: Circle, label: 'Initialized' },
  running: { variant: 'default', icon: Loader2, label: 'Running' },
  replanning: { variant: 'warning', icon: RotateCcw, label: 'Re-planning' },
  completed: { variant: 'success', icon: CheckCircle2, label: 'Completed' },
  failed: { variant: 'destructive', icon: XCircle, label: 'Failed' },
}

export function RunStatusHeader({ runId, status, connectionState }: RunStatusHeaderProps) {
  const style = (status && STATUS_STYLES[status.status]) ?? STATUS_STYLES.initialized
  const Icon = style.icon
  const isSpinning = status?.status === 'running' || status?.status === 'replanning'

  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex flex-col gap-1">
        <span className="text-muted-foreground font-mono text-xs">{runId}</span>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={style.variant}>
            <Icon className={cn('size-3', isSpinning && 'animate-spin')} />
            {style.label}
          </Badge>
          {status?.current_node && (
            <Badge variant="outline">node: {status.current_node}</Badge>
          )}
          {status && <Badge variant="outline">attempt {status.attempt}</Badge>}
        </div>
      </div>
      <div className="flex items-center gap-1.5 text-xs">
        {connectionState === 'open' && (
          <span className="text-success-foreground flex items-center gap-1">
            <span className="bg-success size-2 animate-pulse rounded-full" /> live
          </span>
        )}
        {connectionState === 'connecting' && (
          <span className="text-muted-foreground flex items-center gap-1">
            <Loader2 className="size-3 animate-spin" /> connecting
          </span>
        )}
        {connectionState === 'closed' && <span className="text-muted-foreground">stream closed</span>}
        {connectionState === 'error' && (
          <span className="text-destructive flex items-center gap-1">
            <WifiOff className="size-3" /> connection lost
          </span>
        )}
      </div>
    </div>
  )
}
