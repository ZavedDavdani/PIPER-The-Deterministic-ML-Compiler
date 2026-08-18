import { AlertTriangle, CheckCircle2, XCircle } from 'lucide-react'
import type { PipelineValidationResult, ValidationCheck } from '@/lib/types'
import { cn } from '@/lib/utils'

interface ValidationChecksPanelProps {
  validation: PipelineValidationResult
}

function CheckRow({ check }: { check: ValidationCheck }) {
  const Icon = check.passed ? CheckCircle2 : check.severity === 'warning' ? AlertTriangle : XCircle
  const colorClass = check.passed ? 'text-success' : check.severity === 'warning' ? 'text-warning' : 'text-destructive'

  return (
    <li className="flex items-start gap-2 text-sm">
      <Icon className={cn('mt-0.5 size-4 shrink-0', colorClass)} />
      <div>
        <span className="font-medium">{check.check.replace(/_/g, ' ')}</span>
        <p className="text-muted-foreground text-xs">{check.message}</p>
      </div>
    </li>
  )
}

export function ValidationChecksPanel({ validation }: ValidationChecksPanelProps) {
  return (
    <div className="flex flex-col gap-3">
      <ul className="flex flex-col gap-2">
        {validation.checks.map((check, i) => (
          <CheckRow key={`${check.check}-${i}`} check={check} />
        ))}
      </ul>
      <p className="text-muted-foreground text-xs italic">{validation.scope_note}</p>
    </div>
  )
}
