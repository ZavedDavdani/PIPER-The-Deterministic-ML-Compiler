import { useState } from 'react'
import { HelpCircle, BookOpen, Layers, Lightbulb } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { OperationExplanation, ExplanationLevel } from '@/lib/types'

interface DecisionWhyCardProps {
  operation: OperationExplanation
  defaultLevel?: ExplanationLevel
}

export function DecisionWhyCard({ operation, defaultLevel = 'beginner' }: DecisionWhyCardProps) {
  const [level, setLevel] = useState<ExplanationLevel>(defaultLevel)

  return (
    <Card className="border-border">
      <CardHeader className="pb-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="font-mono text-xs">
              {operation.tool_name}
            </Badge>
            <CardTitle className="text-sm font-semibold text-foreground">
              Why did PIPER do this?
            </CardTitle>
          </div>
          {/* Level Switcher */}
          <div className="flex items-center gap-1 bg-muted p-0.5 rounded-lg border border-border/60">
            {(['beginner', 'intermediate', 'advanced'] as ExplanationLevel[]).map((lvl) => (
              <button
                key={lvl}
                onClick={() => setLevel(lvl)}
                className={`px-2 py-1 rounded text-xs font-medium capitalize transition-all ${
                  level === lvl
                    ? 'bg-background text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {lvl}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-2">
        {/* What Happened */}
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1 mb-1">
            <HelpCircle className="w-3.5 h-3.5 text-primary" /> What happened?
          </span>
          <p className="text-xs sm:text-sm text-foreground bg-muted/30 p-2.5 rounded-lg border border-border/50">
            {operation.what_happened}
          </p>
        </div>

        {/* Why */}
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1 mb-1">
            <Lightbulb className="w-3.5 h-3.5 text-amber-500" /> Why PIPER made this decision
          </span>
          <p className="text-xs sm:text-sm text-foreground bg-muted/30 p-2.5 rounded-lg border border-border/50">
            {operation.why}
          </p>
        </div>

        {/* Concept */}
        {operation.concept && (
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1 mb-1">
              <BookOpen className="w-3.5 h-3.5 text-primary" /> Relevant ML Concept
            </span>
            <div className="text-xs font-medium text-primary bg-primary/5 px-2.5 py-1.5 rounded-md border border-primary/20 inline-block">
              {operation.concept}
            </div>
          </div>
        )}

        {/* Alternatives */}
        {operation.alternative_consideration && (
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1 mb-1">
              <Layers className="w-3.5 h-3.5 text-indigo-500" /> What could have happened with an alternative?
            </span>
            <p className="text-xs text-muted-foreground bg-muted/20 p-2.5 rounded-lg border border-border/40">
              {operation.alternative_consideration}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
