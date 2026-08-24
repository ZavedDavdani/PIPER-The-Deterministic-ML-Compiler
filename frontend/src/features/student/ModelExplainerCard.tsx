import { Cpu, Trophy, Check, AlertTriangle, ShieldCheck } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { ModelConceptExplanation } from '@/lib/types'

interface ModelExplainerCardProps {
  modelConcepts: ModelConceptExplanation[]
  winningModelId?: string | null
  selectionJustification?: string | null
}

export function ModelExplainerCard({
  modelConcepts,
  selectionJustification,
}: ModelExplainerCardProps) {
  if (!modelConcepts || modelConcepts.length === 0) {
    return null
  }

  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center gap-2">
              <Cpu className="w-5 h-5 text-primary" /> Model Architectures Explained
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              How candidate algorithms work, why PIPER compared them, and what their tradeoffs are.
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {selectionJustification && (
          <div className="p-3 bg-primary/5 rounded-xl border border-primary/20 flex items-start gap-2.5">
            <Trophy className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />
            <div className="text-xs">
              <span className="font-semibold text-foreground">Model Selection Rationale:</span>
              <p className="text-muted-foreground mt-0.5">{selectionJustification}</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {modelConcepts.map((mc) => (
            <div
              key={mc.algorithm}
              className={`p-4 rounded-xl border space-y-3 transition-all ${
                mc.is_winner
                  ? 'border-primary/50 bg-primary/5 shadow-sm'
                  : 'border-border bg-card/60'
              }`}
            >
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-foreground flex items-center gap-1.5">
                  {mc.is_winner && <Trophy className="w-3.5 h-3.5 text-amber-500" />}
                  {mc.name}
                </h4>
                {mc.is_winner ? (
                  <Badge className="bg-primary/20 text-primary border-primary/30 text-xs">
                    Winning Model
                  </Badge>
                ) : (
                  <Badge variant="outline" className="text-muted-foreground text-xs">
                    Candidate
                  </Badge>
                )}
              </div>

              <p className="text-xs text-muted-foreground leading-relaxed">{mc.concept}</p>

              <div className="space-y-2 pt-2 border-t border-border/50 text-xs">
                <div>
                  <span className="font-semibold text-foreground/90 flex items-center gap-1 mb-1">
                    <Check className="w-3.5 h-3.5 text-emerald-500" /> Key Strengths
                  </span>
                  <ul className="list-disc list-inside text-muted-foreground space-y-0.5 pl-1">
                    {mc.strengths.map((s, idx) => (
                      <li key={idx}>{s}</li>
                    ))}
                  </ul>
                </div>

                <div>
                  <span className="font-semibold text-foreground/90 flex items-center gap-1 mb-1">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-500" /> Trade-offs & Limitations
                  </span>
                  <ul className="list-disc list-inside text-muted-foreground space-y-0.5 pl-1">
                    {mc.tradeoffs.map((t, idx) => (
                      <li key={idx}>{t}</li>
                    ))}
                  </ul>
                </div>

                <div className="bg-background/80 p-2 rounded border border-border/50">
                  <span className="font-semibold text-foreground/90 flex items-center gap-1">
                    <ShieldCheck className="w-3.5 h-3.5 text-primary" /> How PIPER Applied It
                  </span>
                  <p className="text-muted-foreground mt-0.5">{mc.how_piper_used_it}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
