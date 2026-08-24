import { useState } from 'react'
import { CheckCircle2, Circle, AlertCircle, Clock, BookOpen } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import type { LearningJourney, LearningJourneyStage } from '@/lib/types'

interface LearningJourneyStepperProps {
  journey: LearningJourney | null
  isLoading?: boolean
}

export function LearningJourneyStepper({ journey, isLoading }: LearningJourneyStepperProps) {
  const [selectedStageId, setSelectedStageId] = useState<number | null>(1)

  if (isLoading) {
    return (
      <Card className="border-border">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-primary" /> Guided ML Learning Journey
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="h-4 bg-muted animate-pulse rounded" />
          <div className="space-y-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-12 bg-muted/60 animate-pulse rounded" />
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!journey || journey.stages.length === 0) {
    return (
      <Card className="border-border">
        <CardContent className="py-8 text-center text-muted-foreground text-sm">
          No learning journey data recorded for this run.
        </CardContent>
      </Card>
    )
  }

  const completedCount = journey.stages.filter((s) => s.status === 'completed').length
  const progressPercent = Math.round((completedCount / journey.stages.length) * 100)
  const selectedStage = journey.stages.find((s) => s.stage_id === selectedStageId) ?? journey.stages[0]

  const getStatusIcon = (status: LearningJourneyStage['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0" />
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-rose-500 flex-shrink-0" />
      case 'in_progress':
        return <Clock className="w-4 h-4 text-amber-500 animate-spin flex-shrink-0" />
      case 'skipped':
        return <Circle className="w-4 h-4 text-muted-foreground/60 flex-shrink-0" />
      default:
        return <Circle className="w-4 h-4 text-muted-foreground/40 flex-shrink-0" />
    }
  }

  const getStatusBadge = (status: LearningJourneyStage['status']) => {
    switch (status) {
      case 'completed':
        return <Badge variant="outline" className="text-emerald-600 border-emerald-300 text-xs">Completed</Badge>
      case 'failed':
        return <Badge variant="outline" className="text-rose-600 border-rose-300 text-xs">Failed</Badge>
      case 'in_progress':
        return <Badge variant="outline" className="text-amber-600 border-amber-300 text-xs">In Progress</Badge>
      case 'skipped':
        return <Badge variant="outline" className="text-muted-foreground text-xs">Skipped</Badge>
      default:
        return <Badge variant="outline" className="text-muted-foreground text-xs">Not Reached</Badge>
    }
  }

  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <CardTitle className="text-lg flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-primary" /> 14-Stage ML Learning Journey
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              Step-by-step educational breakdown of what PIPER executed on this dataset.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-medium text-muted-foreground">
              {completedCount} of {journey.stages.length} Stages ({progressPercent}%)
            </span>
            <Progress value={progressPercent} className="w-24 h-2" />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Stages List */}
          <div className="lg:col-span-5 space-y-1.5 max-h-[520px] overflow-y-auto pr-2">
            {journey.stages.map((stage) => {
              const isSelected = stage.stage_id === selectedStageId
              return (
                <button
                  key={stage.stage_id}
                  onClick={() => setSelectedStageId(stage.stage_id)}
                  className={`w-full text-left p-2.5 rounded-lg border text-sm transition-all flex items-center justify-between gap-2 ${
                    isSelected
                      ? 'border-primary bg-primary/5 shadow-sm'
                      : 'border-border/60 hover:border-border hover:bg-muted/30'
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span className="text-xs font-mono font-semibold text-muted-foreground w-5 text-right">
                      {stage.stage_id}.
                    </span>
                    {getStatusIcon(stage.status)}
                    <span className={`truncate font-medium text-xs sm:text-sm ${isSelected ? 'text-primary' : 'text-foreground'}`}>
                      {stage.title}
                    </span>
                  </div>
                  {getStatusBadge(stage.status)}
                </button>
              )
            })}
          </div>

          {/* Selected Stage Detail Panel */}
          <div className="lg:col-span-7">
            {selectedStage ? (
              <div className="p-4 rounded-xl border border-border bg-card/80 space-y-4 shadow-sm">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <span className="text-xs font-semibold uppercase tracking-wider text-primary">
                      Stage {selectedStage.stage_id} of 14 • {selectedStage.concept}
                    </span>
                    <h3 className="text-base font-semibold text-foreground mt-0.5">
                      {selectedStage.title}
                    </h3>
                  </div>
                  {getStatusBadge(selectedStage.status)}
                </div>

                <div className="text-xs text-muted-foreground bg-muted/40 p-3 rounded-lg border border-border/50">
                  {selectedStage.description}
                </div>

                <div className="space-y-3">
                  <div>
                    <h4 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-1">
                      What Happened in This Run
                    </h4>
                    <p className="text-xs sm:text-sm text-foreground bg-background p-3 rounded-lg border border-border/70">
                      {selectedStage.summary}
                    </p>
                  </div>

                  <div className="bg-primary/5 p-3 rounded-lg border border-primary/20 space-y-1">
                    <h4 className="text-xs font-semibold text-primary flex items-center gap-1.5">
                      <BookOpen className="w-3.5 h-3.5" /> Key ML Concept: {selectedStage.concept}
                    </h4>
                    <p className="text-xs text-muted-foreground">
                      Understanding how this stage contributes to generalization on unseen test data without leaking information.
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-xs text-muted-foreground">
                Select a stage on the left to inspect its educational evidence.
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
