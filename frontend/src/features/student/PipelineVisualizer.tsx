import { useState } from 'react'
import { ArrowRight, CheckCircle2, AlertCircle, Clock, Layers } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { PipelineVisualization, PipelineNode } from '@/lib/types'

interface PipelineVisualizerProps {
  pipeline: PipelineVisualization | null
  isLoading?: boolean
}

export function PipelineVisualizer({ pipeline, isLoading }: PipelineVisualizerProps) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>('dataset')

  if (isLoading) {
    return (
      <Card className="border-border">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Layers className="w-5 h-5 text-primary" /> ML Pipeline Architecture
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-24 bg-muted/60 animate-pulse rounded-lg" />
        </CardContent>
      </Card>
    )
  }

  if (!pipeline || pipeline.nodes.length === 0) {
    return (
      <Card className="border-border">
        <CardContent className="py-8 text-center text-muted-foreground text-sm">
          No pipeline visualization data available.
        </CardContent>
      </Card>
    )
  }

  const selectedNode = pipeline.nodes.find((n) => n.id === selectedNodeId) ?? pipeline.nodes[0]

  const getNodeIcon = (status: PipelineNode['status']) => {
    switch (status) {
      case 'passed':
        return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
      case 'failed':
        return <AlertCircle className="w-3.5 h-3.5 text-rose-500" />
      case 'pending':
        return <Clock className="w-3.5 h-3.5 text-amber-500" />
      default:
        return null
    }
  }

  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center gap-2">
              <Layers className="w-5 h-5 text-primary" /> End-to-End Pipeline Visualization
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              Click any stage in the flow to inspect recorded execution details.
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Flowchart Ribbon */}
        <div className="flex items-center gap-2 overflow-x-auto pb-3 pt-1 px-1">
          {pipeline.nodes.map((node, idx) => {
            const isSelected = node.id === selectedNodeId
            return (
              <div key={node.id} className="flex items-center gap-2 flex-shrink-0">
                <button
                  onClick={() => setSelectedNodeId(node.id)}
                  className={`px-3 py-2 rounded-lg border text-xs font-medium flex items-center gap-1.5 transition-all ${
                    isSelected
                      ? 'border-primary bg-primary/10 text-primary ring-2 ring-primary/20 shadow-sm'
                      : 'border-border bg-card hover:border-border hover:bg-muted/40 text-foreground'
                  }`}
                >
                  {getNodeIcon(node.status)}
                  <span>{node.name}</span>
                </button>
                {idx < pipeline.nodes.length - 1 && (
                  <ArrowRight className="w-3.5 h-3.5 text-muted-foreground/50 flex-shrink-0" />
                )}
              </div>
            )
          })}
        </div>

        {/* Selected Node Details Card */}
        {selectedNode && (
          <div className="p-4 rounded-xl border border-border bg-muted/20 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="text-xs font-mono">
                  {selectedNode.stage}
                </Badge>
                <h4 className="text-sm font-semibold text-foreground">{selectedNode.name}</h4>
              </div>
              <Badge
                variant="outline"
                className={`text-xs ${
                  selectedNode.status === 'passed'
                    ? 'text-emerald-600 border-emerald-300'
                    : selectedNode.status === 'failed'
                    ? 'text-rose-600 border-rose-300'
                    : 'text-amber-600 border-amber-300'
                }`}
              >
                {selectedNode.status.toUpperCase()}
              </Badge>
            </div>

            <p className="text-xs sm:text-sm text-foreground bg-background p-3 rounded-lg border border-border/70">
              {selectedNode.summary}
            </p>

            {Object.keys(selectedNode.details).length > 0 && (
              <div className="bg-muted/50 p-2.5 rounded-lg border border-border/40 text-xs font-mono text-muted-foreground">
                <span className="font-semibold text-foreground/80 block mb-1 font-sans">Recorded Evidence:</span>
                <pre className="whitespace-pre-wrap">{JSON.stringify(selectedNode.details, null, 2)}</pre>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
