import { useState } from 'react'
import { Loader2, Play } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { ApiError, createRun } from '@/lib/api'

interface RunCreateFormProps {
  datasetId: string
  columns: string[]
  onCreated: (runId: string) => void
}

export function RunCreateForm({ datasetId, columns, onCreated }: RunCreateFormProps) {
  const [targetColumn, setTargetColumn] = useState(columns[0] ?? '')
  const [maxRetries, setMaxRetries] = useState(2)
  const [isCreating, setIsCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!targetColumn) return
    setError(null)
    setIsCreating(true)
    try {
      const response = await createRun({ dataset_id: datasetId, target_column: targetColumn, max_retries: maxRetries })
      onCreated(response.run_id)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not start the run — is the PIPER backend running?')
    } finally {
      setIsCreating(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="target-column">Objective — predict</Label>
        <Select value={targetColumn} onValueChange={setTargetColumn} disabled={columns.length === 0}>
          <SelectTrigger id="target-column">
            <SelectValue placeholder="Select a column" />
          </SelectTrigger>
          <SelectContent>
            {columns.map((col) => (
              <SelectItem key={col} value={col}>
                {col}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-muted-foreground text-xs">PIPER will plan, clean, train, and validate a model that predicts this column from the rest.</p>
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="max-retries">Max REPLAN retries</Label>
        <Input
          id="max-retries"
          type="number"
          min={0}
          max={20}
          value={maxRetries}
          onChange={(e) => setMaxRetries(Number(e.target.value))}
          className="w-28"
        />
        <p className="text-muted-foreground text-xs">
          If a guardrail rejects the pipeline, PIPER re-plans up to this many times before reporting a final failure.
        </p>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Could not start run</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Button type="submit" disabled={isCreating || !targetColumn} className="self-start">
        {isCreating ? <Loader2 className="animate-spin" /> : <Play />}
        Start run
      </Button>
    </form>
  )
}
