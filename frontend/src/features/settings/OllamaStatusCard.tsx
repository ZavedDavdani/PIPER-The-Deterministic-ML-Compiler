import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { getOllamaStatus, updateOllamaConfig } from '@/lib/api'
import type { OllamaStatusResponse } from '@/lib/types'

export function OllamaStatusCard() {
  const [status, setStatus] = useState<OllamaStatusResponse | null>(null)
  const [model, setModel] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  function refresh() {
    getOllamaStatus()
      .then((body) => {
        setStatus(body)
        setModel(body.model)
        setError(null)
      })
      .catch((err: Error) => setError(err.message))
  }

  useEffect(() => {
    refresh()
  }, [])

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    setSaving(true)
    try {
      const body = await updateOllamaConfig({ model })
      setStatus(body)
      setModel(body.model)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update Ollama config.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Ollama</CardTitle>
        <CardDescription>
          Local planner process only. PIPER does not send temperature or other sampling options.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        {error && <p className="text-destructive">{error}</p>}
        {status && (
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
            <dt className="text-muted-foreground">Host</dt>
            <dd className="font-mono">{status.host}</dd>
            <dt className="text-muted-foreground">Reachable</dt>
            <dd>{status.reachable ? 'yes' : 'no'}</dd>
            <dt className="text-muted-foreground">Installed</dt>
            <dd>{status.models.length ? status.models.join(', ') : '—'}</dd>
          </dl>
        )}
        <form className="flex items-end gap-2" onSubmit={onSubmit}>
          <label className="flex flex-1 flex-col gap-1">
            <span className="text-muted-foreground text-xs">Model</span>
            <input
              className="border-input bg-background h-9 rounded-md border px-2 font-mono text-sm"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              name="model"
            />
          </label>
          <Button type="submit" size="sm" disabled={saving || !model.trim()}>
            Save
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
