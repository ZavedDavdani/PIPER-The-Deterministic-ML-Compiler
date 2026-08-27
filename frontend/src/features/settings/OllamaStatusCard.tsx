import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { getProviderStatus, updateProviderConfig } from '@/lib/api'
import type { ProviderStatusResponse } from '@/lib/types'

type ProviderChoice = 'openai' | 'ollama' | 'gemini'

export function OllamaStatusCard() {
  const [status, setStatus] = useState<ProviderStatusResponse | null>(null)
  const [provider, setProvider] = useState<ProviderChoice>('ollama')
  const [model, setModel] = useState('')
  const [host, setHost] = useState('http://localhost:11434')
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  function applyStatus(body: ProviderStatusResponse) {
    setStatus(body)
    if (body.provider === 'openai' || body.provider === 'ollama' || body.provider === 'gemini') {
      setProvider(body.provider)
    } else {
      setProvider('ollama')
    }
    setModel(body.model)
    if (typeof body.details.host === 'string') {
      setHost(body.details.host)
    }
    setError(body.error)
  }

  function refresh() {
    getProviderStatus()
      .then((body) => {
        applyStatus(body)
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
      const body = await updateProviderConfig({
        provider,
        model,
        ...(provider === 'ollama' ? { host } : {}),
      })
      applyStatus(body)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update provider config.')
    } finally {
      setSaving(false)
    }
  }

  const availableModels = status?.available_models ?? []
  const modelUsesSelect =
    (provider === 'ollama' || provider === 'gemini') && availableModels.length > 0

  return (
    <Card>
      <CardHeader>
        <CardTitle>LLM Provider</CardTitle>
        <CardDescription>
          Choose the planner provider and model. Credentials stay on the backend; only provider settings are sent here.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        {error && <p className="text-destructive">{error}</p>}
        {status && (
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
            <dt className="text-muted-foreground">Reachable</dt>
            <dd>{status.reachable ? 'yes' : 'no'}</dd>
            {(provider === 'openai' || provider === 'gemini') && (
              <>
                <dt className="text-muted-foreground">API key</dt>
                <dd>{status.details.has_api_key ? 'configured' : 'not set'}</dd>
              </>
            )}
            {provider === 'ollama' && availableModels.length > 0 && (
              <>
                <dt className="text-muted-foreground">Installed</dt>
                <dd>{availableModels.join(', ')}</dd>
              </>
            )}
          </dl>
        )}
        <form className="flex flex-col gap-3" onSubmit={onSubmit}>
          <label className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">Provider</span>
            <Select value={provider} onValueChange={(value) => setProvider(value as ProviderChoice)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="openai">OpenAI</SelectItem>
                <SelectItem value="ollama">Ollama</SelectItem>
                <SelectItem value="gemini">Gemini</SelectItem>
              </SelectContent>
            </Select>
          </label>

          {provider === 'ollama' && (
            <label className="flex flex-col gap-1">
              <span className="text-muted-foreground text-xs">Host</span>
              <input
                className="border-input bg-background h-9 rounded-md border px-2 font-mono text-sm"
                value={host}
                onChange={(e) => setHost(e.target.value)}
                name="host"
              />
            </label>
          )}

          <label className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">Model</span>
            {modelUsesSelect ? (
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map((item) => (
                    <SelectItem key={item} value={item}>
                      {item}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <input
                className="border-input bg-background h-9 rounded-md border px-2 font-mono text-sm"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                name="model"
                placeholder={
                  provider === 'openai'
                    ? 'gpt-5.6-luna'
                    : provider === 'gemini'
                      ? 'gemini-model-id'
                      : 'qwen3:8b'
                }
              />
            )}
          </label>

          <Button type="submit" size="sm" disabled={saving || !model.trim() || (provider === 'ollama' && !host.trim())}>
            Save
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
