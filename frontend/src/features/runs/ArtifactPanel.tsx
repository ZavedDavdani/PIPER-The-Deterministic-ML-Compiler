import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { artifactFileUrl, generateArtifacts, getArtifactStatus } from '@/lib/api'
import type { ArtifactStatusResponse } from '@/lib/types'

interface ArtifactPanelProps {
  runId: string
  runStatus: string
}

const FILE_LABELS: Record<string, string> = {
  'pipeline.joblib': 'Fitted pipeline',
  'pipeline.py': 'Inference script',
  'training_reproduction.ipynb': 'Reproduction notebook',
  'manifest.json': 'Manifest',
  'evidence.json': 'Evidence',
  'hashes.json': 'SHA-256 hashes',
}

export function ArtifactPanel({ runId, runStatus }: ArtifactPanelProps) {
  const [status, setStatus] = useState<ArtifactStatusResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!runId || (runStatus !== 'completed' && runStatus !== 'failed')) return
    let cancelled = false
    void getArtifactStatus(runId)
      .then((next) => {
        if (!cancelled) setStatus(next)
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Could not load artifact status.')
      })
    return () => {
      cancelled = true
    }
  }, [runId, runStatus])

  async function onGenerate() {
    setError(null)
    setBusy(true)
    try {
      const next = await generateArtifacts(runId)
      setStatus(next)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Artifact generation failed.')
      try {
        setStatus(await getArtifactStatus(runId))
      } catch {
        /* keep previous */
      }
    } finally {
      setBusy(false)
    }
  }

  const verified = status?.artifact_status === 'VERIFIED'
  const failed = status?.artifact_status === 'FAILED'
  const canGenerate = runStatus === 'completed'

  return (
    <Card>
      <CardHeader>
        <CardTitle>Artifacts</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        {verified && (
          <p className="rounded-md border border-green-600/40 bg-green-600/10 px-3 py-2 font-semibold text-green-800 dark:text-green-400">
            VERIFIED ARTIFACT
          </p>
        )}
        {failed && (
          <p className="rounded-md border border-destructive/40 bg-destructive/10 text-destructive px-3 py-2 font-semibold">
            ARTIFACT GENERATION FAILED
          </p>
        )}
        <p>
          Status: <span className="font-medium">{status?.artifact_status ?? 'NOT_GENERATED'}</span>
        </p>
        <p>
          Model: <span className="font-medium">{status?.algorithm ?? '—'}</span>
        </p>
        <p>
          Parity: <span className="font-medium">{status?.parity_status ?? 'not_run'}</span>
        </p>
        {status?.error?.message && <p className="text-destructive">{status.error.message}</p>}
        {error && <p className="text-destructive">{error}</p>}
        {canGenerate && (
          <Button type="button" onClick={() => void onGenerate()} disabled={busy}>
            {busy ? 'Generating…' : 'Generate artifacts'}
          </Button>
        )}
        {!canGenerate && (
          <p className="text-muted-foreground">
            Deployable artifacts are only generated for completed, guardrail-valid runs.
          </p>
        )}
        {verified && status.files.length > 0 && (
          <ul className="flex flex-col gap-1">
            {status.files.map((name) => (
              <li key={name}>
                <a
                  className="text-primary underline-offset-4 hover:underline"
                  href={artifactFileUrl(runId, name)}
                  download={name}
                >
                  Download {FILE_LABELS[name] ?? name}
                </a>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}
