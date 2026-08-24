import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  deploymentPackageFileUrl,
  generateDeploymentPackage,
  getArtifactStatus,
  getDeployment,
  testFlight,
  testFlightCsvBlob,
} from '@/lib/api'
import type { ArtifactStatusResponse, DeploymentReadinessResponse, PredictResponse } from '@/lib/types'

interface TestFlightPanelProps {
  runId: string
  runStatus: string
}

export function TestFlightPanel({ runId, runStatus }: TestFlightPanelProps) {
  const [artifact, setArtifact] = useState<ArtifactStatusResponse | null>(null)
  const [readiness, setReadiness] = useState<DeploymentReadinessResponse | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [packaged, setPackaged] = useState(false)

  async function refresh() {
    const status = await getArtifactStatus(runId)
    setArtifact(status)
    if (status.artifact_status === 'VERIFIED') {
      setReadiness(await getDeployment(runId))
    } else {
      setReadiness(null)
    }
  }

  useEffect(() => {
    if (!runId || (runStatus !== 'completed' && runStatus !== 'failed')) return
    let cancelled = false
    void getArtifactStatus(runId)
      .then(async (status) => {
        if (cancelled) return
        setArtifact(status)
        if (status.artifact_status === 'VERIFIED') {
          setReadiness(await getDeployment(runId))
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Could not load deployment status.')
      })
    return () => {
      cancelled = true
    }
  }, [runId, runStatus])

  const verified = artifact?.artifact_status === 'VERIFIED'
  const ready = readiness?.status === 'READY'

  async function onPredict() {
    if (!file) return
    setError(null)
    setBusy(true)
    try {
      setResult(await testFlight(runId, file))
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Prediction failed.')
    } finally {
      setBusy(false)
    }
  }

  async function onDownload() {
    if (!file) return
    setError(null)
    try {
      const blob = await testFlightCsvBlob(runId, file)
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'predictions.csv'
      link.click()
      URL.revokeObjectURL(url)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Download failed.')
    }
  }

  async function onPackage() {
    setError(null)
    try {
      await generateDeploymentPackage(runId)
      setPackaged(true)
      await refresh()
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Package generation failed.')
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Test Flight</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        <div className="grid gap-2 sm:grid-cols-2">
          <p className="rounded-md border px-3 py-2">
            <span className="font-semibold">TRAINING DATA</span>
            <span className="text-muted-foreground mt-1 block">
              Used to fit this run. Test Flight never retrains on it.
            </span>
          </p>
          <p className="rounded-md border border-green-600/40 bg-green-600/10 px-3 py-2">
            <span className="font-semibold">NEW UNSEEN DATA</span>
            <span className="mt-1 block">Upload a CSV the model has not been refit on.</span>
          </p>
        </div>

        {error && <p className="text-destructive">{error}</p>}

        <p>
          Artifact: <span className="font-medium">{artifact?.artifact_status ?? 'NOT_GENERATED'}</span>
          {artifact?.algorithm ? ` (${artifact.algorithm})` : ''}
        </p>
        <p>
          Deployment: <span className="font-medium">{readiness?.status ?? 'NOT_READY'}</span>
        </p>
        {readiness?.required_columns?.length ? (
          <p>Required columns: {readiness.required_columns.join(', ')}</p>
        ) : null}

        <Button type="button" variant="outline" onClick={() => void refresh()}>
          Select verified artifact
        </Button>

        {!verified && (
          <p className="text-muted-foreground">
            Generate a VERIFIED artifact first. Test Flight only loads pipeline.joblib — it does not call the planner.
          </p>
        )}

        {verified && ready && (
          <>
            <label className="flex flex-col gap-1">
              <span className="font-medium">Unseen CSV</span>
              <input
                aria-label="Unseen CSV"
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </label>
            <Button type="button" onClick={() => void onPredict()} disabled={!file || busy}>
              {busy ? 'Scoring…' : 'Run prediction'}
            </Button>
          </>
        )}

        {result && (
          <>
            <p>
              Schema: <span className="font-medium">{result.schema_status}</span> · rows:{' '}
              <span className="font-medium">{result.row_count}</span> · parity:{' '}
              <span className="font-medium">{result.parity.parity_status}</span>
            </p>
            <p className="font-semibold">Sample predictions</p>
            <ul className="list-disc pl-5">
              {result.sample.slice(0, 8).map((row, index) => (
                <li key={index}>
                  {String(row.prediction)} (row {index + 1})
                </li>
              ))}
            </ul>
            <Button type="button" variant="outline" onClick={() => void onDownload()} disabled={!file}>
              Download prediction CSV
            </Button>
          </>
        )}

        {verified && (
          <div className="flex flex-col gap-1">
            <Button type="button" variant="outline" onClick={() => void onPackage()}>
              Generate optional deployment package
            </Button>
            {packaged && (
              <>
                <a className="text-primary underline-offset-4 hover:underline" href={deploymentPackageFileUrl(runId, 'inference.py')} download="inference.py">
                  Download inference.py
                </a>
                <a className="text-primary underline-offset-4 hover:underline" href={deploymentPackageFileUrl(runId, 'Dockerfile')} download="Dockerfile">
                  Download optional Dockerfile
                </a>
              </>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
