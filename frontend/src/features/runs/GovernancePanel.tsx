import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { getFairness, getGovernance, governanceDocumentUrl } from '@/lib/api'
import type { FairnessReport, GovernanceBundle } from '@/lib/types'

interface GovernancePanelProps {
  runId: string
  runStatus: string
}

const DOCUMENTS = [
  { file: 'model_card.md', label: 'Model Card (Markdown)' },
  { file: 'model_card.json', label: 'Model Card (JSON)' },
  { file: 'data_card.md', label: 'Data Card (Markdown)' },
  { file: 'data_card.json', label: 'Data Card (JSON)' },
  { file: 'fingerprints.json', label: 'Fingerprints' },
  { file: 'feature_importance.json', label: 'Feature importance' },
  { file: 'fairness.json', label: 'Subgroup analysis' },
]

export function GovernancePanel({ runId, runStatus }: GovernancePanelProps) {
  const [bundle, setBundle] = useState<GovernanceBundle | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [columns, setColumns] = useState('')
  const [fairness, setFairness] = useState<FairnessReport | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!runId || (runStatus !== 'completed' && runStatus !== 'failed')) return
    let cancelled = false
    void getGovernance(runId)
      .then((next) => {
        if (!cancelled) {
          setBundle(next)
          setFairness(next.fairness)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Could not load governance.')
      })
    return () => {
      cancelled = true
    }
  }, [runId, runStatus])

  async function onAnalyze() {
    setError(null)
    setBusy(true)
    try {
      const requested = columns.split(',').map((item) => item.trim()).filter(Boolean)
      const next = await getFairness(runId, requested)
      setFairness(next)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Subgroup analysis failed.')
    } finally {
      setBusy(false)
    }
  }

  const top = bundle?.feature_importance.rows.slice(0, 8) ?? []
  const requested = columns.split(',').map((item) => item.trim()).filter(Boolean)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Governance</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 text-sm">
        {error && <p className="text-destructive">{error}</p>}
        {!bundle && !error && <p className="text-muted-foreground">Loading governance…</p>}
        {bundle && (
          <>
            <section className="flex flex-col gap-1">
              <h3 className="font-semibold">Model Card</h3>
              <p>Status: <span className="font-medium">{bundle.model_card.status}</span></p>
              <p>Winning model: <span className="font-medium">{bundle.model_card.winning_algorithm ?? '—'}</span></p>
              <p>Target: <span className="font-medium">{bundle.model_card.target ?? '—'}</span></p>
              {bundle.model_card.evaluation_metrics.map((metric) => (
                <p key={metric.name}>
                  {metric.name}: <span className="font-medium">{metric.value ?? '—'}</span>
                </p>
              ))}
            </section>

            <section className="flex flex-col gap-1">
              <h3 className="font-semibold">Data Card</h3>
              <p>
                Dimensions:{' '}
                <span className="font-medium">
                  {bundle.data_card.rows ?? '—'} × {bundle.data_card.columns ?? '—'}
                </span>
              </p>
              <p>Features: {bundle.data_card.feature_list.slice(0, 8).join(', ') || '—'}</p>
            </section>

            <section className="flex flex-col gap-1">
              <h3 className="font-semibold">Reproducibility fingerprints</h3>
              <p>Algorithm: <span className="font-medium">{bundle.fingerprints.hash_algorithm}</span></p>
              <p className="text-muted-foreground">{bundle.fingerprints.caveat}</p>
              <ul className="list-disc pl-5">
                {bundle.fingerprints.content_hashes.slice(0, 8).map((entry) => (
                  <li key={entry.name}>
                    {entry.name}: {entry.available ? `${entry.digest?.slice(0, 12)}…` : (entry.reason ?? 'unavailable')}
                  </li>
                ))}
              </ul>
            </section>

            <section className="flex flex-col gap-1">
              <h3 className="font-semibold">Feature importance</h3>
              <p className="text-muted-foreground">{bundle.feature_importance.disclaimer}</p>
              {bundle.feature_importance.status === 'NOT_AVAILABLE' && (
                <p>NOT_AVAILABLE{bundle.feature_importance.reason ? `: ${bundle.feature_importance.reason}` : ''}</p>
              )}
              {top.length > 0 && (
                <ul className="list-disc pl-5">
                  {top.map((row) => (
                    <li key={row.transformed_feature}>
                      {row.transformed_feature}: {row.importance.toFixed(4)}
                      {row.direction ? ` (${row.direction})` : ''}
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section className="flex flex-col gap-2">
              <h3 className="font-semibold">Subgroup analysis</h3>
              <p className="text-muted-foreground">
                Statistical subgroup measurements, not a legal compliance decision. Columns are never inferred from names.
              </p>
              <div className="flex gap-2">
                <Input
                  aria-label="Subgroup columns"
                  placeholder="column names, comma-separated"
                  value={columns}
                  onChange={(event) => setColumns(event.target.value)}
                />
                <Button type="button" onClick={() => void onAnalyze()} disabled={busy}>
                  {busy ? 'Analyzing…' : 'Analyze subgroups'}
                </Button>
              </div>
              {fairness && (
                <>
                  <p>Status: <span className="font-medium">{fairness.status}</span></p>
                  {fairness.warnings.map((warning) => (
                    <p key={warning} className="text-destructive">{warning}</p>
                  ))}
                  {fairness.groups.map((group) => (
                    <p key={`${group.column}-${group.group}`}>
                      {group.column}={group.group} n={group.n}
                      {group.sufficient
                        ? ` F1=${group.f1?.toFixed(3) ?? '—'} selection_rate=${group.selection_rate?.toFixed(3) ?? '—'}`
                        : ` ${group.warning ?? 'insufficient sample'}`}
                    </p>
                  ))}
                </>
              )}
            </section>

            <section className="flex flex-col gap-1">
              <h3 className="font-semibold">Limitations</h3>
              <ul className="list-disc pl-5">
                {bundle.limitations.slice(0, 6).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
              <p>
                Artifact status:{' '}
                <span className="font-medium">
                  {String(bundle.artifact_status?.artifact_status ?? 'NOT_GENERATED')}
                </span>
              </p>
            </section>

            <section className="flex flex-col gap-1">
              <h3 className="font-semibold">Downloads</h3>
              {DOCUMENTS.map((doc) => (
                <a
                  key={doc.file}
                  className="text-primary underline-offset-4 hover:underline"
                  href={governanceDocumentUrl(runId, doc.file, requested)}
                  download={doc.file}
                >
                  Download {doc.label}
                </a>
              ))}
            </section>
          </>
        )}
      </CardContent>
    </Card>
  )
}
