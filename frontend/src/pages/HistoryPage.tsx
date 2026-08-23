import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { listRuns } from '@/lib/api'
import type { RunListItem } from '@/lib/types'

export function HistoryPage() {
  const [runs, setRuns] = useState<RunListItem[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    listRuns()
      .then((body) => {
        if (!cancelled) setRuns(body.runs)
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-8">
      <header className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <Link to="/" className="text-muted-foreground hover:text-foreground text-sm">
            ← Home
          </Link>
          <h1 className="text-2xl font-semibold tracking-tight">Run history</h1>
          <p className="text-muted-foreground text-sm">
            Reopen a stored run. Replay uses persisted events and state only — it does not call the LLM.
          </p>
        </div>
        <ThemeToggle />
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Local runs</CardTitle>
          <CardDescription>SQLite-backed history when PIPER is started with the default store.</CardDescription>
        </CardHeader>
        <CardContent>
          {error && <p className="text-destructive text-sm">{error}</p>}
          {!error && runs.length === 0 && (
            <p className="text-muted-foreground text-sm">No runs stored yet.</p>
          )}
          <ul className="flex flex-col gap-2">
            {runs.map((run) => (
              <li key={run.run_id}>
                <Link
                  to={`/runs/${run.run_id}`}
                  className="hover:bg-accent flex flex-col rounded-md border px-3 py-2 text-sm"
                >
                  <span className="font-mono">{run.run_id}</span>
                  <span className="text-muted-foreground">
                    {run.status} · {run.target_column} · attempt {run.attempt}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
