import { useEffect, useState } from 'react'
import { ApiError, getRunResult } from './api'
import type { RunResultResponse } from './types'

/**
 * Fetches GET /runs/{runId}/result once (when `enabled` — the caller
 * should pass true only once the run's status is already terminal;
 * see useRunStatus). record.final_state is populated by the backend
 * in the SAME store update that flips status to terminal, so there's
 * no race to retry around here.
 */
export function useRunResult(runId: string | null, enabled: boolean) {
  const [result, setResult] = useState<RunResultResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!runId || !enabled) return
    let cancelled = false
    getRunResult(runId)
      .then((r) => {
        if (!cancelled) setResult(r)
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : 'Could not load the run result.')
      })
    return () => {
      cancelled = true
    }
  }, [runId, enabled])

  return { result, error }
}
