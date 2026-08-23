import { useEffect, useRef, useState } from 'react'
import { ApiError, getDecisionTrace } from './api'
import type { DecisionTrace } from './types'

const POLL_INTERVAL_MS = 1000

/** Polls GET /runs/{id}/decision-trace (available mid-run, like /timeline). */
export function useDecisionTrace(runId: string | null, isTerminal: boolean) {
  const [trace, setTrace] = useState<DecisionTrace | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!runId) return
    let cancelled = false

    async function poll() {
      try {
        const result = await getDecisionTrace(runId!)
        if (cancelled) return
        setTrace(result)
        setError(null)
        if (!isTerminal) {
          timerRef.current = setTimeout(poll, POLL_INTERVAL_MS)
        }
      } catch (e) {
        if (cancelled) return
        setError(e instanceof ApiError ? e.message : 'Could not load the decision trace.')
        if (!isTerminal) {
          timerRef.current = setTimeout(poll, POLL_INTERVAL_MS)
        }
      }
    }

    void poll()
    return () => {
      cancelled = true
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [runId, isTerminal])

  return { trace, error }
}
