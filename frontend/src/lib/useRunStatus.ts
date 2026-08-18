import { useEffect, useRef, useState } from 'react'
import { ApiError, getRunStatus } from './api'
import type { RunStatusResponse } from './types'

const TERMINAL_STATUSES = new Set(['completed', 'failed'])
const POLL_INTERVAL_MS = 1000

/**
 * Polls the real GET /runs/{runId} endpoint — the authoritative
 * source for status/current_node/attempt (independent of whether the
 * SSE connection is healthy). Stops polling once the run reaches a
 * terminal status.
 */
export function useRunStatus(runId: string | null) {
  const [status, setStatus] = useState<RunStatusResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!runId) return
    let cancelled = false

    async function poll() {
      try {
        const result = await getRunStatus(runId!)
        if (cancelled) return
        setStatus(result)
        setError(null)
        if (!TERMINAL_STATUSES.has(result.status)) {
          timerRef.current = setTimeout(poll, POLL_INTERVAL_MS)
        }
      } catch (e) {
        if (cancelled) return
        setError(e instanceof ApiError ? e.message : 'Could not reach the PIPER backend.')
        timerRef.current = setTimeout(poll, POLL_INTERVAL_MS)
      }
    }

    void poll()

    return () => {
      cancelled = true
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [runId])

  return { status, error, isTerminal: status ? TERMINAL_STATUSES.has(status.status) : false }
}
