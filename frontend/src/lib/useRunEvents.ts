import { useEffect, useRef, useState } from 'react'
import { runEventsUrl } from './api'
import type { TraceEvent } from './types'

export type SseConnectionState = 'idle' | 'connecting' | 'open' | 'closed' | 'error'

const TERMINAL_EVENT_TYPES = new Set(['run_completed', 'run_failed'])

/**
 * Live progress via the real backend's SSE endpoint
 * (GET /runs/{runId}/events) — genuine EventSource, no polling
 * fallback, no simulated/mock events. Accumulates every TraceEvent
 * received, in arrival order, and tracks connection state so the UI
 * can show a real "live"/"reconnecting"/"closed" indicator instead of
 * silently doing nothing on failure.
 */
export function useRunEvents(runId: string | null, enabled: boolean) {
  const [events, setEvents] = useState<TraceEvent[]>([])
  const [connectionState, setConnectionState] = useState<SseConnectionState>('idle')
  const sourceRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!runId || !enabled) {
      setConnectionState('idle')
      return
    }

    setEvents([])
    setConnectionState('connecting')

    const source = new EventSource(runEventsUrl(runId))
    sourceRef.current = source

    source.onopen = () => setConnectionState('open')

    source.onmessage = (message: MessageEvent<string>) => {
      try {
        const event = JSON.parse(message.data) as TraceEvent
        setEvents((prev) => [...prev, event])
        if (TERMINAL_EVENT_TYPES.has(event.event_type)) {
          source.close()
          setConnectionState('closed')
        }
      } catch {
        // A malformed line would be a real backend bug, but must
        // never crash the live view — skip it.
      }
    }

    source.onerror = () => {
      // The browser's EventSource auto-retries by default; if the
      // stream has already closed cleanly (server-side, on terminal
      // status) readyState will be CLOSED and this is expected, not
      // an actual error.
      if (source.readyState === EventSource.CLOSED) {
        setConnectionState('closed')
      } else {
        setConnectionState('error')
      }
    }

    return () => {
      source.close()
      sourceRef.current = null
    }
  }, [runId, enabled])

  return { events, connectionState }
}
