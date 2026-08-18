/**
 * jsdom (the test DOM environment) has no native EventSource
 * implementation, and MSW's Node-side interception has nothing to
 * intercept if `new EventSource(...)` throws before any request is
 * even made. This fake stands in for the browser's real EventSource
 * ONLY in tests, so useRunEvents/LiveEventFeed can be exercised
 * against realistic, test-controlled event sequences deterministically.
 * It never runs in the actual app — app/main dev/prod code always
 * uses the browser's real EventSource. The genuine, real-network SSE
 * path is verified separately via a live backend + real browser (see
 * the manual verification step, not this test double).
 */
export class FakeEventSource {
  static instances: FakeEventSource[] = []
  static CONNECTING = 0
  static OPEN = 1
  static CLOSED = 2

  url: string
  readyState = FakeEventSource.CONNECTING
  onopen: (() => void) | null = null
  onmessage: ((event: MessageEvent<string>) => void) | null = null
  onerror: (() => void) | null = null

  constructor(url: string) {
    this.url = url
    FakeEventSource.instances.push(this)
  }

  /** Test helper: simulate the connection opening. */
  simulateOpen() {
    this.readyState = FakeEventSource.OPEN
    this.onopen?.()
  }

  /** Test helper: simulate one SSE `data: ...` line arriving. */
  simulateMessage(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent<string>)
  }

  /** Test helper: simulate a transport error (not a clean server-side close). */
  simulateError() {
    this.onerror?.()
  }

  close() {
    this.readyState = FakeEventSource.CLOSED
  }

  static reset() {
    FakeEventSource.instances = []
  }

  static latest(): FakeEventSource | undefined {
    return FakeEventSource.instances[FakeEventSource.instances.length - 1]
  }
}
