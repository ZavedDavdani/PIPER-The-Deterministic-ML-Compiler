import '@testing-library/jest-dom/vitest'
import { afterAll, afterEach, beforeAll } from 'vitest'
import { server } from './mswServer'
import { FakeEventSource } from './fakeEventSource'

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => {
  server.resetHandlers()
  FakeEventSource.reset()
})
afterAll(() => server.close())

// jsdom has no native EventSource — every test gets the deterministic
// fake by default (see fakeEventSource.ts for why).
;(globalThis as { EventSource?: unknown }).EventSource = FakeEventSource

// jsdom doesn't implement the Pointer Events / scroll APIs Radix UI's
// Select relies on (pointer capture, scrollIntoView) — standard
// no-op polyfills for the test environment only, real browsers already
// have these.
if (!Element.prototype.hasPointerCapture) {
  Element.prototype.hasPointerCapture = () => false
}
if (!Element.prototype.setPointerCapture) {
  Element.prototype.setPointerCapture = () => {}
}
if (!Element.prototype.releasePointerCapture) {
  Element.prototype.releasePointerCapture = () => {}
}
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {}
}
