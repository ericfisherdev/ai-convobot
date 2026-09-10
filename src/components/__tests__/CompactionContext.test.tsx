import { useState } from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { CompactionProvider, useCompaction } from '../context/compactionContext'
import { MessagesProvider } from '../context/messageContext'
import { toast } from 'sonner'

vi.mock('sonner', () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

const jsonResponse = (body: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: () => Promise.resolve(body),
  text: () => Promise.resolve(typeof body === 'string' ? body : JSON.stringify(body)),
})

const emptyMessagePage = { messages: [], total_count: 0, has_more: false }
const emptyListing = { checkpoints: [], pending_draft: null }

const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <MessagesProvider>
    <CompactionProvider>{children}</CompactionProvider>
  </MessagesProvider>
)

// Exposes `useCompaction()` to the DOM so a test can drive `draftReady`/
// `commit` without reaching into the provider's internals, mirroring
// `ParticipantsStrip.test.tsx`'s pattern of rendering a real consumer.
const Probe: React.FC = () => {
  const { draftReady, commit } = useCompaction()
  const [commitResult, setCommitResult] = useState('')
  return (
    <div>
      <button onClick={() => draftReady(3)}>watch draft 3</button>
      <button
        onClick={async () => {
          const result = await commit(3, { items: [] })
          setCommitResult(JSON.stringify(result))
        }}
      >
        commit
      </button>
      <div data-testid="commit-result">{commitResult}</div>
    </div>
  )
}

describe('CompactionContext', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('draftReady polls the draft until it reaches review phase, then stops', async () => {
    let detailPhase: string = 'extracting'
    let detailCalls = 0
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/message')) return Promise.resolve(jsonResponse(emptyMessagePage))
      if (url === '/api/compaction/3') {
        detailCalls++
        return Promise.resolve(jsonResponse({ id: 3, phase: detailPhase }))
      }
      if (url.startsWith('/api/compaction')) return Promise.resolve(jsonResponse(emptyListing))
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    fireEvent.click(screen.getByText('watch draft 3'))

    // `draftReady` itself only calls `refresh()`; the first detail fetch
    // happens on the first poll tick, five seconds later.
    expect(detailCalls).toBe(0)

    detailPhase = 'review'
    await vi.advanceTimersByTimeAsync(5_000)
    await vi.waitFor(() => expect(detailCalls).toBe(1))

    // The poll must stop once `review` is seen: a further tick makes no
    // more calls.
    await vi.advanceTimersByTimeAsync(5_000)
    expect(detailCalls).toBe(1)
  })

  it('draftReady stops polling once a discarded draft reports phase: null, not just review', async () => {
    let detailCalls = 0
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/message')) return Promise.resolve(jsonResponse(emptyMessagePage))
      if (url === '/api/compaction/3') {
        detailCalls++
        // A discarded checkpoint: status is no longer `draft`, so the
        // backend's `phase_of` reports `null`, not `'review'`.
        return Promise.resolve(jsonResponse({ id: 3, status: 'discarded', phase: null }))
      }
      if (url.startsWith('/api/compaction')) return Promise.resolve(jsonResponse(emptyListing))
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    fireEvent.click(screen.getByText('watch draft 3'))

    await vi.advanceTimersByTimeAsync(5_000)
    await vi.waitFor(() => expect(detailCalls).toBe(1))

    await vi.advanceTimersByTimeAsync(5_000)
    expect(detailCalls).toBe(1)
  })

  it('draftReady stops polling once the draft disappears (a failed lookup)', async () => {
    let detailCalls = 0
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/message')) return Promise.resolve(jsonResponse(emptyMessagePage))
      if (url === '/api/compaction/3') {
        detailCalls++
        return Promise.resolve(jsonResponse('not found', 404))
      }
      if (url.startsWith('/api/compaction')) return Promise.resolve(jsonResponse(emptyListing))
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    fireEvent.click(screen.getByText('watch draft 3'))

    await vi.advanceTimersByTimeAsync(5_000)
    await vi.waitFor(() => expect(detailCalls).toBe(1))

    await vi.advanceTimersByTimeAsync(5_000)
    expect(detailCalls).toBe(1)
  })

  it('commit passes the 422 body back to the caller instead of toasting', async () => {
    const rejected = [{ item_id: 12, reason: 'quote is not verbatim in any cited message' }]
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/message')) return Promise.resolve(jsonResponse(emptyMessagePage))
      if (url === '/api/compaction/3/commit') return Promise.resolve(jsonResponse(rejected, 422))
      if (url.startsWith('/api/compaction')) return Promise.resolve(jsonResponse(emptyListing))
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    fireEvent.click(screen.getByText('commit'))

    await vi.waitFor(() => {
      expect(screen.getByTestId('commit-result').textContent).toContain('quote is not verbatim')
    })
    const parsed = JSON.parse(screen.getByTestId('commit-result').textContent ?? '')
    expect(parsed).toEqual({ ok: false, status: 422, body: rejected })
    expect(toast.error).not.toHaveBeenCalled()
  })

  it('commit toasts and reports the status for a non-422 failure', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/message')) return Promise.resolve(jsonResponse(emptyMessagePage))
      if (url === '/api/compaction/3/commit') return Promise.resolve(jsonResponse('draft not found', 404))
      if (url.startsWith('/api/compaction')) return Promise.resolve(jsonResponse(emptyListing))
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    fireEvent.click(screen.getByText('commit'))

    await vi.waitFor(() => {
      expect(screen.getByTestId('commit-result').textContent).toContain('"status":404')
    })
    expect(toast.error).toHaveBeenCalled()
  })
})
