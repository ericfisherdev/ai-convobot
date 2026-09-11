import { useState } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { RunningThoughtsProvider, useRunningThoughts } from '../context/runningThoughtsContext'
import { ConfigProvider } from '../context/configContext'
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

const baseConfig = { multiplayer_mode: 'solo' }

const aThought = {
  id: 1,
  companion_id: 1,
  speaker_id: 'char',
  from_message_id: 1,
  through_message_id: 2,
  text: 'the user seems pleased',
  edited: false,
  created_at: '2026-01-01T00:00:00Z',
}

const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <ConfigProvider>
    <RunningThoughtsProvider>{children}</RunningThoughtsProvider>
  </ConfigProvider>
)

// Exposes `useRunningThoughts()` to the DOM so a test can drive it without
// reaching into the provider's internals, mirroring
// `CompactionContext.test.tsx`'s `Probe` pattern.
const Probe: React.FC = () => {
  const { thoughts, regenerating, editThought, deleteThought, regenerateFrom } = useRunningThoughts()
  const [editResult, setEditResult] = useState<string>('')

  return (
    <div>
      <div data-testid="thoughts">{JSON.stringify(thoughts)}</div>
      <div data-testid="regenerating">{JSON.stringify(regenerating)}</div>
      <button
        onClick={async () => {
          const ok = await editThought(1, 'a rewritten thought')
          setEditResult(String(ok))
        }}
      >
        edit
      </button>
      <div data-testid="edit-result">{editResult}</div>
      <button onClick={() => deleteThought(1)}>delete</button>
      <button onClick={() => regenerateFrom(5)}>regenerate</button>
    </div>
  )
}

// Encodes chunks the way `/api/thoughts/regenerate` does, one SSE record
// each, mirroring `ChatWindow.test.tsx`'s `streamResponse` helper.
const streamResponse = (chunks: unknown[]) => {
  const encoder = new TextEncoder()
  return {
    ok: true,
    status: 200,
    body: new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
        }
        controller.close()
      },
    }),
  }
}

describe('RunningThoughtsContext', () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('mounts and fetches /api/thoughts', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse(baseConfig))
      if (url === '/api/thoughts') return Promise.resolve(jsonResponse({ thoughts: [aThought] }))
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    await waitFor(() => {
      expect(screen.getByTestId('thoughts').textContent).toContain('the user seems pleased')
    })
    expect(fetchMock.mock.calls.some(([input]) => String(input) === '/api/thoughts')).toBe(true)
  })

  it('editThought PATCHes { text } and the row carries the response', async () => {
    const updated = { ...aThought, text: 'a rewritten thought', edited: true }
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse(baseConfig))
      if (url === '/api/thoughts') return Promise.resolve(jsonResponse({ thoughts: [aThought] }))
      if (url === '/api/thoughts/1' && init?.method === 'PATCH') {
        expect(JSON.parse(init.body as string)).toEqual({ text: 'a rewritten thought' })
        return Promise.resolve(jsonResponse(updated))
      }
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    await waitFor(() => {
      expect(screen.getByTestId('thoughts').textContent).toContain('the user seems pleased')
    })

    fireEvent.click(screen.getByText('edit'))

    await waitFor(() => {
      expect(screen.getByTestId('edit-result').textContent).toBe('true')
    })
    expect(screen.getByTestId('thoughts').textContent).toContain('"edited":true')
  })

  it('a failed PATCH toasts the 422 reason and returns false', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse(baseConfig))
      if (url === '/api/thoughts') return Promise.resolve(jsonResponse({ thoughts: [aThought] }))
      if (url === '/api/thoughts/1' && init?.method === 'PATCH') {
        return Promise.resolve(jsonResponse({ reason: 'text must not be empty' }, 422))
      }
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    await waitFor(() => {
      expect(screen.getByTestId('thoughts').textContent).toContain('the user seems pleased')
    })

    fireEvent.click(screen.getByText('edit'))

    await waitFor(() => {
      expect(screen.getByTestId('edit-result').textContent).toBe('false')
    })
    expect(toast.error).toHaveBeenCalledWith('text must not be empty')
  })

  it('a 409 PATCH toasts the backend reason and returns false', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse(baseConfig))
      if (url === '/api/thoughts') return Promise.resolve(jsonResponse({ thoughts: [aThought] }))
      if (url === '/api/thoughts/1' && init?.method === 'PATCH') {
        return Promise.resolve({
          ok: false,
          status: 409,
          text: () => Promise.resolve('A reply is still being generated; wait for it to finish before editing a thought'),
        })
      }
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    await waitFor(() => {
      expect(screen.getByTestId('thoughts').textContent).toContain('the user seems pleased')
    })

    fireEvent.click(screen.getByText('edit'))

    await waitFor(() => {
      expect(screen.getByTestId('edit-result').textContent).toBe('false')
    })
    expect(toast.error).toHaveBeenCalledWith(
      'A reply is still being generated; wait for it to finish before editing a thought'
    )
    // The row is unchanged, not dropped or mutated.
    expect(screen.getByTestId('thoughts').textContent).toContain('the user seems pleased')
  })

  it('a 409 DELETE toasts the backend reason and leaves the row in place', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse(baseConfig))
      if (url === '/api/thoughts') return Promise.resolve(jsonResponse({ thoughts: [aThought] }))
      if (url === '/api/thoughts/1' && init?.method === 'DELETE') {
        return Promise.resolve({
          ok: false,
          status: 409,
          text: () => Promise.resolve('A reply is still being generated; wait for it to finish before deleting a thought'),
        })
      }
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    await waitFor(() => {
      expect(screen.getByTestId('thoughts').textContent).toContain('the user seems pleased')
    })

    fireEvent.click(screen.getByText('delete'))

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(
        'A reply is still being generated; wait for it to finish before deleting a thought'
      )
    })
    expect(screen.getByTestId('thoughts').textContent).toContain('the user seems pleased')
  })

  it('deleteThought DELETEs and removes the row', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse(baseConfig))
      if (url === '/api/thoughts') return Promise.resolve(jsonResponse({ thoughts: [aThought] }))
      if (url === '/api/thoughts/1' && init?.method === 'DELETE') {
        return Promise.resolve(jsonResponse('Thought deleted at id 1!'))
      }
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    await waitFor(() => {
      expect(screen.getByTestId('thoughts').textContent).toContain('the user seems pleased')
    })

    fireEvent.click(screen.getByText('delete'))

    await waitFor(() => {
      expect(screen.getByTestId('thoughts').textContent).toBe('[]')
    })
  })

  it('regenerateFrom(5) POSTs from_message_id, streams in the rewritten thought, and resyncs from the server so a non-owned speaker\'s re-inserted row reappears', async () => {
    // `through_message_id`, not `from_message_id`, is the boundary the
    // backend's `delete_from` (and so the optimistic drop) uses: kept has
    // through_message_id 2 (< 5, survives), stale and bot1 both have
    // through_message_id >= 5 (deleted).
    const keptThought = { ...aThought, id: 1, from_message_id: 1, through_message_id: 2 }
    const staleThought = { ...aThought, id: 2, from_message_id: 5, through_message_id: 5 }
    // A non-owned speaker's row in the same range: the backend re-inserts
    // this unchanged under a new id, but the regenerate stream never emits
    // a chunk for it -- only `refresh()` after the stream ends picks it up.
    const bot1Thought = {
      ...aThought,
      id: 3,
      speaker_id: 'bot1',
      from_message_id: 5,
      through_message_id: 6,
      text: 'bot1 is glad too',
    }
    const rewritten = { ...aThought, id: 2, from_message_id: 5, through_message_id: 5, text: 'rewritten', edited: false }
    const reinsertedBot1 = { ...bot1Thought, id: 10 }

    const streamChunks = [
      { request_id: 'r1', event: 'thought_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'token', content: '', is_complete: false, speaker_id: 'char', thought: rewritten },
      { request_id: 'r1', event: 'round_complete', content: '', is_complete: true, speaker_id: '' },
    ]

    let thoughtsGetCalls = 0
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse(baseConfig))
      if (url === '/api/thoughts' && (!init || init.method === undefined)) {
        thoughtsGetCalls++
        // First call is the mount fetch; the second is `regenerateFrom`'s
        // post-stream resync, which the backend would answer with the
        // rewritten char row plus bot1's row re-inserted under a new id.
        if (thoughtsGetCalls === 1) {
          return Promise.resolve(jsonResponse({ thoughts: [keptThought, staleThought, bot1Thought] }))
        }
        return Promise.resolve(jsonResponse({ thoughts: [keptThought, rewritten, reinsertedBot1] }))
      }
      if (url === '/api/thoughts/regenerate') {
        expect(JSON.parse(init?.body as string)).toEqual({ from_message_id: 5 })
        return Promise.resolve(streamResponse(streamChunks))
      }
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    await waitFor(() => {
      expect(screen.getByTestId('thoughts').textContent).toContain('bot1 is glad too')
    })

    fireEvent.click(screen.getByText('regenerate'))

    await waitFor(() => {
      expect(screen.getByTestId('regenerating').textContent).toBe('null')
    })
    expect(screen.getByTestId('thoughts').textContent).toContain('rewritten')
    expect(screen.getByTestId('thoughts').textContent).toContain('bot1 is glad too')
    expect(screen.getByTestId('thoughts').textContent).toContain('"id":1')
    expect(thoughtsGetCalls).toBe(2)
  })

  it('a 409 on regenerate toasts and leaves thoughts untouched', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse(baseConfig))
      if (url === '/api/thoughts' && (!init || init.method === undefined)) {
        return Promise.resolve(jsonResponse({ thoughts: [aThought] }))
      }
      if (url === '/api/thoughts/regenerate') {
        return Promise.resolve({ ok: false, status: 409, body: null })
      }
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    await waitFor(() => {
      expect(screen.getByTestId('thoughts').textContent).toContain('the user seems pleased')
    })

    fireEvent.click(screen.getByText('regenerate'))

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('Still replying'))
    })
    expect(screen.getByTestId('thoughts').textContent).toContain('the user seems pleased')
  })

  it('on a joiner instance, a 409 on /api/thoughts produces no toast and the route is refetched after 10s', async () => {
    vi.useFakeTimers()
    let thoughtsCalls = 0
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse({ multiplayer_mode: 'joiner' }))
      if (url === '/api/thoughts') {
        thoughtsCalls++
        // First load succeeds so the poll tick's 409 has a visible
        // non-empty -> empty transition to wait on -- otherwise `'[]'` is
        // indistinguishable from the state never having been touched at
        // all, and the "no toast" assertions could pass before the 409
        // branch has actually run.
        if (thoughtsCalls === 1) return Promise.resolve(jsonResponse({ thoughts: [aThought] }))
        return Promise.resolve({ ok: false, status: 409, body: null })
      }
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <Probe />
      </MockProviders>
    )

    await vi.waitFor(() =>
      expect(screen.getByTestId('thoughts').textContent).toContain('the user seems pleased')
    )

    await vi.advanceTimersByTimeAsync(10_000)
    await vi.waitFor(() => expect(thoughtsCalls).toBe(2))
    await vi.waitFor(() => expect(screen.getByTestId('thoughts').textContent).toBe('[]'))
    expect(toast.error).not.toHaveBeenCalled()
  })
})
