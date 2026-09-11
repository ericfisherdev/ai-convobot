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

  it('regenerateFrom(5) POSTs from_message_id, drops rows >= 5 up front, streams in each rewritten thought, and clears regenerating on round_complete', async () => {
    const staleThought = { ...aThought, id: 2, from_message_id: 5 }
    const keptThought = { ...aThought, id: 1, from_message_id: 1 }
    const rewritten = { ...aThought, id: 2, from_message_id: 5, text: 'rewritten', edited: false }

    const streamChunks = [
      { request_id: 'r1', event: 'thought_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'token', content: '', is_complete: false, speaker_id: 'char', thought: rewritten },
      { request_id: 'r1', event: 'round_complete', content: '', is_complete: true, speaker_id: '' },
    ]

    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse(baseConfig))
      if (url === '/api/thoughts' && (!init || init.method === undefined)) {
        return Promise.resolve(jsonResponse({ thoughts: [keptThought, staleThought] }))
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
      expect(screen.getByTestId('thoughts').textContent).toContain('"id":2')
    })

    fireEvent.click(screen.getByText('regenerate'))

    await waitFor(() => {
      expect(screen.getByTestId('thoughts').textContent).toContain('rewritten')
    })
    expect(screen.getByTestId('thoughts').textContent).toContain('"id":1')

    await waitFor(() => {
      expect(screen.getByTestId('regenerating').textContent).toBe('null')
    })
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

    await vi.waitFor(() => expect(thoughtsCalls).toBe(1))
    expect(toast.error).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(10_000)
    await vi.waitFor(() => expect(thoughtsCalls).toBe(2))
    expect(toast.error).not.toHaveBeenCalled()
    expect(screen.getByTestId('thoughts').textContent).toBe('[]')
  })
})
