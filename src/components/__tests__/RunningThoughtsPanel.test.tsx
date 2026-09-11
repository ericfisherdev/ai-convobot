import { useEffect } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import { RunningThoughtsPanel } from '../thoughts/RunningThoughtsPanel'
import { RunningThoughtsProvider, useRunningThoughts } from '../context/runningThoughtsContext'
import { ConfigProvider } from '../context/configContext'
import { UserDataProvider } from '../context/userContext'
import { CompanionDataProvider } from '../context/companionContext'
import { ParticipantsProvider } from '../context/participantsContext'

vi.mock('sonner', () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

const jsonResponse = (body: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: () => Promise.resolve(body),
  text: () => Promise.resolve(typeof body === 'string' ? body : JSON.stringify(body)),
})

const hostConfig = { multiplayer_mode: 'host', running_thoughts_enabled: true }
const companion = { name: 'Aria', avatar_path: 'aria.png' }
const user = { name: 'Alex' }
const bot1 = { id: 'bot1', display_name: 'Ada', kind: 'RemoteBot', avatar_url: null, connected: true }

const thoughtChar = {
  id: 1,
  companion_id: 1,
  speaker_id: 'char',
  from_message_id: 1,
  through_message_id: 2,
  text: 'the user seems pleased',
  edited: false,
  created_at: '2026-01-01T00:00:00Z',
}
const thoughtBot1 = {
  id: 2,
  companion_id: 1,
  speaker_id: 'bot1',
  from_message_id: 3,
  through_message_id: 4,
  text: 'the conversation is going well',
  edited: true,
  created_at: '2026-01-01T00:01:00Z',
}

const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <ConfigProvider>
    <UserDataProvider>
      <CompanionDataProvider>
        <ParticipantsProvider>
          <RunningThoughtsProvider>{children}</RunningThoughtsProvider>
        </ParticipantsProvider>
      </CompanionDataProvider>
    </UserDataProvider>
  </ConfigProvider>
)

// Drives `pendingSpeakerId` directly through the real provider (no stream
// mock needed), mirroring `CompactionContext.test.tsx`'s `Probe` pattern.
const PendingDriver: React.FC<{ speakerId: string }> = ({ speakerId }) => {
  const { beginPending } = useRunningThoughts()
  useEffect(() => {
    beginPending(speakerId)
  }, [beginPending, speakerId])
  return null
}

function mockFetch(overrides: {
  thoughts?: unknown[]
  onRegenerate?: (init?: RequestInit) => Promise<unknown>
} = {}) {
  return vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString()
    if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse(hostConfig))
    if (url.startsWith('/api/companion')) return Promise.resolve(jsonResponse(companion))
    if (url.startsWith('/api/user')) return Promise.resolve(jsonResponse(user))
    if (url.startsWith('/api/multiplayer/participants')) return Promise.resolve(jsonResponse([bot1]))
    if (url === '/api/thoughts') return Promise.resolve(jsonResponse({ thoughts: overrides.thoughts ?? [] }))
    if (url.startsWith('/api/thoughts/') && init?.method === 'PATCH') {
      const updated = { ...thoughtChar, text: JSON.parse(init.body as string).text, edited: true }
      return Promise.resolve(jsonResponse(updated))
    }
    if (url.startsWith('/api/thoughts/') && init?.method === 'DELETE') {
      return Promise.resolve(jsonResponse('Thought deleted!'))
    }
    if (url === '/api/thoughts/regenerate' && overrides.onRegenerate) {
      return overrides.onRegenerate(init)
    }
    return Promise.resolve(jsonResponse({}))
  })
}

describe('RunningThoughtsPanel', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('renders two thoughts in id order attributed to char and a bot1 participant', async () => {
    const fetchMock = mockFetch({ thoughts: [thoughtChar, thoughtBot1] })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <RunningThoughtsPanel />
      </MockProviders>
    )

    await screen.findByText('the user seems pleased')
    const names = screen.getAllByText(/^(Aria|Ada)$/)
    expect(names.map((n) => n.textContent)).toEqual(['Aria', 'Ada'])
  })

  it('shows the edited badge only on the edited row', async () => {
    const fetchMock = mockFetch({ thoughts: [thoughtChar, thoughtBot1] })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <RunningThoughtsPanel />
      </MockProviders>
    )

    await screen.findByText('the user seems pleased')
    expect(screen.getAllByText('Your wording')).toHaveLength(1)
  })

  it('editing a thought sends the PATCH body from the last call', async () => {
    const fetchMock = mockFetch({ thoughts: [thoughtChar] })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <RunningThoughtsPanel />
      </MockProviders>
    )

    await screen.findByText('the user seems pleased')
    fireEvent.click(screen.getByLabelText('Edit thought'))

    const textarea = screen.getByRole('textbox')
    fireEvent.change(textarea, { target: { value: 'a rewritten thought' } })
    fireEvent.click(screen.getByLabelText('Save thought'))

    await waitFor(() => {
      const last = fetchMock.mock.calls[fetchMock.mock.calls.length - 1]
      expect(last[0]).toBe('/api/thoughts/1')
      expect(JSON.parse(last[1].body)).toEqual({ text: 'a rewritten thought' })
    })
  })

  it('deleting a thought sends the DELETE request', async () => {
    const fetchMock = mockFetch({ thoughts: [thoughtChar] })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <RunningThoughtsPanel />
      </MockProviders>
    )

    await screen.findByText('the user seems pleased')
    fireEvent.click(screen.getByLabelText('Delete thought'))

    await waitFor(() => {
      const last = fetchMock.mock.calls[fetchMock.mock.calls.length - 1]
      expect(last[0]).toBe('/api/thoughts/1')
      expect(last[1].method).toBe('DELETE')
    })
  })

  it('renders a pending speaker as "is thinking..." with no action buttons besides the header toggle', async () => {
    const fetchMock = mockFetch({ thoughts: [] })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <PendingDriver speakerId="bot1" />
        <RunningThoughtsPanel />
      </MockProviders>
    )

    await screen.findByText('is thinking...')
    expect(screen.queryAllByRole('button')).toHaveLength(1)
    expect(screen.getByLabelText('Collapse running thoughts')).toBeInTheDocument()
  })

  it('the collapse toggle hides the thought list', async () => {
    const fetchMock = mockFetch({ thoughts: [thoughtChar] })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <RunningThoughtsPanel />
      </MockProviders>
    )

    await screen.findByText('the user seems pleased')
    fireEvent.click(screen.getByLabelText('Collapse running thoughts'))

    expect(screen.queryByText('the user seems pleased')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Expand running thoughts')).toBeInTheDocument()
  })

  it('confirming "Regenerate from here" on the first of two rows POSTs that row\'s from_message_id, and hides row actions while regenerating', async () => {
    let releaseStream: (() => void) | undefined
    const streamGate = new Promise<void>((resolve) => {
      releaseStream = resolve
    })
    let regenerateBody: unknown = null

    const fetchMock = mockFetch({
      thoughts: [thoughtChar, thoughtBot1],
      onRegenerate: async (init) => {
        regenerateBody = JSON.parse(init?.body as string)
        return {
          ok: true,
          status: 200,
          body: new ReadableStream<Uint8Array>({
            async start(controller) {
              await streamGate
              controller.close()
            },
          }),
        }
      },
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <RunningThoughtsPanel />
      </MockProviders>
    )

    await screen.findByText('the user seems pleased')
    const firstRow = screen.getByText('the user seems pleased').closest('.message-container') as HTMLElement
    fireEvent.click(within(firstRow).getByLabelText('Regenerate from here'))

    await screen.findByText('Rewrite this and 1 later thought?')
    fireEvent.click(screen.getByLabelText('Confirm regenerate'))

    await waitFor(() => {
      expect(regenerateBody).toEqual({ from_message_id: thoughtChar.from_message_id })
    })

    await screen.findByRole('status')
    expect(screen.getByRole('status')).toHaveTextContent('Rewriting thoughts')
    expect(screen.queryByLabelText('Edit thought')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Delete thought')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Regenerate from here')).not.toBeInTheDocument()

    releaseStream?.()
  })
})
