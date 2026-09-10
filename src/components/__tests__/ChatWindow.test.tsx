import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ChatWindow from '../ChatWindow'
import { MessagesProvider, useMessages } from '../context/messageContext'
import { CompactionProvider } from '../context/compactionContext'
import { UserDataProvider } from '../context/userContext'
import { CompanionDataProvider } from '../context/companionContext'
import { ConfigProvider } from '../context/configContext'
import { ParticipantsProvider } from '../context/participantsContext'
import { AttitudeProvider } from '../context/attitudeContext'
import { SessionProvider } from '../context/sessionContext'
import { ThemeProvider } from '../theme-provider'
import { toast } from 'sonner'

vi.mock('sonner', () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

// The real message list renders replies through a `lazy()` react-markdown
// import, which suspends inside the discrete click event and blanks the tree.
// None of these tests assert on message rendering.
vi.mock('../message/MessageScroll', () => ({
  MessageScroll: () => <div data-testid="message-scroll" />,
}))

const session = {
  id: 's1',
  companion_id: 1,
  user_id: 1,
  created_at: '2024-01-15 09:00',
  last_activity: '2024-01-15 10:00',
  is_active: true,
}

const attitude = {
  id: 1,
  companion_id: 1,
  target_id: 1,
  target_type: 'user',
  attraction: 0,
  trust: 0,
  fear: 0,
  anger: 0,
  joy: 0,
  sorrow: 0,
  disgust: 0,
  surprise: 0,
  curiosity: 0,
  respect: 0,
  suspicion: 0,
  gratitude: 0,
  jealousy: 0,
  empathy: 0,
  lust: 0,
  love: 0,
  anxiety: 0,
  butterflies: 0,
  submissiveness: 0,
  dominance: 0,
  relationship_score: 0,
  last_updated: '2024-01-15 10:00',
  created_at: '2024-01-15 09:00',
}

const jsonResponse = (body: unknown) => ({ ok: true, status: 200, json: () => Promise.resolve(body) })

// Encodes chunks the way `/api/prompt/stream` does: one SSE record each.
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

// Renders every message's speaker and content, so a test can assert on the
// list `MessagesProvider` built without the real (mocked) `MessageScroll`.
const MessageSpy: React.FC = () => {
  const { messages } = useMessages()
  return (
    <div data-testid="message-spy">
      {messages.map(message => (
        <span key={message.id} data-testid={`message-${message.speaker_id}`}>
          {message.content}
        </span>
      ))}
    </div>
  )
}

// Mock the contexts with minimal implementations
const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
    <MessagesProvider>
      <UserDataProvider>
        <CompanionDataProvider>
          <ConfigProvider>
            <ParticipantsProvider>
              <AttitudeProvider>
                <SessionProvider>
                  <CompactionProvider>
                    {children}
                  </CompactionProvider>
                </SessionProvider>
              </AttitudeProvider>
            </ParticipantsProvider>
          </ConfigProvider>
        </CompanionDataProvider>
      </UserDataProvider>
    </MessagesProvider>
  </ThemeProvider>
)

describe('ChatWindow Component', () => {
  beforeEach(() => {
    localStorage.clear()
    // Mock fetch for API calls
    global.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) {
        return Promise.resolve(jsonResponse({ multiplayer_mode: 'solo' }))
      }
      if (url.startsWith('/api/session')) {
        return Promise.resolve(jsonResponse(session))
      }
      if (url.startsWith('/api/attitude/summary/')) {
        return Promise.resolve(jsonResponse({ attitude, summary: 'neutral' }))
      }
      if (url.startsWith('/api/compaction')) {
        return Promise.resolve(jsonResponse({ checkpoints: [], pending_draft: null }))
      }
      return Promise.resolve(jsonResponse([]))
    }) as unknown as typeof fetch
  })

  it('renders chat window', () => {
    render(
      <MockProviders>
        <ChatWindow />
      </MockProviders>
    )

    // Check if main chat elements are present
    expect(screen.getByRole('main')).toBeInTheDocument()
  })

  it('displays message input area', () => {
    render(
      <MockProviders>
        <ChatWindow />
      </MockProviders>
    )

    const textarea = screen.getByRole('textbox')
    expect(textarea).toBeInTheDocument()
  })

  it('shows send button', () => {
    render(
      <MockProviders>
        <ChatWindow />
      </MockProviders>
    )

    const sendButton = screen.getByRole('button', { name: /send/i })
    expect(sendButton).toBeInTheDocument()
  })

  it('handles message input', async () => {
    const user = userEvent.setup()

    render(
      <MockProviders>
        <ChatWindow />
      </MockProviders>
    )

    const textarea = screen.getByRole('textbox')
    await user.type(textarea, 'Hello, this is a test message')
    expect(textarea).toHaveValue('Hello, this is a test message')
  })

  it('applies the stream attitude chunk without refetching the summary', async () => {
    const user = userEvent.setup()
    const streamChunks = [
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'token', content: 'hi', is_complete: false, token_count: 1, speaker_id: 'char' },
      {
        request_id: 'r1',
        event: 'token',
        content: '',
        is_complete: false,
        token_count: 1,
        speaker_id: '',
        attitude: {
          attitude: { ...attitude, trust: 7 },
          summary: 'warmer',
          deltas: [{ dimension: 'trust', delta: 3 }],
        },
      },
      { request_id: 'r1', event: 'reply_complete', content: 'hi', is_complete: false, speaker_id: 'char', message_id: 2 },
      { request_id: 'r1', event: 'round_complete', content: '', is_complete: true, speaker_id: '' },
    ]

    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/prompt/stream')) {
        return Promise.resolve(streamResponse(streamChunks))
      }
      if (url.startsWith('/api/session')) {
        return Promise.resolve(jsonResponse(session))
      }
      if (url.startsWith('/api/attitude/summary/')) {
        return Promise.resolve(jsonResponse({ attitude, summary: 'neutral' }))
      }
      if (url.startsWith('/api/compaction')) {
        return Promise.resolve(jsonResponse({ checkpoints: [], pending_draft: null }))
      }
      return Promise.resolve(jsonResponse([]))
    })
    global.fetch = fetchMock as unknown as typeof fetch

    render(
      <MockProviders>
        <ChatWindow />
      </MockProviders>
    )

    // The bar's own mount fetch has to settle first, so the count below only
    // covers refetches the send would have caused.
    await screen.findByTestId('attitude-summary-bar')
    const summaryFetchesBeforeSend = fetchMock.mock.calls.filter(([input]) =>
      String(input).startsWith('/api/attitude/summary/')
    ).length

    const textarea = screen.getByRole('textbox')
    await user.type(textarea, 'hello')
    await user.click(screen.getByRole('button', { name: /^send message$/i }))

    await waitFor(() => {
      expect(screen.getByTestId('attitude-delta-trust')).toHaveTextContent('+3')
    })
    // The stream carried the attitude, so the `attitude-update` fallback never
    // fired.
    expect(
      fetchMock.mock.calls.filter(([input]) =>
        String(input).startsWith('/api/attitude/summary/')
      ).length
    ).toBe(summaryFetchesBeforeSend)
  })

  it('disables send controls while a reply streams and re-enables afterwards', async () => {
    const user = userEvent.setup()
    let releaseStream: (() => void) | undefined
    const streamGate = new Promise<void>(resolve => { releaseStream = resolve })

    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/prompt/stream')) {
        const encoder = new TextEncoder()
        return Promise.resolve({
          ok: true,
          status: 200,
          body: new ReadableStream<Uint8Array>({
            async start(controller) {
              await streamGate
              const chunks = [
                { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
                { request_id: 'r1', event: 'reply_complete', content: 'hi', is_complete: false, speaker_id: 'char', message_id: 2 },
                { request_id: 'r1', event: 'round_complete', content: '', is_complete: true, speaker_id: '' },
              ]
              for (const chunk of chunks) {
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
              }
              controller.close()
            },
          }),
        })
      }
      if (url.startsWith('/api/session')) {
        return Promise.resolve(jsonResponse(session))
      }
      if (url.startsWith('/api/attitude/summary/')) {
        return Promise.resolve(jsonResponse({ attitude, summary: 'neutral' }))
      }
      if (url.startsWith('/api/compaction')) {
        return Promise.resolve(jsonResponse({ checkpoints: [], pending_draft: null }))
      }
      return Promise.resolve(jsonResponse([]))
    })
    global.fetch = fetchMock as unknown as typeof fetch

    render(
      <MockProviders>
        <ChatWindow />
      </MockProviders>
    )

    const textarea = screen.getByRole('textbox')
    const sendButton = screen.getByRole('button', { name: /^send message$/i })
    await user.type(textarea, 'hello')
    await user.click(sendButton)

    await waitFor(() => {
      expect(textarea).toBeDisabled()
      expect(sendButton).toBeDisabled()
    })

    const streamCallsWhileSending = fetchMock.mock.calls.filter(([input]) =>
      String(input).startsWith('/api/prompt/stream')
    ).length

    // Disabled controls should not let a second send slip through mid-stream.
    await user.click(sendButton)
    await user.keyboard('{Enter}')
    expect(
      fetchMock.mock.calls.filter(([input]) => String(input).startsWith('/api/prompt/stream')).length
    ).toBe(streamCallsWhileSending)

    releaseStream?.()

    // The textarea empties on send, so re-enabling is checked by typing again
    // and confirming the button responds rather than by its disabled state
    // alone (which an empty textarea would also produce).
    await waitFor(() => {
      expect(textarea).not.toBeDisabled()
      expect(textarea).toHaveFocus()
    })
    await user.type(textarea, 'again')
    expect(sendButton).not.toBeDisabled()
  })

  it('streams speaker-tagged bubbles for a two-speaker round and gates re-enabling on round_complete', async () => {
    const user = userEvent.setup()
    let releaseRoundComplete: (() => void) | undefined
    const roundCompleteGate = new Promise<void>(resolve => { releaseRoundComplete = resolve })

    const repliesChunks = [
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'reply_complete', content: 'hi from char', is_complete: false, speaker_id: 'char', message_id: 10 },
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'bot1' },
      { request_id: 'r1', event: 'reply_complete', content: 'hi from bot1', is_complete: false, speaker_id: 'bot1', message_id: 11 },
    ]
    const roundCompleteChunk = { request_id: 'r1', event: 'round_complete', content: '', is_complete: true, speaker_id: '' }

    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/prompt/stream')) {
        const encoder = new TextEncoder()
        return Promise.resolve({
          ok: true,
          status: 200,
          body: new ReadableStream<Uint8Array>({
            async start(controller) {
              for (const chunk of repliesChunks) {
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
              }
              await roundCompleteGate
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(roundCompleteChunk)}\n\n`))
              controller.close()
            },
          }),
        })
      }
      if (url.startsWith('/api/session')) {
        return Promise.resolve(jsonResponse(session))
      }
      if (url.startsWith('/api/attitude/summary/')) {
        return Promise.resolve(jsonResponse({ attitude, summary: 'neutral' }))
      }
      if (url.startsWith('/api/compaction')) {
        return Promise.resolve(jsonResponse({ checkpoints: [], pending_draft: null }))
      }
      return Promise.resolve(jsonResponse([]))
    })
    global.fetch = fetchMock as unknown as typeof fetch

    render(
      <MockProviders>
        <ChatWindow />
        <MessageSpy />
      </MockProviders>
    )

    const textarea = screen.getByRole('textbox')
    await user.type(textarea, 'hello')
    await user.click(screen.getByRole('button', { name: /^send message$/i }))

    await waitFor(() => {
      expect(screen.getByTestId('message-char')).toHaveTextContent('hi from char')
      expect(screen.getByTestId('message-bot1')).toHaveTextContent('hi from bot1')
    })

    // round_complete has not arrived yet, so input is still gated.
    expect(textarea).toBeDisabled()

    releaseRoundComplete?.()

    await waitFor(() => {
      expect(textarea).not.toBeDisabled()
    })
  })

  it('settles a skipped speaker\'s bubble under the system speaker, not the speaker that was skipped', async () => {
    const user = userEvent.setup()
    // bot1 gets its own `reply_started` before the round learns it will not
    // respond, so the notice must settle that bubble as `system` rather
    // than leaving it tagged `bot1`. `round_complete` is gated so the
    // assertion runs before `refreshMessages()` replaces the optimistic
    // bubbles with the (unmocked) persisted rows.
    let releaseRoundComplete: (() => void) | undefined
    const roundCompleteGate = new Promise<void>(resolve => { releaseRoundComplete = resolve })

    const repliesChunks = [
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'reply_complete', content: 'hi from char', is_complete: false, speaker_id: 'char', message_id: 1 },
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'bot1' },
      { request_id: 'r1', event: 'reply_complete', content: 'bot1 did not respond', is_complete: false, speaker_id: 'system', message_id: 2 },
    ]
    const roundCompleteChunk = { request_id: 'r1', event: 'round_complete', content: '', is_complete: true, speaker_id: '' }

    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/prompt/stream')) {
        const encoder = new TextEncoder()
        return Promise.resolve({
          ok: true,
          status: 200,
          body: new ReadableStream<Uint8Array>({
            async start(controller) {
              for (const chunk of repliesChunks) {
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
              }
              await roundCompleteGate
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(roundCompleteChunk)}\n\n`))
              controller.close()
            },
          }),
        })
      }
      if (url.startsWith('/api/session')) {
        return Promise.resolve(jsonResponse(session))
      }
      if (url.startsWith('/api/attitude/summary/')) {
        return Promise.resolve(jsonResponse({ attitude, summary: 'neutral' }))
      }
      if (url.startsWith('/api/compaction')) {
        return Promise.resolve(jsonResponse({ checkpoints: [], pending_draft: null }))
      }
      return Promise.resolve(jsonResponse([]))
    })
    global.fetch = fetchMock as unknown as typeof fetch

    render(
      <MockProviders>
        <ChatWindow />
        <MessageSpy />
      </MockProviders>
    )

    const textarea = screen.getByRole('textbox')
    await user.type(textarea, 'hello')
    await user.click(screen.getByRole('button', { name: /^send message$/i }))

    await waitFor(() => {
      expect(screen.getByTestId('message-system')).toHaveTextContent('bot1 did not respond')
    })
    expect(screen.queryByTestId('message-bot1')).not.toBeInTheDocument()

    releaseRoundComplete?.()
    await waitFor(() => {
      expect(textarea).not.toBeDisabled()
    })
  })

  it('surfaces a mid-round error as a toast and re-enables input', async () => {
    const user = userEvent.setup()
    const streamChunks = [
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'reply_complete', content: 'hi', is_complete: false, speaker_id: 'char', message_id: 1 },
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'bot1' },
      { request_id: 'r1', event: 'error', content: '', is_complete: true, speaker_id: '', error: 'bot1 timed out' },
    ]

    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/prompt/stream')) {
        return Promise.resolve(streamResponse(streamChunks))
      }
      if (url.startsWith('/api/session')) {
        return Promise.resolve(jsonResponse(session))
      }
      if (url.startsWith('/api/attitude/summary/')) {
        return Promise.resolve(jsonResponse({ attitude, summary: 'neutral' }))
      }
      if (url.startsWith('/api/compaction')) {
        return Promise.resolve(jsonResponse({ checkpoints: [], pending_draft: null }))
      }
      return Promise.resolve(jsonResponse([]))
    })
    global.fetch = fetchMock as unknown as typeof fetch

    render(
      <MockProviders>
        <ChatWindow />
      </MockProviders>
    )

    const textarea = screen.getByRole('textbox')
    await user.type(textarea, 'hello')
    await user.click(screen.getByRole('button', { name: /^send message$/i }))

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('bot1 timed out'))
    })
    await user.type(textarea, 'again')
    expect(screen.getByRole('button', { name: /^send message$/i })).not.toBeDisabled()
  })

  it('surfaces a 409 as a still-replying toast', async () => {
    const user = userEvent.setup()

    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/prompt/stream')) {
        return Promise.resolve({ ok: false, status: 409, body: null })
      }
      if (url.startsWith('/api/session')) {
        return Promise.resolve(jsonResponse(session))
      }
      if (url.startsWith('/api/attitude/summary/')) {
        return Promise.resolve(jsonResponse({ attitude, summary: 'neutral' }))
      }
      if (url.startsWith('/api/compaction')) {
        return Promise.resolve(jsonResponse({ checkpoints: [], pending_draft: null }))
      }
      return Promise.resolve(jsonResponse([]))
    })
    global.fetch = fetchMock as unknown as typeof fetch

    render(
      <MockProviders>
        <ChatWindow />
      </MockProviders>
    )

    const textarea = screen.getByRole('textbox')
    await user.type(textarea, 'hello')
    await user.click(screen.getByRole('button', { name: /^send message$/i }))

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('still replying'))
    })
    // The textarea empties on send, so re-enabling is checked by typing again
    // and confirming the button responds rather than by its disabled state
    // alone (which an empty textarea would also produce).
    expect(textarea).not.toBeDisabled()
    await user.type(textarea, 'again')
    expect(screen.getByRole('button', { name: /^send message$/i })).not.toBeDisabled()
  })

  it('shows the participants strip in host mode', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) {
        return Promise.resolve(jsonResponse({ multiplayer_mode: 'host' }))
      }
      if (url.startsWith('/api/multiplayer/participants')) {
        return Promise.resolve(jsonResponse([
          { id: 'bot1', display_name: 'Ada', kind: 'RemoteBot', avatar_url: null, connected: true },
        ]))
      }
      if (url.startsWith('/api/session')) {
        return Promise.resolve(jsonResponse(session))
      }
      if (url.startsWith('/api/attitude/summary/')) {
        return Promise.resolve(jsonResponse({ attitude, summary: 'neutral' }))
      }
      if (url.startsWith('/api/compaction')) {
        return Promise.resolve(jsonResponse({ checkpoints: [], pending_draft: null }))
      }
      return Promise.resolve(jsonResponse([]))
    })
    global.fetch = fetchMock as unknown as typeof fetch

    render(
      <MockProviders>
        <ChatWindow />
      </MockProviders>
    )

    await waitFor(() => {
      expect(screen.getByTestId('participants-strip')).toBeInTheDocument()
    })
    expect(screen.getByRole('textbox')).toBeInTheDocument()
  })

  it('shows the mirrored-chat banner and disables the input in joiner mode', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) {
        return Promise.resolve(jsonResponse({ multiplayer_mode: 'joiner' }))
      }
      if (url.startsWith('/api/multiplayer/status')) {
        return Promise.resolve(jsonResponse({
          mode: 'joiner',
          state: 'connected',
          attempts: 1,
          host_address: '192.168.0.20:3000',
          participant_id: 'bot1',
          participants: [
            { id: 'user', display_name: 'Alice', kind: 'Human', avatar_url: null, connected: true },
          ],
        }))
      }
      if (url.startsWith('/api/session')) {
        return Promise.resolve(jsonResponse(session))
      }
      if (url.startsWith('/api/attitude/summary/')) {
        return Promise.resolve(jsonResponse({ attitude, summary: 'neutral' }))
      }
      if (url.startsWith('/api/compaction')) {
        return Promise.resolve(jsonResponse({ checkpoints: [], pending_draft: null }))
      }
      return Promise.resolve(jsonResponse([]))
    })
    global.fetch = fetchMock as unknown as typeof fetch

    render(
      <MockProviders>
        <ChatWindow />
      </MockProviders>
    )

    await waitFor(() => {
      expect(screen.getByTestId('joiner-banner')).toBeInTheDocument()
    })
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument()
  })
})
