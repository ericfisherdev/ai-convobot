import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ChatWindow from '../ChatWindow'
import { MessagesProvider } from '../context/messageContext'
import { UserDataProvider } from '../context/userContext'
import { CompanionDataProvider } from '../context/companionContext'
import { ConfigProvider } from '../context/configContext'
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

// Mock the contexts with minimal implementations
const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
    <MessagesProvider>
      <UserDataProvider>
        <CompanionDataProvider>
          <ConfigProvider>
            <AttitudeProvider>
              <SessionProvider>
                {children}
              </SessionProvider>
            </AttitudeProvider>
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
      if (url.startsWith('/api/session')) {
        return Promise.resolve(jsonResponse(session))
      }
      if (url.startsWith('/api/attitude/summary/')) {
        return Promise.resolve(jsonResponse({ attitude, summary: 'neutral' }))
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
      { request_id: 'r1', content: 'hi', is_complete: false, token_count: 1 },
      {
        request_id: 'r1',
        content: '',
        is_complete: false,
        token_count: 1,
        attitude: {
          attitude: { ...attitude, trust: 7 },
          summary: 'warmer',
          deltas: [{ dimension: 'trust', delta: 3 }],
        },
      },
      { request_id: 'r1', content: 'hi', is_complete: true, token_count: 1 },
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
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify({ request_id: 'r1', content: 'hi', is_complete: true, token_count: 1 })}\n\n`)
              )
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
})
