import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Message } from '../message/Message'
import { MessagesProvider } from '../context/messageContext'
import { UserDataProvider } from '../context/userContext'
import { CompanionDataProvider } from '../context/companionContext'
import { ConfigProvider } from '../context/configContext'
import { ParticipantsProvider } from '../context/participantsContext'
import { formatMessageDate } from '../../lib/utils'
import { isBotSpeaker } from '../../lib/speakers'
import { toast } from 'sonner'

vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
  },
}))

const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <MessagesProvider>
    <UserDataProvider>
      <CompanionDataProvider>
        <ConfigProvider>
          <ParticipantsProvider>
            {children}
          </ParticipantsProvider>
        </ConfigProvider>
      </CompanionDataProvider>
    </UserDataProvider>
  </MessagesProvider>
)

describe('Message Component', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({}),
        text: () => Promise.resolve(''),
      })
    ))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders user message correctly', async () => {
    render(
      <MockProviders>
        <Message received={false} regenerate={false} id={1} content="Hello, this is a test message" created_at="2024-01-15 10:30" speakerId="user" />
      </MockProviders>
    )

    expect(await screen.findByText('Hello, this is a test message')).toBeInTheDocument()
    expect(screen.getByText(formatMessageDate('2024-01-15 10:30'))).toBeInTheDocument()
  })

  it('renders AI message correctly', async () => {
    render(
      <MockProviders>
        <Message received={true} regenerate={false} id={2} content="Hello! How can I help you today?" created_at="2024-01-15 10:31" speakerId="char" />
      </MockProviders>
    )

    expect(await screen.findByText('Hello! How can I help you today?')).toBeInTheDocument()
  })

  it('displays edit and delete buttons for messages', async () => {
    render(
      <MockProviders>
        <Message received={false} regenerate={false} id={1} content="Hello, this is a test message" created_at="2024-01-15 10:30" speakerId="user" />
      </MockProviders>
    )

    // Look for edit and delete buttons (they might be icon buttons)
    await screen.findByText('Hello, this is a test message')
    const buttons = screen.getAllByRole('button')
    expect(buttons.length).toBeGreaterThanOrEqual(2)
  })

  it('shows markdown content correctly', async () => {
    render(
      <MockProviders>
        <Message received={true} regenerate={false} id={3} content="**Bold text** and *italic text*" created_at="2024-01-15 10:32" speakerId="char" />
      </MockProviders>
    )

    expect(await screen.findByText('Bold text')).toBeInTheDocument()
    expect(screen.getByText('italic text')).toBeInTheDocument()
  })

  it('renders the regenerate control on a trailing AI message', async () => {
    render(
      <MockProviders>
        <Message received={true} regenerate={true} id={4} content="Hello! How can I help you today?" created_at="2024-01-15 10:33" speakerId="char" />
      </MockProviders>
    )

    await screen.findByText('Hello! How can I help you today?')
    expect(screen.getByRole('button', { name: 'Regenerate message' })).toBeInTheDocument()
  })

  it('never renders the regenerate control on a user message', async () => {
    render(
      <MockProviders>
        <Message received={false} regenerate={true} id={5} content="Hello, this is a test message" created_at="2024-01-15 10:34" speakerId="user" />
      </MockProviders>
    )

    await screen.findByText('Hello, this is a test message')
    expect(screen.queryByRole('button', { name: 'Regenerate message' })).not.toBeInTheDocument()
  })

  it('editing an AI message sends only the new content', async () => {
    const user = userEvent.setup()

    render(
      <MockProviders>
        <Message received={true} regenerate={false} id={2} content="Hello! How can I help you today?" created_at="2024-01-15 10:31" speakerId="char" />
      </MockProviders>
    )

    await screen.findByText('Hello! How can I help you today?')
    await user.click(screen.getByRole('button', { name: 'Edit message' }))
    const textbox = screen.getByRole('textbox')
    await user.clear(textbox)
    await user.type(textbox, 'edited ai reply')
    await user.click(screen.getByRole('button', { name: 'Save message' }))

    const mockFetch = fetch as unknown as ReturnType<typeof vi.fn>
    const putCalls = mockFetch.mock.calls.filter(call => call[1]?.method === 'PUT')
    const lastPutCall = putCalls[putCalls.length - 1]
    expect(lastPutCall[0]).toBe('/api/message/2')
    expect(JSON.parse(lastPutCall[1].body)).toEqual({ content: 'edited ai reply' })
  })

  it('editing a user message sends only the new content', async () => {
    const user = userEvent.setup()

    render(
      <MockProviders>
        <Message received={false} regenerate={false} id={1} content="Hello, this is a test message" created_at="2024-01-15 10:30" speakerId="user" />
      </MockProviders>
    )

    await screen.findByText('Hello, this is a test message')
    await user.click(screen.getByRole('button', { name: 'Edit message' }))
    const textbox = screen.getByRole('textbox')
    await user.clear(textbox)
    await user.type(textbox, 'edited user message')
    await user.click(screen.getByRole('button', { name: 'Save message' }))

    const mockFetch = fetch as unknown as ReturnType<typeof vi.fn>
    const putCalls = mockFetch.mock.calls.filter(call => call[1]?.method === 'PUT')
    const lastPutCall = putCalls[putCalls.length - 1]
    expect(lastPutCall[0]).toBe('/api/message/1')
    expect(JSON.parse(lastPutCall[1].body)).toEqual({ content: 'edited user message' })
  })

  it('shows a bot participant\'s display name and avatar for a multiplayer speaker', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ multiplayer_mode: 'host' }) })
      }
      if (url.startsWith('/api/multiplayer/participants')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve([
            { id: 'bot1', display_name: 'Ada', kind: 'RemoteBot', avatar_url: '/api/multiplayer/participants/bot1/avatar', connected: true },
          ]),
        })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}), text: () => Promise.resolve('') })
    }))

    render(
      <MockProviders>
        <Message received={true} regenerate={false} id={6} content="hi from bot1" created_at="2024-01-15 10:35" speakerId="bot1" />
      </MockProviders>
    )

    expect(await screen.findByText('Ada')).toBeInTheDocument()
    expect(await screen.findByAltText('Ada avatar')).toHaveAttribute('src', '/api/multiplayer/participants/bot1/avatar')
  })

  it('renders a system message as a centred notice with no action buttons', async () => {
    render(
      <MockProviders>
        <Message received={true} regenerate={false} id={7} content="bot1 did not respond" created_at="2024-01-15 10:36" speakerId="system" />
      </MockProviders>
    )

    expect(await screen.findByText('bot1 did not respond')).toBeInTheDocument()
    // A system notice carries none of `UserMessage`/`AiMessage`'s edit,
    // delete, reaction or regenerate controls.
    expect(screen.queryAllByRole('button')).toHaveLength(0)
  })

  it('renders the regenerate control for a trailing bot1 reply', async () => {
    render(
      <MockProviders>
        <Message
          received={true}
          regenerate={isBotSpeaker({ speaker_id: 'bot1' })}
          id={8}
          content="hi from bot1"
          created_at="2024-01-15 10:37"
          speakerId="bot1"
        />
      </MockProviders>
    )

    await screen.findByText('hi from bot1')
    expect(screen.getByRole('button', { name: 'Regenerate message' })).toBeInTheDocument()
  })

  it('suppresses the regenerate control in joiner mode even when the caller passes regenerate true', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ multiplayer_mode: 'joiner' }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}), text: () => Promise.resolve('') })
    }))

    render(
      <MockProviders>
        <Message received={true} regenerate={true} id={11} content="hi from bot1" created_at="2024-01-15 10:40" speakerId="bot1" />
      </MockProviders>
    )

    await screen.findByText('hi from bot1')
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'Regenerate message' })).not.toBeInTheDocument()
    )
  })

  it('does not render the regenerate control before config has finished loading', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) {
        // Never resolves, so `config` stays `null` for the life of the test
        // — the same state `ConfigProvider` starts in before its fetch
        // settles. A joiner instance must not be treated as `host` during
        // this window just because `multiplayer_mode` isn't known yet.
        return new Promise(() => {})
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}), text: () => Promise.resolve('') })
    }))

    render(
      <MockProviders>
        <Message received={true} regenerate={true} id={12} content="hi from bot1" created_at="2024-01-15 10:41" speakerId="bot1" />
      </MockProviders>
    )

    await screen.findByText('hi from bot1')
    expect(screen.queryByRole('button', { name: 'Regenerate message' })).not.toBeInTheDocument()
  })

  it('never renders the regenerate control on a trailing system notice, even if the caller passes regenerate true', async () => {
    render(
      <MockProviders>
        <Message received={true} regenerate={true} id={9} content="bot1 did not respond" created_at="2024-01-15 10:38" speakerId="system" />
      </MockProviders>
    )

    await screen.findByText('bot1 did not respond')
    expect(screen.queryByRole('button', { name: 'Regenerate message' })).not.toBeInTheDocument()
  })

  it('surfaces the regenerate 409 body through the toast', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url === '/api/prompt/regenerate') {
        return Promise.resolve({
          ok: false,
          text: () => Promise.resolve('bot1 is not connected, so its reply cannot be regenerated'),
        })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}), text: () => Promise.resolve('') })
    }))

    const user = userEvent.setup()
    render(
      <MockProviders>
        <Message received={true} regenerate={true} id={10} content="hi from bot1" created_at="2024-01-15 10:39" speakerId="bot1" />
      </MockProviders>
    )

    await screen.findByText('hi from bot1')
    await user.click(screen.getByRole('button', { name: 'Regenerate message' }))

    const mockError = toast.error as unknown as ReturnType<typeof vi.fn>
    await waitFor(() => expect(mockError).toHaveBeenCalled())
    const lastCall = mockError.mock.calls[mockError.mock.calls.length - 1]
    expect(lastCall[0]).toBe('bot1 is not connected, so its reply cannot be regenerated')
  })
})
