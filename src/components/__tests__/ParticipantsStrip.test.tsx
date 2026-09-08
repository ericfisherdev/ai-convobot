import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ParticipantsStrip } from '../multiplayer/ParticipantsStrip'
import { ParticipantsProvider } from '../context/participantsContext'
import { ConfigProvider } from '../context/configContext'
import { UserDataProvider } from '../context/userContext'
import { CompanionDataProvider } from '../context/companionContext'

const jsonResponse = (body: unknown) => ({ ok: true, status: 200, json: () => Promise.resolve(body) })

const hostConfig = { multiplayer_mode: 'host' }

const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <ConfigProvider>
    <UserDataProvider>
      <CompanionDataProvider>
        <ParticipantsProvider>{children}</ParticipantsProvider>
      </CompanionDataProvider>
    </UserDataProvider>
  </ConfigProvider>
)

describe('ParticipantsStrip', () => {
  // `vi.waitFor` (not testing-library's `waitFor`) is used throughout: it
  // polls with the real, un-mocked timers `getSafeTimers()` captures, so it
  // still resolves while `vi.useFakeTimers()` is active.
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('renders one chip per participant with a dot reflecting connected state', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse(hostConfig))
      if (url.startsWith('/api/multiplayer/participants')) {
        return Promise.resolve(jsonResponse([
          { id: 'bot1', display_name: 'Ada', kind: 'RemoteBot', avatar_url: null, connected: true },
          { id: 'bot2', display_name: 'Grace', kind: 'RemoteBot', avatar_url: null, connected: false },
        ]))
      }
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <ParticipantsStrip />
      </MockProviders>
    )

    await vi.waitFor(() => {
      expect(screen.getByTestId('participant-chip-bot1')).toBeInTheDocument()
    })
    expect(screen.getByTestId('participant-chip-user')).toBeInTheDocument()
    expect(screen.getByTestId('participant-chip-char')).toBeInTheDocument()
    expect(screen.getByTestId('participant-status-bot1')).toHaveAttribute('data-connected', 'true')
    expect(screen.getByTestId('participant-status-bot2')).toHaveAttribute('data-connected', 'false')
  })

  it('refetches on the poll interval and drops a participant that vanished from the response', async () => {
    let participantsResponse = [
      { id: 'bot1', display_name: 'Ada', kind: 'RemoteBot', avatar_url: null, connected: true },
    ]
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/config')) return Promise.resolve(jsonResponse(hostConfig))
      if (url.startsWith('/api/multiplayer/participants')) {
        return Promise.resolve(jsonResponse(participantsResponse))
      }
      return Promise.resolve(jsonResponse({}))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <MockProviders>
        <ParticipantsStrip />
      </MockProviders>
    )

    await vi.waitFor(() => {
      expect(screen.getByTestId('participant-chip-bot1')).toBeInTheDocument()
    })

    participantsResponse = []
    await vi.advanceTimersByTimeAsync(10_000)

    await vi.waitFor(() => {
      expect(screen.queryByTestId('participant-chip-bot1')).not.toBeInTheDocument()
    })
  })
})
