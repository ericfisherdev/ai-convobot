import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act, waitFor } from '@testing-library/react'
import { AttitudeSummaryBar } from '../attitude/AttitudeSummaryBar'
import { AttitudeProvider, useAttitude } from '../context/attitudeContext'
import { CompanionDataProvider } from '../context/companionContext'
import { UserDataProvider } from '../context/userContext'
import { AttitudeData, AttitudeStreamUpdate } from '../interfaces/AttitudeData'

const buildAttitude = (overrides: Partial<AttitudeData> = {}): AttitudeData => ({
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
  ...overrides,
})

const summaryResponse = (attitude: AttitudeData, summary = '{{companion}} feels neutral toward {{user}}') => ({
  ok: true,
  status: 200,
  json: () => Promise.resolve({ attitude, summary }),
})

// The bar reads names out of the companion/user contexts, which fetch on mount.
const nonAttitudeResponse = { ok: true, status: 200, json: () => Promise.resolve({}) }

const renderBar = (children?: React.ReactNode) =>
  render(
    <UserDataProvider>
      <CompanionDataProvider>
        <AttitudeProvider>
          <AttitudeSummaryBar companionId={1} userId={1} />
          {children}
        </AttitudeProvider>
      </CompanionDataProvider>
    </UserDataProvider>
  )

// Lets a test drive the provider the way the SSE stream does.
let applyUpdate: (update: AttitudeStreamUpdate) => void
const StreamHarness: React.FC = () => {
  const { applyAttitudeStreamUpdate } = useAttitude()
  applyUpdate = applyAttitudeStreamUpdate
  return null
}

describe('AttitudeSummaryBar Component', () => {
  beforeEach(() => {
    global.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/attitude/summary/')) {
        return Promise.resolve(summaryResponse(buildAttitude({ trust: 4 })))
      }
      return Promise.resolve(nonAttitudeResponse)
    }) as unknown as typeof fetch
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders nothing until the first summary fetch resolves, then shows core dimensions', async () => {
    renderBar()

    expect(screen.queryByTestId('attitude-summary-bar')).not.toBeInTheDocument()

    expect(await screen.findByTestId('attitude-summary-bar')).toBeInTheDocument()
    // Every value is under the display threshold, so only the core set carries
    // the bar.
    expect(screen.getByText('Trust')).toBeInTheDocument()
    expect(screen.getByText('Love')).toBeInTheDocument()
    expect(screen.getByText('Curiosity')).toBeInTheDocument()
    expect(screen.getByText('Anger')).toBeInTheDocument()
  })

  it('keeps the previous values on screen while a refetch is in flight', async () => {
    renderBar()
    await screen.findByTestId('attitude-summary-bar')
    expect(screen.getByText('+4')).toBeInTheDocument()

    // The refetch never settles, so anything still rendered is the old state.
    global.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/attitude/summary/')) {
        return new Promise(() => {})
      }
      return Promise.resolve(nonAttitudeResponse)
    }) as unknown as typeof fetch

    act(() => {
      window.dispatchEvent(new CustomEvent('attitude-update'))
    })

    expect(screen.getByTestId('attitude-summary-bar')).toBeInTheDocument()
    expect(screen.getByText('+4')).toBeInTheDocument()
  })

  it('badges only the dimensions the last turn moved', async () => {
    renderBar(<StreamHarness />)
    await screen.findByTestId('attitude-summary-bar')

    act(() => {
      applyUpdate({
        attitude: buildAttitude({ trust: 7 }),
        summary: '{{companion}} trusts {{user}} a little more',
        deltas: [{ dimension: 'trust', delta: 3 }],
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId('attitude-delta-trust')).toHaveTextContent('+3')
    })
    expect(screen.queryByTestId('attitude-delta-love')).not.toBeInTheDocument()
  })

  it('shows a non-core dimension above the threshold and hides a quiet one', async () => {
    global.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.startsWith('/api/attitude/summary/')) {
        return Promise.resolve(summaryResponse(buildAttitude({ empathy: 40, jealousy: 5 })))
      }
      return Promise.resolve(nonAttitudeResponse)
    }) as unknown as typeof fetch

    renderBar()
    await screen.findByTestId('attitude-summary-bar')

    expect(screen.getByText('Empathy')).toBeInTheDocument()
    expect(screen.queryByText('Jealousy')).not.toBeInTheDocument()
  })

  it('substitutes companion and user placeholders into the summary', async () => {
    renderBar()

    expect(await screen.findByText('Companion feels neutral toward User')).toBeInTheDocument()
  })
})
