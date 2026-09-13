import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EditData } from '../editData/EditData'
import { UserDataProvider } from '../context/userContext'
import { CompanionDataProvider } from '../context/companionContext'
import { ConfigProvider } from '../context/configContext'
import { MessagesProvider } from '../context/messageContext'
import { CompactionProvider } from '../context/compactionContext'
import { RunningThoughtsProvider } from '../context/runningThoughtsContext'

// `MemorySettings` (the Memory tab) reads `useCompaction()`, which itself
// reads `useMessages()`, so `CompactionProvider` must sit inside
// `MessagesProvider`; `EditData` itself also reads `useRunningThoughts()`
// (#238's "Clear running thoughts" button), which reads `useConfigData()` --
// matching `App.tsx`'s nesting order.
const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <UserDataProvider>
    <CompanionDataProvider>
      <ConfigProvider>
        <MessagesProvider>
          <CompactionProvider>
            <RunningThoughtsProvider>
              {children}
            </RunningThoughtsProvider>
          </CompactionProvider>
        </MessagesProvider>
      </ConfigProvider>
    </CompanionDataProvider>
  </UserDataProvider>
)

const FULL_CONFIG = {
  device: 'CPU',
  llm_model_path: '',
  gpu_layers: 0,
  prompt_template: 'Auto',
  context_window_size: 2048,
  max_response_tokens: 512,
  enable_dynamic_context: true,
  vram_limit_gb: 4,
  dynamic_gpu_allocation: false,
  gpu_safety_margin: 0.8,
  min_free_vram_mb: 512,
  multiplayer_mode: 'solo',
  multiplayer_password_set: false,
  multiplayer_host_address: '',
  multiplayer_participant_id: '',
  mention_followup_depth: 1,
  remote_generation_timeout_secs: 120,
  compact_threshold_tokens: null,
  compact_min_messages: 8,
  compaction_model_path: null,
  heuristic_person_detection: true,
  compaction_attitude_weight: 0.5,
  running_thoughts_enabled: false,
}

describe('EditData Component', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn((url: string) => {
      if (url.startsWith('/api/llm') || url.startsWith('/api/message')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve([]),
          text: () => Promise.resolve(''),
        })
      }
      if (url.startsWith('/api/compaction')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ checkpoints: [], pending_draft: null }),
          text: () => Promise.resolve(''),
        })
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({}),
        text: () => Promise.resolve(''),
      })
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders edit data tabs', () => {
    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )

    expect(screen.getByRole('tablist')).toBeInTheDocument()
    const tabs = screen.getAllByRole('tab').map(t => t.textContent)
    expect(tabs).toEqual(['Companion', 'User', 'Attitudes', 'Theme', 'Config', 'Memory', 'Multiplayer'])
  })

  it('shows user data tab', async () => {
    const user = userEvent.setup()

    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )

    await user.click(screen.getByRole('tab', { name: 'User' }))
    expect(await screen.findByLabelText('Your name')).toBeInTheDocument()
  })

  it('shows companion data tab', async () => {
    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )

    expect(screen.getByRole('tab', { name: 'Companion' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByLabelText('Your companion name')).toBeInTheDocument()
  })

  it('shows config data tab', async () => {
    const user = userEvent.setup()

    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )

    await user.click(screen.getByRole('tab', { name: 'Config' }))
    expect(await screen.findByText('Prompt template')).toBeInTheDocument()
  })

  it('handles form submission', async () => {
    const user = userEvent.setup()

    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )

    await user.click(screen.getByRole('button', { name: /save changes/i }))

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/api/companion', expect.objectContaining({ method: 'PUT' }))
    })
  })

  it('shows the memory tab with the extraction model picker', async () => {
    const user = userEvent.setup()

    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )

    await user.click(screen.getByRole('tab', { name: 'Memory' }))
    expect(await screen.findByLabelText('Extraction model')).toBeInTheDocument()
    // These two fields' labels also wrap a tooltip trigger button, so
    // `getByLabelText` would match both it and the input; `spinbutton`
    // narrows it back to just the `<input>`.
    expect(screen.getByRole('spinbutton', { name: 'Compaction threshold (tokens)' })).toBeInTheDocument()
    expect(screen.getByRole('spinbutton', { name: 'Narrative attitude weight' })).toBeInTheDocument()
  })

  it('saves the memory tab through PUT /api/config with the compaction fields', async () => {
    vi.stubGlobal('fetch', vi.fn((url: string, options?: RequestInit) => {
      if (url === '/api/config' && !options) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(FULL_CONFIG),
          text: () => Promise.resolve(''),
        })
      }
      if (url.startsWith('/api/llm') || url.startsWith('/api/message')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([]), text: () => Promise.resolve('') })
      }
      if (url.startsWith('/api/compaction')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ checkpoints: [], pending_draft: null }),
          text: () => Promise.resolve(''),
        })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}), text: () => Promise.resolve('') })
    }))

    const user = userEvent.setup()

    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )

    await user.click(screen.getByRole('tab', { name: 'Memory' }))
    await screen.findByLabelText('Extraction model')
    await user.click(screen.getByRole('button', { name: /save changes/i }))

    const mockFetch = fetch as unknown as ReturnType<typeof vi.fn>
    await waitFor(() => {
      const putCalls = mockFetch.mock.calls.filter(call => call[0] === '/api/config' && call[1]?.method === 'PUT')
      expect(putCalls.length).toBeGreaterThan(0)
    })
    const putCalls = mockFetch.mock.calls.filter(call => call[0] === '/api/config' && call[1]?.method === 'PUT')
    const body = JSON.parse(putCalls[putCalls.length - 1][1].body)
    expect(body).toHaveProperty('compact_threshold_tokens')
    expect(body).toHaveProperty('compact_min_messages')
    expect(body).toHaveProperty('compaction_attitude_weight')
    expect(body).toHaveProperty('compaction_model_path')
    expect(body).toHaveProperty('heuristic_person_detection')
    expect(body).toHaveProperty('running_thoughts_enabled')
  })

  // #238: each new clear button opens a confirm dialog. `DialogTrigger`
  // renders its own `<button>` around the `<Button>` child rather than
  // merging with it (no `asChild` here, matching every other trigger in
  // this file), so two elements already share the trigger's accessible
  // name before the dialog even opens; `getAllByRole(...)[0]` opens it,
  // and once open the dialog's own confirm button is the last match.
  // Confirming calls the route and refreshes the panel that route's data
  // feeds.
  it('clears running thoughts and refreshes the thoughts panel', async () => {
    const user = userEvent.setup()

    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )

    await user.click(screen.getAllByRole('button', { name: 'Clear running thoughts' })[0])
    const confirmButtons = await screen.findAllByRole('button', { name: 'Clear running thoughts' })
    await user.click(confirmButtons[confirmButtons.length - 1])

    const mockFetch = fetch as unknown as ReturnType<typeof vi.fn>
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/thoughts/clear', expect.objectContaining({ method: 'DELETE' }))
    })
    // The initial mount already issued one `GET /api/thoughts` for
    // `RunningThoughtsProvider`'s own load; a successful clear must issue a
    // second one to drop the now-deleted rows from the panel.
    const thoughtGets = mockFetch.mock.calls.filter(call => call[0] === '/api/thoughts')
    expect(thoughtGets.length).toBeGreaterThan(1)
  })

  it('clears compaction history and refreshes the compaction checkpoints', async () => {
    const user = userEvent.setup()

    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )

    await user.click(screen.getAllByRole('button', { name: 'Clear compaction history' })[0])
    const confirmButtons = await screen.findAllByRole('button', { name: 'Clear compaction history' })
    await user.click(confirmButtons[confirmButtons.length - 1])

    const mockFetch = fetch as unknown as ReturnType<typeof vi.fn>
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/compaction/clear', expect.objectContaining({ method: 'DELETE' }))
    })
    const compactionGets = mockFetch.mock.calls.filter(call => call[0] === '/api/compaction')
    expect(compactionGets.length).toBeGreaterThan(1)
  })

  it('clears known people', async () => {
    const user = userEvent.setup()

    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )

    await user.click(screen.getAllByRole('button', { name: 'Clear known people' })[0])
    const confirmButtons = await screen.findAllByRole('button', { name: 'Clear known people' })
    await user.click(confirmButtons[confirmButtons.length - 1])

    const mockFetch = fetch as unknown as ReturnType<typeof vi.fn>
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/persons/clear', expect.objectContaining({ method: 'DELETE' }))
    })
  })
})
