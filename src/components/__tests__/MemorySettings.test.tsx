import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, within, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemorySettings } from '../editData/MemorySettings'
import { MessagesProvider } from '../context/messageContext'
import { CompactionProvider } from '../context/compactionContext'
import { ConfigInterface, Device, MultiplayerMode, PromptTemplate } from '../interfaces/Config'
import { CheckpointSummary } from '../interfaces/Compaction'
import { toast } from 'sonner'

vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
  },
}))

// `MemorySettings` reads `useCompaction()`, which itself reads
// `useMessages()` for `refreshMessages` on a successful pin/unpin, matching
// `App.tsx`'s nesting order.
const renderMemorySettings = (ui: React.ReactElement) =>
  render(
    <MessagesProvider>
      <CompactionProvider>{ui}</CompactionProvider>
    </MessagesProvider>
  )

const baseConfig: ConfigInterface = {
  device: Device.CPU,
  llm_model_path: '',
  gpu_layers: 0,
  prompt_template: PromptTemplate.Auto,
  context_window_size: 2048,
  max_response_tokens: 512,
  enable_dynamic_context: true,
  vram_limit_gb: 4,
  dynamic_gpu_allocation: true,
  gpu_safety_margin: 0.8,
  min_free_vram_mb: 512,
  multiplayer_mode: MultiplayerMode.Solo,
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

const MODELS = [
  { path: '/models/qwen3-4b-instruct.Q4_K_M.gguf', filename: 'qwen3-4b-instruct.Q4_K_M.gguf', size_bytes: 1, directory: '/models', last_modified: '' },
  { path: '/models/llama-3.2-3b-it.Q4_K_M.gguf', filename: 'llama-3.2-3b-it.Q4_K_M.gguf', size_bytes: 1, directory: '/models', last_modified: '' },
  { path: '/models/mistral-7b-chat.Q4_K_M.gguf', filename: 'mistral-7b-chat.Q4_K_M.gguf', size_bytes: 1, directory: '/models', last_modified: '' },
  { path: '/models/roleplay-base.Q4_K_M.gguf', filename: 'roleplay-base.Q4_K_M.gguf', size_bytes: 1, directory: '/models', last_modified: '' },
]

const CHECKPOINTS: CheckpointSummary[] = [
  { id: 1, from_message_id: 1, through_message_id: 20, status: 'committed', trigger: 'threshold', committed_at: '2024-01-15 10:30', needs_merge: false, extraction_error: null },
  { id: 2, from_message_id: 21, through_message_id: 40, status: 'stale', trigger: 'threshold', committed_at: '2024-01-16 10:30', needs_merge: true, extraction_error: null },
]

const stubFetch = (overrides: Record<string, () => Promise<unknown>> = {}) => {
  vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString()
    for (const [prefix, respond] of Object.entries(overrides)) {
      if (url.startsWith(prefix)) {
        return respond()
      }
    }
    if (url.startsWith('/api/message')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve([]), text: () => Promise.resolve('') })
    }
    if (url.startsWith('/api/compaction')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ checkpoints: CHECKPOINTS, pending_draft: null }),
        text: () => Promise.resolve(''),
      })
    }
    if (url.startsWith('/api/llm/models')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(MODELS), text: () => Promise.resolve('') })
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}), text: () => Promise.resolve(''), status: init?.method === 'POST' ? 404 : 200 })
  }))
}

describe('MemorySettings', () => {
  beforeEach(() => {
    // jsdom implements neither: the extraction model picker's `Select`
    // (Radix) needs `hasPointerCapture` on open/close and `scrollIntoView`
    // when an item scrolls into view.
    Element.prototype.hasPointerCapture = vi.fn(() => false)
    Element.prototype.scrollIntoView = vi.fn()
    stubFetch()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('updates compact_min_messages on change', () => {
    const onChange = vi.fn()
    renderMemorySettings(<MemorySettings config={baseConfig} onChange={onChange} />)

    const input = screen.getByLabelText('Minimum messages before compaction')
    fireEvent.change(input, { target: { value: '5' } })

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ compact_min_messages: 5 }))
  })

  it('maps a blank compaction threshold to null', () => {
    const onChange = vi.fn()
    renderMemorySettings(
      <MemorySettings config={{ ...baseConfig, compact_threshold_tokens: 4096 }} onChange={onChange} />
    )

    // A `FieldLabel`-wrapped input's accessible name also picks up its
    // tooltip trigger button, so `getByLabelText` matches both; the
    // `spinbutton` role narrows it back to just the `<input>`.
    const input = screen.getByRole('spinbutton', { name: 'Compaction threshold (tokens)' })
    fireEvent.change(input, { target: { value: '' } })

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ compact_threshold_tokens: null }))
  })

  it('sets a numeric compaction threshold on change', () => {
    const onChange = vi.fn()
    renderMemorySettings(<MemorySettings config={baseConfig} onChange={onChange} />)

    const input = screen.getByRole('spinbutton', { name: 'Compaction threshold (tokens)' })
    fireEvent.change(input, { target: { value: '4096' } })

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ compact_threshold_tokens: 4096 }))
  })

  it('updates compaction_attitude_weight on change', () => {
    const onChange = vi.fn()
    renderMemorySettings(<MemorySettings config={baseConfig} onChange={onChange} />)

    const input = screen.getByRole('spinbutton', { name: 'Narrative attitude weight' })
    fireEvent.change(input, { target: { value: '0.75' } })

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ compaction_attitude_weight: 0.75 }))
  })

  it('toggles heuristic_person_detection', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    renderMemorySettings(
      <MemorySettings config={{ ...baseConfig, heuristic_person_detection: true }} onChange={onChange} />
    )

    await user.click(screen.getByRole('switch', { name: 'Heuristic person detection' }))

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ heuristic_person_detection: false })
    )
  })

  it('toggles running_thoughts_enabled', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    renderMemorySettings(
      <MemorySettings config={{ ...baseConfig, running_thoughts_enabled: false }} onChange={onChange} />
    )

    await user.click(screen.getByRole('switch', { name: 'Running thoughts' }))

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ running_thoughts_enabled: true })
    )
  })

  it('posts to the long-term index rebuild endpoint and toasts the backend result', async () => {
    stubFetch({
      '/api/memory/longTerm/rebuild': () => Promise.resolve({ ok: true, text: () => Promise.resolve('Long term memory rebuilt from 42 facts') }),
    })
    const user = userEvent.setup()
    renderMemorySettings(<MemorySettings config={baseConfig} onChange={vi.fn()} />)

    await user.click(screen.getByRole('button', { name: 'Rebuild long-term index' }))

    const mockFetch = fetch as unknown as ReturnType<typeof vi.fn>
    await waitFor(() => {
      expect(mockFetch.mock.calls.some(call => call[0] === '/api/memory/longTerm/rebuild' && call[1]?.method === 'POST')).toBe(true)
    })

    const mockSuccess = toast.success as unknown as ReturnType<typeof vi.fn>
    await waitFor(() => expect(mockSuccess).toHaveBeenCalledWith('Long term memory rebuilt from 42 facts'))
  })

  it('toasts the backend error text when the rebuild fails', async () => {
    stubFetch({
      '/api/memory/longTerm/rebuild': () => Promise.resolve({
        ok: false,
        text: () => Promise.resolve('A reply is still being generated; wait for it to finish before rebuilding long term memory'),
      }),
    })
    const user = userEvent.setup()
    renderMemorySettings(<MemorySettings config={baseConfig} onChange={vi.fn()} />)

    await user.click(screen.getByRole('button', { name: 'Rebuild long-term index' }))

    const mockError = toast.error as unknown as ReturnType<typeof vi.fn>
    await waitFor(() =>
      expect(mockError).toHaveBeenCalledWith('A reply is still being generated; wait for it to finish before rebuilding long term memory')
    )
  })

  it('hides non-instruct models until "Show all models" is toggled', async () => {
    const user = userEvent.setup()
    renderMemorySettings(<MemorySettings config={baseConfig} onChange={vi.fn()} />)

    await user.click(screen.getByLabelText('Extraction model'))
    expect(await screen.findByText('qwen3-4b-instruct.Q4_K_M.gguf')).toBeInTheDocument()
    expect(screen.getByText('llama-3.2-3b-it.Q4_K_M.gguf')).toBeInTheDocument()
    expect(screen.getByText('mistral-7b-chat.Q4_K_M.gguf')).toBeInTheDocument()
    expect(screen.queryByText('roleplay-base.Q4_K_M.gguf')).not.toBeInTheDocument()

    await user.keyboard('{Escape}')
    await user.click(screen.getByLabelText('Show all models'))
    await user.click(screen.getByLabelText('Extraction model'))
    expect(await screen.findByText('roleplay-base.Q4_K_M.gguf')).toBeInTheDocument()
  })

  it('shows a "Same as chat model" option that writes null', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    renderMemorySettings(
      <MemorySettings config={{ ...baseConfig, compaction_model_path: '/models/qwen3-4b-instruct.Q4_K_M.gguf' }} onChange={onChange} />
    )

    await user.click(screen.getByLabelText('Extraction model'))
    await user.click(await screen.findByText('Same as chat model'))

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ compaction_model_path: null })
    )
  })

  it('renders checkpoint history with stale and needs-merge badges', async () => {
    renderMemorySettings(<MemorySettings config={baseConfig} onChange={vi.fn()} />)

    // `checkpoints` populates asynchronously via `CompactionProvider`'s own
    // `GET /api/compaction` fetch on mount.
    expect(await screen.findByText('Messages 1-20')).toBeInTheDocument()
    expect(screen.getByText('Messages 21-40')).toBeInTheDocument()
    expect(screen.getByText('Stale')).toBeInTheDocument()
    expect(screen.getByText('Needs merge')).toBeInTheDocument()
  })

  it('fetches and shows rendered notes for a checkpoint', async () => {
    stubFetch({
      '/api/debug/prompt': () => Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          compaction: {
            user_overlay: 'USER_OVERLAY_TEXT',
            companion_overlay: '',
            rules: '',
            story_so_far: '',
            recent_detail: '',
            pins: '',
          },
        }),
      }),
    })
    const user = userEvent.setup()
    renderMemorySettings(<MemorySettings config={baseConfig} onChange={vi.fn()} />)

    const [firstViewButton] = await screen.findAllByRole('button', { name: 'View rendered notes' })
    await user.click(firstViewButton)

    const dialog = await screen.findByRole('dialog')
    expect(await within(dialog).findByText('USER_OVERLAY_TEXT')).toBeInTheDocument()
  })
})
