import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LlmModelSelector } from '../llm/LlmModelSelector'
import { ModelInfo } from '../interfaces/Config'

const MODELS: ModelInfo[] = [
  { path: '/models/qwen3-4b-instruct.Q4_K_M.gguf', filename: 'qwen3-4b-instruct.Q4_K_M.gguf', size_bytes: 1, directory: '/models', last_modified: '' },
  { path: '/models/roleplay-base.Q4_K_M.gguf', filename: 'roleplay-base.Q4_K_M.gguf', size_bytes: 1, directory: '/models', last_modified: '' },
]

const INSTRUCT_FILTER = (model: ModelInfo) => /instruct/i.test(model.filename);

describe('LlmModelSelector', () => {
  beforeEach(() => {
    // jsdom implements neither: Radix `Select` needs `hasPointerCapture` on
    // open/close and `scrollIntoView` when an item scrolls into view.
    Element.prototype.hasPointerCapture = vi.fn(() => false)
    Element.prototype.scrollIntoView = vi.fn()
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({ ok: true, json: () => Promise.resolve(MODELS) })
    ))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('keeps the currently selected model visible even when a filter would otherwise hide it', async () => {
    const user = userEvent.setup()
    render(
      <LlmModelSelector
        id="test-select"
        label="Model"
        selectedModel="/models/roleplay-base.Q4_K_M.gguf"
        onModelSelect={vi.fn()}
        filter={INSTRUCT_FILTER}
      />
    )

    // The trigger shows the selected model's own text, not a blank
    // placeholder, and the "Current: ..." line below still renders.
    expect(await screen.findByText('roleplay-base.Q4_K_M.gguf')).toBeInTheDocument()

    await user.click(screen.getByLabelText('Model'))
    expect(await screen.findAllByText('roleplay-base.Q4_K_M.gguf')).not.toHaveLength(0)
  })

  it('still hides a non-matching, non-selected model', async () => {
    const user = userEvent.setup()
    render(
      <LlmModelSelector
        id="test-select"
        label="Model"
        selectedModel={undefined}
        onModelSelect={vi.fn()}
        filter={INSTRUCT_FILTER}
      />
    )

    await user.click(screen.getByLabelText('Model'))
    expect(await screen.findByText('qwen3-4b-instruct.Q4_K_M.gguf')).toBeInTheDocument()
    expect(screen.queryByText('roleplay-base.Q4_K_M.gguf')).not.toBeInTheDocument()
  })
})
