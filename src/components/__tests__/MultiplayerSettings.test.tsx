import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MultiplayerSettings } from '../editData/MultiplayerSettings'
import { ConfigInterface, Device, MultiplayerMode, PromptTemplate } from '../interfaces/Config'

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
}

describe('MultiplayerSettings', () => {
  it('shows only the mode description in solo mode', () => {
    render(<MultiplayerSettings config={baseConfig} onChange={vi.fn()} />)

    expect(screen.queryByLabelText('Host address')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Participant ID')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Password')).not.toBeInTheDocument()
  })

  it('shows host address and participant ID inputs in joiner mode', () => {
    render(
      <MultiplayerSettings
        config={{ ...baseConfig, multiplayer_mode: MultiplayerMode.Joiner }}
        onChange={vi.fn()}
      />
    )

    expect(screen.getByLabelText('Host address')).toBeInTheDocument()
    expect(screen.getByLabelText('Participant ID')).toBeInTheDocument()
    expect(screen.getByLabelText('Password')).toBeInTheDocument()
  })

  it('hides host address and participant ID inputs in solo mode', () => {
    render(
      <MultiplayerSettings
        config={{ ...baseConfig, multiplayer_mode: MultiplayerMode.Solo }}
        onChange={vi.fn()}
      />
    )

    expect(screen.queryByLabelText('Host address')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Participant ID')).not.toBeInTheDocument()
  })

  it('shows the password field with the "leave blank" placeholder when a password is already set', () => {
    render(
      <MultiplayerSettings
        config={{ ...baseConfig, multiplayer_mode: MultiplayerMode.Host, multiplayer_password_set: true }}
        onChange={vi.fn()}
      />
    )

    expect(
      screen.getByPlaceholderText('Password is set. Leave blank to keep it')
    ).toBeInTheDocument()
  })

  it('shows the required placeholder in host mode when no password is stored yet', () => {
    render(
      <MultiplayerSettings
        config={{ ...baseConfig, multiplayer_mode: MultiplayerMode.Host, multiplayer_password_set: false }}
        onChange={vi.fn()}
      />
    )

    expect(screen.getByPlaceholderText('Required for host mode')).toBeInTheDocument()
  })
})
