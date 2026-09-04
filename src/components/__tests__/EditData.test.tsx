import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EditData } from '../editData/EditData'
import { UserDataProvider } from '../context/userContext'
import { CompanionDataProvider } from '../context/companionContext'
import { ConfigProvider } from '../context/configContext'
import { MessagesProvider } from '../context/messageContext'

const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <UserDataProvider>
    <CompanionDataProvider>
      <ConfigProvider>
        <MessagesProvider>
          {children}
        </MessagesProvider>
      </ConfigProvider>
    </CompanionDataProvider>
  </UserDataProvider>
)

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
    expect(tabs).toEqual(['Companion', 'User', 'Attitudes', 'Theme', 'Config'])
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
})
