import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import EditData from '../editData/EditData'
import { UserDataProvider } from '../context/userContext'
import { CompanionDataProvider } from '../context/companionContext'
import { ConfigProvider } from '../context/configContext'

const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <UserDataProvider>
    <CompanionDataProvider>
      <ConfigProvider>
        {children}
      </ConfigProvider>
    </CompanionDataProvider>
  </UserDataProvider>
)

describe('EditData Component', () => {
  beforeEach(() => {
    // Mock fetch for API calls
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({}),
      })
    ) as vi.Mock
  })

  it('renders edit data tabs', () => {
    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )
    
    // Check for tab navigation
    expect(screen.getByRole('tablist')).toBeInTheDocument()
  })

  it('shows user data tab', () => {
    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )
    
    // Look for user-related form elements
    const nameInput = screen.getByLabelText(/name/i) || screen.getByPlaceholderText(/name/i)
    expect(nameInput).toBeInTheDocument()
  })

  it('shows companion data tab', async () => {
    const user = userEvent.setup()
    
    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )
    
    // Click on companion tab if it exists
    const companionTab = screen.getByText(/companion/i)
    if (companionTab) {
      await user.click(companionTab)
    }
    
    // Check for companion-related elements
    expect(screen.getByText(/companion/i)).toBeInTheDocument()
  })

  it('shows config data tab', async () => {
    const user = userEvent.setup()
    
    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )
    
    // Click on config tab if it exists
    const configTab = screen.getByText(/config/i)
    if (configTab) {
      await user.click(configTab)
    }
    
    // Check for config-related elements
    expect(screen.getByText(/config/i)).toBeInTheDocument()
  })

  it('handles form submission', async () => {
    const user = userEvent.setup()
    
    render(
      <MockProviders>
        <EditData />
      </MockProviders>
    )
    
    // Find and interact with form elements
    const saveButton = screen.getByRole('button', { name: /save/i }) || screen.getByRole('button', { name: /update/i })
    if (saveButton) {
      await user.click(saveButton)
      
      // Verify that fetch was called for saving
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalled()
      })
    }
  })
})