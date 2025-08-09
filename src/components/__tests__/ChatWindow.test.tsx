import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ChatWindow from '../ChatWindow'
import { MessageProvider } from '../context/messageContext'
import { UserDataProvider } from '../context/userContext'
import { CompanionDataProvider } from '../context/companionContext'
import { ConfigProvider } from '../context/configContext'
import { ThemeProvider } from '../theme-provider'

// Mock the contexts with minimal implementations
const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
    <MessageProvider>
      <UserDataProvider>
        <CompanionDataProvider>
          <ConfigProvider>
            {children}
          </ConfigProvider>
        </CompanionDataProvider>
      </UserDataProvider>
    </MessageProvider>
  </ThemeProvider>
)

describe('ChatWindow Component', () => {
  beforeEach(() => {
    // Mock fetch for API calls
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve([]),
      })
    ) as vi.Mock
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
})