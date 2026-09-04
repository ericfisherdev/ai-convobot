import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Message } from '../message/Message'
import { MessagesProvider } from '../context/messageContext'
import { UserDataProvider } from '../context/userContext'
import { CompanionDataProvider } from '../context/companionContext'
import { formatMessageDate } from '../../lib/utils'

const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <MessagesProvider>
    <UserDataProvider>
      <CompanionDataProvider>
        {children}
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
        <Message received={false} regenerate={false} id={1} content="Hello, this is a test message" created_at="2024-01-15 10:30" />
      </MockProviders>
    )

    expect(await screen.findByText('Hello, this is a test message')).toBeInTheDocument()
    expect(screen.getByText(formatMessageDate('2024-01-15 10:30'))).toBeInTheDocument()
  })

  it('renders AI message correctly', async () => {
    render(
      <MockProviders>
        <Message received={true} regenerate={false} id={2} content="Hello! How can I help you today?" created_at="2024-01-15 10:31" />
      </MockProviders>
    )

    expect(await screen.findByText('Hello! How can I help you today?')).toBeInTheDocument()
  })

  it('displays edit and delete buttons for messages', async () => {
    render(
      <MockProviders>
        <Message received={false} regenerate={false} id={1} content="Hello, this is a test message" created_at="2024-01-15 10:30" />
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
        <Message received={true} regenerate={false} id={3} content="**Bold text** and *italic text*" created_at="2024-01-15 10:32" />
      </MockProviders>
    )

    expect(await screen.findByText('Bold text')).toBeInTheDocument()
    expect(screen.getByText('italic text')).toBeInTheDocument()
  })
})
