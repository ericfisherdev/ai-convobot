import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
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

  it('renders the regenerate control on a trailing AI message', async () => {
    render(
      <MockProviders>
        <Message received={true} regenerate={true} id={4} content="Hello! How can I help you today?" created_at="2024-01-15 10:33" />
      </MockProviders>
    )

    await screen.findByText('Hello! How can I help you today?')
    expect(screen.getByRole('button', { name: 'Regenerate message' })).toBeInTheDocument()
  })

  it('never renders the regenerate control on a user message', async () => {
    render(
      <MockProviders>
        <Message received={false} regenerate={true} id={5} content="Hello, this is a test message" created_at="2024-01-15 10:34" />
      </MockProviders>
    )

    await screen.findByText('Hello, this is a test message')
    expect(screen.queryByRole('button', { name: 'Regenerate message' })).not.toBeInTheDocument()
  })

  it('editing an AI message sends only the new content', async () => {
    const user = userEvent.setup()

    render(
      <MockProviders>
        <Message received={true} regenerate={false} id={2} content="Hello! How can I help you today?" created_at="2024-01-15 10:31" />
      </MockProviders>
    )

    await screen.findByText('Hello! How can I help you today?')
    await user.click(screen.getByRole('button', { name: 'Edit message' }))
    const textbox = screen.getByRole('textbox')
    await user.clear(textbox)
    await user.type(textbox, 'edited ai reply')
    await user.click(screen.getByRole('button', { name: 'Save message' }))

    const mockFetch = fetch as unknown as ReturnType<typeof vi.fn>
    const putCalls = mockFetch.mock.calls.filter(call => call[1]?.method === 'PUT')
    const lastPutCall = putCalls[putCalls.length - 1]
    expect(lastPutCall[0]).toBe('/api/message/2')
    expect(JSON.parse(lastPutCall[1].body)).toEqual({ content: 'edited ai reply' })
  })

  it('editing a user message sends only the new content', async () => {
    const user = userEvent.setup()

    render(
      <MockProviders>
        <Message received={false} regenerate={false} id={1} content="Hello, this is a test message" created_at="2024-01-15 10:30" />
      </MockProviders>
    )

    await screen.findByText('Hello, this is a test message')
    await user.click(screen.getByRole('button', { name: 'Edit message' }))
    const textbox = screen.getByRole('textbox')
    await user.clear(textbox)
    await user.type(textbox, 'edited user message')
    await user.click(screen.getByRole('button', { name: 'Save message' }))

    const mockFetch = fetch as unknown as ReturnType<typeof vi.fn>
    const putCalls = mockFetch.mock.calls.filter(call => call[1]?.method === 'PUT')
    const lastPutCall = putCalls[putCalls.length - 1]
    expect(lastPutCall[0]).toBe('/api/message/1')
    expect(JSON.parse(lastPutCall[1].body)).toEqual({ content: 'edited user message' })
  })
})
