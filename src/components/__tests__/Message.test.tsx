import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import MessageComponent from '../message/Message'
import { MessageProvider } from '../context/messageContext'
import { UserDataProvider } from '../context/userContext'
import { CompanionDataProvider } from '../context/companionContext'

const mockMessage = {
  id: 1,
  ai: false,
  content: 'Hello, this is a test message',
  created_at: '2024-01-15 10:30'
}

const mockAIMessage = {
  id: 2,
  ai: true,
  content: 'Hello! How can I help you today?',
  created_at: '2024-01-15 10:31'
}

const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <MessageProvider>
    <UserDataProvider>
      <CompanionDataProvider>
        {children}
      </CompanionDataProvider>
    </UserDataProvider>
  </MessageProvider>
)

describe('Message Component', () => {
  it('renders user message correctly', () => {
    render(
      <MockProviders>
        <MessageComponent 
          message={mockMessage}
          isEditing={false}
          onEdit={() => {}}
          onDelete={() => {}}
          onSave={() => {}}
          onCancel={() => {}}
        />
      </MockProviders>
    )
    
    expect(screen.getByText('Hello, this is a test message')).toBeInTheDocument()
    expect(screen.getByText('2024-01-15 10:30')).toBeInTheDocument()
  })

  it('renders AI message correctly', () => {
    render(
      <MockProviders>
        <MessageComponent 
          message={mockAIMessage}
          isEditing={false}
          onEdit={() => {}}
          onDelete={() => {}}
          onSave={() => {}}
          onCancel={() => {}}
        />
      </MockProviders>
    )
    
    expect(screen.getByText('Hello! How can I help you today?')).toBeInTheDocument()
  })

  it('displays edit and delete buttons for messages', () => {
    const onEdit = vi.fn()
    const onDelete = vi.fn()
    
    render(
      <MockProviders>
        <MessageComponent 
          message={mockMessage}
          isEditing={false}
          onEdit={onEdit}
          onDelete={onDelete}
          onSave={() => {}}
          onCancel={() => {}}
        />
      </MockProviders>
    )
    
    // Look for edit and delete buttons (they might be icon buttons)
    const buttons = screen.getAllByRole('button')
    expect(buttons.length).toBeGreaterThanOrEqual(2)
  })

  it('shows markdown content correctly', () => {
    const markdownMessage = {
      id: 3,
      ai: true,
      content: '**Bold text** and *italic text*',
      created_at: '2024-01-15 10:32'
    }
    
    render(
      <MockProviders>
        <MessageComponent 
          message={markdownMessage}
          isEditing={false}
          onEdit={() => {}}
          onDelete={() => {}}
          onSave={() => {}}
          onCancel={() => {}}
        />
      </MockProviders>
    )
    
    expect(screen.getByText('Bold text')).toBeInTheDocument()
    expect(screen.getByText('italic text')).toBeInTheDocument()
  })
})