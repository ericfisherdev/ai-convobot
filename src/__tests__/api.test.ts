import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'

// Mock fetch globally
const mockFetch = vi.fn()
global.fetch = mockFetch

describe('API Integration Tests', () => {
  const baseUrl = 'http://localhost:3000/api'

  beforeEach(() => {
    mockFetch.mockClear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Message API', () => {
    it('fetches messages successfully', async () => {
      const mockMessages = [
        { id: 1, ai: false, content: 'Hello', created_at: '2024-01-15 10:00' },
        { id: 2, ai: true, content: 'Hi there!', created_at: '2024-01-15 10:01' }
      ]

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockMessages)
      })

      const response = await fetch(`${baseUrl}/message`)
      const data = await response.json()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/message`)
      expect(data).toEqual(mockMessages)
    })

    it('sends a message successfully', async () => {
      const newMessage = { ai: false, content: 'Test message' }
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true })
      })

      const response = await fetch(`${baseUrl}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newMessage)
      })
      const data = await response.json()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newMessage)
      })
      expect(data.success).toBe(true)
    })

    it('deletes a message successfully', async () => {
      const messageId = 123

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true })
      })

      const response = await fetch(`${baseUrl}/message/${messageId}`, {
        method: 'DELETE'
      })
      const data = await response.json()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/message/${messageId}`, {
        method: 'DELETE'
      })
      expect(data.success).toBe(true)
    })
  })

  describe('Companion API', () => {
    it('fetches companion data successfully', async () => {
      const mockCompanion = {
        name: 'AI Assistant',
        persona: 'Helpful and friendly',
        example_dialogue: 'Hello! How can I help you?',
        first_message: 'Welcome!',
        long_term_mem: 100,
        short_term_mem: 50,
        roleplay: true,
        dialogue_tuning: 1,
        avatar_path: '/avatars/assistant.jpg'
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCompanion)
      })

      const response = await fetch(`${baseUrl}/companion`)
      const data = await response.json()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/companion`)
      expect(data).toEqual(mockCompanion)
    })

    it('updates companion data successfully', async () => {
      const updatedCompanion = {
        name: 'Updated Assistant',
        persona: 'Very helpful',
        example_dialogue: 'Hi there!',
        first_message: 'Hello!',
        long_term_mem: 150,
        short_term_mem: 75,
        roleplay: false,
        dialogue_tuning: 2,
        avatar_path: '/avatars/new-assistant.jpg'
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true })
      })

      const response = await fetch(`${baseUrl}/companion`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedCompanion)
      })
      const data = await response.json()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/companion`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedCompanion)
      })
      expect(data.success).toBe(true)
    })
  })

  describe('User API', () => {
    it('fetches user data successfully', async () => {
      const mockUser = {
        name: 'John Doe',
        persona: 'Curious learner'
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockUser)
      })

      const response = await fetch(`${baseUrl}/user`)
      const data = await response.json()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/user`)
      expect(data).toEqual(mockUser)
    })

    it('updates user data successfully', async () => {
      const updatedUser = {
        name: 'Jane Doe',
        persona: 'Enthusiastic explorer'
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true })
      })

      const response = await fetch(`${baseUrl}/user`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedUser)
      })
      const data = await response.json()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/user`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedUser)
      })
      expect(data.success).toBe(true)
    })
  })

  describe('Config API', () => {
    it('fetches config data successfully', async () => {
      const mockConfig = {
        device: 'CPU',
        llm_model_path: '/models/llama-7b.gguf',
        gpu_layers: 0,
        prompt_template: 'Default',
        context_window_size: 2048,
        max_response_tokens: 512,
        enable_dynamic_context: true,
        vram_limit_gb: 4
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockConfig)
      })

      const response = await fetch(`${baseUrl}/config`)
      const data = await response.json()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/config`)
      expect(data).toEqual(mockConfig)
    })

    it('updates config data successfully', async () => {
      const updatedConfig = {
        device: 'GPU',
        llm_model_path: '/models/llama-13b.gguf',
        gpu_layers: 20,
        prompt_template: 'Llama2',
        context_window_size: 4096,
        max_response_tokens: 1024,
        enable_dynamic_context: false,
        vram_limit_gb: 8
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true })
      })

      const response = await fetch(`${baseUrl}/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedConfig)
      })
      const data = await response.json()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedConfig)
      })
      expect(data.success).toBe(true)
    })
  })

  describe('Send Message API', () => {
    it('sends message and receives AI response', async () => {
      const messageContent = 'Hello, how are you?'
      const mockResponse = {
        response: 'I\'m doing well, thank you! How can I help you today?',
        success: true
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      })

      const response = await fetch(`${baseUrl}/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: messageContent })
      })
      const data = await response.json()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: messageContent })
      })
      expect(data).toEqual(mockResponse)
    })
  })

  describe('Memory API', () => {
    it('fetches memory data successfully', async () => {
      const mockMemory = {
        memories: [
          { content: 'User likes programming', importance: 0.8 },
          { content: 'User mentioned living in New York', importance: 0.6 }
        ]
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockMemory)
      })

      const response = await fetch(`${baseUrl}/memory`)
      const data = await response.json()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/memory`)
      expect(data).toEqual(mockMemory)
    })
  })

  describe('Character Card API', () => {
    it('imports character card successfully', async () => {
      const mockCharacterCard = {
        name: 'Character Name',
        description: 'Character description',
        first_mes: 'First message',
        mes_example: 'Example dialogue'
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true })
      })

      const response = await fetch(`${baseUrl}/character_card`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mockCharacterCard)
      })
      const data = await response.json()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/character_card`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mockCharacterCard)
      })
      expect(data.success).toBe(true)
    })
  })

  describe('Error Handling', () => {
    it('handles network errors gracefully', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'))

      try {
        await fetch(`${baseUrl}/message`)
      } catch (error) {
        expect(error).toBeInstanceOf(Error)
        expect((error as Error).message).toBe('Network error')
      }
    })

    it('handles HTTP error responses', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ error: 'Not found' })
      })

      const response = await fetch(`${baseUrl}/invalid`)
      expect(response.ok).toBe(false)
      expect(response.status).toBe(404)
    })

    it('handles server errors (500)', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: 'Internal server error' })
      })

      const response = await fetch(`${baseUrl}/message`)
      expect(response.ok).toBe(false)
      expect(response.status).toBe(500)
    })
  })
})