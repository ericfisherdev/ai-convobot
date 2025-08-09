import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { renderHook } from '@testing-library/react'
import { useMobile } from '../useMobile'

// Mock window.matchMedia
const mockMatchMedia = vi.fn()

describe('useMobile Hook', () => {
  beforeEach(() => {
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: mockMatchMedia,
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('returns false for desktop viewport', () => {
    mockMatchMedia.mockImplementation((query) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }))

    const { result } = renderHook(() => useMobile())
    expect(result.current).toBe(false)
  })

  it('returns true for mobile viewport', () => {
    mockMatchMedia.mockImplementation((query) => ({
      matches: true,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }))

    const { result } = renderHook(() => useMobile())
    expect(result.current).toBe(true)
  })

  it('calls matchMedia with correct query', () => {
    mockMatchMedia.mockImplementation((query) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }))

    renderHook(() => useMobile())
    expect(mockMatchMedia).toHaveBeenCalledWith('(max-width: 768px)')
  })
})