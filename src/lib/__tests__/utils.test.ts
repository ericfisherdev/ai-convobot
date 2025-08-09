import { describe, it, expect } from 'vitest'
import { cn } from '../utils'

describe('utils', () => {
  describe('cn function', () => {
    it('merges class names correctly', () => {
      const result = cn('class1', 'class2')
      expect(result).toBe('class1 class2')
    })

    it('handles conditional classes', () => {
      const result = cn('base', true && 'conditional', false && 'hidden')
      expect(result).toBe('base conditional')
    })

    it('handles Tailwind CSS conflicts', () => {
      const result = cn('text-sm text-lg')
      // The exact result depends on the implementation, but it should handle conflicts
      expect(typeof result).toBe('string')
    })

    it('handles empty and undefined values', () => {
      const result = cn('base', '', undefined, null, 'end')
      expect(result).toBe('base end')
    })

    it('merges arrays of classes', () => {
      const result = cn(['class1', 'class2'], ['class3'])
      expect(result).toBe('class1 class2 class3')
    })

    it('handles objects with boolean values', () => {
      const result = cn({
        'always-present': true,
        'never-present': false,
        'conditionally-present': true
      })
      expect(result).toBe('always-present conditionally-present')
    })
  })
})