import { describe, it, expect } from 'vitest'
import { isBotSpeaker, USER_SPEAKER_ID, SYSTEM_SPEAKER_ID } from '../speakers'

describe('speakers', () => {
  describe('isBotSpeaker', () => {
    it('is true for the host companion', () => {
      expect(isBotSpeaker({ speaker_id: 'char' })).toBe(true)
    })

    it('is true for a joined remote bot', () => {
      expect(isBotSpeaker({ speaker_id: 'bot1' })).toBe(true)
    })

    it('is false for the user', () => {
      expect(isBotSpeaker({ speaker_id: USER_SPEAKER_ID })).toBe(false)
    })

    it('is false for a system notice', () => {
      expect(isBotSpeaker({ speaker_id: SYSTEM_SPEAKER_ID })).toBe(false)
    })
  })
})
