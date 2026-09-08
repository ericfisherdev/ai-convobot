import { describe, it, expect } from 'vitest'
import { resolveSpeaker } from '../speakerResolution'
import companionAvatarDefault from '../../../assets/companion_avatar.jpg'
import { Participant } from '../../interfaces/Participant'

const fallbacks = {
  userName: 'Alice',
  companionName: 'Bob',
  companionAvatarUrl: '/companion-avatar.jpg?timestamp=1',
}

const participants: Participant[] = [
  {
    id: 'bot1',
    display_name: 'Ada',
    kind: 'RemoteBot',
    avatar_url: '/api/multiplayer/participants/bot1/avatar',
    connected: true,
  },
]

describe('resolveSpeaker', () => {
  it('resolves the user speaker to the user name with no avatar', () => {
    expect(resolveSpeaker('user', participants, fallbacks)).toEqual({
      displayName: 'Alice',
      avatarUrl: null,
      kind: 'user',
    })
  })

  it('resolves the char speaker to the companion name and avatar', () => {
    expect(resolveSpeaker('char', participants, fallbacks)).toEqual({
      displayName: 'Bob',
      avatarUrl: '/companion-avatar.jpg?timestamp=1',
      kind: 'bot',
    })
  })

  it('falls back to the bundled default avatar when the companion has none', () => {
    expect(resolveSpeaker('char', participants, { ...fallbacks, companionAvatarUrl: '' })).toEqual({
      displayName: 'Bob',
      avatarUrl: companionAvatarDefault,
      kind: 'bot',
    })
  })

  it('resolves the system speaker to a muted notice with no avatar', () => {
    expect(resolveSpeaker('system', participants, fallbacks)).toEqual({
      displayName: 'System',
      avatarUrl: null,
      kind: 'system',
    })
  })

  it('resolves a registered joiner bot to its registry entry', () => {
    expect(resolveSpeaker('bot1', participants, fallbacks)).toEqual({
      displayName: 'Ada',
      avatarUrl: '/api/multiplayer/participants/bot1/avatar',
      kind: 'bot',
    })
  })

  it('falls back to the raw id and the bundled avatar for a participant who has left', () => {
    expect(resolveSpeaker('bot2', participants, fallbacks)).toEqual({
      displayName: 'bot2',
      avatarUrl: companionAvatarDefault,
      kind: 'bot',
    })
  })
})
