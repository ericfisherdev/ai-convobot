// Pure resolution from a message's `speaker_id` to what `Message.tsx`
// renders: a display name, an avatar, and which of the three message shells
// (`UserMessage`/`AiMessage`/`SystemNotice`) applies. Kept side-effect free
// (no context reads) so it is unit-testable on its own.
import { Participant } from '../interfaces/Participant';
import { RESERVED_SPEAKERS } from '../interfaces/Participant';
import companionAvatarDefault from '../../assets/companion_avatar.jpg';

export type ResolvedSpeakerKind = 'user' | 'bot' | 'system';

export interface ResolvedSpeaker {
  displayName: string;
  avatarUrl: string | null;
  kind: ResolvedSpeakerKind;
}

export interface SpeakerFallbacks {
  userName: string;
  companionName: string;
  /** Empty string (or falsy) falls back to the bundled default avatar. */
  companionAvatarUrl: string;
}

const SYSTEM_DISPLAY_NAME = 'System';

// Resolves `speakerId` against `participants` (the merged local + remote
// list `useParticipants()` exposes). `user`/`char`/`system` are resolved
// from `fallbacks` without a registry lookup, since every registry always
// carries the first two and the third is never a registry member (see
// `ParticipantId::SYSTEM`'s doc comment on the backend). Any other id is
// looked up in `participants`; an id no longer present (a participant who
// has since left) falls back to the raw id and the bundled default avatar,
// so old transcripts still render.
export function resolveSpeaker(
  speakerId: string,
  participants: Participant[],
  fallbacks: SpeakerFallbacks,
): ResolvedSpeaker {
  if (speakerId === RESERVED_SPEAKERS.user) {
    return { displayName: fallbacks.userName, avatarUrl: null, kind: 'user' };
  }

  if (speakerId === RESERVED_SPEAKERS.char) {
    return {
      displayName: fallbacks.companionName,
      avatarUrl: fallbacks.companionAvatarUrl || companionAvatarDefault,
      kind: 'bot',
    };
  }

  if (speakerId === RESERVED_SPEAKERS.system) {
    return { displayName: SYSTEM_DISPLAY_NAME, avatarUrl: null, kind: 'system' };
  }

  const participant = participants.find((p) => p.id === speakerId);
  if (participant) {
    return {
      displayName: participant.display_name,
      avatarUrl: participant.avatar_url || companionAvatarDefault,
      kind: 'bot',
    };
  }

  return { displayName: speakerId, avatarUrl: companionAvatarDefault, kind: 'bot' };
}
