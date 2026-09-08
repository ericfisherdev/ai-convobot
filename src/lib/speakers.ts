import { MessageInterface } from '../components/interfaces/Message';

// Mirrors the backend's reserved speaker ids (`USER_SPEAKER_ID`/
// `SYSTEM_SPEAKER_ID`, `backend/src/database.rs`). `char` (the host
// companion) and any joined bot id (`bot1`, ...) are the only other values
// `speaker_id` ever takes, so "not user, not system" is exactly "a bot".
export const USER_SPEAKER_ID = 'user';
export const SYSTEM_SPEAKER_ID = 'system';

// Whether `message` was sent by a bot (the host companion or a joined remote
// bot), as opposed to the user or a system notice. The Regenerate control
// (`MessageScroll.tsx`, `VirtualMessageList.tsx`) is gated on this rather
// than on `message.ai`: #125 derives `ai` as `speaker_id != 'user'`, which a
// trailing system notice would also satisfy, wrongly showing Regenerate on
// it.
export function isBotSpeaker(message: Pick<MessageInterface, 'speaker_id'>): boolean {
  return message.speaker_id !== USER_SPEAKER_ID && message.speaker_id !== SYSTEM_SPEAKER_ID;
}
