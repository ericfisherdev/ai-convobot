// Mirrors `backend/src/multiplayer/protocol.rs::ParticipantSummary`, the row
// shape `GET /api/multiplayer/participants` and the joiner's
// `GET /api/multiplayer/status` (its `participants` field) both return.
// `ParticipantKind` has no `#[serde(rename_all)]` on the backend enum, so
// the wire values are the exact Rust variant names, not snake_case.
export type ParticipantKind = 'Human' | 'HostBot' | 'RemoteBot';

export interface Participant {
  id: string; // ParticipantId: ^[a-z][a-z0-9_]{0,15}$
  display_name: string;
  kind: ParticipantKind;
  avatar_url: string | null;
  connected: boolean;
}

// Mirrors `backend/src/multiplayer/joiner.rs::JoinerState`, tagged
// `#[serde(tag = "state", rename_all = "snake_case")]`. `null` in solo and
// host mode.
export type MultiplayerConnectionState = 'disconnected' | 'connecting' | 'connected' | 'rejected';

// The body of `GET /api/multiplayer/status`. `solo`/`host` mode returns just
// `{ mode, state: null }`; `joiner` mode flattens `JoinerState` in alongside
// `attempts`, `host_address`, `participant_id` and `participants`.
export interface MultiplayerStatus {
  mode: 'solo' | 'host' | 'joiner';
  state: MultiplayerConnectionState | null;
  reason?: string; // present only when state is "rejected"
  last_error?: string | null; // present only when state is "disconnected"
  attempts?: number;
  host_address?: string;
  participant_id?: string;
  participants?: Participant[]; // joiner mode only
}

// The three speaker ids every chat carries regardless of multiplayer mode,
// mirroring `backend/src/database.rs`'s `USER_SPEAKER_ID`/`CHAR_SPEAKER_ID`/
// `SYSTEM_SPEAKER_ID`.
export const RESERVED_SPEAKERS = { user: 'user', char: 'char', system: 'system' } as const;
