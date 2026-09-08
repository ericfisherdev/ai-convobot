import { AttitudeStreamUpdate } from './AttitudeData';

export interface MessageInterface {
    id: number;
    ai: boolean;
    speaker_id: string;
    content: string;
    created_at: string;
}

// What kind of `StreamChunk` this is, mirroring the backend's
// `inference_optimizer::StreamEvent`. `reply_started` opens a new bubble for
// `speaker_id`; `token` appends `content` to the current speaker's bubble
// (or, when `attitude` is set instead, carries the attitude update and no
// content); `reply_complete` replaces the current bubble's content with the
// sanitized `content` and carries `message_id`; `round_complete` and `error`
// are the two terminal events (`is_complete: true`).
export type StreamEvent = 'reply_started' | 'token' | 'reply_complete' | 'round_complete' | 'error';

// One Server-Sent Event on `/api/prompt/stream`, one per speaker action in a
// round. `event` says which kind this is; `speaker_id` is who the chunk is
// about (empty on the attitude chunk and on `round_complete`/`error`, which
// are round-wide rather than per-speaker). `is_complete` is `true` only on
// `round_complete` and `error`.
export interface StreamChunk {
    request_id: string;
    event: StreamEvent;
    content: string;
    is_complete: boolean;
    token_count?: number;
    speaker_id: string;
    message_id?: number;
    error?: string;
    attitude?: AttitudeStreamUpdate;
}
