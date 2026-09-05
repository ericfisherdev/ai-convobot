import { AttitudeStreamUpdate } from './AttitudeData';

export interface MessageInterface {
    id: number;
    ai: boolean;
    content: string;
    created_at: string;
}

// One Server-Sent Event on `/api/prompt/stream`. Token chunks carry `content`,
// the attitude chunk carries `attitude`, and the final chunk sets
// `is_complete` plus either the sanitized reply or `error`.
export interface StreamChunk {
    request_id: string;
    content: string;
    is_complete: boolean;
    token_count?: number;
    error?: string;
    attitude?: AttitudeStreamUpdate;
}
