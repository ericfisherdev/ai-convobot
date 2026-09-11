// Mirrors the backend's `running_thoughts::types::RunningThought` (#215/#216).
// #217/#218 extend this file rather than creating their own.
export interface RunningThought {
    id: number;
    companion_id: number;
    speaker_id: string;
    from_message_id: number;
    through_message_id: number;
    text: string;
    // `true` once the user has rewritten `text` (#217's PATCH).
    edited: boolean;
    created_at: string;
}
