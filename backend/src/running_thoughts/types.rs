//! Domain structs for running thoughts (#215), kept out of `store.rs` so a
//! later pure module (#216's generation, if it needs one) can depend on the
//! shape without pulling in rusqlite-backed code, matching
//! `compaction::types`.

use serde::{Deserialize, Serialize};

/// One `running_thoughts` row.
///
/// `speaker_id` is the author's `ParticipantId` as a string (matches
/// `Message::speaker_id`; `ParticipantId` has no `FromSql`/`ToSql`, so the
/// store stays string-typed and #216 converts at the boundary with
/// `ParticipantId::as_str()`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunningThought {
    pub id: i64,
    pub companion_id: i32,
    pub speaker_id: String,
    pub from_message_id: i32,
    pub through_message_id: i32,
    pub text: String,
    /// `true` once the user has rewritten `text` (#217's PATCH); #216 passes
    /// such notes to the generator as the user's words, not the model's.
    pub edited: bool,
    pub created_at: String,
}

/// What [`crate::running_thoughts::store::RunningThoughtStore::insert`]
/// takes. `created_at` is not part of this shape: it is always
/// `crate::database::get_current_date()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewRunningThought {
    pub companion_id: i32,
    pub speaker_id: String,
    pub from_message_id: i32,
    pub through_message_id: i32,
    pub text: String,
    /// Normally `false` (a freshly generated note). #217's regenerate
    /// re-inserts the originals it did not reach on failure, so the flag
    /// must be settable on insert to keep the user's edits marked as such.
    pub edited: bool,
}
