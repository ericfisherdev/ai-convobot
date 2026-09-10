//! Conversation compaction (#171-#186): checkpoints that roll a run of
//! messages into a summary plus extracted facts, so old turns can drop out
//! of the prompt without the companion losing what happened.
//!
//! `types.rs` (#171) holds the enums and domain structs (`Checkpoint`,
//! `Fact`, `Pin`, ...) kept out of `database.rs` and out of `store.rs` so
//! the pure modules added by later issues can depend on the shapes without
//! pulling in rusqlite-backed code.
//!
//! `store.rs` (#171) is the persistence seam: the `CompactionStore` trait,
//! its production `SqliteCompactionStore` impl (backed by the same
//! `companion_database.db` every other `Database` associated fn uses), and
//! a `#[cfg(test)]` in-memory `RecordingStore` other modules' tests can use
//! too.
//!
//! `trigger.rs` (#172) is pure and I/O-free: `CompactionConfig`,
//! `is_scene_break`, and `should_compact`, which decide *when* a checkpoint
//! draft should be queued.
//!
//! `range.rs` (#172) is pure and I/O-free: `CompactionRange` and
//! `select_range`, which decide *which* messages a queued draft should
//! span.
//!
//! `hook.rs` (#172) is the one impure piece: `after_round`, called from
//! `multiplayer::round::run_round` right after a turn finishes, which reads
//! the uncompacted tail through `crate::chat_turn::TurnStore` and queues a
//! draft when `trigger`/`range` say one is due.

pub mod hook;
pub mod range;
pub mod store;
pub mod trigger;
pub mod types;

use crate::database::Message;

/// One message's identity as the trigger/range logic in [`trigger`] and
/// [`range`] needs it: no content, just enough to sum tokens and locate
/// boundaries. #173's validator reuses this struct for source-id and canon
/// checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MessageRef {
    pub id: i32,
    pub is_human: bool,
    pub tokens: usize,
}

impl From<&Message> for MessageRef {
    fn from(message: &Message) -> Self {
        MessageRef {
            id: message.id,
            // `Message::ai` is already derived from `speaker_id` via
            // `database.rs::is_ai_speaker`, which treats everything except
            // `user` as AI, including `system` notices.
            is_human: !message.ai,
            tokens: crate::context_manager::ContextManager::estimate_tokens(&message.content),
        }
    }
}
