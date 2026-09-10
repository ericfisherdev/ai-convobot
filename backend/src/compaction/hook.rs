//! The hook that runs at the end of every finished round (#172):
//! [`after_round`] reads the uncompacted tail through
//! [`crate::chat_turn::TurnStore`], decides whether to queue a new
//! checkpoint draft (`trigger`/`range`), and if so, queues it.
//!
//! Called from `multiplayer::round::run_round`, right after
//! `PendingTurn::finish`, never from `TurnStore::finish_turn` itself: a
//! joiner calls `finish_turn` directly (`remote_generation.rs`) with an
//! empty messages table, so compaction only ever runs on the host/solo path
//! that actually owns the conversation.
//!
//! A compaction failure must never fail the chat reply, so every store
//! error here is logged and swallowed, same as `chat_turn::finish_turn`.

use crate::compaction::range::{select_range, CompactionRange};
use crate::compaction::store::{CompactionStore, SqliteCompactionStore};
use crate::compaction::trigger::{should_compact, CompactionConfig};
use crate::compaction::types::{CompactionTrigger, NewDraft};
use crate::compaction::MessageRef;
use crate::database::Database;

/// What [`crate::chat_turn::TurnStore::compaction_tail`] reads once per
/// round: everything [`should_compact`] and [`select_range`] need, read in
/// one shot right after the round's inserts.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionTailView {
    pub compacted_through: Option<i32>,
    pub messages: Vec<MessageRef>,
    /// The last human message's content in `messages`, or empty when the
    /// tail has no human turn. Kept separate from `MessageRef` (which has
    /// no content) rather than growing it, to keep that struct small.
    pub last_user_turn: String,
    pub short_term_mem: usize,
    pub draft_pending: bool,
    pub config: CompactionConfig,
}

/// A checkpoint draft [`after_round`] just queued.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedDraft {
    pub draft_id: i64,
    pub range: CompactionRange,
    pub trigger: CompactionTrigger,
}

/// Runs the trigger/range decision for `companion_id` and queues a draft
/// when one is due. `None` on "nothing to do" (no trigger fired) and on any
/// store failure (logged internally).
pub fn after_round(
    store: &impl crate::chat_turn::TurnStore,
    companion_id: i32,
) -> Option<QueuedDraft> {
    let tail = match store.compaction_tail(companion_id) {
        Ok(tail) => tail,
        Err(e) => {
            eprintln!(
                "compaction: failed to read the uncompacted tail for companion {}: {}",
                companion_id, e
            );
            return None;
        }
    };

    let tail_tokens: usize = tail.messages.iter().map(|m| m.tokens).sum();
    let tail_messages_before_break = tail.messages.iter().rposition(|m| m.is_human).unwrap_or(0);

    let trigger = should_compact(
        tail_tokens,
        tail_messages_before_break,
        &tail.last_user_turn,
        tail.draft_pending,
        &tail.config,
    )?;

    let range = select_range(
        tail.compacted_through,
        &tail.messages,
        tail.short_term_mem,
        tail.config.min_messages,
        trigger,
    )?;

    match store.queue_compaction_draft(companion_id, range, trigger) {
        Ok(draft_id) => {
            println!(
                "Compaction draft queued ({:?}): messages #{}..#{}",
                trigger, range.from_id, range.through_id
            );
            Some(QueuedDraft {
                draft_id,
                range,
                trigger,
            })
        }
        Err(e) => {
            eprintln!(
                "compaction: failed to queue a draft for companion {}: {}",
                companion_id, e
            );
            None
        }
    }
}

/// Production [`CompactionTailView`] reader, shared by
/// [`crate::chat_turn::SqliteTurnStore::compaction_tail`] so that impl stays
/// a one-liner like every other `TurnStore` method.
pub(crate) fn compaction_tail_on(companion_id: i32) -> rusqlite::Result<CompactionTailView> {
    let compaction_store = SqliteCompactionStore;
    let compacted_through = compaction_store.compacted_through(companion_id)?;
    let companion = Database::get_companion_data()?;
    let rows = Database::get_messages_after(compacted_through.unwrap_or(0))?;
    let last_user_turn = rows
        .iter()
        .rev()
        .find(|m| !m.ai)
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let messages = rows.iter().map(MessageRef::from).collect();
    let draft_pending = compaction_store.pending_draft(companion_id)?.is_some();
    let config = CompactionConfig::from_config(&Database::get_config()?);

    Ok(CompactionTailView {
        compacted_through,
        messages,
        last_user_turn,
        short_term_mem: companion.short_term_mem,
        draft_pending,
        config,
    })
}

/// Production draft-queueing, shared by
/// [`crate::chat_turn::SqliteTurnStore::queue_compaction_draft`].
pub(crate) fn queue_compaction_draft_on(
    companion_id: i32,
    range: CompactionRange,
    trigger: CompactionTrigger,
) -> rusqlite::Result<i64> {
    SqliteCompactionStore.insert_draft(NewDraft {
        companion_id,
        from_message_id: range.from_id,
        through_message_id: range.through_id,
        trigger,
        raw_model_output: None,
    })
}
