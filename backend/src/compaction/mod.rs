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
//! span. Both work over [`MessageRef`], the lightweight `id`/`is_human`/
//! `tokens` view of the *uncompacted tail* they scan every round.
//!
//! `hook.rs` (#172) is the one impure piece: `after_round`, called from
//! `multiplayer::round::run_round` right after a turn finishes, which reads
//! the uncompacted tail through `crate::chat_turn::TurnStore` and queues a
//! draft when `trigger`/`range` say one is due.
//!
//! `extract.rs` (#173) is the pure half of the extraction pass: `serde`
//! structs mirroring the model's JSON output schema, `parse_extraction`, and
//! `to_fact_drafts`, which maps a parsed [`extract::ExtractionOutput`] to
//! [`types::FactDraft`]s. No model, no grammar, no chunking — those are
//! #185's `Extractor`-backed half, built on these same types.
//!
//! `validate.rs` (#173) is the canon validator: `validate` runs every draft
//! through five rejection rules (missing/out-of-range sources, the canon
//! rule, verbatim quotes, length, duplicates) plus a `replaces`-filtering
//! step, never dropping an item — a rejected draft keeps its
//! [`validate::RejectReason`] so #180's review card can show why. Pure, no
//! `Database`, no I/O. It works over [`CitedMessage`], not [`MessageRef`]:
//! the verbatim-quote and per-speaker canon checks need `content` and
//! `speaker_id`, which `MessageRef` deliberately omits to stay cheap on
//! `range.rs`'s every-round tail scan. `validate`/`extract` only ever see
//! the small, already-selected checkpoint range (`Database::
//! get_messages_between`), so carrying full content there costs nothing.
//!
//! [`SpeakerInfo`] is shared by `validate.rs` and `extract.rs`: what the
//! extraction pass needs to know about a speaker, independent of either
//! message view above.
//!
//! `context.rs` (#174) is the pure, SQLite-free input to rendering:
//! `CompactionContext` buckets active facts by category, plus the latest
//! committed checkpoint's summaries and pinned messages, built either
//! directly (`from_facts`, for tests and #182's joiner) or from the store
//! (`load`). `render.rs` (#174) turns a `CompactionContext` into the
//! `RenderedBlocks` `llm.rs::build_base_components` splices into the
//! system prompt, re-splitting the compaction token slice when overlays,
//! rules, summaries and pins do not all fit.
#![allow(dead_code)]

pub mod context;
pub mod extract;
pub mod hook;
pub mod range;
pub mod render;
pub mod store;
pub mod trigger;
pub mod types;
pub mod validate;

use serde::Deserialize;

use crate::database::{self, Message};

/// One message's identity as the trigger/range logic in [`trigger`] and
/// [`range`] needs it: no content, just enough to sum tokens and locate
/// boundaries. Deliberately excludes `content`/`speaker_id` to stay cheap on
/// `range.rs`'s every-round scan of the uncompacted tail; #173's
/// `extract`/`validate` need those and use [`CitedMessage`] instead.
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

/// A message as `extract.rs`/`validate.rs` need it: full `speaker_id` (for
/// the per-speaker canon predicate) and `content` (for the verbatim-quote
/// check). Only ever built over one checkpoint's already-selected range
/// (`Database::get_messages_between`), never the whole uncompacted tail, so
/// carrying full content is cheap here even though [`MessageRef`]
/// deliberately avoids it. `Deserialize` is only used by the
/// `#[cfg(test)] fixtures` module below, to load `synthetic_range.json`
/// directly into this shape.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct CitedMessage {
    pub id: i32,
    pub speaker_id: String,
    pub content: String,
}

impl From<&Message> for CitedMessage {
    fn from(message: &Message) -> Self {
        CitedMessage {
            id: message.id,
            speaker_id: message.speaker_id.clone(),
            content: message.content.clone(),
        }
    }
}

/// What the extraction pass needs to know about a speaker: a display name
/// for prompts/rendering, and whether the speaker's turns count as canon.
pub trait SpeakerInfo {
    fn display_name(&self, speaker_id: &str) -> String;
    fn is_canon(&self, speaker_id: &str) -> bool;
}

/// [`SpeakerInfo`] for the solo (non-multiplayer) case: exactly two
/// speakers, `user` and `char`. #182 supplies a registry-backed impl for
/// multiplayer (only `ParticipantKind::Human` is canon; `system` never is).
pub struct SoloSpeakers {
    pub user_name: String,
    pub companion_name: String,
}

impl SpeakerInfo for SoloSpeakers {
    fn display_name(&self, speaker_id: &str) -> String {
        if speaker_id == database::USER_SPEAKER_ID {
            self.user_name.clone()
        } else {
            self.companion_name.clone()
        }
    }

    fn is_canon(&self, speaker_id: &str) -> bool {
        speaker_id == database::USER_SPEAKER_ID
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_message(id: i32, speaker_id: &str, content: &str) -> Message {
        Message {
            id,
            ai: speaker_id != database::USER_SPEAKER_ID,
            speaker_id: speaker_id.to_string(),
            content: content.to_string(),
            created_at: "now".to_string(),
        }
    }

    #[test]
    fn cited_message_from_message_carries_id_speaker_and_content_but_not_created_at() {
        let message = a_message(46, "user", "hello there");
        let cited = CitedMessage::from(&message);
        assert_eq!(cited.id, 46);
        assert_eq!(cited.speaker_id, "user");
        assert_eq!(cited.content, "hello there");
    }

    #[test]
    fn solo_speakers_maps_user_and_char_to_the_two_names() {
        let speakers = SoloSpeakers {
            user_name: "Eric".to_string(),
            companion_name: "Vi".to_string(),
        };
        assert_eq!(speakers.display_name("user"), "Eric");
        assert_eq!(speakers.display_name("char"), "Vi");
        assert!(speakers.is_canon("user"));
        assert!(!speakers.is_canon("char"));
    }
}

/// Synthetic fixtures shared by every compaction test in this crate
/// (`validate.rs`, `extract.rs`, and #175/#180's tests once they land):
/// `synthetic_range` is a 20-message two-character scene plus a
/// multiplayer-style third speaker; `bad_draft` is an `ExtractionOutput`
/// over that range with four deliberate defects, one per rejection rule
/// `validate.rs` doesn't already cover with a synthetic example. No real
/// chat text.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::CitedMessage;
    use crate::compaction::extract::{parse_extraction, ExtractionOutput};

    /// Ids 46-65 (non-zero-based on purpose): a human turn stating a world
    /// fact (46, 48), a bot turn inventing a world fact (49), a bot turn
    /// introducing a person from a pronoun with no name given (51-52, "Wren"
    /// from "He"), a human line later used as a verbatim rule (53), a bot
    /// line a bad draft misquotes (55), and two lines from a third speaker
    /// `vex` (58, 60) for the multiplayer-shaped canon predicate test.
    pub(crate) fn synthetic_range() -> Vec<CitedMessage> {
        serde_json::from_str(include_str!("fixtures/synthetic_range.json"))
            .expect("synthetic_range.json is well-formed")
    }

    /// An `ExtractionOutput` over [`synthetic_range`] with four deliberate
    /// defects: a `key_quotes` item citing an out-of-range source id, a
    /// `rules` item that misquotes message 53, a `people` item sourced only
    /// from a bot turn (52), and an over-length `milestones` item. Every
    /// other item is valid, so `validate` on this fixture accepts some and
    /// rejects some.
    pub(crate) fn bad_draft() -> ExtractionOutput {
        parse_extraction(include_str!("fixtures/bad_draft.json"))
            .expect("bad_draft.json is well-formed JSON matching the extraction schema")
    }
}
