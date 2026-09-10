//! The input to rendering (#174): [`CompactionContext`] is a plain,
//! SQLite-free snapshot of everything a turn's prompt needs from
//! compaction — active facts already bucketed by category, the latest
//! committed checkpoint's summaries, and pinned messages. `render.rs`
//! consumes it without ever touching a `Connection`, so its own tests never
//! need a database either; [`CompactionContext::load`] is the one function
//! that does, built from the seams #171 already exposes
//! ([`CompactionStore`]) plus a message lookup the caller supplies.
//!
//! #182 mirrors this struct's fields 1:1 onto a wire type
//! (`multiplayer::protocol::ContinuityPayload`), minus `companion_state`
//! and `recalled_facts`; #186's joiner-side `HostContinuity` builds this
//! same struct back from that payload, so nothing here should be renamed
//! without checking both plans. #178 fills in `recalled_facts`, still
//! empty as of this issue.

use serde::{Deserialize, Serialize};

use crate::compaction::store::CompactionStore;
use crate::compaction::types::{Fact, FactCategory};
use crate::database::Message;

/// Who is speaking in a [`QuoteLine`]: a `Rule` or `KeyQuote` fact's `text`,
/// rendered verbatim and attributed to whichever of the two named
/// participants said it. `Serialize`/`Deserialize` (additive, #182) so it
/// can ride on `multiplayer::protocol::ContinuityPayload` unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuoteSpeaker {
    User,
    Companion,
}

/// One quoted line, ready to render as `"{name}: \"{text}\"\n"`.
/// `Eq`/`Serialize`/`Deserialize` (additive, #182) for the same reason as
/// [`QuoteSpeaker`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuoteLine {
    pub speaker: QuoteSpeaker,
    pub text: String,
}

/// One pinned message, with its text already resolved through
/// `message_by_id` at load time. `Eq`/`Serialize`/`Deserialize` (additive,
/// #182) for the same reason as [`QuoteSpeaker`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PinnedMessage {
    pub message_id: i32,
    pub speaker_id: String,
    pub content: String,
}

/// Everything [`crate::compaction::render::render`] needs, already
/// categorised and constructible without SQLite. `Default` is the "never
/// compacted" shape a solo chat that has never triggered compaction renders
/// from — `llm.rs`'s prompt-assembly tests check that shape produces a
/// byte-identical prompt to today's.
///
/// #182 maps its wire payload onto these fields 1:1: renaming a field here
/// needs to update that plan too.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct CompactionContext {
    pub compacted_through: Option<i32>,
    /// Active `UserState` facts, insertion (fact id) order.
    pub user_state: Vec<String>,
    /// Active `CompanionState` facts, insertion order.
    pub companion_state: Vec<String>,
    /// Active `Rule` facts, insertion order.
    pub rules: Vec<QuoteLine>,
    /// Active `Backstory` facts, any subject, insertion order.
    pub backstory: Vec<String>,
    /// Active `OpenThread` facts, insertion order.
    pub open_threads: Vec<String>,
    /// Active `KeyQuote` facts, insertion order.
    pub key_quotes: Vec<QuoteLine>,
    /// The latest committed checkpoint's `rolling_summary`, `""` if none.
    pub rolling_summary: String,
    /// The latest committed checkpoint's `summary`, `""` if none.
    pub recent_detail: String,
    /// Pinned messages, ascending `message_id`.
    pub pins: Vec<PinnedMessage>,
    /// Tantivy-recalled facts relevant to the current turn. Always empty
    /// until #178 fills it in; already the first thing `render::render`
    /// trims when the compaction slice is tight.
    pub recalled_facts: Vec<String>,
}

/// Reads a `Rule`/`KeyQuote` fact's `quote_speaker` column. `Some("user")`
/// is [`QuoteSpeaker::User`]; anything else — `Some("companion")`, an
/// unrecognised value, or `None` (#173's extraction has not landed yet, so
/// no fact carries a real value today) — falls back to
/// [`QuoteSpeaker::Companion`] rather than erroring: a quoted line without a
/// resolvable speaker still needs a name to render under.
fn quote_speaker_of(raw: Option<&str>) -> QuoteSpeaker {
    match raw {
        Some("user") => QuoteSpeaker::User,
        _ => QuoteSpeaker::Companion,
    }
}

impl CompactionContext {
    /// Pure: buckets `facts` by [`FactCategory`], skipping `Milestone`
    /// (already folded into the rolling summary prose; #178 indexes it for
    /// recall rather than rendering it directly) and `Person` (#177 routes
    /// third-party facts through the existing "Known individuals" path).
    /// Filters to `fact.active` itself, so a caller that has not already
    /// filtered (e.g. a test fixture built by hand) still gets the same
    /// result `CompactionStore::active_facts` would have produced.
    pub fn from_facts(facts: &[Fact]) -> Self {
        let mut ctx = CompactionContext::default();
        for fact in facts.iter().filter(|f| f.active) {
            match fact.category {
                FactCategory::UserState => ctx.user_state.push(fact.text.clone()),
                FactCategory::CompanionState => ctx.companion_state.push(fact.text.clone()),
                FactCategory::Rule => ctx.rules.push(QuoteLine {
                    speaker: quote_speaker_of(fact.quote_speaker.as_deref()),
                    text: fact.text.clone(),
                }),
                FactCategory::Backstory => ctx.backstory.push(fact.text.clone()),
                FactCategory::OpenThread => ctx.open_threads.push(fact.text.clone()),
                FactCategory::KeyQuote => ctx.key_quotes.push(QuoteLine {
                    speaker: quote_speaker_of(fact.quote_speaker.as_deref()),
                    text: fact.text.clone(),
                }),
                FactCategory::Milestone | FactCategory::Person => {}
            }
        }
        ctx
    }

    /// Store-backed builder: [`Self::from_facts`] over one consistent
    /// `store.context_snapshot(companion_id)` read (active facts,
    /// `compacted_through`, and the latest committed checkpoint's
    /// summaries, all from the same transaction on the production impl —
    /// see [`CompactionStore::context_snapshot`] — so a checkpoint commit
    /// racing this load can never combine facts from one side of the
    /// commit with the cutoff/summary from the other), plus pinned
    /// messages resolved through `message_by_id` (production passes
    /// `Database::get_message`).
    ///
    /// # Errors
    /// Propagates the store's `rusqlite::Error`. A pinned message whose row
    /// is gone (deleted after pinning) is skipped rather than treated as an
    /// error; any other `message_by_id` failure is propagated.
    pub fn load(
        store: &dyn CompactionStore,
        message_by_id: &dyn Fn(i32) -> rusqlite::Result<Message>,
        companion_id: i32,
    ) -> rusqlite::Result<Self> {
        let (facts, compacted_through, latest_committed) = store.context_snapshot(companion_id)?;
        let mut ctx = Self::from_facts(&facts);

        ctx.compacted_through = compacted_through;

        if let Some(checkpoint) = latest_committed {
            ctx.rolling_summary = checkpoint.rolling_summary.unwrap_or_default();
            ctx.recent_detail = checkpoint.summary.unwrap_or_default();
        }

        let mut pins = Vec::new();
        for pin in store.pins()? {
            match message_by_id(pin.message_id) {
                Ok(message) => pins.push(PinnedMessage {
                    message_id: pin.message_id,
                    speaker_id: message.speaker_id,
                    content: message.content,
                }),
                Err(rusqlite::Error::QueryReturnedNoRows) => {}
                Err(e) => return Err(e),
            }
        }
        ctx.pins = pins;

        Ok(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::store::RecordingStore;
    use crate::compaction::types::{CompactionStatus, CompactionTrigger, FactDraft, NewDraft};
    use std::collections::HashMap;

    fn fact(category: FactCategory, text: &str) -> Fact {
        Fact {
            id: 0,
            compaction_id: 0,
            category,
            subject: None,
            text: text.to_string(),
            quote_speaker: None,
            sources: vec![],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            active: true,
            superseded_by: None,
            rejected_reason: None,
        }
    }

    #[test]
    fn from_facts_buckets_by_category_and_skips_milestone_and_person() {
        let facts = vec![
            fact(FactCategory::UserState, "user fact"),
            fact(FactCategory::CompanionState, "companion fact"),
            fact(FactCategory::Backstory, "backstory fact"),
            fact(FactCategory::OpenThread, "open thread fact"),
            fact(FactCategory::Milestone, "milestone fact"),
            fact(FactCategory::Person, "person fact"),
        ];

        let ctx = CompactionContext::from_facts(&facts);

        assert_eq!(ctx.user_state, vec!["user fact".to_string()]);
        assert_eq!(ctx.companion_state, vec!["companion fact".to_string()]);
        assert_eq!(ctx.backstory, vec!["backstory fact".to_string()]);
        assert_eq!(ctx.open_threads, vec!["open thread fact".to_string()]);
        assert!(ctx.rules.is_empty());
        assert!(ctx.key_quotes.is_empty());
    }

    #[test]
    fn from_facts_preserves_insertion_order_within_a_category() {
        let facts = vec![
            fact(FactCategory::OpenThread, "first"),
            fact(FactCategory::OpenThread, "second"),
        ];
        let ctx = CompactionContext::from_facts(&facts);
        assert_eq!(
            ctx.open_threads,
            vec!["first".to_string(), "second".to_string()]
        );
    }

    #[test]
    fn from_facts_filters_out_inactive_rows() {
        let mut inactive = fact(FactCategory::UserState, "stale");
        inactive.active = false;
        let ctx = CompactionContext::from_facts(&[inactive]);
        assert!(ctx.user_state.is_empty());
    }

    #[test]
    fn from_facts_parses_quote_speaker_and_defaults_unknown_to_companion() {
        let mut user_rule = fact(FactCategory::Rule, "always be honest");
        user_rule.quote_speaker = Some("user".to_string());
        let mut unspecified_quote = fact(FactCategory::KeyQuote, "I promise");
        unspecified_quote.quote_speaker = None;

        let ctx = CompactionContext::from_facts(&[user_rule, unspecified_quote]);

        assert_eq!(ctx.rules[0].speaker, QuoteSpeaker::User);
        assert_eq!(ctx.key_quotes[0].speaker, QuoteSpeaker::Companion);
    }

    #[test]
    fn load_combines_facts_checkpoint_and_pins_skipping_a_deleted_pinned_message() {
        let store = RecordingStore::new();
        let compaction_id = store
            .insert_draft(NewDraft {
                companion_id: 1,
                from_message_id: 1,
                through_message_id: 3,
                trigger: CompactionTrigger::Threshold,
                raw_model_output: None,
            })
            .unwrap();
        store
            .insert_facts(
                compaction_id,
                &[FactDraft {
                    category: FactCategory::UserState,
                    subject: None,
                    text: "loves cats".to_string(),
                    quote_speaker: None,
                    sources: vec![1],
                    replaces: vec![],
                    relation_to: None,
                    relation: None,
                    canon: true,
                    rejected_reason: None,
                }],
            )
            .unwrap();
        store
            .update_status(compaction_id, CompactionStatus::Committed)
            .unwrap();
        store.set_compacted_through(1, Some(3)).unwrap();
        // message 1 still exists; message 2 was pinned then deleted.
        store.pin(1).unwrap();
        store.pin(2).unwrap();

        let mut messages = HashMap::new();
        messages.insert(
            1,
            Message {
                id: 1,
                ai: false,
                speaker_id: "user".to_string(),
                content: "hi".to_string(),
                created_at: String::new(),
            },
        );
        let lookup = move |id: i32| -> rusqlite::Result<Message> {
            messages
                .get(&id)
                .cloned()
                .ok_or(rusqlite::Error::QueryReturnedNoRows)
        };

        let ctx = CompactionContext::load(&store, &lookup, 1).unwrap();

        assert_eq!(ctx.user_state, vec!["loves cats".to_string()]);
        assert_eq!(ctx.compacted_through, Some(3));
        assert_eq!(ctx.pins.len(), 1);
        assert_eq!(ctx.pins[0].message_id, 1);
        assert_eq!(ctx.pins[0].content, "hi");
    }

    #[test]
    fn load_propagates_a_message_lookup_error_other_than_no_rows() {
        let store = RecordingStore::new();
        store.pin(1).unwrap();
        let lookup = |_id: i32| -> rusqlite::Result<Message> { Err(rusqlite::Error::InvalidQuery) };

        let err = CompactionContext::load(&store, &lookup, 1).unwrap_err();
        assert!(matches!(err, rusqlite::Error::InvalidQuery));
    }

    #[test]
    fn load_defaults_to_empty_summaries_when_nothing_was_ever_committed() {
        let store = RecordingStore::new();
        let lookup =
            |_id: i32| -> rusqlite::Result<Message> { Err(rusqlite::Error::QueryReturnedNoRows) };

        let ctx = CompactionContext::load(&store, &lookup, 1).unwrap();

        assert_eq!(ctx.rolling_summary, "");
        assert_eq!(ctx.recent_detail, "");
        assert_eq!(ctx.compacted_through, None);
        assert!(ctx.pins.is_empty());
    }
}
