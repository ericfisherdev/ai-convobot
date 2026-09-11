//! The seam-based half of running-thought generation (#216): everything a
//! caller needs to build a [`ThoughtInputs`] for one round, split so a
//! joiner (#220) can reuse [`thought_range`], [`resolve_round_start`] and
//! [`thought_inputs_for_range`] with its own [`RunningThoughtStore`] and
//! `InMemoryTranscript`. Only [`thought_inputs_on`] touches `Database`
//! directly — it is the host's live-round reader behind
//! `chat_turn::SqliteTurnStore::thought_inputs`.

use std::io;

use crate::compaction::store::{CompactionStore, SqliteCompactionStore};
use crate::database::{Database, Message};
use crate::llm::{SqliteTranscript, TranscriptSource};
use crate::participants::ParticipantId;
use crate::running_thoughts::prompt::{
    ThoughtInputs, THOUGHT_CHAIN_LENGTH, THOUGHT_ROUND_MAX_MESSAGES,
};
use crate::running_thoughts::store::{RunningThoughtStore, SqliteRunningThoughtStore};
use crate::running_thoughts::types::RunningThought;

/// The round's messages a thought is written about: every message with
/// `from <= id <= through`, oldest first, truncated to the newest
/// [`THOUGHT_ROUND_MAX_MESSAGES`] so the first note after enabling the flag
/// on a long chat does not render the whole history.
///
/// Works over `SqliteTranscript` (host) and `InMemoryTranscript` (joiner)
/// alike.
///
/// `NO_LIMIT`, not `usize::MAX`: `SqliteTranscript::recent_messages` binds
/// the limit as a SQLite `LIMIT` parameter, and rusqlite's checked `usize`
/// -> `i64` conversion rejects `usize::MAX` on a 64-bit host with
/// `ToSqlConversionFailure` — which would silently disable running-thought
/// generation entirely (`thought_inputs_on`'s caller logs the error and
/// returns `None`). `i64::MAX` messages is not a real limit in practice.
pub fn thought_range(
    transcript: &dyn TranscriptSource,
    from: i32,
    through: i32,
) -> io::Result<Vec<Message>> {
    const NO_LIMIT: usize = i64::MAX as usize;
    let mut messages = transcript.recent_messages(Some(from - 1), NO_LIMIT)?;
    messages.retain(|m| m.id <= through);
    if messages.len() > THOUGHT_ROUND_MAX_MESSAGES {
        let start = messages.len() - THOUGHT_ROUND_MAX_MESSAGES;
        messages = messages.split_off(start);
    }
    Ok(messages)
}

/// Where a new round starts, in the absence (or presence) of a prior note
/// for this speaker: right after the last note's own range when there is
/// one, otherwise the message right after the last `ai` message strictly
/// before `through` in `tail`, or `tail`'s first message when there is no
/// earlier `ai` message either (this speaker's first-ever note).
pub fn resolve_round_start(latest: Option<&RunningThought>, tail: &[Message], through: i32) -> i32 {
    if let Some(latest) = latest {
        return latest.through_message_id + 1;
    }
    tail.iter()
        .filter(|m| m.id < through && m.ai)
        .map(|m| m.id + 1)
        .max()
        .or_else(|| tail.first().map(|m| m.id))
        .unwrap_or(through)
}

/// A joiner's own analogue of [`resolve_round_start`] (#220): the range a
/// speaker's next thought should cover, given the newest `through_message_id`
/// it has already written one for (`RunningThoughtStore::latest_for`) and
/// `visible` — the messages it is about to reply over, oldest first (the
/// joiner's own `GenerateRequest.transcript`, already post-`compacted_through`
/// and clamped to whatever its mirror holds).
///
/// Named differently from [`resolve_round_start`] because a joiner has no
/// `tail` to scan for the last `ai` message when there is no prior note —
/// its very first thought simply starts at `visible`'s own first id, not at
/// a message before it a joiner may never have seen.
///
/// `from` is `previous_through + 1`, clamped up to `visible.first().id`;
/// `through` is `visible.last().id`. Returns `None` when `visible` is empty
/// or `from` ends up past `through`: a regenerate re-asks the same speaker
/// over a transcript whose newest id this speaker already covered, and that
/// case must not write a second thought.
pub fn pending_thought_range(
    previous_through: Option<i32>,
    visible: &[Message],
) -> Option<(i32, i32)> {
    let first = visible.first()?.id;
    let through = visible.last()?.id;
    let from = previous_through.map_or(first, |t| (t + 1).max(first));
    (from <= through).then_some((from, through))
}

/// Builds a [`ThoughtInputs`] for `[from, through]`: this speaker's chained
/// context (`store.recent_for`) and the round itself (`thought_range`). No
/// flag check and no `Database` access — the building block both the host's
/// live round (via [`thought_inputs_on`]) and #217's regenerate loop (which
/// already knows its own range) call.
pub fn thought_inputs_for_range(
    store: &dyn RunningThoughtStore,
    transcript: &dyn TranscriptSource,
    companion_id: i32,
    speaker: &ParticipantId,
    from: i32,
    through: i32,
) -> io::Result<ThoughtInputs> {
    let previous = store
        .recent_for(companion_id, speaker.as_str(), THOUGHT_CHAIN_LENGTH)
        .map_err(|e| io::Error::other(e.to_string()))?;
    let round = thought_range(transcript, from, through)?;
    Ok(ThoughtInputs {
        companion_id,
        speaker_id: speaker.clone(),
        previous,
        round,
        from_message_id: from,
        through_message_id: through,
    })
}

/// Wraps an [`io::Error`] (from the [`TranscriptSource`] seam) as a
/// [`rusqlite::Error`], so [`thought_inputs_on`] can propagate a single
/// error type. `ToSqlConversionFailure` carries an arbitrary boxed error and
/// is never otherwise produced by a read path, so it cannot be confused with
/// a real conversion failure here.
fn as_rusqlite_error(e: io::Error) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(e))
}

/// The host's live-round reader: `None` when running thoughts are disabled,
/// so the caller never announces or runs a generation. The only
/// `Database`-touching function in this module — everything else here is
/// seam-based so #220's joiner can reuse it with its own store and
/// transcript.
pub(crate) fn thought_inputs_on(
    companion_id: i32,
    speaker: &ParticipantId,
    through_message_id: i32,
) -> rusqlite::Result<Option<ThoughtInputs>> {
    if !Database::get_config()?.running_thoughts_enabled {
        return Ok(None);
    }

    let store = SqliteRunningThoughtStore;
    let latest = store.latest_for(companion_id, speaker.as_str())?;
    let compacted_through = SqliteCompactionStore.compacted_through(companion_id)?;
    let tail = Database::get_messages_after(compacted_through.unwrap_or(0))?;
    let from = resolve_round_start(latest.as_ref(), &tail, through_message_id);

    thought_inputs_for_range(
        &store,
        &SqliteTranscript,
        companion_id,
        speaker,
        from,
        through_message_id,
    )
    .map(Some)
    .map_err(as_rusqlite_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::InMemoryTranscript;

    fn message(id: i32, ai: bool) -> Message {
        Message {
            id,
            ai,
            speaker_id: if ai {
                "char".to_string()
            } else {
                "user".to_string()
            },
            content: format!("message {id}"),
            created_at: String::new(),
        }
    }

    fn a_thought(through: i32) -> RunningThought {
        RunningThought {
            id: 1,
            companion_id: 1,
            speaker_id: "char".to_string(),
            from_message_id: 1,
            through_message_id: through,
            text: "a note".to_string(),
            edited: false,
            created_at: String::new(),
        }
    }

    #[test]
    fn thought_range_keeps_only_the_from_through_window() {
        let transcript = InMemoryTranscript(vec![
            message(1, false),
            message(2, true),
            message(3, false),
            message(4, true),
        ]);

        let range = thought_range(&transcript, 2, 3).unwrap();

        assert_eq!(range.iter().map(|m| m.id).collect::<Vec<_>>(), vec![2, 3]);
    }

    #[test]
    fn thought_range_truncates_to_the_newest_max_messages() {
        let messages: Vec<Message> = (1..=(THOUGHT_ROUND_MAX_MESSAGES as i32 + 5))
            .map(|id| message(id, id % 2 == 0))
            .collect();
        let through = messages.last().unwrap().id;
        let transcript = InMemoryTranscript(messages);

        let range = thought_range(&transcript, 1, through).unwrap();

        assert_eq!(range.len(), THOUGHT_ROUND_MAX_MESSAGES);
        assert_eq!(range.last().unwrap().id, through);
    }

    #[test]
    fn resolve_round_start_continues_right_after_the_prior_notes_range() {
        let latest = a_thought(5);

        let start = resolve_round_start(Some(&latest), &[], 10);

        assert_eq!(start, 6);
    }

    #[test]
    fn resolve_round_start_with_no_prior_note_starts_after_the_last_ai_message() {
        let tail = vec![message(1, false), message(2, true), message(3, false)];

        let start = resolve_round_start(None, &tail, 3);

        assert_eq!(
            start, 3,
            "right after id 2, the last ai message before through"
        );
    }

    #[test]
    fn resolve_round_start_with_no_prior_note_and_no_ai_message_starts_at_the_tails_first_id() {
        let tail = vec![message(5, false), message(6, false)];

        let start = resolve_round_start(None, &tail, 6);

        assert_eq!(start, 5);
    }

    #[test]
    fn pending_thought_range_with_no_prior_note_starts_at_visibles_first_id() {
        let visible = vec![message(3, false), message(4, true)];

        let range = pending_thought_range(None, &visible);

        assert_eq!(range, Some((3, 4)));
    }

    #[test]
    fn pending_thought_range_with_a_prior_note_advances_right_after_it() {
        let visible = vec![message(3, false), message(4, true), message(5, false)];

        let range = pending_thought_range(Some(3), &visible);

        assert_eq!(range, Some((4, 5)));
    }

    #[test]
    fn pending_thought_range_clamps_to_visibles_first_id_when_the_prior_note_predates_the_mirror() {
        // The mirror only holds the newest 50 messages post-`compacted_through`,
        // so a prior note's range can start before anything this joiner's
        // mirror still has.
        let visible = vec![message(10, false), message(11, true)];

        let range = pending_thought_range(Some(2), &visible);

        assert_eq!(range, Some((10, 11)));
    }

    #[test]
    fn pending_thought_range_is_none_when_the_prior_note_already_covers_visibles_newest_id() {
        // The regenerate case: the same speaker is asked again over a
        // transcript whose newest id it already wrote a thought through.
        let visible = vec![message(3, false), message(4, true)];

        let range = pending_thought_range(Some(4), &visible);

        assert_eq!(range, None);
    }

    #[test]
    fn pending_thought_range_is_none_for_an_empty_visible_window() {
        let range = pending_thought_range(None, &[]);

        assert_eq!(range, None);
    }
}
