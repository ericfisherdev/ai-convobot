//! The regenerate-from loop for running thoughts (#217): rewrites every
//! thought a speaker owns from a given message forward, replacing #214's
//! captured-but-stale notes with fresh ones written against the *current*
//! round content. Kept out of `main.rs`, the way `compaction::commit` keeps
//! commit logic out of its handler.
//!
//! Reuses #216's generation seam rather than defining its own:
//! [`crate::running_thoughts::hook::thought_inputs_for_range`] for the
//! per-round inputs and `main.rs`'s `host_thought_writer(speakers)` (the
//! same `(inputs, insert) -> Result<RunningThought, ThoughtError>` closure
//! `PendingTurn::think`'s two live-round callers already use) for
//! generation. This never goes through `PendingTurn::think` itself — that
//! runs inside a live chat turn and needs a user message; a regenerate
//! request has neither.

use std::io;

use crate::participants::ParticipantId;
use crate::running_thoughts::generate::ThoughtError;
use crate::running_thoughts::prompt::ThoughtInputs;
use crate::running_thoughts::store::RunningThoughtStore;
use crate::running_thoughts::types::{NewRunningThought, RunningThought};

/// The generation closure [`regenerate_from`] drives: `(inputs, insert) ->
/// Result<RunningThought, ThoughtError>`, the same shape `main.rs`'s
/// `host_thought_writer(speakers)` returns and `PendingTurn::think`'s two
/// live-round callers already pass around, so a type alias keeps this
/// module's own signatures (and clippy) readable rather than spelling the
/// nested `dyn Fn`s out at every call site.
pub type ThoughtGenerator<'a> = dyn FnMut(
        &ThoughtInputs,
        &dyn Fn(NewRunningThought) -> rusqlite::Result<RunningThought>,
    ) -> Result<RunningThought, ThoughtError>
    + 'a;

/// One regenerate-from request: rewrite every thought `speaker_id` owns
/// with `through_message_id >= from_message_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegenerateRequest {
    pub companion_id: i32,
    pub speaker_id: ParticipantId,
    pub from_message_id: i32,
}

/// What [`regenerate_from`] reports as it rewrites each thought it owns. The
/// SSE adapter in `main.rs` (`SseThoughtSink`) implements this; tests use a
/// `Vec` recorder.
pub trait RegenerateSink {
    fn thought_started(&mut self, speaker: &ParticipantId);
    fn thought_regenerated(&mut self, thought: &RunningThought);
}

/// Why [`regenerate_from`] could not finish. `Display`'s wording says which
/// step failed, so `main.rs` can send it straight out as the terminal error
/// chunk's message.
#[derive(Debug)]
pub enum ThoughtRegenerateError {
    /// Nothing has `through_message_id >= request.from_message_id`. Nothing
    /// was touched.
    NothingToRegenerate,
    /// `store.delete_from`, or the re-insert of an original this run did
    /// not reach, failed.
    Store(rusqlite::Error),
    /// [`ThoughtInputs`] could not be read (the `TranscriptSource` seam
    /// failed) for the round starting at `from_message_id`.
    Inputs {
        from_message_id: i32,
        source: io::Error,
    },
    /// Generation failed (or produced an empty note, `ThoughtError::Empty`)
    /// for the round starting at `from_message_id`. Unlike a live round,
    /// which skips an empty note silently, a rewrite the user explicitly
    /// asked for treats an empty result as a failure.
    Generate {
        from_message_id: i32,
        source: ThoughtError,
    },
}

impl std::fmt::Display for ThoughtRegenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThoughtRegenerateError::NothingToRegenerate => {
                write!(f, "there is nothing to regenerate from that message")
            }
            ThoughtRegenerateError::Store(e) => write!(f, "failed to update running thoughts: {e}"),
            ThoughtRegenerateError::Inputs {
                from_message_id,
                source,
            } => write!(
                f,
                "failed to read inputs for the round starting at message {from_message_id}: {source}"
            ),
            ThoughtRegenerateError::Generate {
                from_message_id,
                source,
            } => write!(
                f,
                "failed to regenerate the thought for the round starting at message {from_message_id}: {source}"
            ),
        }
    }
}

impl std::error::Error for ThoughtRegenerateError {}

/// Turns a caught panic payload into a message, mirroring
/// `compaction::extract::panic_message`'s reasoning (private to that module,
/// so duplicated here rather than exposed across a module boundary just for
/// this one caller).
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!("thought generation panicked: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("thought generation panicked: {message}")
    } else {
        "thought generation panicked with a non-string payload".to_string()
    }
}

/// Re-inserts `original` exactly as captured (new id, same text/`edited`),
/// the shared body of both the success path (a non-owned speaker's row) and
/// the failure path (rows this run did not reach).
fn reinsert_original(
    store: &dyn RunningThoughtStore,
    original: &RunningThought,
) -> rusqlite::Result<()> {
    store.insert(NewRunningThought {
        companion_id: original.companion_id,
        speaker_id: original.speaker_id.clone(),
        from_message_id: original.from_message_id,
        through_message_id: original.through_message_id,
        text: original.text.clone(),
        edited: original.edited,
    })?;
    Ok(())
}

/// Rewrites every thought `request.speaker_id` owns whose `through_message_id`
/// reaches at least `request.from_message_id`, in id order; every other
/// speaker's thought in that same range is re-inserted unchanged (each
/// instance regenerates only the speaker it generates for, #220's ownership
/// rule).
///
/// 1. `store.delete_from` captures and removes every matching row (any
///    speaker) in one transaction, so `inputs_for`'s `previous` (read live,
///    via `recent_for`) can never see a later round's stale thought while an
///    earlier one is being rewritten.
/// 2. For each captured row, oldest first: a non-owned speaker's row is
///    re-inserted as-is. An owned row announces `sink.thought_started`,
///    reads fresh inputs over its own captured `[from_message_id,
///    through_message_id]` (never re-derived), and calls `generate`; success
///    reports `sink.thought_regenerated` and the row joins the chain the
///    next `inputs_for` call reads via `recent_for`.
/// 3. A failure re-inserts every row from the failed one onward (inclusive)
///    exactly as captured — so a failed run loses at most the one thought it
///    was rewriting — and returns the error. Rows already regenerated stay
///    regenerated.
///
/// Returns the count of thoughts actually regenerated (excludes untouched
/// re-inserts of other speakers' rows).
pub fn regenerate_from(
    store: &dyn RunningThoughtStore,
    inputs_for: &mut dyn FnMut(&ParticipantId, i32, i32) -> io::Result<ThoughtInputs>,
    generate: &mut ThoughtGenerator,
    request: &RegenerateRequest,
    sink: &mut dyn RegenerateSink,
) -> Result<usize, ThoughtRegenerateError> {
    let captured = store
        .delete_from(request.companion_id, request.from_message_id)
        .map_err(ThoughtRegenerateError::Store)?;
    if captured.is_empty() {
        return Err(ThoughtRegenerateError::NothingToRegenerate);
    }

    let insert = |thought: NewRunningThought| -> rusqlite::Result<RunningThought> {
        let id = store.insert(thought)?;
        store.get(id)?.ok_or(rusqlite::Error::QueryReturnedNoRows)
    };

    let mut regenerated = 0usize;
    for (index, original) in captured.iter().enumerate() {
        if original.speaker_id != request.speaker_id.as_str() {
            reinsert_original(store, original).map_err(ThoughtRegenerateError::Store)?;
            continue;
        }

        sink.thought_started(&request.speaker_id);
        let outcome = inputs_for(
            &request.speaker_id,
            original.from_message_id,
            original.through_message_id,
        )
        .map_err(|source| ThoughtRegenerateError::Inputs {
            from_message_id: original.from_message_id,
            source,
        })
        .and_then(|inputs| {
            // Caught, not just propagated: `llama-cpp-2`'s
            // `LlamaModel::load_from_file` panics (rather than returning
            // `Err`) in a debug build when the configured GGUF path does not
            // exist (`compaction::extract::run_extraction_job`'s identical
            // `catch_unwind` documents the same behaviour). Left uncaught, a
            // single missing model would unwind straight out of this loop
            // and skip the restore step below entirely, defeating the "a
            // failed run loses at most the one thought it was rewriting"
            // guarantee this function exists for.
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| generate(&inputs, &insert)))
                .unwrap_or_else(|payload| {
                    Err(ThoughtError::Generate(std::io::Error::other(
                        panic_message(payload.as_ref()),
                    )))
                })
                .map_err(|source| ThoughtRegenerateError::Generate {
                    from_message_id: original.from_message_id,
                    source,
                })
        });

        match outcome {
            Ok(row) => {
                sink.thought_regenerated(&row);
                regenerated += 1;
            }
            Err(err) => {
                // This row and every one after it (owned or not) never got a
                // fresh generation attempt; restore them all exactly as
                // captured before reporting the failure.
                for remaining in &captured[index..] {
                    reinsert_original(store, remaining).map_err(ThoughtRegenerateError::Store)?;
                }
                return Err(err);
            }
        }
    }

    Ok(regenerated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::running_thoughts::store::RecordingStore;

    fn seed(store: &RecordingStore, speaker: &str, from: i32, through: i32, text: &str) -> i64 {
        store
            .insert(NewRunningThought {
                companion_id: 1,
                speaker_id: speaker.to_string(),
                from_message_id: from,
                through_message_id: through,
                text: text.to_string(),
                edited: false,
            })
            .unwrap()
    }

    fn a_request(from_message_id: i32) -> RegenerateRequest {
        RegenerateRequest {
            companion_id: 1,
            speaker_id: ParticipantId::CHAR,
            from_message_id,
        }
    }

    fn inputs_of(speaker: &ParticipantId, from: i32, through: i32) -> ThoughtInputs {
        ThoughtInputs {
            companion_id: 1,
            speaker_id: speaker.clone(),
            previous: vec![],
            round: vec![],
            from_message_id: from,
            through_message_id: through,
        }
    }

    /// A `generate` closure that always succeeds, tagging the produced text
    /// with the round it was asked to regenerate so assertions can tell
    /// which round produced which row.
    fn always_succeeds() -> impl FnMut(
        &ThoughtInputs,
        &dyn Fn(NewRunningThought) -> rusqlite::Result<RunningThought>,
    ) -> Result<RunningThought, ThoughtError> {
        |inputs, insert| {
            insert(NewRunningThought {
                companion_id: inputs.companion_id,
                speaker_id: inputs.speaker_id.to_string(),
                from_message_id: inputs.from_message_id,
                through_message_id: inputs.through_message_id,
                text: format!(
                    "fresh note for [{}, {}]",
                    inputs.from_message_id, inputs.through_message_id
                ),
                edited: false,
            })
            .map_err(ThoughtError::Store)
        }
    }

    #[derive(Default)]
    struct RecordingSink {
        started: Vec<ParticipantId>,
        regenerated: Vec<RunningThought>,
    }

    impl RegenerateSink for RecordingSink {
        fn thought_started(&mut self, speaker: &ParticipantId) {
            self.started.push(speaker.clone());
        }

        fn thought_regenerated(&mut self, thought: &RunningThought) {
            self.regenerated.push(thought.clone());
        }
    }

    #[test]
    fn replaces_exactly_the_rows_from_the_given_message_forward_leaving_earlier_ones_untouched() {
        let store = RecordingStore::new();
        let earlier = seed(&store, "char", 1, 4, "earlier note");
        seed(&store, "char", 5, 8, "stale note one");
        seed(&store, "char", 9, 12, "stale note two");

        let mut sink = RecordingSink::default();
        let regenerated = regenerate_from(
            &store,
            &mut |speaker, from, through| Ok(inputs_of(speaker, from, through)),
            &mut always_succeeds(),
            &a_request(5),
            &mut sink,
        )
        .unwrap();

        assert_eq!(regenerated, 2);
        let after = store.list(1).unwrap();
        assert_eq!(after.len(), 3);
        let earlier_row = after.iter().find(|t| t.id == earlier).unwrap();
        assert_eq!(earlier_row.text, "earlier note");
        assert!(after
            .iter()
            .any(|t| t.from_message_id == 5 && t.text.contains("fresh note")));
        assert!(after
            .iter()
            .any(|t| t.from_message_id == 9 && t.text.contains("fresh note")));
    }

    #[test]
    fn a_non_owned_speakers_row_inside_the_range_is_re_inserted_with_its_text_and_edited_intact() {
        let store = RecordingStore::new();
        seed(&store, "char", 1, 4, "char's note");
        let bot_id = store
            .insert(NewRunningThought {
                companion_id: 1,
                speaker_id: "bot1".to_string(),
                from_message_id: 2,
                through_message_id: 3,
                text: "bot1's own note".to_string(),
                edited: true,
            })
            .unwrap();

        let mut sink = RecordingSink::default();
        regenerate_from(
            &store,
            &mut |speaker, from, through| Ok(inputs_of(speaker, from, through)),
            &mut always_succeeds(),
            &a_request(1),
            &mut sink,
        )
        .unwrap();

        let after = store.list(1).unwrap();
        // `bot_id` itself was deleted and re-inserted, so it has a new id;
        // find it by its untouched content instead.
        let bot_row = after
            .iter()
            .find(|t| t.speaker_id == "bot1")
            .expect("bot1's row should have been re-inserted");
        assert_ne!(
            bot_row.id, bot_id,
            "the row is re-inserted, not kept in place"
        );
        assert_eq!(bot_row.text, "bot1's own note");
        assert!(bot_row.edited);
        assert_eq!(bot_row.from_message_id, 2);
        assert_eq!(bot_row.through_message_id, 3);
    }

    #[test]
    fn a_failure_on_the_second_of_three_restores_the_third_and_keeps_the_first_regenerated() {
        let store = RecordingStore::new();
        seed(&store, "char", 1, 3, "round one");
        seed(&store, "char", 4, 6, "round two");
        seed(&store, "char", 7, 9, "round three");

        let mut sink = RecordingSink::default();
        let err = regenerate_from(
            &store,
            &mut |speaker, from, through| Ok(inputs_of(speaker, from, through)),
            &mut |inputs, insert| {
                if inputs.from_message_id == 4 {
                    return Err(ThoughtError::Generate(io::Error::other("model failed")));
                }
                insert(NewRunningThought {
                    companion_id: inputs.companion_id,
                    speaker_id: inputs.speaker_id.to_string(),
                    from_message_id: inputs.from_message_id,
                    through_message_id: inputs.through_message_id,
                    text: "fresh".to_string(),
                    edited: false,
                })
                .map_err(ThoughtError::Store)
            },
            &a_request(1),
            &mut sink,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ThoughtRegenerateError::Generate {
                from_message_id: 4,
                ..
            }
        ));

        let after = store.list(1).unwrap();
        assert_eq!(after.len(), 3);
        assert!(after
            .iter()
            .any(|t| t.from_message_id == 1 && t.text == "fresh"));
        let restored_two = after.iter().find(|t| t.from_message_id == 4).unwrap();
        assert_eq!(restored_two.text, "round two");
        let restored_three = after.iter().find(|t| t.from_message_id == 7).unwrap();
        assert_eq!(restored_three.text, "round three");
        assert!(!restored_three.edited);
    }

    #[test]
    fn an_empty_generation_is_a_failure_that_restores_the_original() {
        let store = RecordingStore::new();
        seed(&store, "char", 1, 3, "original note");

        let err = regenerate_from(
            &store,
            &mut |speaker, from, through| Ok(inputs_of(speaker, from, through)),
            &mut |_inputs, _insert| Err(ThoughtError::Empty),
            &a_request(1),
            &mut RecordingSink::default(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ThoughtRegenerateError::Generate {
                source: ThoughtError::Empty,
                ..
            }
        ));
        let after = store.list(1).unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].text, "original note");
    }

    #[test]
    fn sink_receives_started_then_regenerated_pairs_in_order() {
        let store = RecordingStore::new();
        seed(&store, "char", 1, 3, "one");
        seed(&store, "char", 4, 6, "two");

        let mut sink = RecordingSink::default();
        regenerate_from(
            &store,
            &mut |speaker, from, through| Ok(inputs_of(speaker, from, through)),
            &mut always_succeeds(),
            &a_request(1),
            &mut sink,
        )
        .unwrap();

        assert_eq!(sink.started.len(), 2);
        assert_eq!(sink.regenerated.len(), 2);
        assert!(sink.started.iter().all(|s| *s == ParticipantId::CHAR));
        assert_eq!(sink.regenerated[0].from_message_id, 1);
        assert_eq!(sink.regenerated[1].from_message_id, 4);
    }

    #[test]
    fn an_empty_range_is_nothing_to_regenerate_and_touches_nothing() {
        let store = RecordingStore::new();
        seed(&store, "char", 1, 3, "keep me");

        let err = regenerate_from(
            &store,
            &mut |speaker, from, through| Ok(inputs_of(speaker, from, through)),
            &mut always_succeeds(),
            &a_request(100),
            &mut RecordingSink::default(),
        )
        .unwrap_err();

        assert!(matches!(err, ThoughtRegenerateError::NothingToRegenerate));
        let after = store.list(1).unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].text, "keep me");
    }
}
