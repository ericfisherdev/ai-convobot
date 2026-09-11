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

/// Re-inserts every row in `rows` exactly as captured, continuing past a
/// failed re-insert rather than stopping at the first one -- `delete_from`
/// already committed the deletes, so every row here is already gone from
/// the store, and skipping the rest of the loop would lose them for good
/// instead of just the one that failed to restore. Returns the first
/// failure, if any (logged as it happens, since a caller reports at most
/// one and the rest would otherwise go unrecorded).
fn restore_all(
    store: &dyn RunningThoughtStore,
    rows: &[RunningThought],
) -> Option<rusqlite::Error> {
    let mut first_failure = None;
    for remaining in rows {
        if let Err(e) = reinsert_original(store, remaining) {
            eprintln!(
                "running thoughts: failed to restore the original for the round starting at message {}: {e}",
                remaining.from_message_id
            );
            first_failure.get_or_insert(e);
        }
    }
    first_failure
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
///    re-inserted as-is (a failure here restores every later row via
///    [`restore_all`] before reporting, same as step 3 — every row in
///    `captured` is already deleted regardless of which branch reaches it).
///    An owned row announces `sink.thought_started`,
///    reads fresh inputs over its own captured `[from_message_id,
///    through_message_id]` (never re-derived), and calls `generate`; success
///    reports `sink.thought_regenerated` and the row joins the chain the
///    next `inputs_for` call reads via `recent_for`. An empty `round` (every
///    message in the window is gone, or the round predates a joiner's
///    mirror) is treated as a failure rather than run through the model, so
///    the restore path below runs instead of replacing the note with one
///    generated from nothing.
/// 3. A failure re-inserts every row from the failed one onward (inclusive)
///    exactly as captured — so a failed run loses at most the one thought it
///    was rewriting — and returns the error. Rows already regenerated stay
///    regenerated. Restoring continues past a single failed re-insert (the
///    rows are already deleted; skipping the rest would lose them for
///    good), and only replaces the reported error when a restore itself
///    fails.
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
            if let Err(e) = reinsert_original(store, original) {
                // Same rule as the failure arm below: every row after this
                // one is already deleted too, so restore them before
                // reporting rather than losing them behind this one error.
                // This row's own failure is logged here, in the same format
                // `restore_all` uses for each of its own: only one error is
                // ever returned, and `e` is the first one chronologically,
                // so it is the one reported -- a later restore failure
                // still runs (for its own side effects) but must not
                // silently swallow this one.
                eprintln!(
                    "running thoughts: failed to restore the original for the round starting at message {}: {e}",
                    original.from_message_id
                );
                restore_all(store, &captured[index + 1..]);
                return Err(ThoughtRegenerateError::Store(e));
            }
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
            // An empty round is treated as a failure, not run through the
            // model: `thought_inputs_for_range` returns `Ok` with `round:
            // vec![]` when nothing resolves in the captured window (every
            // message in it was deleted since the original was written, or
            // — on a joiner — the round predates the mirror), and generating
            // from a blank "What just happened:" section would silently
            // replace a real note with one the model invented from nothing.
            // Routing it through the same `Inputs` variant runs the restore
            // path below exactly as a genuine read failure would.
            if inputs.round.is_empty() {
                return Err(ThoughtRegenerateError::Inputs {
                    from_message_id: original.from_message_id,
                    source: io::Error::other(format!(
                        "no messages remain in [{}, {}]",
                        original.from_message_id, original.through_message_id
                    )),
                });
            }
            Ok(inputs)
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
                // captured before reporting the failure. `delete_from`
                // already committed the deletes, so one failed re-insert
                // here must not stop the rest of the loop from running --
                // that would lose every later captured row permanently
                // instead of just the one this run was rewriting. The
                // original `Inputs`/`Generate` error is what gets returned;
                // a restore failure only replaces it when restoring itself
                // failed, since that is the more urgent thing to report --
                // but `err` is logged first (same rule as the non-owned
                // branch above): it is what actually caused this row to
                // need restoring at all, and returning `Store(e)` instead
                // must not silently erase that reason from the record.
                return Err(match restore_all(store, &captured[index..]) {
                    Some(e) => {
                        eprintln!(
                            "running thoughts: regeneration failed for the round starting at message {}: {err}",
                            original.from_message_id
                        );
                        ThoughtRegenerateError::Store(e)
                    }
                    None => err,
                });
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

    /// A one-message round, not an empty one: `regenerate_from` treats an
    /// empty `round` as a failure (a captured range whose messages are all
    /// gone), so a helper other tests build ordinary success-path inputs
    /// from must not accidentally produce one.
    fn inputs_of(speaker: &ParticipantId, from: i32, through: i32) -> ThoughtInputs {
        ThoughtInputs {
            companion_id: 1,
            speaker_id: speaker.clone(),
            previous: vec![],
            round: vec![crate::database::Message {
                id: from,
                ai: false,
                speaker_id: "user".to_string(),
                content: "the round's own content".to_string(),
                created_at: String::new(),
            }],
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

    #[test]
    fn an_empty_round_is_a_failure_that_restores_the_original_instead_of_generating_from_nothing() {
        let store = RecordingStore::new();
        seed(&store, "char", 1, 3, "first note");
        seed(&store, "char", 4, 6, "second note");

        let err = regenerate_from(
            &store,
            &mut |speaker, from, through| {
                // The second round's messages are gone: an empty `round`,
                // the same shape `thought_inputs_for_range` returns when
                // nothing resolves in the window.
                let mut inputs = inputs_of(speaker, from, through);
                if from == 4 {
                    inputs.round = vec![];
                }
                Ok(inputs)
            },
            &mut always_succeeds(),
            &a_request(1),
            &mut RecordingSink::default(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ThoughtRegenerateError::Inputs {
                from_message_id: 4,
                ..
            }
        ));

        let after = store.list(1).unwrap();
        assert_eq!(after.len(), 2);
        assert!(after
            .iter()
            .any(|t| t.from_message_id == 1 && t.text.contains("fresh note")));
        let restored = after.iter().find(|t| t.from_message_id == 4).unwrap();
        assert_eq!(restored.text, "second note");
        assert!(!restored.edited);
    }

    /// One (`text`, error constructor) pair [`FailingReinsertStore`] fails
    /// `insert` on.
    type FailingText = (&'static str, fn() -> rusqlite::Error);

    /// A [`RunningThoughtStore`] wrapping a [`RecordingStore`] whose
    /// `insert` fails for each marked piece of text with the error paired
    /// with it -- a distinct error per text, not one shared `InvalidQuery`
    /// for all of them, so a test asserting on which failure came back
    /// actually exercises that (rather than every marked text producing an
    /// indistinguishable error `matches!` would accept regardless of which
    /// one is reported) -- everything else delegated straight through, the
    /// minimal seam needed to exercise a restore failure without touching
    /// real SQLite.
    struct FailingReinsertStore {
        inner: RecordingStore,
        fails_for_texts: &'static [FailingText],
    }

    impl RunningThoughtStore for FailingReinsertStore {
        fn insert(&self, thought: NewRunningThought) -> rusqlite::Result<i64> {
            if let Some((_, make_err)) = self
                .fails_for_texts
                .iter()
                .find(|(text, _)| *text == thought.text.as_str())
            {
                return Err(make_err());
            }
            self.inner.insert(thought)
        }

        fn get(&self, id: i64) -> rusqlite::Result<Option<RunningThought>> {
            self.inner.get(id)
        }

        fn list(&self, companion_id: i32) -> rusqlite::Result<Vec<RunningThought>> {
            self.inner.list(companion_id)
        }

        fn recent_for(
            &self,
            companion_id: i32,
            speaker_id: &str,
            limit: usize,
        ) -> rusqlite::Result<Vec<RunningThought>> {
            self.inner.recent_for(companion_id, speaker_id, limit)
        }

        fn latest_for(
            &self,
            companion_id: i32,
            speaker_id: &str,
        ) -> rusqlite::Result<Option<RunningThought>> {
            self.inner.latest_for(companion_id, speaker_id)
        }

        fn in_range(
            &self,
            companion_id: i32,
            from: i32,
            through: i32,
        ) -> rusqlite::Result<Vec<RunningThought>> {
            self.inner.in_range(companion_id, from, through)
        }

        fn update_text(&self, id: i64, text: &str) -> rusqlite::Result<()> {
            self.inner.update_text(id, text)
        }

        fn delete(&self, id: i64) -> rusqlite::Result<()> {
            self.inner.delete(id)
        }

        fn delete_from(
            &self,
            companion_id: i32,
            message_id: i32,
        ) -> rusqlite::Result<Vec<RunningThought>> {
            self.inner.delete_from(companion_id, message_id)
        }
    }

    #[test]
    fn a_failed_restore_still_restores_every_other_row_instead_of_stopping_at_the_first_failure() {
        let store = FailingReinsertStore {
            inner: RecordingStore::new(),
            fails_for_texts: &[("round two", || rusqlite::Error::InvalidQuery)],
        };
        seed(&store.inner, "char", 1, 3, "round one");
        seed(&store.inner, "char", 4, 6, "round two");
        seed(&store.inner, "char", 7, 9, "round three");

        // The first row regenerates; the second's generation itself fails,
        // triggering the restore path. Restoring "round two" fails (the
        // seam above), but "round three" must still come back.
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
            &mut RecordingSink::default(),
        )
        .unwrap_err();

        // The restore failure is what gets reported, not the original
        // `Generate` error it happened while handling.
        assert!(matches!(err, ThoughtRegenerateError::Store(_)));

        let after = store.list(1).unwrap();
        assert!(
            after
                .iter()
                .any(|t| t.from_message_id == 1 && t.text == "fresh"),
            "round one should still have regenerated"
        );
        assert!(
            !after.iter().any(|t| t.from_message_id == 4),
            "round two's restore failed, so it is genuinely gone"
        );
        let restored_three = after
            .iter()
            .find(|t| t.from_message_id == 7)
            .expect("round three must still be restored even though round two's restore failed");
        assert_eq!(restored_three.text, "round three");
    }

    #[test]
    fn a_failed_restore_of_a_non_owned_row_still_restores_every_later_row() {
        // PR #227 review, round 3: the non-owned-speaker branch used `?`
        // and returned immediately on a failed `reinsert_original`, losing
        // every row after it -- the same bug the failure-arm test above
        // pins, just in the other branch.
        let store = FailingReinsertStore {
            inner: RecordingStore::new(),
            fails_for_texts: &[("bot1's own note", || rusqlite::Error::InvalidQuery)],
        };
        seed(&store.inner, "char", 1, 3, "round one");
        store
            .inner
            .insert(NewRunningThought {
                companion_id: 1,
                speaker_id: "bot1".to_string(),
                from_message_id: 4,
                through_message_id: 6,
                text: "bot1's own note".to_string(),
                edited: false,
            })
            .unwrap();
        seed(&store.inner, "char", 7, 9, "round three");

        let err = regenerate_from(
            &store,
            &mut |speaker, from, through| Ok(inputs_of(speaker, from, through)),
            &mut always_succeeds(),
            &a_request(1),
            &mut RecordingSink::default(),
        )
        .unwrap_err();

        assert!(matches!(err, ThoughtRegenerateError::Store(_)));

        let after = store.list(1).unwrap();
        assert!(
            after
                .iter()
                .any(|t| t.from_message_id == 1 && t.text.contains("fresh note")),
            "round one (owned by char, first in the loop) should still have regenerated"
        );
        assert!(
            !after.iter().any(|t| t.speaker_id == "bot1"),
            "bot1's row's own restore failed, so it is genuinely gone"
        );
        let restored_three = after
            .iter()
            .find(|t| t.from_message_id == 7)
            .expect("round three must still be restored even though bot1's restore failed");
        assert_eq!(restored_three.text, "round three");
    }

    #[test]
    fn when_a_non_owned_rows_restore_and_a_later_restore_both_fail_the_non_owned_rows_error_is_reported(
    ) {
        // PR #227 review, round 4: the non-owned-speaker branch's own
        // failure (`e`) was discarded unlogged whenever the subsequent
        // `restore_all(&captured[index + 1..])` call also failed --
        // `later_failure.unwrap_or(e)` silently prefers the later error.
        // `e` is chronologically first, so it is the one that must be
        // logged and returned; `round three`'s own failure still needs
        // `restore_all` to run (for its own side effects/logging) even
        // though this row's error is what gets reported.
        let store = FailingReinsertStore {
            inner: RecordingStore::new(),
            fails_for_texts: &[
                ("bot1's own note", || rusqlite::Error::InvalidQuery),
                ("round three", || rusqlite::Error::QueryReturnedNoRows),
            ],
        };
        seed(&store.inner, "char", 1, 3, "round one");
        store
            .inner
            .insert(NewRunningThought {
                companion_id: 1,
                speaker_id: "bot1".to_string(),
                from_message_id: 4,
                through_message_id: 6,
                text: "bot1's own note".to_string(),
                edited: false,
            })
            .unwrap();
        seed(&store.inner, "char", 7, 9, "round three");

        let err = regenerate_from(
            &store,
            &mut |speaker, from, through| Ok(inputs_of(speaker, from, through)),
            &mut always_succeeds(),
            &a_request(1),
            &mut RecordingSink::default(),
        )
        .unwrap_err();

        // The two marked texts fail with distinct errors specifically so
        // this assertion can tell bot1's own (chronologically first)
        // failure apart from round three's later one -- a `Store(_)` match
        // alone would pass even if the code regressed to reporting the
        // later error instead.
        assert!(
            matches!(
                err,
                ThoughtRegenerateError::Store(rusqlite::Error::InvalidQuery)
            ),
            "expected bot1's own InvalidQuery to be reported, got {err:?}"
        );

        let after = store.list(1).unwrap();
        assert!(
            after
                .iter()
                .any(|t| t.from_message_id == 1 && t.text.contains("fresh note")),
            "round one should still have regenerated"
        );
        assert!(
            !after.iter().any(|t| t.speaker_id == "bot1"),
            "bot1's row's own restore failed, so it is genuinely gone"
        );
        assert!(
            !after.iter().any(|t| t.from_message_id == 7),
            "round three's restore also failed, so it is genuinely gone too"
        );
    }
}
