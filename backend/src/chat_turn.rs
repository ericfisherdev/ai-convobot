//! Owns the turn lifecycle shared between the prompting HTTP handlers
//! (`/api/prompt` and `/api/prompt/stream`): insert the user's turn, generate
//! a reply from it, then score and persist the turn's attitude effect.
//!
//! # Why a two-step type, not one function
//!
//! The streaming handler inserts the user's turn inside an `off_worker`
//! closure but generates on a separately spawned thread, so a single
//! function covering "insert then generate" would have to move the insert
//! onto the generation thread too, turning a pre-stream failure (500) into
//! an SSE error chunk instead. Splitting the sequence into [`PendingTurn::begin`]
//! (the insert) and [`PendingTurn::complete`] (generate, then score and
//! persist) lets each handler run the two steps on whichever thread it
//! already uses, while the type system still enforces the order —
//! `complete` needs a `PendingTurn`, which only `begin` can produce — and
//! that generation is attempted at most once, since `complete` consumes
//! `self`.

use crate::attitude_engine::{LexiconScorer, ScorerConfig, TurnScorer};
use crate::database::{CompanionAttitude, Database, NewMessage};
use crate::turn_slot::TurnGuard;

/// The persistence seam between [`PendingTurn`] and the database, so the
/// turn lifecycle can be unit-tested against an in-memory store instead of
/// the hardwired `companion_database.db` (`Database::open()` has no path
/// parameter).
pub trait TurnStore {
    /// Pre-processing shared by both prompting handlers: third-party mention
    /// tracking, new-person detection and interaction detection. Returns the
    /// prompt to generate from when an interaction with a recorded outcome
    /// matched, so the caller can generate with that added context.
    fn preprocess(&self, user_message: &str, companion_id: i32) -> Option<String>;

    /// Persists the user's half of the turn.
    fn insert_user_turn(&self, content: &str) -> rusqlite::Result<()>;

    /// Scores the turn and persists the resulting attitude change.
    ///
    /// An attitude failure must never fail the chat reply, so every error
    /// path is logged internally and this returns `None` rather than
    /// propagating. On success, returns the (previous, current) attitude
    /// pair for the user target.
    fn finish_turn(
        &self,
        companion_id: i32,
        user_id: i32,
        user_message: &str,
        companion_reply: &str,
    ) -> Option<(CompanionAttitude, CompanionAttitude)>;
}

/// The production [`TurnStore`], backed by `companion_database.db`.
pub struct SqliteTurnStore;

impl TurnStore for SqliteTurnStore {
    fn preprocess(&self, user_message: &str, companion_id: i32) -> Option<String> {
        preprocess_user_message(user_message, companion_id)
    }

    fn insert_user_turn(&self, content: &str) -> rusqlite::Result<()> {
        Database::insert_message(NewMessage {
            ai: false,
            content: content.to_string(),
        })
    }

    fn finish_turn(
        &self,
        companion_id: i32,
        user_id: i32,
        user_message: &str,
        companion_reply: &str,
    ) -> Option<(CompanionAttitude, CompanionAttitude)> {
        finish_turn(companion_id, user_id, user_message, companion_reply)
    }
}

/// Pre-processing shared by `/api/prompt` and `/api/prompt/stream`: third-party
/// mention tracking, new-person detection and interaction detection.
///
/// Returns the prompt to generate from when an interaction with a recorded
/// outcome matched, so the caller can generate with that added context.
fn preprocess_user_message(user_message: &str, companion_id: i32) -> Option<String> {
    // Track third-party mentions and display console output
    match Database::track_third_party_mentions(user_message) {
        Ok(mention_output) => {
            if !mention_output.is_empty() {
                println!("{}", mention_output);
            }
        }
        Err(e) => eprintln!("Failed to track third-party mentions: {}", e),
    }

    // Automatically detect new persons in the message
    if let Err(e) = Database::detect_new_persons_in_message(user_message, companion_id) {
        eprintln!("Failed to detect persons in message: {}", e);
        // Continue processing even if person detection fails
    }

    // Detect and handle interaction requests
    if let Ok(Some(interaction)) = Database::detect_interaction_request(user_message, companion_id)
    {
        if let Some(outcome) = interaction.outcome.as_ref() {
            let third_party_name = Database::get_third_party_by_id(interaction.third_party_id)
                .ok()
                .flatten()
                .map(|p| p.name)
                .unwrap_or_else(|| "unknown".to_string());
            return Some(format!(
                "{}\n[Context: Interaction with {} - {}]",
                user_message, third_party_name, outcome
            ));
        }
    }

    None
}

/// Longest user-turn excerpt stored on an attitude memory.
pub(crate) const MEMORY_EXCERPT_CHARS: usize = 200;

/// Single-line excerpt of a user turn, for `attitude_memories.message_context`.
///
/// Truncates on a character boundary, so a multi-byte message can never split
/// mid-codepoint.
pub(crate) fn message_excerpt(user_message: &str) -> String {
    let single_line = user_message
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    match single_line.char_indices().nth(MEMORY_EXCERPT_CHARS) {
        Some((byte_index, _)) => format!("{}…", &single_line[..byte_index]),
        None => single_line,
    }
}

/// Derives attitude deltas from one conversation turn and persists them.
///
/// Called after generation, once both sides of the turn are known, from
/// [`PendingTurn::complete`] on both the non-streaming and streaming paths.
/// An attitude failure must never fail the chat reply, so every error path
/// here is logged and returns `None` rather than propagating.
///
/// On success, returns the (previous, current) attitude pair for the user
/// target so callers can report or persist the change (e.g. into the SSE
/// chunk, or as an attitude memory).
fn finish_turn(
    companion_id: i32,
    user_id: i32,
    user_message: &str,
    companion_reply: &str,
) -> Option<(CompanionAttitude, CompanionAttitude)> {
    // `get_attitude` propagates real SQL failures (e.g. a busy write lock) as
    // `Err` rather than collapsing them into `Ok(None)`, so `Ok(None)` here
    // reliably means the row is absent, never "the read failed".
    let current = match Database::get_attitude(companion_id, user_id, "user") {
        Ok(Some(attitude)) => attitude,
        Ok(None) => {
            // Fresh database: seed the row from the companion's persona before
            // scoring, otherwise the UPDATE below would silently touch zero rows.
            // `seed_missing_user_attitude` is insert-only (never falls back to
            // an UPDATE), so if this "row absent" read raced a concurrent
            // writer that has since inserted the real row, the seed silently
            // no-ops instead of wiping accumulated state.
            let persona = match Database::get_companion_data() {
                Ok(companion_data) => companion_data.persona,
                Err(e) => {
                    eprintln!("Failed to load companion persona for attitude seed: {}", e);
                    return None;
                }
            };
            if let Err(e) = Database::seed_missing_user_attitude(companion_id, user_id, &persona) {
                eprintln!("Failed to seed initial user attitude: {}", e);
                return None;
            }
            match Database::get_attitude(companion_id, user_id, "user") {
                Ok(Some(attitude)) => attitude,
                Ok(None) => {
                    eprintln!("Attitude row missing immediately after seeding");
                    return None;
                }
                Err(e) => {
                    eprintln!("Failed to reload seeded attitude: {}", e);
                    return None;
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to load attitude before scoring turn: {}", e);
            return None;
        }
    };

    let persona = match Database::get_companion_data() {
        Ok(companion_data) => companion_data.persona,
        Err(e) => {
            eprintln!(
                "Failed to load companion persona for attitude baseline: {}",
                e
            );
            return None;
        }
    };
    let baseline = Database::adjust_attitude_for_persona(
        &Database::default_user_attitude(companion_id, user_id),
        &persona,
    );

    let scorer = LexiconScorer::new(ScorerConfig::new(baseline));
    let deltas = scorer.evaluate_turn(user_message, companion_reply, &current);

    match Database::apply_attitude_deltas(companion_id, user_id, "user", &deltas) {
        Ok(Some((previous, updated))) => {
            let formatter = crate::attitude_formatter::AttitudeFormatter::new();
            let attitude_changes =
                formatter.format_attitude_changes_for_console(&previous, &updated);
            if !attitude_changes.is_empty() {
                println!("{}", attitude_changes);
            }
            // One memory per turn at most, carrying what the user said so the
            // companion remembers why its feelings moved.
            if let Err(e) = Database::detect_attitude_change(
                companion_id,
                user_id,
                "user",
                &previous,
                &updated,
                Some(&message_excerpt(user_message)),
            ) {
                eprintln!("Failed to record attitude memory: {}", e);
            }
            Some((previous, updated))
        }
        Ok(None) => {
            eprintln!("Attitude row missing when applying deltas after seeding");
            None
        }
        Err(e) => {
            eprintln!("Failed to apply attitude deltas: {}", e);
            None
        }
    }
}

/// A user turn that has been persisted and is waiting for its reply.
///
/// Fields are private and plain data (`String`/`i32`), so a `PendingTurn` is
/// `Send + 'static` and can be returned out of an `off_worker` closure and
/// moved into a separately spawned generation thread.
pub struct PendingTurn {
    companion_id: i32,
    user_id: i32,
    /// What the user actually said. The attitude engine scores the turn
    /// against this, never against `generation_prompt`.
    user_message: String,
    /// The prompt to generate from: the interaction-context prompt when one
    /// matched during `begin`, otherwise `user_message` itself.
    generation_prompt: String,
}

/// The reply half of a turn, once generation and scoring have both run.
pub struct CompletedTurn {
    pub reply: String,
    pub attitude: Option<(CompanionAttitude, CompanionAttitude)>,
}

impl PendingTurn {
    /// Persists the user's turn and prepares the prompt to generate from.
    ///
    /// `_claimed` is compile-time proof that the turn slot was claimed
    /// before this call (the #95 invariant: the insert must never happen
    /// outside a claimed turn slot); the guard itself is not stored. Errors
    /// are the insert's `rusqlite::Error`; a `preprocess` failure is logged
    /// internally by the store and never fails the turn.
    pub fn begin(
        _claimed: &TurnGuard,
        store: &impl TurnStore,
        companion_id: i32,
        user_id: i32,
        user_message: String,
    ) -> rusqlite::Result<PendingTurn> {
        let interaction_prompt = store.preprocess(&user_message, companion_id);
        store.insert_user_turn(&user_message)?;
        let generation_prompt = interaction_prompt.unwrap_or_else(|| user_message.clone());
        Ok(PendingTurn {
            companion_id,
            user_id,
            user_message,
            generation_prompt,
        })
    }

    /// Generates the reply and, on success, scores and persists the turn's
    /// attitude effect.
    ///
    /// On `Err`, the store is never touched: no second insert, no second
    /// generation, no `finish_turn` call. This is the #84 regression guard.
    /// Consuming `self` is what makes "generation is attempted at most
    /// once" a compile error to violate.
    pub fn complete(
        self,
        store: &impl TurnStore,
        generate: impl FnOnce(&str) -> std::io::Result<String>,
    ) -> std::io::Result<CompletedTurn> {
        let reply = generate(&self.generation_prompt)?;
        let attitude =
            store.finish_turn(self.companion_id, self.user_id, &self.user_message, &reply);
        Ok(CompletedTurn { reply, attitude })
    }
}

#[cfg(test)]
pub(crate) struct RecordingStore {
    pub(crate) inserted: std::sync::Mutex<Vec<String>>,
    pub(crate) finished: std::sync::Mutex<Vec<(String, String)>>,
    interaction_prompt: Option<String>,
}

#[cfg(test)]
impl RecordingStore {
    pub(crate) fn new(interaction_prompt: Option<String>) -> Self {
        Self {
            inserted: std::sync::Mutex::new(Vec::new()),
            finished: std::sync::Mutex::new(Vec::new()),
            interaction_prompt,
        }
    }
}

#[cfg(test)]
impl TurnStore for RecordingStore {
    fn preprocess(&self, _user_message: &str, _companion_id: i32) -> Option<String> {
        self.interaction_prompt.clone()
    }

    fn insert_user_turn(&self, content: &str) -> rusqlite::Result<()> {
        self.inserted.lock().unwrap().push(content.to_string());
        Ok(())
    }

    fn finish_turn(
        &self,
        _companion_id: i32,
        _user_id: i32,
        user_message: &str,
        companion_reply: &str,
    ) -> Option<(CompanionAttitude, CompanionAttitude)> {
        self.finished
            .lock()
            .unwrap()
            .push((user_message.to_string(), companion_reply.to_string()));
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn_slot::TurnSlot;

    #[test]
    fn begin_inserts_the_user_turn_exactly_once_and_generates_from_the_interaction_prompt() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(Some("augmented prompt".to_string()));

        let pending = PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string())
            .expect("insert should succeed");

        assert_eq!(*store.inserted.lock().unwrap(), vec!["hello".to_string()]);

        let mut seen_prompt = None;
        pending
            .complete(&store, |prompt| {
                seen_prompt = Some(prompt.to_string());
                Ok("reply".to_string())
            })
            .expect("generation should succeed");

        assert_eq!(seen_prompt.as_deref(), Some("augmented prompt"));
    }

    #[test]
    fn failed_generation_leaves_one_user_turn_and_never_finishes() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);

        let pending = PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string())
            .expect("insert should succeed");

        let result = pending.complete(&store, |_prompt| Err(std::io::Error::other("no model")));

        assert!(result.is_err());
        assert_eq!(store.inserted.lock().unwrap().len(), 1);
        assert!(store.finished.lock().unwrap().is_empty());
    }

    #[test]
    fn successful_generation_finishes_the_turn_against_what_the_user_said() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(Some("augmented prompt".to_string()));

        let pending = PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string())
            .expect("insert should succeed");

        let completed = pending
            .complete(&store, |_prompt| Ok("reply".to_string()))
            .expect("generation should succeed");

        assert_eq!(completed.reply, "reply");
        assert_eq!(
            *store.finished.lock().unwrap(),
            vec![("hello".to_string(), "reply".to_string())]
        );
    }
}
