//! Owns the turn lifecycle shared between the prompting HTTP handlers
//! (`/api/prompt` and `/api/prompt/stream`): insert the user's turn, then
//! generate and persist each speaker's reply in turn (#131's round
//! orchestrator, `multiplayer::round::run_round`), and finally score and
//! persist the turn's attitude effect.
//!
//! # Why a two-step type, not one function
//!
//! The streaming handler inserts the user's turn inside an `off_worker`
//! closure but generates on a separately spawned thread, so a single
//! function covering "insert then generate" would have to move the insert
//! onto the generation thread too, turning a pre-stream failure (500) into
//! an SSE error chunk instead. Splitting the sequence into [`PendingTurn::begin`]
//! (the insert) and [`PendingTurn::reply`]/[`PendingTurn::finish`] (generate
//! and persist one speaker's reply; once the round is over, score and
//! persist the attitude effect) lets each handler run the steps on whichever
//! thread it already uses, while the type system still enforces the order —
//! `reply` and `finish` need a `PendingTurn`, which only `begin` can produce
//! — and that scoring is attempted at most once per round, since `finish`
//! consumes `self`.

use crate::attitude_engine::{LexiconScorer, ScorerConfig, TurnScorer};
use crate::database::{CompanionAttitude, Database, Message, NewMessage};
use crate::participants::{normalise_mentions, ParticipantId, ParticipantRegistry};
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

    /// Persists the user's half of the turn and returns the new message's
    /// id, for [`PendingTurn::user_message_id`] — #154's round orchestrator
    /// reads it back through [`TurnStore::get_message`] to broadcast the
    /// user's turn to every joiner.
    fn insert_user_turn(&self, content: &str) -> rusqlite::Result<i32>;

    /// Persists one speaker's reply and returns the new message's id, for
    /// [`PersistedReply::message_id`].
    fn insert_reply(&self, speaker_id: &ParticipantId, content: &str) -> rusqlite::Result<i32>;

    /// The newest `limit` messages, oldest first — what a round hands a
    /// remote speaker as its view of the conversation so far (including any
    /// replies earlier in the same round).
    fn transcript_tail(&self, limit: usize) -> rusqlite::Result<Vec<Message>>;

    /// The full persisted row for a message id `insert_user_turn` or
    /// `insert_reply` just returned — #154's round orchestrator uses this to
    /// fetch the row it broadcasts to every joiner as `ServerFrame::Message`,
    /// so a joiner's mirror carries the exact same id and `created_at` the
    /// host database recorded.
    fn get_message(&self, id: i32) -> rusqlite::Result<Message>;

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
///
/// Carries the current chat's participant display names (constructor
/// injection from a snapshot of the shared registry) so `preprocess` never
/// mistakes the user, the host companion or a joined bot for a new
/// third-party person mentioned in the conversation.
pub struct SqliteTurnStore {
    participant_names: Vec<String>,
}

impl SqliteTurnStore {
    pub fn new(participant_names: Vec<String>) -> Self {
        SqliteTurnStore { participant_names }
    }
}

impl TurnStore for SqliteTurnStore {
    fn preprocess(&self, user_message: &str, companion_id: i32) -> Option<String> {
        preprocess_user_message(user_message, companion_id, &self.participant_names)
    }

    fn insert_user_turn(&self, content: &str) -> rusqlite::Result<i32> {
        Database::insert_message(NewMessage::from_user(content))
    }

    fn insert_reply(&self, speaker_id: &ParticipantId, content: &str) -> rusqlite::Result<i32> {
        Database::insert_message(NewMessage::new(speaker_id.to_string(), content))
    }

    fn get_message(&self, id: i32) -> rusqlite::Result<Message> {
        Database::get_message(id)
    }

    fn transcript_tail(&self, limit: usize) -> rusqlite::Result<Vec<Message>> {
        Database::get_x_messages(limit, 0)
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
/// `excluded_names` are the current chat's participant display names (user,
/// host companion, any joined bots), so none of them is ever mistaken for a
/// newly mentioned third party.
///
/// Returns the prompt to generate from when an interaction with a recorded
/// outcome matched, so the caller can generate with that added context.
fn preprocess_user_message(
    user_message: &str,
    companion_id: i32,
    excluded_names: &[String],
) -> Option<String> {
    // Track third-party mentions and display console output
    match Database::track_third_party_mentions(user_message, excluded_names) {
        Ok(mention_output) => {
            if !mention_output.is_empty() {
                println!("{}", mention_output);
            }
        }
        Err(e) => eprintln!("Failed to track third-party mentions: {}", e),
    }

    // Automatically detect new persons in the message
    if let Err(e) =
        Database::detect_new_persons_in_message(user_message, companion_id, excluded_names)
    {
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
    /// What the user actually said, already normalised to `@id` mention
    /// form. The attitude engine scores the turn against this, never
    /// against `generation_prompt`.
    user_message: String,
    /// The prompt to generate from: the interaction-context prompt when one
    /// matched during `begin`, otherwise `user_message` itself.
    generation_prompt: String,
    /// The chat's participant snapshot `begin` was called with, kept so
    /// [`PendingTurn::reply`] can normalise every speaker's `@mention`s to
    /// `@id` form before it persists them, regardless of which model
    /// produced the text.
    registry: ParticipantRegistry,
    /// The id `store.insert_user_turn` gave the user's turn — what
    /// [`PendingTurn::user_message_id`] exposes so #154's round orchestrator
    /// can look the row back up (`TurnStore::get_message`) and broadcast it
    /// to every joiner before the round's first speaker generates.
    user_message_id: i32,
}

/// One speaker's reply, once generated and persisted: the new message's id,
/// who said it, and the cleaned text. `message_id` is what #133's
/// `reply_complete` SSE chunks carry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistedReply {
    pub message_id: i32,
    pub speaker_id: ParticipantId,
    pub text: String,
}

impl PendingTurn {
    /// Persists the user's turn and prepares the prompt to generate from.
    ///
    /// `user_message` is normalised to `@id` mention form (`registry`,
    /// #126/#132) before `preprocess` and the insert, so the stored row, the
    /// attitude scorer input and `generation_prompt` all carry `@id` rather
    /// than whatever display-name form the client sent.
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
        registry: ParticipantRegistry,
    ) -> rusqlite::Result<PendingTurn> {
        let user_message = normalise_mentions(&user_message, &registry);
        let interaction_prompt = store.preprocess(&user_message, companion_id);
        let user_message_id = store.insert_user_turn(&user_message)?;
        let generation_prompt = interaction_prompt.unwrap_or_else(|| user_message.clone());
        Ok(PendingTurn {
            companion_id,
            user_id,
            user_message,
            generation_prompt,
            registry,
            user_message_id,
        })
    }

    /// The id `store.insert_user_turn` gave the user's turn during
    /// [`PendingTurn::begin`] — #154's round orchestrator reads it back
    /// through [`TurnStore::get_message`] to broadcast the row to every
    /// joiner.
    pub fn user_message_id(&self) -> i32 {
        self.user_message_id
    }

    /// Generates one speaker's reply and persists it.
    ///
    /// The generated text is normalised to `@id` mention form (`self.registry`)
    /// before it is persisted, so a model's `@Ada` is stored as `@bot2`
    /// regardless of which speaker produced it — this is the single choke
    /// point both the host bot's reply and every remote reply pass through.
    ///
    /// Takes `&self` rather than consuming it, so a round
    /// (`multiplayer::round::run_round`) can call this once per speaker
    /// while still generating every reply from the same
    /// `generation_prompt` `begin` prepared. On `Err`, the store is never
    /// touched: no insert for this speaker. This is the #84 regression
    /// guard, now scoped to one speaker's attempt instead of the whole turn.
    pub fn reply(
        &self,
        store: &impl TurnStore,
        speaker_id: ParticipantId,
        generate: impl FnOnce(&str) -> std::io::Result<String>,
    ) -> std::io::Result<PersistedReply> {
        let text = generate(&self.generation_prompt)?;
        let text = normalise_mentions(&text, &self.registry);
        let message_id = store
            .insert_reply(&speaker_id, &text)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(PersistedReply {
            message_id,
            speaker_id,
            text,
        })
    }

    /// Scores the round and persists the resulting attitude effect.
    ///
    /// Scores against what the user said (`self.user_message`), never
    /// against `generation_prompt`. `host_reply` is `None` when the host
    /// companion did not speak this round (a mention-filtered round, #132),
    /// in which case this returns `None` without touching the store.
    /// Consuming `self` is what makes "scoring is attempted at most once per
    /// round" a compile error to violate.
    pub fn finish(
        self,
        store: &impl TurnStore,
        host_reply: Option<&str>,
    ) -> Option<(CompanionAttitude, CompanionAttitude)> {
        let host_reply = host_reply?;
        store.finish_turn(
            self.companion_id,
            self.user_id,
            &self.user_message,
            host_reply,
        )
    }
}

#[cfg(test)]
pub(crate) struct RecordingStore {
    pub(crate) inserted: std::sync::Mutex<Vec<String>>,
    pub(crate) replies: std::sync::Mutex<Vec<(ParticipantId, String)>>,
    pub(crate) finished: std::sync::Mutex<Vec<(String, String)>>,
    /// Every insert (`insert_user_turn` and `insert_reply`), in call order —
    /// what `transcript_tail` reads its tail from, so a round test can
    /// assert a later speaker saw an earlier one's reply.
    log: std::sync::Mutex<Vec<Message>>,
    interaction_prompt: Option<String>,
}

#[cfg(test)]
impl RecordingStore {
    pub(crate) fn new(interaction_prompt: Option<String>) -> Self {
        Self {
            inserted: std::sync::Mutex::new(Vec::new()),
            replies: std::sync::Mutex::new(Vec::new()),
            finished: std::sync::Mutex::new(Vec::new()),
            log: std::sync::Mutex::new(Vec::new()),
            interaction_prompt,
        }
    }

    /// Appends `speaker_id`/`content` to `log` and returns the row id it was
    /// given, shared by `insert_user_turn` and `insert_reply` so the two can
    /// never assign a duplicate id.
    fn log_message(&self, speaker_id: &ParticipantId, content: &str) -> i32 {
        let mut log = self.log.lock().unwrap();
        let id = log.len() as i32 + 1;
        log.push(Message {
            id,
            ai: speaker_id != &ParticipantId::USER,
            speaker_id: speaker_id.to_string(),
            content: content.to_string(),
            created_at: String::new(),
        });
        id
    }
}

#[cfg(test)]
impl TurnStore for RecordingStore {
    fn preprocess(&self, _user_message: &str, _companion_id: i32) -> Option<String> {
        self.interaction_prompt.clone()
    }

    fn insert_user_turn(&self, content: &str) -> rusqlite::Result<i32> {
        self.inserted.lock().unwrap().push(content.to_string());
        Ok(self.log_message(&ParticipantId::USER, content))
    }

    fn insert_reply(&self, speaker_id: &ParticipantId, content: &str) -> rusqlite::Result<i32> {
        self.replies
            .lock()
            .unwrap()
            .push((speaker_id.clone(), content.to_string()));
        Ok(self.log_message(speaker_id, content))
    }

    fn get_message(&self, id: i32) -> rusqlite::Result<Message> {
        self.log
            .lock()
            .unwrap()
            .iter()
            .find(|m| m.id == id)
            .cloned()
            .ok_or(rusqlite::Error::QueryReturnedNoRows)
    }

    fn transcript_tail(&self, limit: usize) -> rusqlite::Result<Vec<Message>> {
        let log = self.log.lock().unwrap();
        let start = log.len().saturating_sub(limit);
        Ok(log[start..].to_vec())
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

    /// `user` ("TestUser") and `char` ("TestCompanion") only, as the
    /// Implementation Plan for #132's normalisation tests specifies.
    fn solo_registry() -> ParticipantRegistry {
        ParticipantRegistry::solo("TestUser", "TestCompanion", None)
    }

    #[test]
    fn begin_inserts_the_user_turn_exactly_once_and_generates_from_the_interaction_prompt() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(Some("augmented prompt".to_string()));

        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), solo_registry())
                .expect("insert should succeed");

        assert_eq!(*store.inserted.lock().unwrap(), vec!["hello".to_string()]);

        let mut seen_prompt = None;
        pending
            .reply(&store, ParticipantId::CHAR, |prompt| {
                seen_prompt = Some(prompt.to_string());
                Ok("reply".to_string())
            })
            .expect("generation should succeed");

        assert_eq!(seen_prompt.as_deref(), Some("augmented prompt"));
    }

    #[test]
    fn begin_normalises_a_display_name_mention_to_its_id_before_inserting() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);

        PendingTurn::begin(
            &guard,
            &store,
            1,
            1,
            "hey @TestCompanion".to_string(),
            solo_registry(),
        )
        .expect("insert should succeed");

        assert_eq!(
            *store.inserted.lock().unwrap(),
            vec!["hey @char".to_string()]
        );
    }

    #[test]
    fn reply_normalises_a_display_name_mention_to_its_id_before_inserting() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);

        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), solo_registry())
                .expect("insert should succeed");

        let persisted = pending
            .reply(&store, ParticipantId::CHAR, |_prompt| {
                Ok("hi @TestCompanion".to_string())
            })
            .expect("generation should succeed");

        assert_eq!(persisted.text, "hi @char");
        assert_eq!(
            *store.replies.lock().unwrap(),
            vec![(ParticipantId::CHAR, "hi @char".to_string())]
        );
    }

    #[test]
    fn failed_generation_leaves_one_user_turn_and_inserts_no_reply() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);

        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), solo_registry())
                .expect("insert should succeed");

        let result = pending.reply(&store, ParticipantId::CHAR, |_prompt| {
            Err(std::io::Error::other("no model"))
        });

        assert!(result.is_err());
        assert_eq!(store.inserted.lock().unwrap().len(), 1);
        assert!(store.replies.lock().unwrap().is_empty());
    }

    #[test]
    fn successful_generation_finishes_the_turn_against_what_the_user_said() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(Some("augmented prompt".to_string()));

        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), solo_registry())
                .expect("insert should succeed");

        let persisted = pending
            .reply(&store, ParticipantId::CHAR, |_prompt| {
                Ok("reply".to_string())
            })
            .expect("generation should succeed");

        assert_eq!(persisted.text, "reply");
        assert_eq!(persisted.speaker_id, ParticipantId::CHAR);

        pending.finish(&store, Some(&persisted.text));

        assert_eq!(
            *store.finished.lock().unwrap(),
            vec![("hello".to_string(), "reply".to_string())]
        );
    }
}
