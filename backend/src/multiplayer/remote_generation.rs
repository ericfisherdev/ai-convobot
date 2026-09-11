//! A joiner's own reply generation (#153, Part B of the joiner — #130 is
//! Part A: connection, handshake, reconnect, transcript mirror).
//!
//! [`LocalModelGeneration`] is the [`GenerateRequestHandler`] `main.rs`
//! wires a joiner up with: it runs the joiner's own model (its own card,
//! config, dialogue tuning, attitudes and long-term memory — every one of
//! those is still read locally, only the participant names come from the
//! host's roster) on the transcript a `GenerateRequest` carried, streaming
//! `Token`s back as they are produced and a `ReplyComplete` once the reply
//! is done, then scores the joiner's own attitude toward the user from the
//! turn it just answered.
//!
//! [`run_remote_turn`] is the frame-sequencing core, free of `ACTIVE_TURN`,
//! `JoinerShared` and any socket, so it is unit-testable with a stub
//! generator and no model, no socket; [`LocalModelGeneration::try_handle`]
//! is the only production caller, claiming the turn slot and spawning the
//! thread [`run_remote_turn`] actually runs on. That same spawned thread,
//! once `run_remote_turn` returns and the reply's own claim on the turn
//! slot has been explicitly released, makes one independent attempt at this
//! joiner's own auto-extraction (#186, `joiner_compaction::maybe_queue_extraction`)
//! — see [`LocalModelGeneration::try_handle`]'s own doc comment for why that
//! ordering is load-bearing.

use std::io;
use std::sync::Arc;

use tokio::sync::mpsc::UnboundedSender;

use crate::chat_turn::{SqliteTurnStore, TurnStore};
use crate::compaction::context::{CompactionContext, QuoteLine};
use crate::compaction::store::SqliteCompactionStore;
use crate::database::{CompanionView, Database, Message, USER_SPEAKER_ID};
use crate::llm::{
    self, CharacterModel, CompactionSource, InMemoryTranscript, PromptSpeakers,
    ResidentCharacterModel, SqliteThoughts,
};
use crate::multiplayer::joiner::{GenerateRequestHandler, JoinerHandle};
use crate::multiplayer::joiner_compaction::{local_overlay, JoinerExtractionJob};
use crate::multiplayer::protocol::{ClientFrame, ContinuityPayload, ParticipantSummary};
use crate::participants::{AvatarRef, Participant, ParticipantId, ParticipantRegistry};
use crate::running_thoughts::generate::{generate_thought_into, ThoughtError};
use crate::running_thoughts::hook::{pending_thought_range, thought_inputs_for_range};
use crate::running_thoughts::store::{RunningThoughtStore, SqliteRunningThoughtStore};
use crate::running_thoughts::types::RunningThought;
use crate::turn_slot::ACTIVE_TURN;

/// The joiner-side [`CompactionSource`] (#186): renders the host's
/// committed summaries/rules (`payload`, #182's [`ContinuityPayload`])
/// together with this joiner's own locally-kept companion overlay and
/// rules (`local_overlay`), so a joiner's own reply is grounded in the same
/// story the host renders for itself, plus whatever this joiner alone has
/// learned about its own character. Replaces #174's placeholder (a fixed,
/// never-compacted `CompactionContext`) `LocalModelGeneration` was wired up
/// with.
pub struct HostContinuity {
    payload: Option<ContinuityPayload>,
    companion_state: Vec<String>,
    rules: Vec<QuoteLine>,
}

impl HostContinuity {
    pub fn new(
        payload: Option<ContinuityPayload>,
        companion_state: Vec<String>,
        rules: Vec<QuoteLine>,
    ) -> Self {
        HostContinuity {
            payload,
            companion_state,
            rules,
        }
    }
}

impl CompactionSource for HostContinuity {
    /// `companion_id` is unused: `payload`/`companion_state`/`rules` are
    /// already scoped to this one joiner's own companion by construction
    /// (there is only ever one, and `local_overlay` was already read
    /// against it before this was built).
    ///
    /// `recalled_facts` is left empty, matching `SqliteCompaction`'s own
    /// current behaviour (`CompactionContext::load` does not fill it in
    /// either, until #178 lands tantivy-backed fact recall).
    fn context(&self, _companion_id: i32) -> std::io::Result<CompactionContext> {
        let mut ctx = match self.payload.clone() {
            Some(payload) => payload.into_context(self.companion_state.clone(), Vec::new()),
            None => CompactionContext {
                companion_state: self.companion_state.clone(),
                ..CompactionContext::default()
            },
        };
        ctx.rules.extend(self.rules.iter().cloned());
        Ok(ctx)
    }
}

/// Builds everything a joiner's own prompt needs from `HostContinuity`
/// onward: the speakers this joiner would generate as, a fresh
/// [`HostContinuity`] read from `handle`'s current state, the transcript
/// mirror a live reply generates from, and the raw payload so a caller can
/// echo it back. The single build site for [`HostContinuity`] — both
/// [`with_local_model`]'s generator (a live reply) and
/// `main.rs::inspect_prompt`'s joiner branch (`GET /api/debug/prompt`) call
/// this rather than rebuilding it inline, so the two can never disagree
/// about what "this joiner's own overlay" means.
pub fn joiner_prompt_inputs(
    handle: &JoinerHandle,
) -> (
    i32,
    PromptSpeakers,
    HostContinuity,
    Vec<Message>,
    Option<ContinuityPayload>,
) {
    let (companion_id, participants, self_id, payload, transcript) = {
        let shared = handle.read().unwrap_or_else(|p| p.into_inner());
        (
            shared.companion_id,
            shared.participants.clone(),
            shared.participant_id.clone(),
            shared.last_continuity.clone(),
            shared.transcript.snapshot(),
        )
    };
    let speakers = PromptSpeakers {
        registry: registry_from_participants(&participants),
        self_id,
    };
    let (companion_state, rules) = local_overlay(&SqliteCompactionStore, companion_id)
        .unwrap_or_else(|e| {
            eprintln!("joiner: failed to read local compaction overlay: {e}");
            (Vec::new(), Vec::new())
        });
    let source = HostContinuity::new(payload.clone(), companion_state, rules);
    (companion_id, speakers, source, transcript, payload)
}

/// The fixed user id every turn is scored against — the same constant every
/// `PendingTurn::begin`/`finish_turn` call site in `main.rs` uses
/// ("Default user ID"). A joiner scores against the same id: there is only
/// ever one human in a chat.
const USER_ID: i32 = 1;

/// Produces one reply for `transcript`/`speakers`, invoking `on_token` with
/// each token as it is produced (mirroring `llm::prompt_streaming`'s own
/// callback contract: runs on the calling thread, must not block).
///
/// Injected through [`LocalModelGeneration::new`] rather than built inside
/// [`LocalModelGeneration::handle`], so a test (or #136's two-instance
/// test, which runs a real host and a real joiner with no model loaded)
/// can supply a stub that never touches a model.
pub type RemoteGenerator = Arc<
    dyn Fn(&[Message], &PromptSpeakers, &mut dyn FnMut(&str)) -> io::Result<String> + Send + Sync,
>;

/// Writes this joiner's own bot's running thought (#220) for the round a
/// `GenerateRequest` just carried, before its reply is generated. Same shape
/// as [`RemoteGenerator`] minus the streaming callback — a thought is never
/// streamed to the host, only ever written to this joiner's own local
/// store — so a test can supply a recording stub with no model, exactly as
/// `RemoteGenerator`'s own stub does.
pub type Thinker = Arc<dyn Fn(&[Message], &PromptSpeakers) + Send + Sync>;

/// The [`GenerateRequestHandler`] `main.rs` wires a joiner up with.
pub struct LocalModelGeneration {
    companion_id: i32,
    self_id: ParticipantId,
    handle: JoinerHandle,
    generator: RemoteGenerator,
    thinker: Thinker,
    extraction: JoinerExtractionJob,
}

impl LocalModelGeneration {
    /// Takes any generator, thinker and extraction job. Used directly by
    /// this module's own tests (a stub generator/thinker that touch no
    /// model, and `joiner_compaction::noop_job()` when a test has no
    /// interest in compaction) and by #136's two-instance test.
    pub fn new(
        companion_id: i32,
        self_id: ParticipantId,
        handle: JoinerHandle,
        generator: RemoteGenerator,
        thinker: Thinker,
        extraction: JoinerExtractionJob,
    ) -> Self {
        LocalModelGeneration {
            companion_id,
            self_id,
            handle,
            generator,
            thinker,
            extraction,
        }
    }

    /// Wraps the production generator and the production
    /// [`JoinerExtractionJob`]. The generator runs the joiner's own model
    /// over `llm::prompt_streaming`, generating from an `InMemoryTranscript`
    /// of the transcript the host sent (never the joiner's own local
    /// `messages` table, which a remote reply never touches) and keyed by
    /// the newest user message in it, the same way a local turn's
    /// long-term memory recall is keyed by what the user just said. Builds
    /// its [`HostContinuity`] through [`joiner_prompt_inputs`] on every
    /// call (discarding the tuple's other elements, already known here) so
    /// a reply and `GET /api/debug/prompt` can never render from two
    /// different builds of "this joiner's own overlay". The extraction job
    /// is [`joiner_compaction::run_joiner_extraction`], run by
    /// [`Self::try_handle`] only after a reply's own turn-slot claim has
    /// already been released (see its doc comment).
    pub fn with_local_model(
        companion_id: i32,
        self_id: ParticipantId,
        handle: JoinerHandle,
    ) -> Self {
        let generator: RemoteGenerator = Arc::new({
            let handle = handle.clone();
            let self_id = self_id.clone();
            move |transcript: &[Message],
                  speakers: &PromptSpeakers,
                  on_token: &mut dyn FnMut(&str)| {
                let prompt = newest_user_message(transcript);
                let (_, _, source, _, _) = joiner_prompt_inputs(&handle);
                llm::prompt_streaming(
                    &prompt,
                    companion_id,
                    on_token,
                    &InMemoryTranscript(transcript.to_vec()),
                    speakers,
                    &source,
                    &joiner_reply_thoughts(&self_id),
                )
            }
        });
        let thinker: Thinker = Arc::new({
            let self_id = self_id.clone();
            move |transcript: &[Message], speakers: &PromptSpeakers| {
                think_and_store(companion_id, &self_id, transcript, speakers);
            }
        });
        let extraction: JoinerExtractionJob = {
            let handle = handle.clone();
            Arc::new(move |request| {
                crate::multiplayer::joiner_compaction::run_joiner_extraction(&handle, request);
            })
        };
        LocalModelGeneration::new(
            companion_id,
            self_id,
            handle,
            generator,
            thinker,
            extraction,
        )
    }

    /// The body of [`GenerateRequestHandler::handle`], returning the
    /// spawned thread's `JoinHandle` so this module's own tests can join it
    /// before asserting the turn slot is free again. `None` when nothing
    /// was spawned (the slot was already claimed, or the spawn itself
    /// failed) — both cases already sent their own `ReplyFailed`.
    ///
    /// After the reply is generated (and *only* after: `turn_guard` is
    /// explicitly dropped first), the spawned thread makes one independent
    /// attempt at this joiner's own auto-extraction
    /// (`joiner_compaction::maybe_queue_extraction`), which claims its own
    /// dedicated `turn_slot::JOINER_EXTRACTION` slot rather than `ACTIVE_TURN`
    /// (PR #204 review finding, twice over): dropping `turn_guard` first
    /// stops extraction from ever winning the race for the *same*
    /// `GenerateRequest`'s reply, but only a slot of its own stops a
    /// *later* `GenerateRequest` — arriving anywhere in extraction's
    /// model-bound extract+merge window, not just on the frame that queued
    /// it — from finding the slot still held and coming back `ReplyFailed`
    /// too. `maybe_queue_extraction` falls back to
    /// `JoinerShared::pending_extraction` if another extraction already
    /// holds its slot.
    fn try_handle(
        &self,
        round_id: u64,
        transcript: Vec<Message>,
        tx: UnboundedSender<ClientFrame>,
    ) -> Option<std::thread::JoinHandle<()>> {
        // Part A's 409 guards already block the local prompting endpoints
        // while a turn is in flight; in practice this only trips on two
        // overlapping `GenerateRequest`s.
        let Some(turn_guard) = ACTIVE_TURN.try_claim() else {
            let _ = tx.send(ClientFrame::ReplyFailed {
                round_id,
                reason: "a local turn is in progress".to_string(),
            });
            return None;
        };

        let speakers = {
            let shared = self.handle.read().unwrap_or_else(|p| p.into_inner());
            PromptSpeakers {
                registry: registry_from_participants(&shared.participants),
                self_id: self.self_id.clone(),
            }
        };
        let generator = Arc::clone(&self.generator);
        let thinker = Arc::clone(&self.thinker);
        let extraction = Arc::clone(&self.extraction);
        let extraction_handle = self.handle.clone();
        let companion_id = self.companion_id;
        // Kept outside the closure below so a failed spawn (which drops the
        // closure, and with it the `tx` moved into it, without running it)
        // still has a sender left to report the failure with.
        let tx_for_failed_spawn = tx.clone();

        let spawn_result = std::thread::Builder::new()
            .name("remote-generation".into())
            .spawn(move || {
                run_remote_turn(
                    round_id,
                    transcript,
                    tx,
                    |transcript| thinker(transcript, &speakers),
                    |transcript, on_token| generator(transcript, &speakers, on_token),
                    |transcript, reply| {
                        let store = SqliteTurnStore::new(Vec::new());
                        score_attitude(&store, companion_id, transcript, reply);
                    },
                );
                drop(turn_guard);
                crate::multiplayer::joiner_compaction::maybe_queue_extraction(
                    &extraction_handle,
                    &extraction,
                );
            });

        match spawn_result {
            Ok(join_handle) => Some(join_handle),
            Err(e) => {
                eprintln!("joiner: failed to spawn remote-generation thread: {}", e);
                let _ = tx_for_failed_spawn.send(ClientFrame::ReplyFailed {
                    round_id,
                    reason: format!("failed to start generation: {e}"),
                });
                None
            }
        }
    }
}

impl GenerateRequestHandler for LocalModelGeneration {
    fn handle(&self, round_id: u64, transcript: Vec<Message>, tx: UnboundedSender<ClientFrame>) {
        self.try_handle(round_id, transcript, tx);
    }
}

/// Runs one joiner reply end to end on the calling thread: writes this
/// joiner's own running thought about `transcript` (#220), generates the
/// reply (streaming a `Token` to `tx` for every `on_token` call), scores the
/// joiner's own attitude against it on success, then sends the terminal
/// frame. A closed `tx` (the socket dropped mid-generation) is ignored
/// exactly as the host's own `stream_round` ignores a hung-up SSE client:
/// every send here is best-effort.
///
/// `think` takes no `Result`: it must already have swallowed its own
/// failure (`LocalModelGeneration::with_local_model`'s production `think_and_store`
/// logs and returns on every error path) — a thought that could not be
/// written must never cost the user this reply, the same #216 rule now
/// applied to a remote turn. Runs before `generate` and before the first
/// `Token` frame is ever sent, on the same thread `try_handle` spawns after
/// `ACTIVE_TURN` is already claimed.
pub(crate) fn run_remote_turn(
    round_id: u64,
    transcript: Vec<Message>,
    tx: UnboundedSender<ClientFrame>,
    think: impl FnOnce(&[Message]),
    generate: impl FnOnce(&[Message], &mut dyn FnMut(&str)) -> io::Result<String>,
    score: impl FnOnce(&[Message], &str),
) {
    think(&transcript);
    let mut on_token = |token: &str| {
        let _ = tx.send(ClientFrame::Token {
            round_id,
            text: token.to_string(),
        });
    };
    match generate(&transcript, &mut on_token) {
        Ok(text) => {
            score(&transcript, &text);
            let _ = tx.send(ClientFrame::ReplyComplete { round_id, text });
        }
        Err(e) => {
            let _ = tx.send(ClientFrame::ReplyFailed {
                round_id,
                reason: e.to_string(),
            });
        }
    }
}

/// Scores the joiner's own attitude toward the user from the turn it just
/// answered: the newest `speaker_id == "user"` row in `transcript`, and
/// `reply`. Skipped silently when no user turn exists — a round can ask a
/// remote speaker to answer before the user has said anything in it (a
/// mention follow-up), and there is nothing to score in that case.
///
/// Any `finish_turn` failure is already logged and swallowed inside it; the
/// reply has already been sent by the time this runs, so attitude scoring
/// can never fail a remote reply.
fn score_attitude(store: &impl TurnStore, companion_id: i32, transcript: &[Message], reply: &str) {
    let Some(user_turn) = transcript
        .iter()
        .rev()
        .find(|m| m.speaker_id == USER_SPEAKER_ID)
    else {
        return;
    };
    store.finish_turn(companion_id, USER_ID, &user_turn.content, reply);
}

/// The [`llm::ThoughtSource`] this joiner's own reply reads from: its own
/// bot's chain, scoped by `self_id` — never `char`'s or another bot's. Split
/// out as its own function so the wiring (which id `with_local_model` feeds
/// `SqliteThoughts`) is testable without a model or a real `Database`; the
/// scoping itself (`SqliteThoughts::recent` -> `RunningThoughtStore::recent_for`)
/// is already covered by `running_thoughts::store`'s own tests.
fn joiner_reply_thoughts(self_id: &ParticipantId) -> SqliteThoughts {
    SqliteThoughts {
        speaker: self_id.clone(),
    }
}

/// Writes this joiner's own bot's running thought (#220) about `transcript`
/// — the same `GenerateRequest.transcript` its reply is about to generate
/// from, so a bot speaking after `char` in the same round writes its
/// thought about `char`'s reply too, and its own reply in turn reads that
/// thought back through [`joiner_reply_thoughts`]. Applies #186's
/// per-instance ownership rule to thoughts: a joiner only ever writes its
/// own bot's row, with its own model, into its own local `running_thoughts`
/// table (`speaker_id = self_id`, `companion_id`).
///
/// The `Database`-touching wrapper around [`think_into`], which is where the
/// actual `latest_for` -> [`pending_thought_range`] -> [`generate_thought_into`]
/// sequence lives, seam-based and unit-tested. A silent no-op when running
/// thoughts are disabled, when reading the config or the companion's own
/// data fails, or on any [`ThoughtError`] `think_into` returns: every error
/// is logged and swallowed here, exactly as [`crate::chat_turn::PendingTurn::think`]
/// does for the host, so a failed thought never costs this joiner its reply.
fn think_and_store(
    companion_id: i32,
    self_id: &ParticipantId,
    transcript: &[Message],
    speakers: &PromptSpeakers,
) {
    let config = match Database::get_config() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("running thoughts: joiner failed to read config: {e}");
            return;
        }
    };
    if !config.running_thoughts_enabled {
        return;
    }

    let companion = match Database::get_companion_data() {
        Ok(companion) => companion,
        Err(e) => {
            eprintln!("running thoughts: joiner failed to read its companion data: {e}");
            return;
        }
    };

    if let Err(e) = think_into(
        &SqliteRunningThoughtStore,
        companion_id,
        self_id,
        transcript,
        &companion,
        speakers,
        &ResidentCharacterModel,
    ) {
        eprintln!("running thoughts: {e}");
    }
}

/// The seam-based half of [`think_and_store`]: `store.latest_for` ->
/// [`pending_thought_range`] -> [`thought_inputs_for_range`] ->
/// [`generate_thought_into`], with no `Database` reads of its own — a test
/// drives it with `running_thoughts::store::RecordingStore` and
/// `llm::FakeCharacterModel`, the same doubles `generate.rs`'s own tests
/// use. `Ok(None)` (not an error) is [`pending_thought_range`]'s "nothing
/// new" case, e.g. a regenerate that resends a transcript whose newest id
/// this speaker already covered.
fn think_into(
    store: &dyn RunningThoughtStore,
    companion_id: i32,
    self_id: &ParticipantId,
    transcript: &[Message],
    companion: &CompanionView,
    speakers: &PromptSpeakers,
    model: &dyn CharacterModel,
) -> Result<Option<RunningThought>, ThoughtError> {
    let latest = store
        .latest_for(companion_id, self_id.as_str())
        .map_err(ThoughtError::Store)?;
    let Some((from, through)) =
        pending_thought_range(latest.map(|t| t.through_message_id), transcript)
    else {
        return Ok(None);
    };

    let inputs = thought_inputs_for_range(
        store,
        &InMemoryTranscript(transcript.to_vec()),
        companion_id,
        self_id,
        from,
        through,
    )
    .map_err(ThoughtError::Inputs)?;

    generate_thought_into(store, &inputs, companion, speakers, model).map(Some)
}

/// The content of the newest `transcript` row from the user, or an empty
/// string when there is none — the same `user_message` `assemble_prompt`
/// keys long-term memory recall from for a local turn.
fn newest_user_message(transcript: &[Message]) -> String {
    transcript
        .iter()
        .rev()
        .find(|m| m.speaker_id == USER_SPEAKER_ID)
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

/// Builds a [`ParticipantRegistry`] from the roster [`JoinerShared`]
/// (`multiplayer::joiner`) keeps current from `Joined`/`ParticipantJoined`/
/// `ParticipantLeft` (Part A), so a request built from it always reflects
/// who is actually in the chat right now, not who was in it when this
/// joiner connected.
///
/// [`JoinerShared`]: crate::multiplayer::joiner::JoinerShared
pub(crate) fn registry_from_participants(
    participants: &[ParticipantSummary],
) -> ParticipantRegistry {
    let user_name = participants
        .iter()
        .find(|p| p.id == ParticipantId::USER)
        .map(|p| p.display_name.as_str())
        .unwrap_or_else(|| ParticipantId::USER.as_str());
    let char_summary = participants.iter().find(|p| p.id == ParticipantId::CHAR);
    let char_name = char_summary
        .map(|p| p.display_name.as_str())
        .unwrap_or_else(|| ParticipantId::CHAR.as_str());
    let char_avatar = char_summary
        .and_then(|p| p.avatar_url.clone())
        .map(AvatarRef::new);

    let mut registry = ParticipantRegistry::solo(user_name, char_name, char_avatar);
    for p in participants {
        if p.id == ParticipantId::USER || p.id == ParticipantId::CHAR {
            continue;
        }
        // `JoinerShared::participants` never carries a duplicate id
        // (`joiner::serve` only pushes a `ParticipantJoined` summary once
        // per id), so `insert` never actually fails here; ignoring its
        // `Result` just means a hypothetical duplicate is dropped instead
        // of panicking the generation thread over it.
        let _ = registry.insert(Participant {
            id: p.id.clone(),
            display_name: p.display_name.clone(),
            kind: p.kind.clone(),
            avatar: p.avatar_url.clone().map(AvatarRef::new),
        });
    }
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_turn::RecordingStore;
    use crate::llm::FakeCharacterModel;
    use crate::multiplayer::joiner::JoinerShared;
    use crate::multiplayer::protocol::AvatarUpload;
    use crate::participants::ParticipantKind;
    use crate::running_thoughts::store::RecordingStore as RecordingThoughtStore;
    use crate::running_thoughts::types::NewRunningThought;
    use std::sync::{Mutex, RwLock};
    use tokio::sync::mpsc;

    fn sample_message(id: i32, speaker_id: &str, content: &str) -> Message {
        Message {
            id,
            ai: speaker_id != USER_SPEAKER_ID,
            speaker_id: speaker_id.to_string(),
            content: content.to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    fn drain(rx: &mut mpsc::UnboundedReceiver<ClientFrame>) -> Vec<ClientFrame> {
        let mut frames = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            frames.push(frame);
        }
        frames
    }

    // -- joiner_reply_thoughts: which speaker a joiner's own reply reads --

    #[test]
    fn joiner_reply_thoughts_scopes_by_this_joiners_own_id_not_char() {
        let self_id = ParticipantId::parse("bot1").unwrap();

        let source = joiner_reply_thoughts(&self_id);

        assert_eq!(source.speaker, self_id);
        assert_ne!(source.speaker, ParticipantId::CHAR);
    }

    // -- think_into: the joiner's own thought-writing seam, no model or Database --

    fn a_companion() -> CompanionView {
        CompanionView {
            name: "Ada".to_string(),
            persona: "a curious, upbeat persona".to_string(),
            example_dialogue: String::new(),
            first_message: String::new(),
            long_term_mem: 0,
            short_term_mem: 0,
            roleplay: false,
            dialogue_tuning: false,
            avatar_path: String::new(),
        }
    }

    fn bot1_speakers() -> PromptSpeakers {
        PromptSpeakers {
            registry: ParticipantRegistry::solo("Alice", "Bob", None),
            self_id: ParticipantId::parse("bot1").unwrap(),
        }
    }

    fn a_prior_thought(speaker_id: &ParticipantId, through: i32) -> NewRunningThought {
        NewRunningThought {
            companion_id: 1,
            speaker_id: speaker_id.to_string(),
            from_message_id: 1,
            through_message_id: through,
            text: "an earlier note".to_string(),
            edited: false,
        }
    }

    #[test]
    fn think_into_skips_a_regenerate_over_a_range_this_speaker_already_covered() {
        let store = RecordingThoughtStore::new();
        let bot1 = ParticipantId::parse("bot1").unwrap();
        store.insert(a_prior_thought(&bot1, 2)).unwrap();
        let transcript = vec![
            sample_message(1, USER_SPEAKER_ID, "hi"),
            sample_message(2, "char", "hello"),
        ];
        let model = FakeCharacterModel::returning(Vec::<io::Result<String>>::new());

        let result = think_into(
            &store,
            1,
            &bot1,
            &transcript,
            &a_companion(),
            &bot1_speakers(),
            &model,
        )
        .expect("reading state and skipping must not itself error");

        assert_eq!(
            result, None,
            "bot1 already covered through id 2, so nothing new"
        );
    }

    #[test]
    fn think_into_is_not_suppressed_by_another_speakers_prior_row() {
        let store = RecordingThoughtStore::new();
        let bot1 = ParticipantId::parse("bot1").unwrap();
        // char has a prior note over the same range; bot1 has never thought.
        store
            .insert(a_prior_thought(&ParticipantId::CHAR, 2))
            .unwrap();
        let transcript = vec![
            sample_message(1, USER_SPEAKER_ID, "hi"),
            sample_message(2, "char", "hello"),
        ];
        let model = FakeCharacterModel::returning([Ok("I'm glad they said hi.".to_string())]);

        let result = think_into(
            &store,
            1,
            &bot1,
            &transcript,
            &a_companion(),
            &bot1_speakers(),
            &model,
        )
        .expect("generation should succeed")
        .expect("bot1 has never thought yet, so a row should be written");

        assert_eq!(result.speaker_id, bot1.to_string());
        assert_eq!((result.from_message_id, result.through_message_id), (1, 2));
    }

    #[test]
    fn think_into_with_no_prior_row_writes_one_under_this_speakers_own_id() {
        let store = RecordingThoughtStore::new();
        let bot1 = ParticipantId::parse("bot1").unwrap();
        let transcript = vec![sample_message(1, USER_SPEAKER_ID, "hi")];
        let model = FakeCharacterModel::returning([Ok("bot1's own first note".to_string())]);

        let result = think_into(
            &store,
            1,
            &bot1,
            &transcript,
            &a_companion(),
            &bot1_speakers(),
            &model,
        )
        .expect("generation should succeed")
        .expect("no prior row, so a thought should be written");

        assert_eq!(result.speaker_id, bot1.to_string());
        assert_eq!(result.text, "bot1's own first note");
        assert_eq!((result.from_message_id, result.through_message_id), (1, 1));
    }

    // -- run_remote_turn: pure frame sequencing, no model, no socket --

    #[test]
    fn a_successful_generation_streams_every_token_then_completes_and_scores() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let transcript = vec![sample_message(1, USER_SPEAKER_ID, "hi")];
        let scored = Mutex::new(None);

        run_remote_turn(
            7,
            transcript.clone(),
            tx,
            |_transcript| {},
            |_transcript, on_token| {
                on_token("hel");
                on_token("lo");
                Ok("hello".to_string())
            },
            |transcript, reply| {
                *scored.lock().unwrap() = Some((transcript.to_vec(), reply.to_string()));
            },
        );

        assert_eq!(
            drain(&mut rx),
            vec![
                ClientFrame::Token {
                    round_id: 7,
                    text: "hel".to_string()
                },
                ClientFrame::Token {
                    round_id: 7,
                    text: "lo".to_string()
                },
                ClientFrame::ReplyComplete {
                    round_id: 7,
                    text: "hello".to_string()
                },
            ]
        );
        assert_eq!(
            scored.into_inner().unwrap(),
            Some((transcript, "hello".to_string()))
        );
    }

    #[test]
    fn a_generation_error_yields_exactly_one_reply_failed_and_never_scores() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let scored = Mutex::new(false);

        run_remote_turn(
            3,
            vec![sample_message(1, USER_SPEAKER_ID, "hi")],
            tx,
            |_transcript| {},
            |_transcript, _on_token| Err(io::Error::other("model failed")),
            |_transcript, _reply| *scored.lock().unwrap() = true,
        );

        assert_eq!(
            drain(&mut rx),
            vec![ClientFrame::ReplyFailed {
                round_id: 3,
                reason: "model failed".to_string(),
            }]
        );
        assert!(!*scored.lock().unwrap(), "an error must never score");
    }

    #[test]
    fn think_runs_before_generate_and_before_the_first_token_is_sent() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let order: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

        run_remote_turn(
            1,
            vec![sample_message(1, USER_SPEAKER_ID, "hi")],
            tx,
            |_transcript| order.lock().unwrap().push("think"),
            |_transcript, on_token| {
                order.lock().unwrap().push("generate");
                on_token("hi");
                Ok("hi".to_string())
            },
            |_transcript, _reply| order.lock().unwrap().push("score"),
        );

        assert_eq!(
            *order.lock().unwrap(),
            vec!["think", "generate", "score"],
            "the thought must be written before the reply is generated"
        );
        assert_eq!(
            drain(&mut rx),
            vec![
                ClientFrame::Token {
                    round_id: 1,
                    text: "hi".to_string()
                },
                ClientFrame::ReplyComplete {
                    round_id: 1,
                    text: "hi".to_string()
                },
            ],
            "thinking must never itself emit a frame"
        );
    }

    // -- score_attitude: newest-user-row lookup and the no-user-row skip --

    #[test]
    fn score_attitude_scores_against_the_newest_user_row() {
        let store = RecordingStore::new(None);
        let transcript = vec![
            sample_message(1, USER_SPEAKER_ID, "first"),
            sample_message(2, "bot1", "a bot reply"),
            sample_message(3, USER_SPEAKER_ID, "second"),
        ];

        score_attitude(&store, 1, &transcript, "my reply");

        assert_eq!(
            *store.finished.lock().unwrap(),
            vec![("second".to_string(), "my reply".to_string())]
        );
    }

    #[test]
    fn score_attitude_skips_silently_when_the_transcript_has_no_user_row() {
        let store = RecordingStore::new(None);
        let transcript = vec![sample_message(1, "bot1", "a bot reply")];

        score_attitude(&store, 1, &transcript, "my reply");

        assert!(store.finished.lock().unwrap().is_empty());
    }

    // -- registry_from_participants --

    fn summary(id: &str, display_name: &str, kind: ParticipantKind) -> ParticipantSummary {
        ParticipantSummary {
            id: ParticipantId::parse(id).unwrap(),
            display_name: display_name.to_string(),
            kind,
            avatar_url: None,
            connected: true,
        }
    }

    #[test]
    fn registry_from_participants_carries_every_bot_and_falls_back_for_missing_user_char() {
        let participants = vec![
            ParticipantSummary {
                id: ParticipantId::USER,
                ..summary("user", "Alice", ParticipantKind::Human)
            },
            ParticipantSummary {
                id: ParticipantId::CHAR,
                ..summary("char", "Bob", ParticipantKind::HostBot)
            },
            summary("bot1", "Ada", ParticipantKind::RemoteBot),
        ];

        let registry = registry_from_participants(&participants);

        assert_eq!(registry.display_name(&ParticipantId::USER), Some("Alice"));
        assert_eq!(registry.display_name(&ParticipantId::CHAR), Some("Bob"));
        assert_eq!(
            registry.display_name(&ParticipantId::parse("bot1").unwrap()),
            Some("Ada")
        );
    }

    #[test]
    fn registry_from_participants_falls_back_to_the_raw_id_when_user_or_char_is_absent() {
        let registry = registry_from_participants(&[]);

        assert_eq!(registry.display_name(&ParticipantId::USER), Some("user"));
        assert_eq!(registry.display_name(&ParticipantId::CHAR), Some("char"));
    }

    // -- LocalModelGeneration: turn-slot claim, spawn, and release --

    fn joiner_handle_with(participants: Vec<ParticipantSummary>) -> JoinerHandle {
        let identity = crate::multiplayer::joiner::JoinerIdentity {
            id: ParticipantId::parse("bot1").unwrap(),
            display_name: "Ada".to_string(),
            avatar: None::<AvatarUpload>,
            password: "hunter2".to_string(),
            host_address: "127.0.0.1:0".to_string(),
        };
        let mut shared = JoinerShared::new(&identity, 1, None);
        shared.participants = participants;
        Arc::new(RwLock::new(shared))
    }

    /// A [`Thinker`] that writes nothing and records nothing, for a test
    /// with no interest in thought generation.
    fn noop_thinker() -> Thinker {
        Arc::new(|_transcript, _speakers| {})
    }

    // Every case below shares the process-wide `ACTIVE_TURN`, so they run as
    // one test function: two separate `#[test]`s touching the same global
    // would race under cargo's default parallel test execution. Also holds
    // `turn_slot::ACTIVE_TURN_TEST_LOCK` for the whole function, so this
    // test can never race `multiplayer::two_instance_tests`'s real
    // host-and-joiner tests, or `main.rs`'s `thoughts_route_tests` (#217),
    // for the same global slot either.
    #[test]
    fn local_model_generation_claims_and_releases_the_shared_turn_slot() {
        let _serial = crate::turn_slot::ACTIVE_TURN_TEST_LOCK.blocking_lock();

        // A pre-claimed slot: `try_handle` must report failure and spawn no
        // thread at all, rather than generate — or think — while a local
        // turn is live.
        let outer_guard = ACTIVE_TURN.try_claim().expect("slot should start free");
        let generation = LocalModelGeneration::new(
            1,
            ParticipantId::parse("bot1").unwrap(),
            joiner_handle_with(vec![]),
            Arc::new(|_transcript, _speakers, _on_token| {
                panic!("must never generate while the slot is claimed")
            }),
            Arc::new(|_transcript, _speakers| panic!("must never think while the slot is claimed")),
            crate::multiplayer::joiner_compaction::noop_job(),
        );
        let (tx, mut rx) = mpsc::unbounded_channel();

        let join_handle = generation.try_handle(1, vec![], tx);

        assert!(join_handle.is_none(), "no thread should have been spawned");
        assert_eq!(
            drain(&mut rx),
            vec![ClientFrame::ReplyFailed {
                round_id: 1,
                reason: "a local turn is in progress".to_string(),
            }]
        );
        drop(outer_guard);

        // The slot is free again: `try_handle` claims it, spawns the
        // generation thread, invokes the injected thinker before the reply
        // is generated, and releases the slot once that thread joins.
        let thinker_calls: Arc<Mutex<Vec<Vec<i32>>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded_thoughts = Arc::clone(&thinker_calls);
        let generation = LocalModelGeneration::new(
            1,
            ParticipantId::parse("bot1").unwrap(),
            joiner_handle_with(vec![]),
            Arc::new(|_transcript, _speakers, on_token| {
                on_token("hi");
                Ok("hi".to_string())
            }),
            Arc::new(move |transcript, _speakers| {
                recorded_thoughts
                    .lock()
                    .unwrap()
                    .push(transcript.iter().map(|m| m.id).collect());
            }),
            crate::multiplayer::joiner_compaction::noop_job(),
        );
        let (tx, mut rx) = mpsc::unbounded_channel();

        let join_handle = generation
            .try_handle(5, vec![sample_message(1, USER_SPEAKER_ID, "hello")], tx)
            .expect("the slot was free, so a thread should have been spawned");
        join_handle
            .join()
            .expect("generation thread should not panic");

        assert_eq!(
            *thinker_calls.lock().unwrap(),
            vec![vec![1]],
            "try_handle should invoke the injected thinker with the request's transcript"
        );
        assert_eq!(
            drain(&mut rx),
            vec![
                ClientFrame::Token {
                    round_id: 5,
                    text: "hi".to_string()
                },
                ClientFrame::ReplyComplete {
                    round_id: 5,
                    text: "hi".to_string()
                },
            ],
            "thinking must never itself send a frame"
        );
        assert!(
            ACTIVE_TURN.try_claim().is_some(),
            "the slot should be free again once the thread has joined"
        );

        // PR #204 review finding: extraction used to claim `ACTIVE_TURN` on
        // the same `GenerateRequest` the reply itself needed it for, so
        // every continuity-advancing frame lost that race and got
        // `ReplyFailed` instead of a reply. `try_handle` must always
        // produce the reply first, only attempting extraction afterward,
        // independently, once the reply's own claim has actually been
        // released.
        let handle = joiner_handle_with(vec![]);
        {
            let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
            shared.last_continuity = Some(crate::multiplayer::protocol::ContinuityPayload {
                compacted_through: 5,
                ..Default::default()
            });
        }

        let extraction_calls: Arc<Mutex<Vec<(i32, i32)>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&extraction_calls);
        let extraction: crate::multiplayer::joiner_compaction::JoinerExtractionJob =
            Arc::new(move |request| {
                recorded
                    .lock()
                    .unwrap()
                    .push((request.from, request.through));
            });

        let generation = LocalModelGeneration::new(
            1,
            ParticipantId::parse("bot1").unwrap(),
            handle.clone(),
            Arc::new(|_transcript, _speakers, on_token| {
                on_token("hi");
                Ok("hi".to_string())
            }),
            noop_thinker(),
            extraction,
        );
        let (tx, mut rx) = mpsc::unbounded_channel();

        let join_handle = generation
            .try_handle(9, vec![sample_message(1, USER_SPEAKER_ID, "hello")], tx)
            .expect("the slot was free, so a thread should have been spawned");
        join_handle
            .join()
            .expect("generation thread should not panic");

        assert_eq!(
            drain(&mut rx),
            vec![
                ClientFrame::Token {
                    round_id: 9,
                    text: "hi".to_string()
                },
                ClientFrame::ReplyComplete {
                    round_id: 9,
                    text: "hi".to_string()
                },
            ],
            "a continuity-advancing GenerateRequest must still produce its reply, not ReplyFailed"
        );

        // Extraction runs on a further spawned thread of its own
        // (`joiner_compaction::spawn_holding`), holding `JOINER_EXTRACTION`
        // rather than `ACTIVE_TURN` (PR #204 review finding: a dedicated
        // slot, so extraction can never contend with a reply even while
        // still running); wait for it to release that slot before asserting
        // it ran, the same polling pattern
        // `local_model_generation_claims_and_releases_the_shared_turn_slot`
        // and `joiner_compaction`'s own tests use for the same reason.
        for _ in 0..100 {
            if crate::turn_slot::JOINER_EXTRACTION.try_claim().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(
            *extraction_calls.lock().unwrap(),
            vec![(1, 5)],
            "extraction should still run, independently, after the reply"
        );

        // PR #204 review finding (reopened): `ACTIVE_TURN` alone was not
        // enough — even after the same-frame race above was fixed,
        // extraction held `ACTIVE_TURN` for its whole model-bound
        // extract+merge, so any `GenerateRequest` landing *during* that
        // window (not just the one that queued it) still came back
        // `ReplyFailed`. With extraction on its own `JOINER_EXTRACTION`
        // slot, a reply must succeed even while an extraction is still
        // running.
        let held_extraction = crate::turn_slot::JOINER_EXTRACTION
            .try_claim()
            .expect("nothing else holds it at this point in the test");
        let generation_during_extraction = LocalModelGeneration::new(
            1,
            ParticipantId::parse("bot1").unwrap(),
            handle.clone(),
            Arc::new(|_transcript, _speakers, on_token| {
                on_token("hi again");
                Ok("hi again".to_string())
            }),
            noop_thinker(),
            crate::multiplayer::joiner_compaction::noop_job(),
        );
        let (tx, mut rx) = mpsc::unbounded_channel();
        let join_handle = generation_during_extraction
            .try_handle(10, vec![sample_message(2, USER_SPEAKER_ID, "hi")], tx)
            .expect("ACTIVE_TURN was free, so a thread should have been spawned");
        join_handle
            .join()
            .expect("generation thread should not panic");
        assert_eq!(
            drain(&mut rx),
            vec![
                ClientFrame::Token {
                    round_id: 10,
                    text: "hi again".to_string()
                },
                ClientFrame::ReplyComplete {
                    round_id: 10,
                    text: "hi again".to_string()
                },
            ],
            "a reply must succeed while a joiner's own extraction is still \
             running, not just outside the one frame that queued it"
        );
        drop(held_extraction);
    }
}
