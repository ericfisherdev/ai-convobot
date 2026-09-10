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
//! thread [`run_remote_turn`] actually runs on.

use std::io;
use std::sync::Arc;

use tokio::sync::mpsc::UnboundedSender;

use crate::chat_turn::{SqliteTurnStore, TurnStore};
use crate::compaction::context::CompactionContext;
use crate::database::{Message, USER_SPEAKER_ID};
use crate::llm::{self, FixedCompaction, InMemoryTranscript, PromptSpeakers};
use crate::multiplayer::joiner::{GenerateRequestHandler, JoinerHandle};
use crate::multiplayer::protocol::{ClientFrame, ParticipantSummary};
use crate::participants::{AvatarRef, Participant, ParticipantId, ParticipantRegistry};
use crate::turn_slot::ACTIVE_TURN;

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

/// The [`GenerateRequestHandler`] `main.rs` wires a joiner up with.
pub struct LocalModelGeneration {
    companion_id: i32,
    self_id: ParticipantId,
    handle: JoinerHandle,
    generator: RemoteGenerator,
}

impl LocalModelGeneration {
    /// Takes any generator. Used directly by this module's own tests (a
    /// stub that emits fixed tokens with no model loaded) and by #136's
    /// two-instance test.
    pub fn new(
        companion_id: i32,
        self_id: ParticipantId,
        handle: JoinerHandle,
        generator: RemoteGenerator,
    ) -> Self {
        LocalModelGeneration {
            companion_id,
            self_id,
            handle,
            generator,
        }
    }

    /// Wraps the production generator: the joiner's own model over
    /// `llm::prompt_streaming`, generating from an `InMemoryTranscript` of
    /// the transcript the host sent (never the joiner's own local
    /// `messages` table, which a remote reply never touches) and keyed by
    /// the newest user message in it, the same way a local turn's
    /// long-term memory recall is keyed by what the user just said. Passes
    /// a default (never-compacted) `CompactionContext` until #182 replaces
    /// this with its own `HostContinuity` impl built from the host's
    /// `ContinuityPayload`.
    pub fn with_local_model(
        companion_id: i32,
        self_id: ParticipantId,
        handle: JoinerHandle,
    ) -> Self {
        let generator: RemoteGenerator = Arc::new(
            move |transcript: &[Message],
                  speakers: &PromptSpeakers,
                  on_token: &mut dyn FnMut(&str)| {
                let prompt = newest_user_message(transcript);
                llm::prompt_streaming(
                    &prompt,
                    companion_id,
                    on_token,
                    &InMemoryTranscript(transcript.to_vec()),
                    speakers,
                    &FixedCompaction(CompactionContext::default()),
                )
            },
        );
        LocalModelGeneration::new(companion_id, self_id, handle, generator)
    }

    /// The body of [`GenerateRequestHandler::handle`], returning the
    /// spawned thread's `JoinHandle` so this module's own tests can join it
    /// before asserting the turn slot is free again. `None` when nothing
    /// was spawned (the slot was already claimed, or the spawn itself
    /// failed) — both cases already sent their own `ReplyFailed`.
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
        let companion_id = self.companion_id;
        // Kept outside the closure below so a failed spawn (which drops the
        // closure, and with it the `tx` moved into it, without running it)
        // still has a sender left to report the failure with.
        let tx_for_failed_spawn = tx.clone();

        let spawn_result = std::thread::Builder::new()
            .name("remote-generation".into())
            .spawn(move || {
                let _turn_guard = turn_guard;
                run_remote_turn(
                    round_id,
                    transcript,
                    tx,
                    |transcript, on_token| generator(transcript, &speakers, on_token),
                    |transcript, reply| {
                        let store = SqliteTurnStore::new(Vec::new());
                        score_attitude(&store, companion_id, transcript, reply);
                    },
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

/// Runs one joiner reply end to end on the calling thread: generates it
/// (streaming a `Token` to `tx` for every `on_token` call), scores the
/// joiner's own attitude against it on success, then sends the terminal
/// frame. A closed `tx` (the socket dropped mid-generation) is ignored
/// exactly as the host's own `stream_round` ignores a hung-up SSE client:
/// every send here is best-effort.
pub(crate) fn run_remote_turn(
    round_id: u64,
    transcript: Vec<Message>,
    tx: UnboundedSender<ClientFrame>,
    generate: impl FnOnce(&[Message], &mut dyn FnMut(&str)) -> io::Result<String>,
    score: impl FnOnce(&[Message], &str),
) {
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
fn registry_from_participants(participants: &[ParticipantSummary]) -> ParticipantRegistry {
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
    use crate::multiplayer::joiner::JoinerShared;
    use crate::multiplayer::protocol::AvatarUpload;
    use crate::participants::ParticipantKind;
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
        let mut shared = JoinerShared::new(&identity);
        shared.participants = participants;
        Arc::new(RwLock::new(shared))
    }

    // Both cases below share the process-wide `ACTIVE_TURN`, so they run as
    // one test function: two separate `#[test]`s touching the same global
    // would race under cargo's default parallel test execution.
    #[test]
    fn local_model_generation_claims_and_releases_the_shared_turn_slot() {
        // A pre-claimed slot: `try_handle` must report failure and spawn no
        // thread at all, rather than generate while a local turn is live.
        let outer_guard = ACTIVE_TURN.try_claim().expect("slot should start free");
        let generation = LocalModelGeneration::new(
            1,
            ParticipantId::parse("bot1").unwrap(),
            joiner_handle_with(vec![]),
            Arc::new(|_transcript, _speakers, _on_token| {
                panic!("must never generate while the slot is claimed")
            }),
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
        // generation thread, and releases it once that thread joins.
        let generation = LocalModelGeneration::new(
            1,
            ParticipantId::parse("bot1").unwrap(),
            joiner_handle_with(vec![]),
            Arc::new(|_transcript, _speakers, on_token| {
                on_token("hi");
                Ok("hi".to_string())
            }),
        );
        let (tx, mut rx) = mpsc::unbounded_channel();

        let join_handle = generation
            .try_handle(5, vec![sample_message(1, USER_SPEAKER_ID, "hello")], tx)
            .expect("the slot was free, so a thread should have been spawned");
        join_handle
            .join()
            .expect("generation thread should not panic");

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
            ]
        );
        assert!(
            ACTIVE_TURN.try_claim().is_some(),
            "the slot should be free again once the thread has joined"
        );
    }
}
