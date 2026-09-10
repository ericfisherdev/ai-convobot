//! The round orchestrator (#131, Part A): turns one user message into a
//! sequence of speaker replies — the host companion, then each connected
//! joiner in join order — all under one held turn slot.
//!
//! Everything here is `Database`-free: [`plan_round`] and [`run_round`] are
//! driven entirely through the [`crate::chat_turn::TurnStore`],
//! [`RemoteGenerator`] and [`RoundSink`] seams, so a round can be exercised
//! with fakes in tests without a real sqlite file or model. [`NoRemotes`]
//! reports every remote speaker offline, so a round with no joiners behaves
//! exactly as `PendingTurn::complete` did before this module existed; the
//! real socket-backed `RemoteGenerator` is #154's `SocketRemoteGenerator`
//! (`multiplayer::remote_generator`), wired over `RemoteBots`. This module's
//! own `broadcast` parameter is the other half of #154: every persisted
//! message of the round (the user's turn, each reply, each skip notice) is
//! handed to it in persistence order, so `main.rs` can mirror the round to
//! every joiner via `RemoteBots::broadcast`.
//!
//! The speaker order itself — who talks, and who an `@mention` in a reply
//! adds mid-round — is `routing.rs`'s (#132): [`RoundPlan`], [`plan_round`]
//! and `schedule_follow_ups` are re-exported from there so this module's own
//! call sites and tests need no separate import.

use std::collections::HashSet;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::chat_turn::{PendingTurn, PersistedReply, TurnStore};
use crate::compaction::hook::{after_round as compaction_after_round, QueuedDraft};
use crate::database::{CompanionAttitude, Message};
use crate::multiplayer::protocol::{ContinuityPayload, ServerFrame};
pub use crate::multiplayer::routing::{plan_round, RoundPlan};
use crate::multiplayer::routing::{schedule_follow_ups, RoutingPolicy};
use crate::participants::{ParticipantId, ParticipantRegistry};
use crate::turn_slot::TurnGuard;

/// The newest-messages page size a remote speaker's request is built from:
/// the same page length `GET /api/message` uses for the frontend's first
/// page.
const TRANSCRIPT_TAIL_MESSAGES: usize = 50;

/// Drops every message at or before `compacted_through` from `tail` (#182):
/// once the host has committed a checkpoint through that id, a joiner
/// should see the [`ContinuityPayload`] covering it instead of the raw
/// messages a second time. `None` (never compacted) passes `tail` through
/// unchanged; a tail that falls entirely inside the compacted range yields
/// an empty transcript, which is still a valid `RemoteRequest`.
fn transcript_after(tail: Vec<Message>, compacted_through: Option<i32>) -> Vec<Message> {
    match compacted_through {
        None => tail,
        Some(through) => tail.into_iter().filter(|m| m.id > through).collect(),
    }
}

/// The seam between the round orchestrator and a remote joiner's socket
/// connection. The production implementation, over the host WebSocket via
/// `RemoteBots`, is #154 (Part B); tests use a scripted fake.
pub trait RemoteGenerator {
    /// Requests a reply from `request.speaker`, invoking `on_token` with
    /// each token as it arrives (mirroring `llm::prompt_streaming`'s
    /// callback: runs on the calling thread, must not block).
    fn generate(
        &self,
        request: RemoteRequest<'_>,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String, RemoteFailure>;
}

/// One remote speaker's turn: what it is generating for, and what it can see.
///
/// `NoRemotes` ignores every field; #154's `SocketRemoteGenerator`
/// (`multiplayer::remote_generator`) is what actually reads them. This
/// module's own tests exercise them through `FakeRemote`.
pub struct RemoteRequest<'a> {
    pub round_id: u64,
    pub speaker: &'a ParticipantId,
    pub transcript: &'a [Message],
    /// A total deadline measured from the moment the request is sent, not an
    /// idle timeout: it bounds the whole round trip, including however long
    /// the remote spends thinking between tokens, not just the gap between
    /// two tokens. This is the one meaning `config.remote_generation_timeout_secs`
    /// (#128) has, wherever it is read.
    pub timeout: Duration,
    /// The host's compaction checkpoint (#182), read once per round (or
    /// once per `regenerate_reply` call) so every remote speaker sees the
    /// same one even if a background job commits mid-round. `None` in solo
    /// mode and in host mode before the host has ever compacted.
    pub continuity: Option<&'a ContinuityPayload>,
}

/// Why a remote speaker did not produce a reply.
///
/// `NoRemotes` always returns `Offline`; `Timeout` and `Failed` are
/// `SocketRemoteGenerator`'s (#154) to construct, once a real socket
/// round-trip times out or the joiner reports `ReplyFailed`. This module's
/// tests construct all three.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteFailure {
    Offline,
    Timeout,
    Failed(String),
}

/// A [`RemoteGenerator`] with no remotes: every request reports the speaker
/// offline. What both prompting handlers use in `Solo` mode, so a round
/// with no joiners is unaffected by `RemoteBots`' existence — `main.rs`
/// switches to `SocketRemoteGenerator` (#154) only in `Host` mode.
pub struct NoRemotes;

impl RemoteGenerator for NoRemotes {
    fn generate(
        &self,
        _request: RemoteRequest<'_>,
        _on_token: &mut dyn FnMut(&str),
    ) -> Result<String, RemoteFailure> {
        Err(RemoteFailure::Offline)
    }
}

/// Observes a round as it runs, one speaker at a time.
///
/// `main.rs` implements this over the SSE stream (`SseRoundSink`; #133
/// extends it with speaker-tagged chunks) and the non-streaming handler uses
/// [`NoopSink`].
pub trait RoundSink {
    fn reply_started(&mut self, speaker: &ParticipantId);
    fn token(&mut self, speaker: &ParticipantId, text: &str);
    fn reply_complete(&mut self, reply: &PersistedReply);
    /// `notice` is the persisted system message explaining the skip, so
    /// #133 can render it as a bubble.
    fn speaker_skipped(&mut self, speaker: &ParticipantId, notice: &PersistedReply);
    /// Scoring runs once at the end of the round (`PendingTurn::finish`),
    /// which in solo mode reproduces today's wire order exactly: the
    /// attitude chunk, then the final chunk.
    fn round_complete(&mut self, attitude: Option<&(CompanionAttitude, CompanionAttitude)>);
}

/// A [`RoundSink`] that observes nothing. Used by the non-streaming
/// `/api/prompt` handler, which has no per-token chunk to send.
pub struct NoopSink;

impl RoundSink for NoopSink {
    fn reply_started(&mut self, _speaker: &ParticipantId) {}
    fn token(&mut self, _speaker: &ParticipantId, _text: &str) {}
    fn reply_complete(&mut self, _reply: &PersistedReply) {}
    fn speaker_skipped(&mut self, _speaker: &ParticipantId, _notice: &PersistedReply) {}
    fn round_complete(&mut self, _attitude: Option<&(CompanionAttitude, CompanionAttitude)>) {}
}

/// What a round produced: every persisted reply, the host's own reply
/// singled out for callers that only care about that (today, both prompting
/// handlers), and the attitude change scored against it.
pub struct RoundOutcome {
    /// `None` when the host companion did not speak this round (a
    /// mention-filtered round, #132) — unreachable today, since
    /// [`plan_round`] always puts `char` first.
    pub host_reply: Option<PersistedReply>,
    /// Every speaker's persisted reply, in speaking order. Does not include
    /// skipped-speaker notices. Neither prompting handler reads this today
    /// (both only care about `host_reply`); #133's speaker-tagged SSE
    /// chunks are what reads it next.
    #[allow(dead_code)]
    pub replies: Vec<PersistedReply>,
    /// Read by neither handler today for the same reason `replies` is not:
    /// both already got the same value out of `sink.round_complete` while
    /// the round ran. Kept on the outcome for a caller that only wants the
    /// end result, not the running commentary.
    #[allow(dead_code)]
    pub attitude: Option<(CompanionAttitude, CompanionAttitude)>,
    /// The compaction draft (#172) this round's `after_round` hook queued,
    /// if any. Read by neither handler today; the SSE draft-ready
    /// notification (a later compaction issue) is what reads this next.
    #[allow(dead_code)]
    pub queued_draft: Option<QueuedDraft>,
}

/// Assigns each round a distinct id, for `RemoteRequest::round_id` — #154's
/// production `RemoteGenerator` uses it to route inbound frames on
/// `RemoteBots::subscribe_round` back to the round that is waiting on them.
static NEXT_ROUND_ID: AtomicU64 = AtomicU64::new(1);

fn next_round_id() -> u64 {
    NEXT_ROUND_ID.fetch_add(1, Ordering::Relaxed)
}

/// Looks `id` back up through `store` and hands it to `broadcast` as a
/// `ServerFrame::Message`, so every joiner's transcript mirror gets the
/// exact row the host just persisted — same id, same `created_at`. Called
/// for the user's turn and for every reply and skip notice `run_round`
/// persists, in the same order they were inserted.
///
/// A lookup failure only means a joiner's mirror misses this one message
/// (it still has everything the `Joined` handshake seeded it with, plus
/// whatever else the round broadcasts); it must never fail the round
/// itself, so this logs and returns rather than propagating.
fn broadcast_message(store: &impl TurnStore, id: i32, broadcast: &dyn Fn(ServerFrame)) {
    match store.get_message(id) {
        Ok(row) => broadcast(ServerFrame::Message(row)),
        Err(e) => eprintln!(
            "multiplayer: failed to load message {} to broadcast to joiners: {}",
            id, e
        ),
    }
}

/// Runs one round on the calling (blocking) thread, exactly like
/// `PendingTurn::complete` did before this module existed.
///
/// Binds `turn_guard` first so the slot is held until this function
/// returns, which is after `sink.round_complete` — an overlapping call
/// cannot start a second round until this one, including its attitude
/// scoring, has fully settled.
///
/// `too_many_arguments`/`type_complexity`: every parameter here is one of
/// this module's own seam types (`TurnStore`, `RemoteGenerator`,
/// `RoundSink`) or plain data; factoring them into a settings struct would
/// just move the same fields one level down without reducing what a caller
/// has to supply.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn run_round(
    turn_guard: TurnGuard,
    pending: PendingTurn,
    mut plan: RoundPlan,
    store: &impl TurnStore,
    registry: &ParticipantRegistry,
    policy: &RoutingPolicy,
    host: &mut dyn FnMut(&str, &mut dyn FnMut(&str)) -> io::Result<String>,
    remotes: &dyn RemoteGenerator,
    broadcast: &dyn Fn(ServerFrame),
    timeout: Duration,
    sink: &mut dyn RoundSink,
) -> io::Result<RoundOutcome> {
    let _turn_guard = turn_guard;
    let round_id = next_round_id();
    // Read once, before the speaker loop, so every remote speaker in this
    // round sees the same checkpoint even if a background extraction job
    // commits a new one mid-round (#182).
    let continuity = store
        .continuity()
        .map_err(|e| io::Error::other(e.to_string()))?;

    broadcast_message(store, pending.user_message_id(), broadcast);

    let mut host_reply: Option<PersistedReply> = None;
    let mut replies: Vec<PersistedReply> = Vec::new();

    while let Some(next) = plan.next_speaker() {
        let speaker = &next.id;
        sink.reply_started(speaker);

        if *speaker == ParticipantId::CHAR {
            // A failed host generate ends the round immediately: no notice,
            // no further speakers, no `finish` — the #84 regression guard,
            // now covering the whole round instead of a single turn.
            let persisted = pending.reply(store, speaker.clone(), |generation_prompt| {
                host(generation_prompt, &mut |token| sink.token(speaker, token))
            })?;
            broadcast_message(store, persisted.message_id, broadcast);
            sink.reply_complete(&persisted);
            for followup in schedule_follow_ups(&mut plan, &persisted.text, &next, registry, policy)
            {
                println!(
                    "Follow-up scheduled: {} (depth {})",
                    followup,
                    next.depth + 1
                );
            }
            host_reply = Some(persisted.clone());
            replies.push(persisted);
            continue;
        }

        let tail = store
            .transcript_tail(TRANSCRIPT_TAIL_MESSAGES)
            .map_err(|e| io::Error::other(e.to_string()))?;
        let tail = transcript_after(tail, continuity.as_ref().map(|c| c.compacted_through));
        let attempt = pending.reply(store, speaker.clone(), |_generation_prompt| {
            remotes
                .generate(
                    RemoteRequest {
                        round_id,
                        speaker,
                        transcript: &tail,
                        timeout,
                        continuity: continuity.as_ref(),
                    },
                    &mut |token| sink.token(speaker, token),
                )
                .map_err(|failure| io::Error::other(format!("{:?}", failure)))
        });

        match attempt {
            Ok(persisted) => {
                broadcast_message(store, persisted.message_id, broadcast);
                sink.reply_complete(&persisted);
                for followup in
                    schedule_follow_ups(&mut plan, &persisted.text, &next, registry, policy)
                {
                    println!(
                        "Follow-up scheduled: {} (depth {})",
                        followup,
                        next.depth + 1
                    );
                }
                replies.push(persisted);
            }
            Err(_) => {
                let notice_text = format!("{} did not respond", speaker);
                let message_id = store
                    .insert_reply(&ParticipantId::SYSTEM, &notice_text)
                    .map_err(|e| io::Error::other(e.to_string()))?;
                let notice = PersistedReply {
                    message_id,
                    speaker_id: ParticipantId::SYSTEM,
                    text: notice_text,
                };
                broadcast_message(store, notice.message_id, broadcast);
                sink.speaker_skipped(speaker, &notice);
            }
        }
    }

    // Read before `finish` consumes `pending`.
    let companion_id = pending.companion_id();
    let attitude = pending.finish(store, host_reply.as_ref().map(|r| r.text.as_str()));
    // Runs whether or not the host spoke this round (a mention-filtered
    // round still finishes a turn worth checking for compaction); a
    // compaction failure must never fail the round, so `after_round` logs
    // and returns `None` internally rather than propagating.
    let queued_draft = compaction_after_round(store, companion_id);
    sink.round_complete(attitude.as_ref());

    Ok(RoundOutcome {
        host_reply,
        replies,
        attitude,
        queued_draft,
    })
}

/// Where a regenerate request for a popped reply's `speaker_id` should be
/// routed. `char` always regenerates locally through the host model; every
/// other id belongs to a remote bot, routed to it only while it is
/// connected — otherwise regenerating would either hang forever or answer
/// with the wrong bot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerateTarget {
    Local,
    Remote(String),
    Offline(String),
}

/// Pure routing decision for `speaker_id`, given the bot ids currently
/// connected. `main.rs::regenerate_prompt` snapshots `RemoteBots`'
/// connections into `connected_bots` before claiming the turn slot, so the
/// whole regenerate stays `Send` across the `web::block` boundary; in
/// `Solo` mode that snapshot is always empty, so a stray remote id there
/// routes `Offline` and `char` still routes `Local`, matching solo
/// behaviour exactly. This is also what `Database::pop_latest_bot_reply`'s
/// `owner_ready` predicate is built from, so the "is this reply's owner
/// available" check and the actual routing can never disagree.
pub fn regenerate_target(speaker_id: &str, connected_bots: &HashSet<String>) -> RegenerateTarget {
    if speaker_id == ParticipantId::CHAR.as_str() {
        return RegenerateTarget::Local;
    }
    if connected_bots.contains(speaker_id) {
        RegenerateTarget::Remote(speaker_id.to_string())
    } else {
        RegenerateTarget::Offline(speaker_id.to_string())
    }
}

/// Why [`regenerate_reply`] could not produce a replacement reply. Mapped to
/// an HTTP response by `main.rs::regenerate_prompt`'s `TurnError`, mirroring
/// how [`run_round`]'s `io::Error` outcome is mapped by its own caller.
#[derive(Debug)]
pub enum RegenerateError {
    /// The reply's owner is a remote bot that is not currently connected.
    Offline(String),
    /// The host, or the remote bot's own generation, failed. Carries the bot
    /// id in the message so the log line names it, mirroring
    /// [`SocketRemoteGenerator`](crate::multiplayer::remote_generator::SocketRemoteGenerator)'s
    /// own failure logging.
    Generate(io::Error),
    Database(rusqlite::Error),
}

/// Regenerates the bot reply `Database::pop_latest_bot_reply` just removed,
/// routed to its owner exactly as a live round would route it: `char`
/// generates locally through `host`, any other id answers over `remotes`
/// with the same transcript-tail request a round sends it.
///
/// Deliberately does not go through [`run_round`]: regenerate answers one
/// speaker, not a whole round, and — unchanged from before #135 — never
/// inserts a user turn, never calls `PendingTurn::finish`, and so never
/// re-scores attitude. On the `Remote` arm, `host` is never called, so no
/// host-side long-term-memory entry is written for a remote reply (`llm::generate`,
/// `backend/src/llm.rs`, is the only place that happens); the joiner's own
/// `generate` is what writes the entry on the joiner's side (#130).
pub fn regenerate_reply(
    target: RegenerateTarget,
    user_turn: &Message,
    store: &impl TurnStore,
    host: impl FnOnce(&str) -> io::Result<String>,
    remotes: &dyn RemoteGenerator,
    timeout: Duration,
) -> Result<PersistedReply, RegenerateError> {
    match target {
        RegenerateTarget::Offline(speaker_id) => Err(RegenerateError::Offline(speaker_id)),
        RegenerateTarget::Local => {
            let text = host(&user_turn.content).map_err(RegenerateError::Generate)?;
            let message_id = store
                .insert_reply(&ParticipantId::CHAR, &text)
                .map_err(RegenerateError::Database)?;
            Ok(PersistedReply {
                message_id,
                speaker_id: ParticipantId::CHAR,
                text,
            })
        }
        RegenerateTarget::Remote(speaker_id) => {
            // Only ever reachable with an id `regenerate_target` already
            // resolved to `Remote`, i.e. one this same process's registry
            // produced as a connected bot's `ParticipantId`, so this parse
            // cannot fail in practice; mapped to `Generate` rather than
            // unwrapped so a future caller cannot turn a bad id into a panic.
            let speaker = ParticipantId::parse(&speaker_id).map_err(|e| {
                RegenerateError::Generate(io::Error::other(format!(
                    "invalid speaker id {:?}: {}",
                    speaker_id, e
                )))
            })?;
            let continuity = store.continuity().map_err(RegenerateError::Database)?;
            let tail = store
                .transcript_tail(TRANSCRIPT_TAIL_MESSAGES)
                .map_err(RegenerateError::Database)?;
            let tail = transcript_after(tail, continuity.as_ref().map(|c| c.compacted_through));
            let text = remotes
                .generate(
                    RemoteRequest {
                        round_id: next_round_id(),
                        speaker: &speaker,
                        transcript: &tail,
                        timeout,
                        continuity: continuity.as_ref(),
                    },
                    &mut |_token| {},
                )
                .map_err(|failure| {
                    RegenerateError::Generate(io::Error::other(format!(
                        "{} failed to reply: {:?}",
                        speaker_id, failure
                    )))
                })?;
            let message_id = store
                .insert_reply(&speaker, &text)
                .map_err(RegenerateError::Database)?;
            Ok(PersistedReply {
                message_id,
                speaker_id: speaker,
                text,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_turn::RecordingStore;
    use crate::compaction::hook::CompactionTailView;
    use crate::compaction::trigger::CompactionConfig;
    use crate::compaction::types::CompactionTrigger;
    use crate::compaction::MessageRef;
    use crate::participants::{Participant, ParticipantKind};
    use crate::turn_slot::TurnSlot;
    use std::collections::HashMap;
    use std::sync::Mutex;

    fn bot(id: &str) -> ParticipantId {
        ParticipantId::parse(id).unwrap()
    }

    fn insert_bot(registry: &mut ParticipantRegistry, id: &str, display_name: &str) {
        registry
            .insert(Participant {
                id: bot(id),
                display_name: display_name.to_string(),
                kind: ParticipantKind::RemoteBot,
                avatar: None,
            })
            .unwrap();
    }

    /// Scripted per-speaker remote outcomes, plus a record of the transcript
    /// each request carried — how the round tests confirm a later speaker
    /// saw an earlier one's reply.
    struct FakeRemote {
        outcomes: HashMap<ParticipantId, Result<String, RemoteFailure>>,
        seen_transcripts: Mutex<HashMap<ParticipantId, Vec<Message>>>,
        /// Every speaker's `request.continuity`, cloned out of the borrow —
        /// how the tests below confirm a round ships the same
        /// `ContinuityPayload` `store.continuity()` returned (#182).
        seen_continuity: Mutex<HashMap<ParticipantId, Option<ContinuityPayload>>>,
    }

    impl FakeRemote {
        fn new(outcomes: Vec<(ParticipantId, Result<&str, RemoteFailure>)>) -> Self {
            Self {
                outcomes: outcomes
                    .into_iter()
                    .map(|(id, result)| (id, result.map(|text| text.to_string())))
                    .collect(),
                seen_transcripts: Mutex::new(HashMap::new()),
                seen_continuity: Mutex::new(HashMap::new()),
            }
        }

        fn transcript_seen_by(&self, id: &ParticipantId) -> Vec<Message> {
            self.seen_transcripts
                .lock()
                .unwrap()
                .get(id)
                .cloned()
                .unwrap_or_default()
        }

        fn continuity_seen_by(&self, id: &ParticipantId) -> Option<ContinuityPayload> {
            self.seen_continuity
                .lock()
                .unwrap()
                .get(id)
                .cloned()
                .flatten()
        }
    }

    impl RemoteGenerator for FakeRemote {
        fn generate(
            &self,
            request: RemoteRequest<'_>,
            _on_token: &mut dyn FnMut(&str),
        ) -> Result<String, RemoteFailure> {
            self.seen_transcripts
                .lock()
                .unwrap()
                .insert(request.speaker.clone(), request.transcript.to_vec());
            self.seen_continuity
                .lock()
                .unwrap()
                .insert(request.speaker.clone(), request.continuity.cloned());
            match self.outcomes.get(request.speaker) {
                Some(outcome) => outcome.clone(),
                None => Err(RemoteFailure::Offline),
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum SinkEvent {
        Started(ParticipantId),
        Complete(ParticipantId, String),
        Skipped(ParticipantId),
        RoundComplete,
    }

    #[derive(Default)]
    struct RecordingSink {
        events: Vec<SinkEvent>,
    }

    impl RoundSink for RecordingSink {
        fn reply_started(&mut self, speaker: &ParticipantId) {
            self.events.push(SinkEvent::Started(speaker.clone()));
        }
        fn token(&mut self, _speaker: &ParticipantId, _text: &str) {}
        fn reply_complete(&mut self, reply: &PersistedReply) {
            self.events.push(SinkEvent::Complete(
                reply.speaker_id.clone(),
                reply.text.clone(),
            ));
        }
        fn speaker_skipped(&mut self, speaker: &ParticipantId, _notice: &PersistedReply) {
            self.events.push(SinkEvent::Skipped(speaker.clone()));
        }
        fn round_complete(&mut self, _attitude: Option<&(CompanionAttitude, CompanionAttitude)>) {
            self.events.push(SinkEvent::RoundComplete);
        }
    }

    /// `user`/`char` plus `bot1`/`bot2`, for tests that need a registry
    /// `run_round`'s `schedule_follow_ups` call can look bots up in.
    fn registry_with_bots() -> ParticipantRegistry {
        let mut registry = ParticipantRegistry::solo("Alice", "Bob", None);
        insert_bot(&mut registry, "bot1", "Ada");
        insert_bot(&mut registry, "bot2", "Grace");
        registry
    }

    /// A policy that never schedules a follow-up. What every `run_round`
    /// test below uses: none of them are testing `routing.rs`'s own
    /// follow-up behaviour (that is `routing.rs`'s tests), so a plan that
    /// never grows keeps these tests' assertions exactly as fixed lists.
    fn no_followups() -> RoutingPolicy {
        RoutingPolicy {
            max_followup_depth: 0,
        }
    }

    #[test]
    fn plan_round_orders_char_then_connected_joiners_by_join_order() {
        let registry = registry_with_bots();

        let plan = plan_round("hi", &registry, &no_followups());

        assert_eq!(
            plan.pending_ids(),
            vec![ParticipantId::CHAR, bot("bot1"), bot("bot2")]
        );
    }

    #[test]
    fn plan_round_excludes_a_disconnected_joiner() {
        // A joiner that never joined (or has since disconnected) is simply
        // absent from the registry: `host.rs` removes it on disconnect.
        let registry = ParticipantRegistry::solo("Alice", "Bob", None);

        let plan = plan_round("hi", &registry, &no_followups());

        assert_eq!(plan.pending_ids(), vec![ParticipantId::CHAR]);
    }

    #[test]
    fn a_round_persists_char_then_each_bot_in_order_and_scores_once() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);
        let registry = registry_with_bots();
        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), registry.clone())
                .expect("insert should succeed");

        let plan = RoundPlan::from_speakers([ParticipantId::CHAR, bot("bot1"), bot("bot2")]);
        let remotes = FakeRemote::new(vec![
            (bot("bot1"), Ok("hi from bot1")),
            (bot("bot2"), Ok("hi from bot2")),
        ]);
        let mut sink = RecordingSink::default();

        let outcome = run_round(
            guard,
            pending,
            plan,
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| Ok("hi from char".to_string()),
            &remotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut sink,
        )
        .expect("round should succeed");

        assert_eq!(
            outcome.host_reply.as_ref().map(|r| r.text.as_str()),
            Some("hi from char")
        );
        assert_eq!(
            outcome
                .replies
                .iter()
                .map(|r| r.text.as_str())
                .collect::<Vec<_>>(),
            vec!["hi from char", "hi from bot1", "hi from bot2"]
        );

        let bot2_transcript = remotes.transcript_seen_by(&bot("bot2"));
        assert!(
            bot2_transcript.iter().any(|m| m.content == "hi from bot1"),
            "bot2 should have seen bot1's reply in its transcript"
        );

        assert_eq!(
            *store.finished.lock().unwrap(),
            vec![("hello".to_string(), "hi from char".to_string())]
        );

        assert_eq!(
            sink.events,
            vec![
                SinkEvent::Started(ParticipantId::CHAR),
                SinkEvent::Complete(ParticipantId::CHAR, "hi from char".to_string()),
                SinkEvent::Started(bot("bot1")),
                SinkEvent::Complete(bot("bot1"), "hi from bot1".to_string()),
                SinkEvent::Started(bot("bot2")),
                SinkEvent::Complete(bot("bot2"), "hi from bot2".to_string()),
                SinkEvent::RoundComplete,
            ]
        );
    }

    #[test]
    fn a_round_ships_the_hosts_continuity_and_trims_the_transcript_below_it() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);
        store.set_continuity(Some(ContinuityPayload {
            compacted_through: 2,
            rules: vec![],
            ..Default::default()
        }));
        let registry = registry_with_bots();
        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), registry.clone())
                .expect("insert should succeed");

        // ids: 1 = the user turn `begin` just inserted, 2 = char's reply
        // below, both at or before `compacted_through`; bot1's reply (3)
        // is the first id above it.
        let plan = RoundPlan::from_speakers([ParticipantId::CHAR, bot("bot1"), bot("bot2")]);
        let remotes = FakeRemote::new(vec![
            (bot("bot1"), Ok("hi from bot1")),
            (bot("bot2"), Ok("hi from bot2")),
        ]);
        let mut sink = RecordingSink::default();

        run_round(
            guard,
            pending,
            plan,
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| Ok("hi from char".to_string()),
            &remotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut sink,
        )
        .expect("round should succeed");

        let expected_payload = Some(ContinuityPayload {
            compacted_through: 2,
            rules: vec![],
            ..Default::default()
        });
        assert_eq!(remotes.continuity_seen_by(&bot("bot1")), expected_payload);
        assert_eq!(remotes.continuity_seen_by(&bot("bot2")), expected_payload);

        let bot1_tail = remotes.transcript_seen_by(&bot("bot1"));
        assert!(
            bot1_tail.iter().all(|m| m.id > 2),
            "bot1 must never see a message at or before compacted_through: {:?}",
            bot1_tail
        );
        let bot2_tail = remotes.transcript_seen_by(&bot("bot2"));
        assert!(
            bot2_tail.iter().all(|m| m.id > 2),
            "bot2 must never see a message at or before compacted_through: {:?}",
            bot2_tail
        );
        assert!(
            bot2_tail.iter().any(|m| m.content == "hi from bot1"),
            "bot2 should still see bot1's reply, which is above the checkpoint"
        );
    }

    #[test]
    fn a_round_with_no_compaction_carries_no_continuity_and_the_full_tail() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        // `RecordingStore::new` defaults `continuity` to `None`, matching a
        // companion that has never been compacted.
        let store = RecordingStore::new(None);
        let registry = registry_with_bots();
        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), registry.clone())
                .expect("insert should succeed");

        let plan = RoundPlan::from_speakers([ParticipantId::CHAR, bot("bot1")]);
        let remotes = FakeRemote::new(vec![(bot("bot1"), Ok("hi from bot1"))]);
        let mut sink = RecordingSink::default();

        run_round(
            guard,
            pending,
            plan,
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| Ok("hi from char".to_string()),
            &remotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut sink,
        )
        .expect("round should succeed");

        assert_eq!(remotes.continuity_seen_by(&bot("bot1")), None);
        let bot1_tail = remotes.transcript_seen_by(&bot("bot1"));
        assert_eq!(
            bot1_tail
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            vec!["hello", "hi from char"],
            "with no compaction, the tail is unchanged from before #182"
        );
    }

    #[test]
    fn a_remote_timeout_inserts_a_notice_and_the_round_still_completes() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);
        let registry = registry_with_bots();
        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), registry.clone())
                .expect("insert should succeed");

        let plan = RoundPlan::from_speakers([ParticipantId::CHAR, bot("bot1"), bot("bot2")]);
        let remotes = FakeRemote::new(vec![
            (bot("bot1"), Err(RemoteFailure::Timeout)),
            (bot("bot2"), Ok("hi from bot2")),
        ]);
        let mut sink = RecordingSink::default();

        let outcome = run_round(
            guard,
            pending,
            plan,
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| Ok("hi from char".to_string()),
            &remotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut sink,
        )
        .expect("a skipped speaker should not fail the round");

        assert_eq!(
            outcome
                .replies
                .iter()
                .map(|r| r.speaker_id.clone())
                .collect::<Vec<_>>(),
            vec![ParticipantId::CHAR, bot("bot2")],
            "bot1 was skipped, so it never produced a reply"
        );
        assert_eq!(
            *store.replies.lock().unwrap(),
            vec![
                (ParticipantId::CHAR, "hi from char".to_string()),
                (ParticipantId::SYSTEM, "bot1 did not respond".to_string()),
                (bot("bot2"), "hi from bot2".to_string()),
            ]
        );
        assert!(sink.events.contains(&SinkEvent::Skipped(bot("bot1"))));
        assert!(sink.events.contains(&SinkEvent::Complete(
            bot("bot2"),
            "hi from bot2".to_string()
        )));
        assert!(sink.events.contains(&SinkEvent::RoundComplete));
    }

    #[test]
    fn a_failed_host_generate_stops_the_round_without_a_notice_or_scoring() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);
        let registry = registry_with_bots();
        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), registry.clone())
                .expect("insert should succeed");

        let plan = RoundPlan::from_speakers([ParticipantId::CHAR, bot("bot1")]);
        let remotes = FakeRemote::new(vec![]);
        let mut sink = RecordingSink::default();

        let result = run_round(
            guard,
            pending,
            plan,
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| Err(std::io::Error::other("no model")),
            &remotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut sink,
        );

        assert!(result.is_err());
        assert_eq!(store.inserted.lock().unwrap().len(), 1);
        assert!(store.replies.lock().unwrap().is_empty());
        assert!(store.finished.lock().unwrap().is_empty());
        assert!(!sink.events.contains(&SinkEvent::RoundComplete));
    }

    #[test]
    fn the_turn_slot_is_held_through_round_complete_and_released_after() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);
        let registry = registry_with_bots();
        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), registry.clone())
                .expect("insert should succeed");

        struct SlotCheckSink {
            claimable_during_round_complete: Option<bool>,
        }
        impl RoundSink for SlotCheckSink {
            fn reply_started(&mut self, _speaker: &ParticipantId) {}
            fn token(&mut self, _speaker: &ParticipantId, _text: &str) {}
            fn reply_complete(&mut self, _reply: &PersistedReply) {}
            fn speaker_skipped(&mut self, _speaker: &ParticipantId, _notice: &PersistedReply) {}
            fn round_complete(
                &mut self,
                _attitude: Option<&(CompanionAttitude, CompanionAttitude)>,
            ) {
                self.claimable_during_round_complete = Some(SLOT.try_claim().is_some());
            }
        }
        let mut sink = SlotCheckSink {
            claimable_during_round_complete: None,
        };

        let plan = RoundPlan::from_speakers([ParticipantId::CHAR]);
        run_round(
            guard,
            pending,
            plan,
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| Ok("hi".to_string()),
            &NoRemotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut sink,
        )
        .expect("round should succeed");

        assert_eq!(
            sink.claimable_during_round_complete,
            Some(false),
            "the turn slot should still be held while round_complete runs"
        );
        assert!(
            SLOT.try_claim().is_some(),
            "the turn slot should be released once run_round returns"
        );
    }

    #[test]
    fn solo_plan_makes_the_same_store_calls_the_old_complete_made() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);
        let registry = ParticipantRegistry::solo("Alice", "Bob", None);
        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), registry.clone())
                .expect("insert should succeed");

        let plan = plan_round("hello", &registry, &no_followups());
        let remotes = FakeRemote::new(vec![]);
        let mut sink = RecordingSink::default();

        run_round(
            guard,
            pending,
            plan,
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| Ok("hi".to_string()),
            &remotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut sink,
        )
        .expect("round should succeed");

        assert_eq!(store.inserted.lock().unwrap().len(), 1);
        assert_eq!(store.replies.lock().unwrap().len(), 1);
        assert_eq!(store.finished.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_host_reply_mentioning_a_bot_persists_exactly_one_extra_reply() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);
        let registry = registry_with_bots();
        let pending = PendingTurn::begin(
            &guard,
            &store,
            1,
            1,
            "@char hi".to_string(),
            registry.clone(),
        )
        .expect("insert should succeed");

        let plan = plan_round(
            "@char hi",
            &registry,
            &RoutingPolicy {
                max_followup_depth: 1,
            },
        );
        let remotes = FakeRemote::new(vec![(bot("bot1"), Ok("sure"))]);
        let mut sink = RecordingSink::default();

        let outcome = run_round(
            guard,
            pending,
            plan,
            &store,
            &registry,
            &RoutingPolicy {
                max_followup_depth: 1,
            },
            &mut |_prompt, _on_token| Ok("hi @bot1 do you agree?".to_string()),
            &remotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut sink,
        )
        .expect("round should succeed");

        assert_eq!(
            outcome
                .replies
                .iter()
                .map(|r| r.speaker_id.clone())
                .collect::<Vec<_>>(),
            vec![ParticipantId::CHAR, bot("bot1")],
            "bot1's mention in char's reply should schedule exactly one follow-up"
        );
        assert_eq!(
            outcome.host_reply.as_ref().map(|r| r.text.as_str()),
            Some("hi @bot1 do you agree?")
        );
    }

    #[test]
    fn a_user_mention_of_one_bot_runs_only_it_and_finish_returns_none() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);
        let registry = registry_with_bots();
        let pending = PendingTurn::begin(
            &guard,
            &store,
            1,
            1,
            "@bot1 hi".to_string(),
            registry.clone(),
        )
        .expect("insert should succeed");

        let plan = plan_round("@bot1 hi", &registry, &no_followups());
        let remotes = FakeRemote::new(vec![(bot("bot1"), Ok("hello"))]);
        let mut sink = RecordingSink::default();

        let outcome = run_round(
            guard,
            pending,
            plan,
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| panic!("char should never be asked to speak"),
            &remotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut sink,
        )
        .expect("round should succeed");

        assert!(outcome.host_reply.is_none());
        assert_eq!(
            outcome
                .replies
                .iter()
                .map(|r| r.speaker_id.clone())
                .collect::<Vec<_>>(),
            vec![bot("bot1")]
        );
        assert!(
            store
                .replies
                .lock()
                .unwrap()
                .iter()
                .all(|(id, _)| *id != ParticipantId::CHAR),
            "char never spoke, so insert_reply(char) must never have been called"
        );
        assert!(
            store.finished.lock().unwrap().is_empty(),
            "finish should return None (and never call finish_turn) when char did not speak"
        );
    }

    #[test]
    fn broadcast_sees_the_user_turn_every_reply_and_the_notice_in_persistence_order() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);
        let registry = registry_with_bots();
        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), registry.clone())
                .expect("insert should succeed");

        let plan = RoundPlan::from_speakers([ParticipantId::CHAR, bot("bot1"), bot("bot2")]);
        let remotes = FakeRemote::new(vec![
            (bot("bot1"), Err(RemoteFailure::Timeout)),
            (bot("bot2"), Ok("hi from bot2")),
        ]);
        let mut sink = RecordingSink::default();
        let broadcasts: Mutex<Vec<ServerFrame>> = Mutex::new(Vec::new());

        run_round(
            guard,
            pending,
            plan,
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| Ok("hi from char".to_string()),
            &remotes,
            &|frame| broadcasts.lock().unwrap().push(frame),
            Duration::from_secs(30),
            &mut sink,
        )
        .expect("a skipped speaker should not fail the round");

        let broadcast_contents: Vec<(String, String)> = broadcasts
            .into_inner()
            .unwrap()
            .into_iter()
            .map(|frame| match frame {
                ServerFrame::Message(message) => (message.speaker_id, message.content),
                other => panic!("expected only Message frames, got {:?}", other),
            })
            .collect();

        assert_eq!(
            broadcast_contents,
            vec![
                ("user".to_string(), "hello".to_string()),
                ("char".to_string(), "hi from char".to_string()),
                ("system".to_string(), "bot1 did not respond".to_string()),
                ("bot2".to_string(), "hi from bot2".to_string()),
            ],
            "the user turn, then char's reply, then bot1's skip notice, then bot2's reply"
        );
    }

    fn user_message(id: i32, content: &str) -> Message {
        Message {
            id,
            ai: false,
            speaker_id: ParticipantId::USER.to_string(),
            content: content.to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    fn connected(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|id| id.to_string()).collect()
    }

    #[test]
    fn regenerate_target_routes_char_locally_regardless_of_who_is_connected() {
        assert_eq!(
            regenerate_target("char", &connected(&[])),
            RegenerateTarget::Local
        );
        assert_eq!(
            regenerate_target("char", &connected(&["bot1"])),
            RegenerateTarget::Local
        );
    }

    #[test]
    fn regenerate_target_routes_a_connected_bot_remotely() {
        assert_eq!(
            regenerate_target("bot1", &connected(&["bot1", "bot2"])),
            RegenerateTarget::Remote("bot1".to_string())
        );
    }

    #[test]
    fn regenerate_target_routes_a_disconnected_bot_offline() {
        assert_eq!(
            regenerate_target("bot1", &connected(&["bot2"])),
            RegenerateTarget::Offline("bot1".to_string())
        );
        // Solo mode's snapshot is always empty, so a stray bot id (reachable
        // only through a direct API call, never through the normal solo
        // flow) still routes Offline rather than panicking or routing Local.
        assert_eq!(
            regenerate_target("bot1", &connected(&[])),
            RegenerateTarget::Offline("bot1".to_string())
        );
    }

    #[test]
    fn regenerate_reply_local_calls_host_once_and_inserts_under_char() {
        let store = RecordingStore::new(None);
        let user_turn = user_message(1, "hi");
        let remotes = FakeRemote::new(vec![]);

        let persisted = regenerate_reply(
            RegenerateTarget::Local,
            &user_turn,
            &store,
            |prompt| {
                assert_eq!(prompt, "hi");
                Ok("hello again".to_string())
            },
            &remotes,
            Duration::from_secs(30),
        )
        .expect("local regenerate should succeed");

        assert_eq!(persisted.speaker_id, ParticipantId::CHAR);
        assert_eq!(persisted.text, "hello again");
        assert_eq!(
            *store.replies.lock().unwrap(),
            vec![(ParticipantId::CHAR, "hello again".to_string())]
        );
    }

    #[test]
    fn regenerate_reply_remote_never_calls_host_and_inserts_under_the_bot() {
        let store = RecordingStore::new(None);
        let user_turn = user_message(1, "hi @bot1");
        let remotes = FakeRemote::new(vec![(bot("bot1"), Ok("hi again from bot1"))]);

        let persisted = regenerate_reply(
            RegenerateTarget::Remote("bot1".to_string()),
            &user_turn,
            &store,
            |_prompt| panic!("the host must never be asked to speak for a remote reply"),
            &remotes,
            Duration::from_secs(30),
        )
        .expect("remote regenerate should succeed");

        assert_eq!(persisted.speaker_id, bot("bot1"));
        assert_eq!(persisted.text, "hi again from bot1");
        assert_eq!(
            *store.replies.lock().unwrap(),
            vec![(bot("bot1"), "hi again from bot1".to_string())]
        );
    }

    #[test]
    fn regenerate_reply_remote_carries_continuity_and_trims_the_transcript_below_it() {
        let store = RecordingStore::new(None);
        // Seed three prior messages (ids 1-3), same as a real chat would
        // have accumulated before this regenerate.
        store.insert_reply(&ParticipantId::USER, "hi").unwrap();
        store.insert_reply(&ParticipantId::CHAR, "hello").unwrap();
        store.insert_reply(&bot("bot1"), "old reply").unwrap();
        store.set_continuity(Some(ContinuityPayload {
            compacted_through: 2,
            ..Default::default()
        }));
        let user_turn = user_message(1, "hi @bot1");
        let remotes = FakeRemote::new(vec![(bot("bot1"), Ok("hi again from bot1"))]);

        let persisted = regenerate_reply(
            RegenerateTarget::Remote("bot1".to_string()),
            &user_turn,
            &store,
            |_prompt| panic!("the host must never be asked to speak for a remote reply"),
            &remotes,
            Duration::from_secs(30),
        )
        .expect("remote regenerate should succeed");

        assert_eq!(persisted.text, "hi again from bot1");
        assert_eq!(
            remotes.continuity_seen_by(&bot("bot1")),
            Some(ContinuityPayload {
                compacted_through: 2,
                ..Default::default()
            })
        );
        let tail = remotes.transcript_seen_by(&bot("bot1"));
        assert!(
            tail.iter().all(|m| m.id > 2),
            "a regenerate must never carry a message at or before compacted_through: {:?}",
            tail
        );
        assert!(tail.iter().any(|m| m.content == "old reply"));
    }

    #[test]
    fn regenerate_reply_remote_with_no_compaction_carries_no_continuity() {
        let store = RecordingStore::new(None);
        let user_turn = user_message(1, "hi @bot1");
        let remotes = FakeRemote::new(vec![(bot("bot1"), Ok("hi again from bot1"))]);

        regenerate_reply(
            RegenerateTarget::Remote("bot1".to_string()),
            &user_turn,
            &store,
            |_prompt| panic!("the host must never be asked to speak for a remote reply"),
            &remotes,
            Duration::from_secs(30),
        )
        .expect("remote regenerate should succeed");

        assert_eq!(remotes.continuity_seen_by(&bot("bot1")), None);
    }

    #[test]
    fn regenerate_reply_remote_failure_inserts_nothing() {
        let store = RecordingStore::new(None);
        let user_turn = user_message(1, "hi @bot1");
        let remotes = FakeRemote::new(vec![(bot("bot1"), Err(RemoteFailure::Timeout))]);

        let result = regenerate_reply(
            RegenerateTarget::Remote("bot1".to_string()),
            &user_turn,
            &store,
            |_prompt| panic!("the host must never be asked to speak for a remote reply"),
            &remotes,
            Duration::from_secs(30),
        );

        assert!(matches!(result, Err(RegenerateError::Generate(_))));
        assert!(store.replies.lock().unwrap().is_empty());
    }

    #[test]
    fn regenerate_reply_offline_target_fails_without_touching_the_store() {
        let store = RecordingStore::new(None);
        let user_turn = user_message(1, "hi @bot1");
        let remotes = FakeRemote::new(vec![]);

        let result = regenerate_reply(
            RegenerateTarget::Offline("bot1".to_string()),
            &user_turn,
            &store,
            |_prompt| panic!("an offline target must never generate"),
            &remotes,
            Duration::from_secs(30),
        );

        match result {
            Err(RegenerateError::Offline(speaker_id)) => assert_eq!(speaker_id, "bot1"),
            other => panic!("expected Offline, got {:?}", other),
        }
        assert!(store.replies.lock().unwrap().is_empty());
    }

    fn message_ref(id: i32, is_human: bool, tokens: usize) -> MessageRef {
        MessageRef {
            id,
            is_human,
            tokens,
        }
    }

    /// A tail whose token sum is well past `threshold_tokens` (each message
    /// carries `tokens: 100`, threshold is `50`), with no scene-break cue in
    /// `last_user_turn`, so a round against it always queues a `Threshold`
    /// draft — used by every "does the hook fire" test below.
    fn over_threshold_tail() -> CompactionTailView {
        CompactionTailView {
            compacted_through: None,
            messages: vec![
                message_ref(1, true, 100),
                message_ref(2, false, 100),
                message_ref(3, true, 100),
            ],
            last_user_turn: "hello there".to_string(),
            short_term_mem: 0,
            draft_pending: false,
            config: CompactionConfig {
                threshold_tokens: 50,
                min_messages: 1,
            },
        }
    }

    #[test]
    fn a_round_over_the_threshold_queues_exactly_one_draft_matching_the_outcome() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None).with_compaction_tail(over_threshold_tail());
        let registry = ParticipantRegistry::solo("Alice", "Bob", None);
        let pending =
            PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string(), registry.clone())
                .expect("insert should succeed");

        let plan = plan_round("hello", &registry, &no_followups());
        let mut sink = RecordingSink::default();

        let outcome = run_round(
            guard,
            pending,
            plan,
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| Ok("hi".to_string()),
            &NoRemotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut sink,
        )
        .expect("round should succeed");

        let queued = outcome
            .queued_draft
            .expect("a tail over threshold should queue a draft");
        assert_eq!(queued.trigger, CompactionTrigger::Threshold);
        assert_eq!(queued.range.from_id, 1);
        assert_eq!(queued.range.through_id, 3);
        assert_eq!(
            *store.queued_drafts.lock().unwrap(),
            vec![(queued.range, queued.trigger)]
        );
    }

    #[test]
    fn a_second_round_with_a_draft_already_pending_queues_none() {
        static SLOT: TurnSlot = TurnSlot::new();
        let store = RecordingStore::new(None).with_compaction_tail(over_threshold_tail());
        let registry = ParticipantRegistry::solo("Alice", "Bob", None);

        let first_guard = SLOT.try_claim().expect("slot should be free");
        let first_pending = PendingTurn::begin(
            &first_guard,
            &store,
            1,
            1,
            "hello".to_string(),
            registry.clone(),
        )
        .expect("insert should succeed");
        let first_outcome = run_round(
            first_guard,
            first_pending,
            plan_round("hello", &registry, &no_followups()),
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| Ok("hi".to_string()),
            &NoRemotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut RecordingSink::default(),
        )
        .expect("round should succeed");
        assert!(first_outcome.queued_draft.is_some());

        let second_guard = SLOT.try_claim().expect("slot should be free again");
        let second_pending = PendingTurn::begin(
            &second_guard,
            &store,
            1,
            1,
            "hello again".to_string(),
            registry.clone(),
        )
        .expect("insert should succeed");
        let second_outcome = run_round(
            second_guard,
            second_pending,
            plan_round("hello again", &registry, &no_followups()),
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| Ok("hi again".to_string()),
            &NoRemotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut RecordingSink::default(),
        )
        .expect("round should succeed");

        assert!(
            second_outcome.queued_draft.is_none(),
            "a pending draft must suppress every trigger"
        );
        assert_eq!(store.queued_drafts.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_scene_break_user_turn_on_a_long_tail_queues_a_scene_break_draft_before_that_turn() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        // human(1) ai(2) human(3) ai(4) human(5); the last human turn (id 5)
        // is where the scene-break cue lives, so the draft must stop at id 4.
        let tail = CompactionTailView {
            compacted_through: None,
            messages: vec![
                message_ref(1, true, 1),
                message_ref(2, false, 1),
                message_ref(3, true, 1),
                message_ref(4, false, 1),
                message_ref(5, true, 1),
            ],
            last_user_turn: "the next morning, everything was different".to_string(),
            short_term_mem: 0,
            draft_pending: false,
            config: CompactionConfig {
                threshold_tokens: 1_000_000,
                min_messages: 2,
            },
        };
        let store = RecordingStore::new(None).with_compaction_tail(tail);
        let registry = ParticipantRegistry::solo("Alice", "Bob", None);
        let pending = PendingTurn::begin(
            &guard,
            &store,
            1,
            1,
            "the next morning, everything was different".to_string(),
            registry.clone(),
        )
        .expect("insert should succeed");

        let outcome = run_round(
            guard,
            pending,
            plan_round(
                "the next morning, everything was different",
                &registry,
                &no_followups(),
            ),
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| Ok("good morning".to_string()),
            &NoRemotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut RecordingSink::default(),
        )
        .expect("round should succeed");

        let queued = outcome
            .queued_draft
            .expect("a long tail with a scene-break cue should queue a draft");
        assert_eq!(queued.trigger, CompactionTrigger::SceneBreak);
        assert_eq!(queued.range.from_id, 1);
        assert_eq!(
            queued.range.through_id, 4,
            "the range must stop at the message before the scene-break turn"
        );
    }

    #[test]
    fn a_mention_filtered_round_with_no_host_reply_still_runs_the_compaction_hook() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None).with_compaction_tail(over_threshold_tail());
        let registry = registry_with_bots();
        let pending = PendingTurn::begin(
            &guard,
            &store,
            1,
            1,
            "@bot1 hi".to_string(),
            registry.clone(),
        )
        .expect("insert should succeed");

        let plan = plan_round("@bot1 hi", &registry, &no_followups());
        let remotes = FakeRemote::new(vec![(bot("bot1"), Ok("hello"))]);

        let outcome = run_round(
            guard,
            pending,
            plan,
            &store,
            &registry,
            &no_followups(),
            &mut |_prompt, _on_token| panic!("char should never be asked to speak"),
            &remotes,
            &|_frame| {},
            Duration::from_secs(30),
            &mut RecordingSink::default(),
        )
        .expect("round should succeed");

        assert!(outcome.host_reply.is_none());
        assert!(
            outcome.queued_draft.is_some(),
            "the compaction hook must run even when the host did not speak"
        );
    }
}
