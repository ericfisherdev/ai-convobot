//! The round orchestrator (#131, Part A): turns one user message into a
//! sequence of speaker replies — the host companion, then each connected
//! joiner in join order — all under one held turn slot.
//!
//! Everything here is `Database`-free: [`plan_round`] and [`run_round`] are
//! driven entirely through the [`crate::chat_turn::TurnStore`],
//! [`RemoteGenerator`] and [`RoundSink`] seams, so a round can be exercised
//! with fakes in tests without a real sqlite file or model. The real
//! socket-backed `RemoteGenerator`, wired over `RemoteBots`, and the
//! transcript broadcasts that go with it, are #154 (Part B); this issue
//! ships [`NoRemotes`], which reports every remote speaker offline, so a
//! round with no joiners behaves exactly as `PendingTurn::complete` did
//! before this module existed.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::chat_turn::{PendingTurn, PersistedReply, TurnStore};
use crate::database::{CompanionAttitude, Message};
use crate::participants::{ParticipantId, ParticipantRegistry};
use crate::turn_slot::TurnGuard;

/// The newest-messages page size a remote speaker's request is built from:
/// the same page length `GET /api/message` uses for the frontend's first
/// page.
const TRANSCRIPT_TAIL_MESSAGES: usize = 50;

/// Who speaks, and in what order, for one round.
pub struct RoundPlan {
    pub speakers: Vec<ParticipantId>,
}

/// Builds a round's speaker order: the host companion, then every connected
/// joiner, in join order.
///
/// A joiner is only ever present in `registry` while its socket is
/// connected (`host.rs` inserts it on join and removes it on disconnect),
/// so `registry.iter_bots()` already is the connected set — there is no
/// separate "is this bot online" check to make here.
///
/// `user_message` is accepted but ignored today; #132 adds @mention-based
/// speaker filtering here without changing any call site.
pub fn plan_round(_user_message: &str, registry: &ParticipantRegistry) -> RoundPlan {
    RoundPlan {
        speakers: registry.iter_bots().map(|p| p.id.clone()).collect(),
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
/// `NoRemotes` (this issue's only `RemoteGenerator`) ignores every field;
/// #154's production implementation is what actually reads them, so they are
/// dead code from `cargo build`'s point of view until then. This module's
/// own tests exercise them today through `FakeRemote`.
#[allow(dead_code)]
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
}

/// Why a remote speaker did not produce a reply.
///
/// Only `Offline` is constructed outside tests today (`NoRemotes` always
/// returns it); `Timeout` and `Failed` are #154's production
/// `RemoteGenerator`'s to construct, once a real socket round-trip can
/// actually time out or fail. This module's tests construct all three.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteFailure {
    Offline,
    #[allow(dead_code)] // constructed by #154's production RemoteGenerator
    Timeout,
    #[allow(dead_code)] // constructed by #154's production RemoteGenerator
    Failed(String),
}

/// A [`RemoteGenerator`] with no remotes: every request reports the speaker
/// offline. Used by both prompting handlers until #154 wires the real,
/// socket-backed implementation in, so a round with no joiners is
/// unaffected by this module's existence.
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
}

/// Assigns each round a distinct id, for `RemoteRequest::round_id` — #154's
/// production `RemoteGenerator` uses it to route inbound frames on
/// `RemoteBots::subscribe_round` back to the round that is waiting on them.
static NEXT_ROUND_ID: AtomicU64 = AtomicU64::new(1);

fn next_round_id() -> u64 {
    NEXT_ROUND_ID.fetch_add(1, Ordering::Relaxed)
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
    plan: RoundPlan,
    store: &impl TurnStore,
    host: &mut dyn FnMut(&str, &mut dyn FnMut(&str)) -> io::Result<String>,
    remotes: &dyn RemoteGenerator,
    timeout: Duration,
    sink: &mut dyn RoundSink,
) -> io::Result<RoundOutcome> {
    let _turn_guard = turn_guard;
    let round_id = next_round_id();

    let mut host_reply: Option<PersistedReply> = None;
    let mut replies: Vec<PersistedReply> = Vec::new();

    for speaker in &plan.speakers {
        sink.reply_started(speaker);

        if *speaker == ParticipantId::CHAR {
            // A failed host generate ends the round immediately: no notice,
            // no further speakers, no `finish` — the #84 regression guard,
            // now covering the whole round instead of a single turn.
            let persisted = pending.reply(store, speaker.clone(), |generation_prompt| {
                host(generation_prompt, &mut |token| sink.token(speaker, token))
            })?;
            sink.reply_complete(&persisted);
            host_reply = Some(persisted.clone());
            replies.push(persisted);
            continue;
        }

        let tail = store
            .transcript_tail(TRANSCRIPT_TAIL_MESSAGES)
            .map_err(|e| io::Error::other(e.to_string()))?;
        let attempt = pending.reply(store, speaker.clone(), |_generation_prompt| {
            remotes
                .generate(
                    RemoteRequest {
                        round_id,
                        speaker,
                        transcript: &tail,
                        timeout,
                    },
                    &mut |token| sink.token(speaker, token),
                )
                .map_err(|failure| io::Error::other(format!("{:?}", failure)))
        });

        match attempt {
            Ok(persisted) => {
                sink.reply_complete(&persisted);
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
                sink.speaker_skipped(speaker, &notice);
            }
        }
    }

    let attitude = pending.finish(store, host_reply.as_ref().map(|r| r.text.as_str()));
    sink.round_complete(attitude.as_ref());

    Ok(RoundOutcome {
        host_reply,
        replies,
        attitude,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_turn::RecordingStore;
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
    }

    impl FakeRemote {
        fn new(outcomes: Vec<(ParticipantId, Result<&str, RemoteFailure>)>) -> Self {
            Self {
                outcomes: outcomes
                    .into_iter()
                    .map(|(id, result)| (id, result.map(|text| text.to_string())))
                    .collect(),
                seen_transcripts: Mutex::new(HashMap::new()),
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

    #[test]
    fn plan_round_orders_char_then_connected_joiners_by_join_order() {
        let mut registry = ParticipantRegistry::solo("Alice", "Bob", None);
        insert_bot(&mut registry, "bot1", "Ada");
        insert_bot(&mut registry, "bot2", "Grace");

        let plan = plan_round("hi", &registry);

        assert_eq!(
            plan.speakers,
            vec![ParticipantId::CHAR, bot("bot1"), bot("bot2")]
        );
    }

    #[test]
    fn plan_round_excludes_a_disconnected_joiner() {
        // A joiner that never joined (or has since disconnected) is simply
        // absent from the registry: `host.rs` removes it on disconnect.
        let registry = ParticipantRegistry::solo("Alice", "Bob", None);

        let plan = plan_round("hi", &registry);

        assert_eq!(plan.speakers, vec![ParticipantId::CHAR]);
    }

    #[test]
    fn a_round_persists_char_then_each_bot_in_order_and_scores_once() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);
        let pending = PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string())
            .expect("insert should succeed");

        let plan = RoundPlan {
            speakers: vec![ParticipantId::CHAR, bot("bot1"), bot("bot2")],
        };
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
            &mut |_prompt, _on_token| Ok("hi from char".to_string()),
            &remotes,
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
    fn a_remote_timeout_inserts_a_notice_and_the_round_still_completes() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = RecordingStore::new(None);
        let pending = PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string())
            .expect("insert should succeed");

        let plan = RoundPlan {
            speakers: vec![ParticipantId::CHAR, bot("bot1"), bot("bot2")],
        };
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
            &mut |_prompt, _on_token| Ok("hi from char".to_string()),
            &remotes,
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
        let pending = PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string())
            .expect("insert should succeed");

        let plan = RoundPlan {
            speakers: vec![ParticipantId::CHAR, bot("bot1")],
        };
        let remotes = FakeRemote::new(vec![]);
        let mut sink = RecordingSink::default();

        let result = run_round(
            guard,
            pending,
            plan,
            &store,
            &mut |_prompt, _on_token| Err(std::io::Error::other("no model")),
            &remotes,
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
        let pending = PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string())
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

        let plan = RoundPlan {
            speakers: vec![ParticipantId::CHAR],
        };
        run_round(
            guard,
            pending,
            plan,
            &store,
            &mut |_prompt, _on_token| Ok("hi".to_string()),
            &NoRemotes,
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
        let pending = PendingTurn::begin(&guard, &store, 1, 1, "hello".to_string())
            .expect("insert should succeed");

        let plan = plan_round("hello", &ParticipantRegistry::solo("Alice", "Bob", None));
        let remotes = FakeRemote::new(vec![]);
        let mut sink = RecordingSink::default();

        run_round(
            guard,
            pending,
            plan,
            &store,
            &mut |_prompt, _on_token| Ok("hi".to_string()),
            &remotes,
            Duration::from_secs(30),
            &mut sink,
        )
        .expect("round should succeed");

        assert_eq!(store.inserted.lock().unwrap().len(), 1);
        assert_eq!(store.replies.lock().unwrap().len(), 1);
        assert_eq!(store.finished.lock().unwrap().len(), 1);
    }
}
