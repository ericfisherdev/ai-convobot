//! The production [`RemoteGenerator`] (#154, Part B of the round
//! orchestrator — #131 is Part A): a joiner's reply, driven over the host
//! WebSocket via [`RemoteBots`] instead of a scripted fake.
//!
//! [`round::run_round`] executes on a blocking thread (like `main.rs`'s
//! `stream_round` does today) and `actix_ws::Session::text` is `async`, so
//! [`SocketRemoteGenerator::generate`] never holds a `Session` directly: it
//! sends the request and waits on [`RemoteBots::subscribe_round`]'s
//! synchronous channel instead, while each connection task in `host.rs`
//! drains the matching outbound channel into its own `Session`.
//!
//! [`round::run_round`]: crate::multiplayer::round::run_round

use std::sync::mpsc::RecvTimeoutError;
use std::time::Instant;

use actix_web::web;

use crate::multiplayer::protocol::{ClientFrame, ServerFrame};
use crate::multiplayer::remote_bots::RemoteBots;
use crate::multiplayer::round::{RemoteFailure, RemoteGenerator, RemoteRequest};

/// The three [`ClientFrame`] variants that answer a `GenerateRequest`,
/// stripped of their `round_id` — [`RemoteBots::route_inbound`] has already
/// used it to route the frame to this round's channel, so
/// [`SocketRemoteGenerator::generate`] does not need to look at it again.
/// [`ClientFrame::Join`] is the only variant this cannot come from: it never
/// carries a round id, so `route_inbound` drops it before it ever reaches a
/// round's channel.
enum ReplyEvent {
    Token(String),
    ReplyComplete(String),
    ReplyFailed(String),
}

impl TryFrom<ClientFrame> for ReplyEvent {
    /// The frame that was not a reply variant, so a caller that wants to
    /// log the surprise still can.
    type Error = ClientFrame;

    fn try_from(frame: ClientFrame) -> Result<Self, Self::Error> {
        match frame {
            ClientFrame::Token { text, .. } => Ok(ReplyEvent::Token(text)),
            ClientFrame::ReplyComplete { text, .. } => Ok(ReplyEvent::ReplyComplete(text)),
            ClientFrame::ReplyFailed { reason, .. } => Ok(ReplyEvent::ReplyFailed(reason)),
            other @ ClientFrame::Join { .. } => Err(other),
        }
    }
}

/// Unsubscribes `round_id` on every exit path out of
/// [`SocketRemoteGenerator::generate`] — success, failure, or an early
/// `return` — so a frame that arrives after this round has moved on (a
/// timed-out bot's reply landing late) is discarded by `route_inbound`
/// instead of being delivered to whichever later round reused the id.
struct RoundSubscription<'a> {
    remote_bots: &'a RemoteBots,
    round_id: u64,
}

impl Drop for RoundSubscription<'_> {
    fn drop(&mut self) {
        self.remote_bots.unsubscribe_round(self.round_id);
    }
}

/// The [`RemoteGenerator`] `main.rs` wires in for `Host` mode: a real
/// socket round trip over [`RemoteBots`], instead of [`crate::multiplayer::round::NoRemotes`]'s
/// unconditional offline.
pub struct SocketRemoteGenerator {
    remote_bots: web::Data<RemoteBots>,
}

impl SocketRemoteGenerator {
    pub fn new(remote_bots: web::Data<RemoteBots>) -> Self {
        SocketRemoteGenerator { remote_bots }
    }
}

impl RemoteGenerator for SocketRemoteGenerator {
    fn generate(
        &self,
        request: RemoteRequest<'_>,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String, RemoteFailure> {
        // Subscribed before the request is sent, so a reply cannot arrive
        // (and be dropped as "unsubscribed") before this round is listening
        // for it.
        let receiver = self.remote_bots.subscribe_round(request.round_id);
        let _subscription = RoundSubscription {
            remote_bots: self.remote_bots.get_ref(),
            round_id: request.round_id,
        };

        if self
            .remote_bots
            .send(
                request.speaker,
                ServerFrame::GenerateRequest {
                    round_id: request.round_id,
                    transcript: request.transcript.to_vec(),
                    continuity: request.continuity.cloned(),
                },
            )
            .is_err()
        {
            println!(
                "multiplayer: {} is not connected, skipping its turn",
                request.speaker
            );
            return Err(RemoteFailure::Offline);
        }

        let deadline = Instant::now() + request.timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match receiver.recv_timeout(remaining) {
                Ok((from, _frame)) if from != *request.speaker => {
                    // A frame from a different bot than the one this round
                    // asked to speak: `RemoteBots::route_inbound` only keys
                    // on round id, not sender, so this can only happen if
                    // two speakers in the same round somehow shared one —
                    // ignored rather than trusted, since it did not answer
                    // this request.
                    continue;
                }
                Ok((_, frame)) => match ReplyEvent::try_from(frame) {
                    Ok(ReplyEvent::Token(text)) => on_token(&text),
                    Ok(ReplyEvent::ReplyComplete(text)) => return Ok(text),
                    Ok(ReplyEvent::ReplyFailed(reason)) => {
                        println!(
                            "multiplayer: {} failed to reply: {}",
                            request.speaker, reason
                        );
                        return Err(RemoteFailure::Failed(reason));
                    }
                    Err(_not_a_reply) => continue,
                },
                Err(RecvTimeoutError::Timeout) => {
                    println!(
                        "multiplayer: {} timed out generating a reply",
                        request.speaker
                    );
                    return Err(RemoteFailure::Timeout);
                }
                Err(RecvTimeoutError::Disconnected) => {
                    println!(
                        "multiplayer: {} disconnected while generating a reply",
                        request.speaker
                    );
                    return Err(RemoteFailure::Offline);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::context::{QuoteLine, QuoteSpeaker};
    use crate::database::Message;
    use crate::multiplayer::protocol::ContinuityPayload;
    use crate::participants::ParticipantId;
    use std::time::Duration;

    fn id(s: &str) -> ParticipantId {
        ParticipantId::parse(s).unwrap()
    }

    fn sample_message(id: i32, speaker_id: &str, content: &str) -> Message {
        Message {
            id,
            ai: speaker_id != "user",
            speaker_id: speaker_id.to_string(),
            content: content.to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    fn request<'a>(
        round_id: u64,
        speaker: &'a ParticipantId,
        transcript: &'a [Message],
    ) -> RemoteRequest<'a> {
        RemoteRequest {
            round_id,
            speaker,
            transcript,
            timeout: Duration::from_millis(200),
            continuity: None,
        }
    }

    #[test]
    fn an_unregistered_speaker_is_reported_offline() {
        let bots = web::Data::new(RemoteBots::new());
        let generator = SocketRemoteGenerator::new(bots);
        let speaker = id("bot1");

        let result = generator.generate(request(1, &speaker, &[]), &mut |_| {});

        assert_eq!(result.unwrap_err(), RemoteFailure::Offline);
    }

    #[test]
    fn tokens_reach_on_token_in_order_and_reply_complete_returns_the_text() {
        let bots = web::Data::new(RemoteBots::new());
        let mut rx = bots.register(id("bot1")).unwrap();
        let generator = SocketRemoteGenerator::new(bots.clone());
        let speaker = id("bot1");

        let handle = std::thread::spawn(move || {
            let mut tokens = Vec::new();
            let result = generator.generate(request(1, &speaker, &[]), &mut |t| {
                tokens.push(t.to_string())
            });
            (result, tokens)
        });

        // Answer the GenerateRequest the generation thread just sent.
        let frame = rx.blocking_recv().expect("GenerateRequest should be sent");
        assert!(matches!(
            frame,
            ServerFrame::GenerateRequest { round_id: 1, .. }
        ));

        bots.route_inbound(
            &id("bot1"),
            ClientFrame::Token {
                round_id: 1,
                text: "hel".to_string(),
            },
        );
        bots.route_inbound(
            &id("bot1"),
            ClientFrame::Token {
                round_id: 1,
                text: "lo".to_string(),
            },
        );
        bots.route_inbound(
            &id("bot1"),
            ClientFrame::ReplyComplete {
                round_id: 1,
                text: "hello".to_string(),
            },
        );

        let (result, tokens) = handle.join().expect("generation thread should not panic");
        assert_eq!(tokens, vec!["hel".to_string(), "lo".to_string()]);
        assert_eq!(result, Ok("hello".to_string()));
    }

    #[test]
    fn reply_failed_maps_to_failed_with_the_reason() {
        let bots = web::Data::new(RemoteBots::new());
        let mut rx = bots.register(id("bot1")).unwrap();
        let generator = SocketRemoteGenerator::new(bots.clone());
        let speaker = id("bot1");

        let handle =
            std::thread::spawn(move || generator.generate(request(1, &speaker, &[]), &mut |_| {}));

        rx.blocking_recv().expect("GenerateRequest should be sent");
        bots.route_inbound(
            &id("bot1"),
            ClientFrame::ReplyFailed {
                round_id: 1,
                reason: "no model loaded".to_string(),
            },
        );

        let result = handle.join().expect("generation thread should not panic");
        assert_eq!(
            result.unwrap_err(),
            RemoteFailure::Failed("no model loaded".to_string())
        );
    }

    #[test]
    fn no_event_before_the_deadline_times_out_and_drops_the_subscription() {
        let bots = web::Data::new(RemoteBots::new());
        let _rx = bots.register(id("bot1")).unwrap();
        let generator = SocketRemoteGenerator::new(bots.clone());
        let speaker = id("bot1");

        let result = generator.generate(request(1, &speaker, &[]), &mut |_| {});

        assert_eq!(result.unwrap_err(), RemoteFailure::Timeout);

        // The subscription is gone: a frame for round 1 arriving now (the
        // timed-out bot's reply landing late) is dropped, not delivered to
        // this generator's already-returned call.
        bots.route_inbound(
            &id("bot1"),
            ClientFrame::ReplyComplete {
                round_id: 1,
                text: "too late".to_string(),
            },
        );
    }

    #[test]
    fn an_event_from_a_different_speaker_is_ignored() {
        let bots = web::Data::new(RemoteBots::new());
        let mut rx1 = bots.register(id("bot1")).unwrap();
        let _rx2 = bots.register(id("bot2")).unwrap();
        let generator = SocketRemoteGenerator::new(bots.clone());
        let speaker = id("bot1");

        let handle =
            std::thread::spawn(move || generator.generate(request(1, &speaker, &[]), &mut |_| {}));

        rx1.blocking_recv().expect("GenerateRequest should be sent");
        // bot2 was never asked to speak this round, so its round-scoped
        // frame is routed to the same subscription (both bots' inbound
        // frames for round 1 land on the one channel `subscribe_round`
        // opened) but must be ignored rather than accepted as bot1's reply.
        bots.route_inbound(
            &id("bot2"),
            ClientFrame::ReplyComplete {
                round_id: 1,
                text: "wrong speaker".to_string(),
            },
        );
        bots.route_inbound(
            &id("bot1"),
            ClientFrame::ReplyComplete {
                round_id: 1,
                text: "right speaker".to_string(),
            },
        );

        let result = handle.join().expect("generation thread should not panic");
        assert_eq!(result, Ok("right speaker".to_string()));
    }

    #[test]
    fn the_generate_request_frame_carries_the_transcript_passed_in() {
        let bots = web::Data::new(RemoteBots::new());
        let mut rx = bots.register(id("bot1")).unwrap();
        let generator = SocketRemoteGenerator::new(bots.clone());
        let speaker = id("bot1");
        let transcript = vec![sample_message(1, "user", "hi")];

        let handle = std::thread::spawn(move || {
            generator.generate(request(1, &speaker, &transcript), &mut |_| {})
        });

        let frame = rx.blocking_recv().expect("GenerateRequest should be sent");
        assert_eq!(
            frame,
            ServerFrame::GenerateRequest {
                round_id: 1,
                transcript: vec![sample_message(1, "user", "hi")],
                continuity: None,
            }
        );

        // Unblock the generation thread so the test does not leak it: any
        // terminal frame will do, this test only cares about the request.
        bots.route_inbound(
            &id("bot1"),
            ClientFrame::ReplyFailed {
                round_id: 1,
                reason: "test cleanup".to_string(),
            },
        );
        let _ = handle.join();
    }

    #[test]
    fn the_generate_request_frame_carries_the_continuity_payload_passed_in() {
        let bots = web::Data::new(RemoteBots::new());
        let mut rx = bots.register(id("bot1")).unwrap();
        let generator = SocketRemoteGenerator::new(bots.clone());
        let speaker = id("bot1");
        let payload = ContinuityPayload {
            compacted_through: 2,
            rules: vec![QuoteLine {
                speaker: QuoteSpeaker::User,
                text: "never call me Bob".to_string(),
            }],
            ..Default::default()
        };

        let handle = std::thread::spawn(move || {
            generator.generate(
                RemoteRequest {
                    round_id: 1,
                    speaker: &speaker,
                    transcript: &[],
                    timeout: Duration::from_millis(200),
                    continuity: Some(&payload),
                },
                &mut |_| {},
            )
        });

        let frame = rx.blocking_recv().expect("GenerateRequest should be sent");
        match frame {
            ServerFrame::GenerateRequest { continuity, .. } => {
                assert_eq!(
                    continuity,
                    Some(ContinuityPayload {
                        compacted_through: 2,
                        rules: vec![QuoteLine {
                            speaker: QuoteSpeaker::User,
                            text: "never call me Bob".to_string(),
                        }],
                        ..Default::default()
                    })
                );
            }
            other => panic!("expected a GenerateRequest, got {:?}", other),
        }

        bots.route_inbound(
            &id("bot1"),
            ClientFrame::ReplyFailed {
                round_id: 1,
                reason: "test cleanup".to_string(),
            },
        );
        let _ = handle.join();
    }
}
