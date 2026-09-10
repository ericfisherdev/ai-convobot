//! Wire frames exchanged over the host's `/api/multiplayer/ws` socket.
//!
//! Pure serde types shared by the host (this crate, #129) and the joiner
//! binary (also this crate, #130), sent as WebSocket text frames. Both
//! frame enums are tagged `{"type": "...", ...fields}` so a future variant
//! addition is additive on the wire.
//!
//! Heartbeats use native WebSocket ping/pong control frames (`host.rs`'s
//! `Session::ping`/`AggregatedMessage::Pong`), not a JSON variant here.
//!
//! `ServerFrame::Message`, `ServerFrame::GenerateRequest` and
//! `ClientFrame::ReplyFailed` are added by #130 (the joiner matches or sends
//! all three) even though #131's host-side round orchestrator is what
//! actually constructs and routes them in production: `RemoteBots::send`
//! targets one bot with `GenerateRequest`, and `RemoteBots::route_inbound`
//! delivers a `ReplyFailed` back to the round that requested it.
//!
//! `ClientFrame::Token` and `ClientFrame::ReplyComplete` are added by #153:
//! the joiner's own `LocalModelGeneration` (`multiplayer::remote_generation`)
//! constructs and sends both, streaming a reply back token by token then the
//! final text. `RemoteBots::route_inbound` reading them on the host side is
//! #154's job.
//!
//! `ServerFrame::GenerateRequest.continuity` is added by #182: once the
//! host has ever compacted a companion's transcript, every request carries
//! a [`ContinuityPayload`] alongside the (now-trimmed) transcript, so a
//! joiner's own reply is grounded in the same committed summaries and
//! rules the host renders for itself. `#[serde(default)]` keeps a
//! pre-#182 `GenerateRequest` (no `continuity` key at all) deserialising
//! as `None`, so `PROTOCOL_VERSION` does not need to change. #186 is what
//! actually reads it on the joiner side (`HostContinuity`); this crate's
//! own joiner code (`multiplayer::joiner`/`remote_generation`) does not
//! look at the field yet.

use serde::{Deserialize, Serialize};

use crate::compaction::context::{CompactionContext, PinnedMessage, QuoteLine};

use crate::database::Message as DbMessage;
use crate::participants::{ParticipantId, ParticipantKind};

/// The wire protocol version. Bumped when a breaking change is made to
/// [`ClientFrame`]/[`ServerFrame`]; `host.rs::await_join` rejects a `Join`
/// whose `protocol_version` does not match.
pub const PROTOCOL_VERSION: u32 = 1;

/// An avatar the joiner uploads as part of [`ClientFrame::Join`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvatarUpload {
    /// The client's claimed MIME type, informational only:
    /// `avatar::validate_avatar` detects the real format from magic bytes.
    pub mime: String,
    /// Standard (padded) base64-encoded image bytes.
    pub data_base64: String,
}

/// A frame sent by a joiner to the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientFrame {
    /// The only frame a joiner may send before it has been admitted to the
    /// registry; every other frame received before this one is rejected.
    Join {
        protocol_version: u32,
        id: ParticipantId,
        display_name: String,
        avatar: Option<AvatarUpload>,
        /// `handshake::encode(&handshake::join_proof(password, &nonce, &id))`,
        /// where `nonce` is `handshake::decode(Challenge.nonce)`.
        proof: String,
    },
    /// Sent by the joiner instead of a generated reply when it cannot
    /// answer a [`ServerFrame::GenerateRequest`]: a local turn already in
    /// progress, a failed generation thread spawn, or the generator itself
    /// erroring (`LocalModelGeneration`, #153).
    ReplyFailed { round_id: u64, reason: String },
    /// One token of a reply as it is generated, sent by
    /// `LocalModelGeneration` (#153) for every token `llm::prompt_streaming`
    /// produces.
    Token { round_id: u64, text: String },
    /// The reply's full, cleaned text once generation finishes, sent by
    /// `LocalModelGeneration` (#153) after the last `Token`.
    ReplyComplete { round_id: u64, text: String },
}

impl ClientFrame {
    /// The round id this frame belongs to, or `None` if it is not
    /// round-scoped. `Join` is the only variant that is not: every other
    /// variant answers one `GenerateRequest`. This is the one accessor
    /// `RemoteBots::route_inbound` keys on.
    pub fn round_id(&self) -> Option<u64> {
        match self {
            ClientFrame::Join { .. } => None,
            ClientFrame::ReplyFailed { round_id, .. } => Some(*round_id),
            ClientFrame::Token { round_id, .. } => Some(*round_id),
            ClientFrame::ReplyComplete { round_id, .. } => Some(*round_id),
        }
    }
}

/// Why the host refused a [`ClientFrame::Join`]. Serialised snake_case, so
/// a tag-only variant becomes the bare string `"bad_proof"` and
/// `InvalidAvatar` becomes `{"invalid_avatar": "<reason>"}` (serde's
/// default externally-tagged representation for a newtype variant); either
/// way the joiner UI (#130) can show `reason` directly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectReason {
    UnsupportedProtocol,
    BadProof,
    NoHostPassword,
    DuplicateId,
    ReservedId,
    InvalidAvatar(String),
    JoinTimeout,
}

/// The compaction summaries and rules the host ships to joiners once it has
/// ever compacted a companion's transcript (#182), carried on every
/// [`ServerFrame::GenerateRequest`]. Field names and types mirror
/// [`CompactionContext`]'s exactly, minus `companion_state` (per-instance
/// by design: every instance renders its own companion's state, never the
/// host's) and `recalled_facts` (a joiner recalls from its own tantivy
/// index, never the host's). Unlike `CompactionContext::compacted_through`,
/// this field is a plain `i32`: a payload only ever exists once the host's
/// own `compacted_through` is `Some`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ContinuityPayload {
    pub compacted_through: i32,
    pub user_state: Vec<String>,
    pub rules: Vec<QuoteLine>,
    pub backstory: Vec<String>,
    pub open_threads: Vec<String>,
    pub key_quotes: Vec<QuoteLine>,
    pub rolling_summary: String,
    pub recent_detail: String,
    pub pins: Vec<PinnedMessage>,
}

impl From<CompactionContext> for ContinuityPayload {
    /// `compacted_through.unwrap_or(0)` never actually falls back to `0` in
    /// production: [`crate::chat_turn::TurnStore::continuity`] only builds a
    /// payload at all when the host's `compacted_through` is `Some`.
    fn from(ctx: CompactionContext) -> Self {
        ContinuityPayload {
            compacted_through: ctx.compacted_through.unwrap_or(0),
            user_state: ctx.user_state,
            rules: ctx.rules,
            backstory: ctx.backstory,
            open_threads: ctx.open_threads,
            key_quotes: ctx.key_quotes,
            rolling_summary: ctx.rolling_summary,
            recent_detail: ctx.recent_detail,
            pins: ctx.pins,
        }
    }
}

impl ContinuityPayload {
    /// Rebuilds a full [`CompactionContext`] from this wire payload plus the
    /// two per-instance fields it dropped. #186's `HostContinuity` is the
    /// only production caller, once the joiner side renders this the same
    /// way the host renders its own `CompactionContext`.
    #[allow(dead_code)] // wired up by #186
    pub fn into_context(
        self,
        companion_state: Vec<String>,
        recalled_facts: Vec<String>,
    ) -> CompactionContext {
        CompactionContext {
            compacted_through: Some(self.compacted_through),
            user_state: self.user_state,
            companion_state,
            rules: self.rules,
            backstory: self.backstory,
            open_threads: self.open_threads,
            key_quotes: self.key_quotes,
            rolling_summary: self.rolling_summary,
            recent_detail: self.recent_detail,
            pins: self.pins,
            recalled_facts,
        }
    }
}

/// One row of [`ServerFrame::Joined`]'s participant list, and the JSON row
/// of `GET /api/multiplayer/participants`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticipantSummary {
    pub id: ParticipantId,
    pub display_name: String,
    pub kind: ParticipantKind,
    pub avatar_url: Option<String>,
    pub connected: bool,
}

/// A frame sent by the host to a joiner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerFrame {
    /// Sent immediately after the socket upgrades, before any `Join` is
    /// read. `nonce` is standard base64 of [`crate::multiplayer::handshake::NONCE_BYTES`]
    /// random bytes.
    Challenge {
        protocol_version: u32,
        nonce: String,
    },
    /// Sent once a `Join` has been authenticated and admitted.
    Joined {
        self_id: ParticipantId,
        participants: Vec<ParticipantSummary>,
        /// The last 50 messages, verbatim, so the joiner can seed its
        /// transcript mirror with the same rows `GET /api/message` returns.
        transcript: Vec<DbMessage>,
    },
    /// Sent instead of `Joined` when a `Join` is refused.
    Rejected { reason: RejectReason },
    /// Broadcast to every other connected participant when a new one joins.
    ParticipantJoined(ParticipantSummary),
    /// Broadcast to every other connected participant when one disconnects.
    ParticipantLeft { id: ParticipantId },
    /// A message the host persisted, mirrored to every joiner so its own
    /// transcript stays in sync (#131 broadcasts this; the joiner's
    /// `RemoteTranscript::push` is the counterpart).
    Message(DbMessage),
    /// A message the host edited (`PUT /api/message/{id}`), mirrored so a
    /// joiner's transcript carries the same content the host now has,
    /// instead of the stale pre-edit text (#135; the joiner's
    /// `RemoteTranscript::replace_message` is the counterpart).
    MessageEdited { message: DbMessage },
    /// A message the host deleted (`DELETE /api/message/{id}`) or popped for
    /// a regenerate (`GET /api/prompt/regenerate`), mirrored so a joiner
    /// drops the same row rather than showing a message the host no longer
    /// has (#135; the joiner's `RemoteTranscript::remove` is the
    /// counterpart). Sent for a regenerate's pop before the replacement
    /// reply is generated, so a joiner never shows the old and new reply
    /// side by side.
    MessageRemoved { id: i32 },
    /// Sent to one remote bot when it is its turn to generate a reply.
    /// `transcript` is the context to generate from, sent by the host so
    /// the joiner never has to trust its own possibly-stale mirror for a
    /// round's correctness.
    GenerateRequest {
        round_id: u64,
        transcript: Vec<DbMessage>,
        /// `Some` once the host has ever compacted this companion's
        /// transcript (#182); `None` in solo mode and in host mode before
        /// the first checkpoint commits. `#[serde(default)]` so a
        /// pre-#182 frame (no `continuity` key at all) still deserialises;
        /// `skip_serializing_if` keeps a `None` payload's wire shape
        /// byte-identical to before this field existed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        continuity: Option<ContinuityPayload>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::context::QuoteSpeaker;

    fn sample_avatar() -> AvatarUpload {
        AvatarUpload {
            mime: "image/png".to_string(),
            data_base64: "aGVsbG8=".to_string(),
        }
    }

    fn sample_join() -> ClientFrame {
        ClientFrame::Join {
            protocol_version: PROTOCOL_VERSION,
            id: ParticipantId::parse("bot1").unwrap(),
            display_name: "Ada".to_string(),
            avatar: Some(sample_avatar()),
            proof: "cHJvb2Y=".to_string(),
        }
    }

    #[test]
    fn join_round_trips_through_json() {
        let frame = sample_join();
        let json = serde_json::to_string(&frame).unwrap();
        let back: ClientFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(back, frame);
    }

    #[test]
    fn join_has_the_exact_wire_shape_a_joiner_depends_on() {
        // Pinned so a future rename of a `Join` field cannot silently break
        // an older joiner build (#130).
        let frame = sample_join();
        let json: serde_json::Value = serde_json::to_value(&frame).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "type": "join",
                "protocol_version": 1,
                "id": "bot1",
                "display_name": "Ada",
                "avatar": {
                    "mime": "image/png",
                    "data_base64": "aGVsbG8="
                },
                "proof": "cHJvb2Y="
            })
        );
    }

    #[test]
    fn join_with_no_avatar_round_trips() {
        let frame = ClientFrame::Join {
            protocol_version: PROTOCOL_VERSION,
            id: ParticipantId::parse("bot1").unwrap(),
            display_name: "Ada".to_string(),
            avatar: None,
            proof: "cHJvb2Y=".to_string(),
        };
        let json = serde_json::to_string(&frame).unwrap();
        let back: ClientFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(back, frame);
    }

    #[test]
    fn round_id_is_none_for_join() {
        assert_eq!(sample_join().round_id(), None);
    }

    #[test]
    fn reply_failed_round_trips_and_carries_its_round_id() {
        let frame = ClientFrame::ReplyFailed {
            round_id: 7,
            reason: "generation not implemented on this joiner".to_string(),
        };
        let json = serde_json::to_string(&frame).unwrap();
        let back: ClientFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(back, frame);
        assert_eq!(frame.round_id(), Some(7));
    }

    #[test]
    fn challenge_round_trips_through_json() {
        let frame = ServerFrame::Challenge {
            protocol_version: PROTOCOL_VERSION,
            nonce: "bm9uY2U=".to_string(),
        };
        let json = serde_json::to_string(&frame).unwrap();
        let back: ServerFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(back, frame);
    }

    #[test]
    fn joined_round_trips_through_json() {
        let frame = ServerFrame::Joined {
            self_id: ParticipantId::parse("bot1").unwrap(),
            participants: vec![ParticipantSummary {
                id: ParticipantId::USER,
                display_name: "Alice".to_string(),
                kind: ParticipantKind::Human,
                avatar_url: None,
                connected: true,
            }],
            transcript: vec![DbMessage {
                id: 1,
                ai: false,
                speaker_id: "user".to_string(),
                content: "hi".to_string(),
                created_at: "2024-01-01T00:00:00Z".to_string(),
            }],
        };
        let json = serde_json::to_string(&frame).unwrap();
        let back: ServerFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(back, frame);
    }

    #[test]
    fn rejected_round_trips_every_tag_only_reason() {
        for reason in [
            RejectReason::UnsupportedProtocol,
            RejectReason::BadProof,
            RejectReason::NoHostPassword,
            RejectReason::DuplicateId,
            RejectReason::ReservedId,
            RejectReason::JoinTimeout,
        ] {
            let frame = ServerFrame::Rejected {
                reason: reason.clone(),
            };
            let json = serde_json::to_string(&frame).unwrap();
            let back: ServerFrame = serde_json::from_str(&json).unwrap();
            assert_eq!(back, frame);
        }
    }

    #[test]
    fn rejected_round_trips_invalid_avatar_with_its_message() {
        let frame = ServerFrame::Rejected {
            reason: RejectReason::InvalidAvatar("too large".to_string()),
        };
        let json = serde_json::to_string(&frame).unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&json).unwrap()["reason"],
            serde_json::json!({"invalid_avatar": "too large"})
        );
        let back: ServerFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(back, frame);
    }

    #[test]
    fn participant_joined_and_left_round_trip() {
        let joined = ServerFrame::ParticipantJoined(ParticipantSummary {
            id: ParticipantId::parse("bot1").unwrap(),
            display_name: "Ada".to_string(),
            kind: ParticipantKind::RemoteBot,
            avatar_url: Some("/api/multiplayer/participants/bot1/avatar".to_string()),
            connected: true,
        });
        let json = serde_json::to_string(&joined).unwrap();
        assert_eq!(serde_json::from_str::<ServerFrame>(&json).unwrap(), joined);

        let left = ServerFrame::ParticipantLeft {
            id: ParticipantId::parse("bot1").unwrap(),
        };
        let json = serde_json::to_string(&left).unwrap();
        assert_eq!(serde_json::from_str::<ServerFrame>(&json).unwrap(), left);
    }

    fn sample_message() -> DbMessage {
        DbMessage {
            id: 1,
            ai: false,
            speaker_id: "user".to_string(),
            content: "hi".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn message_round_trips_through_json() {
        let frame = ServerFrame::Message(sample_message());
        let json = serde_json::to_string(&frame).unwrap();
        assert_eq!(serde_json::from_str::<ServerFrame>(&json).unwrap(), frame);
    }

    #[test]
    fn message_edited_round_trips_through_json() {
        let frame = ServerFrame::MessageEdited {
            message: sample_message(),
        };
        let json = serde_json::to_string(&frame).unwrap();
        assert_eq!(serde_json::from_str::<ServerFrame>(&json).unwrap(), frame);
    }

    #[test]
    fn message_removed_round_trips_through_json() {
        let frame = ServerFrame::MessageRemoved { id: 42 };
        let json = serde_json::to_string(&frame).unwrap();
        assert_eq!(serde_json::from_str::<ServerFrame>(&json).unwrap(), frame);
    }

    #[test]
    fn generate_request_round_trips_through_json() {
        let frame = ServerFrame::GenerateRequest {
            round_id: 3,
            transcript: vec![sample_message()],
            continuity: None,
        };
        let json = serde_json::to_string(&frame).unwrap();
        assert_eq!(serde_json::from_str::<ServerFrame>(&json).unwrap(), frame);
    }

    fn sample_continuity() -> ContinuityPayload {
        ContinuityPayload {
            compacted_through: 12,
            user_state: vec!["loves the sea".to_string()],
            rules: vec![QuoteLine {
                speaker: QuoteSpeaker::User,
                text: "never call me Bob".to_string(),
            }],
            backstory: vec!["grew up near a lighthouse".to_string()],
            open_threads: vec!["waiting on Rina's visit".to_string()],
            key_quotes: vec![QuoteLine {
                speaker: QuoteSpeaker::Companion,
                text: "I will remember that promise always".to_string(),
            }],
            rolling_summary: "settling into the new place".to_string(),
            recent_detail: "moved into the lighthouse".to_string(),
            pins: vec![PinnedMessage {
                message_id: 5,
                speaker_id: "user".to_string(),
                content: "I promise I will never lie to you".to_string(),
            }],
        }
    }

    #[test]
    fn generate_request_with_continuity_round_trips_through_json() {
        let frame = ServerFrame::GenerateRequest {
            round_id: 3,
            transcript: vec![sample_message()],
            continuity: Some(sample_continuity()),
        };
        let json = serde_json::to_string(&frame).unwrap();
        assert_eq!(serde_json::from_str::<ServerFrame>(&json).unwrap(), frame);
    }

    #[test]
    fn a_pre_182_generate_request_with_no_continuity_key_still_deserialises() {
        // Pinned so a pre-#182 host/joiner pair (neither side aware of
        // `continuity`) stays interoperable with a #182 build on either
        // end: `#[serde(default)]` is what makes a missing key parse as
        // `None` rather than a deserialize error.
        let json = serde_json::json!({
            "type": "generate_request",
            "round_id": 1,
            "transcript": []
        })
        .to_string();
        let frame: ServerFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(
            frame,
            ServerFrame::GenerateRequest {
                round_id: 1,
                transcript: vec![],
                continuity: None,
            }
        );
    }

    #[test]
    fn continuity_payload_from_compaction_context_drops_companion_state_and_recalled_facts() {
        let ctx = CompactionContext {
            compacted_through: Some(12),
            user_state: vec!["loves the sea".to_string()],
            companion_state: vec!["is protective of the lighthouse".to_string()],
            rules: sample_continuity().rules,
            backstory: vec!["grew up near a lighthouse".to_string()],
            open_threads: vec!["waiting on Rina's visit".to_string()],
            key_quotes: sample_continuity().key_quotes,
            rolling_summary: "settling into the new place".to_string(),
            recent_detail: "moved into the lighthouse".to_string(),
            pins: sample_continuity().pins,
            recalled_facts: vec!["the lighthouse was built in 1890".to_string()],
        };

        let payload = ContinuityPayload::from(ctx.clone());

        assert_eq!(payload.compacted_through, 12);
        assert_eq!(payload.user_state, ctx.user_state);
        assert_eq!(payload.rules, ctx.rules);
        assert_eq!(payload.backstory, ctx.backstory);
        assert_eq!(payload.open_threads, ctx.open_threads);
        assert_eq!(payload.key_quotes, ctx.key_quotes);
        assert_eq!(payload.rolling_summary, ctx.rolling_summary);
        assert_eq!(payload.recent_detail, ctx.recent_detail);
        assert_eq!(payload.pins, ctx.pins);
    }

    #[test]
    fn into_context_restores_the_two_per_instance_fields_the_conversion_dropped() {
        let payload = sample_continuity();
        let companion_state = vec!["is protective of the lighthouse".to_string()];
        let recalled_facts = vec!["the lighthouse was built in 1890".to_string()];

        let ctx = payload
            .clone()
            .into_context(companion_state.clone(), recalled_facts.clone());

        assert_eq!(ctx.compacted_through, Some(payload.compacted_through));
        assert_eq!(ctx.user_state, payload.user_state);
        assert_eq!(ctx.companion_state, companion_state);
        assert_eq!(ctx.rules, payload.rules);
        assert_eq!(ctx.backstory, payload.backstory);
        assert_eq!(ctx.open_threads, payload.open_threads);
        assert_eq!(ctx.key_quotes, payload.key_quotes);
        assert_eq!(ctx.rolling_summary, payload.rolling_summary);
        assert_eq!(ctx.recent_detail, payload.recent_detail);
        assert_eq!(ctx.pins, payload.pins);
        assert_eq!(ctx.recalled_facts, recalled_facts);
    }

    #[test]
    fn context_to_payload_and_back_round_trips_every_shared_field() {
        let companion_state = vec!["is protective of the lighthouse".to_string()];
        let recalled_facts = vec!["the lighthouse was built in 1890".to_string()];
        let original = CompactionContext {
            compacted_through: Some(12),
            companion_state: companion_state.clone(),
            recalled_facts: recalled_facts.clone(),
            ..sample_continuity().into_context(Vec::new(), Vec::new())
        };

        let round_tripped =
            ContinuityPayload::from(original.clone()).into_context(companion_state, recalled_facts);

        assert_eq!(round_tripped, original);
    }

    #[test]
    fn generate_request_with_no_continuity_serialises_with_no_continuity_key() {
        let frame = ServerFrame::GenerateRequest {
            round_id: 1,
            transcript: vec![],
            continuity: None,
        };
        let json: serde_json::Value = serde_json::to_value(&frame).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "type": "generate_request",
                "round_id": 1,
                "transcript": []
            })
        );
    }

    #[test]
    fn token_round_trips_and_carries_its_round_id() {
        let frame = ClientFrame::Token {
            round_id: 9,
            text: "hel".to_string(),
        };
        let json = serde_json::to_string(&frame).unwrap();
        let back: ClientFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(back, frame);
        assert_eq!(frame.round_id(), Some(9));
    }

    #[test]
    fn reply_complete_round_trips_and_carries_its_round_id() {
        let frame = ClientFrame::ReplyComplete {
            round_id: 9,
            text: "hello there".to_string(),
        };
        let json = serde_json::to_string(&frame).unwrap();
        let back: ClientFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(back, frame);
        assert_eq!(frame.round_id(), Some(9));
    }
}
