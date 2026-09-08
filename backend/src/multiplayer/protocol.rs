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
//! Reserved, not sent or received by this issue (listed so a future issue's
//! variant addition is the only change needed, and so clippy's dead-code
//! lint has nothing to fire on until then):
//! - `ServerFrame::Message(database::Message)`: broadcast by #131 for every
//!   reply it persists.
//! - `ClientFrame::GenerateRequest { round_id: u64, transcript: Vec<database::Message> }`,
//!   `ClientFrame::Token { round_id: u64, text: String }`,
//!   `ClientFrame::ReplyComplete { round_id: u64, text: String }`,
//!   `ClientFrame::ReplyFailed { round_id: u64, reason: String }`: #131's
//!   remote-generation round protocol.

use serde::{Deserialize, Serialize};

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
}

impl ClientFrame {
    /// The round id this frame belongs to, or `None` if it is not
    /// round-scoped. Every variant in this issue returns `None` (`Join` is
    /// not round-scoped); #131 adds the `Token`/`ReplyComplete`/
    /// `ReplyFailed` arms when it adds those variants. This is the one
    /// accessor `RemoteBots::route_inbound` keys on.
    pub fn round_id(&self) -> Option<u64> {
        match self {
            ClientFrame::Join { .. } => None,
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
