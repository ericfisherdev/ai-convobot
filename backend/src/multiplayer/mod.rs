//! Multiplayer (multi-instance chat) support.
//!
//! `config.rs` owns the network-role config (#128): mode, host address,
//! participant id, and the derived validation rules. It has no `Database` or
//! HTTP dependency, so it is fully unit-testable on its own.
//!
//! `protocol.rs` (#129) is the wire frame shapes shared by host and joiner.
//! `handshake.rs` is the HMAC join-proof helper both sides call. `avatar.rs`
//! validates and stores transferred avatars; `join_throttle.rs` is the
//! per-address failed-join rate limit. `remote_bots.rs` and `host.rs` own
//! the socket handler and connection lifecycle: `host::multiplayer_ws` is
//! the `/api/multiplayer/ws` handler `main.rs` registers.
//!
//! `joiner.rs` (#130) is the other side of that socket: it connects out to
//! a host, drives the handshake, and mirrors the transcript into
//! `remote_transcript.rs`'s pure `RemoteTranscript`, reconnecting with
//! `backoff.rs`'s `ReconnectBackoff` when the connection drops. Its
//! `GenerateRequestHandler` trait is the seam a `GenerateRequest` frame is
//! answered through; `remote_generation.rs` (#153) is the implementation
//! `main.rs` wires in.
//!
//! `remote_generation.rs` (#153) is a joiner's own reply generation:
//! `LocalModelGeneration` runs the joiner's own model (its own card,
//! config, dialogue tuning and long-term memory) on the transcript a
//! `GenerateRequest` carried, streaming `Token`s then a `ReplyComplete`
//! back over the socket, and scores the joiner's own attitude toward the
//! user from the turn it just answered.
//!
//! `round.rs` (#131) is the round orchestrator: `run_round` turns one user
//! message into a sequence of speaker replies, all under one held turn slot.
//! Its `RemoteGenerator` trait is the seam a joiner's reply is generated
//! through, on the *host* side; #154 implements it over `remote_bots.rs`.
//!
//! `routing.rs` (#132) is where `round.rs`'s speaker order actually comes
//! from: `plan_round` turns an `@mention`d user message into a speaker
//! order (falling back to `round.rs`'s old "host then every joiner" order
//! when nothing is mentioned), and `schedule_follow_ups` lets a bot's own
//! `@mention` of another bot queue it a follow-up turn, bounded by
//! `RoutingPolicy::max_followup_depth` so a chain can never run forever.

pub mod avatar;
pub mod backoff;
pub mod config;
pub mod handshake;
pub mod host;
pub mod join_throttle;
pub mod joiner;
pub mod protocol;
pub mod remote_bots;
pub mod remote_generation;
pub mod remote_transcript;
pub mod round;
pub mod routing;
