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
//! `backoff.rs`'s `ReconnectBackoff` when the connection drops.

pub mod avatar;
pub mod backoff;
pub mod config;
pub mod handshake;
pub mod host;
pub mod join_throttle;
pub mod joiner;
pub mod protocol;
pub mod remote_bots;
pub mod remote_transcript;
