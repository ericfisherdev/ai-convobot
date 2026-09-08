//! Multiplayer (multi-instance chat) support.
//!
//! `config.rs` owns the network-role config (#128): mode, host address,
//! participant id, and the derived validation rules. It has no `Database` or
//! HTTP dependency, so it is fully unit-testable on its own. `protocol.rs`
//! (#129) will live alongside it here.

pub mod config;
