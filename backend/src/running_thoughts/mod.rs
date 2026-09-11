//! Running thoughts (#214): a companion-authored, first-person memory note
//! written per exchange, distinct from every other place a note about a
//! conversation can live.
//!
//! `compaction_facts` is the wrong home: those rows carry
//! `compaction_id NOT NULL` and cascade-delete with the checkpoint that
//! produced them, so a fact's lifetime is tied to a range of messages that
//! can later go stale or be retired. A running thought is standalone and
//! durable — it survives message deletion (deleting or editing the messages
//! a thought is *about* does not retroactively invalidate it, since the
//! thought describes what the companion took away from them, not the
//! messages themselves) and has no checkpoint of its own to cascade with.
//! `pinned_messages` pins an existing message; a thought is new text the
//! companion wrote, not a flag on something the user or companion already
//! said. `third_party_individuals` covers people other than the two
//! speakers in a one-on-one chat; a running thought is always authored by
//! (and, in the multiplayer case, potentially about) a participant in the
//! conversation itself, never a third party.
//!
//! This module (#215) adds the table and the store trait only: no
//! generation and no UI, so the rest of the epic (#216 generation, #217
//! routes/DTOs, #218 rendering, #219 checkpoint-overlap handling, #220
//! per-speaker isolation) has something to write against.
//!
//! `types.rs`: the domain structs (`RunningThought`, `NewRunningThought`),
//! kept free of anything beyond `FromSql`/`ToSql`, mirroring
//! `compaction::types`.
//!
//! `store.rs`: the DDL, `_on` helpers, the `RunningThoughtStore` trait, its
//! production `SqliteRunningThoughtStore` impl, and a `#[cfg(test)]`
//! in-memory `RecordingStore`, mirroring `compaction::store`.
#![allow(dead_code)]

pub mod store;
pub mod types;
