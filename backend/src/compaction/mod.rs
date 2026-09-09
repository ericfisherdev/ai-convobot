//! Conversation compaction (#171-#186): checkpoints that roll a run of
//! messages into a summary plus extracted facts, so old turns can drop out
//! of the prompt without the companion losing what happened.
//!
//! `types.rs` (#171) holds the enums and domain structs (`Checkpoint`,
//! `Fact`, `Pin`, ...) kept out of `database.rs` and out of `store.rs` so
//! the pure modules added by later issues can depend on the shapes without
//! pulling in rusqlite-backed code.
//!
//! `store.rs` (#171) is the persistence seam: the `CompactionStore` trait,
//! its production `SqliteCompactionStore` impl (backed by the same
//! `companion_database.db` every other `Database` associated fn uses), and
//! a `#[cfg(test)]` in-memory `RecordingStore` other modules' tests can use
//! too.

pub mod store;
pub mod types;
