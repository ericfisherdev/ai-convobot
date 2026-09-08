//! The joiner's mirror of the host's `messages` table.
//!
//! Pure, no I/O, no `Database` dependency, unit-tested on its own (like
//! `handshake.rs`). Held inside `JoinerShared` behind a lock; every method
//! here takes `&self`/`&mut self` and returns owned data, so a caller never
//! has to hold the lock across an `.await`.

use crate::database::Message;

/// The joiner's copy of the host's transcript, kept in the same order the
/// host's `id` column implies (oldest first).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RemoteTranscript {
    messages: Vec<Message>,
}

impl RemoteTranscript {
    pub fn new() -> Self {
        RemoteTranscript::default()
    }

    /// Replaces the whole mirror, e.g. with the seed transcript a `Joined`
    /// frame carries. `messages` is expected oldest-first, the same order
    /// `Joined.transcript` and `Database::get_x_messages` (reversed) use.
    pub fn replace(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    /// Appends one message from a [`crate::multiplayer::protocol::ServerFrame::Message`]
    /// frame. A no-op if `message.id` is already present, so a reconnect
    /// that replays the tail of the host's transcript (the last 50 rows in
    /// a fresh `Joined`) never duplicates a row this mirror already has.
    pub fn push(&mut self, message: Message) {
        if self.messages.iter().any(|m| m.id == message.id) {
            return;
        }
        self.messages.push(message);
    }

    /// The same windowing `GET /api/message` returns from
    /// `Database::get_x_messages` + `Database::get_total_message_count`: the
    /// `limit` messages ending `start_index` back from the newest, returned
    /// oldest-first, plus the total count and whether older messages remain.
    pub fn page(&self, start_index: usize, limit: usize) -> (Vec<Message>, usize, bool) {
        let total_count = self.messages.len();
        let end = total_count.saturating_sub(start_index);
        let start = end.saturating_sub(limit);
        let page = self.messages[start..end].to_vec();
        let has_more = start_index + page.len() < total_count;
        (page, total_count, has_more)
    }

    /// The full mirrored transcript, oldest first. Used by #153's
    /// generation handler to build a prompt from the current mirror rather
    /// than a `GenerateRequest`'s own `transcript` field.
    #[allow(dead_code)] // wired up by #153
    pub fn snapshot(&self) -> Vec<Message> {
        self.messages.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(id: i32) -> Message {
        Message {
            id,
            ai: id % 2 == 0,
            speaker_id: if id % 2 == 0 { "char" } else { "user" }.to_string(),
            content: format!("message {id}"),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    fn messages(ids: impl IntoIterator<Item = i32>) -> Vec<Message> {
        ids.into_iter().map(message).collect()
    }

    #[test]
    fn replace_sets_the_whole_mirror() {
        let mut transcript = RemoteTranscript::new();
        transcript.replace(messages(1..=3));
        assert_eq!(transcript.snapshot(), messages(1..=3));
    }

    #[test]
    fn push_appends_a_new_message() {
        let mut transcript = RemoteTranscript::new();
        transcript.replace(messages(1..=2));
        transcript.push(message(3));
        assert_eq!(transcript.snapshot(), messages(1..=3));
    }

    #[test]
    fn push_ignores_a_duplicate_id() {
        let mut transcript = RemoteTranscript::new();
        transcript.replace(messages(1..=3));
        transcript.push(message(2));
        assert_eq!(transcript.snapshot(), messages(1..=3));
    }

    #[test]
    fn page_from_the_start_returns_the_newest_window() {
        let mut transcript = RemoteTranscript::new();
        transcript.replace(messages(1..=10));

        let (page, total_count, has_more) = transcript.page(0, 5);
        assert_eq!(page, messages(6..=10));
        assert_eq!(total_count, 10);
        assert!(has_more);
    }

    #[test]
    fn page_offset_past_the_newest_window_returns_the_older_window() {
        let mut transcript = RemoteTranscript::new();
        transcript.replace(messages(1..=10));

        let (page, total_count, has_more) = transcript.page(5, 5);
        assert_eq!(page, messages(1..=5));
        assert_eq!(total_count, 10);
        assert!(!has_more);
    }

    #[test]
    fn page_start_index_past_the_end_is_empty() {
        let mut transcript = RemoteTranscript::new();
        transcript.replace(messages(1..=3));

        let (page, total_count, has_more) = transcript.page(10, 5);
        assert!(page.is_empty());
        assert_eq!(total_count, 3);
        assert!(!has_more);
    }

    #[test]
    fn page_limit_larger_than_the_transcript_returns_everything() {
        let mut transcript = RemoteTranscript::new();
        transcript.replace(messages(1..=3));

        let (page, total_count, has_more) = transcript.page(0, 50);
        assert_eq!(page, messages(1..=3));
        assert_eq!(total_count, 3);
        assert!(!has_more);
    }
}
