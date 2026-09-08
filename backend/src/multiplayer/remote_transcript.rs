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

    /// Merges a host transcript snapshot into the mirror, e.g. the seed
    /// transcript a `Joined` frame carries. `snapshot` is expected
    /// oldest-first, the same order `Joined.transcript` and
    /// `Database::get_x_messages` (reversed) use.
    ///
    /// Deliberately a merge, not an overwrite: `Joined.transcript` is only
    /// ever the host's *last 50* messages (`host.rs::admit`), so a joiner
    /// reconnecting after already having mirrored more history than that
    /// would lose the older rows — and shrink `total_count` — if this
    /// replaced `self.messages` outright. Each message in `snapshot` goes
    /// through the same dedup-and-sorted-insert [`RemoteTranscript::push`]
    /// uses, so calling this on a fresh mirror (the common case: the first
    /// `Joined` of a run) is equivalent to a plain assignment.
    pub fn replace(&mut self, snapshot: Vec<Message>) {
        for message in snapshot {
            self.push(message);
        }
    }

    /// Inserts one message from a [`crate::multiplayer::protocol::ServerFrame::Message`]
    /// frame, or one row of a [`RemoteTranscript::replace`] snapshot, at its
    /// sorted position by `id`. A no-op if `message.id` is already present,
    /// so a reconnect that replays the tail of the host's transcript never
    /// duplicates a row this mirror already has. Inserting by position
    /// (rather than always appending) keeps the mirror correctly ordered
    /// even if `replace` ever merges a snapshot whose new ids interleave
    /// with an existing gap, not just the common case of new ids landing
    /// after the current maximum.
    pub fn push(&mut self, message: Message) {
        if let Err(pos) = self.messages.binary_search_by_key(&message.id, |m| m.id) {
            self.messages.insert(pos, message);
        }
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

    /// Applies a [`crate::multiplayer::protocol::ServerFrame::MessageEdited`]
    /// frame: replaces the row with `message.id`'s content in place, keeping
    /// its sorted position. A no-op if the id is not mirrored (e.g. the
    /// edited row is older than this mirror's history), since there is no
    /// row here to update.
    pub fn replace_message(&mut self, message: Message) {
        if let Ok(pos) = self.messages.binary_search_by_key(&message.id, |m| m.id) {
            self.messages[pos] = message;
        }
    }

    /// Applies a [`crate::multiplayer::protocol::ServerFrame::MessageRemoved`]
    /// frame: drops the row with this id. A no-op if it is not mirrored.
    pub fn remove(&mut self, id: i32) {
        if let Ok(pos) = self.messages.binary_search_by_key(&id, |m| m.id) {
            self.messages.remove(pos);
        }
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
    fn push_inserts_an_out_of_order_id_at_its_sorted_position() {
        let mut transcript = RemoteTranscript::new();
        transcript.replace(messages([1, 2, 5]));
        transcript.push(message(3));
        assert_eq!(
            transcript.snapshot(),
            vec![message(1), message(2), message(3), message(5)]
        );
    }

    #[test]
    fn replace_message_updates_the_row_in_place() {
        let mut transcript = RemoteTranscript::new();
        transcript.replace(messages(1..=3));

        let mut edited = message(2);
        edited.content = "edited content".to_string();
        transcript.replace_message(edited.clone());

        assert_eq!(transcript.snapshot(), vec![message(1), edited, message(3)]);
    }

    #[test]
    fn replace_message_is_a_no_op_for_an_id_not_in_the_mirror() {
        let mut transcript = RemoteTranscript::new();
        transcript.replace(messages(1..=3));

        transcript.replace_message(message(99));

        assert_eq!(transcript.snapshot(), messages(1..=3));
    }

    #[test]
    fn remove_drops_the_row_with_the_given_id() {
        let mut transcript = RemoteTranscript::new();
        transcript.replace(messages(1..=3));

        transcript.remove(2);

        assert_eq!(transcript.snapshot(), vec![message(1), message(3)]);
    }

    #[test]
    fn remove_is_a_no_op_for_an_id_not_in_the_mirror() {
        let mut transcript = RemoteTranscript::new();
        transcript.replace(messages(1..=3));

        transcript.remove(99);

        assert_eq!(transcript.snapshot(), messages(1..=3));
    }

    #[test]
    fn replace_merges_a_later_snapshot_without_losing_earlier_history() {
        let mut transcript = RemoteTranscript::new();
        // First `Joined`, early in the run: only 5 messages exist yet.
        transcript.replace(messages(1..=5));
        // The connection drops; more messages arrive on the host while this
        // joiner is disconnected. A reconnect's `Joined.transcript` is only
        // ever the host's last 50 (here: 4..=8), overlapping rather than
        // starting exactly where this mirror left off.
        transcript.replace(messages(4..=8));
        assert_eq!(transcript.snapshot(), messages(1..=8));
        let (_, total_count, _) = transcript.page(0, 50);
        assert_eq!(total_count, 8);
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
