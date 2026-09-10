use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, PoisonError, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;

use crate::attitude_formatter::AttitudeDelta;
use crate::database::CompanionAttitude;
use crate::participants::ParticipantId;

/// Post-turn attitude state, carried by the stream's attitude chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttitudeStreamUpdate {
    /// The companion's attitude toward the user after the turn was scored.
    pub attitude: CompanionAttitude,
    /// Natural language rendering of `attitude`, with `{{companion}}` and
    /// `{{user}}` placeholders the client substitutes names into.
    pub summary: String,
    /// Only the dimensions this turn actually moved.
    pub deltas: Vec<AttitudeDelta>,
}

/// What kind of [`StreamChunk`] this is, so a client can switch on it
/// instead of inferring meaning from which optional fields are set.
///
/// `ReplyStarted` opens a new bubble for `speaker_id`; `Token` appends
/// `content` to the current speaker's bubble (or, when `attitude` is set
/// instead, carries the attitude update and no content); `ReplyComplete`
/// replaces the current bubble's content with the sanitized `content` and
/// carries `message_id`; `RoundComplete` and `Error` are the two terminal
/// events (`is_complete: true`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamEvent {
    ReplyStarted,
    Token,
    ReplyComplete,
    RoundComplete,
    Error,
}

/// One Server-Sent Event on `/api/prompt/stream`, one per speaker action in
/// the round.
///
/// `event` says which of five kinds this is; `speaker_id` is the
/// [`ParticipantId`] the chunk is about (empty on the attitude chunk and on
/// `round_complete`/`error`, which are round-wide rather than per-speaker).
/// A full round is `reply_started`, then zero or more `token`s, then
/// `reply_complete`, repeated once per speaker (a skipped speaker's notice
/// arrives as a bare `reply_complete` for `speaker_id: "system"`, with no
/// preceding `reply_started`); an optional attitude chunk (a `token`-event
/// chunk with empty `content` and `attitude` set, unchanged shape from
/// before this struct grew `event`) follows the last `reply_complete`; an
/// optional compaction-draft-ready chunk (same shape, `compaction_draft_id`
/// set instead, #179) follows that; then `round_complete` ends the stream.
/// `is_complete` is `true` only on `round_complete` and `error`, so a
/// client that only tracks that field still terminates correctly.
///
/// `message_id`, `error`, `attitude` and `compaction_draft_id` are omitted
/// when `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub request_id: String,
    pub event: StreamEvent,
    pub content: String,
    pub is_complete: bool,
    pub token_count: Option<usize>,
    pub speaker_id: String,
    /// Set on `reply_complete`: the persisted message's row id.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub message_id: Option<i32>,
    /// Set on the terminal `error` chunk, so a client can tell a failure
    /// apart from a normal completion.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub error: Option<String>,
    /// Set only on the attitude chunk.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub attitude: Option<AttitudeStreamUpdate>,
    /// Set only on the compaction-draft-ready chunk (#179): the id of the
    /// checkpoint draft `multiplayer::round::run_round`'s compaction hook
    /// just queued, so the client can start polling
    /// `GET /api/compaction/{id}` without waiting for a page refresh.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub compaction_draft_id: Option<i64>,
}

impl StreamChunk {
    /// Opens a bubble for `speaker` about to generate.
    pub fn reply_started(request_id: String, speaker: &ParticipantId) -> Self {
        StreamChunk {
            request_id,
            event: StreamEvent::ReplyStarted,
            content: String::new(),
            is_complete: false,
            token_count: None,
            speaker_id: speaker.as_str().to_string(),
            message_id: None,
            error: None,
            attitude: None,
            compaction_draft_id: None,
        }
    }

    /// One token from `speaker`, with the stream's running token count.
    pub fn token(request_id: String, speaker: &ParticipantId, text: &str, count: usize) -> Self {
        StreamChunk {
            request_id,
            event: StreamEvent::Token,
            content: text.to_string(),
            is_complete: false,
            token_count: Some(count),
            speaker_id: speaker.as_str().to_string(),
            message_id: None,
            error: None,
            attitude: None,
            compaction_draft_id: None,
        }
    }

    /// `speaker`'s finished, persisted reply.
    pub fn reply_complete(
        request_id: String,
        speaker: &ParticipantId,
        text: &str,
        message_id: i32,
        count: usize,
    ) -> Self {
        StreamChunk {
            request_id,
            event: StreamEvent::ReplyComplete,
            content: text.to_string(),
            is_complete: false,
            token_count: Some(count),
            speaker_id: speaker.as_str().to_string(),
            message_id: Some(message_id),
            error: None,
            attitude: None,
            compaction_draft_id: None,
        }
    }

    /// The attitude chunk: a `token`-event chunk with empty content and
    /// `attitude` set, kept exactly the shape it had before `event` existed.
    pub fn attitude(request_id: String, update: AttitudeStreamUpdate, count: usize) -> Self {
        StreamChunk {
            request_id,
            event: StreamEvent::Token,
            content: String::new(),
            is_complete: false,
            token_count: Some(count),
            speaker_id: String::new(),
            message_id: None,
            error: None,
            attitude: Some(update),
            compaction_draft_id: None,
        }
    }

    /// The compaction-draft-ready chunk (#179): a `token`-event chunk with
    /// empty content and `compaction_draft_id` set, sent once between the
    /// last `reply_complete`/the attitude chunk and `round_complete` when
    /// this round's compaction hook queued a draft.
    pub fn compaction_draft(request_id: String, draft_id: i64, count: usize) -> Self {
        StreamChunk {
            request_id,
            event: StreamEvent::Token,
            content: String::new(),
            is_complete: false,
            token_count: Some(count),
            speaker_id: String::new(),
            message_id: None,
            error: None,
            attitude: None,
            compaction_draft_id: Some(draft_id),
        }
    }

    /// The terminal chunk on a successful round.
    pub fn round_complete(request_id: String, count: Option<usize>) -> Self {
        StreamChunk {
            request_id,
            event: StreamEvent::RoundComplete,
            content: String::new(),
            is_complete: true,
            token_count: count,
            speaker_id: String::new(),
            message_id: None,
            error: None,
            attitude: None,
            compaction_draft_id: None,
        }
    }

    /// The terminal chunk on a failed round, including `StreamSession::Drop`'s
    /// own terminal chunk when a caller never reaches `finish`.
    pub fn error(request_id: String, message: String, count: Option<usize>) -> Self {
        StreamChunk {
            request_id,
            event: StreamEvent::Error,
            content: String::new(),
            is_complete: true,
            token_count: count,
            speaker_id: String::new(),
            message_id: None,
            error: Some(message),
            attitude: None,
            compaction_draft_id: None,
        }
    }
}

/// Inference optimization statistics
#[derive(Debug, Clone, Serialize)]
pub struct InferenceStats {
    pub total_requests: usize,
    pub avg_response_time: Duration,
    pub batch_processed: usize,
    pub streaming_sessions: usize,
}

/// Main inference optimizer: tracks streaming sessions and response-time stats
pub struct InferenceOptimizer {
    /// Active streaming sessions
    streaming_sessions: Arc<RwLock<HashMap<String, mpsc::UnboundedSender<StreamChunk>>>>,
    /// Performance statistics
    stats: Arc<RwLock<InferenceStats>>,
}

impl InferenceOptimizer {
    /// Create a new inference optimizer
    pub fn new() -> Self {
        Self {
            streaming_sessions: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(InferenceStats {
                total_requests: 0,
                avg_response_time: Duration::from_millis(0),
                batch_processed: 0,
                streaming_sessions: 0,
            })),
        }
    }

    /// Start response streaming session.
    ///
    /// Returns a [`StreamSession`] guard alongside the receiver: the guard is
    /// the only way to end the session, and it ends it on drop (with a
    /// terminal error chunk) if the caller never reaches [`StreamSession::finish`],
    /// including when the caller's thread panics.
    pub fn start_streaming_session(
        &'static self,
        session_id: String,
    ) -> (StreamSession, mpsc::UnboundedReceiver<StreamChunk>) {
        let (tx, rx) = mpsc::unbounded_channel();

        let mut sessions = self
            .streaming_sessions
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        sessions.insert(session_id.clone(), tx);

        let mut stats = self.stats.write().unwrap_or_else(PoisonError::into_inner);
        stats.streaming_sessions += 1;

        (
            StreamSession {
                optimizer: self,
                session_id,
                finished: false,
            },
            rx,
        )
    }

    /// Stream response chunk to client
    pub fn stream_chunk(&self, session_id: &str, chunk: StreamChunk) -> Result<(), String> {
        let sessions = self
            .streaming_sessions
            .read()
            .unwrap_or_else(PoisonError::into_inner);

        if let Some(tx) = sessions.get(session_id) {
            tx.send(chunk)
                .map_err(|e| format!("Failed to stream chunk: {}", e))?;
            Ok(())
        } else {
            Err("Session not found".to_string())
        }
    }

    /// End streaming session
    pub fn end_streaming_session(&self, session_id: &str) {
        let mut sessions = self
            .streaming_sessions
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        sessions.remove(session_id);
    }

    /// Get current performance statistics
    pub fn get_stats(&self) -> InferenceStats {
        self.stats
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Update response time statistics
    pub fn record_response_time(&self, duration: Duration) {
        let mut stats = self.stats.write().unwrap_or_else(PoisonError::into_inner);
        stats.total_requests += 1;

        // Calculate running average
        let total_requests = stats.total_requests as u64;
        let current_avg_nanos = stats.avg_response_time.as_nanos() as u64;
        let new_duration_nanos = duration.as_nanos() as u64;

        let new_avg_nanos =
            ((current_avg_nanos * (total_requests - 1)) + new_duration_nanos) / total_requests;
        stats.avg_response_time = Duration::from_nanos(new_avg_nanos);
    }
}

impl Default for InferenceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Owns a streaming session's lifetime.
///
/// `InferenceOptimizer::streaming_sessions` is a `HashMap` entry, not owned
/// by the worker thread, so nothing closed it if the thread that started the
/// session exited without saying so — the `mpsc` receiver's `recv()` would
/// then never return `None`, leaving the SSE response (and the client
/// waiting on it) hanging forever. `StreamSession` closes that gap: call
/// [`StreamSession::finish`] on every normal exit path, and `Drop` ends the
/// session with a terminal error chunk on every other path (an early
/// `return`, or the holder's thread panicking).
///
/// `Drop` runs during unwinding, so it must never itself panic — that would
/// abort the process rather than merely fail the request.
pub struct StreamSession {
    optimizer: &'static InferenceOptimizer,
    session_id: String,
    finished: bool,
}

impl StreamSession {
    /// The session id, used as `StreamChunk::request_id`.
    pub fn id(&self) -> &str {
        &self.session_id
    }

    /// Sends a non-terminal chunk (a token or the attitude update).
    pub fn send(&self, chunk: StreamChunk) -> Result<(), String> {
        self.optimizer.stream_chunk(&self.session_id, chunk)
    }

    /// Sends the final chunk and ends the session.
    ///
    /// Consumes `self` so a finished session cannot be sent through or
    /// finished again; `Drop` sees `finished == true` and does nothing.
    pub fn finish(mut self, final_chunk: StreamChunk) {
        let _ = self.optimizer.stream_chunk(&self.session_id, final_chunk);
        self.finished = true;
        self.optimizer.end_streaming_session(&self.session_id);
    }
}

impl Drop for StreamSession {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        // The default panic hook already printed the payload and location to
        // stderr by the time `Drop` runs during unwinding, so the client only
        // needs to know which of the two happened.
        let message = if std::thread::panicking() {
            "generation panicked, check logs for more information"
        } else {
            "generation ended without a reply, check logs for more information"
        };
        // Ignored: the client may already have hung up, and this is the
        // failure path regardless of whether the send lands.
        let _ = self.optimizer.stream_chunk(
            &self.session_id,
            StreamChunk::error(self.session_id.clone(), message.to_string(), None),
        );
        self.optimizer.end_streaming_session(&self.session_id);
    }
}

lazy_static::lazy_static! {
    /// Global inference optimizer instance
    pub static ref INFERENCE_OPTIMIZER: InferenceOptimizer = InferenceOptimizer::new();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panicking_worker_ends_the_session_with_an_error_chunk() {
        let session_id = format!("test-{}", uuid::Uuid::new_v4());
        let (stream, mut rx) = INFERENCE_OPTIMIZER.start_streaming_session(session_id.clone());

        let handle = std::thread::spawn(move || {
            let _stream = stream;
            panic!("simulated generation panic");
        });
        assert!(handle.join().is_err(), "worker thread should have panicked");

        let chunk = rx
            .try_recv()
            .expect("dropping the guard on panic should send a terminal chunk");
        assert!(chunk.is_complete);
        assert!(chunk.error.is_some());

        assert!(matches!(
            rx.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
        assert_eq!(
            INFERENCE_OPTIMIZER.stream_chunk(&session_id, dummy_chunk(&session_id)),
            Err("Session not found".to_string())
        );
    }

    #[test]
    fn finish_sends_exactly_the_final_chunk_and_closes_the_channel() {
        let session_id = format!("test-{}", uuid::Uuid::new_v4());
        let (stream, mut rx) = INFERENCE_OPTIMIZER.start_streaming_session(session_id.clone());

        stream.finish(dummy_chunk(&session_id));

        let chunk = rx.try_recv().expect("finish should send the final chunk");
        assert!(chunk.is_complete);
        assert!(chunk.error.is_none());

        assert!(matches!(
            rx.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn dropping_an_unfinished_session_without_a_panic_still_closes_it() {
        let session_id = format!("test-{}", uuid::Uuid::new_v4());
        let (stream, mut rx) = INFERENCE_OPTIMIZER.start_streaming_session(session_id.clone());

        drop(stream);

        let chunk = rx
            .try_recv()
            .expect("dropping without finishing should send a terminal chunk");
        assert!(chunk.is_complete);
        assert!(chunk.error.is_some());

        assert!(matches!(
            rx.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
    }

    fn dummy_chunk(session_id: &str) -> StreamChunk {
        StreamChunk::round_complete(session_id.to_string(), Some(1))
    }
}
