use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, PoisonError, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;

use crate::attitude_formatter::AttitudeDelta;
use crate::database::CompanionAttitude;

/// Post-turn attitude state, carried by the stream's attitude chunk.
#[derive(Debug, Clone, Serialize)]
pub struct AttitudeStreamUpdate {
    /// The companion's attitude toward the user after the turn was scored.
    pub attitude: CompanionAttitude,
    /// Natural language rendering of `attitude`, with `{{companion}}` and
    /// `{{user}}` placeholders the client substitutes names into.
    pub summary: String,
    /// Only the dimensions this turn actually moved.
    pub deltas: Vec<AttitudeDelta>,
}

/// One Server-Sent Event on `/api/prompt/stream`.
///
/// Three kinds travel over the same struct:
/// - token chunks: `is_complete: false`, `content` holds the next token;
/// - the attitude chunk: `is_complete: false`, empty `content`, `attitude` set,
///   sent once after generation when the turn moved any dimension;
/// - the final chunk: `is_complete: true`, `content` holds the sanitized reply,
///   or `error` is set when generation failed.
///
/// `attitude` and `error` are omitted when `None`, so token and final chunks
/// keep the shape older clients expect.
#[derive(Debug, Clone, Serialize)]
pub struct StreamChunk {
    pub request_id: String,
    pub content: String,
    pub is_complete: bool,
    pub token_count: Option<usize>,
    /// Set on the final chunk when generation failed, so a client can tell a
    /// failure apart from a normal completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Set only on the attitude chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attitude: Option<AttitudeStreamUpdate>,
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
            StreamChunk {
                request_id: self.session_id.clone(),
                content: String::new(),
                is_complete: true,
                token_count: None,
                error: Some(message.to_string()),
                attitude: None,
            },
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
        StreamChunk {
            request_id: session_id.to_string(),
            content: "reply".to_string(),
            is_complete: true,
            token_count: Some(1),
            error: None,
            attitude: None,
        }
    }
}
