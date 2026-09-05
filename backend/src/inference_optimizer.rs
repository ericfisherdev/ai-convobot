use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
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

    /// Start response streaming session
    pub fn start_streaming_session(
        &self,
        session_id: String,
    ) -> mpsc::UnboundedReceiver<StreamChunk> {
        let (tx, rx) = mpsc::unbounded_channel();

        let mut sessions = self.streaming_sessions.write().unwrap();
        sessions.insert(session_id.clone(), tx);

        let mut stats = self.stats.write().unwrap();
        stats.streaming_sessions += 1;

        rx
    }

    /// Stream response chunk to client
    pub fn stream_chunk(&self, session_id: &str, chunk: StreamChunk) -> Result<(), String> {
        let sessions = self.streaming_sessions.read().unwrap();

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
        let mut sessions = self.streaming_sessions.write().unwrap();
        sessions.remove(session_id);
    }

    /// Get current performance statistics
    pub fn get_stats(&self) -> InferenceStats {
        self.stats.read().unwrap().clone()
    }

    /// Update response time statistics
    pub fn record_response_time(&self, duration: Duration) {
        let mut stats = self.stats.write().unwrap();
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

lazy_static::lazy_static! {
    /// Global inference optimizer instance
    pub static ref INFERENCE_OPTIMIZER: InferenceOptimizer = InferenceOptimizer::new();
}
