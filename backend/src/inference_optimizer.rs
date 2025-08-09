use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use crate::database::Message;

/// Cache entry for frequently used prompts
#[derive(Debug, Clone)]
pub struct CachedPrompt {
    pub prompt_hash: String,
    pub base_prompt: String,
    pub timestamp: Instant,
    pub hit_count: usize,
    pub estimated_tokens: usize,
}

/// Batch inference request for multiple prompts
#[derive(Clone)]
pub struct BatchInferenceRequest {
    pub id: String,
    pub prompts: Vec<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

/// Response streaming chunk
#[derive(Debug, Clone, Serialize)]
pub struct StreamChunk {
    pub request_id: String,
    pub content: String,
    pub is_complete: bool,
    pub token_count: Option<usize>,
}

/// Inference optimization statistics
#[derive(Debug, Clone, Serialize)]
pub struct InferenceStats {
    pub total_requests: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub avg_response_time: Duration,
    pub batch_processed: usize,
    pub streaming_sessions: usize,
}

/// Main inference optimizer with caching and batching capabilities
pub struct InferenceOptimizer {
    /// Cache for frequently used prompt segments
    prompt_cache: Arc<RwLock<HashMap<String, CachedPrompt>>>,
    /// Batch processing queue
    batch_queue: Arc<Mutex<Vec<BatchInferenceRequest>>>,
    /// Active streaming sessions
    streaming_sessions: Arc<RwLock<HashMap<String, mpsc::UnboundedSender<StreamChunk>>>>,
    /// Performance statistics
    stats: Arc<RwLock<InferenceStats>>,
    /// Configuration
    cache_max_size: usize,
    cache_ttl: Duration,
    batch_size: usize,
    batch_timeout: Duration,
}

impl InferenceOptimizer {
    /// Create a new inference optimizer
    pub fn new() -> Self {
        Self {
            prompt_cache: Arc::new(RwLock::new(HashMap::new())),
            batch_queue: Arc::new(Mutex::new(Vec::new())),
            streaming_sessions: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(InferenceStats {
                total_requests: 0,
                cache_hits: 0,
                cache_misses: 0,
                avg_response_time: Duration::from_millis(0),
                batch_processed: 0,
                streaming_sessions: 0,
            })),
            cache_max_size: 1000,
            cache_ttl: Duration::from_secs(3600), // 1 hour
            batch_size: 4,
            batch_timeout: Duration::from_millis(100),
        }
    }

    /// Generate a hash for prompt caching
    pub fn hash_prompt(&self, prompt: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Check cache for existing prompt
    pub fn get_cached_prompt(&self, prompt: &str) -> Option<CachedPrompt> {
        let hash = self.hash_prompt(prompt);
        let cache = self.prompt_cache.read().unwrap();

        if let Some(cached) = cache.get(&hash) {
            // Check if cache entry is still valid
            if cached.timestamp.elapsed() < self.cache_ttl {
                let mut stats = self.stats.write().unwrap();
                stats.cache_hits += 1;

                // Clone and update hit count
                let mut updated_cache = cached.clone();
                updated_cache.hit_count += 1;
                drop(cache);

                // Update cache with new hit count
                let mut cache_write = self.prompt_cache.write().unwrap();
                cache_write.insert(hash, updated_cache.clone());

                return Some(updated_cache);
            }
        }

        let mut stats = self.stats.write().unwrap();
        stats.cache_misses += 1;
        None
    }

    /// Cache a prompt for future use
    pub fn cache_prompt(&self, prompt: &str, base_prompt: &str, estimated_tokens: usize) {
        let hash = self.hash_prompt(prompt);
        let cached = CachedPrompt {
            prompt_hash: hash.clone(),
            base_prompt: base_prompt.to_string(),
            timestamp: Instant::now(),
            hit_count: 1,
            estimated_tokens,
        };

        let mut cache = self.prompt_cache.write().unwrap();

        // Implement LRU eviction if cache is full
        if cache.len() >= self.cache_max_size {
            self.evict_lru_entries();
        }

        cache.insert(hash, cached);
    }

    /// Evict least recently used cache entries
    fn evict_lru_entries(&self) {
        let mut cache = self.prompt_cache.write().unwrap();
        // Find oldest entries and remove 25% of cache
        let mut entries: Vec<_> = cache
            .iter()
            .map(|(k, v)| (k.clone(), v.timestamp))
            .collect();
        entries.sort_by_key(|(_, timestamp)| *timestamp);

        let to_remove = cache.len() / 4;
        for (hash, _) in entries.into_iter().take(to_remove) {
            cache.remove(&hash);
        }
    }

    /// Optimize prompt construction with caching
    pub fn optimize_prompt_construction(
        &self,
        base_components: &[String],
        dynamic_content: &str,
        _messages: &[Message],
    ) -> (String, bool) {
        let base_prompt = base_components.join("");
        let _base_hash = self.hash_prompt(&base_prompt);

        // Check if base prompt is cached
        if let Some(cached) = self.get_cached_prompt(&base_prompt) {
            // Construct full prompt with dynamic content
            let full_prompt = format!("{}{}", cached.base_prompt, dynamic_content);
            return (full_prompt, true);
        }

        // Not cached, construct normally and cache for future use
        let estimated_tokens = self.estimate_tokens(&base_prompt);
        self.cache_prompt(&base_prompt, &base_prompt, estimated_tokens);

        let full_prompt = format!("{}{}", base_prompt, dynamic_content);
        (full_prompt, false)
    }

    /// Add request to batch processing queue
    pub async fn add_to_batch(&self, request: BatchInferenceRequest) -> Result<String, String> {
        let mut queue = self.batch_queue.lock().unwrap();
        let request_id = request.id.clone();
        queue.push(request);

        // Process batch if it reaches target size
        if queue.len() >= self.batch_size {
            drop(queue);
            self.process_batch().await?;
        }

        Ok(request_id)
    }

    /// Process batch of inference requests
    async fn process_batch(&self) -> Result<(), String> {
        let mut queue = self.batch_queue.lock().unwrap();
        if queue.is_empty() {
            return Ok(());
        }

        let batch: Vec<_> = queue.drain(..).collect();
        drop(queue);

        println!("Processing batch of {} requests", batch.len());

        // In a real implementation, this would use the LLM's batch inference capabilities
        // For now, we'll process them sequentially but track batch statistics
        for _request in batch {
            // Simulate batch processing - in reality this would be optimized
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let mut stats = self.stats.write().unwrap();
        stats.batch_processed += 1;

        Ok(())
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

    /// Estimate token count for a text (simplified implementation)
    pub fn estimate_tokens(&self, text: &str) -> usize {
        // Simple estimation: ~4 characters per token on average
        // In a real implementation, this would use proper tokenization
        (text.len() + 3) / 4
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

    /// Clear expired cache entries
    pub fn cleanup_cache(&self) {
        let mut cache = self.prompt_cache.write().unwrap();
        let now = Instant::now();

        cache.retain(|_, cached| now.duration_since(cached.timestamp) < self.cache_ttl);

        println!(
            "Cache cleanup completed. Entries remaining: {}",
            cache.len()
        );
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize, f64) {
        let cache = self.prompt_cache.read().unwrap();
        let stats = self.stats.read().unwrap();

        let cache_size = cache.len();
        let total_hits = stats.cache_hits;
        let total_requests = stats.cache_hits + stats.cache_misses;
        let hit_rate = if total_requests > 0 {
            stats.cache_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        (cache_size, total_hits, hit_rate)
    }
}

impl Default for InferenceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Global inference optimizer instance
lazy_static::lazy_static! {
    pub static ref INFERENCE_OPTIMIZER: InferenceOptimizer = InferenceOptimizer::new();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_hashing() {
        let optimizer = InferenceOptimizer::new();
        let prompt = "Hello, world!";
        let hash1 = optimizer.hash_prompt(prompt);
        let hash2 = optimizer.hash_prompt(prompt);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_cache_operations() {
        let optimizer = InferenceOptimizer::new();
        let prompt = "Test prompt";
        let base_prompt = "Base: Test prompt";

        // Should be cache miss initially
        assert!(optimizer.get_cached_prompt(prompt).is_none());

        // Cache the prompt
        optimizer.cache_prompt(prompt, base_prompt, 10);

        // Should be cache hit now
        assert!(optimizer.get_cached_prompt(prompt).is_some());
    }

    #[test]
    fn test_token_estimation() {
        let optimizer = InferenceOptimizer::new();
        let text = "This is a test";
        let tokens = optimizer.estimate_tokens(text);
        assert!(tokens > 0);
        assert!(tokens <= text.len()); // Should be reasonable estimate
    }
}
