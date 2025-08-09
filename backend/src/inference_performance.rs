use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
// Database import removed - using direct connection
use rusqlite::params;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub model_path: String,
    pub gpu_layers: i32,
    pub avg_tokens_per_second: f64,
    pub avg_time_to_first_token: f64,
    pub sample_count: u32,
    pub last_updated: String,
    pub confidence_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseEstimate {
    pub min_seconds: u32,
    pub expected_seconds: u32,
    pub max_seconds: u32,
    pub confidence: f64,
    pub factors: Vec<String>,
}

#[derive(Debug)]
pub struct InferenceSession {
    pub start_time: Instant,
    pub first_token_time: Option<Instant>,
    pub tokens_generated: u32,
    pub input_tokens: u32,
    pub model_config: ModelConfig,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_path: String,
    pub gpu_layers: i32,
    pub device_type: String,
}

pub struct InferencePerformanceTracker {
    current_sessions: HashMap<String, InferenceSession>,
    cached_metrics: HashMap<String, InferenceMetrics>,
}

impl InferencePerformanceTracker {
    pub fn new() -> Self {
        Self {
            current_sessions: HashMap::new(),
            cached_metrics: HashMap::new(),
        }
    }

    /// Start tracking a new inference session
    pub fn start_session(
        &mut self,
        session_id: String,
        model_config: ModelConfig,
        input_tokens: u32,
    ) {
        let session = InferenceSession {
            start_time: Instant::now(),
            first_token_time: None,
            tokens_generated: 0,
            input_tokens,
            model_config,
        };
        
        self.current_sessions.insert(session_id, session);
    }

    /// Record first token generation (important for perceived responsiveness)
    pub fn record_first_token(&mut self, session_id: &str) {
        if let Some(session) = self.current_sessions.get_mut(session_id) {
            if session.first_token_time.is_none() {
                session.first_token_time = Some(Instant::now());
            }
        }
    }

    /// Update token count during generation
    pub fn update_token_count(&mut self, session_id: &str, tokens_generated: u32) {
        if let Some(session) = self.current_sessions.get_mut(session_id) {
            session.tokens_generated = tokens_generated;
        }
    }

    /// Complete the session and store metrics
    pub fn complete_session(&mut self, session_id: &str) -> rusqlite::Result<()> {
        if let Some(session) = self.current_sessions.remove(session_id) {
            let total_time = session.start_time.elapsed();
            let time_to_first_token = session.first_token_time
                .map(|t| t.duration_since(session.start_time))
                .unwrap_or(Duration::from_secs(0));

            if session.tokens_generated > 0 {
                let tokens_per_second = session.tokens_generated as f64 / total_time.as_secs_f64();
                let ttft_seconds = time_to_first_token.as_secs_f64();

                // Store in database
                self.store_performance_metrics(
                    &session.model_config,
                    tokens_per_second,
                    ttft_seconds,
                    session.input_tokens,
                    session.tokens_generated,
                )?;

                // Update cached metrics
                let cache_key = format!("{}:{}", session.model_config.model_path, session.model_config.gpu_layers);
                self.update_cached_metrics(&cache_key, tokens_per_second, ttft_seconds)?;
            }
        }
        Ok(())
    }

    /// Get current performance estimate based on historical data
    pub fn estimate_response_time(
        &mut self,
        message: &str,
        model_config: &ModelConfig,
    ) -> ResponseEstimate {
        let input_tokens = self.estimate_input_tokens(message);
        let expected_output_tokens = self.estimate_output_tokens(message);
        
        let cache_key = format!("{}:{}", model_config.model_path, model_config.gpu_layers);
        
        // Get or load metrics
        let metrics = if let Some(cached) = self.cached_metrics.get(&cache_key) {
            cached.clone()
        } else {
            match self.load_metrics_from_db(model_config) {
                Ok(Some(metrics)) => {
                    self.cached_metrics.insert(cache_key.clone(), metrics.clone());
                    metrics
                }
                _ => {
                    // Use conservative defaults if no historical data
                    return self.conservative_estimate(message, expected_output_tokens);
                }
            }
        };

        // Calculate estimates based on historical performance
        let base_generation_time = expected_output_tokens as f64 / metrics.avg_tokens_per_second;
        let time_to_first_token = metrics.avg_time_to_first_token;
        
        // Apply context size penalty (larger contexts slow down inference)
        let context_penalty = self.calculate_context_penalty(input_tokens);
        let adjusted_generation_time = base_generation_time * context_penalty;
        
        // Add startup/warmup time if model not recently used
        let warmup_time = self.estimate_warmup_time(&metrics);
        
        let total_time = time_to_first_token + adjusted_generation_time + warmup_time;
        
        let mut factors = Vec::new();
        factors.push(format!("Expected {} output tokens", expected_output_tokens));
        factors.push(format!("Historical TPS: {:.1}", metrics.avg_tokens_per_second));
        if context_penalty > 1.1 {
            factors.push(format!("Context penalty: {:.1}x", context_penalty));
        }
        if warmup_time > 1.0 {
            factors.push(format!("Model warmup: {:.1}s", warmup_time));
        }

        ResponseEstimate {
            min_seconds: (total_time * 0.7) as u32,
            expected_seconds: total_time as u32,
            max_seconds: (total_time * 1.8) as u32,
            confidence: metrics.confidence_score,
            factors,
        }
    }

    /// Conservative estimate when no historical data is available
    fn conservative_estimate(&self, message: &str, expected_output_tokens: u32) -> ResponseEstimate {
        let _word_count = message.split_whitespace().count();
        let base_time = 15.0; // More realistic base time
        let token_time = expected_output_tokens as f64 * 0.5; // 0.5 seconds per token (very conservative)
        let complexity_bonus = self.analyze_complexity(message) * 10.0;
        
        let total_time = base_time + token_time + complexity_bonus as f64;
        
        ResponseEstimate {
            min_seconds: (total_time * 0.5) as u32,
            expected_seconds: total_time as u32,
            max_seconds: (total_time * 2.0) as u32,
            confidence: 0.3, // Low confidence without historical data
            factors: vec![
                format!("No historical data - using conservative estimate"),
                format!("Expected {} output tokens", expected_output_tokens),
                format!("Message complexity factor: {:.1}", complexity_bonus),
            ],
        }
    }

    /// Estimate input tokens from message length
    fn estimate_input_tokens(&self, message: &str) -> u32 {
        // Rough estimate: ~4 characters per token
        (message.len() as f32 / 4.0).ceil() as u32
    }

    /// Estimate output tokens based on message content and type
    fn estimate_output_tokens(&self, message: &str) -> u32 {
        let word_count = message.split_whitespace().count() as f32;
        let msg_lower = message.to_lowercase();
        
        // Base estimate: similar length to input
        let mut estimated_tokens = word_count * 1.2;
        
        // Adjust based on request type
        if msg_lower.contains("write") || msg_lower.contains("create") || msg_lower.contains("generate") {
            estimated_tokens *= 3.0; // Creative tasks produce longer responses
        } else if msg_lower.contains("explain") || msg_lower.contains("describe") {
            estimated_tokens *= 2.0; // Explanations are typically longer
        } else if msg_lower.contains("list") || msg_lower.contains("summarize") {
            estimated_tokens *= 1.5; // Lists and summaries
        } else if msg_lower.contains("?") && word_count < 10.0 {
            estimated_tokens *= 0.8; // Short questions often get concise answers
        }
        
        // Apply reasonable bounds
        let min_tokens = 20;
        let max_tokens = 1000; // Reasonable upper bound for most responses
        
        (estimated_tokens.max(min_tokens as f32).min(max_tokens as f32)) as u32
    }

    /// Analyze message complexity for estimation purposes
    fn analyze_complexity(&self, message: &str) -> f32 {
        let msg_lower = message.to_lowercase();
        let mut complexity = 0.0;
        
        // Technical/creative content complexity
        if msg_lower.contains("code") || msg_lower.contains("program") || msg_lower.contains("algorithm") {
            complexity += 2.0;
        }
        if msg_lower.contains("write") || msg_lower.contains("create") || msg_lower.contains("compose") {
            complexity += 1.5;
        }
        if msg_lower.contains("analyze") || msg_lower.contains("compare") || msg_lower.contains("evaluate") {
            complexity += 1.0;
        }
        if msg_lower.contains("explain") || msg_lower.contains("describe") || msg_lower.contains("how") {
            complexity += 0.5;
        }
        
        // Question mark adds slight complexity
        if msg_lower.contains("?") {
            complexity += 0.2;
        }
        
        complexity
    }

    /// Calculate penalty for larger context sizes
    fn calculate_context_penalty(&self, input_tokens: u32) -> f64 {
        match input_tokens {
            0..=100 => 1.0,     // No penalty for small contexts
            101..=500 => 1.1,   // Small penalty
            501..=1000 => 1.3,  // Moderate penalty
            1001..=2000 => 1.6, // Large penalty
            _ => 2.0,           // Maximum penalty for very large contexts
        }
    }

    /// Estimate warmup time if model hasn't been used recently
    fn estimate_warmup_time(&self, metrics: &InferenceMetrics) -> f64 {
        // If we have recent data, assume model is warm
        if metrics.sample_count > 0 && metrics.confidence_score > 0.5 {
            0.0
        } else {
            // Cold start penalty
            5.0
        }
    }

    /// Store performance metrics in database
    fn store_performance_metrics(
        &self,
        config: &ModelConfig,
        tokens_per_second: f64,
        time_to_first_token: f64,
        input_tokens: u32,
        output_tokens: u32,
    ) -> rusqlite::Result<()> {
        let con = rusqlite::Connection::open("companion_database.db")?;
        
        con.execute(
            "INSERT INTO inference_metrics (
                model_path, gpu_layers, device_type, tokens_per_second, 
                time_to_first_token, input_tokens, output_tokens, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, datetime('now'))",
            params![
                config.model_path,
                config.gpu_layers,
                config.device_type,
                tokens_per_second,
                time_to_first_token,
                input_tokens,
                output_tokens
            ],
        )?;
        
        Ok(())
    }

    /// Update cached metrics with new data point
    fn update_cached_metrics(
        &mut self,
        cache_key: &str,
        tokens_per_second: f64,
        time_to_first_token: f64,
    ) -> rusqlite::Result<()> {
        if let Some(metrics) = self.cached_metrics.get_mut(cache_key) {
            // Update rolling averages
            let weight = (metrics.sample_count as f64 / (metrics.sample_count as f64 + 1.0)).min(0.9);
            metrics.avg_tokens_per_second = metrics.avg_tokens_per_second * weight + tokens_per_second * (1.0 - weight);
            metrics.avg_time_to_first_token = metrics.avg_time_to_first_token * weight + time_to_first_token * (1.0 - weight);
            metrics.sample_count += 1;
            metrics.confidence_score = (metrics.sample_count as f64 / (metrics.sample_count as f64 + 10.0)).min(0.95);
            metrics.last_updated = chrono::Utc::now().to_string();
        }
        Ok(())
    }

    /// Load metrics from database
    fn load_metrics_from_db(&self, config: &ModelConfig) -> rusqlite::Result<Option<InferenceMetrics>> {
        let con = rusqlite::Connection::open("companion_database.db")?;
        
        // Get aggregated metrics for this configuration
        let mut stmt = con.prepare("
            SELECT 
                AVG(tokens_per_second) as avg_tps,
                AVG(time_to_first_token) as avg_ttft,
                COUNT(*) as sample_count,
                MAX(created_at) as last_updated
            FROM inference_metrics 
            WHERE model_path = ?1 AND gpu_layers = ?2
            AND created_at > datetime('now', '-30 days')
        ")?;

        let metrics = stmt.query_row(
            params![config.model_path, config.gpu_layers],
            |row| {
                let sample_count: u32 = row.get("sample_count")?;
                if sample_count == 0 {
                    return Ok(None);
                }
                
                let confidence = (sample_count as f64 / (sample_count as f64 + 10.0)).min(0.95);
                
                Ok(Some(InferenceMetrics {
                    model_path: config.model_path.clone(),
                    gpu_layers: config.gpu_layers,
                    avg_tokens_per_second: row.get("avg_tps")?,
                    avg_time_to_first_token: row.get("avg_ttft")?,
                    sample_count,
                    last_updated: row.get("last_updated")?,
                    confidence_score: confidence,
                }))
            },
        )?;

        Ok(metrics)
    }

    /// Get real-time progress estimate during generation
    pub fn get_progress_estimate(&self, session_id: &str) -> Option<(f64, u32)> {
        if let Some(session) = self.current_sessions.get(session_id) {
            let elapsed = session.start_time.elapsed().as_secs_f64();
            
            if session.tokens_generated > 0 {
                let current_tps = session.tokens_generated as f64 / elapsed;
                let estimated_total_tokens = self.estimate_output_tokens(""); // This could be improved
                let remaining_tokens = estimated_total_tokens.saturating_sub(session.tokens_generated);
                let estimated_remaining_time = remaining_tokens as f64 / current_tps.max(0.1);
                
                return Some((current_tps, estimated_remaining_time as u32));
            }
        }
        None
    }
}

// Global instance
lazy_static::lazy_static! {
    pub static ref INFERENCE_TRACKER: std::sync::Mutex<InferencePerformanceTracker> = 
        std::sync::Mutex::new(InferencePerformanceTracker::new());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_estimation() {
        let tracker = InferencePerformanceTracker::new();
        
        assert_eq!(tracker.estimate_input_tokens("hello world"), 3);
        assert_eq!(tracker.estimate_input_tokens("a longer message with more words"), 9);
    }

    #[test]
    fn test_output_estimation() {
        let tracker = InferencePerformanceTracker::new();
        
        let short_q = tracker.estimate_output_tokens("What time is it?");
        let creative = tracker.estimate_output_tokens("Write a story about dragons");
        let explain = tracker.estimate_output_tokens("Explain how computers work");
        
        assert!(creative > explain);
        assert!(explain > short_q);
    }

    #[test]
    fn test_complexity_analysis() {
        let tracker = InferencePerformanceTracker::new();
        
        assert!(tracker.analyze_complexity("write code for sorting") > tracker.analyze_complexity("what time is it"));
        assert!(tracker.analyze_complexity("explain algorithms") > tracker.analyze_complexity("hello"));
    }

    #[test]
    fn test_context_penalty() {
        let tracker = InferencePerformanceTracker::new();
        
        assert_eq!(tracker.calculate_context_penalty(50), 1.0);
        assert!(tracker.calculate_context_penalty(200) > 1.0);
        assert!(tracker.calculate_context_penalty(1500) > tracker.calculate_context_penalty(200));
    }
}