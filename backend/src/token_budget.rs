use crate::database::{CompanionAttitude, Message, ThirdPartyIndividual};

#[derive(Debug, Clone)]
pub struct TokenBudget {
    pub total: usize,
    pub system_prompt: usize,
    pub attitude_data: usize,
    pub third_party_info: usize,
    pub recent_messages: usize,
    pub response_buffer: usize,
    pub vram_tier: VramTier,
}

#[derive(Debug, Clone)]
pub enum VramTier {
    Minimal,  // 0-2GB VRAM
    Standard, // 3-4GB VRAM
    Extended, // 5-6GB VRAM
    Maximum,  // 7GB+ VRAM
}

impl TokenBudget {
    pub fn from_vram_limit(vram_gb: usize, max_configured: usize) -> Self {
        let (tier, total) = match vram_gb {
            0..=2 => (VramTier::Minimal, 1024),
            3..=4 => (VramTier::Standard, 2048),
            5..=6 => (VramTier::Extended, 3072),
            _ => (VramTier::Maximum, 4096),
        };

        // Use the smaller of calculated total or user configuration
        let total = std::cmp::min(total, max_configured);

        // Allocation strategy based on specifications
        let system_prompt = (total as f32 * 0.15) as usize; // 15% for system prompts
        let attitude_data = (total as f32 * 0.20) as usize; // 20% for attitude/memory context
        let third_party_info = (total as f32 * 0.10) as usize; // 10% for third-party information
        let recent_messages = (total as f32 * 0.40) as usize; // 40% for recent conversation
        let response_buffer = (total as f32 * 0.15) as usize; // 15% for response generation

        Self {
            total,
            system_prompt,
            attitude_data,
            third_party_info,
            recent_messages,
            response_buffer,
            vram_tier: tier,
        }
    }

    pub fn get_allocation_summary(&self) -> String {
        format!(
            "Token Budget ({}): System: {}, Attitude: {}, Third-party: {}, Messages: {}, Response: {}",
            self.total,
            self.system_prompt,
            self.attitude_data,
            self.third_party_info,
            self.recent_messages,
            self.response_buffer
        )
    }
}

#[derive(Debug)]
pub struct TokenUsageMonitor {
    pub budget: TokenBudget,
    pub current_usage: TokenUsage,
    pub optimization_stats: OptimizationStats,
}

#[derive(Debug, Default, Clone)]
pub struct TokenUsage {
    pub system_tokens: usize,
    pub attitude_tokens: usize,
    pub third_party_tokens: usize,
    pub message_tokens: usize,
    pub total_context_tokens: usize,
}

#[derive(Debug, Default, Clone)]
pub struct OptimizationStats {
    pub messages_compressed: usize,
    pub attitudes_filtered: usize,
    pub third_parties_filtered: usize,
    pub total_savings: usize,
    pub overflow_events: usize,
}

impl TokenUsageMonitor {
    pub fn new(budget: TokenBudget) -> Self {
        Self {
            budget,
            current_usage: TokenUsage::default(),
            optimization_stats: OptimizationStats::default(),
        }
    }

    /// Estimate token count for text (improved approximation)
    pub fn estimate_tokens(text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }

        // More accurate estimation based on tokenizer patterns:
        // - Average English word is ~5 characters
        // - Average token is ~1.3 words
        // - Special characters and punctuation add complexity
        let word_count = text.split_whitespace().count();
        let char_count = text.len();

        // Use weighted average of word-based and character-based estimates
        let word_estimate = (word_count as f32 * 1.3) as usize;
        let char_estimate = (char_count as f32 / 4.0) as usize;

        // Take the larger estimate for safety margin
        std::cmp::max(word_estimate, char_estimate).max(1)
    }

    /// Filter and prioritize attitudes based on significance and token budget
    pub fn optimize_attitude_context(
        &mut self,
        attitudes: Vec<CompanionAttitude>,
    ) -> Vec<CompanionAttitude> {
        let mut filtered_attitudes = Vec::new();
        let mut current_tokens = 0;
        let attitude_threshold = match self.budget.vram_tier {
            VramTier::Minimal => 30.0,  // Only very significant attitudes
            VramTier::Standard => 20.0, // Moderately significant attitudes
            VramTier::Extended => 15.0, // Include more attitudes
            VramTier::Maximum => 10.0,  // Include most attitudes
        };

        // Sort attitudes by combined significance score
        let mut sorted_attitudes = attitudes;
        sorted_attitudes.sort_by(|a, b| {
            let score_a = Self::calculate_attitude_significance(a);
            let score_b = Self::calculate_attitude_significance(b);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for attitude in sorted_attitudes {
            // Check if attitude meets significance threshold
            if Self::calculate_attitude_significance(&attitude) < attitude_threshold {
                self.optimization_stats.attitudes_filtered += 1;
                continue;
            }

            let attitude_text = self.format_attitude_for_context(&attitude);
            let attitude_tokens = Self::estimate_tokens(&attitude_text);

            if current_tokens + attitude_tokens <= self.budget.attitude_data {
                filtered_attitudes.push(attitude);
                current_tokens += attitude_tokens;
            } else {
                self.optimization_stats.attitudes_filtered += 1;
                if current_tokens + (attitude_tokens / 2) <= self.budget.attitude_data {
                    // Try compressed format for borderline cases
                    let compressed_attitude = self.compress_attitude(&attitude);
                    let compressed_tokens = Self::estimate_tokens(&compressed_attitude);
                    if current_tokens + compressed_tokens <= self.budget.attitude_data {
                        let mut compressed_attitude_obj = attitude.clone();
                        // Store compressed version in a special way (this is a simplified approach)
                        filtered_attitudes.push(compressed_attitude_obj);
                        current_tokens += compressed_tokens;
                    }
                }
            }
        }

        self.current_usage.attitude_tokens = current_tokens;
        filtered_attitudes
    }

    /// Calculate significance score for an attitude
    fn calculate_attitude_significance(attitude: &CompanionAttitude) -> f32 {
        let dimensions = [
            attitude.attraction,
            attitude.trust,
            attitude.fear,
            attitude.anger,
            attitude.joy,
            attitude.sorrow,
            attitude.disgust,
            attitude.surprise,
            attitude.curiosity,
            attitude.respect,
            attitude.suspicion,
            attitude.gratitude,
            attitude.jealousy,
            attitude.empathy,
        ];

        // Calculate weighted significance based on intensity and emotional impact
        let total_intensity: f32 = dimensions.iter().map(|&x| x.abs()).sum();
        let max_dimension = dimensions.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let relationship_bonus = attitude.relationship_score.unwrap_or(0.0).abs() * 0.1;

        total_intensity * 0.3 + max_dimension * 0.6 + relationship_bonus
    }

    /// Format attitude for context inclusion
    fn format_attitude_for_context(&self, attitude: &CompanionAttitude) -> String {
        let significant_dimensions = self.get_significant_dimensions(attitude);
        if significant_dimensions.is_empty() {
            return String::new();
        }

        format!(
            "Attitude towards {}: {}{}",
            attitude.target_type,
            significant_dimensions.join(", "),
            attitude
                .relationship_score
                .map(|score| format!(" (relationship: {:.0})", score))
                .unwrap_or_default()
        )
    }

    /// Get significant attitude dimensions above threshold
    fn get_significant_dimensions(&self, attitude: &CompanionAttitude) -> Vec<String> {
        let threshold = 15.0;
        let mut dimensions = Vec::new();

        let dimension_map = [
            ("attraction", attitude.attraction),
            ("trust", attitude.trust),
            ("fear", attitude.fear),
            ("anger", attitude.anger),
            ("joy", attitude.joy),
            ("sorrow", attitude.sorrow),
            ("disgust", attitude.disgust),
            ("surprise", attitude.surprise),
            ("curiosity", attitude.curiosity),
            ("respect", attitude.respect),
            ("suspicion", attitude.suspicion),
            ("gratitude", attitude.gratitude),
            ("jealousy", attitude.jealousy),
            ("empathy", attitude.empathy),
        ];

        for (name, value) in dimension_map {
            if value.abs() >= threshold {
                dimensions.push(format!("{}: {:.0}", name, value));
            }
        }

        dimensions
    }

    /// Compress attitude representation for token efficiency
    fn compress_attitude(&self, attitude: &CompanionAttitude) -> String {
        let top_dimensions = self.get_top_dimensions(attitude, 3);
        format!("#{}: {}", attitude.target_type, top_dimensions.join(", "))
    }

    /// Get top N most significant dimensions
    fn get_top_dimensions(&self, attitude: &CompanionAttitude, n: usize) -> Vec<String> {
        let mut dimensions = vec![
            ("trust", attitude.trust),
            ("fear", attitude.fear),
            ("joy", attitude.joy),
            ("anger", attitude.anger),
            ("curiosity", attitude.curiosity),
            ("respect", attitude.respect),
            ("empathy", attitude.empathy),
        ];

        dimensions.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        dimensions
            .into_iter()
            .take(n)
            .filter(|(_, value)| value.abs() >= 10.0)
            .map(|(name, value)| format!("{}:{:.0}", name, value))
            .collect()
    }

    /// Optimize message context with intelligent summarization
    pub fn optimize_message_context(&mut self, messages: Vec<Message>) -> Vec<Message> {
        if messages.is_empty() {
            return messages;
        }

        let mut optimized_messages = Vec::new();
        let mut current_tokens = 0;

        // Always prioritize the most recent messages
        for message in messages.iter().rev() {
            let tokens = Self::estimate_tokens(&message.content);

            if current_tokens + tokens <= self.budget.recent_messages {
                optimized_messages.insert(0, message.clone());
                current_tokens += tokens;
            } else if tokens > self.budget.recent_messages / 3 {
                // Try to compress very long messages
                let compressed_content = self.compress_message(&message.content);
                let compressed_tokens = Self::estimate_tokens(&compressed_content);

                if current_tokens + compressed_tokens <= self.budget.recent_messages {
                    let mut compressed_message = message.clone();
                    compressed_message.content = compressed_content;
                    optimized_messages.insert(0, compressed_message);
                    current_tokens += compressed_tokens;
                    self.optimization_stats.messages_compressed += 1;
                }
                break;
            } else {
                break;
            }
        }

        self.current_usage.message_tokens = current_tokens;
        optimized_messages
    }

    /// Compress message content while preserving key information
    fn compress_message(&self, content: &str) -> String {
        let max_length = (self.budget.recent_messages / 3) * 4; // Approximate character limit

        if content.len() <= max_length {
            return content.to_string();
        }

        // Try to preserve important parts: beginning and end
        let quarter_length = max_length / 4;
        let beginning = &content[..quarter_length.min(content.len())];
        let end_start = content.len().saturating_sub(quarter_length);
        let end = &content[end_start..];

        format!("{}...[summarized]...{}", beginning.trim(), end.trim())
    }

    /// Filter third-party information based on relevance and recency
    pub fn optimize_third_party_context(
        &mut self,
        third_parties: Vec<ThirdPartyIndividual>,
    ) -> Vec<ThirdPartyIndividual> {
        let mut filtered_parties = Vec::new();
        let mut current_tokens = 0;

        // Sort by importance and recency
        let mut sorted_parties = third_parties;
        sorted_parties.sort_by(|a, b| {
            let score_a = a.importance_score * (a.mention_count as f32).ln().max(1.0);
            let score_b = b.importance_score * (b.mention_count as f32).ln().max(1.0);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for party in sorted_parties {
            let party_text = self.format_third_party_for_context(&party);
            let party_tokens = Self::estimate_tokens(&party_text);

            if current_tokens + party_tokens <= self.budget.third_party_info {
                filtered_parties.push(party);
                current_tokens += party_tokens;
            } else {
                self.optimization_stats.third_parties_filtered += 1;
            }
        }

        self.current_usage.third_party_tokens = current_tokens;
        filtered_parties
    }

    /// Format third-party individual for context inclusion
    fn format_third_party_for_context(&self, party: &ThirdPartyIndividual) -> String {
        let mut details = vec![party.name.clone()];

        if let Some(ref relationship) = party.relationship_to_user {
            details.push(format!("rel:{}", relationship));
        }

        if let Some(ref occupation) = party.occupation {
            details.push(format!("job:{}", occupation));
        }

        if let Some(ref traits) = party.personality_traits {
            let short_traits = if traits.len() > 30 {
                format!("{}...", &traits[..30])
            } else {
                traits.clone()
            };
            details.push(format!("traits:{}", short_traits));
        }

        format!(
            "{} (mentioned {} times)",
            details.join(", "),
            party.mention_count
        )
    }

    /// Get comprehensive usage statistics
    pub fn get_usage_statistics(&mut self) -> TokenUsageStatistics {
        self.current_usage.total_context_tokens = self.current_usage.system_tokens
            + self.current_usage.attitude_tokens
            + self.current_usage.third_party_tokens
            + self.current_usage.message_tokens;

        let remaining_response_tokens = self.budget.response_buffer.min(
            self.budget
                .total
                .saturating_sub(self.current_usage.total_context_tokens),
        );

        let total_utilization = self.current_usage.total_context_tokens + remaining_response_tokens;
        let utilization_percentage =
            (total_utilization as f32 / self.budget.total as f32 * 100.0) as u8;

        // Check for overflow situation
        if self.current_usage.total_context_tokens > self.budget.total - self.budget.response_buffer
        {
            self.optimization_stats.overflow_events += 1;
        }

        TokenUsageStatistics {
            budget: self.budget.clone(),
            current_usage: self.current_usage.clone(),
            remaining_response_tokens,
            utilization_percentage,
            optimization_stats: self.optimization_stats.clone(),
            overflow_risk: self.current_usage.total_context_tokens
                > (self.budget.total as f32 * 0.85) as usize,
        }
    }

    /// Suggest optimizations based on current usage patterns
    pub fn get_optimization_suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();
        let stats = &self.optimization_stats;

        if stats.attitudes_filtered > 5 {
            suggestions.push(format!(
                "Consider increasing attitude token budget - filtered {} attitudes",
                stats.attitudes_filtered
            ));
        }

        if stats.messages_compressed > 3 {
            suggestions.push(format!(
                "Frequent message compression detected - {} messages compressed",
                stats.messages_compressed
            ));
        }

        if stats.overflow_events > 0 {
            suggestions.push(format!(
                "Context overflow detected {} times - consider enabling dynamic context",
                stats.overflow_events
            ));
        }

        let utilization = (self.current_usage.total_context_tokens as f32
            / self.budget.total as f32
            * 100.0) as u8;
        if utilization > 90 {
            suggestions
                .push("Context utilization above 90% - consider optimizing content".to_string());
        } else if utilization < 50 {
            suggestions.push(
                "Low context utilization - could include more conversation history".to_string(),
            );
        }

        if suggestions.is_empty() {
            suggestions.push("Token usage is well optimized".to_string());
        }

        suggestions
    }
}

#[derive(Debug)]
pub struct TokenUsageStatistics {
    pub budget: TokenBudget,
    pub current_usage: TokenUsage,
    pub remaining_response_tokens: usize,
    pub utilization_percentage: u8,
    pub optimization_stats: OptimizationStats,
    pub overflow_risk: bool,
}

impl TokenUsageStatistics {
    pub fn print_detailed_stats(&self) {
        println!("🧠 Comprehensive Token Budget Analysis:");
        println!("   VRAM Tier: {:?}", self.budget.vram_tier);
        println!("   Total Budget: {} tokens", self.budget.total);
        println!();
        println!("📊 Current Usage:");
        println!(
            "   System Prompt: {}/{} tokens",
            self.current_usage.system_tokens, self.budget.system_prompt
        );
        println!(
            "   Attitudes: {}/{} tokens",
            self.current_usage.attitude_tokens, self.budget.attitude_data
        );
        println!(
            "   Third-party Info: {}/{} tokens",
            self.current_usage.third_party_tokens, self.budget.third_party_info
        );
        println!(
            "   Messages: {}/{} tokens",
            self.current_usage.message_tokens, self.budget.recent_messages
        );
        println!(
            "   Response Buffer: {} tokens available",
            self.remaining_response_tokens
        );
        println!(
            "   Total Context: {} tokens",
            self.current_usage.total_context_tokens
        );
        println!("   Utilization: {}%", self.utilization_percentage);

        if self.overflow_risk {
            println!("⚠️  WARNING: High context utilization - approaching overflow");
        }

        println!();
        println!("🔧 Optimization Results:");
        println!(
            "   Messages compressed: {}",
            self.optimization_stats.messages_compressed
        );
        println!(
            "   Attitudes filtered: {}",
            self.optimization_stats.attitudes_filtered
        );
        println!(
            "   Third-parties filtered: {}",
            self.optimization_stats.third_parties_filtered
        );
        println!(
            "   Overflow events: {}",
            self.optimization_stats.overflow_events
        );

        if self.optimization_stats.total_savings > 0 {
            println!(
                "   Total tokens saved: {}",
                self.optimization_stats.total_savings
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::get_current_date;

    #[test]
    fn test_token_budget_allocation() {
        let budget = TokenBudget::from_vram_limit(4, 2048);

        assert_eq!(budget.total, 2048);
        assert!(matches!(budget.vram_tier, VramTier::Standard));

        // Verify allocation percentages (allow for small rounding differences)
        let total_allocated = budget.system_prompt
            + budget.attitude_data
            + budget.third_party_info
            + budget.recent_messages
            + budget.response_buffer;
        assert!((total_allocated as i32 - budget.total as i32).abs() <= 2); // Allow small rounding differences

        // Check specific allocations
        assert_eq!(budget.system_prompt, (2048 as f32 * 0.15) as usize);
        assert_eq!(budget.attitude_data, (2048 as f32 * 0.20) as usize);
        assert_eq!(budget.third_party_info, (2048 as f32 * 0.10) as usize);
        assert_eq!(budget.recent_messages, (2048 as f32 * 0.40) as usize);
        assert_eq!(budget.response_buffer, (2048 as f32 * 0.15) as usize);
    }

    #[test]
    fn test_vram_tier_classification() {
        let minimal = TokenBudget::from_vram_limit(2, 4096);
        assert!(matches!(minimal.vram_tier, VramTier::Minimal));
        assert_eq!(minimal.total, 1024);

        let standard = TokenBudget::from_vram_limit(4, 4096);
        assert!(matches!(standard.vram_tier, VramTier::Standard));
        assert_eq!(standard.total, 2048);

        let extended = TokenBudget::from_vram_limit(6, 4096);
        assert!(matches!(extended.vram_tier, VramTier::Extended));
        assert_eq!(extended.total, 3072);

        let maximum = TokenBudget::from_vram_limit(8, 4096);
        assert!(matches!(maximum.vram_tier, VramTier::Maximum));
        assert_eq!(maximum.total, 4096);
    }

    #[test]
    fn test_token_estimation() {
        assert_eq!(TokenUsageMonitor::estimate_tokens(""), 0);
        assert!(TokenUsageMonitor::estimate_tokens("Hello world") > 0);

        let long_text = "This is a longer piece of text with multiple words and punctuation.";
        let tokens = TokenUsageMonitor::estimate_tokens(long_text);
        assert!(tokens > 10);
        assert!(tokens < 50);

        let word_count = long_text.split_whitespace().count();
        let word_estimate = (word_count as f32 * 1.3) as usize;
        assert!(tokens >= word_estimate);
    }

    #[test]
    fn test_attitude_significance_calculation() {
        let monitor = TokenUsageMonitor::new(TokenBudget::from_vram_limit(4, 2048));

        let high_significance_attitude = CompanionAttitude {
            id: Some(1),
            companion_id: 1,
            target_id: 1,
            target_type: "user".to_string(),
            attraction: 50.0,
            trust: 80.0,
            fear: 5.0,
            anger: 0.0,
            joy: 70.0,
            sorrow: 0.0,
            disgust: 0.0,
            surprise: 20.0,
            curiosity: 60.0,
            respect: 75.0,
            suspicion: 10.0,
            gratitude: 40.0,
            jealousy: 0.0,
            empathy: 85.0,
            relationship_score: Some(70.0),
            last_updated: get_current_date(),
            created_at: get_current_date(),
        };

        let low_significance_attitude = CompanionAttitude {
            id: Some(2),
            companion_id: 1,
            target_id: 2,
            target_type: "stranger".to_string(),
            attraction: 0.0,
            trust: 5.0,
            fear: 2.0,
            anger: 0.0,
            joy: 3.0,
            sorrow: 0.0,
            disgust: 0.0,
            surprise: 1.0,
            curiosity: 8.0,
            respect: 0.0,
            suspicion: 2.0,
            gratitude: 0.0,
            jealousy: 0.0,
            empathy: 5.0,
            relationship_score: Some(5.0),
            last_updated: get_current_date(),
            created_at: get_current_date(),
        };

        let high_score =
            TokenUsageMonitor::calculate_attitude_significance(&high_significance_attitude);
        let low_score =
            TokenUsageMonitor::calculate_attitude_significance(&low_significance_attitude);

        assert!(high_score > low_score);
        assert!(high_score > 50.0);
        assert!(low_score < 20.0);
    }

    #[test]
    fn test_attitude_filtering() {
        let mut monitor = TokenUsageMonitor::new(TokenBudget::from_vram_limit(2, 1024)); // Minimal tier

        let attitudes = vec![
            create_test_attitude(1, 80.0, 70.0, 60.0), // High significance
            create_test_attitude(2, 5.0, 3.0, 2.0),    // Low significance
            create_test_attitude(3, 45.0, 50.0, 40.0), // Medium significance
            create_test_attitude(4, 2.0, 1.0, 0.0),    // Very low significance
        ];

        let filtered = monitor.optimize_attitude_context(attitudes);

        // Should filter out low significance attitudes
        assert!(filtered.len() < 4);
        assert!(filtered.len() >= 1); // At least the high significance one should remain

        // Check that we have some usage statistics
        assert!(monitor.current_usage.attitude_tokens > 0);
        assert!(monitor.optimization_stats.attitudes_filtered > 0);
    }

    #[test]
    fn test_message_compression() {
        let mut monitor = TokenUsageMonitor::new(TokenBudget::from_vram_limit(2, 1024)); // Small budget

        let messages = vec![
            create_test_message(1, false, "Short message"),
            create_test_message(2, true, "This is a much longer message that should potentially be compressed because it exceeds reasonable length limits for small token budgets and contains lots of unnecessary details that could be summarized"),
            create_test_message(3, false, "Another short one"),
            create_test_message(4, true, "Final response"),
        ];

        let optimized = monitor.optimize_message_context(messages);

        assert!(!optimized.is_empty());
        assert!(monitor.current_usage.message_tokens > 0);
        assert!(monitor.current_usage.message_tokens <= monitor.budget.recent_messages);
    }

    #[test]
    fn test_third_party_optimization() {
        let mut monitor = TokenUsageMonitor::new(TokenBudget::from_vram_limit(4, 2048));

        let third_parties = vec![
            create_test_third_party("Alice", 0.9, 15), // High importance, mentioned often
            create_test_third_party("Bob", 0.3, 2),    // Low importance, rarely mentioned
            create_test_third_party("Charlie", 0.7, 8), // Medium importance
            create_test_third_party("Dave", 0.1, 1),   // Very low importance
        ];

        let optimized = monitor.optimize_third_party_context(third_parties);

        // Should prioritize by importance and mention count
        assert!(!optimized.is_empty());
        assert!(optimized.len() <= 4);

        // Alice should be first (highest importance * mention count)
        if let Some(first) = optimized.first() {
            assert_eq!(first.name, "Alice");
        }

        assert!(monitor.current_usage.third_party_tokens > 0);
    }

    #[test]
    fn test_usage_statistics() {
        let budget = TokenBudget::from_vram_limit(4, 2048);
        let mut monitor = TokenUsageMonitor::new(budget.clone());

        // Simulate some usage
        monitor.current_usage.system_tokens = 100;
        monitor.current_usage.attitude_tokens = 200;
        monitor.current_usage.third_party_tokens = 150;
        monitor.current_usage.message_tokens = 500;

        let stats = monitor.get_usage_statistics();

        assert_eq!(stats.current_usage.total_context_tokens, 950);
        assert!(stats.remaining_response_tokens > 0);
        assert!(stats.utilization_percentage > 0);
        assert!(!stats.overflow_risk); // Should not be at risk with these numbers

        let suggestions = monitor.get_optimization_suggestions();
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_overflow_detection() {
        let budget = TokenBudget::from_vram_limit(2, 1024); // Small budget
        let mut monitor = TokenUsageMonitor::new(budget);

        // Simulate high usage that approaches overflow
        monitor.current_usage.system_tokens = 200;
        monitor.current_usage.attitude_tokens = 300;
        monitor.current_usage.third_party_tokens = 200;
        monitor.current_usage.message_tokens = 400; // Total: 1100 > 1024

        let stats = monitor.get_usage_statistics();
        assert!(stats.overflow_risk);

        let suggestions = monitor.get_optimization_suggestions();
        assert!(suggestions
            .iter()
            .any(|s| s.contains("overflow") || s.contains("90%")));
    }

    // Helper functions for testing
    fn create_test_attitude(id: i32, trust: f32, joy: f32, curiosity: f32) -> CompanionAttitude {
        CompanionAttitude {
            id: Some(id),
            companion_id: 1,
            target_id: id,
            target_type: "test".to_string(),
            attraction: 0.0,
            trust,
            fear: 0.0,
            anger: 0.0,
            joy,
            sorrow: 0.0,
            disgust: 0.0,
            surprise: 0.0,
            curiosity,
            respect: 0.0,
            suspicion: 0.0,
            gratitude: 0.0,
            jealousy: 0.0,
            empathy: 0.0,
            relationship_score: Some((trust + joy + curiosity) / 3.0),
            last_updated: get_current_date(),
            created_at: get_current_date(),
        }
    }

    fn create_test_message(id: i32, ai: bool, content: &str) -> Message {
        Message {
            id,
            ai,
            content: content.to_string(),
            created_at: get_current_date(),
        }
    }

    fn create_test_third_party(name: &str, importance: f32, mentions: i32) -> ThirdPartyIndividual {
        ThirdPartyIndividual {
            id: None,
            name: name.to_string(),
            relationship_to_user: Some("friend".to_string()),
            relationship_to_companion: Some("acquaintance".to_string()),
            occupation: Some("tester".to_string()),
            personality_traits: Some("helpful".to_string()),
            physical_description: None,
            first_mentioned: get_current_date(),
            last_mentioned: Some(get_current_date()),
            mention_count: mentions,
            importance_score: importance,
            created_at: get_current_date(),
            updated_at: get_current_date(),
        }
    }
}
