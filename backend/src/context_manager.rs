use crate::database::{CompanionAttitude, ConfigView, Message, ThirdPartyIndividual};
use crate::token_budget::{TokenBudget, TokenUsageMonitor, TokenUsageStatistics};
use crate::system_memory::{SystemMemoryDetector, SystemMemoryInfo, MemoryStrategy};

pub struct ContextManager {
    pub config: ConfigView,
    pub token_budget: TokenBudget,
    pub usage_monitor: TokenUsageMonitor,
    pub system_memory_detector: SystemMemoryDetector,
    pub hybrid_context_allocation: Option<HybridContextAllocation>,
    // Legacy fields for backward compatibility
    pub system_token_budget: usize,
    pub attitude_token_budget: usize,
    pub message_token_budget: usize,
    pub response_token_budget: usize,
}

#[derive(Debug, Clone)]
pub struct HybridContextAllocation {
    pub total_context_tokens: usize,
    pub vram_context_tokens: usize,
    pub system_ram_context_tokens: usize,
    pub allocation_strategy: HybridStrategy,
    pub system_memory_info: SystemMemoryInfo,
}

#[derive(Debug, Clone)]
pub enum HybridStrategy {
    VramOnly,
    Conservative, // +25% context using RAM
    Balanced,     // +50% context using RAM
    Aggressive,   // +100% context using RAM
}

impl ContextManager {
    pub fn new(config: ConfigView) -> Self {
        // Initialize system memory detector
        let system_memory_detector = SystemMemoryDetector::new()
            .with_safety_margin(config.ram_safety_margin_gb as f32)
            .with_max_usage(config.max_system_ram_usage_gb as f32);

        // Calculate context size with hybrid approach if enabled
        let (context_size, hybrid_allocation) = if config.enable_hybrid_context {
            Self::calculate_hybrid_context_size(&config, &system_memory_detector)
        } else if config.enable_dynamic_context {
            let size = Self::calculate_dynamic_context_size(config.vram_limit_gb, config.context_window_size);
            (size, None)
        } else {
            (config.context_window_size, None)
        };

        // Create comprehensive token budget system
        let token_budget = TokenBudget::from_vram_limit(config.vram_limit_gb, context_size);
        let usage_monitor = TokenUsageMonitor::new(token_budget.clone());

        // Legacy allocations for backward compatibility
        let system_token_budget = token_budget.system_prompt;
        let attitude_token_budget = token_budget.attitude_data;
        let message_token_budget = token_budget.recent_messages;
        let response_token_budget = token_budget.response_buffer;

        Self {
            config,
            token_budget,
            usage_monitor,
            system_memory_detector,
            hybrid_context_allocation: hybrid_allocation,
            system_token_budget,
            attitude_token_budget,
            message_token_budget,
            response_token_budget,
        }
    }

    /// Calculate dynamic context size based on VRAM availability
    fn calculate_dynamic_context_size(vram_gb: usize, configured_size: usize) -> usize {
        let dynamic_size = match vram_gb {
            0..=2 => 1024, // 2GB VRAM: minimal context
            3..=4 => 2048, // 4GB VRAM: standard context
            5..=6 => 3072, // 6GB VRAM: extended context
            _ => 4096,     // 6GB+ VRAM: maximum context
        };

        // Use the smaller of dynamic calculation or user configuration
        std::cmp::min(dynamic_size, configured_size)
    }

    /// Calculate hybrid context size using both VRAM and system RAM
    fn calculate_hybrid_context_size(
        config: &ConfigView,
        system_memory_detector: &SystemMemoryDetector,
    ) -> (usize, Option<HybridContextAllocation>) {
        // Get system memory information
        let system_memory_info = match system_memory_detector.detect_system_memory() {
            Ok(info) => info,
            Err(e) => {
                eprintln!("Warning: Failed to detect system memory, falling back to VRAM-only: {}", e);
                let vram_size = Self::calculate_dynamic_context_size(config.vram_limit_gb, config.context_window_size);
                return (vram_size, None);
            }
        };

        // Get memory allocation recommendation
        let memory_allocation = system_memory_detector.calculate_memory_allocation(&system_memory_info);

        // Check if we should use hybrid approach
        if memory_allocation.recommended_usage_gb < 1.0 || system_memory_detector.is_memory_pressure(&system_memory_info) {
            // Not enough RAM or system under pressure, use VRAM-only
            let vram_size = Self::calculate_dynamic_context_size(config.vram_limit_gb, config.context_window_size);
            return (vram_size, None);
        }

        // Calculate base VRAM context size
        let base_vram_context = Self::calculate_dynamic_context_size(config.vram_limit_gb, config.context_window_size);
        
        // Determine hybrid strategy based on configuration and available RAM
        let hybrid_strategy = match config.context_expansion_strategy.as_str() {
            "conservative" => HybridStrategy::Conservative,
            "balanced" => HybridStrategy::Balanced,
            "aggressive" => HybridStrategy::Aggressive,
            "vram_only" => HybridStrategy::VramOnly,
            _ => {
                // Auto-determine based on system memory info
                match memory_allocation.allocation_strategy {
                    MemoryStrategy::VramOnly => HybridStrategy::VramOnly,
                    MemoryStrategy::Conservative => HybridStrategy::Conservative,
                    MemoryStrategy::Balanced => HybridStrategy::Balanced,
                    MemoryStrategy::Aggressive => HybridStrategy::Aggressive,
                }
            }
        };

        // Calculate RAM context bonus based on strategy
        let ram_context_tokens = match hybrid_strategy {
            HybridStrategy::VramOnly => 0,
            HybridStrategy::Conservative => (base_vram_context as f32 * 0.25) as usize,
            HybridStrategy::Balanced => (base_vram_context as f32 * 0.50) as usize,
            HybridStrategy::Aggressive => (base_vram_context as f32 * 1.00) as usize,
        };

        // Ensure we don't exceed configured limits
        let max_ram_tokens = (memory_allocation.recommended_usage_gb * 1024.0 * 256.0) as usize; // ~256 tokens per MB
        let final_ram_context_tokens = ram_context_tokens.min(max_ram_tokens);
        
        let total_context_tokens = base_vram_context + final_ram_context_tokens;
        
        // Don't exceed user's configured maximum
        let final_total_context = total_context_tokens.min(config.context_window_size);
        let final_ram_context = if final_total_context > base_vram_context {
            final_total_context - base_vram_context
        } else {
            0
        };

        let hybrid_allocation = HybridContextAllocation {
            total_context_tokens: final_total_context,
            vram_context_tokens: base_vram_context,
            system_ram_context_tokens: final_ram_context,
            allocation_strategy: hybrid_strategy,
            system_memory_info: system_memory_info.clone(),
        };

        println!("🧠 Hybrid Context Allocation: {} total tokens (VRAM: {}, RAM: {}) - Strategy: {:?}", 
                 final_total_context, base_vram_context, final_ram_context, hybrid_allocation.allocation_strategy);
        println!("📊 {}", system_memory_detector.get_memory_summary(&system_memory_info));

        (final_total_context, Some(hybrid_allocation))
    }

    /// Estimate token count for a string (rough approximation: 1 token ≈ 4 chars)
    pub fn estimate_tokens(text: &str) -> usize {
        // Simple approximation: average 4 characters per token
        // This is a rough estimate - real tokenizers are more complex
        (text.len() as f32 / 4.0).ceil() as usize
    }

    /// Prioritize and trim messages to fit within token budget
    pub fn manage_message_context(&self, messages: Vec<Message>) -> Vec<Message> {
        if messages.is_empty() {
            return messages;
        }

        let mut selected_messages = Vec::new();
        let mut current_tokens = 0;

        // Always include the most recent message (user's latest input)
        if let Some(last_message) = messages.last() {
            let tokens = Self::estimate_tokens(&last_message.content);
            if tokens <= self.message_token_budget {
                selected_messages.push(last_message.clone());
                current_tokens += tokens;
            }
        }

        // Work backwards through messages, prioritizing recent ones
        for message in messages.iter().rev().skip(1) {
            let tokens = Self::estimate_tokens(&message.content);

            if current_tokens + tokens <= self.message_token_budget {
                selected_messages.insert(0, message.clone()); // Insert at beginning to maintain order
                current_tokens += tokens;
            } else {
                // If message is too long, try to summarize or truncate
                if tokens > self.message_token_budget / 4 {
                    // If single message uses >25% of budget
                    let truncated_content =
                        self.truncate_message(&message.content, self.message_token_budget / 4);
                    let truncated_tokens = Self::estimate_tokens(&truncated_content);

                    if current_tokens + truncated_tokens <= self.message_token_budget {
                        let mut truncated_message = message.clone();
                        truncated_message.content = truncated_content;
                        selected_messages.insert(0, truncated_message);
                        current_tokens += truncated_tokens;
                    }
                }
                break; // Stop adding more messages
            }
        }

        selected_messages
    }

    /// Truncate message content to fit within token limit
    fn truncate_message(&self, content: &str, max_tokens: usize) -> String {
        let max_chars = max_tokens * 4; // Approximate character limit
        if content.len() <= max_chars {
            content.to_string()
        } else {
            let truncated = &content[..max_chars.saturating_sub(20)];
            format!("{}... [truncated]", truncated)
        }
    }

    /// Calculate remaining tokens available for response generation
    pub fn get_response_token_limit(&self, used_context_tokens: usize) -> usize {
        let available_response_tokens = self
            .response_token_budget
            .min(self.config.max_response_tokens);

        // Ensure we don't exceed total context window
        let total_used = used_context_tokens + available_response_tokens;
        if total_used > self.config.context_window_size {
            let excess = total_used - self.config.context_window_size;
            available_response_tokens.saturating_sub(excess)
        } else {
            available_response_tokens
        }
    }

    /// Check if context management is working within memory constraints
    pub fn validate_context_size(
        &self,
        system_tokens: usize,
        attitude_tokens: usize,
        message_tokens: usize,
    ) -> bool {
        let total_tokens =
            system_tokens + attitude_tokens + message_tokens + self.response_token_budget;
        total_tokens <= self.config.context_window_size
    }

    /// Get memory usage statistics (legacy method)
    pub fn get_memory_stats(
        &self,
        system_tokens: usize,
        attitude_tokens: usize,
        message_tokens: usize,
    ) -> MemoryStats {
        let used_tokens = system_tokens + attitude_tokens + message_tokens;
        let available_response_tokens = self.get_response_token_limit(used_tokens);
        let total_used = used_tokens + available_response_tokens;

        MemoryStats {
            system_tokens,
            attitude_tokens,
            message_tokens,
            response_tokens: available_response_tokens,
            total_used_tokens: total_used,
            total_available_tokens: self.config.context_window_size,
            utilization_percentage: (total_used as f32 / self.config.context_window_size as f32
                * 100.0) as u8,
        }
    }

    /// Comprehensive context optimization using the new token budget system
    pub fn optimize_full_context(
        &mut self,
        system_prompt: &str,
        messages: Vec<Message>,
        attitudes: Vec<CompanionAttitude>,
        third_parties: Vec<ThirdPartyIndividual>,
    ) -> OptimizedContext {
        // Estimate system prompt tokens
        let system_tokens = TokenUsageMonitor::estimate_tokens(system_prompt);
        self.usage_monitor.current_usage.system_tokens = system_tokens;

        // Optimize each component using the advanced token budget system
        let optimized_attitudes = self.usage_monitor.optimize_attitude_context(attitudes);
        let optimized_third_parties = self
            .usage_monitor
            .optimize_third_party_context(third_parties);
        let optimized_messages = self.usage_monitor.optimize_message_context(messages);

        // Get comprehensive usage statistics
        let usage_stats = self.usage_monitor.get_usage_statistics();
        let optimization_suggestions = self.usage_monitor.get_optimization_suggestions();

        OptimizedContext {
            system_prompt: system_prompt.to_string(),
            messages: optimized_messages,
            attitudes: optimized_attitudes,
            third_parties: optimized_third_parties,
            usage_statistics: usage_stats,
            optimization_suggestions,
            overflow_detected: self.check_overflow_risk(),
        }
    }

    /// Check if we're at risk of context overflow
    fn check_overflow_risk(&self) -> bool {
        let total_context = self.usage_monitor.current_usage.total_context_tokens;
        let safety_threshold = (self.token_budget.total as f32 * 0.85) as usize;
        total_context > safety_threshold
    }

    /// Get token budget allocation summary
    pub fn get_budget_summary(&self) -> String {
        self.token_budget.get_allocation_summary()
    }

    /// Reset usage monitor for new conversation
    pub fn reset_usage_monitor(&mut self) {
        self.usage_monitor = TokenUsageMonitor::new(self.token_budget.clone());
    }

    /// Get optimization suggestions based on usage patterns
    pub fn get_context_optimization_suggestions(&self) -> Vec<String> {
        self.usage_monitor.get_optimization_suggestions()
    }

    /// Format context for LLM prompt with priority-based inclusion
    pub fn format_optimized_prompt(&self, context: &OptimizedContext) -> String {
        let mut prompt_parts = Vec::new();

        // System prompt (always included)
        prompt_parts.push(context.system_prompt.clone());

        // Third-party information (if any)
        if !context.third_parties.is_empty() {
            let third_party_info = context
                .third_parties
                .iter()
                .map(|tp| format!("- {} (mentioned {} times)", tp.name, tp.mention_count))
                .collect::<Vec<_>>()
                .join("\n");
            prompt_parts.push(format!("Known individuals:\n{}", third_party_info));
        }

        // Attitude context (if any significant attitudes)
        if !context.attitudes.is_empty() {
            let attitude_info = context
                .attitudes
                .iter()
                .map(|att| {
                    format!(
                        "Attitude towards {}: trust:{:.0}, joy:{:.0}, curiosity:{:.0}",
                        att.target_type, att.trust, att.joy, att.curiosity
                    )
                })
                .collect::<Vec<_>>()
                .join("\n- ");
            if !attitude_info.is_empty() {
                prompt_parts.push(format!("Current attitudes:\n- {}", attitude_info));
            }
        }

        // Recent conversation history
        if !context.messages.is_empty() {
            let conversation = context
                .messages
                .iter()
                .map(|msg| {
                    let speaker = if msg.ai { "Assistant" } else { "Human" };
                    format!("{}: {}", speaker, msg.content)
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            prompt_parts.push(format!("Recent conversation:\n{}", conversation));
        }

        prompt_parts.join("\n\n")
    }

    /// Handle context overflow by intelligently trimming content
    pub fn handle_context_overflow(&mut self, context: &mut OptimizedContext) -> bool {
        if !context.overflow_detected {
            return false;
        }

        let mut overflow_handled = false;

        // Strategy 1: Further compress messages
        if context.messages.len() > 3 {
            let compressed_messages = self.compress_message_history(&context.messages, 3);
            if compressed_messages.len() < context.messages.len() {
                context.messages = compressed_messages;
                overflow_handled = true;
            }
        }

        // Strategy 2: Reduce attitude details
        if context.attitudes.len() > 5 {
            context.attitudes.truncate(5);
            overflow_handled = true;
        }

        // Strategy 3: Limit third-party information
        if context.third_parties.len() > 3 {
            context.third_parties.truncate(3);
            overflow_handled = true;
        }

        // Recalculate usage after modifications
        if overflow_handled {
            context.usage_statistics = self.usage_monitor.get_usage_statistics();
            context.overflow_detected = self.check_overflow_risk();
        }

        overflow_handled
    }

    /// Compress message history to essential messages
    fn compress_message_history(&self, messages: &[Message], target_count: usize) -> Vec<Message> {
        if messages.len() <= target_count {
            return messages.to_vec();
        }

        let mut compressed = Vec::new();

        // Always keep the most recent message
        if let Some(last) = messages.last() {
            compressed.push(last.clone());
        }

        // Try to keep a balanced selection of AI and human messages
        let remaining_slots = target_count - 1;
        let mut selected_indices = Vec::new();
        let step = messages.len() / remaining_slots.max(1);

        for i in (0..messages.len()).step_by(step).take(remaining_slots) {
            if i != messages.len() - 1 {
                // Don't duplicate the last message
                selected_indices.push(i);
            }
        }

        for &index in selected_indices.iter().rev() {
            compressed.insert(0, messages[index].clone());
        }

        compressed
    }

    /// Check if response budget is critically low and attempt to resolve it
    pub fn handle_response_budget_crisis(&mut self, current_response_budget: usize) -> bool {
        if current_response_budget >= 100 {
            return false; // No crisis
        }

        println!("⚠️  Response budget crisis detected: {} tokens available", current_response_budget);

        // Strategy 1: Try to expand context window if hybrid mode is available
        if self.config.enable_hybrid_context && self.hybrid_context_allocation.is_none() {
            if let Some(expanded_allocation) = self.try_expand_context_window() {
                println!("🔄 Expanded context window using system RAM");
                self.hybrid_context_allocation = Some(expanded_allocation);
                self.recalculate_token_budget();
                return true;
            }
        }

        // Strategy 2: Optimize existing allocation
        if let Some(ref hybrid_allocation) = self.hybrid_context_allocation {
            if self.can_expand_ram_allocation(hybrid_allocation) {
                if let Some(expanded_allocation) = self.expand_ram_allocation() {
                    println!("🔄 Expanded RAM allocation for context");
                    self.hybrid_context_allocation = Some(expanded_allocation);
                    self.recalculate_token_budget();
                    return true;
                }
            }
        }

        // Strategy 3: Reallocate existing tokens more efficiently
        self.reallocate_token_budget_for_response();
        println!("🔄 Reallocated token budget to prioritize response generation");
        
        false
    }

    /// Try to expand context window using system RAM
    fn try_expand_context_window(&self) -> Option<HybridContextAllocation> {
        let (_, hybrid_allocation) = Self::calculate_hybrid_context_size(
            &self.config,
            &self.system_memory_detector,
        );
        
        hybrid_allocation
    }

    /// Check if we can expand RAM allocation for context
    fn can_expand_ram_allocation(&self, current_allocation: &HybridContextAllocation) -> bool {
        // Check if system memory situation has improved
        match self.system_memory_detector.detect_system_memory() {
            Ok(memory_info) => {
                let allocation = self.system_memory_detector.calculate_memory_allocation(&memory_info);
                allocation.recommended_usage_gb > 1.0 && !self.system_memory_detector.is_memory_pressure(&memory_info)
            }
            Err(_) => false,
        }
    }

    /// Expand RAM allocation for context
    fn expand_ram_allocation(&self) -> Option<HybridContextAllocation> {
        if let Some(ref current) = self.hybrid_context_allocation {
            let current_ram_gb = current.system_ram_context_tokens as f32 / (1024.0 * 256.0);
            
            // Try to increase RAM allocation by 50%
            let new_ram_tokens = (current.system_ram_context_tokens as f32 * 1.5) as usize;
            let new_total_tokens = current.vram_context_tokens + new_ram_tokens;
            
            // Don't exceed configured maximum
            let final_total = new_total_tokens.min(self.config.context_window_size);
            let final_ram = if final_total > current.vram_context_tokens {
                final_total - current.vram_context_tokens
            } else {
                current.system_ram_context_tokens
            };
            
            if final_ram > current.system_ram_context_tokens {
                return Some(HybridContextAllocation {
                    total_context_tokens: final_total,
                    vram_context_tokens: current.vram_context_tokens,
                    system_ram_context_tokens: final_ram,
                    allocation_strategy: current.allocation_strategy.clone(),
                    system_memory_info: current.system_memory_info.clone(),
                });
            }
        }
        None
    }

    /// Reallocate token budget to prioritize response generation
    fn reallocate_token_budget_for_response(&mut self) {
        // Temporarily reduce other allocations to boost response budget
        let current_total = self.token_budget.total;
        
        // Reduce attitude allocation by 25%
        let attitude_reduction = (self.token_budget.attitude_data as f32 * 0.25) as usize;
        // Reduce third-party allocation by 50%
        let third_party_reduction = (self.token_budget.third_party_info as f32 * 0.50) as usize;
        
        // Increase response buffer with the freed tokens
        let additional_response_tokens = attitude_reduction + third_party_reduction;
        
        self.token_budget.attitude_data -= attitude_reduction;
        self.token_budget.third_party_info -= third_party_reduction;
        self.token_budget.response_buffer += additional_response_tokens;
        
        // Update legacy fields
        self.attitude_token_budget = self.token_budget.attitude_data;
        self.response_token_budget = self.token_budget.response_buffer;
    }

    /// Recalculate token budget based on new context size
    fn recalculate_token_budget(&mut self) {
        if let Some(ref hybrid_allocation) = self.hybrid_context_allocation {
            self.token_budget = TokenBudget::from_vram_limit(
                self.config.vram_limit_gb,
                hybrid_allocation.total_context_tokens,
            );
            
            // Update usage monitor with new budget
            self.usage_monitor = TokenUsageMonitor::new(self.token_budget.clone());
            
            // Update legacy fields
            self.system_token_budget = self.token_budget.system_prompt;
            self.attitude_token_budget = self.token_budget.attitude_data;
            self.message_token_budget = self.token_budget.recent_messages;
            self.response_token_budget = self.token_budget.response_buffer;
        }
    }

    /// Get current memory usage summary including hybrid allocation
    pub fn get_hybrid_memory_summary(&self) -> String {
        let base_summary = self.get_budget_summary();
        
        if let Some(ref hybrid_allocation) = self.hybrid_context_allocation {
            format!(
                "{}\nHybrid Context: {} total tokens (VRAM: {}, RAM: {}) - Strategy: {:?}",
                base_summary,
                hybrid_allocation.total_context_tokens,
                hybrid_allocation.vram_context_tokens,
                hybrid_allocation.system_ram_context_tokens,
                hybrid_allocation.allocation_strategy
            )
        } else {
            format!("{}\nHybrid Context: Disabled", base_summary)
        }
    }

    /// Check if system can benefit from hybrid context expansion
    pub fn can_benefit_from_hybrid_expansion(&self) -> bool {
        if !self.config.enable_hybrid_context || self.hybrid_context_allocation.is_some() {
            return false;
        }
        
        match self.system_memory_detector.detect_system_memory() {
            Ok(memory_info) => {
                let allocation = self.system_memory_detector.calculate_memory_allocation(&memory_info);
                allocation.recommended_usage_gb > 1.0 && !self.system_memory_detector.is_memory_pressure(&memory_info)
            }
            Err(_) => false,
        }
    }
}

#[derive(Debug)]
pub struct OptimizedContext {
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub attitudes: Vec<CompanionAttitude>,
    pub third_parties: Vec<ThirdPartyIndividual>,
    pub usage_statistics: TokenUsageStatistics,
    pub optimization_suggestions: Vec<String>,
    pub overflow_detected: bool,
}

impl OptimizedContext {
    pub fn print_optimization_summary(&self) {
        println!("🔧 Context Optimization Summary:");
        println!("   Messages included: {}", self.messages.len());
        println!("   Attitudes included: {}", self.attitudes.len());
        println!("   Third-parties included: {}", self.third_parties.len());
        println!("   Overflow detected: {}", self.overflow_detected);

        if !self.optimization_suggestions.is_empty() {
            println!("💡 Optimization suggestions:");
            for suggestion in &self.optimization_suggestions {
                println!("   - {}", suggestion);
            }
        }

        self.usage_statistics.print_detailed_stats();
    }
}

#[derive(Debug)]
pub struct MemoryStats {
    pub system_tokens: usize,
    pub attitude_tokens: usize,
    pub message_tokens: usize,
    pub response_tokens: usize,
    pub total_used_tokens: usize,
    pub total_available_tokens: usize,
    pub utilization_percentage: u8,
}

impl MemoryStats {
    pub fn print_stats(&self) {
        println!("🧠 Context Window Memory Usage:");
        println!("   System Prompt: {} tokens", self.system_tokens);
        println!("   Attitude Data: {} tokens", self.attitude_tokens);
        println!("   Messages: {} tokens", self.message_tokens);
        println!("   Response Budget: {} tokens", self.response_tokens);
        println!(
            "   Total Used: {}/{} tokens ({}%)",
            self.total_used_tokens, self.total_available_tokens, self.utilization_percentage
        );
    }
}
