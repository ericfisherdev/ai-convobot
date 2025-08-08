use crate::database::{ConfigView, Message};

pub struct ContextManager {
    pub config: ConfigView,
    pub system_token_budget: usize,
    pub attitude_token_budget: usize,
    pub message_token_budget: usize,
    pub response_token_budget: usize,
}

impl ContextManager {
    pub fn new(config: ConfigView) -> Self {
        let context_size = if config.enable_dynamic_context {
            Self::calculate_dynamic_context_size(config.vram_limit_gb, config.context_window_size)
        } else {
            config.context_window_size
        };
        
        // Allocate token budget based on the specification
        let system_token_budget = (context_size as f32 * 0.20) as usize;     // 20% for system prompt
        let attitude_token_budget = (context_size as f32 * 0.10) as usize;   // 10% for attitude data
        let message_token_budget = (context_size as f32 * 0.50) as usize;    // 50% for messages
        let response_token_budget = (context_size as f32 * 0.20) as usize;   // 20% for response buffer
        
        Self {
            config,
            system_token_budget,
            attitude_token_budget,
            message_token_budget,
            response_token_budget,
        }
    }
    
    /// Calculate dynamic context size based on VRAM availability
    fn calculate_dynamic_context_size(vram_gb: usize, configured_size: usize) -> usize {
        let dynamic_size = match vram_gb {
            0..=2 => 1024,    // 2GB VRAM: minimal context
            3..=4 => 2048,    // 4GB VRAM: standard context  
            5..=6 => 3072,    // 6GB VRAM: extended context
            _ => 4096,        // 6GB+ VRAM: maximum context
        };
        
        // Use the smaller of dynamic calculation or user configuration
        std::cmp::min(dynamic_size, configured_size)
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
                if tokens > self.message_token_budget / 4 { // If single message uses >25% of budget
                    let truncated_content = self.truncate_message(&message.content, self.message_token_budget / 4);
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
        let available_response_tokens = self.response_token_budget.min(self.config.max_response_tokens);
        
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
    pub fn validate_context_size(&self, system_tokens: usize, attitude_tokens: usize, message_tokens: usize) -> bool {
        let total_tokens = system_tokens + attitude_tokens + message_tokens + self.response_token_budget;
        total_tokens <= self.config.context_window_size
    }
    
    /// Get memory usage statistics
    pub fn get_memory_stats(&self, system_tokens: usize, attitude_tokens: usize, message_tokens: usize) -> MemoryStats {
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
            utilization_percentage: (total_used as f32 / self.config.context_window_size as f32 * 100.0) as u8,
        }
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
        println!("   Total Used: {}/{} tokens ({}%)", 
                self.total_used_tokens, 
                self.total_available_tokens, 
                self.utilization_percentage);
    }
}