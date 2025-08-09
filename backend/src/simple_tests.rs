#[cfg(test)]
mod tests {
    use crate::database::*;
    use crate::inference_optimizer::*;

    #[test]
    fn test_date_functions() {
        let date = get_current_date();
        assert!(!date.is_empty());
        assert!(date.len() > 10);
    }

    #[test]
    fn test_time_question_detection() {
        assert!(contains_time_question("What time is it?"));
        assert!(contains_time_question("What's the date today?"));
        assert!(contains_time_question("It's morning here"));
        assert!(!contains_time_question("How are you doing?"));
        assert!(!contains_time_question("Tell me a story"));
    }

    #[test]
    fn test_inference_optimizer() {
        let optimizer = InferenceOptimizer::new();

        // Test token estimation
        let text = "This is a test";
        let tokens = optimizer.estimate_tokens(text);
        assert!(tokens > 0);
        assert!(tokens <= text.len());

        // Test prompt hashing
        let prompt = "Hello, world!";
        let hash1 = optimizer.hash_prompt(prompt);
        let hash2 = optimizer.hash_prompt(prompt);
        assert_eq!(hash1, hash2);
        assert!(!hash1.is_empty());

        // Test cache operations
        assert!(optimizer.get_cached_prompt("nonexistent").is_none());

        optimizer.cache_prompt("test", "base test", 10);
        let cached = optimizer.get_cached_prompt("test");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().estimated_tokens, 10);
    }

    #[test]
    fn test_message_struct() {
        let message = Message {
            id: 1,
            ai: true,
            content: "Hello world".to_string(),
            created_at: "2024-01-15 10:00".to_string(),
        };

        assert_eq!(message.id, 1);
        assert!(message.ai);
        assert_eq!(message.content, "Hello world");
    }

    #[test]
    fn test_new_message_struct() {
        let new_message = NewMessage {
            ai: false,
            content: "User message".to_string(),
        };

        assert!(!new_message.ai);
        assert_eq!(new_message.content, "User message");
    }

    #[test]
    fn test_companion_attitude_struct() {
        let attitude = CompanionAttitude {
            id: Some(1),
            companion_id: 1,
            target_id: 123,
            target_type: "user".to_string(),
            attraction: 25.0,
            trust: 75.0,
            fear: 10.0,
            anger: 5.0,
            joy: 60.0,
            sorrow: 15.0,
            disgust: 8.0,
            surprise: 30.0,
            curiosity: 85.0,
            respect: 70.0,
            suspicion: 20.0,
            gratitude: 40.0,
            jealousy: 12.0,
            empathy: 80.0,
            lust: 0.0,
            love: 0.0,
            anxiety: 0.0,
            butterflies: 0.0,
            submissiveness: 0.0,
            dominance: 0.0,
            relationship_score: Some(65.5),
            last_updated: "2024-01-15 10:00".to_string(),
            created_at: "2024-01-15 09:00".to_string(),
        };

        assert_eq!(attitude.trust, 75.0);
        assert_eq!(attitude.curiosity, 85.0);
        assert_eq!(attitude.relationship_score, Some(65.5));
    }

    #[test]
    fn test_prompt_optimization() {
        let optimizer = InferenceOptimizer::new();
        let base_components = vec![
            "System: You are a helpful assistant.".to_string(),
            "Human: ".to_string(),
        ];
        let dynamic_content = "Hello, how are you?";
        let messages = vec![];

        let (optimized_prompt, _was_cached) =
            optimizer.optimize_prompt_construction(&base_components, dynamic_content, &messages);

        assert!(optimized_prompt.contains("System: You are a helpful assistant."));
        assert!(optimized_prompt.contains("Hello, how are you?"));
    }

    #[test]
    fn test_stats_tracking() {
        let optimizer = InferenceOptimizer::new();

        let initial_stats = optimizer.get_stats();
        assert_eq!(initial_stats.total_requests, 0);
        assert_eq!(initial_stats.cache_hits, 0);

        // Simulate recording response time
        optimizer.record_response_time(std::time::Duration::from_millis(150));

        let updated_stats = optimizer.get_stats();
        assert_eq!(updated_stats.total_requests, 1);
        assert!(updated_stats.avg_response_time.as_millis() > 0);
    }

    #[test]
    fn test_cache_statistics() {
        let optimizer = InferenceOptimizer::new();

        let (cache_size, hits, hit_rate) = optimizer.get_cache_stats();
        assert_eq!(cache_size, 0);
        assert_eq!(hits, 0);
        assert_eq!(hit_rate, 0.0);

        // Add some cache entries
        optimizer.cache_prompt("test1", "base1", 10);
        optimizer.cache_prompt("test2", "base2", 15);

        let (cache_size_after, _, _) = optimizer.get_cache_stats();
        assert_eq!(cache_size_after, 2);
    }
}
