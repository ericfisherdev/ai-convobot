#[cfg(test)]
mod tests {
    use crate::attitude_formatter::AttitudeFormatter;
    use crate::database::*;
    use crate::inference_optimizer::*;
    use crate::message_excerpt;

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

    fn attitude_fixture() -> CompanionAttitude {
        Database::default_user_attitude(1, 1)
    }

    #[test]
    fn diff_attitudes_reports_nothing_when_unchanged() {
        let formatter = AttitudeFormatter::new();
        let attitude = attitude_fixture();

        assert!(formatter.diff_attitudes(&attitude, &attitude).is_empty());
    }

    #[test]
    fn diff_attitudes_reports_one_moved_dimension() {
        let formatter = AttitudeFormatter::new();
        let previous = attitude_fixture();
        let mut current = previous.clone();
        current.trust += 3.0;

        let deltas = formatter.diff_attitudes(&previous, &current);

        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].dimension, "trust");
        assert!((deltas[0].delta - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn diff_attitudes_ignores_sub_threshold_movement() {
        let formatter = AttitudeFormatter::new();
        let previous = attitude_fixture();
        let mut current = previous.clone();
        current.joy += 0.4;
        current.anger -= 0.9;

        assert!(formatter.diff_attitudes(&previous, &current).is_empty());
    }

    #[test]
    fn diff_attitudes_covers_dimensions_the_console_list_omitted() {
        let formatter = AttitudeFormatter::new();
        let previous = attitude_fixture();
        let mut current = previous.clone();
        current.disgust += 2.0;
        current.gratitude -= 4.0;
        current.dominance += 5.0;

        let mut dimensions: Vec<String> = formatter
            .diff_attitudes(&previous, &current)
            .into_iter()
            .map(|delta| delta.dimension)
            .collect();
        dimensions.sort();

        assert_eq!(dimensions, vec!["disgust", "dominance", "gratitude"]);
    }

    #[test]
    fn console_changes_render_signed_labels() {
        let formatter = AttitudeFormatter::new();
        let previous = attitude_fixture();
        let mut current = previous.clone();
        current.love += 2.0;
        current.anger -= 5.0;

        let rendered = formatter.format_attitude_changes_for_console(&previous, &current);

        assert!(rendered.contains("Anger -5"));
        assert!(rendered.contains("Love +2"));
    }

    fn memory_fixture(target_type: &str, attitude_delta_json: &str) -> AttitudeMemory {
        AttitudeMemory {
            id: Some(1),
            companion_id: 1,
            target_id: 1,
            target_type: target_type.to_string(),
            memory_type: "BondingMoment".to_string(),
            description: "A bonding moment occurred".to_string(),
            priority_score: 80.0,
            attitude_delta_json: attitude_delta_json.to_string(),
            impact_score: 20.0,
            message_context: "thanks for listening".to_string(),
            created_at: "2024-01-15 10:00".to_string(),
        }
    }

    #[test]
    fn evaluate_attitude_shift_ignores_negligible_movement() {
        let previous = attitude_fixture();
        let mut current = previous.clone();
        current.trust += 1.0;

        assert!(evaluate_attitude_shift(&previous, &current).is_none());
    }

    #[test]
    fn evaluate_attitude_shift_reports_a_multi_dimension_turn() {
        let previous = attitude_fixture();
        let mut current = previous.clone();
        current.trust += 5.0;
        current.joy += 5.0;
        current.respect += 5.0;
        current.anger -= 5.0;

        let draft = evaluate_attitude_shift(&previous, &current).expect("shift is significant");

        assert!(draft.impact_score > SIGNIFICANT_IMPACT_THRESHOLD);
        assert!(draft.priority_score > 0.0);
        assert!(!draft.description.is_empty());
        assert!((draft.delta.trust - 5.0).abs() < f32::EPSILON);
        assert!((draft.delta.anger + 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn evaluate_attitude_shift_scores_dimensions_the_old_delta_omitted() {
        let previous = attitude_fixture();
        let mut current = previous.clone();
        // Love alone used to score zero impact, so a turn like this could never
        // become a memory.
        current.love += 12.0;

        let draft =
            evaluate_attitude_shift(&previous, &current).expect("love shift is significant");

        assert!((draft.delta.love - 12.0).abs() < f32::EPSILON);
        assert!(draft.impact_score > SIGNIFICANT_IMPACT_THRESHOLD);
    }

    #[test]
    fn message_excerpt_truncates_multi_byte_text_on_a_char_boundary() {
        let long = "é".repeat(500);

        let excerpt = message_excerpt(&long);

        // 200 characters plus the ellipsis marker.
        assert_eq!(excerpt.chars().count(), 201);
        assert!(excerpt.ends_with('…'));
    }

    #[test]
    fn message_excerpt_collapses_whitespace_and_keeps_short_text() {
        assert_eq!(message_excerpt("hello\n  there"), "hello there");
    }

    #[test]
    fn format_attitude_memories_is_empty_without_memories() {
        let formatter = AttitudeFormatter::new();

        assert_eq!(formatter.format_attitude_memories(&[]), "");
    }

    #[test]
    fn format_attitude_memories_renders_context_and_movement() {
        let formatter = AttitudeFormatter::new();
        let delta = serde_json::json!({ "trust": 4.0, "joy": 5.0 }).to_string();

        let block = formatter.format_attitude_memories(&[memory_fixture("user", &delta)]);

        assert!(block.contains("A bonding moment occurred"));
        assert!(block.contains("when you said: \"thanks for listening\""));
        assert!(block.contains("trust +4"));
        assert!(block.contains("joy +5"));
    }

    #[test]
    fn format_attitude_memories_survives_malformed_delta_json() {
        let formatter = AttitudeFormatter::new();

        let block = formatter.format_attitude_memories(&[memory_fixture("user", "not json")]);

        assert!(block.contains("A bonding moment occurred"));
        assert!(!block.contains("moved:"));
    }

    #[test]
    fn format_attitude_memories_excludes_third_party_rows() {
        let formatter = AttitudeFormatter::new();
        let delta = serde_json::json!({ "trust": 4.0 }).to_string();

        let block = formatter.format_attitude_memories(&[memory_fixture("third_party", &delta)]);

        assert_eq!(block, "");
    }
}
