//! Pure, I/O-free decision of *when* compaction (#172) should queue a new
//! draft checkpoint. Nothing here touches the database or the model:
//! [`crate::compaction::hook::after_round`] is the only caller, and it is
//! what supplies the numbers this module reasons about.

use crate::compaction::types::CompactionTrigger;
use crate::database::ConfigView;
use crate::token_budget::TokenBudget;

/// Fewest uncompacted messages compaction will ever fire on when
/// `ConfigView::compact_min_messages` is unset (`0`).
pub const DEFAULT_MIN_MESSAGES: usize = 8;

/// The knobs [`should_compact`] reads, derived once per round from
/// [`ConfigView`] rather than threaded through as separate arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionConfig {
    pub threshold_tokens: usize,
    pub min_messages: usize,
}

impl CompactionConfig {
    /// `threshold_tokens` is `config.compact_threshold_tokens` when set and
    /// non-zero, else twice the pure, unexpanded
    /// `TokenBudget::from_vram_limit(..).recent_messages` budget —
    /// deliberately not `ContextManager::new`, which probes system memory.
    /// `min_messages` is `config.compact_min_messages`, falling back to
    /// [`DEFAULT_MIN_MESSAGES`] when zero.
    pub fn from_config(config: &ConfigView) -> Self {
        let threshold_tokens = match config.compact_threshold_tokens {
            Some(threshold) if threshold != 0 => threshold,
            _ => {
                2 * TokenBudget::from_vram_limit(config.vram_limit_gb, config.context_window_size)
                    .recent_messages
            }
        };
        let min_messages = if config.compact_min_messages == 0 {
            DEFAULT_MIN_MESSAGES
        } else {
            config.compact_min_messages
        };
        CompactionConfig {
            threshold_tokens,
            min_messages,
        }
    }
}

/// Phrases that mark a scene or time break in the human's latest turn.
/// A plain cue list, not a `regex`: easier to read and to extend than a
/// pattern, and every cue here is a fixed phrase anyway.
pub const SCENE_BREAK_CUES: &[&str] = &[
    "a few hours later",
    "hours later",
    "the next morning",
    "the following morning",
    "later that night",
    "the next day",
    "the following day",
    "days later",
    "weeks later",
];

/// Whether `user_turn` contains a scene-break cue, ignoring case,
/// markdown emphasis (`*`/`_`) and line breaks around it (so `*a few
/// hours later*` and a cue split across a newline still match).
pub fn is_scene_break(user_turn: &str) -> bool {
    let normalized: String = user_turn
        .to_lowercase()
        .chars()
        .map(|c| {
            if c == '*' || c == '_' || c == '\n' {
                ' '
            } else {
                c
            }
        })
        .collect();
    let collapsed = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    SCENE_BREAK_CUES.iter().any(|cue| collapsed.contains(cue))
}

/// Decides whether a round should queue a new checkpoint draft, and which
/// trigger caused it. `None` while `pending_draft` is `true`: only one
/// draft may be pending at a time.
///
/// Scene break wins over threshold when both hold, because it gives the
/// cleaner boundary — cutting right before the human's "next morning" turn
/// reads better than cutting mid-scene just because the tail also happens
/// to be long.
pub fn should_compact(
    tail_tokens: usize,
    tail_messages_before_break: usize,
    last_user_turn: &str,
    pending_draft: bool,
    config: &CompactionConfig,
) -> Option<CompactionTrigger> {
    if pending_draft {
        return None;
    }
    if is_scene_break(last_user_turn) && tail_messages_before_break >= config.min_messages {
        return Some(CompactionTrigger::SceneBreak);
    }
    if tail_tokens > config.threshold_tokens {
        return Some(CompactionTrigger::Threshold);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(threshold_tokens: usize, min_messages: usize) -> CompactionConfig {
        CompactionConfig {
            threshold_tokens,
            min_messages,
        }
    }

    #[test]
    fn below_threshold_does_not_trigger() {
        assert_eq!(should_compact(100, 10, "hi", false, &config(200, 2)), None);
    }

    #[test]
    fn above_threshold_triggers() {
        assert_eq!(
            should_compact(201, 10, "hi", false, &config(200, 2)),
            Some(CompactionTrigger::Threshold)
        );
    }

    #[test]
    fn exactly_at_threshold_does_not_trigger() {
        assert_eq!(should_compact(200, 10, "hi", false, &config(200, 2)), None);
    }

    #[test]
    fn scene_break_with_too_short_a_tail_does_not_trigger() {
        assert_eq!(
            should_compact(0, 1, "the next morning, I woke up", false, &config(200, 2)),
            None
        );
    }

    #[test]
    fn scene_break_with_long_enough_tail_triggers() {
        assert_eq!(
            should_compact(0, 2, "the next morning, I woke up", false, &config(200, 2)),
            Some(CompactionTrigger::SceneBreak)
        );
    }

    #[test]
    fn both_triggers_true_picks_scene_break() {
        assert_eq!(
            should_compact(500, 2, "the next morning", false, &config(200, 2)),
            Some(CompactionTrigger::SceneBreak)
        );
    }

    #[test]
    fn pending_draft_suppresses_the_threshold_trigger() {
        assert_eq!(should_compact(500, 10, "hi", true, &config(200, 2)), None);
    }

    #[test]
    fn pending_draft_suppresses_the_scene_break_trigger() {
        assert_eq!(
            should_compact(0, 10, "the next morning", true, &config(200, 2)),
            None
        );
    }

    #[test]
    fn is_scene_break_matches_wrapped_in_markdown_emphasis() {
        assert!(is_scene_break("*a few hours later*"));
    }

    #[test]
    fn is_scene_break_matches_capitalised_mid_sentence() {
        assert!(is_scene_break(
            "The next morning, everything felt different."
        ));
    }

    #[test]
    fn is_scene_break_matches_mid_sentence_without_capitalisation() {
        assert!(is_scene_break("we talked for hours later that evening"));
    }

    #[test]
    fn is_scene_break_rejects_an_unrelated_lateness_mention() {
        assert!(!is_scene_break("I will be late"));
    }

    #[test]
    fn from_config_uses_the_default_min_messages_when_unset() {
        let mut config = default_config_view();
        config.compact_min_messages = 0;
        assert_eq!(
            CompactionConfig::from_config(&config).min_messages,
            DEFAULT_MIN_MESSAGES
        );
    }

    #[test]
    fn from_config_derives_the_threshold_from_the_vram_budget_when_unset() {
        let mut config = default_config_view();
        config.compact_threshold_tokens = None;
        config.vram_limit_gb = 4;
        config.context_window_size = 100_000;
        let expected = 2 * TokenBudget::from_vram_limit(4, 100_000).recent_messages;
        assert_eq!(
            CompactionConfig::from_config(&config).threshold_tokens,
            expected
        );
    }

    #[test]
    fn from_config_uses_an_explicit_threshold_when_set() {
        let mut config = default_config_view();
        config.compact_threshold_tokens = Some(4096);
        assert_eq!(
            CompactionConfig::from_config(&config).threshold_tokens,
            4096
        );
    }

    /// Minimal `ConfigView` for the `from_config` tests above: only the
    /// fields `CompactionConfig::from_config` reads are exercised, the rest
    /// just need any valid value.
    fn default_config_view() -> ConfigView {
        ConfigView {
            device: crate::database::Device::CPU,
            llm_model_path: String::new(),
            gpu_layers: 0,
            prompt_template: crate::database::PromptTemplate::Auto,
            context_window_size: 2048,
            max_response_tokens: 512,
            enable_dynamic_context: true,
            vram_limit_gb: 4,
            dynamic_gpu_allocation: true,
            gpu_safety_margin: 0.8,
            min_free_vram_mb: 512,
            enable_hybrid_context: true,
            max_system_ram_usage_gb: 8,
            context_expansion_strategy: "balanced".to_string(),
            ram_safety_margin_gb: 2,
            multiplayer_mode: crate::multiplayer::config::MultiplayerMode::Solo,
            multiplayer_host_address: String::new(),
            multiplayer_participant_id: String::new(),
            mention_followup_depth: 1,
            remote_generation_timeout_secs: 120,
            multiplayer_password_set: false,
            multiplayer_password: String::new(),
            compact_threshold_tokens: None,
            compact_min_messages: DEFAULT_MIN_MESSAGES,
            compaction_model_path: None,
            heuristic_person_detection: false,
        }
    }
}
