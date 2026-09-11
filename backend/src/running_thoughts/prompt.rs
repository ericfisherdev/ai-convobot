//! Pure prompt assembly for running thoughts (#216): no `Database`, no
//! model, so this module is unit-tested with plain fixtures. Production
//! callers build a [`ThoughtInputs`] through `running_thoughts::hook`
//! (the host's live-round reader is `hook::thought_inputs_on`) and hand it
//! to [`build_thought_prompt`]; `llm.rs::assemble_prompt` calls
//! [`render_reply_block`] to splice a companion's recent notes into its own
//! reply prompt.

use crate::context_manager::ContextManager;
use crate::database::Message;
use crate::llm::PromptSpeakers;
use crate::participants::{expand_placeholders, render_mentions, ParticipantId};
use crate::running_thoughts::types::RunningThought;

/// Previous notes shown to the thought generator.
pub const THOUGHT_CHAIN_LENGTH: usize = 6;
/// Output cap for one note: a short paragraph, so the added latency is a
/// fraction of the reply's.
pub const THOUGHT_MAX_TOKENS: usize = 96;
/// Ceiling on the messages a single round can carry into the thought
/// prompt: the first note after enabling the flag on a long chat must not
/// render the whole history.
pub const THOUGHT_ROUND_MAX_MESSAGES: usize = 12;
/// Newest notes carried into the reply prompt before budget trimming.
pub const REPLY_THOUGHTS_LIMIT: usize = 8;
/// Share of `ContextManager::compaction_token_budget` the reply-prompt
/// thoughts block may take. The single knob for #214's open budget
/// question; oldest notes are trimmed first.
pub const REPLY_THOUGHTS_BUDGET_SHARE: f32 = 0.25;

/// Everything the thought generator needs for one companion's note about
/// the round that just closed. Built by `TurnStore::thought_inputs`
/// (production: `running_thoughts::hook::thought_inputs_on`), consumed by
/// `running_thoughts::generate::generate_thought`.
#[derive(Debug, Clone, PartialEq)]
pub struct ThoughtInputs {
    pub companion_id: i32,
    pub speaker_id: ParticipantId,
    /// The previous notes, oldest first: the chain (#214) the new one
    /// continues. At most [`THOUGHT_CHAIN_LENGTH`].
    pub previous: Vec<RunningThought>,
    /// The round's messages, oldest first, ending with the user's new turn.
    pub round: Vec<Message>,
    pub from_message_id: i32,
    pub through_message_id: i32,
}

/// A thought-generation prompt split into a system part and a user part, so
/// `llm.rs` can render it through the GGUF chat template exactly as
/// `run_extraction` does (system + one `user` turn).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThoughtPrompt {
    pub system: String,
    pub user: String,
}

/// Builds the prompt for one thought-generation call: the companion's
/// persona and instructions as the system part, the previous notes plus the
/// round that just closed as the user part.
pub fn build_thought_prompt(
    inputs: &ThoughtInputs,
    persona: &str,
    speakers: &PromptSpeakers,
) -> ThoughtPrompt {
    let self_name = speakers.self_name();
    let user_name = speakers.user_name();
    let system = format!(
        "You are {self_name}. {persona}\n\
         Write one short, first-person note in {self_name}'s own voice, recording what \
         {self_name} took from the exchange below — including anything {self_name} now \
         believes or suspects, even if it was never said outright. Write it as a private \
         thought, in plain first-person prose.",
        self_name = self_name,
        persona = expand_placeholders(persona, &speakers.registry),
    );

    let mut user = String::from("Your previous notes:\n");
    if inputs.previous.is_empty() {
        user.push_str("none yet\n");
    } else {
        for (index, note) in inputs.previous.iter().enumerate() {
            let n = index + 1;
            if note.edited {
                user.push_str(&format!(
                    "{n}. (in your own words, as {user_name} corrected it) {text}\n",
                    n = n,
                    user_name = user_name,
                    text = note.text,
                ));
            } else {
                user.push_str(&format!("{n}. {text}\n", n = n, text = note.text));
            }
        }
    }

    user.push_str("\nWhat just happened:\n");
    for message in &inputs.round {
        let display_name = ParticipantId::parse(&message.speaker_id)
            .ok()
            .and_then(|id| speakers.registry.display_name(&id).map(str::to_string))
            .unwrap_or_else(|| message.speaker_id.clone());
        let text = render_mentions(&message.content, &speakers.registry);
        user.push_str(&format!("{display_name}: {text}\n"));
    }
    user.push_str("\nYour note:");

    ThoughtPrompt { system, user }
}

/// The block spliced into the reply prompt, carrying a companion's recent
/// running thoughts. Empty input yields an empty string — what keeps the
/// disabled path (an empty `thoughts` slice) byte-identical.
///
/// Drops the oldest note, one at a time, until the rendered block fits
/// `budget_tokens`; always keeps at least the newest note, even if that
/// alone is over budget.
pub fn render_reply_block(
    thoughts: &[RunningThought],
    self_name: &str,
    user_name: &str,
    budget_tokens: usize,
) -> String {
    if thoughts.is_empty() {
        return String::new();
    }

    let header = format!(
        "{self_name}'s private thoughts so far (what {self_name} actually believes; \
         {self_name} may say something different to {user_name} only when deliberately \
         deceiving them):\n"
    );

    let mut kept = thoughts;
    loop {
        let block = render_thought_block(&header, kept);
        if kept.len() <= 1 || ContextManager::estimate_tokens(&block) <= budget_tokens {
            return block;
        }
        kept = &kept[1..];
    }
}

fn render_thought_block(header: &str, notes: &[RunningThought]) -> String {
    let mut block = header.to_string();
    for note in notes {
        block.push_str(&format!("- {}\n", note.text));
    }
    block
}

/// Cleans one raw thought-generation completion: trims surrounding
/// whitespace, cuts at the first blank line (the model sometimes continues
/// past its one note into a second paragraph or a reply), and strips a
/// leading `"{self_name}:"` self-attribution or a surrounding quote wrapper
/// the model sometimes adds. `None` for an empty result, so the caller never
/// stores a blank note.
pub fn clean_thought(raw: &str, self_name: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Cut at the first blank line.
    let cut = trimmed.split("\n\n").next().unwrap_or(trimmed).trim();

    let prefix = format!("{self_name}:");
    let without_name = cut
        .strip_prefix(prefix.as_str())
        .map(str::trim_start)
        .unwrap_or(cut);

    let unquoted = without_name.trim_matches(|c: char| matches!(c, '"' | '\'' | '“' | '”'));
    let cleaned = unquoted.trim();

    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::participants::ParticipantRegistry;

    fn speakers() -> PromptSpeakers {
        PromptSpeakers {
            registry: ParticipantRegistry::solo("Alice", "Bob", None),
            self_id: ParticipantId::CHAR,
        }
    }

    fn a_thought(text: &str, edited: bool) -> RunningThought {
        RunningThought {
            id: 1,
            companion_id: 1,
            speaker_id: ParticipantId::CHAR.to_string(),
            from_message_id: 1,
            through_message_id: 2,
            text: text.to_string(),
            edited,
            created_at: String::new(),
        }
    }

    fn a_message(speaker_id: &str, content: &str) -> Message {
        Message {
            id: 1,
            ai: speaker_id != "user",
            speaker_id: speaker_id.to_string(),
            content: content.to_string(),
            created_at: String::new(),
        }
    }

    #[test]
    fn build_thought_prompt_renders_none_yet_when_there_are_no_previous_notes() {
        let inputs = ThoughtInputs {
            companion_id: 1,
            speaker_id: ParticipantId::CHAR,
            previous: vec![],
            round: vec![a_message("user", "hi")],
            from_message_id: 1,
            through_message_id: 1,
        };

        let prompt = build_thought_prompt(&inputs, "a friendly persona", &speakers());

        assert!(prompt.user.contains("Your previous notes:\nnone yet\n"));
        assert!(prompt.system.contains("Bob"));
        assert!(prompt.system.contains("a friendly persona"));
    }

    #[test]
    fn build_thought_prompt_marks_an_edited_note_as_the_users_words_and_leaves_others_bare() {
        let inputs = ThoughtInputs {
            companion_id: 1,
            speaker_id: ParticipantId::CHAR,
            previous: vec![
                a_thought("original note", false),
                a_thought("corrected note", true),
            ],
            round: vec![],
            from_message_id: 1,
            through_message_id: 1,
        };

        let prompt = build_thought_prompt(&inputs, "persona", &speakers());

        assert!(prompt.user.contains("1. original note\n"));
        assert!(prompt
            .user
            .contains("2. (in your own words, as Alice corrected it) corrected note\n"));
    }

    #[test]
    fn build_thought_prompt_renders_every_speaker_by_display_name_with_mentions_expanded() {
        let inputs = ThoughtInputs {
            companion_id: 1,
            speaker_id: ParticipantId::CHAR,
            previous: vec![],
            round: vec![
                a_message("user", "hey @char, how are you?"),
                a_message("char", "I'm well, @user"),
            ],
            from_message_id: 1,
            through_message_id: 2,
        };

        let prompt = build_thought_prompt(&inputs, "persona", &speakers());

        assert!(prompt.user.contains("Alice: hey @Bob, how are you?\n"));
        assert!(prompt.user.contains("Bob: I'm well, @Alice\n"));
    }

    #[test]
    fn render_reply_block_on_empty_input_is_empty() {
        assert_eq!(render_reply_block(&[], "Bob", "Alice", 1000), "");
    }

    #[test]
    fn render_reply_block_trims_the_oldest_note_first_and_always_keeps_the_newest() {
        let thoughts = vec![
            a_thought(
                "first note, quite long and wordy so it costs several tokens",
                false,
            ),
            a_thought("second note", false),
            a_thought("third note", false),
        ];

        // A budget too small to fit all three, but big enough for the last
        // two once the first is dropped.
        let full = render_reply_block(&thoughts, "Bob", "Alice", 10_000);
        let trimmed = render_reply_block(&thoughts, "Bob", "Alice", 8);

        assert!(full.contains("first note"));
        assert!(!trimmed.contains("first note"));
        assert!(
            trimmed.contains("third note"),
            "the newest note is always kept"
        );
    }

    #[test]
    fn render_reply_block_keeps_the_newest_note_even_when_it_alone_is_over_budget() {
        let thoughts = vec![a_thought(
            "a very long note that on its own already exceeds any reasonable budget here",
            false,
        )];

        let block = render_reply_block(&thoughts, "Bob", "Alice", 1);

        assert!(block.contains("a very long note"));
    }

    #[test]
    fn clean_thought_on_blank_input_is_none() {
        assert_eq!(clean_thought("", "Bob"), None);
        assert_eq!(clean_thought("   ", "Bob"), None);
    }

    #[test]
    fn clean_thought_on_a_double_newline_only_input_is_none() {
        assert_eq!(clean_thought("\n\n", "Bob"), None);
    }

    #[test]
    fn clean_thought_cuts_at_the_first_blank_line() {
        assert_eq!(
            clean_thought("This is my note.\n\nUser: something else entirely", "Bob"),
            Some("This is my note.".to_string())
        );
    }

    #[test]
    fn clean_thought_strips_a_leading_name_prefix_and_quotes() {
        assert_eq!(
            clean_thought("Bob: \"I feel like today went well.\"", "Bob"),
            Some("I feel like today went well.".to_string())
        );
    }

    #[test]
    fn clean_thought_leaves_a_note_with_an_internal_colon_alone() {
        assert_eq!(
            clean_thought("She said: I should trust her more.", "Bob"),
            Some("She said: I should trust her more.".to_string())
        );
    }
}
