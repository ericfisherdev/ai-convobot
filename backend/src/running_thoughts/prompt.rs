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

/// A closing quote or bracket, in any script the thought generator might
/// write in: ASCII/curly quotes, French/CJK closing guillemets, brackets.
fn is_closing_mark(c: char) -> bool {
    matches!(c, '"' | '\'' | '”' | '’' | '»' | '」' | '』' | ')' | ']')
}

/// A closer that is unambiguous even when a space precedes it. `"` and `'`
/// are excluded: in English they open a quotation at least as often as
/// they close one, so a space followed by one of them is the start of the
/// *next* sentence (`"Fine," she said`), not a closer for the one just cut.
/// Only used for the whitespace-crossing step below; a closer immediately
/// adjacent to the terminator (no space) is unambiguous regardless of which
/// mark it is, since nothing legitimately opens a quote there.
fn is_unambiguous_closing_mark(c: char) -> bool {
    is_closing_mark(c) && !matches!(c, '"' | '\'')
}

/// Trims a capped completion back to its last sentence boundary (#235), so
/// a thought that hit `THOUGHT_MAX_TOKENS` is never stored mid-word. Keeps
/// any run of closing quotes/brackets that immediately follows the
/// terminator (`He said "fine."` stays whole; so does a stacked
/// `("fine.")`), and one separated from the terminator by a single space
/// when that mark cannot also be an opener — the French convention around
/// a closing guillemet (`ça va. »`). A space before an ASCII `"`/`'`
/// is never crossed, since that pair usually opens the next sentence
/// (`I trust her now. "Fine," she said` must cut after `now.`, not swallow
/// the opening quote). When `text` has no `. ! ? …`, it is returned
/// unchanged: an abrupt run-on beats dropping the only note the model
/// wrote, and this is the invariant `clean_thought` relies on to never
/// turn a non-empty completion into `None` because of this step.
fn trim_to_last_sentence_boundary(text: &str) -> &str {
    let Some(idx) = text.rfind(['.', '!', '?', '…']) else {
        return text;
    };
    let terminator_len = text[idx..].chars().next().map_or(1, char::len_utf8);
    let mut end = idx + terminator_len;
    loop {
        let rest = &text[end..];
        let Some(c) = rest.chars().next() else {
            break;
        };
        if is_closing_mark(c) {
            end += c.len_utf8();
            continue;
        }
        if c.is_whitespace() {
            let after_whitespace = &rest[c.len_utf8()..];
            if after_whitespace
                .chars()
                .next()
                .is_some_and(is_unambiguous_closing_mark)
            {
                end += c.len_utf8();
                continue;
            }
        }
        break;
    }
    &text[..end]
}

/// Cleans one raw thought-generation completion: trims surrounding
/// whitespace, cuts at the first blank line (the model sometimes continues
/// past its one note into a second paragraph or a reply), strips a leading
/// `"{self_name}:"` self-attribution, strips a surrounding quote wrapper the
/// model sometimes adds, then — last, so it can never re-expose a wrapper
/// quote the previous step already removed — trims a capped completion to
/// its last sentence boundary (#235; skipped when the blank-line cut
/// already kept a naturally-finished paragraph). Running the quote strip
/// before the cap trim is what lets a *closing* quote that follows the
/// sentence boundary survive: at strip time it is not yet at the string's
/// edge, so `trim_matches` leaves it alone, and the cap trim afterwards
/// keeps it as part of the kept sentence. `None` for an empty result, so
/// the caller never stores a blank note; `trim_to_last_sentence_boundary`
/// never empties a non-empty input, so that branch is reachable only by
/// inputs that were already empty before the cap trim.
pub fn clean_thought(raw: &str, self_name: &str, hit_token_cap: bool) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Cut at the first blank line.
    let cut = trimmed.split("\n\n").next().unwrap_or(trimmed).trim();
    let paragraph_finished_naturally = cut.len() != trimmed.len();

    let prefix = format!("{self_name}:");
    let without_name = cut
        .strip_prefix(prefix.as_str())
        .map(str::trim_start)
        .unwrap_or(cut);

    let unquoted = without_name.trim_matches(|c: char| matches!(c, '"' | '\'' | '“' | '”'));

    let cap_trimmed = if hit_token_cap && !paragraph_finished_naturally {
        trim_to_last_sentence_boundary(unquoted)
    } else {
        unquoted
    };
    let cleaned = cap_trimmed.trim();

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
        assert_eq!(clean_thought("", "Bob", false), None);
        assert_eq!(clean_thought("   ", "Bob", false), None);
    }

    #[test]
    fn clean_thought_on_a_double_newline_only_input_is_none() {
        assert_eq!(clean_thought("\n\n", "Bob", false), None);
    }

    #[test]
    fn clean_thought_cuts_at_the_first_blank_line() {
        assert_eq!(
            clean_thought(
                "This is my note.\n\nUser: something else entirely",
                "Bob",
                false
            ),
            Some("This is my note.".to_string())
        );
    }

    #[test]
    fn clean_thought_strips_a_leading_name_prefix_and_quotes() {
        assert_eq!(
            clean_thought("Bob: \"I feel like today went well.\"", "Bob", false),
            Some("I feel like today went well.".to_string())
        );
    }

    #[test]
    fn clean_thought_leaves_a_note_with_an_internal_colon_alone() {
        assert_eq!(
            clean_thought("She said: I should trust her more.", "Bob", false),
            Some("She said: I should trust her more.".to_string())
        );
    }

    #[test]
    fn clean_thought_trims_a_capped_completion_to_the_last_sentence_boundary() {
        assert_eq!(
            clean_thought(
                "I trust her now. She seems to be worried about",
                "Bob",
                true
            ),
            Some("I trust her now.".to_string())
        );
    }

    #[test]
    fn clean_thought_on_a_capped_completion_keeps_a_closing_quote_after_the_terminator() {
        assert_eq!(
            clean_thought("She said \"I'm fine.\" I wonder ab", "Bob", true),
            Some("She said \"I'm fine.\"".to_string())
        );
    }

    #[test]
    fn clean_thought_on_a_capped_completion_with_no_boundary_is_left_unchanged_not_dropped() {
        assert_eq!(
            clean_thought(
                "one long run-on that never ends and keeps going ab",
                "Bob",
                true
            ),
            Some("one long run-on that never ends and keeps going ab".to_string())
        );
    }

    #[test]
    fn clean_thought_on_an_uncapped_completion_with_no_terminal_punctuation_is_unchanged() {
        assert_eq!(
            clean_thought(
                "this note just trails off without punctuation",
                "Bob",
                false
            ),
            Some("this note just trails off without punctuation".to_string())
        );
    }

    #[test]
    fn clean_thought_on_an_uncapped_completion_never_trims_even_when_a_boundary_exists() {
        // Unlike the case above, this input DOES have a sentence boundary
        // partway through, so it would be trimmed if the `hit_token_cap`
        // guard were dropped or ignored. `hit_token_cap: false` must leave
        // it alone.
        assert_eq!(
            clean_thought("I trust her now. She seems to be worried", "Bob", false),
            Some("I trust her now. She seems to be worried".to_string())
        );
    }

    #[test]
    fn clean_thought_on_a_capped_completion_keeps_stacked_closing_brackets_and_quotes() {
        assert_eq!(
            clean_thought("She said (\"fine.\") and then ab", "Bob", true),
            Some("She said (\"fine.\")".to_string())
        );
    }

    #[test]
    fn clean_thought_trims_a_capped_multi_byte_completion_at_the_last_terminator() {
        assert_eq!(
            clean_thought(
                "Elle m'a dit « ça va ». Je pense qu'elle est inquiète à propos",
                "Bob",
                true
            ),
            Some("Elle m'a dit « ça va ».".to_string())
        );
    }

    #[test]
    fn clean_thought_on_a_capped_completion_keeps_a_closing_guillemet_after_a_space() {
        // French typographic convention puts a space before a closing
        // guillemet, so the terminator and the closer are not literally
        // adjacent.
        assert_eq!(
            clean_thought("Elle a dit « ça va. » Je pense ab", "Bob", true),
            Some("Elle a dit « ça va. »".to_string())
        );
    }

    #[test]
    fn clean_thought_on_a_capped_completion_does_not_cross_a_space_before_an_ascii_quote() {
        // Unlike a guillemet, a space before an ASCII quote almost always
        // opens the *next* sentence rather than closing the one just cut,
        // so the trim must stop at the terminator and leave the quote for
        // the (dropped) remainder.
        assert_eq!(
            clean_thought(
                "I trust her now. \"Fine,\" she said, but she seems ab",
                "Bob",
                true
            ),
            Some("I trust her now.".to_string())
        );
        assert_eq!(
            clean_thought("I trust her. 'Tis a fine thing ab", "Bob", true),
            Some("I trust her.".to_string())
        );
    }

    #[test]
    fn clean_thought_trims_a_capped_completion_at_a_horizontal_ellipsis() {
        assert_eq!(
            clean_thought(
                "She trailed off mid-thought… and then kept going ab",
                "Bob",
                true
            ),
            Some("She trailed off mid-thought…".to_string())
        );
    }

    #[test]
    fn clean_thought_does_not_cap_trim_a_paragraph_that_already_ended_naturally() {
        assert_eq!(
            clean_thought("Done thinking\n\nMore words that got cut ab", "Bob", true),
            Some("Done thinking".to_string())
        );
    }
}
