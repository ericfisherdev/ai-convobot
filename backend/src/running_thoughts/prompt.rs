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

/// Previous notes shown to the thought generator. Two, not more: with a
/// longer chain the model paraphrased or copied its older notes instead of
/// reading the round (the 2026-09-12 eval saw the same sentence restated in
/// four consecutive notes at six).
pub const THOUGHT_CHAIN_LENGTH: usize = 2;
/// Sentences a note may run to. Length is enforced by stopping generation
/// at the sentence boundary (`count_sentences` in the decode callback);
/// asking the model for "at most three sentences" had no effect.
pub const THOUGHT_SENTENCE_LIMIT: usize = 3;
/// Backstop token cap for one note, behind the sentence stop: a note that
/// never closes a sentence still ends here, so the added latency stays a
/// fraction of the reply's. Three sentences normally finish well under it.
pub const THOUGHT_MAX_TOKENS: usize = 120;
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
/// persona, instructions and the card's example dialogue (its register) as
/// the system part; the previous notes plus the round that just closed as
/// the user part.
///
/// The instruction asks for two things in one paragraph — what the user
/// just said or did, then what the companion makes of it — and pins the
/// tense ("has not responded yet"), because a note written *before* the
/// reply otherwise narrates the pending exchange as already finished. The
/// worked example is about nobody in particular so it carries only the
/// shape; voice comes from the card's example dialogue — its spoken lines
/// only, since quoting its `*actions*` primed the model to narrate the
/// companion from the outside instead of thinking as her.
pub fn build_thought_prompt(
    inputs: &ThoughtInputs,
    persona: &str,
    example_dialogue: &str,
    speakers: &PromptSpeakers,
) -> ThoughtPrompt {
    let self_name = speakers.self_name();
    let user_name = speakers.user_name();
    let mut system = format!(
        "You are {self_name}. {persona}\n\
         Before you answer {user_name}, you think to yourself, in your own words: one \
         paragraph, at most three sentences, written as \"I\". You never call yourself \
         {self_name} in it and never describe yourself from the outside. First what \
         {user_name} just said or did, keeping strictly to what is in the exchange below \
         and nothing that has not happened yet; then what you make of it — what you \
         suspect {user_name} is after, and how you feel about {user_name} right now. \
         Guesses about {user_name}'s intentions are welcome; invented events are not. \
         Plain prose only: no asterisks, no stage directions.\n\
         An example of the shape, about somebody else entirely: \"She asked where I'd been \
         and let it go when I dodged. I think she already knows and is waiting for me to \
         say it, and I don't like being waited on.\"",
        self_name = self_name,
        user_name = user_name,
        persona = expand_placeholders(persona, &speakers.registry),
    );
    let dialogue = spoken_lines(&expand_placeholders(example_dialogue, &speakers.registry));
    if !dialogue.is_empty() {
        system.push_str("\nThis is how you talk:\n");
        system.push_str(&dialogue);
    }

    let mut user = String::new();
    if inputs.previous.is_empty() {
        user.push_str("Your earlier thoughts: none yet.\n");
    } else {
        user.push_str(
            "Your earlier thoughts, already noted (do not repeat them; only add what is new):\n",
        );
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

    user.push_str(&format!(
        "\nWhat just happened ({self_name} has not responded yet):\n"
    ));
    for message in &inputs.round {
        let display_name = ParticipantId::parse(&message.speaker_id)
            .ok()
            .and_then(|id| speakers.registry.display_name(&id).map(str::to_string))
            .unwrap_or_else(|| message.speaker_id.clone());
        let text = render_mentions(&message.content, &speakers.registry);
        user.push_str(&format!("{display_name}: {text}\n"));
    }
    user.push_str("\nYour thought:");

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

fn is_terminator(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…')
}

/// Byte offsets just past each sentence end in `text`: a run of terminators
/// (so `...` is one end, not three) plus any closing quotes/brackets that
/// immediately follow, where the run is not sandwiched between digits
/// (`3.5`) and is followed by whitespace or the end of the text. A run made
/// only of ellipsis at the very end is a pause, not an end, until the
/// whitespace after it proves the sentence closed — otherwise a streamed
/// `But...` stops generation mid-thought.
fn sentence_ends(text: &str) -> Vec<usize> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut ends = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !is_terminator(chars[i].1) {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && is_terminator(chars[i].1) {
            i += 1;
        }
        let run = &chars[start..i];
        let before_is_digit = start > 0 && chars[start - 1].1.is_ascii_digit();
        let after_is_digit = i < chars.len() && chars[i].1.is_ascii_digit();
        if before_is_digit && after_is_digit {
            continue;
        }
        // `1. ` at the very start is the list number `strip_list_number`
        // removes, not a sentence.
        let is_leading_list_number = run.len() == 1
            && run[0].1 == '.'
            && chars[..start].iter().all(|(_, c)| c.is_ascii_digit());
        if before_is_digit && is_leading_list_number {
            continue;
        }
        let run_is_ellipsis =
            run.iter().all(|(_, c)| matches!(c, '.' | '…')) && (run.len() > 1 || run[0].1 == '…');
        while i < chars.len() && is_closing_mark(chars[i].1) {
            i += 1;
        }
        let at_end = i >= chars.len();
        let at_boundary = if at_end {
            !run_is_ellipsis
        } else {
            chars[i].1.is_whitespace()
        };
        if at_boundary {
            ends.push(chars.get(i).map_or(text.len(), |(b, _)| *b));
        }
    }
    ends
}

/// Sentences closed so far in `text`. Fed the streamed completion after
/// every token by `generate_thought`, which stops at
/// [`THOUGHT_SENTENCE_LIMIT`].
pub fn count_sentences(text: &str) -> usize {
    sentence_ends(text).len()
}

/// `text` cut just after its `limit`-th sentence, or whole when it has
/// fewer. The decode loop stops one token late at best (the token that
/// closed the sentence may carry the start of the next), so this trims the
/// remainder.
pub fn truncate_to_sentences(text: &str, limit: usize) -> &str {
    let ends = sentence_ends(text);
    match limit.checked_sub(1).and_then(|i| ends.get(i)) {
        Some(&end) => &text[..end],
        None => text,
    }
}

/// The example dialogue with every `*action*` removed, line by line, and
/// any line that was only an action (or only a speaker label) dropped.
/// What the thought prompt quotes for register: how the companion speaks,
/// not how a roleplay narrates her.
fn spoken_lines(example_dialogue: &str) -> String {
    example_dialogue
        .lines()
        .map(strip_asides)
        .filter(|line| !line.is_empty() && !line.ends_with(':'))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Strips a leading `N. ` — the model sometimes continues the numbered
/// "earlier thoughts" list into its own note.
fn strip_list_number(text: &str) -> &str {
    let digits = text.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return text;
    }
    match text[digits..].strip_prefix('.') {
        Some(rest) if rest.starts_with(char::is_whitespace) => rest.trim_start(),
        _ => text,
    }
}

/// Drops every `*…*` span — the asterisk-delimited stage direction a
/// roleplay model slips into prose (`*I scribble in my notebook.*`) — and
/// collapses the whitespace it leaves behind. An unmatched `*` drops the
/// rest of the text, which is the lesser evil: a dangling action beat is
/// never part of the thought.
fn strip_asides(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_aside = false;
    for c in text.chars() {
        if c == '*' {
            in_aside = !in_aside;
            continue;
        }
        if !in_aside {
            out.push(c);
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
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
/// whitespace, cuts at the first blank line, strips `*stage directions*`
/// from the kept paragraph, strips a leading list number (`3. `) copied
/// from the numbered chain (the model sometimes continues
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

    // Asides are stripped only inside the kept paragraph: `strip_asides`
    // collapses all whitespace, so running it on `raw` would erase the
    // blank line the cut above keys on.
    let without_asides = if cut.contains('*') {
        strip_asides(cut)
    } else {
        cut.to_string()
    };
    let without_number = strip_list_number(without_asides.trim());
    let prefix = format!("{self_name}:");
    let without_name = without_number
        .strip_prefix(prefix.as_str())
        .map(str::trim_start)
        .unwrap_or(without_number);

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

        let prompt = build_thought_prompt(&inputs, "a friendly persona", "", &speakers());

        assert!(prompt.user.contains("Your earlier thoughts: none yet.\n"));
        assert!(prompt.system.contains("Bob"));
        assert!(prompt.system.contains("a friendly persona"));
        assert!(
            !prompt.system.contains("This is how you talk"),
            "an empty example dialogue adds no register block"
        );
    }

    #[test]
    fn build_thought_prompt_quotes_the_example_dialogue_as_the_companions_register() {
        let inputs = ThoughtInputs {
            companion_id: 1,
            speaker_id: ParticipantId::CHAR,
            previous: vec![],
            round: vec![a_message("user", "hi")],
            from_message_id: 1,
            through_message_id: 1,
        };

        let prompt = build_thought_prompt(
            &inputs,
            "persona",
            "{{user}}: hey\n{{char}}: what.",
            &speakers(),
        );

        assert!(prompt
            .system
            .contains("This is how you talk:\nAlice: hey\nBob: what."));
    }

    #[test]
    fn build_thought_prompt_quotes_only_the_spoken_part_of_the_example_dialogue() {
        let inputs = ThoughtInputs {
            companion_id: 1,
            speaker_id: ParticipantId::CHAR,
            previous: vec![],
            round: vec![a_message("user", "hi")],
            from_message_id: 1,
            through_message_id: 1,
        };

        let prompt = build_thought_prompt(
            &inputs,
            "persona",
            "{{user}}: You're bleeding.\n{{char}}: *doesn't look down* Wow. Real observant. *shifts* It's fine.\n{{char}}: *very still*\n{{user}}: Who?",
            &speakers(),
        );

        assert!(prompt.system.contains(
            "This is how you talk:\nAlice: You're bleeding.\nBob: Wow. Real observant. It's fine.\nAlice: Who?"
        ));
        assert!(!prompt.system.contains('*'));
    }

    #[test]
    fn build_thought_prompt_anchors_the_tense_and_asks_for_plain_prose() {
        let inputs = ThoughtInputs {
            companion_id: 1,
            speaker_id: ParticipantId::CHAR,
            previous: vec![],
            round: vec![a_message("user", "hi")],
            from_message_id: 1,
            through_message_id: 1,
        };

        let prompt = build_thought_prompt(&inputs, "persona", "", &speakers());

        assert!(prompt
            .user
            .contains("What just happened (Bob has not responded yet):"));
        assert!(prompt.system.contains("nothing that has not happened yet"));
        assert!(prompt.system.contains("no asterisks, no stage directions"));
        assert!(prompt.user.ends_with("Your thought:"));
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

        let prompt = build_thought_prompt(&inputs, "persona", "", &speakers());

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

        let prompt = build_thought_prompt(&inputs, "persona", "", &speakers());

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

    #[test]
    fn clean_thought_strips_a_leading_list_number_but_not_a_number_in_prose() {
        assert_eq!(
            clean_thought("1. He gave me a gun.", "Bob", false),
            Some("He gave me a gun.".to_string())
        );
        assert_eq!(
            clean_thought("14 basilisks. Great.", "Bob", false),
            Some("14 basilisks. Great.".to_string())
        );
        assert_eq!(
            clean_thought("3.5 miles is nothing.", "Bob", false),
            Some("3.5 miles is nothing.".to_string())
        );
    }

    #[test]
    fn clean_thought_cuts_at_the_first_blank_line_even_when_an_aside_precedes_it() {
        assert_eq!(
            clean_thought(
                "I'm glad. *smiles*\n\nBob: So how was your day?",
                "Bob",
                false
            ),
            Some("I'm glad.".to_string())
        );
    }

    #[test]
    fn count_sentences_does_not_count_a_leading_list_number() {
        assert_eq!(count_sentences("1. He gave me a gun. Really. Truly."), 3);
        assert_eq!(count_sentences("14 basilisks. Great."), 2);
        assert_eq!(count_sentences("He gave me 3. Really."), 2);
    }

    #[test]
    fn clean_thought_strips_stage_directions() {
        assert_eq!(
            clean_thought(
                "He's got nerve. *I scribble in my notebook.* But still.",
                "Bob",
                false
            ),
            Some("He's got nerve. But still.".to_string())
        );
        assert_eq!(clean_thought("*shrugs*", "Bob", false), None);
    }

    #[test]
    fn count_sentences_counts_plain_sentences_and_a_terminator_at_the_end() {
        assert_eq!(count_sentences("He said fine. I think so! Really?"), 3);
        assert_eq!(count_sentences("First. Second. Third."), 3);
        assert_eq!(count_sentences("First. Second. Thi"), 2);
    }

    #[test]
    fn count_sentences_does_not_split_decimals_or_ellipsis_runs() {
        assert_eq!(count_sentences("It's 3.5 miles. Wait... what? No."), 4);
        assert_eq!(count_sentences("Wait... what"), 1);
    }

    #[test]
    fn count_sentences_treats_a_trailing_ellipsis_as_a_pause_until_whitespace_follows() {
        assert_eq!(count_sentences("He's got some nerve. But..."), 1);
        assert_eq!(count_sentences("He's got some nerve. But… "), 2);
        assert_eq!(count_sentences("Nope."), 1);
    }

    #[test]
    fn truncate_to_sentences_cuts_after_the_limit_and_keeps_a_closing_quote() {
        assert_eq!(truncate_to_sentences("A. B. C. D.", 3), "A. B. C.");
        assert_eq!(truncate_to_sentences("A. B.", 3), "A. B.");
        assert_eq!(
            truncate_to_sentences("He said \"fine.\" Then left. And more. Extra.", 3),
            "He said \"fine.\" Then left. And more."
        );
        assert_eq!(truncate_to_sentences("A. B.", 0), "A. B.");
    }
}
