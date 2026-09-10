//! Renders a [`CompactionContext`] into prompt-ready text (#174). Pure — no
//! I/O, no SQLite — so `llm.rs::assemble_prompt` can call it with a
//! pre-built context and this module's own tests never need a database.
//!
//! [`overlays_and_rules_fit`] is the one place the "overlays and rules are
//! never trimmed" rule is checked: [`render`] calls it too, and #173/#175
//! call it directly before they ever open a transaction, so the size rule
//! can never drift between the two call sites.

use serde::Serialize;

use crate::compaction::context::{CompactionContext, PinnedMessage, QuoteLine, QuoteSpeaker};
use crate::context_manager::ContextManager;

/// The rendered prompt fragments `llm.rs::build_base_components` splices
/// into the system portion, in the order they are meant to appear:
/// `user_overlay`, `companion_overlay`, `rules`, `story_so_far`,
/// `recent_detail`, `pins`. Every field is `""` when its section has
/// nothing to say, so an empty [`CompactionContext`] renders every field
/// empty and adds no text to a prompt that never compacted.
#[derive(Serialize, Default, PartialEq, Debug)]
pub struct RenderedBlocks {
    pub user_overlay: String,
    pub companion_overlay: String,
    pub rules: String,
    pub story_so_far: String,
    pub recent_detail: String,
    pub pins: String,
    /// `Some(n)` when `user_overlay` + `companion_overlay` + `rules` alone
    /// exceed the budget by `n` tokens. Those three blocks are never
    /// trimmed, so this is the only way the render can go over budget; a
    /// chat turn still proceeds, just larger than intended (`llm.rs` logs a
    /// warning instead of failing the turn).
    pub over_budget_by: Option<usize>,
}

/// The result of [`overlays_and_rules_fit`] finding the never-trimmed
/// blocks too large for the slice they were given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverBudget {
    pub needed: usize,
    pub budget: usize,
}

/// Whether the user overlay, companion overlay, and rules block together
/// fit inside `budget_tokens`. These three are never trimmed by [`render`],
/// so this is the one place — called by `render` itself, and directly by
/// #173's whole-draft rejection and #175's commit — that decides whether a
/// checkpoint's overlays and rules can ever be rendered inside a given
/// compaction slice.
pub fn overlays_and_rules_fit(
    ctx: &CompactionContext,
    user_name: &str,
    companion_name: &str,
    budget_tokens: usize,
) -> Result<(), OverBudget> {
    let never_trimmed = format!(
        "{}{}{}",
        render_overlay(user_name, &ctx.user_state),
        render_overlay(companion_name, &ctx.companion_state),
        render_rules(&ctx.rules, user_name, companion_name),
    );
    let needed = ContextManager::estimate_tokens(&never_trimmed);
    if needed > budget_tokens {
        Err(OverBudget {
            needed,
            budget: budget_tokens,
        })
    } else {
        Ok(())
    }
}

/// Renders `ctx` into [`RenderedBlocks`], trimming the compaction slice's
/// disposable content to fit `budget_tokens` before ever touching the
/// overlays or rules block. Trim order, each step fully exhausted before
/// the next: recalled facts (all at once) → `recent_detail` sentences from
/// the front → `story_so_far` (rolling-summary sentences from the front,
/// then Background lines oldest first, then Open-thread lines oldest
/// first) → pins block (key quotes oldest first, then pinned messages
/// oldest first). If the overlays and rules alone exceed `budget_tokens`,
/// every trimmable block is dropped and `over_budget_by` reports the
/// excess.
pub fn render(
    ctx: &CompactionContext,
    user_name: &str,
    companion_name: &str,
    budget_tokens: usize,
) -> RenderedBlocks {
    let user_overlay = render_overlay(user_name, &ctx.user_state);
    let companion_overlay = render_overlay(companion_name, &ctx.companion_state);
    let rules = render_rules(&ctx.rules, user_name, companion_name);

    match overlays_and_rules_fit(ctx, user_name, companion_name, budget_tokens) {
        Err(over) => RenderedBlocks {
            user_overlay,
            companion_overlay,
            rules,
            story_so_far: String::new(),
            recent_detail: String::new(),
            pins: String::new(),
            over_budget_by: Some(over.needed - over.budget),
        },
        Ok(()) => {
            let never_trimmed_tokens = ContextManager::estimate_tokens(&format!(
                "{user_overlay}{companion_overlay}{rules}"
            ));
            let remaining_budget = budget_tokens.saturating_sub(never_trimmed_tokens);
            let (story_so_far, recent_detail, pins) =
                trim_and_render(ctx, user_name, companion_name, remaining_budget);
            RenderedBlocks {
                user_overlay,
                companion_overlay,
                rules,
                story_so_far,
                recent_detail,
                pins,
                over_budget_by: None,
            }
        }
    }
}

/// `"{name} now (this supersedes the persona above):\n"` followed by one
/// `"- {line}\n"` per entry; `""` when `lines` is empty, so a joiner
/// context with an empty `companion_state` omits the header entirely.
fn render_overlay(name: &str, lines: &[String]) -> String {
    if lines.is_empty() {
        return String::new();
    }
    let mut out = format!("{name} now (this supersedes the persona above):\n");
    for line in lines {
        out.push_str(&format!("- {line}\n"));
    }
    out
}

fn speaker_name<'a>(speaker: QuoteSpeaker, user_name: &'a str, companion_name: &'a str) -> &'a str {
    match speaker {
        QuoteSpeaker::User => user_name,
        QuoteSpeaker::Companion => companion_name,
    }
}

/// `"Rules (verbatim):\n"` followed by one `"- {name}: \"{text}\"\n"` per
/// rule; `""` when there are no rules.
fn render_rules(rules: &[QuoteLine], user_name: &str, companion_name: &str) -> String {
    if rules.is_empty() {
        return String::new();
    }
    let mut out = String::from("Rules (verbatim):\n");
    for line in rules {
        out.push_str(&format!(
            "- {}: \"{}\"\n",
            speaker_name(line.speaker, user_name, companion_name),
            line.text
        ));
    }
    out
}

/// `"Recently: {text}\n"`; `""` once every sentence has been trimmed away
/// (or there was never any `recent_detail` to begin with).
fn render_recent_detail(sentences: &[String]) -> String {
    if sentences.is_empty() {
        return String::new();
    }
    format!("Recently: {}\n", sentences.concat())
}

/// `"Story so far: {rolling_summary}\n"`, then one `"- Background: {text}\n"`
/// per backstory line, one `"- Open thread: {text}\n"` per open thread, and
/// (the plan gives `recalled_facts` its own `"Recalled:\n- {text}\n"`
/// format but no dedicated `RenderedBlocks` field — see the module-level
/// note in `context.rs`) a trailing `"Recalled:\n"` section, one `"-
/// {text}\n"` per recalled fact. `""` only when all four inputs are empty,
/// so a companion with just a rolling summary still renders the header.
fn render_story_so_far(
    rolling_sentences: &[String],
    backstory: &[String],
    open_threads: &[String],
    recalled_facts: &[String],
) -> String {
    if rolling_sentences.is_empty()
        && backstory.is_empty()
        && open_threads.is_empty()
        && recalled_facts.is_empty()
    {
        return String::new();
    }
    let mut out = format!("Story so far: {}\n", rolling_sentences.concat());
    for line in backstory {
        out.push_str(&format!("- Background: {line}\n"));
    }
    for line in open_threads {
        out.push_str(&format!("- Open thread: {line}\n"));
    }
    if !recalled_facts.is_empty() {
        out.push_str("Recalled:\n");
        for fact in recalled_facts {
            out.push_str(&format!("- {fact}\n"));
        }
    }
    out
}

/// `"Pinned (verbatim):\n"`, one `"{name}: {content}\n"` per pinned message
/// (speaker name from `speaker_id`: `user`/`char` map to the passed names,
/// anything else renders as the raw id), then one `"{name}: \"{text}\"\n"`
/// per key quote. `""` when both are empty.
fn render_pins(
    pins: &[PinnedMessage],
    key_quotes: &[QuoteLine],
    user_name: &str,
    companion_name: &str,
) -> String {
    if pins.is_empty() && key_quotes.is_empty() {
        return String::new();
    }
    let mut out = String::from("Pinned (verbatim):\n");
    for pin in pins {
        let name = match pin.speaker_id.as_str() {
            "user" => user_name,
            "char" => companion_name,
            other => other,
        };
        out.push_str(&format!("{name}: {}\n", pin.content));
    }
    for line in key_quotes {
        out.push_str(&format!(
            "{}: \"{}\"\n",
            speaker_name(line.speaker, user_name, companion_name),
            line.text
        ));
    }
    out
}

/// Splits `text` into sentences on `.`, `!`, or `?` followed by whitespace,
/// keeping that trailing whitespace attached to the sentence it ends so
/// `sentences.concat()` always reproduces `text` exactly. The final
/// fragment (no terminator, or a terminator with nothing after it) is its
/// own trailing "sentence" so no text is ever dropped by the split itself.
fn split_sentences(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut sentences = Vec::new();
    let mut current = String::new();
    let mut chars = text.char_indices().peekable();
    while let Some((_, c)) = chars.next() {
        current.push(c);
        if c == '.' || c == '!' || c == '?' {
            match chars.peek() {
                Some(&(_, next)) if next.is_whitespace() => {
                    while let Some(&(_, ws)) = chars.peek() {
                        if ws.is_whitespace() {
                            current.push(ws);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    sentences.push(std::mem::take(&mut current));
                }
                None => sentences.push(std::mem::take(&mut current)),
                _ => {}
            }
        }
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    sentences
}

/// Trims the disposable parts of `ctx` (everything but the overlays and
/// rules, already rendered by the caller) to fit `budget`, returning the
/// three trimmable [`RenderedBlocks`] fields. See [`render`] for the trim
/// order.
fn trim_and_render(
    ctx: &CompactionContext,
    user_name: &str,
    companion_name: &str,
    budget: usize,
) -> (String, String, String) {
    let mut recalled_facts = ctx.recalled_facts.clone();
    let mut recent_sentences = split_sentences(&ctx.recent_detail);
    let mut rolling_sentences = split_sentences(&ctx.rolling_summary);
    let mut backstory = ctx.backstory.clone();
    let mut open_threads = ctx.open_threads.clone();
    let mut key_quotes = ctx.key_quotes.clone();
    let mut pins = ctx.pins.clone();

    let assemble = |recalled_facts: &[String],
                    recent_sentences: &[String],
                    rolling_sentences: &[String],
                    backstory: &[String],
                    open_threads: &[String],
                    key_quotes: &[QuoteLine],
                    pins: &[PinnedMessage]| {
        let story_so_far =
            render_story_so_far(rolling_sentences, backstory, open_threads, recalled_facts);
        let recent_detail = render_recent_detail(recent_sentences);
        let pins_block = render_pins(pins, key_quotes, user_name, companion_name);
        (story_so_far, recent_detail, pins_block)
    };
    let fits = |story_so_far: &str, recent_detail: &str, pins_block: &str| {
        ContextManager::estimate_tokens(&format!("{story_so_far}{recent_detail}{pins_block}"))
            <= budget
    };

    macro_rules! current {
        () => {
            assemble(
                &recalled_facts,
                &recent_sentences,
                &rolling_sentences,
                &backstory,
                &open_threads,
                &key_quotes,
                &pins,
            )
        };
    }

    let (mut story_so_far, mut recent_detail, mut pins_block) = current!();
    if fits(&story_so_far, &recent_detail, &pins_block) {
        return (story_so_far, recent_detail, pins_block);
    }

    // Step 1: recalled facts, all at once.
    if !recalled_facts.is_empty() {
        recalled_facts.clear();
        (story_so_far, recent_detail, pins_block) = current!();
        if fits(&story_so_far, &recent_detail, &pins_block) {
            return (story_so_far, recent_detail, pins_block);
        }
    }

    // Step 2: recent_detail sentences, from the front.
    while !recent_sentences.is_empty() {
        recent_sentences.remove(0);
        (story_so_far, recent_detail, pins_block) = current!();
        if fits(&story_so_far, &recent_detail, &pins_block) {
            return (story_so_far, recent_detail, pins_block);
        }
    }

    // Step 3: rolling-summary sentences from the front, then Background
    // lines oldest first, then Open-thread lines oldest first.
    while !rolling_sentences.is_empty() {
        rolling_sentences.remove(0);
        (story_so_far, recent_detail, pins_block) = current!();
        if fits(&story_so_far, &recent_detail, &pins_block) {
            return (story_so_far, recent_detail, pins_block);
        }
    }
    while !backstory.is_empty() {
        backstory.remove(0);
        (story_so_far, recent_detail, pins_block) = current!();
        if fits(&story_so_far, &recent_detail, &pins_block) {
            return (story_so_far, recent_detail, pins_block);
        }
    }
    while !open_threads.is_empty() {
        open_threads.remove(0);
        (story_so_far, recent_detail, pins_block) = current!();
        if fits(&story_so_far, &recent_detail, &pins_block) {
            return (story_so_far, recent_detail, pins_block);
        }
    }

    // Step 4: pins block — key quotes oldest first, then pinned messages
    // oldest first.
    while !key_quotes.is_empty() {
        key_quotes.remove(0);
        (story_so_far, recent_detail, pins_block) = current!();
        if fits(&story_so_far, &recent_detail, &pins_block) {
            return (story_so_far, recent_detail, pins_block);
        }
    }
    while !pins.is_empty() {
        pins.remove(0);
        (story_so_far, recent_detail, pins_block) = current!();
        if fits(&story_so_far, &recent_detail, &pins_block) {
            return (story_so_far, recent_detail, pins_block);
        }
    }

    current!()
}

#[cfg(test)]
mod tests {
    use super::*;

    const USER: &str = "Alice";
    const COMPANION: &str = "Ada";

    #[test]
    fn empty_context_renders_every_field_empty() {
        let ctx = CompactionContext::default();
        let blocks = render(&ctx, USER, COMPANION, 1000);
        assert_eq!(blocks, RenderedBlocks::default());
    }

    #[test]
    fn overlay_lines_render_in_fact_id_order_with_no_header_when_empty() {
        let ctx = CompactionContext {
            user_state: vec!["is tired".to_string(), "just got a promotion".to_string()],
            ..Default::default()
        };
        let blocks = render(&ctx, USER, COMPANION, 1000);
        assert_eq!(
            blocks.user_overlay,
            "Alice now (this supersedes the persona above):\n- is tired\n- just got a promotion\n"
        );
        assert_eq!(blocks.companion_overlay, "");
    }

    #[test]
    fn rules_render_with_the_correct_speaker_name() {
        let ctx = CompactionContext {
            rules: vec![QuoteLine {
                speaker: QuoteSpeaker::User,
                text: "never mention my ex".to_string(),
            }],
            ..Default::default()
        };
        let blocks = render(&ctx, USER, COMPANION, 1000);
        assert_eq!(
            blocks.rules,
            "Rules (verbatim):\n- Alice: \"never mention my ex\"\n"
        );
    }

    #[test]
    fn trim_order_drops_recalled_then_recent_detail_then_story_then_pins_never_overlays_or_rules() {
        let ctx = CompactionContext {
            user_state: vec!["overlay line".to_string()],
            rules: vec![QuoteLine {
                speaker: QuoteSpeaker::Companion,
                text: "a rule".to_string(),
            }],
            recalled_facts: vec!["a recalled fact".to_string()],
            recent_detail: "Something happened recently.".to_string(),
            rolling_summary: "A long story so far.".to_string(),
            backstory: vec!["background detail".to_string()],
            pins: vec![PinnedMessage {
                message_id: 1,
                speaker_id: "user".to_string(),
                content: "pinned text".to_string(),
            }],
            ..Default::default()
        };

        // A budget wide enough for the never-trimmed blocks but nothing
        // else: every trimmable block empties out, overlays and rules
        // survive untouched.
        let never_trimmed_tokens = ContextManager::estimate_tokens(&format!(
            "{}{}",
            render_overlay(USER, &ctx.user_state),
            render_rules(&ctx.rules, USER, COMPANION),
        ));
        let blocks = render(&ctx, USER, COMPANION, never_trimmed_tokens);

        assert_eq!(
            blocks.user_overlay,
            "Alice now (this supersedes the persona above):\n- overlay line\n"
        );
        assert_eq!(blocks.rules, "Rules (verbatim):\n- Ada: \"a rule\"\n");
        assert_eq!(blocks.story_so_far, "");
        assert_eq!(blocks.recent_detail, "");
        assert_eq!(blocks.pins, "");
        assert_eq!(blocks.over_budget_by, None);
    }

    #[test]
    fn overlays_alone_exceeding_the_budget_reports_over_budget_by_and_keeps_them_intact() {
        let ctx = CompactionContext {
            user_state: vec!["a fairly long overlay line that costs several tokens".to_string()],
            recent_detail: "this should be dropped".to_string(),
            ..Default::default()
        };

        let blocks = render(&ctx, USER, COMPANION, 1);

        assert!(blocks.over_budget_by.unwrap() > 0);
        assert!(!blocks.user_overlay.is_empty());
        assert_eq!(blocks.recent_detail, "");
        assert_eq!(blocks.story_so_far, "");
        assert_eq!(blocks.pins, "");
    }

    #[test]
    fn overlays_and_rules_fit_reports_the_exact_excess() {
        let ctx = CompactionContext {
            user_state: vec!["x".repeat(40)],
            ..Default::default()
        };
        let needed = ContextManager::estimate_tokens(&render_overlay(USER, &ctx.user_state));
        let err = overlays_and_rules_fit(&ctx, USER, COMPANION, needed - 1).unwrap_err();
        assert_eq!(err.needed, needed);
        assert_eq!(err.budget, needed - 1);
    }

    #[test]
    fn split_sentences_reconstructs_the_original_text() {
        let text = "First sentence. Second one! Third?  Trailing fragment";
        let sentences = split_sentences(text);
        assert_eq!(sentences.concat(), text);
        assert_eq!(sentences.len(), 4);
    }
}
