//! The pure half of the extraction pass (#173): `serde` structs mirroring
//! the model's JSON output schema, [`parse_extraction`], and
//! [`to_fact_drafts`], which maps a parsed [`ExtractionOutput`] to
//! [`FactDraft`]s. The prompt, GBNF grammar that constrains the model to
//! this exact shape, chunking, and the `Extractor`-backed runner are #185's
//! half, built on these same types.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::compaction::registry_speakers::RegistrySpeakers;
use crate::compaction::store::{CompactionStore, SqliteCompactionStore};
use crate::compaction::types::{
    Checkpoint, CompactionStatus, FactCategory, FactDraft, FactSubject,
};
use crate::compaction::validate::{overlays_fit, validate};
use crate::compaction::{CitedMessage, SpeakerInfo};
use crate::context_manager::ContextManager;
use crate::database::Database;
use crate::llm::{Extractor, ResidentExtractor};
use crate::participants::ParticipantRegistry;
use crate::turn_slot::TurnGuard;

/// The model's extraction output, one JSON object per compacted range.
/// `#[serde(deny_unknown_fields)]` so a model that drifts from the schema
/// fails parse instead of being silently accepted.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionOutput {
    pub companion_state: Vec<TextItem>,
    pub user_state: Vec<TextItem>,
    pub milestones: Vec<TextItem>,
    pub backstory: Vec<BackstoryItem>,
    pub open_threads: Vec<TextItem>,
    pub rules: Vec<QuoteItem>,
    pub people: Vec<PersonItem>,
    pub key_quotes: Vec<QuoteItem>,
    pub summary: String,
    pub attitude: AttitudeRatings,
}

/// One `companion_state`/`user_state`/`milestones`/`open_threads` entry.
/// `replaces` is only meaningful on `companion_state`/`user_state`: the ids
/// of prior overlay facts this item updates. `#[serde(default)]` since a
/// `milestones`/`open_threads` item never carries it.
#[derive(Debug, Clone, Deserialize)]
pub struct TextItem {
    pub text: String,
    pub sources: Vec<i32>,
    #[serde(default)]
    pub replaces: Vec<i64>,
}

/// One `backstory` entry: `about` picks the subject between the user and
/// the companion.
#[derive(Debug, Clone, Deserialize)]
pub struct BackstoryItem {
    pub about: Party,
    pub text: String,
    pub sources: Vec<i32>,
}

/// One `rules`/`key_quotes` entry: a verbatim quote plus who said it.
#[derive(Debug, Clone, Deserialize)]
pub struct QuoteItem {
    pub quote: String,
    pub speaker: Party,
    pub sources: Vec<i32>,
}

/// One `people` entry: a person other than the user or companion,
/// introduced in the compacted range.
#[derive(Debug, Clone, Deserialize)]
pub struct PersonItem {
    pub name: String,
    pub relation_to: Party,
    pub relation: String,
    pub sources: Vec<i32>,
}

/// Who a `backstory`/`rules`/`key_quotes`/`people` item is about or spoken
/// by. Serializes/deserializes lowercase (`"user"`/`"companion"`) to match
/// the grammar's constrained output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Party {
    User,
    Companion,
}

/// The eight `attitude_engine::AttitudeDimension` ratings the model emits
/// alongside the extracted facts, on a 0-100 scale (the grammar in #185
/// already restricts each to three digits; deserializing through
/// [`RawAttitudeRatings`] clamps anyway so a malformed value, positive or
/// negative, can never escape this module).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "RawAttitudeRatings")]
pub struct AttitudeRatings {
    pub trust: u8,
    pub love: u8,
    pub fear: u8,
    pub anger: u8,
    pub joy: u8,
    pub sorrow: u8,
    pub suspicion: u8,
    pub gratitude: u8,
}

/// Largest value an [`AttitudeRatings`] field may hold after clamping.
const MAX_ATTITUDE_RATING: i32 = 100;

/// The wire shape `AttitudeRatings` actually deserializes through: signed,
/// so an out-of-grammar value (a hallucinated `999`, or a stray `-1`) still
/// parses instead of `serde_json` rejecting it outright the way it would
/// for a `u8` field — `clamp_to_valid_range` (via `From`) is what brings it
/// back into `0..=100` before it becomes an `AttitudeRatings`.
#[derive(Debug, Deserialize)]
struct RawAttitudeRatings {
    trust: i32,
    love: i32,
    fear: i32,
    anger: i32,
    joy: i32,
    sorrow: i32,
    suspicion: i32,
    gratitude: i32,
}

/// Clamps one raw signed rating to `0..=100` before narrowing to `u8`.
fn clamp_rating(value: i32) -> u8 {
    value.clamp(0, MAX_ATTITUDE_RATING) as u8
}

impl From<RawAttitudeRatings> for AttitudeRatings {
    fn from(raw: RawAttitudeRatings) -> Self {
        AttitudeRatings {
            trust: clamp_rating(raw.trust),
            love: clamp_rating(raw.love),
            fear: clamp_rating(raw.fear),
            anger: clamp_rating(raw.anger),
            joy: clamp_rating(raw.joy),
            sorrow: clamp_rating(raw.sorrow),
            suspicion: clamp_rating(raw.suspicion),
            gratitude: clamp_rating(raw.gratitude),
        }
    }
}

/// Parses one model completion into an [`ExtractionOutput`]. Every
/// [`AttitudeRatings`] field is already clamped to `0..=100` by the time
/// this returns — see [`RawAttitudeRatings`].
pub fn parse_extraction(raw: &str) -> Result<ExtractionOutput, serde_json::Error> {
    serde_json::from_str(raw)
}

/// Maps a `Party` to the [`FactSubject`] it stands for. `Party` can only
/// ever be `User`/`Companion`, so this never produces `FactSubject::Person`.
fn subject_of(party: Party) -> FactSubject {
    match party {
        Party::User => FactSubject::User,
        Party::Companion => FactSubject::Companion,
    }
}

/// Maps one [`ExtractionOutput`] to the [`FactDraft`]s `validate::validate`
/// will run the canon rule and the rest over. Preserves each array's order
/// and item count; `validate` is what may reject an item, never this
/// function. `output.summary` and `output.attitude` go on the checkpoint
/// row, not into facts.
pub fn to_fact_drafts(output: &ExtractionOutput) -> Vec<FactDraft> {
    let mut drafts = Vec::new();

    for item in &output.companion_state {
        drafts.push(text_item_draft(
            item,
            FactCategory::CompanionState,
            Some(FactSubject::Companion),
        ));
    }
    for item in &output.user_state {
        drafts.push(text_item_draft(
            item,
            FactCategory::UserState,
            Some(FactSubject::User),
        ));
    }
    for item in &output.milestones {
        drafts.push(text_item_draft(item, FactCategory::Milestone, None));
    }
    for item in &output.backstory {
        drafts.push(FactDraft {
            category: FactCategory::Backstory,
            subject: Some(subject_of(item.about)),
            text: item.text.clone(),
            quote_speaker: None,
            sources: item.sources.clone(),
            replaces: Vec::new(),
            relation_to: None,
            relation: None,
            canon: false,
            rejected_reason: None,
        });
    }
    for item in &output.open_threads {
        drafts.push(text_item_draft(item, FactCategory::OpenThread, None));
    }
    for item in &output.rules {
        drafts.push(quote_item_draft(item, FactCategory::Rule));
    }
    for item in &output.people {
        drafts.push(FactDraft {
            category: FactCategory::Person,
            subject: Some(FactSubject::Person(item.name.clone())),
            // Carries the name, not just `relation`: `check_duplicate`
            // keys on `(category, normalised text)` alone, so two distinct
            // people sharing a relation string (e.g. two different "a
            // neighbor"s) would otherwise collide as the same duplicate.
            text: format!("{}: {}", item.name, item.relation),
            quote_speaker: None,
            sources: item.sources.clone(),
            replaces: Vec::new(),
            relation_to: Some(subject_of(item.relation_to)),
            relation: Some(item.relation.clone()),
            canon: false,
            rejected_reason: None,
        });
    }
    for item in &output.key_quotes {
        drafts.push(quote_item_draft(item, FactCategory::KeyQuote));
    }

    drafts
}

/// Shared by every `TextItem`-shaped array (`companion_state`, `user_state`,
/// `milestones`, `open_threads`): only `subject` and `replaces` differ, and
/// `replaces` is only ever non-empty for the two state categories, since
/// only `TextItem` carries it.
fn text_item_draft(
    item: &TextItem,
    category: FactCategory,
    subject: Option<FactSubject>,
) -> FactDraft {
    FactDraft {
        category,
        subject,
        text: item.text.clone(),
        quote_speaker: None,
        sources: item.sources.clone(),
        replaces: item.replaces.clone(),
        relation_to: None,
        relation: None,
        canon: false,
        rejected_reason: None,
    }
}

/// Shared by `rules` and `key_quotes`: both are a verbatim `QuoteItem`, only
/// the category differs.
fn quote_item_draft(item: &QuoteItem, category: FactCategory) -> FactDraft {
    FactDraft {
        category,
        subject: None,
        text: item.quote.clone(),
        quote_speaker: Some(item.speaker_token().to_string()),
        sources: item.sources.clone(),
        replaces: Vec::new(),
        relation_to: None,
        relation: None,
        canon: false,
        rejected_reason: None,
    }
}

impl QuoteItem {
    /// `quote_speaker` is stored as the same lowercase token every other
    /// speaker-shaped column uses (`"user"`/`"companion"`), not `Party`'s
    /// `Debug` form.
    fn speaker_token(&self) -> &'static str {
        match self.speaker {
            Party::User => "user",
            Party::Companion => "companion",
        }
    }
}

/// One instruct completion's token budget for a single extraction chunk,
/// generous enough for up to twelve items per array ([`EXTRACTION_GRAMMAR`]'s
/// own bound) without asking a small local model for more than it can
/// produce in one call.
const EXTRACTION_MAX_TOKENS: usize = 1536;

/// Token budget for the summary-merge pass over already-summarised chunks
/// ([`SUMMARY_GRAMMAR`]'s single `summary` string is short by construction).
const SUMMARY_MAX_TOKENS: usize = 300;

/// Tokens reserved off the top of an extractor's context window before
/// [`chunk_range`] sizes a chunk against what is left: headroom for the
/// model's own chat-template wrapping and the completion itself.
pub const CONTEXT_RESERVE_TOKENS: usize = 1024;

/// What the extraction prompt renders as "already known" before the
/// transcript: prior `companion_state`/`user_state` overlay facts (`[F<id>]
/// text`, so the model can cite them in a new item's `replaces`) and the
/// prose `rolling_summary` from the latest committed checkpoint, if any.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PriorNotes {
    pub user_overlay: Vec<(i64, String)>,
    pub companion_overlay: Vec<(i64, String)>,
    pub rolling_summary: String,
}

/// Renders one transcript line for the extraction prompt: `[#id] (canon)
/// Name: text` when `speakers.is_canon` says the speaker counts as canon,
/// `[#id] Name: text` otherwise. `content`'s newlines are flattened to
/// spaces first, so one message is always one line.
pub fn render_range_line(m: &CitedMessage, speakers: &dyn SpeakerInfo) -> String {
    let content = m.content.replace('\n', " ");
    let name = speakers.display_name(&m.speaker_id);
    if speakers.is_canon(&m.speaker_id) {
        format!("[#{}] (canon) {}: {}", m.id, name, content)
    } else {
        format!("[#{}] {}: {}", m.id, name, content)
    }
}

/// The exact JSON shape [`EXTRACTION_GRAMMAR`] constrains the model to,
/// spelled out for the model rather than left implicit.
const SCHEMA_SKELETON: &str = r#"{
  "companion_state": [{"text": "...", "sources": [id, ...], "replaces": [fact_id, ...]}],
  "user_state": [{"text": "...", "sources": [id, ...], "replaces": [fact_id, ...]}],
  "milestones": [{"text": "...", "sources": [id, ...]}],
  "backstory": [{"about": "user"|"companion", "text": "...", "sources": [id, ...]}],
  "open_threads": [{"text": "...", "sources": [id, ...]}],
  "rules": [{"quote": "...", "speaker": "user"|"companion", "sources": [id, ...]}],
  "people": [{"name": "...", "relation_to": "user"|"companion", "relation": "...", "sources": [id, ...]}],
  "key_quotes": [{"quote": "...", "speaker": "user"|"companion", "sources": [id, ...]}],
  "summary": "...",
  "attitude": {"trust": 0-100, "love": 0-100, "fear": 0-100, "anger": 0-100, "joy": 0-100, "sorrow": 0-100, "suspicion": 0-100, "gratitude": 0-100}
}"#;

/// Builds the full extraction prompt for one range of messages: who the two
/// participants are and the canon rule, the "already known" overlay/rolling
/// summary block, the rendered transcript, then the JSON schema the model
/// must fill in. Takes no token budget: [`chunk_range`] is what decides how
/// much of `range` fits in one call, this only renders whatever it is given.
pub fn build_extraction_prompt(
    prior: &PriorNotes,
    range: &[CitedMessage],
    speakers: &dyn SpeakerInfo,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(
        "You are extracting durable facts from a conversation between two participants.\n",
    );
    prompt.push_str(
        "Lines marked (canon) were said by the real human user; every other line is the \
         companion's (or a third party's) in-character speech and may not be literally true.\n",
    );
    prompt.push_str(
        "Cite [#id] for every item you produce. Do not invent anything; if you are unsure, omit it.\n",
    );
    prompt.push_str("Keep each item under 25 words.\n");
    prompt.push_str(
        "When a companion_state or user_state item updates one of the previous notes below, \
         list that note's [F..] id(s) in the item's \"replaces\" array.\n\n",
    );

    prompt.push_str("Previous notes (already known, update or keep, do not repeat unchanged):\n");
    if prior.companion_overlay.is_empty()
        && prior.user_overlay.is_empty()
        && prior.rolling_summary.is_empty()
    {
        prompt.push_str("none\n");
    } else {
        for (id, text) in &prior.companion_overlay {
            prompt.push_str(&format!("[F{id}] {text}\n"));
        }
        for (id, text) in &prior.user_overlay {
            prompt.push_str(&format!("[F{id}] {text}\n"));
        }
        if !prior.rolling_summary.is_empty() {
            prompt.push_str(&format!("Rolling summary: {}\n", prior.rolling_summary));
        }
    }
    prompt.push('\n');

    prompt.push_str("Transcript to process:\n");
    for message in range {
        prompt.push_str(&render_range_line(message, speakers));
        prompt.push('\n');
    }
    prompt.push('\n');

    prompt.push_str("Produce JSON with exactly these keys:\n");
    prompt.push_str(SCHEMA_SKELETON);
    prompt
}

/// GBNF grammar constraining the model's extraction output to exactly
/// [`ExtractionOutput`]'s shape: fixed key order, every key required, arrays
/// bounded to 12 items so a runaway model cannot fill the context, strings
/// bounded to 400 characters. The rule is literally named `root`, as
/// `LlamaSampler::grammar` requires. `companion_state`/`user_state` items may
/// carry an optional trailing `"replaces"` array of prior fact ids; every
/// other item type has no such key.
///
/// Every rule definition here is a single physical line (`root`/`attitude`
/// reference named per-field sub-rules rather than wrapping), because
/// llama.cpp's C grammar parser only treats a bare newline as insignificant
/// while inside an unclosed `(...)` group (or right after `::=`/`|`) —
/// otherwise, a newline ends the current rule, and the next line is then
/// parsed as if it must start a new `name ::=` definition. A rule split
/// across lines like the pre-#207 version of `root` therefore fails with
/// `expecting name at ...` against a real model, even though every unit test
/// here (which never hands this string to llama.cpp) passes. `mod tests`'s
/// `// --- #207` section checks this constraint on every grammar constant
/// below via `check_gbnf_rule_boundaries`, without a GGUF; this same
/// module's own `extracts_from_a_real_gguf` test exercises this exact
/// constant against a real GGUF when one is available.
pub const EXTRACTION_GRAMMAR: &str = r#"root ::= "{" ws companion-state-field ws "," ws user-state-field ws "," ws milestones-field ws "," ws backstory-field ws "," ws open-threads-field ws "," ws rules-field ws "," ws people-field ws "," ws key-quotes-field ws "," ws summary-field ws "," ws attitude-field ws "}"

companion-state-field ::= "\"companion_state\"" ws ":" ws state-array
user-state-field ::= "\"user_state\"" ws ":" ws state-array
milestones-field ::= "\"milestones\"" ws ":" ws text-array
backstory-field ::= "\"backstory\"" ws ":" ws backstory-array
open-threads-field ::= "\"open_threads\"" ws ":" ws text-array
rules-field ::= "\"rules\"" ws ":" ws quote-array
people-field ::= "\"people\"" ws ":" ws person-array
key-quotes-field ::= "\"key_quotes\"" ws ":" ws quote-array
summary-field ::= "\"summary\"" ws ":" ws string
attitude-field ::= "\"attitude\"" ws ":" ws attitude

state-array ::= "[" ws (state-item (ws "," ws state-item){0,11})? ws "]"
text-array ::= "[" ws (text-item (ws "," ws text-item){0,11})? ws "]"
backstory-array ::= "[" ws (backstory-item (ws "," ws backstory-item){0,11})? ws "]"
quote-array ::= "[" ws (quote-item (ws "," ws quote-item){0,11})? ws "]"
person-array ::= "[" ws (person-item (ws "," ws person-item){0,11})? ws "]"

state-item ::= "{" ws "\"text\"" ws ":" ws string ws "," ws "\"sources\"" ws ":" ws sources (ws "," ws "\"replaces\"" ws ":" ws fact-ids)? ws "}"
text-item ::= "{" ws "\"text\"" ws ":" ws string ws "," ws "\"sources\"" ws ":" ws sources ws "}"
backstory-item ::= "{" ws "\"about\"" ws ":" ws party ws "," ws "\"text\"" ws ":" ws string ws "," ws "\"sources\"" ws ":" ws sources ws "}"
quote-item ::= "{" ws "\"quote\"" ws ":" ws string ws "," ws "\"speaker\"" ws ":" ws party ws "," ws "\"sources\"" ws ":" ws sources ws "}"
person-item ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"relation_to\"" ws ":" ws party ws "," ws "\"relation\"" ws ":" ws string ws "," ws "\"sources\"" ws ":" ws sources ws "}"

attitude ::= "{" ws trust-field ws "," ws love-field ws "," ws fear-field ws "," ws anger-field ws "," ws joy-field ws "," ws sorrow-field ws "," ws suspicion-field ws "," ws gratitude-field ws "}"

trust-field ::= "\"trust\"" ws ":" ws rating
love-field ::= "\"love\"" ws ":" ws rating
fear-field ::= "\"fear\"" ws ":" ws rating
anger-field ::= "\"anger\"" ws ":" ws rating
joy-field ::= "\"joy\"" ws ":" ws rating
sorrow-field ::= "\"sorrow\"" ws ":" ws rating
suspicion-field ::= "\"suspicion\"" ws ":" ws rating
gratitude-field ::= "\"gratitude\"" ws ":" ws rating

fact-ids ::= "[" ws (int (ws "," ws int){0,7})? ws "]"
sources ::= "[" ws int (ws "," ws int){0,7} ws "]"
int ::= [0-9]{1,7}
rating ::= [0-9] | [1-9] [0-9] | "100"
party ::= "\"user\"" | "\"companion\""
string ::= "\"" char{1,400} "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \n\t]{0,20}
"#;

/// GBNF grammar for the single-key re-summarise pass over already-summarised
/// chunk outputs ([`fill_draft`]'s multi-chunk path): `{"summary": "..."}`.
pub const SUMMARY_GRAMMAR: &str = r#"root ::= "{" ws "\"summary\"" ws ":" ws string ws "}"
string ::= "\"" char{1,400} "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \n\t]{0,20}
"#;

/// Model-free GBNF syntax guard (#207): `llama-cpp-2` exposes no way to
/// parse a grammar without a loaded `LlamaModel` (`LlamaSampler::grammar`
/// takes `&LlamaModel`; the crate's own `src/grammar/` parser is not wired
/// into `lib.rs` and is unreachable outside its own crate), so this
/// reimplements the one constraint of llama.cpp's C parser
/// (`llama-grammar.cpp`'s `parse_sequence`) that actually broke
/// `EXTRACTION_GRAMMAR`: at paren depth 0, a term is followed by
/// `parse_space(pos, is_nested)` with `is_nested = false`, which does not
/// treat `\n` as insignificant whitespace — so a bare newline outside any
/// `(...)` group ends the current rule's production, and the parser then
/// requires the next line to start a new `name ::=` definition. This is not
/// a full GBNF grammar checker; it only catches that one failure mode, since
/// that is the one nothing else in CI catches (every extraction test here
/// uses `FakeExtractor` and never hands this text to a real parser).
///
/// Returns `Err` describing the offending line on the first bare newline
/// found at depth 0 that isn't immediately followed (modulo whitespace and
/// `#` comment lines) by a new `name ::=` rule or the end of the grammar.
#[cfg(test)]
fn check_gbnf_rule_boundaries(gbnf: &str) -> Result<(), String> {
    fn rule_starts_or_grammar_ends(rest: &str) -> bool {
        let mut s = rest;
        loop {
            s = s.trim_start_matches([' ', '\t', '\r', '\n']);
            match s.strip_prefix('#') {
                Some(after_hash) => {
                    s = match after_hash.find('\n') {
                        Some(idx) => &after_hash[idx + 1..],
                        None => "",
                    };
                }
                None => break,
            }
        }
        if s.is_empty() {
            return true;
        }
        let name_end = s
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
            .unwrap_or(s.len());
        name_end > 0
            && s[name_end..]
                .trim_start_matches([' ', '\t'])
                .starts_with("::=")
    }

    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut in_char_class = false;
    let mut escaped = false;
    let mut line_no = 1usize;
    let mut skip_to_eol = false;
    // Tracks whether the most recent depth-0 non-whitespace token was `::=`
    // or `|`: `llama-grammar.cpp`'s `parse_rule` (line 675, immediately
    // after consuming `::=`) and `parse_alternates` (line 443, immediately
    // after consuming `|`) both call `parse_space(pos, /*newline_ok=*/true)`
    // unconditionally, regardless of nesting — so a newline right there is
    // always legal, unlike every other depth-0 position. Whitespace and `#`
    // comments never clear this; any other real token does.
    let mut colon_run = 0u8;
    let mut after_free_newline_token = false;
    for (byte_idx, ch) in gbnf.char_indices() {
        if skip_to_eol {
            if ch == '\n' {
                skip_to_eol = false;
            } else {
                continue;
            }
        }
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string || in_char_class => escaped = true,
            '"' if !in_char_class => in_string = !in_string,
            '[' if !in_string => in_char_class = true,
            ']' if !in_string => in_char_class = false,
            '#' if !in_string && !in_char_class => skip_to_eol = true,
            '(' if !in_string && !in_char_class => depth += 1,
            ')' if !in_string && !in_char_class => depth -= 1,
            '\n' if depth == 0
                && !in_string
                && !in_char_class
                && !after_free_newline_token
                && !rule_starts_or_grammar_ends(&gbnf[byte_idx + 1..]) =>
            {
                return Err(format!(
                    "line {line_no}: bare newline at paren depth 0, not right after \
                     `::=` or `|`, and outside a rule boundary; llama.cpp's grammar \
                     parser ends the rule here and then fails to parse the next line \
                     as a new `name ::=` definition"
                ));
            }
            _ => {}
        }

        if !in_string && !in_char_class && depth == 0 && !skip_to_eol {
            match ch {
                ':' => {
                    colon_run = (colon_run + 1).min(2);
                    after_free_newline_token = false;
                }
                '=' if colon_run == 2 => {
                    after_free_newline_token = true;
                    colon_run = 0;
                }
                '|' => {
                    after_free_newline_token = true;
                    colon_run = 0;
                }
                ' ' | '\t' | '\r' | '\n' | '#' => {
                    // Whitespace and comments never cancel a pending `::=`/`|`.
                }
                _ => {
                    after_free_newline_token = false;
                    colon_run = 0;
                }
            }
        }

        if ch == '\n' {
            line_no += 1;
        }
    }
    Ok(())
}

/// Splits `range` into chunks that each fit `context_window -
/// CONTEXT_RESERVE_TOKENS - scaffold_tokens`, greedily at message
/// boundaries, using [`ContextManager::estimate_tokens`] over each rendered
/// [`render_range_line`]. A single message that alone exceeds the budget
/// still becomes its own (oversized) chunk rather than being split
/// mid-message or dropped; the model truncates and the validator still runs
/// over whatever it produced. `scaffold_tokens` is the token cost of
/// everything in the prompt besides the transcript itself, typically
/// `estimate_tokens(&build_extraction_prompt(prior, &[], speakers))`.
pub fn chunk_range<'a>(
    range: &'a [CitedMessage],
    speakers: &dyn SpeakerInfo,
    scaffold_tokens: usize,
    context_window: usize,
) -> Vec<&'a [CitedMessage]> {
    let budget = context_window
        .saturating_sub(CONTEXT_RESERVE_TOKENS)
        .saturating_sub(scaffold_tokens);

    let mut chunks = Vec::new();
    let mut start = 0;
    let mut running_tokens = 0usize;
    for (i, message) in range.iter().enumerate() {
        let line_tokens = ContextManager::estimate_tokens(&render_range_line(message, speakers));
        if i > start && running_tokens + line_tokens > budget {
            chunks.push(&range[start..i]);
            start = i;
            running_tokens = 0;
        }
        running_tokens += line_tokens;
    }
    if start < range.len() {
        chunks.push(&range[start..]);
    }
    chunks
}

/// Merges every chunk's [`ExtractionOutput`] into one: concatenates each
/// array in chunk order, keeps the last chunk's `attitude` (the most recent
/// read on how the companion feels), and joins every chunk's `summary` with
/// a single space (`fill_draft` re-summarises the joined text into one
/// narrative when there was more than one chunk).
///
/// Panics if `chunks` is empty; every caller only ever calls this over
/// [`chunk_range`]'s output for a non-empty range, which always yields at
/// least one chunk.
pub fn merge_outputs(chunks: Vec<ExtractionOutput>) -> ExtractionOutput {
    let mut chunks = chunks.into_iter();
    let mut merged = chunks
        .next()
        .expect("merge_outputs requires at least one chunk");
    let mut summaries = vec![std::mem::take(&mut merged.summary)];

    for chunk in chunks {
        merged.companion_state.extend(chunk.companion_state);
        merged.user_state.extend(chunk.user_state);
        merged.milestones.extend(chunk.milestones);
        merged.backstory.extend(chunk.backstory);
        merged.open_threads.extend(chunk.open_threads);
        merged.rules.extend(chunk.rules);
        merged.people.extend(chunk.people);
        merged.key_quotes.extend(chunk.key_quotes);
        summaries.push(chunk.summary);
        merged.attitude = chunk.attitude;
    }

    merged.summary = summaries.join(" ");
    merged
}

/// Why [`fill_draft`] could not fill a draft.
#[derive(Debug)]
pub enum DraftError {
    /// `draft.status` was not [`CompactionStatus::Draft`], or
    /// `raw_model_output` was already set: a stale or double-dispatched job
    /// must never overwrite a reviewed draft.
    DraftNotPending(i64),
    /// A [`CompactionStore`] call failed.
    Store(rusqlite::Error),
    /// The [`Extractor`] itself failed (model load, decode, ...).
    Model(std::io::Error),
    /// Two extraction attempts in a row failed to parse as
    /// [`ExtractionOutput`]. The draft is flipped to
    /// [`CompactionStatus::Discarded`] before this is returned.
    Unparseable { first: String, second: String },
    /// The accepted `companion_state`/`user_state`/`rule` items alone would
    /// not fit `overlay_budget_tokens`. The draft is flipped to
    /// [`CompactionStatus::Discarded`] before this is returned.
    OverlayBudget { needed: usize, budget: usize },
    /// `range` was empty — every message it covered was deleted or edited
    /// out from under a pending draft (#181), so there is nothing left to
    /// extract. The draft is flipped to [`CompactionStatus::Discarded`]
    /// before this is returned, the same terminal state every other
    /// unrecoverable path here leaves it in.
    EmptyRange(i64),
}

impl fmt::Display for DraftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DraftError::DraftNotPending(id) => {
                write!(f, "checkpoint {id} is not a pending draft")
            }
            DraftError::Store(e) => write!(f, "compaction store error: {e}"),
            DraftError::Model(e) => write!(f, "extraction model error: {e}"),
            DraftError::Unparseable { .. } => {
                write!(f, "extraction output did not parse after one retry")
            }
            DraftError::OverlayBudget { needed, budget } => write!(
                f,
                "accepted overlay/rule items need {needed} tokens, over the {budget}-token budget"
            ),
            DraftError::EmptyRange(id) => {
                write!(f, "checkpoint {id}'s range is empty, nothing to extract")
            }
        }
    }
}

/// One chunk's extraction attempt, including the raw text that parsed (the
/// second attempt's, when a retry was needed) so [`fill_draft`] can record
/// exactly what the model produced.
enum ChunkOutcome {
    Parsed {
        output: ExtractionOutput,
        raw: String,
    },
    Unparseable {
        first: String,
        second: String,
    },
}

/// Runs `extractor.extract` over `prompt` with [`EXTRACTION_GRAMMAR`],
/// retrying once (with the serde error appended to the prompt) if the first
/// attempt does not parse as [`ExtractionOutput`]. Never retries a second
/// time: a model that fails grammar-constrained JSON twice in a row is not
/// going to succeed on a third attempt either.
fn extract_chunk(extractor: &impl Extractor, prompt: &str) -> Result<ChunkOutcome, std::io::Error> {
    let first_raw = extractor.extract(prompt, EXTRACTION_GRAMMAR, EXTRACTION_MAX_TOKENS)?;
    match parse_extraction(&first_raw) {
        Ok(output) => Ok(ChunkOutcome::Parsed {
            output,
            raw: first_raw,
        }),
        Err(parse_error) => {
            let retry_prompt = format!(
                "{prompt}\nYour previous output was not valid JSON: {parse_error}. Produce the JSON again."
            );
            let second_raw =
                extractor.extract(&retry_prompt, EXTRACTION_GRAMMAR, EXTRACTION_MAX_TOKENS)?;
            match parse_extraction(&second_raw) {
                Ok(output) => Ok(ChunkOutcome::Parsed {
                    output,
                    raw: second_raw,
                }),
                Err(_) => Ok(ChunkOutcome::Unparseable {
                    first: first_raw,
                    second: second_raw,
                }),
            }
        }
    }
}

/// Combines already-summarised chunk summaries into one narrative via
/// [`SUMMARY_GRAMMAR`], kept constrained here (unlike #175's own merge pass,
/// which uses `complete`) so the result is guaranteed to be a single string.
fn resummarise(extractor: &impl Extractor, joined_summary: &str) -> Result<String, std::io::Error> {
    let prompt = format!("Combine these into one 3-6 sentence narrative:\n{joined_summary}");
    let raw = extractor.extract(&prompt, SUMMARY_GRAMMAR, SUMMARY_MAX_TOKENS)?;

    #[derive(Deserialize)]
    struct SummaryOnly {
        summary: String,
    }
    let parsed: SummaryOnly = serde_json::from_str(&raw)
        .map_err(|e| std::io::Error::other(format!("summary merge output did not parse: {e}")))?;
    Ok(parsed.summary)
}

/// Builds [`PriorNotes`] from `store`'s active overlay facts and the latest
/// committed checkpoint's `rolling_summary`.
fn build_prior_notes(
    store: &impl CompactionStore,
    companion_id: i32,
) -> Result<PriorNotes, DraftError> {
    let active = store
        .active_facts(companion_id)
        .map_err(DraftError::Store)?;
    let user_overlay = active
        .iter()
        .filter(|fact| fact.category == FactCategory::UserState)
        .map(|fact| (fact.id, fact.text.clone()))
        .collect();
    let companion_overlay = active
        .iter()
        .filter(|fact| fact.category == FactCategory::CompanionState)
        .map(|fact| (fact.id, fact.text.clone()))
        .collect();
    let rolling_summary = store
        .latest_committed(companion_id)
        .map_err(DraftError::Store)?
        .and_then(|checkpoint| checkpoint.rolling_summary)
        .unwrap_or_default();

    Ok(PriorNotes {
        user_overlay,
        companion_overlay,
        rolling_summary,
    })
}

/// Discards `draft_id`, recording `raw` as its `raw_model_output` (`None`
/// when there was nothing to run the model over, e.g. an empty range) with
/// no summary/attitude. Shared by every one of [`fill_draft`]'s failure
/// paths (empty range, unparseable output, overlay budget exceeded), so a
/// discarded draft is always left in the same shape for #175's commit and
/// #181's stale-retirement logic to reason about.
fn discard_draft(
    store: &impl CompactionStore,
    draft_id: i64,
    raw: Option<String>,
    reason: &str,
) -> Result<(), DraftError> {
    store
        .set_extraction_result(draft_id, raw, None, None)
        .map_err(DraftError::Store)?;
    store
        .update_status(draft_id, CompactionStatus::Discarded)
        .map_err(DraftError::Store)?;
    eprintln!("compaction: draft {draft_id} discarded, {reason}");
    Ok(())
}

/// Fills a pre-inserted `Draft`-status checkpoint row: builds the prompt(s),
/// runs the model through `extractor`, validates the result, and either
/// stores the accepted/rejected facts (status stays `Draft`) or discards the
/// checkpoint. Pure with respect to HTTP and locks — callers
/// ([`spawn_extraction`], #182's joiner-side job) supply the row, the
/// already-selected range, and the speaker policy.
pub fn fill_draft(
    store: &impl CompactionStore,
    extractor: &impl Extractor,
    draft: &Checkpoint,
    range: &[CitedMessage],
    speakers: &dyn SpeakerInfo,
    overlay_budget_tokens: usize,
) -> Result<(), DraftError> {
    if draft.status != CompactionStatus::Draft || draft.raw_model_output.is_some() {
        return Err(DraftError::DraftNotPending(draft.id));
    }
    if range.is_empty() {
        discard_draft(
            store,
            draft.id,
            None,
            "range is empty (its messages were likely deleted or edited out from under it)",
        )?;
        return Err(DraftError::EmptyRange(draft.id));
    }

    let prior = build_prior_notes(store, draft.companion_id)?;
    let scaffold_tokens =
        ContextManager::estimate_tokens(&build_extraction_prompt(&prior, &[], speakers));
    let chunks = chunk_range(range, speakers, scaffold_tokens, extractor.context_window());

    let mut outputs = Vec::with_capacity(chunks.len());
    let mut raw_outputs = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let prompt = build_extraction_prompt(&prior, chunk, speakers);
        match extract_chunk(extractor, &prompt).map_err(DraftError::Model)? {
            ChunkOutcome::Parsed { output, raw } => {
                outputs.push(output);
                raw_outputs.push(raw);
            }
            ChunkOutcome::Unparseable { first, second } => {
                discard_draft(
                    store,
                    draft.id,
                    Some(format!("{first}\n---\n{second}")),
                    "extraction output was unparseable twice",
                )?;
                return Err(DraftError::Unparseable { first, second });
            }
        }
    }

    let was_chunked = outputs.len() > 1;
    let mut merged = merge_outputs(outputs);
    if was_chunked {
        merged.summary = resummarise(extractor, &merged.summary).map_err(DraftError::Model)?;
    }

    let drafts = to_fact_drafts(&merged);
    let active = store
        .active_facts(draft.companion_id)
        .map_err(DraftError::Store)?;
    let is_canon = |speaker_id: &str| speakers.is_canon(speaker_id);
    let validated = validate(drafts, range, &active, &is_canon);

    if let Err(needed) = overlays_fit(&validated, overlay_budget_tokens) {
        discard_draft(
            store,
            draft.id,
            Some(raw_outputs.join("\n---\n")),
            &format!(
                "overlay/rule items need {needed} tokens, over the {overlay_budget_tokens}-token budget"
            ),
        )?;
        return Err(DraftError::OverlayBudget {
            needed,
            budget: overlay_budget_tokens,
        });
    }

    let attitude_json =
        serde_json::to_string(&merged.attitude).expect("AttitudeRatings always serializes");
    store
        .set_extraction_result(
            draft.id,
            Some(raw_outputs.join("\n---\n")),
            Some(merged.summary.clone()),
            Some(attitude_json),
        )
        .map_err(DraftError::Store)?;
    store
        .insert_facts(draft.id, &validated)
        .map_err(DraftError::Store)?;

    Ok(())
}

/// Moves an already-claimed [`TurnGuard`] into a spawned thread running
/// `job`, releasing the guard when the thread ends — including when `job`
/// panics, since `TurnGuard::drop` runs regardless. Callers claim the slot
/// themselves before calling this, so there is no `Busy` variant here and no
/// window between "turn finished" and "extraction started" in which another
/// chat turn could slip in.
pub fn spawn_holding(
    guard: TurnGuard,
    job: impl FnOnce() + Send + 'static,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let _guard = guard;
        job();
    })
}

/// The single production entry point for running extraction on a queued
/// draft: loads the row and the range it covers, builds the canon-aware
/// speaker policy from `registry` (#182's `RegistrySpeakers` — equivalent
/// to the old hard-coded `SoloSpeakers` in solo mode, since a solo registry
/// is just `user` then `char`), and calls [`fill_draft`] against the
/// production [`SqliteCompactionStore`]/[`ResidentExtractor`]. Errors are
/// logged, like every other background job in this codebase, but (#208)
/// never merely swallowed once a draft row is known: see
/// [`run_extraction_job`]/[`fail_pending_draft`].
///
/// `registry` is a snapshot (`Clone`, owned, `Send`), not a live handle: the
/// caller takes it from the same shared registry a round already snapshots
/// (`main.rs`'s `snapshot_speakers(&registry).registry`), so a joined or
/// disconnected bot mid-extraction cannot mutate the policy this thread is
/// running against.
///
/// The overlay budget is a flat 15% of the chat model's token budget total
/// until #174 lands its own `compaction` slice on `TokenBudget` for this to
/// read instead.
pub fn spawn_extraction(
    guard: TurnGuard,
    draft_id: i64,
    registry: ParticipantRegistry,
) -> std::thread::JoinHandle<()> {
    spawn_holding(guard, move || {
        if let Err(e) = run_extraction_job(draft_id, registry) {
            eprintln!("compaction: extraction job for draft {draft_id} failed: {e}");
        }
    })
}

/// Loads `draft_id` and runs extraction over it via [`run_extraction`]. A
/// missing or unreadable row (before we even know a draft exists to
/// transition) just ends the thread, same as before #208 — there is
/// nothing to flip a status on. Once the row *is* known, every failure
/// [`run_extraction`] returns *or panics with* goes through
/// [`fail_pending_draft`] before this function returns it, so a failed
/// extraction always leaves the checkpoint in an explicit terminal state
/// instead of stuck in `Draft`/`extracting` forever (the bug #208 fixes:
/// the trigger in `hook.rs` treats any pending `Draft` row as a reason not
/// to queue another one).
///
/// The `catch_unwind` is load-bearing, not defensive boilerplate:
/// `llama-cpp-2`'s `LlamaModel::load_from_file` panics (does not return
/// `Err`) when the configured GGUF path does not exist, confirmed against
/// this binary directly while building this fix. Without it, that one
/// panic would unwind straight out of this function -- `spawn_holding`'s
/// own doc comment already documents that the *turn slot* survives a
/// panicking job (`TurnGuard::drop` runs regardless), but nothing downstream
/// of that ever ran `fail_pending_draft`, so the checkpoint stayed wedged
/// in `Draft` exactly as before this fix. `AssertUnwindSafe` is safe here:
/// every value the closure captures (`store`'s reference to a unit struct,
/// `draft`'s owned plain-data snapshot, `registry`'s owned snapshot) is
/// left untouched by an aborted call on this thread, and `Database::open()`
/// inside `run_extraction` starts a fresh connection per call regardless.
fn run_extraction_job(draft_id: i64, registry: ParticipantRegistry) -> Result<(), String> {
    let store = SqliteCompactionStore;
    let draft = load_draft_for_extraction(&store, draft_id)?;

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_extraction(&store, &draft, registry)
    }))
    .unwrap_or_else(|payload| Err(panic_message(payload.as_ref())));

    if let Err(err) = &outcome {
        fail_pending_draft(&store, draft.id, err);
    }
    outcome
}

/// Loads `draft_id`, split out from [`run_extraction_job`] so its one
/// review-caught edge case is unit-testable against [`RecordingStore`]
/// rather than only reachable through the hardwired `Database`/
/// `SqliteCompactionStore` globals `run_extraction_job` itself is glued to.
///
/// `Ok(None)` (`draft_id` genuinely does not exist) has nothing to
/// transition and returns its error as-is, same as before #208. A `Err`
/// read (#208 review, CodeRabbit) is different: `fail_draft` opens its own
/// store connection and only needs `draft_id`, so it can still succeed even
/// though *this* read failed -- a transient error here (lock contention, a
/// blip) must not rule out the write right after it. Skipping that would
/// leave exactly the wedge this issue exists to close, just triggered one
/// step earlier than every other path in [`run_extraction_job`].
fn load_draft_for_extraction(
    store: &impl CompactionStore,
    draft_id: i64,
) -> Result<Checkpoint, String> {
    match store.get_checkpoint(draft_id) {
        Ok(Some(draft)) => Ok(draft),
        Ok(None) => Err(format!("draft {draft_id} not found")),
        Err(e) => {
            let error = format!("failed to load draft {draft_id}: {e}");
            fail_pending_draft(store, draft_id, &error);
            Err(error)
        }
    }
}

/// Turns a caught panic payload into a readable message: `&str`/`String`
/// cover every panic macro (`panic!`, `.unwrap()`, `.expect()`, and the
/// bare `&str` `llama-cpp-2` itself panics with — see
/// [`run_extraction_job`]'s doc comment), falling back to a fixed message
/// for the rare payload that is neither.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = panic_payload_text(payload) {
        format!("extraction panicked: {message}")
    } else {
        "extraction panicked with a non-string payload".to_string()
    }
}

/// The `&str`/`String` payload every `panic!`/`.unwrap()`/`.expect()` call
/// produces, with a defensive fallback that unwraps one extra layer of
/// `Box<dyn Any + Send>` if the payload was itself re-boxed (e.g. a
/// `resume_unwind(Box::new(inner))` re-raise elsewhere in the call stack).
///
/// That extra layer is *not* what `catch_unwind` normally hands back here,
/// and earlier revisions of this comment wrongly blamed it on the
/// actix/tokio worker machinery: the real cause, caught in review, was this
/// module's own call site passing `&payload` (`payload: Box<dyn Any +
/// Send>`) into [`panic_message`] instead of `payload.as_ref()`. `&payload`
/// unsizes the *`Box` itself* into the trait object -- `Box<dyn Any +
/// Send>` is `Sized` and blanket-implements `Any`, so rustc's unsize
/// coercion wins over deref coercion at that call site, for every panic,
/// not just `llama-cpp-2`'s -- so `downcast_ref::<Box<dyn Any + Send>>()`
/// matched first regardless of what was actually boxed. `run_extraction_job`
/// now calls `panic_message(payload.as_ref())`, which derefs through the
/// box first and lands on the real payload directly, so this recursive arm
/// is not the shape production hits; it stays as a fallback in case
/// something upstream ever does re-box a payload before it reaches here.
fn panic_payload_text(payload: &(dyn std::any::Any + Send)) -> Option<String> {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return Some((*s).to_string());
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return Some(s.clone());
    }
    if let Some(inner) = payload.downcast_ref::<Box<dyn std::any::Any + Send>>() {
        return panic_payload_text(inner.as_ref());
    }
    None
}

/// The extraction pipeline for an already-loaded `draft` row: builds the
/// range and speaker policy, then calls [`fill_draft`]. Split out from
/// [`run_extraction_job`] so every failure after the row is known —
/// `Database::get_messages_between`, `Database::get_config`, and whatever
/// `fill_draft` itself returns — funnels through this one `Result<(), String>`
/// for [`run_extraction_job`] to hand to [`fail_pending_draft`].
fn run_extraction(
    store: &impl CompactionStore,
    draft: &Checkpoint,
    registry: ParticipantRegistry,
) -> Result<(), String> {
    let speakers = RegistrySpeakers(registry);

    let messages = Database::get_messages_between(draft.from_message_id, draft.through_message_id)
        .map_err(|e| e.to_string())?;
    let range: Vec<CitedMessage> = messages.iter().map(CitedMessage::from).collect();

    let config = Database::get_config().map_err(|e| e.to_string())?;
    // #174 introduces a dedicated `compaction` slice on `TokenBudget`; until
    // it lands, reserve a flat 15% of the chat model's total token budget
    // for the overlay/rule items this draft's facts will render into.
    let overlay_budget_tokens = ContextManager::new(config).token_budget.total * 15 / 100;

    fill_draft(
        store,
        &ResidentExtractor,
        draft,
        &range,
        &speakers,
        overlay_budget_tokens,
    )
    .map_err(|e| e.to_string())
}

/// Gives a failed extraction an explicit terminal state (#208): flips
/// `draft_id` from `Draft` to `Failed` via [`CompactionStore::fail_draft`],
/// recording `error` so `GET /api/compaction/{id}` can explain it. Uses the
/// same conditional-update pattern #175's `transition_status` established
/// (`UPDATE ... WHERE id = ? AND status = ?`, `changes() == 0` read as a
/// lost race), so this can never clobber a status `fill_draft` or a
/// concurrent commit/discard already moved on. That covers two distinct
/// "already handled" cases, both surfacing as `QueryReturnedNoRows` here:
/// `fill_draft`'s own `discard_draft` already flipped the row to
/// `Discarded` for a content reason (empty range, unparseable output twice,
/// an over-budget overlay) before returning its error, and an unrelated
/// commit/discard raced this thread from the API. Either way the row is
/// already terminal, so this is a no-op, not an error — logged at a level
/// that says so rather than reads as a new failure.
fn fail_pending_draft(store: &impl CompactionStore, draft_id: i64, error: &str) {
    match store.fail_draft(draft_id, error) {
        Ok(()) => {
            eprintln!("compaction: draft {draft_id} failed extraction: {error}");
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            eprintln!(
                "compaction: draft {draft_id} failed extraction ({error}), but had already left Draft status (its own discard, or a concurrent commit/discard) -- nothing to do"
            );
        }
        Err(e) => {
            eprintln!(
                "compaction: draft {draft_id} failed extraction ({error}), and recording that failure itself failed: {e}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::fixtures::{bad_draft, synthetic_range};
    use crate::compaction::store::RecordingStore;
    use crate::compaction::types::{CompactionTrigger, Fact, NewDraft};
    use crate::compaction::SoloSpeakers;
    use crate::llm::FakeExtractor;

    #[test]
    fn bad_draft_fixture_parses() {
        let output = bad_draft();
        assert!(!output.rules.is_empty());
    }

    #[test]
    fn an_extra_top_level_key_fails_to_parse() {
        let raw = r#"{
            "companion_state": [], "user_state": [], "milestones": [],
            "backstory": [], "open_threads": [], "rules": [], "people": [],
            "key_quotes": [], "summary": "s",
            "attitude": {"trust":0,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0},
            "extra_field": "not in the schema"
        }"#;
        assert!(parse_extraction(raw).is_err());
    }

    #[test]
    fn a_rating_of_250_clamps_to_100() {
        let raw = r#"{
            "companion_state": [], "user_state": [], "milestones": [],
            "backstory": [], "open_threads": [], "rules": [], "people": [],
            "key_quotes": [], "summary": "s",
            "attitude": {"trust":250,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0}
        }"#;
        let output = parse_extraction(raw).unwrap();
        assert_eq!(output.attitude.trust, 100);
    }

    #[test]
    fn a_rating_of_999_parses_and_clamps_to_100_instead_of_failing_to_deserialize() {
        // 999 does not fit in a `u8` (max 255); deserializing straight into
        // a `u8` field would make `serde_json` reject it before clamping
        // ever ran. `RawAttitudeRatings` is signed precisely so this parses.
        let raw = r#"{
            "companion_state": [], "user_state": [], "milestones": [],
            "backstory": [], "open_threads": [], "rules": [], "people": [],
            "key_quotes": [], "summary": "s",
            "attitude": {"trust":999,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0}
        }"#;
        let output = parse_extraction(raw).unwrap();
        assert_eq!(output.attitude.trust, 100);
    }

    #[test]
    fn a_negative_rating_parses_and_clamps_to_0() {
        let raw = r#"{
            "companion_state": [], "user_state": [], "milestones": [],
            "backstory": [], "open_threads": [], "rules": [], "people": [],
            "key_quotes": [], "summary": "s",
            "attitude": {"trust":-1,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0}
        }"#;
        let output = parse_extraction(raw).unwrap();
        assert_eq!(output.attitude.trust, 0);
    }

    #[test]
    fn to_fact_drafts_preserves_order_and_item_count() {
        let output = bad_draft();
        let total_items = output.companion_state.len()
            + output.user_state.len()
            + output.milestones.len()
            + output.backstory.len()
            + output.open_threads.len()
            + output.rules.len()
            + output.people.len()
            + output.key_quotes.len();
        let drafts = to_fact_drafts(&output);
        assert_eq!(drafts.len(), total_items);
    }

    #[test]
    fn a_people_item_maps_to_a_person_draft_with_relation_to_relation_and_no_quote_speaker() {
        let raw = r#"{
            "companion_state": [], "user_state": [], "milestones": [],
            "backstory": [], "open_threads": [],
            "rules": [], "key_quotes": [],
            "people": [{"name":"Ann","relation_to":"user","relation":"sister","sources":[46]}],
            "summary": "s",
            "attitude": {"trust":0,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0}
        }"#;
        let output = parse_extraction(raw).unwrap();
        let drafts = to_fact_drafts(&output);
        assert_eq!(drafts.len(), 1);
        let person = &drafts[0];
        assert_eq!(person.category, FactCategory::Person);
        assert_eq!(person.relation_to, Some(FactSubject::User));
        assert_eq!(person.relation.as_deref(), Some("sister"));
        assert_eq!(person.quote_speaker, None);
        assert_eq!(person.text, "Ann: sister");
    }

    #[test]
    fn two_people_sharing_a_relation_string_get_distinct_text_from_their_names() {
        let raw = r#"{
            "companion_state": [], "user_state": [], "milestones": [],
            "backstory": [], "open_threads": [],
            "rules": [], "key_quotes": [],
            "people": [
                {"name":"Ann","relation_to":"user","relation":"a neighbor","sources":[46]},
                {"name":"Bo","relation_to":"user","relation":"a neighbor","sources":[46]}
            ],
            "summary": "s",
            "attitude": {"trust":0,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0}
        }"#;
        let output = parse_extraction(raw).unwrap();
        let drafts = to_fact_drafts(&output);
        assert_ne!(drafts[0].text, drafts[1].text);
    }

    #[test]
    fn a_companion_state_item_with_replaces_maps_through_while_a_milestone_leaves_it_empty_and_has_no_relation_to(
    ) {
        let raw = r#"{
            "companion_state": [{"text":"is happier now","sources":[46],"replaces":[3,4]}],
            "user_state": [], "milestones": [{"text":"left home","sources":[47]}],
            "backstory": [], "open_threads": [], "rules": [], "people": [],
            "key_quotes": [], "summary": "s",
            "attitude": {"trust":0,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0}
        }"#;
        let output = parse_extraction(raw).unwrap();
        let drafts = to_fact_drafts(&output);
        assert_eq!(drafts[0].replaces, vec![3, 4]);
        assert!(drafts[1].replaces.is_empty());
        assert_eq!(drafts[1].relation_to, None);
    }

    // --- render_range_line / build_extraction_prompt ---

    fn solo_speakers() -> SoloSpeakers {
        SoloSpeakers {
            user_name: "Eric".to_string(),
            companion_name: "Vi".to_string(),
        }
    }

    #[test]
    fn render_range_line_marks_canon_only_for_the_user_and_flattens_newlines() {
        let speakers = solo_speakers();
        let user_line = render_range_line(
            &CitedMessage {
                id: 46,
                speaker_id: "user".to_string(),
                content: "hello\nthere".to_string(),
            },
            &speakers,
        );
        assert_eq!(user_line, "[#46] (canon) Eric: hello there");

        let char_line = render_range_line(
            &CitedMessage {
                id: 47,
                speaker_id: "char".to_string(),
                content: "hi".to_string(),
            },
            &speakers,
        );
        assert_eq!(char_line, "[#47] Vi: hi");
    }

    #[test]
    fn build_extraction_prompt_marks_only_canon_lines_and_cites_every_id_and_schema_key() {
        let range = synthetic_range();
        let speakers = solo_speakers();
        let prompt = build_extraction_prompt(&PriorNotes::default(), &range, &speakers);

        for message in &range {
            assert!(
                prompt.contains(&format!("[#{}]", message.id)),
                "prompt is missing message id {}",
                message.id
            );
            let marked_canon = prompt.contains(&format!("[#{}] (canon)", message.id));
            assert_eq!(marked_canon, speakers.is_canon(&message.speaker_id));
        }

        for key in [
            "companion_state",
            "user_state",
            "milestones",
            "backstory",
            "open_threads",
            "rules",
            "people",
            "key_quotes",
            "summary",
            "attitude",
        ] {
            assert!(prompt.contains(key), "prompt is missing schema key {key}");
        }
    }

    #[test]
    fn build_extraction_prompt_renders_previous_notes_as_none_when_empty_and_lists_overlay_ids_otherwise(
    ) {
        let speakers = solo_speakers();
        let empty_prompt = build_extraction_prompt(&PriorNotes::default(), &[], &speakers);
        assert!(empty_prompt.contains("none"));

        let prior = PriorNotes {
            user_overlay: vec![(12, "feels at ease".to_string())],
            companion_overlay: vec![(7, "is cautious".to_string())],
            rolling_summary: "They moved into a lighthouse.".to_string(),
        };
        let prompt = build_extraction_prompt(&prior, &[], &speakers);
        assert!(prompt.contains("[F12] feels at ease"));
        assert!(prompt.contains("[F7] is cautious"));
        assert!(prompt.contains("They moved into a lighthouse."));
    }

    // --- EXTRACTION_GRAMMAR / SUMMARY_GRAMMAR ---

    #[test]
    fn extraction_grammar_contains_the_root_rule_and_no_nul_bytes() {
        assert!(EXTRACTION_GRAMMAR.contains("root ::="));
        assert!(!EXTRACTION_GRAMMAR.contains('\0'));
    }

    #[test]
    fn summary_grammar_contains_the_root_rule_and_no_nul_bytes() {
        assert!(SUMMARY_GRAMMAR.contains("root ::="));
        assert!(!SUMMARY_GRAMMAR.contains('\0'));
    }

    #[test]
    fn a_hand_written_sample_in_the_grammars_shape_round_trips_through_parse_extraction() {
        let sample = r#"{
            "companion_state": [{"text": "is warmer toward the user", "sources": [54], "replaces": [3]}],
            "user_state": [{"text": "feels at home", "sources": [62]}],
            "milestones": [{"text": "first night settled in", "sources": [62]}],
            "backstory": [{"about": "user", "text": "grew up near Millbrook", "sources": [46]}],
            "open_threads": [{"text": "whether Rina will visit", "sources": [64]}],
            "rules": [{"quote": "I promise I will never lie to you, no matter what happens.", "speaker": "user", "sources": [53]}],
            "people": [{"name": "Wren", "relation_to": "companion", "relation": "a neighbor", "sources": [52]}],
            "key_quotes": [{"quote": "The old lighthouse keeper's ghost still walks these halls every midnight.", "speaker": "companion", "sources": [55]}],
            "summary": "A quiet night in the lighthouse.",
            "attitude": {"trust": 80, "love": 60, "fear": 10, "anger": 0, "joy": 70, "sorrow": 5, "suspicion": 15, "gratitude": 55}
        }"#;
        let output = parse_extraction(sample).expect("shape the grammar produces should parse");
        assert_eq!(output.companion_state[0].replaces, vec![3]);
    }

    // --- chunk_range ---

    #[test]
    fn chunk_range_returns_one_chunk_when_everything_fits() {
        let range = synthetic_range();
        let speakers = solo_speakers();

        let chunks = chunk_range(&range, &speakers, 0, EXTRACTOR_MAX_CONTEXT_FOR_TESTS);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), range.len());
    }

    /// Stand-in for `llm::EXTRACTOR_MAX_CONTEXT`, kept local since this test
    /// only needs "generously larger than the synthetic range", not the
    /// crate's real extractor constant.
    const EXTRACTOR_MAX_CONTEXT_FOR_TESTS: usize = 8192;

    #[test]
    fn chunk_range_splits_a_20_message_range_at_boundaries_without_exceeding_the_budget() {
        let range = synthetic_range();
        let speakers = solo_speakers();
        let scaffold_tokens = 0;
        // Small enough that most messages cannot share a chunk.
        let context_window = CONTEXT_RESERVE_TOKENS + 40;
        let budget = context_window - CONTEXT_RESERVE_TOKENS - scaffold_tokens;

        let chunks = chunk_range(&range, &speakers, scaffold_tokens, context_window);

        assert!(chunks.len() > 1);
        let total_messages: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total_messages, range.len());
        for chunk in &chunks {
            let chunk_tokens: usize = chunk
                .iter()
                .map(|m| ContextManager::estimate_tokens(&render_range_line(m, &speakers)))
                .sum();
            // A chunk of exactly one message is allowed to exceed the
            // budget (an oversized single message becomes its own chunk).
            assert!(chunk_tokens <= budget || chunk.len() == 1);
        }
    }

    // --- merge_outputs ---

    fn an_output(text: &str, summary: &str, rating: i32) -> ExtractionOutput {
        let raw = format!(
            r#"{{
                "companion_state": [{{"text": "{text}", "sources": [1], "replaces": []}}],
                "user_state": [], "milestones": [], "backstory": [], "open_threads": [],
                "rules": [], "people": [], "key_quotes": [],
                "summary": "{summary}",
                "attitude": {{"trust": {rating}, "love": {rating}, "fear": {rating}, "anger": {rating}, "joy": {rating}, "sorrow": {rating}, "suspicion": {rating}, "gratitude": {rating}}}
            }}"#
        );
        parse_extraction(&raw).unwrap()
    }

    #[test]
    fn merge_outputs_concatenates_arrays_in_order_and_keeps_the_last_chunks_attitude() {
        let merged = merge_outputs(vec![
            an_output("a", "first", 10),
            an_output("b", "second", 90),
        ]);

        assert_eq!(merged.companion_state.len(), 2);
        assert_eq!(merged.companion_state[0].text, "a");
        assert_eq!(merged.companion_state[1].text, "b");
        assert_eq!(merged.summary, "first second");
        assert_eq!(merged.attitude.trust, 90);
    }

    // --- fill_draft ---

    fn a_pending_draft(store: &RecordingStore, range: &[CitedMessage]) -> Checkpoint {
        let draft_id = store
            .insert_draft(NewDraft {
                companion_id: 1,
                from_message_id: range.first().unwrap().id,
                through_message_id: range.last().unwrap().id,
                trigger: CompactionTrigger::Threshold,
                raw_model_output: None,
            })
            .unwrap();
        store.get_checkpoint(draft_id).unwrap().unwrap()
    }

    #[test]
    fn fill_draft_fills_a_pending_draft_with_every_item_and_leaves_status_draft() {
        let store = RecordingStore::new();
        let range = synthetic_range();
        let draft = a_pending_draft(&store, &range);
        let speakers = solo_speakers();
        let extractor =
            FakeExtractor::returning(vec![
                Ok(include_str!("fixtures/bad_draft.json").to_string()),
            ]);

        fill_draft(&store, &extractor, &draft, &range, &speakers, usize::MAX)
            .expect("fill_draft should succeed");

        let updated = store.get_checkpoint(draft.id).unwrap().unwrap();
        assert_eq!(updated.status, CompactionStatus::Draft);
        assert!(updated.raw_model_output.is_some());
        assert!(updated.summary.is_some());

        let expected_item_count = to_fact_drafts(&bad_draft()).len();
        let facts = store.facts_for(draft.id).unwrap();
        assert_eq!(facts.len(), expected_item_count);
        assert!(facts.iter().any(|f| f.rejected_reason.is_some()));
        assert!(facts.iter().any(|f| f.rejected_reason.is_none()));
    }

    #[test]
    fn a_successful_first_attempt_never_calls_extract_a_second_time() {
        let store = RecordingStore::new();
        let range = synthetic_range();
        let draft = a_pending_draft(&store, &range);
        let speakers = solo_speakers();
        let extractor =
            FakeExtractor::returning(vec![
                Ok(include_str!("fixtures/bad_draft.json").to_string()),
            ]);

        fill_draft(&store, &extractor, &draft, &range, &speakers, usize::MAX)
            .expect("fill_draft should succeed");

        assert_eq!(extractor.prompts.lock().unwrap().len(), 1);
    }

    #[test]
    fn two_unparseable_attempts_discard_the_draft_and_the_retry_prompt_names_the_parse_error() {
        let store = RecordingStore::new();
        let range = synthetic_range();
        let draft = a_pending_draft(&store, &range);
        let speakers = solo_speakers();
        let extractor = FakeExtractor::returning(vec![
            Ok("not json".to_string()),
            Ok("still not json".to_string()),
        ]);

        let err = fill_draft(&store, &extractor, &draft, &range, &speakers, usize::MAX)
            .expect_err("garbage twice should fail");
        assert!(matches!(err, DraftError::Unparseable { .. }));

        let updated = store.get_checkpoint(draft.id).unwrap().unwrap();
        assert_eq!(updated.status, CompactionStatus::Discarded);
        let raw = updated
            .raw_model_output
            .expect("raw output should be stored");
        assert!(raw.contains("not json"));
        assert!(raw.contains("still not json"));
        assert!(updated.summary.is_none());
        assert!(store.facts_for(draft.id).unwrap().is_empty());

        let prompts = extractor.prompts.lock().unwrap();
        assert_eq!(prompts.len(), 2);
        assert!(prompts[1].contains("was not valid JSON"));
    }

    #[test]
    fn a_draft_that_already_has_raw_model_output_is_never_overwritten() {
        let store = RecordingStore::new();
        let range = synthetic_range();
        let draft_id = store
            .insert_draft(NewDraft {
                companion_id: 1,
                from_message_id: range.first().unwrap().id,
                through_message_id: range.last().unwrap().id,
                trigger: CompactionTrigger::Threshold,
                raw_model_output: Some("already extracted".to_string()),
            })
            .unwrap();
        let draft = store.get_checkpoint(draft_id).unwrap().unwrap();
        let speakers = solo_speakers();
        let extractor = FakeExtractor::returning(Vec::<std::io::Result<String>>::new());

        let err = fill_draft(&store, &extractor, &draft, &range, &speakers, usize::MAX)
            .expect_err("a draft with raw_model_output already set must be rejected");
        assert!(matches!(err, DraftError::DraftNotPending(id) if id == draft_id));
        assert!(store.facts_for(draft_id).unwrap().is_empty());
        assert_eq!(extractor.prompts.lock().unwrap().len(), 0);
    }

    #[test]
    fn an_empty_range_discards_the_draft_instead_of_panicking() {
        // Regression test: a draft whose messages were all deleted or
        // edited out from under it (#181) leaves `range` empty by the time
        // `fill_draft` runs. `chunk_range` then returns no chunks, so
        // `merge_outputs`'s "at least one chunk" precondition would panic
        // without the guard at the top of `fill_draft`.
        let store = RecordingStore::new();
        let draft = a_pending_draft(&store, &synthetic_range());
        let speakers = solo_speakers();
        let extractor = FakeExtractor::returning(Vec::<std::io::Result<String>>::new());

        let err = fill_draft(&store, &extractor, &draft, &[], &speakers, usize::MAX)
            .expect_err("an empty range should be discarded, not extracted");
        assert!(matches!(err, DraftError::EmptyRange(id) if id == draft.id));

        let updated = store.get_checkpoint(draft.id).unwrap().unwrap();
        assert_eq!(updated.status, CompactionStatus::Discarded);
        assert!(updated.raw_model_output.is_none());
        assert!(store.facts_for(draft.id).unwrap().is_empty());
        assert_eq!(extractor.prompts.lock().unwrap().len(), 0);
    }

    #[test]
    fn an_overlay_budget_of_zero_discards_the_draft() {
        let store = RecordingStore::new();
        let range = synthetic_range();
        let draft = a_pending_draft(&store, &range);
        let speakers = solo_speakers();
        let extractor =
            FakeExtractor::returning(vec![
                Ok(include_str!("fixtures/bad_draft.json").to_string()),
            ]);

        let err = fill_draft(&store, &extractor, &draft, &range, &speakers, 0)
            .expect_err("zero overlay budget should discard the draft");
        assert!(matches!(err, DraftError::OverlayBudget { .. }));

        let updated = store.get_checkpoint(draft.id).unwrap().unwrap();
        assert_eq!(updated.status, CompactionStatus::Discarded);
        assert!(store.facts_for(draft.id).unwrap().is_empty());
    }

    // --- fail_pending_draft (#208) ---

    #[test]
    fn fail_pending_draft_flips_a_pending_draft_to_failed_with_its_reason() {
        let store = RecordingStore::new();
        let range = synthetic_range();
        let draft = a_pending_draft(&store, &range);

        fail_pending_draft(
            &store,
            draft.id,
            "extraction model error: model load failed",
        );

        let updated = store.get_checkpoint(draft.id).unwrap().unwrap();
        assert_eq!(updated.status, CompactionStatus::Failed);
        assert_eq!(
            updated.extraction_error.as_deref(),
            Some("extraction model error: model load failed")
        );
    }

    #[test]
    fn fail_pending_draft_does_not_block_the_next_trigger() {
        // The core of #208's bug: a stuck `Draft` row makes
        // `pending_draft` keep reporting it, which `should_compact`/the
        // manual-trigger route both read as "a draft is already pending".
        // Once `fail_pending_draft` runs, `pending_draft` must go back to
        // `None` so a fresh draft can be queued.
        let store = RecordingStore::new();
        let range = synthetic_range();
        let draft = a_pending_draft(&store, &range);
        assert!(store.pending_draft(draft.companion_id).unwrap().is_some());

        fail_pending_draft(&store, draft.id, "boom");

        assert!(store.pending_draft(draft.companion_id).unwrap().is_none());
    }

    #[test]
    fn fail_pending_draft_never_clobbers_a_draft_that_already_moved_on() {
        // A concurrent commit/discard (or `fill_draft`'s own content-reason
        // discard, already run by the time this is called) must win: the
        // conditional `Draft -> Failed` transition has to lose that race
        // silently rather than overwrite a row that already reached some
        // other terminal state.
        let store = RecordingStore::new();
        let range = synthetic_range();
        let draft = a_pending_draft(&store, &range);
        store
            .transition_status(
                draft.id,
                CompactionStatus::Draft,
                CompactionStatus::Discarded,
            )
            .unwrap();

        fail_pending_draft(&store, draft.id, "too late");

        let updated = store.get_checkpoint(draft.id).unwrap().unwrap();
        assert_eq!(updated.status, CompactionStatus::Discarded);
        assert_eq!(updated.extraction_error, None);
    }

    #[test]
    fn fail_pending_draft_on_an_unknown_id_does_not_panic() {
        let store = RecordingStore::new();
        // No draft was ever inserted; `fail_draft` returns
        // `QueryReturnedNoRows`, the same "lost race" path a concurrent
        // discard would produce. Just asserting this returns at all is the
        // regression test: `fail_pending_draft` must not `.unwrap()`.
        fail_pending_draft(&store, 999, "boom");
    }

    // --- load_draft_for_extraction (#208 review: CodeRabbit) ---

    /// Wraps a [`RecordingStore`] so `get_checkpoint` always fails, so
    /// [`load_draft_for_extraction`]'s checkpoint-read-error branch can be
    /// exercised without a real, breakable `Database` connection — every
    /// other method just delegates, mirroring `commit.rs`'s own
    /// `RacyDiscardStore` test double.
    struct FailingCheckpointStore<'a> {
        inner: &'a RecordingStore,
    }

    impl CompactionStore for FailingCheckpointStore<'_> {
        fn insert_draft(&self, draft: NewDraft) -> rusqlite::Result<i64> {
            self.inner.insert_draft(draft)
        }
        fn get_checkpoint(&self, _id: i64) -> rusqlite::Result<Option<Checkpoint>> {
            Err(rusqlite::Error::SqliteSingleThreadedMode)
        }
        fn pending_draft(&self, companion_id: i32) -> rusqlite::Result<Option<Checkpoint>> {
            self.inner.pending_draft(companion_id)
        }
        fn list_checkpoints(&self, companion_id: i32) -> rusqlite::Result<Vec<Checkpoint>> {
            self.inner.list_checkpoints(companion_id)
        }
        fn latest_committed(&self, companion_id: i32) -> rusqlite::Result<Option<Checkpoint>> {
            self.inner.latest_committed(companion_id)
        }
        fn latest_committed_before(
            &self,
            companion_id: i32,
            from_message_id: i32,
        ) -> rusqlite::Result<Option<Checkpoint>> {
            self.inner
                .latest_committed_before(companion_id, from_message_id)
        }
        fn context_snapshot(
            &self,
            companion_id: i32,
        ) -> rusqlite::Result<(Vec<Fact>, Option<i32>, Option<Checkpoint>)> {
            self.inner.context_snapshot(companion_id)
        }
        fn update_status(&self, id: i64, status: CompactionStatus) -> rusqlite::Result<()> {
            self.inner.update_status(id, status)
        }
        fn transition_status(
            &self,
            id: i64,
            from: CompactionStatus,
            to: CompactionStatus,
        ) -> rusqlite::Result<()> {
            self.inner.transition_status(id, from, to)
        }
        fn set_extraction_result(
            &self,
            id: i64,
            raw_model_output: Option<String>,
            summary: Option<String>,
            attitude_ratings: Option<String>,
        ) -> rusqlite::Result<()> {
            self.inner
                .set_extraction_result(id, raw_model_output, summary, attitude_ratings)
        }
        fn fail_draft(&self, id: i64, error: &str) -> rusqlite::Result<()> {
            self.inner.fail_draft(id, error)
        }
        fn insert_facts(
            &self,
            compaction_id: i64,
            facts: &[FactDraft],
        ) -> rusqlite::Result<Vec<i64>> {
            self.inner.insert_facts(compaction_id, facts)
        }
        fn active_facts(&self, companion_id: i32) -> rusqlite::Result<Vec<Fact>> {
            self.inner.active_facts(companion_id)
        }
        fn facts_for(&self, compaction_id: i64) -> rusqlite::Result<Vec<Fact>> {
            self.inner.facts_for(compaction_id)
        }
        fn supersede(&self, fact_id: i64, by: i64) -> rusqlite::Result<()> {
            self.inner.supersede(fact_id, by)
        }
        fn mark_stale_containing(
            &self,
            companion_id: i32,
            message_id: i32,
        ) -> rusqlite::Result<usize> {
            self.inner.mark_stale_containing(companion_id, message_id)
        }
        fn oldest_stale_from(&self, companion_id: i32) -> rusqlite::Result<Option<i32>> {
            self.inner.oldest_stale_from(companion_id)
        }
        fn compacted_through(&self, companion_id: i32) -> rusqlite::Result<Option<i32>> {
            self.inner.compacted_through(companion_id)
        }
        fn set_compacted_through(
            &self,
            companion_id: i32,
            through: Option<i32>,
        ) -> rusqlite::Result<()> {
            self.inner.set_compacted_through(companion_id, through)
        }
        fn pin(&self, message_id: i32) -> rusqlite::Result<()> {
            self.inner.pin(message_id)
        }
        fn unpin(&self, message_id: i32) -> rusqlite::Result<()> {
            self.inner.unpin(message_id)
        }
        fn pins(&self) -> rusqlite::Result<Vec<crate::compaction::types::Pin>> {
            self.inner.pins()
        }
        fn commit_checkpoint(
            &self,
            record: crate::compaction::store::CommitRecord,
        ) -> rusqlite::Result<Checkpoint> {
            self.inner.commit_checkpoint(record)
        }
    }

    #[test]
    fn load_draft_for_extraction_still_fails_the_draft_when_the_read_itself_errors() {
        // The bug CodeRabbit caught: before this fix, a `get_checkpoint`
        // error propagated straight out of `run_extraction_job` without
        // ever calling `fail_pending_draft`, so a transient read failure
        // left the row wedged in `Draft` exactly like every other
        // unhandled failure #208 exists to close -- just one step earlier.
        let inner = RecordingStore::new();
        let range = synthetic_range();
        let draft = a_pending_draft(&inner, &range);
        let store = FailingCheckpointStore { inner: &inner };

        let err = load_draft_for_extraction(&store, draft.id).unwrap_err();
        assert!(err.contains("failed to load draft"));

        let updated = inner.get_checkpoint(draft.id).unwrap().unwrap();
        assert_eq!(updated.status, CompactionStatus::Failed);
        assert_eq!(updated.extraction_error.as_deref(), Some(err.as_str()));
    }

    #[test]
    fn load_draft_for_extraction_on_a_genuinely_unknown_id_does_not_touch_the_store() {
        // `Ok(None)` (no row at all) is not the same failure: nothing to
        // transition, so this must not call `fail_draft` at all -- there is
        // no `RecordingStore` checkpoint to assert against, so the
        // assertion here is just that this returns cleanly rather than
        // panicking on a `fail_draft` call against an id that was never
        // inserted.
        let store = RecordingStore::new();
        let err = load_draft_for_extraction(&store, 999).unwrap_err();
        assert_eq!(err, "draft 999 not found");
    }

    // --- panic_message / panic_payload_text (#208) ---
    //
    // Exercises the exact failure mode discovered while building this fix:
    // `llama-cpp-2`'s `LlamaModel::load_from_file` does not return `Err` on
    // a missing GGUF path in a debug build, it `debug_assert!`s and panics
    // -- confirmed by running this binary directly against the default,
    // unconfigured `llm_model_path`. Without `run_extraction_job`'s
    // `catch_unwind`, that panic unwinds straight past `fail_pending_draft`
    // and the checkpoint stays wedged in `Draft` exactly like the bug this
    // issue fixes.
    //
    // The last test below is a second one, added in review: an earlier
    // revision of `run_extraction_job` called `panic_message(&payload)`
    // instead of `panic_message(payload.as_ref())`. That compiles
    // (`&payload` unsizes the `Box<dyn Any + Send>` itself into the trait
    // object, since `Box<T>` is `Sized` and blanket-implements `Any`) and
    // still produces the correct message -- `panic_payload_text`'s
    // defensive nested-`Box` fallback (kept deliberately) absorbs it -- but
    // it does so by relying on a fallback meant for a different situation,
    // and the doc comment on that fallback used to describe the wrong
    // cause entirely (blamed the actix/tokio worker machinery instead of
    // this call site's own coercion). `payload.as_ref()` reaches the same
    // message directly, with no fallback involved.

    #[test]
    fn panic_message_formats_a_str_payload() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("boom");
        assert_eq!(panic_message(payload.as_ref()), "extraction panicked: boom");
    }

    #[test]
    fn panic_message_formats_a_string_payload() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("boom".to_string());
        assert_eq!(panic_message(payload.as_ref()), "extraction panicked: boom");
    }

    #[test]
    fn panic_message_unwraps_one_layer_of_boxed_any_as_a_defensive_fallback() {
        // Not the shape production hits (see the group comment above) --
        // this covers `panic_payload_text`'s recursive arm on its own
        // terms, in case something upstream of `run_extraction_job` ever
        // re-boxes a payload before it reaches here.
        let inner: Box<dyn std::any::Any + Send> = Box::new("nested boom".to_string());
        let payload: Box<dyn std::any::Any + Send> = Box::new(inner);
        assert_eq!(
            panic_message(payload.as_ref()),
            "extraction panicked: nested boom"
        );
    }

    #[test]
    fn panic_message_falls_back_for_a_non_string_payload() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(42i32);
        assert_eq!(
            panic_message(payload.as_ref()),
            "extraction panicked with a non-string payload"
        );
    }

    #[test]
    fn panic_message_reaches_the_real_text_either_way_but_as_ref_skips_the_fallback() {
        // Runs against a real `catch_unwind`, not a hand-built `Box<dyn
        // Any>` (like the two tests above), since only a real one
        // reproduces the exact coercion `run_extraction_job`'s call site
        // hits. `&payload` (`payload: Box<dyn Any + Send>`) unsizes the
        // *`Box` itself* into the trait object -- `Box<T>` is `Sized` and
        // blanket-implements `Any`, so `downcast_ref::<&str>()`/`<String>()`
        // both miss and this only resolves through
        // `panic_payload_text`'s one-layer fallback. `payload.as_ref()`
        // derefs through the box first and resolves directly, on the first
        // check, with no fallback involved.
        //
        // Both still land on the identical, correct message: the retained
        // fallback (kept deliberately, see `panic_payload_text`'s doc
        // comment) is exactly why an earlier revision's `panic_message(
        // &payload)` call site (caught in review, since fixed to
        // `payload.as_ref()`) never actually produced a *wrong* message --
        // its bug was a misattributed doc comment about which path
        // handled it, not an observable output difference. `.as_ref()` is
        // still the right call: it is correct without leaning on a
        // fallback meant for a payload boxed by something else upstream.
        let payload = std::panic::catch_unwind(|| {
            panic!("simulated model load panic");
        })
        .unwrap_err();

        assert_eq!(
            panic_message(&payload),
            "extraction panicked: simulated model load panic"
        );
        assert_eq!(
            panic_message(payload.as_ref()),
            "extraction panicked: simulated model load panic"
        );
    }

    #[test]
    fn run_extraction_jobs_catch_unwind_converts_a_panic_to_a_failed_draft() {
        // `run_extraction_job` itself is wired to the real `Database`/
        // `ResidentExtractor` globals and cannot run in a unit test (see the
        // comment on `run_extraction_surfaces_fill_drafts_model_error_...`
        // below), so this exercises the exact same `catch_unwind` ->
        // `fail_pending_draft` sequence `run_extraction_job` runs, against
        // a closure that panics the same way `fill_draft`'s real call chain
        // does -- proving the wiring this issue is actually about, not just
        // `fail_pending_draft` in isolation (that only proves the DB write
        // works once something remembers to call it).
        let store = RecordingStore::new();
        let range = synthetic_range();
        let draft = a_pending_draft(&store, &range);

        let outcome: Result<(), String> = std::panic::catch_unwind(|| {
            panic!("{}", "simulated model load panic");
        })
        .unwrap_or_else(|payload| Err(panic_message(payload.as_ref())));

        assert!(outcome.is_err());
        if let Err(err) = &outcome {
            fail_pending_draft(&store, draft.id, err);
        }

        let updated = store.get_checkpoint(draft.id).unwrap().unwrap();
        assert_eq!(updated.status, CompactionStatus::Failed);
        assert_eq!(
            updated.extraction_error.as_deref(),
            Some("extraction panicked: simulated model load panic")
        );
    }

    // --- run_extraction (#208) ---

    #[test]
    fn run_extraction_surfaces_fill_drafts_model_error_without_panicking() {
        // `extract_chunk` maps an `Extractor::extract` I/O error straight to
        // `DraftError::Model` and returns early -- `fill_draft` never calls
        // `discard_draft` for this path (unlike `Unparseable`), so this is
        // exactly the failure #208 exists to terminal-ize: without the fix,
        // this draft is left in `Draft` forever.
        let store = RecordingStore::new();
        let range = synthetic_range();
        let draft = a_pending_draft(&store, &range);
        let speakers = solo_speakers();
        let extractor = FakeExtractor::returning(vec![Err(std::io::Error::other(
            "simulated model load failure",
        ))]);

        let err = fill_draft(&store, &extractor, &draft, &range, &speakers, usize::MAX)
            .expect_err("a model error should fail, not silently succeed");
        assert!(matches!(err, DraftError::Model(_)));

        // `fill_draft` alone (what this test exercises directly, since
        // `run_extraction`/`run_extraction_job` are wired to the real
        // `Database`/`ResidentExtractor` globals and cannot run in a unit
        // test) leaves the row exactly as it found it -- still `Draft`.
        // `fail_pending_draft`'s own tests above cover the transition this
        // error is meant to trigger once `run_extraction_job` calls it.
        let updated = store.get_checkpoint(draft.id).unwrap().unwrap();
        assert_eq!(updated.status, CompactionStatus::Draft);
    }

    // --- spawn_holding ---

    #[test]
    fn spawn_holding_keeps_the_slot_claimed_while_the_job_runs_and_releases_it_after() {
        static SLOT: crate::turn_slot::TurnSlot = crate::turn_slot::TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let start_barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let job_barrier = start_barrier.clone();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();

        let handle = spawn_holding(guard, move || {
            job_barrier.wait();
            release_rx
                .recv()
                .expect("main thread should signal before the job returns");
        });

        start_barrier.wait();
        assert!(
            SLOT.try_claim().is_none(),
            "slot should still be claimed while the job runs"
        );

        release_tx.send(()).unwrap();
        handle.join().expect("job should not panic");
        assert!(
            SLOT.try_claim().is_some(),
            "slot should be released once the job returns"
        );
    }

    #[test]
    fn spawn_holding_releases_the_slot_even_when_the_job_panics() {
        static SLOT: crate::turn_slot::TurnSlot = crate::turn_slot::TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");

        let handle = spawn_holding(guard, || panic!("simulated job panic"));

        handle.join().expect_err("job should have panicked");
        assert!(
            SLOT.try_claim().is_some(),
            "slot should be released even though the job panicked"
        );
    }

    // --- #207: model-free GBNF syntax guard ---
    //
    // Neither of these needs a GGUF, so unlike `extracts_from_a_real_gguf`
    // below they always run in CI. This is the guard the issue calls "the
    // actual root cause" — a malformed grammar constant must fail here, not
    // only on hardware.

    #[test]
    fn every_shipped_grammar_constant_passes_the_gbnf_rule_boundary_lint() {
        for (name, gbnf) in [
            ("EXTRACTION_GRAMMAR", EXTRACTION_GRAMMAR),
            ("SUMMARY_GRAMMAR", SUMMARY_GRAMMAR),
        ] {
            assert_eq!(
                check_gbnf_rule_boundaries(gbnf),
                Ok(()),
                "{name} should pass the GBNF rule-boundary lint"
            );
        }
    }

    /// Falsifies the lint above: reconstructs the exact pre-#207 shape of
    /// `root` (a multi-line top-level sequence, unindented sub-rules split
    /// across lines with no enclosing parens) and confirms the lint rejects
    /// it. Without this, `every_shipped_grammar_constant_passes_the_gbnf_rule_boundary_lint`
    /// could pass merely because `check_gbnf_rule_boundaries` is a no-op.
    #[test]
    fn the_lint_rejects_the_original_pre_207_multi_line_root() {
        let pre_207_root = r#"root ::= "{" ws
  "\"companion_state\"" ws ":" ws state-array ws "," ws
  "\"summary\"" ws ":" ws string
ws "}"

state-array ::= "[" ws "]"
string ::= "\"" char{1,400} "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \n\t]{0,20}
"#;
        assert!(
            check_gbnf_rule_boundaries(pre_207_root).is_err(),
            "the lint should reject a rule whose production is split across \
             lines outside any parentheses, matching the pre-#207 bug"
        );
    }

    /// Confirms the lint does not merely reject every multi-line grammar:
    /// a newline nested inside an unclosed `(...)` group, as llama.cpp's
    /// own bundled `json.gbnf` uses, is legal.
    #[test]
    fn the_lint_allows_a_newline_nested_inside_parens() {
        let nested = r#"object ::= "{" (
  string ":" value
)? "}"
string ::= "\"" [a-z]* "\""
value ::= string
"#;
        assert_eq!(check_gbnf_rule_boundaries(nested), Ok(()));
    }

    /// A newline directly after `::=`, at paren depth 0, is legal per
    /// `llama-grammar.cpp::parse_rule` (line 675:
    /// `pos = parse_space(pos + 3, /*newline_ok=*/true);`, called
    /// immediately after matching `::=`) — this is exactly the shape
    /// llama.cpp's own bundled `json.gbnf` uses for every multi-line rule
    /// (`object ::=\n  "{" (...)? "}"`). The pre-fix version of this lint
    /// rejected it as a false positive.
    #[test]
    fn the_lint_allows_a_newline_directly_after_coloncoloneq() {
        let after_coloncoloneq = r#"object ::=
  "{" ws "}"
ws ::= [ \n\t]*
"#;
        assert_eq!(check_gbnf_rule_boundaries(after_coloncoloneq), Ok(()));
    }

    /// A newline directly after a trailing `|`, at paren depth 0, is legal
    /// per `llama-grammar.cpp::parse_alternates` (line 443:
    /// `pos = parse_space(pos + 1, true);`, called immediately after
    /// matching `|`, unconditionally — not gated on `is_nested`). A newline
    /// *before* a leading `|` on the next line is not: `parse_sequence`
    /// stops at the newline (its own `parse_space` calls pass
    /// `is_nested`, which is `false` at depth 0), so `parse_alternates`'s
    /// `while (*pos == '|')` loop never sees it and the rule ends there —
    /// the next line is then read as a new `name ::=` definition and fails,
    /// exactly like the original bug.
    #[test]
    fn the_lint_allows_a_newline_directly_after_a_trailing_pipe_but_not_before_a_leading_one() {
        let trailing_pipe = r#"value ::= "a" |
  "b" | "c"
"#;
        assert_eq!(check_gbnf_rule_boundaries(trailing_pipe), Ok(()));

        let leading_pipe = r#"value ::= "a" | "b"
  | "c"
"#;
        assert!(
            check_gbnf_rule_boundaries(leading_pipe).is_err(),
            "a newline before a leading `|` is not the exception llama.cpp grants; \
             the lint should still reject it"
        );
    }

    // --- manual/CI-optional acceptance test ---

    /// A `ConfigModify` that passes every validation rule, mirroring
    /// `llm.rs`'s own `valid_config_modify` test fixture (not reusable
    /// across modules since it is private there).
    fn valid_config_modify() -> crate::database::ConfigModify {
        crate::database::ConfigModify {
            device: "CPU".to_string(),
            llm_model_path: String::new(),
            gpu_layers: 0,
            prompt_template: "Auto".to_string(),
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
            multiplayer_mode: "solo".to_string(),
            multiplayer_host_address: String::new(),
            multiplayer_participant_id: String::new(),
            mention_followup_depth: 1,
            remote_generation_timeout_secs: 120,
            multiplayer_password: None,
            compact_threshold_tokens: None,
            compact_min_messages: 8,
            compaction_model_path: None,
            heuristic_person_detection: true,
            compaction_attitude_weight: 0.5,
        }
    }

    /// Manual/CI-optional acceptance test: with `AI_COMPANION_TEST_GGUF` set
    /// to a small instruct GGUF, exercises `ResidentExtractor::extract`
    /// end to end over `synthetic_range()`. Unset (the default for `cargo
    /// test`, including CI), it prints why it skipped and returns rather
    /// than using `#[ignore]`, matching `llm.rs`'s own real-GGUF test, whose
    /// `paths::init` caveat applies here too: run this test alone, not
    /// alongside another real-GGUF test in the same process.
    #[test]
    fn extracts_from_a_real_gguf() {
        let gguf_path = match std::env::var("AI_COMPANION_TEST_GGUF") {
            Ok(path) => path,
            Err(_) => {
                println!("skipped: set AI_COMPANION_TEST_GGUF to a small instruct GGUF");
                return;
            }
        };

        let dir = tempfile::tempdir().unwrap();
        crate::paths::init(dir.path().to_path_buf())
            .expect("paths::init should not already be set in this process");
        Database::init().expect("failed to initialise the test database");

        let mut modify = valid_config_modify();
        modify.compaction_model_path = Some(gguf_path);
        Database::change_config(modify).expect("failed to save the extractor config");

        let speakers = solo_speakers();
        let range = synthetic_range();
        let prompt = build_extraction_prompt(&PriorNotes::default(), &range, &speakers);

        let raw = ResidentExtractor
            .extract(&prompt, EXTRACTION_GRAMMAR, EXTRACTION_MAX_TOKENS)
            .expect("extraction should succeed against a real GGUF");
        let output = parse_extraction(&raw)
            .expect("extractor output should match the extraction schema on the first attempt");

        assert!(!output.companion_state.is_empty());
        assert!(!output.user_state.is_empty());
        assert!(!output.rules.is_empty());
    }
}
