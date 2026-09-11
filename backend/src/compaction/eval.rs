//! A deterministic eval harness for the extraction pipeline: runs
//! [`extract`](crate::compaction::extract) over a set of fixed message
//! ranges and prints a diffable report, so a prompt or grammar change
//! becomes a measurable delta instead of an impression formed from one
//! hand-run.
//!
//! # Why this is a `#[test]` and not an `examples/` binary
//!
//! `ai-companion` is a binary crate with no `lib.rs`, so neither
//! `examples/` nor `tests/` can reach `crate::compaction` at all — an
//! integration test here can only drive the built binary over HTTP, which
//! is the wrong altitude for measuring one pure pass. An in-crate
//! `#[cfg(test)]` module is the only place this pipeline is callable
//! directly, and it is also where `extract.rs`'s own
//! `extracts_from_a_real_gguf` already lives, so this follows a pattern the
//! repo has rather than inventing one.
//!
//! # Running it
//!
//! ```text
//! cd backend
//! AI_COMPANION_TEST_GGUF=../models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
//!   cargo test --bin ai-companion compaction::eval::extraction_eval \
//!   -- --nocapture --test-threads=1
//! ```
//!
//! Both gates skip cleanly and loudly:
//!
//! * `AI_COMPANION_TEST_GGUF` unset — no extractor model, nothing to run.
//! * the local fixture directory absent — only the committed synthetic
//!   range runs.
//!
//! The test name filter is not optional: `crate::paths::init` can only be
//! called once per process and `extract.rs`'s own real-GGUF test calls it
//! too, so the two must not run in the same binary invocation.
//!
//! Other knobs:
//!
//! * `AI_COMPANION_EVAL_FIXTURES` — directory of range fixtures, default
//!   `backend/tests/fixtures/local/eval` (gitignored in full; see
//!   `scripts/make_eval_fixtures.sh`, which derives it from a real
//!   database).
//! * `AI_COMPANION_EVAL_RANGES` — comma-separated range names to run, for
//!   iterating on one range instead of all of them.
//!
//! # Gold labels
//!
//! The headline metric is subject inversion, which needs ground truth the
//! model's own output cannot supply. Each range may have a
//! `<name>.gold.json` next to its fixture:
//!
//! ```json
//! { "labels": [ { "key": "jinx keeps her guard up around strangers",
//!                 "about": "companion",
//!                 "note": "companion_state" } ] }
//! ```
//!
//! `key` is [`label_key`] of the item's text — lowercased, punctuation
//! stripped, first [`LABEL_KEY_WORDS`] words — so a label stays attached to
//! a piece of text no matter which array a later prompt files it under,
//! which is exactly the error being measured. `about` is
//! `"user"`/`"companion"` (scored) or `"ambiguous"` (deliberately excluded).
//!
//! To add a range: write its fixture, run the harness, and paste the
//! `unlabelled items` block it prints into `<name>.gold.json` — **after
//! correcting each `about` by reading who the text is actually about.**
//! Recording what the model said would make the metric measure nothing.
//!
//! # What this deliberately does not do
//!
//! Extraction runs against empty [`PriorNotes`] and no active facts, so the
//! run is independent of database state and reproducible: the `replaces`
//! and `Duplicate` paths are therefore not exercised. The multi-chunk
//! re-summarise pass is skipped too — summary prose is not one of the
//! measured metrics, and skipping it keeps wall time comparable between a
//! chunked and an unchunked range.
#![cfg(test)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::compaction::extract::{
    build_extraction_grammar, build_extraction_prompt, chunk_range, merge_outputs,
    parse_extraction, to_fact_drafts, AttitudeRatings, ExtractionOutput, PriorNotes,
    RangeParticipants, EXTRACTION_MAX_TOKENS,
};
use crate::compaction::types::{FactCategory, FactDraft, FactSubject};
use crate::compaction::validate::{validate, RejectReason, MAX_ITEM_WORDS};
use crate::compaction::{CitedMessage, SoloSpeakers, SpeakerInfo};
use crate::context_manager::ContextManager;
use crate::llm::{Extractor, ResidentExtractor};

// --- configuration ---

/// Env var naming the extractor GGUF, shared with `extract.rs`'s and
/// `llm.rs`'s own real-model tests.
const GGUF_ENV: &str = "AI_COMPANION_TEST_GGUF";

/// Env var overriding [`default_fixture_dir`].
const FIXTURE_DIR_ENV: &str = "AI_COMPANION_EVAL_FIXTURES";

/// Env var holding a comma-separated allowlist of range names to run.
const RANGES_ENV: &str = "AI_COMPANION_EVAL_RANGES";

/// How many leading words of an item's text form its gold-label key. Long
/// enough to be unambiguous within one range, short enough that a reworded
/// tail does not orphan the label.
const LABEL_KEY_WORDS: usize = 10;

/// The per-item word limit `build_extraction_prompt` asks the model for.
/// The validator's own limit is [`MAX_ITEM_WORDS`], which is higher; the
/// gap between them is what `item length` reports on.
const PROMPT_WORD_LIMIT: usize = 25;

/// Where range fixtures live when [`FIXTURE_DIR_ENV`] is unset: a
/// gitignored directory, because anything derived from a real database
/// carries user chat text.
fn default_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/local/eval")
}

// --- fixtures ---

/// One fixed range to extract over. `user_name`/`companion_name` build the
/// [`SoloSpeakers`] policy, which is what decides both the `(canon)` marker
/// on each rendered line and the names the `#ROLES:` block interpolates.
#[derive(Debug, Clone, Deserialize)]
struct RangeFixture {
    name: String,
    description: String,
    user_name: String,
    companion_name: String,
    messages: Vec<CitedMessage>,
}

impl RangeFixture {
    fn speakers(&self) -> SoloSpeakers {
        SoloSpeakers {
            user_name: self.user_name.clone(),
            companion_name: self.companion_name.clone(),
        }
    }

    /// `first-last` message id, for the report header.
    fn id_span(&self) -> String {
        match (self.messages.first(), self.messages.last()) {
            (Some(first), Some(last)) => format!("{}-{}", first.id, last.id),
            _ => "empty".to_string(),
        }
    }
}

/// The one range that ships in git: `fixtures/synthetic_range.json` holds
/// no real chat text, so it is always available and gives the harness a
/// floor even on a machine with no local fixtures.
fn synthetic_fixture() -> RangeFixture {
    RangeFixture {
        name: "synthetic_range".to_string(),
        description: "the committed synthetic fixture (no real chat text)".to_string(),
        user_name: "Eric".to_string(),
        companion_name: "Vi".to_string(),
        messages: crate::compaction::fixtures::synthetic_range(),
    }
}

/// Every `*.json` in `dir` that is not a `*.gold.json`, sorted by name so
/// two runs report ranges in the same order.
fn load_local_fixtures(dir: &Path) -> Vec<RangeFixture> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "json")
                && !path.to_string_lossy().ends_with(".gold.json")
        })
        .collect();
    paths.sort();

    paths
        .iter()
        .map(|path| {
            let text = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
            serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("fixture {} is not a RangeFixture: {e}", path.display()))
        })
        .collect()
}

// --- gold labels ---

/// Who a labelled item is actually about, as judged by a human reading the
/// text — not as the model filed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum About {
    User,
    Companion,
    /// Genuinely about both or neither; excluded from the inversion rate
    /// rather than being scored as a coin flip.
    Ambiguous,
}

impl About {
    /// The [`FactSubject`] a correctly-filed item would carry, or `None`
    /// for [`About::Ambiguous`].
    fn expected_subject(self) -> Option<FactSubject> {
        match self {
            About::User => Some(FactSubject::User),
            About::Companion => Some(FactSubject::Companion),
            About::Ambiguous => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct GoldLabel {
    key: String,
    about: About,
    #[serde(default)]
    #[allow(dead_code)] // free-text aide-memoire for whoever labels the file
    note: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct GoldLabels {
    labels: Vec<GoldLabel>,
}

impl GoldLabels {
    fn lookup(&self, key: &str) -> Option<About> {
        self.labels
            .iter()
            .find(|label| label.key == key)
            .map(|label| label.about)
    }
}

/// Loads `<dir>/<name>.gold.json`, treating an absent file as "no labels
/// yet" — the harness still runs and reports every item as unlabelled,
/// which is how a new range gets seeded.
fn load_gold_labels(dir: &Path, name: &str) -> GoldLabels {
    let path = dir.join(format!("{name}.gold.json"));
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("gold labels {} did not parse: {e}", path.display())),
        Err(_) => GoldLabels::default(),
    }
}

/// The stable key an item's text is labelled under: lowercased, every
/// non-alphanumeric character treated as a separator, first
/// [`LABEL_KEY_WORDS`] words joined by single spaces. Keyed on text alone,
/// never on category, so a label survives the model moving the same
/// sentence from `user_state` to `companion_state` — which is the whole
/// error class being counted.
fn label_key(text: &str) -> String {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .take(LABEL_KEY_WORDS)
        .collect::<Vec<_>>()
        .join(" ")
}

// --- the counting extractor ---

/// Wraps the production extraction call so the harness can report prompt
/// and generated token counts, which [`Extractor`] itself does not expose.
/// Every decode still goes through `llm::run_extraction` with the same
/// grammar and the same greedy sampler production uses, so nothing about
/// the measured behaviour differs from a real run.
struct CountingExtractor {
    prompt_tokens: std::cell::Cell<usize>,
    generated_tokens: std::cell::Cell<usize>,
}

impl CountingExtractor {
    fn new() -> Self {
        CountingExtractor {
            prompt_tokens: std::cell::Cell::new(0),
            generated_tokens: std::cell::Cell::new(0),
        }
    }

    fn record(&self, extraction: &crate::llm::Extraction) {
        self.prompt_tokens
            .set(self.prompt_tokens.get() + extraction.prompt_tokens);
        self.generated_tokens
            .set(self.generated_tokens.get() + extraction.tokens_generated);
    }
}

impl Extractor for CountingExtractor {
    fn context_window(&self) -> usize {
        ResidentExtractor.context_window()
    }

    fn complete(&self, prompt: &str, max_tokens: usize) -> std::io::Result<String> {
        let extraction = crate::llm::run_extraction(prompt, None, max_tokens)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        self.record(&extraction);
        Ok(extraction.text)
    }

    fn extract(&self, prompt: &str, grammar: &str, max_tokens: usize) -> std::io::Result<String> {
        let extraction = crate::llm::run_extraction(prompt, Some(grammar), max_tokens)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        self.record(&extraction);
        Ok(extraction.text)
    }
}

// --- running one range ---

/// Everything one range's extraction produced, before it is scored.
struct RunOutput {
    chunks: usize,
    parse_retries: usize,
    scaffold_fingerprint: u64,
    merged: ExtractionOutput,
    validated: Vec<FactDraft>,
    prompt_tokens: usize,
    generated_tokens: usize,
    elapsed: Duration,
}

/// Mirrors `fill_draft`'s pipeline without its store: chunk, extract each
/// chunk under the range's own built grammar, merge, map to drafts, validate.
/// Deliberately not `fill_draft` itself — that writes checkpoint rows and
/// reads prior notes out of the database, neither of which belongs in a
/// measurement whose whole value is being reproducible.
fn run_range(fixture: &RangeFixture) -> Result<RunOutput, String> {
    let speakers = fixture.speakers();
    let prior = PriorNotes::default();
    let extractor = CountingExtractor::new();

    let participants = RangeParticipants::from_range(&fixture.messages, &speakers);
    let grammar = build_extraction_grammar(&participants);
    let scaffold = build_extraction_prompt(&prior, &[], &speakers, &participants);
    let scaffold_tokens = ContextManager::estimate_tokens(&scaffold);
    let chunks = chunk_range(
        &fixture.messages,
        &speakers,
        scaffold_tokens,
        extractor.context_window(),
    );

    // `make_eval_fixtures.sh` writes a fixture file even when its id range
    // selects no rows, and `merge_outputs` panics on an empty chunk list.
    // Reported as this range failing, so one empty fixture does not abort
    // the whole run.
    if chunks.is_empty() {
        return Err("fixture has no messages to extract".to_string());
    }

    let started = Instant::now();
    let mut outputs = Vec::with_capacity(chunks.len());
    let mut parse_retries = 0;
    for (index, chunk) in chunks.iter().enumerate() {
        println!(
            "  [{}] chunk {}/{} ({} messages)...",
            fixture.name,
            index + 1,
            chunks.len(),
            chunk.len()
        );
        let prompt = build_extraction_prompt(&prior, chunk, &speakers, &participants);
        let (output, retried) = extract_chunk(&extractor, &prompt, &grammar)
            .map_err(|e| format!("chunk {} failed: {e}", index + 1))?;
        parse_retries += usize::from(retried);
        outputs.push(output);
    }
    let elapsed = started.elapsed();

    let merged = merge_outputs(outputs);
    let is_canon = |speaker_id: &str| speakers.is_canon(speaker_id);
    let validated = validate(
        to_fact_drafts(&merged, &participants),
        &fixture.messages,
        &[],
        &is_canon,
    );

    Ok(RunOutput {
        chunks: chunks.len(),
        parse_retries,
        scaffold_fingerprint: fingerprint(&scaffold),
        merged,
        validated,
        prompt_tokens: extractor.prompt_tokens.get(),
        generated_tokens: extractor.generated_tokens.get(),
        elapsed,
    })
}

/// One chunk, with the same single retry-on-unparseable that
/// `extract.rs`'s own `extract_chunk` performs, so the harness measures the
/// pipeline production runs rather than a stricter one. Returns whether the
/// retry was needed, since a range that only parses on the second attempt
/// is itself a finding. A second failure ends the range: that is what
/// `fill_draft` treats as terminal too.
fn extract_chunk(
    extractor: &CountingExtractor,
    prompt: &str,
    grammar: &str,
) -> Result<(ExtractionOutput, bool), String> {
    let first = extractor
        .extract(prompt, grammar, EXTRACTION_MAX_TOKENS)
        .map_err(|e| e.to_string())?;
    let parse_error = match parse_extraction(&first) {
        Ok(output) => return Ok((output, false)),
        Err(e) => e,
    };

    println!("    (output did not parse: {parse_error}; retrying once, as production does)");
    let retry_prompt = format!(
        "{prompt}\nYour previous output was not valid JSON: {parse_error}. Produce the JSON again."
    );
    let second = extractor
        .extract(&retry_prompt, grammar, EXTRACTION_MAX_TOKENS)
        .map_err(|e| e.to_string())?;
    match parse_extraction(&second) {
        Ok(output) => Ok((output, true)),
        Err(e) => Err(format!("output did not parse twice in a row: {e}")),
    }
}

/// A cheap fingerprint of the prompt scaffold (everything
/// `build_extraction_prompt` renders around the transcript), printed per
/// range so two reports can be compared even when neither names a commit —
/// a clean tree with an edited prompt is otherwise indistinguishable from
/// an unedited one. `DefaultHasher` is not stable across Rust releases,
/// which is fine: this only ever compares runs from the same toolchain.
fn fingerprint(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

// --- metrics ---

/// One item scored against the gold labels.
struct SubjectVerdict {
    category: FactCategory,
    assigned: Option<FactSubject>,
    expected: Option<FactSubject>,
    accepted: bool,
    text: String,
}

impl SubjectVerdict {
    fn inverted(&self) -> bool {
        match (&self.assigned, &self.expected) {
            (Some(assigned), Some(expected)) => assigned != expected,
            _ => false,
        }
    }

    fn scored(&self) -> bool {
        self.assigned.is_some() && self.expected.is_some()
    }
}

/// The categories whose subject the model chooses, and which the inversion
/// metric therefore scores: the two state arrays (subject implied by which
/// array the item landed in) and `backstory` (subject stated outright as
/// `about`).
fn subject_bearing(category: FactCategory) -> bool {
    matches!(
        category,
        FactCategory::CompanionState | FactCategory::UserState | FactCategory::Backstory
    )
}

/// Buckets a stored `rejected_reason` back to its [`RejectReason`] variant
/// name. `RejectReason` has no `FromStr` and two variants interpolate a
/// number into their `Display`, so the parameterless variants are matched
/// exactly and those two by the fixed prefix they start with.
/// `every_reject_reason_buckets_to_its_own_variant` pins every arm against
/// the enum, so a wording change in `validate.rs` fails there rather than
/// silently splitting a bucket.
fn reason_bucket(reason: &str) -> &'static str {
    for (variant, label) in [
        (RejectReason::NoSources, "NoSources"),
        (RejectReason::NotCanon, "NotCanon"),
        (RejectReason::QuoteNotVerbatim, "QuoteNotVerbatim"),
        (RejectReason::SpeakerMismatch, "SpeakerMismatch"),
        (RejectReason::Duplicate, "Duplicate"),
        (RejectReason::UnknownSubject, "UnknownSubject"),
        (RejectReason::PrincipalAsPerson, "PrincipalAsPerson"),
    ] {
        if reason == variant.to_string() {
            return label;
        }
    }
    if reason.starts_with("source message ") {
        return "SourceOutOfRange";
    }
    if reason.starts_with("item is ") {
        return "TooLong";
    }
    "Unrecognised"
}

/// How an item's word count sits against the two different limits in play.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct LengthBuckets {
    /// Within the limit the prompt asks for.
    within_prompt_limit: usize,
    /// Over the prompt's limit but under the validator's, so accepted
    /// despite the instruction — the "lost in between" band.
    over_prompt_under_validator: usize,
    /// Over the validator's limit, so rejected outright.
    over_validator_limit: usize,
}

impl LengthBuckets {
    fn add(&mut self, words: usize) {
        if words > MAX_ITEM_WORDS {
            self.over_validator_limit += 1;
        } else if words > PROMPT_WORD_LIMIT {
            self.over_prompt_under_validator += 1;
        } else {
            self.within_prompt_limit += 1;
        }
    }

    fn merge(&mut self, other: &LengthBuckets) {
        self.within_prompt_limit += other.within_prompt_limit;
        self.over_prompt_under_validator += other.over_prompt_under_validator;
        self.over_validator_limit += other.over_validator_limit;
    }
}

/// Quote-specific counts: `rules` and `key_quotes` are the two categories
/// the verbatim and speaker checks apply to.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct QuoteCounts {
    rules: usize,
    key_quotes: usize,
    accepted: usize,
    not_verbatim: usize,
    wrong_speaker: usize,
    other_rejection: usize,
}

impl QuoteCounts {
    fn emitted(&self) -> usize {
        self.rules + self.key_quotes
    }

    fn merge(&mut self, other: &QuoteCounts) {
        self.rules += other.rules;
        self.key_quotes += other.key_quotes;
        self.accepted += other.accepted;
        self.not_verbatim += other.not_verbatim;
        self.wrong_speaker += other.wrong_speaker;
        self.other_rejection += other.other_rejection;
    }
}

/// Everything the report prints for one range, and the unit the aggregate
/// sums over.
struct RangeReport {
    name: String,
    description: String,
    id_span: String,
    scaffold_fingerprint: u64,
    message_count: usize,
    chunks: usize,
    parse_retries: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    elapsed: Duration,
    attitude: AttitudeRatings,
    emitted: usize,
    accepted: usize,
    rejections: BTreeMap<&'static str, usize>,
    verdicts: Vec<SubjectVerdict>,
    unlabelled: Vec<(FactCategory, String)>,
    quotes: QuoteCounts,
    lengths: LengthBuckets,
    principals_in_people: Vec<String>,
}

impl RangeReport {
    fn inverted(&self) -> usize {
        self.verdicts.iter().filter(|v| v.inverted()).count()
    }

    fn scored(&self) -> usize {
        self.verdicts.iter().filter(|v| v.scored()).count()
    }

    fn inverted_and_accepted(&self) -> usize {
        self.verdicts
            .iter()
            .filter(|v| v.inverted() && v.accepted)
            .count()
    }
}

/// Scores one completed run against `gold`.
fn score_run(fixture: &RangeFixture, run: RunOutput, gold: &GoldLabels) -> RangeReport {
    let mut report = RangeReport {
        name: fixture.name.clone(),
        description: fixture.description.clone(),
        id_span: fixture.id_span(),
        message_count: fixture.messages.len(),
        chunks: run.chunks,
        parse_retries: run.parse_retries,
        scaffold_fingerprint: run.scaffold_fingerprint,
        prompt_tokens: run.prompt_tokens,
        generated_tokens: run.generated_tokens,
        elapsed: run.elapsed,
        attitude: run.merged.attitude.clone(),
        emitted: run.validated.len(),
        accepted: 0,
        rejections: BTreeMap::new(),
        verdicts: Vec::new(),
        unlabelled: Vec::new(),
        quotes: QuoteCounts::default(),
        lengths: LengthBuckets::default(),
        principals_in_people: Vec::new(),
    };

    for draft in &run.validated {
        let accepted = draft.rejected_reason.is_none();
        if accepted {
            report.accepted += 1;
        } else if let Some(reason) = &draft.rejected_reason {
            *report.rejections.entry(reason_bucket(reason)).or_insert(0) += 1;
        }

        report.lengths.add(draft.text.split_whitespace().count());
        tally_quote(&mut report.quotes, draft);
        collect_principal(&mut report.principals_in_people, draft, fixture);
        score_subject(&mut report, draft, gold, accepted);
    }

    report
}

/// Adds `draft` to the quote counts when it is one of the two quote
/// categories.
fn tally_quote(quotes: &mut QuoteCounts, draft: &FactDraft) {
    match draft.category {
        FactCategory::Rule => quotes.rules += 1,
        FactCategory::KeyQuote => quotes.key_quotes += 1,
        _ => return,
    }
    match draft.rejected_reason.as_deref().map(reason_bucket) {
        None => quotes.accepted += 1,
        Some("QuoteNotVerbatim") => quotes.not_verbatim += 1,
        Some("SpeakerMismatch") => quotes.wrong_speaker += 1,
        Some(_) => quotes.other_rejection += 1,
    }
}

/// Records a `people` item that names one of the two principals.
/// Deliberately independent of `to_fact_drafts`'s own
/// `RejectReason::PrincipalAsPerson` filter, and counted on emission
/// rather than acceptance: the filter is what stops these becoming facts,
/// this line is what says whether the model still produces them.
fn collect_principal(found: &mut Vec<String>, draft: &FactDraft, fixture: &RangeFixture) {
    let Some(FactSubject::Person(name)) = &draft.subject else {
        return;
    };
    // Whole words, matching `RangeParticipants::is_principal`: substring
    // containment would count "Erica" as the principal "Eric" and report a
    // prompt-adherence failure the production filter does not see.
    let candidate = eval_words(name);
    for principal in [&fixture.user_name, &fixture.companion_name] {
        let wanted = eval_words(principal);
        if !wanted.is_empty()
            && candidate
                .windows(wanted.len())
                .any(|window| window == wanted.as_slice())
        {
            found.push(name.clone());
            return;
        }
    }
}

/// `text`'s lowercased alphanumeric words. Mirrors `extract.rs`'s `words_of`,
/// kept separate because the harness deliberately measures emission with its
/// own code rather than sharing the filter it is scoring.
fn eval_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Scores one draft's subject against the gold labels, or records it as
/// unlabelled so the report can print it ready to be labelled.
fn score_subject(report: &mut RangeReport, draft: &FactDraft, gold: &GoldLabels, accepted: bool) {
    if !subject_bearing(draft.category) {
        return;
    }
    let key = label_key(&draft.text);
    match gold.lookup(&key) {
        None => report.unlabelled.push((draft.category, draft.text.clone())),
        Some(about) => report.verdicts.push(SubjectVerdict {
            category: draft.category,
            assigned: draft.subject.clone(),
            expected: about.expected_subject(),
            accepted,
            text: draft.text.clone(),
        }),
    }
}

// --- rendering ---

/// `<short sha>` plus a dirty marker, so a printed report names the code
/// state that produced it. Falls back to `unknown` outside a git checkout.
fn code_state() -> String {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args(args)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let head = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let dirty = match git(&["status", "--porcelain"]) {
        Some(status) if !status.is_empty() => " (dirty working tree)",
        Some(_) => " (clean working tree)",
        None => "",
    };
    format!("{head}{dirty}")
}

fn render_attitude(attitude: &AttitudeRatings) -> String {
    format!(
        "trust={} love={} fear={} anger={} joy={} sorrow={} suspicion={} gratitude={}",
        attitude.trust,
        attitude.love,
        attitude.fear,
        attitude.anger,
        attitude.joy,
        attitude.sorrow,
        attitude.suspicion,
        attitude.gratitude
    )
}

fn percent(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "n/a".to_string()
    } else {
        format!("{:.1}%", 100.0 * numerator as f64 / denominator as f64)
    }
}

fn subject_token(subject: &Option<FactSubject>) -> String {
    match subject {
        Some(subject) => subject.to_string(),
        None => "none".to_string(),
    }
}

/// Truncates to `max` characters for a one-line report entry, on a char
/// boundary so a multi-byte character is never split.
fn snippet(text: &str, max: usize) -> String {
    let truncated: String = text.chars().take(max).collect();
    if truncated.chars().count() < text.chars().count() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn print_range_report(report: &RangeReport) {
    println!("\n## {} — {}", report.name, report.description);
    println!(
        "messages: {} (ids {}), chunks: {}, chunks needing a parse retry: {}",
        report.message_count, report.id_span, report.chunks, report.parse_retries
    );
    println!(
        "prompt scaffold fingerprint: {:016x}",
        report.scaffold_fingerprint
    );
    println!(
        "prompt tokens: {}, generated tokens: {}, wall: {:.1}s",
        report.prompt_tokens,
        report.generated_tokens,
        report.elapsed.as_secs_f64()
    );
    println!("attitude: {}", render_attitude(&report.attitude));
    println!(
        "items: {} emitted, {} accepted, {} rejected",
        report.emitted,
        report.accepted,
        report.emitted - report.accepted
    );
    if report.rejections.is_empty() {
        println!("rejections: none");
    } else {
        println!("rejections by reason:");
        for (reason, count) in &report.rejections {
            println!("  {reason:<20} {count}");
        }
    }

    println!(
        "subject inversion: {}/{} scored items inverted ({}), {} of those were accepted; \
         {} unlabelled",
        report.inverted(),
        report.scored(),
        percent(report.inverted(), report.scored()),
        report.inverted_and_accepted(),
        report.unlabelled.len()
    );
    for verdict in report.verdicts.iter().filter(|v| v.inverted()) {
        println!(
            "  INVERTED [{}] filed as {}, is about {}{}: \"{}\"",
            verdict.category,
            subject_token(&verdict.assigned),
            subject_token(&verdict.expected),
            if verdict.accepted { "" } else { ", rejected" },
            snippet(&verdict.text, 70)
        );
    }

    println!(
        "quotes: {} emitted (rules {}, key_quotes {}), {} accepted, {} not-verbatim, \
         {} wrong-speaker, {} rejected for another reason",
        report.quotes.emitted(),
        report.quotes.rules,
        report.quotes.key_quotes,
        report.quotes.accepted,
        report.quotes.not_verbatim,
        report.quotes.wrong_speaker,
        report.quotes.other_rejection
    );
    println!(
        "item length: {} within the prompt's {}-word limit, {} over it but under the \
         validator's {} (accepted anyway), {} over {} (rejected)",
        report.lengths.within_prompt_limit,
        PROMPT_WORD_LIMIT,
        report.lengths.over_prompt_under_validator,
        MAX_ITEM_WORDS,
        report.lengths.over_validator_limit,
        MAX_ITEM_WORDS
    );
    println!(
        "people items naming a principal: {} (should be 0){}",
        report.principals_in_people.len(),
        if report.principals_in_people.is_empty() {
            String::new()
        } else {
            format!(" -> {}", report.principals_in_people.join(", "))
        }
    );

    print_unlabelled(report);
}

/// Prints unlabelled items as pasteable gold-label JSON. `about` is
/// deliberately left as a placeholder: whoever pastes this must read the
/// text and decide, because seeding it from the model's own filing would
/// make the inversion rate measure nothing.
fn print_unlabelled(report: &RangeReport) {
    if report.unlabelled.is_empty() {
        return;
    }
    println!(
        "unlabelled items — set each \"about\" by hand, then save as {}.gold.json:",
        report.name
    );
    println!("  {{\"labels\": [");
    // No trailing comma on the last entry: the block is meant to be pasted
    // straight into a `.gold.json`, and JSON has no tolerance for one.
    let last = report.unlabelled.len() - 1;
    for (index, (category, text)) in report.unlabelled.iter().enumerate() {
        println!(
            "    {{\"key\": {}, \"about\": \"user|companion|ambiguous\", \"note\": {}}}{}",
            serde_json::to_string(&label_key(text)).unwrap_or_default(),
            serde_json::to_string(&format!("{category}: {text}")).unwrap_or_default(),
            if index == last { "" } else { "," }
        );
    }
    println!("  ]}}");
}

fn print_aggregate(reports: &[RangeReport]) {
    let mut rejections: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut quotes = QuoteCounts::default();
    let mut lengths = LengthBuckets::default();
    let (mut emitted, mut accepted, mut inverted, mut scored) = (0, 0, 0, 0);
    let (mut unlabelled, mut principals, mut prompt_tokens, mut generated_tokens) = (0, 0, 0, 0);
    let mut parse_retries = 0;
    let mut elapsed = Duration::ZERO;

    for report in reports {
        emitted += report.emitted;
        accepted += report.accepted;
        inverted += report.inverted();
        scored += report.scored();
        unlabelled += report.unlabelled.len();
        principals += report.principals_in_people.len();
        prompt_tokens += report.prompt_tokens;
        generated_tokens += report.generated_tokens;
        elapsed += report.elapsed;
        parse_retries += report.parse_retries;
        quotes.merge(&report.quotes);
        lengths.merge(&report.lengths);
        for (reason, count) in &report.rejections {
            *rejections.entry(reason).or_insert(0) += count;
        }
    }

    println!("\n## aggregate over {} range(s)", reports.len());
    println!(
        "items: {emitted} emitted, {accepted} accepted ({}), {} rejected",
        percent(accepted, emitted),
        emitted - accepted
    );
    if !rejections.is_empty() {
        println!("rejections by reason:");
        for (reason, count) in &rejections {
            println!("  {reason:<20} {count}");
        }
    }
    println!(
        "SUBJECT INVERSION: {inverted}/{scored} ({}), {unlabelled} unlabelled",
        percent(inverted, scored)
    );
    println!(
        "quotes: {} emitted, {} accepted ({}), {} not-verbatim, {} wrong-speaker",
        quotes.emitted(),
        quotes.accepted,
        percent(quotes.accepted, quotes.emitted()),
        quotes.not_verbatim,
        quotes.wrong_speaker
    );
    println!(
        "item length: {} within {} words, {} in the {}-{} band, {} over {}",
        lengths.within_prompt_limit,
        PROMPT_WORD_LIMIT,
        lengths.over_prompt_under_validator,
        PROMPT_WORD_LIMIT + 1,
        MAX_ITEM_WORDS,
        lengths.over_validator_limit,
        MAX_ITEM_WORDS
    );
    println!("people items naming a principal: {principals} (should be 0)");
    println!("chunks needing a parse retry: {parse_retries}");
    println!(
        "tokens: {prompt_tokens} prompt, {generated_tokens} generated; wall {:.1}s",
        elapsed.as_secs_f64()
    );
}

// --- the harness itself ---

/// A `ConfigModify` that passes every validation rule, with `gguf_path`
/// installed as the extraction model. Mirrors `extract.rs`'s own fixture of
/// the same shape, which is private there.
fn extractor_config(gguf_path: String) -> crate::database::ConfigModify {
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
        compaction_model_path: Some(gguf_path),
        heuristic_person_detection: true,
        compaction_attitude_weight: 0.5,
        running_thoughts_enabled: false,
    }
}

/// The ranges to run: the committed synthetic one plus every local fixture,
/// narrowed to [`RANGES_ENV`] when it is set.
fn selected_fixtures(fixture_dir: &Path) -> Vec<RangeFixture> {
    let mut fixtures = vec![synthetic_fixture()];
    fixtures.extend(load_local_fixtures(fixture_dir));

    match std::env::var(RANGES_ENV) {
        Err(_) => fixtures,
        Ok(list) => {
            let wanted: Vec<&str> = list.split(',').map(str::trim).collect();
            fixtures
                .into_iter()
                .filter(|fixture| wanted.contains(&fixture.name.as_str()))
                .collect()
        }
    }
}

/// Runs extraction over every selected range and prints the report. Gated
/// on [`GGUF_ENV`]; skips with an explanation rather than `#[ignore]`, the
/// same shape `extract.rs`'s and `llm.rs`'s real-model tests use.
#[test]
fn extraction_eval() {
    let Ok(gguf_path) = std::env::var(GGUF_ENV) else {
        println!("skipped: set {GGUF_ENV} to the extractor GGUF to run the extraction eval");
        return;
    };

    let fixture_dir = std::env::var(FIXTURE_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_fixture_dir());
    let fixtures = selected_fixtures(&fixture_dir);
    if fixtures.is_empty() {
        println!("skipped: no ranges selected (fixture dir {} — see scripts/make_eval_fixtures.sh; {RANGES_ENV} may also be filtering everything out)", fixture_dir.display());
        return;
    }

    let data_dir = tempfile::tempdir().expect("failed to create the harness data dir");
    crate::paths::init(data_dir.path().to_path_buf())
        .expect("paths::init should not already be set — run this test with a name filter");
    crate::database::Database::init().expect("failed to initialise the harness database");
    crate::database::Database::change_config(extractor_config(gguf_path.clone()))
        .expect("failed to save the extractor config");

    println!("\n# Extraction eval");
    println!("code: {}", code_state());
    println!("model: {gguf_path}");
    println!("fixtures: {}", fixture_dir.display());
    println!(
        "note: the first range's wall time includes the one-off model load; \
         later ranges reuse the resident extractor."
    );
    println!(
        "ranges: {}",
        fixtures
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Each range's block prints as soon as that range finishes rather than
    // at the end: a full run is minutes per range on CPU, and a report you
    // cannot read until all of them are done is a report you end up
    // watching a progress line for instead.
    let mut reports = Vec::with_capacity(fixtures.len());
    for fixture in &fixtures {
        println!("\n-- running {} --", fixture.name);
        match run_range(fixture) {
            Ok(run) => {
                let gold = load_gold_labels(&fixture_dir, &fixture.name);
                let report = score_run(fixture, run, &gold);
                print_range_report(&report);
                reports.push(report);
            }
            Err(e) => println!("!! {} failed: {e}", fixture.name),
        }
    }

    print_aggregate(&reports);

    assert!(
        !reports.is_empty(),
        "every selected range failed to extract; see the errors above"
    );
}

// --- unit tests for the scoring, which need no model ---

#[cfg(test)]
mod tests {
    use super::*;

    fn a_draft(category: FactCategory, subject: Option<FactSubject>, text: &str) -> FactDraft {
        FactDraft {
            category,
            subject,
            text: text.to_string(),
            quote_speaker: None,
            sources: vec![46],
            replaces: Vec::new(),
            relation_to: None,
            relation: None,
            canon: false,
            rejected_reason: None,
        }
    }

    fn a_fixture(messages: Vec<CitedMessage>) -> RangeFixture {
        RangeFixture {
            name: "unit".to_string(),
            description: "unit-test fixture".to_string(),
            user_name: "Eric".to_string(),
            companion_name: "Jinx".to_string(),
            messages,
        }
    }

    fn a_run(validated: Vec<FactDraft>) -> RunOutput {
        RunOutput {
            chunks: 1,
            parse_retries: 0,
            scaffold_fingerprint: 0,
            merged: crate::compaction::fixtures::bad_draft(),
            validated,
            prompt_tokens: 0,
            generated_tokens: 0,
            elapsed: Duration::ZERO,
        }
    }

    fn gold(entries: &[(&str, About)]) -> GoldLabels {
        GoldLabels {
            labels: entries
                .iter()
                .map(|(text, about)| GoldLabel {
                    key: label_key(text),
                    about: *about,
                    note: String::new(),
                })
                .collect(),
        }
    }

    #[test]
    fn every_reject_reason_buckets_to_its_own_variant() {
        for (reason, expected) in [
            (RejectReason::NoSources, "NoSources"),
            (RejectReason::SourceOutOfRange(999), "SourceOutOfRange"),
            (RejectReason::NotCanon, "NotCanon"),
            (RejectReason::QuoteNotVerbatim, "QuoteNotVerbatim"),
            (RejectReason::SpeakerMismatch, "SpeakerMismatch"),
            (RejectReason::TooLong { words: 41 }, "TooLong"),
            (RejectReason::Duplicate, "Duplicate"),
            (RejectReason::UnknownSubject, "UnknownSubject"),
            (RejectReason::PrincipalAsPerson, "PrincipalAsPerson"),
        ] {
            assert_eq!(
                reason_bucket(&reason.to_string()),
                expected,
                "`{reason}` should bucket as {expected}"
            );
        }
    }

    #[test]
    fn label_key_ignores_case_punctuation_and_everything_past_the_key_length() {
        let key = label_key("Jinx's guard is UP — around strangers, always, without fail, ever");
        assert_eq!(
            key,
            "jinx s guard is up around strangers always without fail"
        );

        assert_eq!(
            label_key("Jinx keeps her guard up."),
            label_key("jinx  keeps her GUARD up!!")
        );
    }

    #[test]
    fn a_label_follows_its_text_across_categories_so_a_misfiled_item_counts_as_inverted() {
        let text = "keeps her guard up around strangers she has not met";
        let fixture = a_fixture(crate::compaction::fixtures::synthetic_range());
        let labels = gold(&[(text, About::Companion)]);

        let correct = score_run(
            &fixture,
            a_run(vec![a_draft(
                FactCategory::CompanionState,
                Some(FactSubject::Companion),
                text,
            )]),
            &labels,
        );
        assert_eq!((correct.scored(), correct.inverted()), (1, 0));

        let misfiled = score_run(
            &fixture,
            a_run(vec![a_draft(
                FactCategory::UserState,
                Some(FactSubject::User),
                text,
            )]),
            &labels,
        );
        assert_eq!((misfiled.scored(), misfiled.inverted()), (1, 1));
    }

    #[test]
    fn an_ambiguous_label_is_loaded_but_never_scored() {
        let text = "the night was quiet and neither of them said much";
        let fixture = a_fixture(crate::compaction::fixtures::synthetic_range());
        let report = score_run(
            &fixture,
            a_run(vec![a_draft(
                FactCategory::UserState,
                Some(FactSubject::User),
                text,
            )]),
            &gold(&[(text, About::Ambiguous)]),
        );
        assert_eq!((report.scored(), report.inverted()), (0, 0));
        assert!(report.unlabelled.is_empty());
    }

    #[test]
    fn an_unlabelled_subject_bearing_item_is_reported_and_a_milestone_is_not() {
        let fixture = a_fixture(crate::compaction::fixtures::synthetic_range());
        let report = score_run(
            &fixture,
            a_run(vec![
                a_draft(
                    FactCategory::CompanionState,
                    Some(FactSubject::Companion),
                    "has no gold label yet",
                ),
                a_draft(FactCategory::Milestone, None, "milestones carry no subject"),
            ]),
            &GoldLabels::default(),
        );
        assert_eq!(report.unlabelled.len(), 1);
        assert_eq!(report.unlabelled[0].0, FactCategory::CompanionState);
    }

    #[test]
    fn a_people_item_naming_either_principal_is_flagged_and_a_third_party_is_not() {
        let fixture = a_fixture(crate::compaction::fixtures::synthetic_range());
        let person = |name: &str| {
            let mut draft = a_draft(FactCategory::Person, None, &format!("{name}: a neighbor"));
            draft.subject = Some(FactSubject::Person(name.to_string()));
            draft
        };
        let report = score_run(
            &fixture,
            a_run(vec![person("Jinx"), person("eric"), person("Wren")]),
            &GoldLabels::default(),
        );
        assert_eq!(report.principals_in_people, vec!["Jinx", "eric"]);
    }

    #[test]
    fn length_buckets_split_at_the_prompts_limit_and_the_validators() {
        let mut buckets = LengthBuckets::default();
        for words in [PROMPT_WORD_LIMIT, PROMPT_WORD_LIMIT + 1, MAX_ITEM_WORDS + 1] {
            buckets.add(words);
        }
        assert_eq!(
            buckets,
            LengthBuckets {
                within_prompt_limit: 1,
                over_prompt_under_validator: 1,
                over_validator_limit: 1,
            }
        );
    }

    #[test]
    fn quote_counts_separate_not_verbatim_from_wrong_speaker() {
        let fixture = a_fixture(crate::compaction::fixtures::synthetic_range());
        let rejected = |category, reason: RejectReason| {
            let mut draft = a_draft(category, None, "a quote");
            draft.rejected_reason = Some(reason.to_string());
            draft
        };
        let report = score_run(
            &fixture,
            a_run(vec![
                a_draft(FactCategory::Rule, None, "an accepted rule"),
                rejected(FactCategory::KeyQuote, RejectReason::QuoteNotVerbatim),
                rejected(FactCategory::KeyQuote, RejectReason::SpeakerMismatch),
                rejected(FactCategory::Rule, RejectReason::NoSources),
            ]),
            &GoldLabels::default(),
        );
        assert_eq!(
            report.quotes,
            QuoteCounts {
                rules: 2,
                key_quotes: 2,
                accepted: 1,
                not_verbatim: 1,
                wrong_speaker: 1,
                other_rejection: 1,
            }
        );
    }

    #[test]
    fn the_committed_synthetic_range_is_always_available_as_a_fixture() {
        let fixture = synthetic_fixture();
        assert_eq!(fixture.messages.len(), 20);
        assert_eq!(fixture.id_span(), "46-65");
    }

    #[test]
    fn an_absent_fixture_directory_yields_no_local_ranges_rather_than_panicking() {
        assert!(load_local_fixtures(Path::new("/nonexistent/eval/fixtures")).is_empty());
    }

    #[test]
    fn snippet_truncates_on_a_character_boundary() {
        assert_eq!(snippet("héllo wörld", 5), "héllo...");
        assert_eq!(snippet("short", 40), "short");
    }
}
