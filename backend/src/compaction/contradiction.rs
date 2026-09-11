//! The judge that checks compaction output against curated running thoughts
//! (#219): this is what gives running thoughts (#214/#215) their purpose —
//! a checkpoint's summary and facts are **checked against** the companion's
//! own curated notes before they can be committed, closing the gap where
//! `compaction_facts` are validated hard (`validate.rs`) but the free-form
//! summary feeds every later prompt unchecked.
//!
//! Pure with respect to storage: [`check`] takes an [`Extractor`] and plain
//! data, never a `Database` or a [`crate::compaction::store::CompactionStore`].
//! [`extract::fill_draft`](crate::compaction::extract::fill_draft) is the one
//! caller, via the [`ThoughtCheck`] bundle below; the persistence seams
//! ([`ContradictionStore`] for the `compaction_contradictions` table,
//! [`CoveringThoughts`] over #215's `RunningThoughtStore`) live in this same
//! module because nothing outside this check needs them.
//!
//! Model output is filtered in code, not trusted (the #212 lesson, applied
//! to this judge): [`parse_contradictions`] keeps a reported contradiction
//! only if both indices are in range and its `quote` is a whitespace-
//! normalised verbatim substring of the candidate it names. A judge that
//! "finds" a contradiction it cannot quote is dropped, the same discipline
//! [`crate::compaction::validate::RejectReason::QuoteNotVerbatim`] applies
//! to the extractor itself.
#![allow(dead_code)]

use serde::Deserialize;

use crate::context_manager::ContextManager;
use crate::llm::Extractor;
use crate::running_thoughts::store::RunningThoughtStore;

/// One curated running thought as the judge sees it. Built from #215's
/// `RunningThought` (id, text, edited) via [`CoveringThoughts`] — a narrow
/// copy so this module does not depend on that struct's full shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CuratedThought {
    pub id: i64,
    pub text: String,
    pub edited: bool,
}

/// What the judge is asked about: the draft summary, or one accepted fact,
/// named by its index into the caller's own `Vec<FactDraft>`/`ReviewedItem`s
/// — never a `Fact.id`, so this module never needs to know whether the
/// caller is judging a fresh draft or a review's already-stored items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandidateKey {
    Summary,
    Fact(usize),
}

/// One thing the judge is asked about: `key` identifies it back to the
/// caller, `text` is what actually gets shown to the model.
#[derive(Debug, Clone, Copy)]
pub struct Candidate<'a> {
    pub key: CandidateKey,
    pub text: &'a str,
}

/// A kept verdict: which candidate, which thought, and the verbatim passage
/// of the candidate the judge quoted. `thought_text` is snapshotted here
/// (rather than requiring a caller to join back to the thought later) so the
/// review card renders without a join — the thought may be edited or
/// deleted by the time anyone reads a stored row, and #217's regenerate
/// rewrites thought ids entirely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contradiction {
    pub candidate: CandidateKey,
    pub thought_id: i64,
    pub thought_text: String,
    pub quote: String,
}

/// Token budget for one contradiction-check completion: a handful of
/// `{"candidate": N, "thought": N, "quote": "..."}` entries, generous enough
/// for the grammar's 12-item cap without asking a small local model for more
/// than it can produce in one call.
pub const CONTRADICTION_MAX_TOKENS: usize = 400;

/// The GBNF grammar constraining the judge's output to `{"contradictions":
/// [{"candidate": N, "thought": N, "quote": "..."}, ...]}`, arrays bounded to
/// 12 items so a runaway model cannot fill the context. Every rule
/// definition is a single physical line, for the same reason
/// [`crate::compaction::extract::EXTRACTION_GRAMMAR_BASE`]'s doc comment
/// explains: llama.cpp's grammar parser only treats a bare newline as
/// insignificant inside an unclosed `(...)` group.
pub const CONTRADICTION_GRAMMAR: &str = r#"root ::= "{" ws "\"contradictions\"" ws ":" ws item-array ws "}"
item-array ::= "[" ws (item (ws "," ws item){0,11})? ws "]"
item ::= "{" ws "\"candidate\"" ws ":" ws index ws "," ws "\"thought\"" ws ":" ws index ws "," ws "\"quote\"" ws ":" ws string ws "}"
index ::= [0-9]{1,3}
string ::= "\"" char{1,400} "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \n\t]{0,20}
"#;

/// Collapses runs of whitespace to a single space and trims the ends,
/// mirroring `validate.rs::normalise_ws` — kept as its own copy here since
/// that one is private to `validate.rs` and this module has no other reason
/// to depend on it.
fn normalise_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Renders the thoughts and candidates the judge is asked about, plus the
/// instruction and one worked example using neutral synthetic text (not the
/// motivating fixture's own wounds example, so a test asserting the fixture
/// text appears in a *verdict* can't pass merely because the prompt itself
/// echoed it).
pub fn build_contradiction_prompt(thoughts: &[CuratedThought], candidates: &[Candidate]) -> String {
    let mut prompt = String::new();
    prompt.push_str(
        "You are checking new text against the companion's own notes for contradictions.\n",
    );
    prompt.push_str("The companion's own notes (treat these as true):\n");
    if thoughts.is_empty() {
        prompt.push_str("none\n");
    } else {
        for (i, thought) in thoughts.iter().enumerate() {
            if thought.edited {
                prompt.push_str(&format!("T{i} (confirmed by the user): {}\n", thought.text));
            } else {
                prompt.push_str(&format!("T{i}: {}\n", thought.text));
            }
        }
    }
    prompt.push('\n');

    prompt.push_str("Candidate passages to check:\n");
    if candidates.is_empty() {
        prompt.push_str("none\n");
    } else {
        for (i, candidate) in candidates.iter().enumerate() {
            prompt.push_str(&format!("C{i}: {}\n", candidate.text));
        }
    }
    prompt.push('\n');

    prompt.push_str(
        "For each candidate that states something a note rules out, output its index, the \
         note's index, and the exact words of the candidate that conflict. Output an empty \
         list when nothing conflicts.\n",
    );
    prompt.push_str(
        "Example: if T0 says \"only one lamp survived the storm\" and C0 says \"both lamps \
         were destroyed\", output {\"candidate\": 0, \"thought\": 0, \"quote\": \"both lamps \
         were destroyed\"}.\n",
    );
    prompt.push_str("Produce JSON with exactly this key:\n");
    prompt.push_str(r#"{"contradictions": [{"candidate": 0, "thought": 0, "quote": "..."}]}"#);
    prompt
}

#[derive(Debug, Deserialize)]
struct RawContradictions {
    contradictions: Vec<RawItem>,
}

#[derive(Debug, Deserialize)]
struct RawItem {
    candidate: usize,
    thought: usize,
    quote: String,
}

/// Parses the judge's raw output and applies the code-side filter the #212
/// lesson demands: a reported contradiction is kept only if both indices are
/// in range and `quote` is a whitespace-normalised verbatim substring of the
/// candidate it names. Two verdicts naming the same candidate keep only the
/// first (in the order the model emitted them). A `serde_json::Error` here
/// is a genuine parse failure (bad JSON shape), never a filtered-out verdict
/// — those are silently dropped, not errors.
pub fn parse_contradictions(
    raw: &str,
    thoughts: &[CuratedThought],
    candidates: &[Candidate],
) -> Result<Vec<Contradiction>, serde_json::Error> {
    let parsed: RawContradictions = serde_json::from_str(raw)?;

    let mut seen = std::collections::HashSet::new();
    let mut kept = Vec::new();
    for item in parsed.contradictions {
        let Some(candidate) = candidates.get(item.candidate) else {
            continue;
        };
        let Some(thought) = thoughts.get(item.thought) else {
            continue;
        };
        if !seen.insert(candidate.key) {
            continue;
        }
        let normalised_quote = normalise_ws(&item.quote);
        if normalised_quote.is_empty() || !normalise_ws(candidate.text).contains(&normalised_quote)
        {
            continue;
        }
        kept.push(Contradiction {
            candidate: candidate.key,
            thought_id: thought.id,
            thought_text: thought.text.clone(),
            quote: item.quote,
        });
    }
    Ok(kept)
}

/// Splits `candidates` into batches that each fit `window - scaffold_tokens`
/// tokens, greedily, using [`ContextManager::estimate_tokens`] over each
/// candidate's text — mirroring
/// [`crate::compaction::extract::chunk_range`]'s own greedy-boundary
/// algorithm. A single candidate that alone exceeds the budget still becomes
/// its own (oversized) batch rather than being dropped or split. Never
/// emits an empty batch; an empty `candidates` slice returns no batches at
/// all.
pub fn pack_candidates<'a>(
    candidates: &'a [Candidate<'a>],
    scaffold_tokens: usize,
    window: usize,
) -> Vec<&'a [Candidate<'a>]> {
    let budget = window.saturating_sub(scaffold_tokens);

    let mut batches = Vec::new();
    let mut start = 0;
    let mut running_tokens = 0usize;
    for (i, candidate) in candidates.iter().enumerate() {
        let tokens = ContextManager::estimate_tokens(candidate.text);
        if i > start && running_tokens + tokens > budget {
            batches.push(&candidates[start..i]);
            start = i;
            running_tokens = 0;
        }
        running_tokens += tokens;
    }
    if start < candidates.len() {
        batches.push(&candidates[start..]);
    }
    batches
}

/// Drops the oldest thoughts (lowest id, per [`CoveringThoughts::covering`]'s
/// `ORDER BY id`) one at a time until the rendered "notes" block alone fits
/// `window` tokens, or only one thought is left. Logs how many were dropped,
/// if any. This is the fallback for a range so densely noted that the
/// thoughts alone would leave no room for even one candidate.
fn fit_thoughts(thoughts: &[CuratedThought], window: usize) -> Vec<CuratedThought> {
    let mut kept: Vec<CuratedThought> = thoughts.to_vec();
    while kept.len() > 1
        && ContextManager::estimate_tokens(&build_contradiction_prompt(&kept, &[])) >= window
    {
        kept.remove(0);
    }
    if kept.len() < thoughts.len() {
        eprintln!(
            "compaction contradiction check: dropped {} oldest thought(s) to fit the context window",
            thoughts.len() - kept.len()
        );
    }
    kept
}

/// Runs the judge over `candidates` against `thoughts`: `Ok(vec![])` with
/// **no model call at all** when either slice is empty (#219 AC: a
/// checkpoint with no covering thoughts costs nothing extra). Otherwise
/// batches `candidates` via [`pack_candidates`] against `extractor`'s own
/// context window (less [`CONTRADICTION_MAX_TOKENS`] headroom for the
/// completion), running one `extract` call per batch; a parse failure
/// retries once with the serde error appended to the prompt (matching
/// `extract::extract_chunk`'s shape), then surfaces as `io::Error::other`.
pub fn check(
    extractor: &impl Extractor,
    thoughts: &[CuratedThought],
    candidates: &[Candidate],
) -> std::io::Result<Vec<Contradiction>> {
    if thoughts.is_empty() || candidates.is_empty() {
        return Ok(Vec::new());
    }

    let window = extractor
        .context_window()
        .saturating_sub(CONTRADICTION_MAX_TOKENS);
    let thoughts = fit_thoughts(thoughts, window);
    let scaffold_tokens =
        ContextManager::estimate_tokens(&build_contradiction_prompt(&thoughts, &[]));
    let batches = pack_candidates(candidates, scaffold_tokens, window);

    let mut found = Vec::new();
    for batch in batches {
        let prompt = build_contradiction_prompt(&thoughts, batch);
        let first_raw =
            extractor.extract(&prompt, CONTRADICTION_GRAMMAR, CONTRADICTION_MAX_TOKENS)?;
        match parse_contradictions(&first_raw, &thoughts, batch) {
            Ok(hits) => found.extend(hits),
            Err(parse_error) => {
                let retry_prompt = format!(
                    "{prompt}\nYour previous output was not valid JSON: {parse_error}. Produce the JSON again."
                );
                let second_raw = extractor.extract(
                    &retry_prompt,
                    CONTRADICTION_GRAMMAR,
                    CONTRADICTION_MAX_TOKENS,
                )?;
                let hits = parse_contradictions(&second_raw, &thoughts, batch).map_err(|e| {
                    std::io::Error::other(format!(
                        "contradiction check output did not parse after one retry: {e}"
                    ))
                })?;
                found.extend(hits);
            }
        }
    }
    Ok(found)
}

/// One stored `compaction_contradictions` row. `fact_id: None` means the
/// contradiction is against the checkpoint's summary rather than a fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredContradiction {
    pub fact_id: Option<i64>,
    pub thought_id: i64,
    pub thought_text: String,
    pub quote: String,
}

/// The persistence seam for `compaction_contradictions` rows. Kept separate
/// from [`crate::compaction::store::CompactionStore`] (ISP): adding these
/// methods there would force an edit on every one of that trait's five
/// impls, almost none of which have anything to do with running-thought
/// contradictions.
pub trait ContradictionStore {
    /// Deletes every row for `compaction_id` and inserts `rows`, in one
    /// transaction, so the table always reflects exactly the most recent
    /// check's verdict for that draft.
    fn replace_contradictions(
        &self,
        compaction_id: i64,
        rows: &[StoredContradiction],
    ) -> rusqlite::Result<()>;
    fn contradictions_for(&self, compaction_id: i64) -> rusqlite::Result<Vec<StoredContradiction>>;
}

pub(crate) fn replace_contradictions_on(
    con: &mut rusqlite::Connection,
    compaction_id: i64,
    rows: &[StoredContradiction],
) -> rusqlite::Result<()> {
    let tx = con.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    tx.execute(
        "DELETE FROM compaction_contradictions WHERE compaction_id = ?",
        rusqlite::params![compaction_id],
    )?;
    for row in rows {
        tx.execute(
            "INSERT INTO compaction_contradictions (compaction_id, fact_id, thought_id, thought_text, quote)
             VALUES (?, ?, ?, ?, ?)",
            rusqlite::params![compaction_id, row.fact_id, row.thought_id, row.thought_text, row.quote],
        )?;
    }
    tx.commit()
}

pub(crate) fn contradictions_for_on(
    con: &rusqlite::Connection,
    compaction_id: i64,
) -> rusqlite::Result<Vec<StoredContradiction>> {
    let mut stmt = con.prepare(
        "SELECT fact_id, thought_id, thought_text, quote FROM compaction_contradictions
         WHERE compaction_id = ? ORDER BY id",
    )?;
    let rows = stmt.query_map(rusqlite::params![compaction_id], |row| {
        Ok(StoredContradiction {
            fact_id: row.get(0)?,
            thought_id: row.get(1)?,
            thought_text: row.get(2)?,
            quote: row.get(3)?,
        })
    })?;
    rows.collect()
}

/// Production [`ContradictionStore`], opening `Database::open()` per call,
/// exactly like every other `Database`-backed store in this crate.
pub struct SqliteContradictionStore;

impl ContradictionStore for SqliteContradictionStore {
    fn replace_contradictions(
        &self,
        compaction_id: i64,
        rows: &[StoredContradiction],
    ) -> rusqlite::Result<()> {
        let mut con = crate::database::Database::open()?;
        replace_contradictions_on(&mut con, compaction_id, rows)
    }

    fn contradictions_for(&self, compaction_id: i64) -> rusqlite::Result<Vec<StoredContradiction>> {
        let con = crate::database::Database::open()?;
        contradictions_for_on(&con, compaction_id)
    }
}

/// The thoughts covering one checkpoint's range, as [`fill_draft`]
/// (crate::compaction::extract::fill_draft) needs them. The blanket impl
/// below maps every #215 [`RunningThoughtStore`] onto this narrower
/// interface, so `fill_draft` depends on [`CuratedThought`] rather than
/// `running_thoughts::types::RunningThought`'s full shape.
pub trait CoveringThoughts {
    fn covering(
        &self,
        companion_id: i32,
        from_message_id: i32,
        through_message_id: i32,
    ) -> rusqlite::Result<Vec<CuratedThought>>;
}

impl<T: RunningThoughtStore> CoveringThoughts for T {
    fn covering(
        &self,
        companion_id: i32,
        from_message_id: i32,
        through_message_id: i32,
    ) -> rusqlite::Result<Vec<CuratedThought>> {
        Ok(self
            .in_range(companion_id, from_message_id, through_message_id)?
            .into_iter()
            .map(|t| CuratedThought {
                id: t.id,
                text: t.text,
                edited: t.edited,
            })
            .collect())
    }
}

/// Bundles the two seams [`fill_draft`](crate::compaction::extract::fill_draft)
/// needs for its contradiction step into one parameter, so that function
/// grows by one argument instead of two.
pub struct ThoughtCheck<'a> {
    pub thoughts: &'a dyn CoveringThoughts,
    pub store: &'a dyn ContradictionStore,
}

#[cfg(test)]
pub(crate) struct FixedThoughts(pub Vec<CuratedThought>);

#[cfg(test)]
impl CoveringThoughts for FixedThoughts {
    fn covering(
        &self,
        _companion_id: i32,
        _from_message_id: i32,
        _through_message_id: i32,
    ) -> rusqlite::Result<Vec<CuratedThought>> {
        Ok(self.0.clone())
    }
}

#[cfg(test)]
pub(crate) struct RecordingContradictionStore {
    rows: std::sync::Mutex<std::collections::HashMap<i64, Vec<StoredContradiction>>>,
}

#[cfg(test)]
impl RecordingContradictionStore {
    pub(crate) fn new() -> Self {
        Self {
            rows: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }
}

#[cfg(test)]
impl ContradictionStore for RecordingContradictionStore {
    fn replace_contradictions(
        &self,
        compaction_id: i64,
        rows: &[StoredContradiction],
    ) -> rusqlite::Result<()> {
        self.rows
            .lock()
            .unwrap()
            .insert(compaction_id, rows.to_vec());
        Ok(())
    }

    fn contradictions_for(&self, compaction_id: i64) -> rusqlite::Result<Vec<StoredContradiction>> {
        Ok(self
            .rows
            .lock()
            .unwrap()
            .get(&compaction_id)
            .cloned()
            .unwrap_or_default())
    }
}

/// A [`FixedThoughts`]/[`RecordingContradictionStore`] pair owned together,
/// so a caller that needs a no-op [`ThoughtCheck`] (every `fill_draft` test
/// that predates #219, plus every test that only cares about the extraction
/// pipeline around this check) can build one without juggling two locals.
#[cfg(test)]
pub(crate) struct TestThoughtCheck {
    thoughts: FixedThoughts,
    store: RecordingContradictionStore,
}

#[cfg(test)]
impl TestThoughtCheck {
    /// No covering thoughts: [`ThoughtCheck::thoughts`] returns `Ok(vec![])`
    /// regardless of the range asked about, so [`check`] never calls the
    /// model.
    pub(crate) fn none() -> Self {
        Self {
            thoughts: FixedThoughts(Vec::new()),
            store: RecordingContradictionStore::new(),
        }
    }

    pub(crate) fn with_thoughts(thoughts: Vec<CuratedThought>) -> Self {
        Self {
            thoughts: FixedThoughts(thoughts),
            store: RecordingContradictionStore::new(),
        }
    }

    pub(crate) fn check(&self) -> ThoughtCheck<'_> {
        ThoughtCheck {
            thoughts: &self.thoughts,
            store: &self.store,
        }
    }

    pub(crate) fn stored_for(&self, compaction_id: i64) -> Vec<StoredContradiction> {
        self.store.contradictions_for(compaction_id).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::FakeExtractor;

    fn thought(id: i64, text: &str, edited: bool) -> CuratedThought {
        CuratedThought {
            id,
            text: text.to_string(),
            edited,
        }
    }

    fn summary_candidate(text: &str) -> Candidate<'_> {
        Candidate {
            key: CandidateKey::Summary,
            text,
        }
    }

    // --- CONTRADICTION_GRAMMAR ---

    #[test]
    fn grammar_has_a_root_rule_and_no_nul_bytes() {
        assert!(CONTRADICTION_GRAMMAR.contains("root ::="));
        assert!(!CONTRADICTION_GRAMMAR.contains('\0'));
        assert_eq!(
            crate::compaction::extract::check_gbnf_rule_boundaries(CONTRADICTION_GRAMMAR),
            Ok(())
        );
    }

    // --- build_contradiction_prompt ---

    #[test]
    fn prompt_mentions_every_thought_and_candidate_index_and_marks_edited_ones() {
        let thoughts = vec![
            thought(1, "the guest room key is under the mat", true),
            thought(2, "the porch key is still missing", false),
        ];
        let candidates = vec![
            summary_candidate("they found the guest room key on the shelf"),
            Candidate {
                key: CandidateKey::Fact(0),
                text: "the porch key was found",
            },
        ];

        let prompt = build_contradiction_prompt(&thoughts, &candidates);

        assert!(prompt.contains("T0 (confirmed by the user):"));
        assert!(prompt.contains("T1:"));
        assert!(!prompt.contains("T1 (confirmed by the user):"));
        assert!(prompt.contains("C0:"));
        assert!(prompt.contains("C1:"));
    }

    #[test]
    fn the_worked_example_does_not_reuse_the_motivating_fixtures_own_wounds_text() {
        // So a test asserting that text appears in a *verdict* cannot pass
        // merely because the prompt itself echoed it back.
        let prompt = build_contradiction_prompt(&[], &[]);
        assert!(!prompt.contains("wounds"));
    }

    #[test]
    fn an_empty_thoughts_or_candidates_list_still_renders_a_well_formed_prompt() {
        let prompt = build_contradiction_prompt(&[], &[]);
        assert!(prompt.contains("none"));
    }

    // --- parse_contradictions ---

    #[test]
    fn the_motivating_case_is_caught() {
        // #219's whole reason for existing: a thought establishing only one
        // character was wounded, against a summary claiming both were.
        let thoughts = vec![thought(7, "He stitched my wounds; he was not hurt.", false)];
        let candidates = vec![summary_candidate(
            "They spent the evening quietly, and tended to each other's wounds.",
        )];
        let raw = r#"{"contradictions": [{"candidate": 0, "thought": 0, "quote": "tended to each other's wounds"}]}"#;

        let found = parse_contradictions(raw, &thoughts, &candidates).unwrap();

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].candidate, CandidateKey::Summary);
        assert_eq!(found[0].thought_id, 7);
        assert_eq!(found[0].quote, "tended to each other's wounds");
    }

    #[test]
    fn a_verdict_whose_quote_is_not_in_the_candidate_is_dropped() {
        let thoughts = vec![thought(1, "only one lamp survived", false)];
        let candidates = vec![summary_candidate("both lamps were fine")];
        let raw = r#"{"contradictions": [{"candidate": 0, "thought": 0, "quote": "not actually in the text"}]}"#;

        let found = parse_contradictions(raw, &thoughts, &candidates).unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn an_out_of_range_candidate_or_thought_index_is_dropped() {
        let thoughts = vec![thought(1, "only one lamp survived", false)];
        let candidates = vec![summary_candidate("both lamps were fine")];

        let bad_candidate = r#"{"contradictions": [{"candidate": 5, "thought": 0, "quote": "both lamps were fine"}]}"#;
        assert!(parse_contradictions(bad_candidate, &thoughts, &candidates)
            .unwrap()
            .is_empty());

        let bad_thought = r#"{"contradictions": [{"candidate": 0, "thought": 5, "quote": "both lamps were fine"}]}"#;
        assert!(parse_contradictions(bad_thought, &thoughts, &candidates)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn two_verdicts_on_one_candidate_keep_only_the_first() {
        let thoughts = vec![
            thought(1, "only one lamp survived", false),
            thought(2, "the door was locked all night", false),
        ];
        let candidates = vec![summary_candidate(
            "both lamps were fine and the door was open",
        )];
        let raw = r#"{"contradictions": [
            {"candidate": 0, "thought": 0, "quote": "both lamps were fine"},
            {"candidate": 0, "thought": 1, "quote": "the door was open"}
        ]}"#;

        let found = parse_contradictions(raw, &thoughts, &candidates).unwrap();

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].thought_id, 1);
    }

    #[test]
    fn empty_thoughts_short_circuits_to_no_model_call() {
        let extractor = FakeExtractor::returning(Vec::<std::io::Result<String>>::new());
        let candidates = vec![summary_candidate("anything")];

        let found = check(&extractor, &[], &candidates).unwrap();

        assert!(found.is_empty());
        assert!(extractor.prompts.lock().unwrap().is_empty());
    }

    #[test]
    fn empty_candidates_short_circuits_to_no_model_call() {
        let extractor = FakeExtractor::returning(Vec::<std::io::Result<String>>::new());
        let thoughts = vec![thought(1, "only one lamp survived", false)];

        let found = check(&extractor, &thoughts, &[]).unwrap();

        assert!(found.is_empty());
        assert!(extractor.prompts.lock().unwrap().is_empty());
    }

    #[test]
    fn check_runs_the_model_once_and_returns_the_filtered_verdict() {
        let extractor = FakeExtractor::returning(vec![Ok(
            r#"{"contradictions": [{"candidate": 0, "thought": 0, "quote": "both lamps were fine"}]}"#
                .to_string(),
        )]);
        let thoughts = vec![thought(3, "only one lamp survived", false)];
        let candidates = vec![summary_candidate("both lamps were fine tonight")];

        let found = check(&extractor, &thoughts, &candidates).unwrap();

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].thought_id, 3);
        assert_eq!(extractor.prompts.lock().unwrap().len(), 1);
    }

    // --- pack_candidates ---

    #[test]
    fn pack_candidates_returns_one_batch_when_everything_fits() {
        let candidates = vec![
            summary_candidate("short"),
            Candidate {
                key: CandidateKey::Fact(0),
                text: "also short",
            },
        ];
        let batches = pack_candidates(&candidates, 0, 10_000);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 2);
    }

    #[test]
    fn pack_candidates_splits_when_over_the_window_and_never_emits_an_empty_batch() {
        let long_text = "word ".repeat(200);
        let candidates: Vec<Candidate> = (0..6)
            .map(|i| Candidate {
                key: CandidateKey::Fact(i),
                text: long_text.as_str(),
            })
            .collect();

        let batches = pack_candidates(&candidates, 0, 150);

        assert!(batches.len() > 1);
        for batch in &batches {
            assert!(!batch.is_empty());
        }
    }

    #[test]
    fn pack_candidates_over_an_empty_slice_returns_no_batches() {
        let candidates: Vec<Candidate> = Vec::new();
        assert!(pack_candidates(&candidates, 0, 10_000).is_empty());
    }

    // --- replace_contradictions_on / contradictions_for_on ---

    /// Builds the schema `replace_contradictions_on`/`contradictions_for_on`
    /// need (`companion`, `compactions`, `compaction_facts`,
    /// `compaction_contradictions`) on a temp-file connection, and seeds one
    /// companion, one compaction and one fact row. Mirrors
    /// `store.rs`'s own `fresh_db` test helper.
    fn fresh_db() -> (tempfile::TempDir, rusqlite::Connection) {
        let dir = tempfile::TempDir::new().unwrap();
        let con = crate::database::Database::open_at(dir.path().join("t.db")).unwrap();
        con.execute(
            "CREATE TABLE IF NOT EXISTS companion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, persona TEXT, example_dialogue TEXT, first_message TEXT,
                long_term_mem INTEGER, short_term_mem INTEGER, roleplay BOOLEAN,
                dialogue_tuning BOOLEAN, avatar_path TEXT, compacted_through INTEGER
            )",
            [],
        )
        .unwrap();
        crate::compaction::store::create_tables(&con).unwrap();
        con.execute(
            "INSERT INTO companion (id, name, persona, example_dialogue, first_message, long_term_mem, short_term_mem, roleplay, dialogue_tuning, avatar_path) VALUES (1, 'Test', '', '', '', 0, 0, 0, 0, '')",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO compactions (id, companion_id, from_message_id, through_message_id, status, trigger, created_at) VALUES (1, 1, 1, 5, 'draft', 'threshold', 'now')",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO compaction_facts (id, compaction_id, category, text, sources, canon, active) VALUES (1, 1, 'milestone', 'a fact', '[1]', 1, 1)",
            [],
        )
        .unwrap();
        (dir, con)
    }

    fn a_row(fact_id: Option<i64>, thought_id: i64) -> StoredContradiction {
        StoredContradiction {
            fact_id,
            thought_id,
            thought_text: "the thought's own words".to_string(),
            quote: "the conflicting quote".to_string(),
        }
    }

    #[test]
    fn replace_then_read_round_trips_a_null_fact_id() {
        let (_dir, mut con) = fresh_db();
        replace_contradictions_on(&mut con, 1, &[a_row(None, 7)]).unwrap();

        let rows = contradictions_for_on(&con, 1).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].fact_id, None);
        assert_eq!(rows[0].thought_id, 7);
    }

    #[test]
    fn replace_then_read_round_trips_a_fact_id() {
        let (_dir, mut con) = fresh_db();
        replace_contradictions_on(&mut con, 1, &[a_row(Some(1), 9)]).unwrap();

        let rows = contradictions_for_on(&con, 1).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].fact_id, Some(1));
    }

    #[test]
    fn replace_is_atomic_and_replaces_every_prior_row() {
        let (_dir, mut con) = fresh_db();
        replace_contradictions_on(&mut con, 1, &[a_row(None, 1), a_row(Some(1), 2)]).unwrap();
        assert_eq!(contradictions_for_on(&con, 1).unwrap().len(), 2);

        replace_contradictions_on(&mut con, 1, &[a_row(None, 3)]).unwrap();
        let rows = contradictions_for_on(&con, 1).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].thought_id, 3);
    }

    #[test]
    fn replacing_with_an_empty_slice_clears_every_row() {
        let (_dir, mut con) = fresh_db();
        replace_contradictions_on(&mut con, 1, &[a_row(None, 1)]).unwrap();
        replace_contradictions_on(&mut con, 1, &[]).unwrap();
        assert!(contradictions_for_on(&con, 1).unwrap().is_empty());
    }

    #[test]
    fn deleting_the_checkpoint_cascades_to_its_contradiction_rows() {
        let (_dir, mut con) = fresh_db();
        replace_contradictions_on(&mut con, 1, &[a_row(None, 1)]).unwrap();

        con.execute("DELETE FROM compactions WHERE id = 1", [])
            .unwrap();

        assert!(contradictions_for_on(&con, 1).unwrap().is_empty());
    }

    #[test]
    fn deleting_the_fact_cascades_to_its_contradiction_row() {
        let (_dir, mut con) = fresh_db();
        replace_contradictions_on(&mut con, 1, &[a_row(Some(1), 1)]).unwrap();

        con.execute("DELETE FROM compaction_facts WHERE id = 1", [])
            .unwrap();

        assert!(contradictions_for_on(&con, 1).unwrap().is_empty());
    }
}
