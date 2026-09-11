//! Commits a reviewed compaction draft, or discards it (#175).
//!
//! [`commit`] promotes the fact rows #185's `fill_draft` already stored
//! (accepted and rejected alike): it never inserts a `compaction_facts` row
//! itself, only rewrites the reviewed text/verdict onto the rows named by
//! [`ReviewedItem::fact_id`], via [`crate::compaction::store::CommitRecord`]
//! and [`CompactionStore::commit_checkpoint`]. [`discard`] is the trivial
//! counterpart: it only flips the checkpoint's status, leaving the fact rows
//! exactly as extraction stored them (invisible to
//! [`crate::compaction::store::active_facts_on`] once the checkpoint stops
//! being `committed`-eligible).
//!
//! [`plan_commit`] is the pure decision core — no store, no I/O beyond the
//! summary merger [`commit`] hands it — so the supersede/merge/over-budget
//! rules are unit-tested without a database. [`CommitDeps`] bundles the two
//! seams later issues plug into: [`SummaryMerger`] (#175's own
//! [`crate::compaction::merge::LlmSummaryMerger`] in production) and
//! [`CommitObserver`] (#176 attitude, #177 persons, #178 tantivy).

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::compaction::context::CompactionContext;
use crate::compaction::render::overlays_and_rules_fit;
use crate::compaction::store::{CommitRecord, CompactionStore, FactPromotion};
use crate::compaction::types::{Checkpoint, CompactionStatus, Fact, FactCategory, FactDraft};
use crate::context_manager::ContextManager;
use crate::database::Database;

/// One reviewed item: `fact_id` names the stored `compaction_facts` row
/// (#185's `fill_draft` inserted it) this item edits; `draft` is the
/// reviewed content, already re-validated by #179's `apply_review`, so
/// `commit` trusts `draft.rejected_reason` as given — a struck item
/// (`accepted: false`) is expected to carry `rejected_reason = "struck at
/// review"`. `accepted` is informational (mirrors `draft.rejected_reason.is_none()`);
/// `commit` derives active/inactive from `rejected_reason` alone, the same
/// rule [`crate::compaction::store::insert_facts_on`] uses.
#[derive(Debug, Clone, PartialEq)]
pub struct ReviewedItem {
    pub fact_id: i64,
    pub draft: FactDraft,
    pub accepted: bool,
}

/// A draft's whole review: every stored fact gets exactly one
/// [`ReviewedItem`], plus the reviewed summary text.
#[derive(Debug, Clone, PartialEq)]
pub struct ReviewedDraft {
    pub draft_id: i64,
    pub items: Vec<ReviewedItem>,
    pub summary: String,
}

/// The token budget and display names [`commit`]'s over-budget guard and
/// summary-merge decision need. The caller (#179's handler) fills
/// `compaction_slice_tokens` from `ContextManager::compaction_token_budget`,
/// `rolling_summary_tokens` as half of it, and the names from
/// `UserView`/`CompanionView`.
#[derive(Debug, Clone, PartialEq)]
pub struct CommitBudget {
    pub compaction_slice_tokens: usize,
    pub rolling_summary_tokens: usize,
    pub user_name: String,
    pub companion_name: String,
}

/// Why [`commit`] or [`discard`] failed.
#[derive(Debug)]
pub enum CommitError {
    /// `get_checkpoint` returned `None` for the given draft id.
    DraftNotFound(i64),
    /// The checkpoint's status is not `Draft`, or it is `Draft` but
    /// `raw_model_output` is still `NULL` (extraction has not finished, so
    /// there is nothing to review yet). `discard` only checks the status,
    /// not `raw_model_output`.
    DraftNotPending { id: i64, status: CompactionStatus },
    /// The user overlay, companion overlay, and rules block together would
    /// exceed the compaction slice; those three are never trimmed, so
    /// nothing was written.
    OverBudget { needed: usize, budget: usize },
    /// The transaction failed; nothing was written. Also the mapping for
    /// `QueryReturnedNoRows` when a `fact_id` in the review does not belong
    /// to the draft. Only ever returned for failures *before* the
    /// checkpoint is durable — `store.commit_checkpoint`'s own doc comment
    /// guarantees an `Err` from it means nothing committed, and every read
    /// `commit` performs after that call succeeds is non-fatal instead of
    /// mapped to this variant.
    Storage(rusqlite::Error),
}

impl fmt::Display for CommitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommitError::DraftNotFound(id) => write!(f, "compaction draft {id} not found"),
            CommitError::DraftNotPending { id, status } => write!(
                f,
                "compaction draft {id} is not pending review (status: {status})"
            ),
            CommitError::OverBudget { needed, budget } => write!(
                f,
                "overlays and rules need {needed} tokens but the compaction slice is only {budget}"
            ),
            CommitError::Storage(e) => write!(f, "compaction storage error: {e}"),
        }
    }
}

impl From<rusqlite::Error> for CommitError {
    fn from(e: rusqlite::Error) -> Self {
        CommitError::Storage(e)
    }
}

/// Why [`SummaryMerger::merge`] could not produce a compressed summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeError {
    ModelUnavailable(String),
}

/// Compresses a rolling summary that has grown past its token budget.
/// [`crate::compaction::merge::LlmSummaryMerger`] is the production impl;
/// tests use `IdentityMerger`/`FailingMerger` below.
pub trait SummaryMerger {
    fn merge(&self, rolling: &str, budget_tokens: usize) -> Result<String, MergeError>;
}

/// Runs after a commit's transaction, with the newly active facts it wrote.
/// `on_committed` is expected to handle its own errors (log and move on);
/// its `Result` return exists so [`commit`] can guarantee an observer
/// failure never fails the commit itself — the data is already durable by
/// the time observers run. #176 (attitude), #177 (persons), and #178
/// (tantivy) each implement one, registered by
/// [`crate::compaction::production_commit_deps`]. `superseded` carries the
/// ids of facts this commit superseded (previously active, now not), since
/// #178's observer needs them to remove stale entries from the tantivy
/// index.
pub trait CommitObserver {
    fn on_committed(
        &self,
        checkpoint: &Checkpoint,
        facts: &[Fact],
        superseded: &[i64],
    ) -> Result<(), String>;
}

/// The dependencies [`commit`] needs beyond the store: the summary merger
/// and the observers to run afterward.
/// [`crate::compaction::production_commit_deps`] builds the production
/// value; tests build one by hand.
pub struct CommitDeps<'a> {
    pub merger: Box<dyn SummaryMerger + 'a>,
    pub observers: Vec<Box<dyn CommitObserver>>,
}

/// Collapses whitespace and case so two differently-formatted quotes of the
/// same line compare equal, matching #175's duplicate-rule/key-quote check.
fn normalize_quote(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// The pure decision core: turns a draft's stored facts, its review, the
/// companion's current active facts, and the latest committed checkpoint
/// into a [`CommitRecord`]. No store access; `deps.merger` may run a model
/// but takes no id or connection.
fn plan_commit(
    draft: &Checkpoint,
    stored_facts: &[Fact],
    review: &ReviewedDraft,
    active_facts: &[Fact],
    prev_committed: Option<&Checkpoint>,
    deps: &CommitDeps<'_>,
    budget: &CommitBudget,
) -> Result<CommitRecord, CommitError> {
    let stored_by_id: HashMap<i64, &Fact> = stored_facts.iter().map(|f| (f.id, f)).collect();

    let mut promote = Vec::with_capacity(review.items.len());
    let mut supersede: Vec<(i64, i64)> = Vec::new();
    let mut merge_into: Vec<(i64, i64, Vec<i32>)> = Vec::new();
    let mut merged_away: HashSet<i64> = HashSet::new();

    for item in &review.items {
        if !stored_by_id.contains_key(&item.fact_id) {
            return Err(CommitError::Storage(rusqlite::Error::QueryReturnedNoRows));
        }
        let is_active = item.draft.rejected_reason.is_none();

        if is_active {
            match item.draft.category {
                FactCategory::CompanionState | FactCategory::UserState => {
                    for &old_id in &item.draft.replaces {
                        match active_facts
                            .iter()
                            .find(|f| f.id == old_id && f.category == item.draft.category)
                        {
                            Some(_) => supersede.push((old_id, item.fact_id)),
                            None => eprintln!(
                                "compaction commit: fact {old_id} named in replaces is no longer an active {}, skipping supersede",
                                item.draft.category
                            ),
                        }
                    }
                }
                FactCategory::Rule | FactCategory::KeyQuote => {
                    let normalized = normalize_quote(&item.draft.text);
                    if let Some(existing) = active_facts.iter().find(|f| {
                        f.category == item.draft.category && normalize_quote(&f.text) == normalized
                    }) {
                        let mut merged_sources = existing.sources.clone();
                        for &source in &item.draft.sources {
                            if !merged_sources.contains(&source) {
                                merged_sources.push(source);
                            }
                        }
                        merge_into.push((existing.id, item.fact_id, merged_sources));
                        merged_away.insert(item.fact_id);
                    }
                }
                FactCategory::Milestone
                | FactCategory::Backstory
                | FactCategory::OpenThread
                | FactCategory::Person => {}
            }
        }

        promote.push(FactPromotion {
            fact_id: item.fact_id,
            draft: item.draft.clone(),
        });
    }

    let (summary, rolling_summary, needs_merge) = match prev_committed {
        None => (review.summary.clone(), String::new(), false),
        Some(prev) => {
            let prev_summary = prev.summary.clone().unwrap_or_default();
            let prev_rolling = prev.rolling_summary.clone().unwrap_or_default();
            let concatenated = format!("{prev_rolling}\n\n{prev_summary}")
                .trim()
                .to_string();
            if ContextManager::estimate_tokens(&concatenated) <= budget.rolling_summary_tokens {
                (review.summary.clone(), concatenated, false)
            } else {
                match deps
                    .merger
                    .merge(&concatenated, budget.rolling_summary_tokens)
                {
                    // Not re-checked against `rolling_summary_tokens` here:
                    // an over-long completion is stored as-is, but
                    // `render::render`'s trim order (#174) still shortens
                    // `story_so_far`'s rolling-summary sentences from the
                    // front at read time if the compaction slice is tight,
                    // so a merger that overshoots the word budget it was
                    // asked for degrades gracefully rather than blowing the
                    // prompt's token budget.
                    Ok(text) => (review.summary.clone(), text, false),
                    Err(_) => (review.summary.clone(), concatenated, true),
                }
            }
        }
    };

    let superseded_ids: HashSet<i64> = supersede.iter().map(|(old, _)| *old).collect();
    let mut would_be_active: Vec<Fact> = active_facts
        .iter()
        .filter(|f| !superseded_ids.contains(&f.id))
        .cloned()
        .collect();
    for item in &review.items {
        if item.draft.rejected_reason.is_none() && !merged_away.contains(&item.fact_id) {
            let stored = stored_by_id[&item.fact_id];
            would_be_active.push(Fact {
                id: stored.id,
                compaction_id: stored.compaction_id,
                category: item.draft.category,
                subject: item.draft.subject.clone(),
                text: item.draft.text.clone(),
                quote_speaker: item.draft.quote_speaker.clone(),
                sources: item.draft.sources.clone(),
                replaces: item.draft.replaces.clone(),
                relation_to: item.draft.relation_to.clone(),
                relation: item.draft.relation.clone(),
                canon: item.draft.canon,
                active: true,
                superseded_by: None,
                rejected_reason: None,
            });
        }
    }

    overlays_and_rules_fit(
        &CompactionContext::from_facts(&would_be_active),
        &budget.user_name,
        &budget.companion_name,
        budget.compaction_slice_tokens,
    )
    .map_err(|over| CommitError::OverBudget {
        needed: over.needed,
        budget: over.budget,
    })?;

    Ok(CommitRecord {
        draft_id: draft.id,
        companion_id: draft.companion_id,
        from_message_id: draft.from_message_id,
        through_message_id: draft.through_message_id,
        summary,
        rolling_summary,
        needs_merge,
        promote,
        supersede,
        merge_into,
    })
}

/// Commits a reviewed draft in one transaction: promotes the fact rows
/// #185's `fill_draft` stored, marks superseded/merged rows, flips the
/// checkpoint to `Committed` with its new summaries, sets
/// `compacted_through`, clears the message cache, then runs `deps.observers`
/// with only the newly active facts. Does not claim `ACTIVE_TURN` itself —
/// #179's handler holds it around this call since the production merger
/// runs the model. Once `store.commit_checkpoint` returns `Ok`, the commit
/// is durable and this function no longer fails: the post-commit read-back
/// of the new facts is non-fatal (an empty slice is fed to the observers
/// instead, and the failure is logged), and each observer's own error is
/// swallowed as documented on [`CommitObserver::on_committed`].
pub fn commit(
    store: &dyn CompactionStore,
    review: ReviewedDraft,
    deps: &CommitDeps<'_>,
    budget: &CommitBudget,
) -> Result<Checkpoint, CommitError> {
    let draft = store
        .get_checkpoint(review.draft_id)?
        .ok_or(CommitError::DraftNotFound(review.draft_id))?;

    if draft.status != CompactionStatus::Draft || draft.raw_model_output.is_none() {
        return Err(CommitError::DraftNotPending {
            id: draft.id,
            status: draft.status,
        });
    }

    let stored_facts = store.facts_for(review.draft_id)?;
    let active_facts = store.active_facts(draft.companion_id)?;
    // The newest committed checkpoint ending *before* this draft's own
    // range, not simply the newest one overall (#181): a re-compaction can
    // start earlier than `compacted_through`, so the highest-id committed
    // checkpoint can lie inside the range this draft is about to re-cover
    // and retire. Folding its summary forward would preserve exactly the
    // stale narrative the re-compaction exists to replace.
    let prev_committed =
        store.latest_committed_before(draft.companion_id, draft.from_message_id)?;

    let record = plan_commit(
        &draft,
        &stored_facts,
        &review,
        &active_facts,
        prev_committed.as_ref(),
        deps,
        budget,
    )?;

    let superseded_ids: Vec<i64> = record.supersede.iter().map(|(old, _)| *old).collect();
    let draft_id = record.draft_id;

    // `commit_checkpoint` re-asserts the `Draft` premise decided above
    // inside its own transaction (`transition_status_on`/its `RecordingStore`
    // equivalent): a discard or a second commit that landed since the read
    // above surfaces here as `QueryReturnedNoRows`, which we re-read and
    // report as the same `DraftNotPending` the caller would have seen had
    // the check run after that race instead of before it.
    let checkpoint = match store.commit_checkpoint(record) {
        Ok(checkpoint) => checkpoint,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            let now = store
                .get_checkpoint(draft_id)?
                .ok_or(CommitError::DraftNotFound(draft_id))?;
            return Err(CommitError::DraftNotPending {
                id: now.id,
                status: now.status,
            });
        }
        Err(e) => return Err(e.into()),
    };

    Database::clear_message_cache();

    // The commit above is already durable — a failure reading the fresh
    // fact rows back must not be reported as a failed commit (that would be
    // the same defect class #226 fixed in `main.rs`'s
    // `commit_reviewed_draft`). Feed the observers an empty slice instead
    // and log it: `AttitudeRecalibrator` ignores `_facts` entirely, and
    // `LtmObserver` still gets the real `superseded_ids` below, so
    // recalibration and superseded-fact removal both still happen. Only
    // the new facts themselves are missing from long-term memory until the
    // index is rebuilt.
    let new_active_facts: Vec<Fact> = match store.facts_for(draft_id) {
        Ok(facts) => facts.into_iter().filter(|f| f.active).collect(),
        Err(e) => {
            eprintln!(
                "compaction commit {draft_id}: committed, but reading back its facts for \
                 observers failed: {e}; new facts will be missing from long-term memory until \
                 POST /api/memory/longTerm/rebuild"
            );
            Vec::new()
        }
    };

    for observer in &deps.observers {
        if let Err(e) = observer.on_committed(&checkpoint, &new_active_facts, &superseded_ids) {
            eprintln!("compaction commit observer failed: {e}");
        }
    }

    Ok(checkpoint)
}

/// Discards a pending draft: only the status flips to `Discarded`. The fact
/// rows #185's `fill_draft` stored are left untouched — they become
/// invisible on their own once the checkpoint stops being
/// `committed`-eligible (see
/// [`crate::compaction::store::active_facts_on`]). Checks only the status,
/// not `raw_model_output`, so a still-extracting draft may be discarded.
/// Uses [`CompactionStore::transition_status`] rather than `update_status`
/// so a commit that landed between the read below and the write fails the
/// write instead of silently undoing it (the interleaving the reviewed
/// facts would otherwise vanish from `active_facts` while
/// `compacted_through` and every observer had already treated them as
/// live).
pub fn discard(store: &dyn CompactionStore, draft_id: i64) -> Result<(), CommitError> {
    let draft = store
        .get_checkpoint(draft_id)?
        .ok_or(CommitError::DraftNotFound(draft_id))?;
    if draft.status != CompactionStatus::Draft {
        return Err(CommitError::DraftNotPending {
            id: draft.id,
            status: draft.status,
        });
    }
    match store.transition_status(
        draft_id,
        CompactionStatus::Draft,
        CompactionStatus::Discarded,
    ) {
        Ok(()) => Ok(()),
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            let now = store
                .get_checkpoint(draft_id)?
                .ok_or(CommitError::DraftNotFound(draft_id))?;
            Err(CommitError::DraftNotPending {
                id: now.id,
                status: now.status,
            })
        }
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::store::RecordingStore;
    use crate::compaction::types::{CompactionTrigger, FactSubject, NewDraft};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// Records every call and returns the input unchanged.
    struct IdentityMerger {
        calls: AtomicUsize,
    }

    impl IdentityMerger {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl SummaryMerger for IdentityMerger {
        fn merge(&self, rolling: &str, _budget_tokens: usize) -> Result<String, MergeError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(rolling.to_string())
        }
    }

    struct FailingMerger;

    impl SummaryMerger for FailingMerger {
        fn merge(&self, _rolling: &str, _budget_tokens: usize) -> Result<String, MergeError> {
            Err(MergeError::ModelUnavailable("no model loaded".to_string()))
        }
    }

    type ObserverCall = (Checkpoint, Vec<Fact>, Vec<i64>);

    #[derive(Default)]
    struct RecordingObserver {
        calls: Mutex<Vec<ObserverCall>>,
    }

    impl CommitObserver for RecordingObserver {
        fn on_committed(
            &self,
            checkpoint: &Checkpoint,
            facts: &[Fact],
            superseded: &[i64],
        ) -> Result<(), String> {
            self.calls.lock().unwrap().push((
                checkpoint.clone(),
                facts.to_vec(),
                superseded.to_vec(),
            ));
            Ok(())
        }
    }

    struct FailingObserver;

    impl CommitObserver for FailingObserver {
        fn on_committed(&self, _: &Checkpoint, _: &[Fact], _: &[i64]) -> Result<(), String> {
            Err("boom".to_string())
        }
    }

    fn deps_with(merger: impl SummaryMerger + 'static) -> CommitDeps<'static> {
        CommitDeps {
            merger: Box::new(merger),
            observers: Vec::new(),
        }
    }

    fn budget() -> CommitBudget {
        CommitBudget {
            compaction_slice_tokens: 10_000,
            rolling_summary_tokens: 1_000,
            user_name: "Eric".to_string(),
            companion_name: "Vi".to_string(),
        }
    }

    fn accepted_item(fact_id: i64, category: FactCategory, text: &str) -> ReviewedItem {
        ReviewedItem {
            fact_id,
            draft: FactDraft {
                category,
                subject: None,
                text: text.to_string(),
                quote_speaker: None,
                sources: vec![1],
                replaces: vec![],
                relation_to: None,
                relation: None,
                canon: true,
                rejected_reason: None,
            },
            accepted: true,
        }
    }

    /// Seeds a draft checkpoint plus the fact rows exactly as #185's
    /// `fill_draft` would leave them (via `insert_draft` + `insert_facts`),
    /// and returns the draft id and the stored facts' ids in insertion
    /// order.
    fn seed_draft(
        store: &RecordingStore,
        companion_id: i32,
        drafts: &[FactDraft],
    ) -> (i64, Vec<i64>) {
        seed_draft_with_range(store, companion_id, 1, 10, drafts)
    }

    /// Like [`seed_draft`], with an explicit range. Tests that seed and
    /// commit a *second* draft for the same companion need this instead of
    /// the [1,10] default: in production a draft's range never overlaps an
    /// already-committed one ([`crate::compaction::range::select_range`]
    /// only ever starts after `compacted_through`, and a re-compaction's own
    /// overlap gets retired by `commit_checkpoint` rather than left as a
    /// second live checkpoint over the same messages) — reusing [1,10] for
    /// both would give `latest_committed_before` (#181) nothing to find.
    fn seed_draft_with_range(
        store: &RecordingStore,
        companion_id: i32,
        from_message_id: i32,
        through_message_id: i32,
        drafts: &[FactDraft],
    ) -> (i64, Vec<i64>) {
        let draft_id = store
            .insert_draft(NewDraft {
                companion_id,
                from_message_id,
                through_message_id,
                trigger: CompactionTrigger::Threshold,
                raw_model_output: Some("raw".to_string()),
            })
            .unwrap();
        let ids = store.insert_facts(draft_id, drafts).unwrap();
        (draft_id, ids)
    }

    #[test]
    fn first_commit_promotes_rows_in_place_and_sets_summary() {
        let store = RecordingStore::new();
        let draft = FactDraft {
            category: FactCategory::UserState,
            subject: None,
            text: "loves cats".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (draft_id, ids) = seed_draft(&store, 1, std::slice::from_ref(&draft));

        let review = ReviewedDraft {
            draft_id,
            items: vec![accepted_item(ids[0], FactCategory::UserState, "loves cats")],
            summary: "a summary".to_string(),
        };

        let deps = deps_with(IdentityMerger::new());
        let checkpoint = commit(&store, review, &deps, &budget()).unwrap();

        assert_eq!(checkpoint.status, CompactionStatus::Committed);
        assert_eq!(checkpoint.summary.as_deref(), Some("a summary"));
        assert_eq!(checkpoint.rolling_summary.as_deref(), Some(""));
        assert_eq!(store.compacted_through(1).unwrap(), Some(10));

        let stored = store.facts_for(draft_id).unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].id, ids[0]);
        assert!(stored[0].active);

        let active = store.active_facts(1).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, ids[0]);
    }

    #[test]
    fn second_commit_folds_first_summary_into_rolling_summary() {
        let store = RecordingStore::new();
        let first_draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "met at the park".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (first_id, first_ids) = seed_draft(&store, 1, std::slice::from_ref(&first_draft));
        let deps = deps_with(IdentityMerger::new());
        commit(
            &store,
            ReviewedDraft {
                draft_id: first_id,
                items: vec![accepted_item(
                    first_ids[0],
                    FactCategory::Milestone,
                    "met at the park",
                )],
                summary: "first summary".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let second_draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "went to dinner".to_string(),
            quote_speaker: None,
            sources: vec![2],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (second_id, second_ids) =
            seed_draft_with_range(&store, 1, 11, 20, std::slice::from_ref(&second_draft));
        let checkpoint = commit(
            &store,
            ReviewedDraft {
                draft_id: second_id,
                items: vec![accepted_item(
                    second_ids[0],
                    FactCategory::Milestone,
                    "went to dinner",
                )],
                summary: "second summary".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        assert_eq!(checkpoint.summary.as_deref(), Some("second summary"));
        assert_eq!(checkpoint.rolling_summary.as_deref(), Some("first summary"));
    }

    #[test]
    fn struck_and_rejected_items_are_stored_inactive_with_their_reason() {
        let store = RecordingStore::new();
        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "a milestone".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (draft_id, ids) = seed_draft(&store, 1, std::slice::from_ref(&draft));

        let mut item = accepted_item(ids[0], FactCategory::Milestone, "a milestone");
        item.accepted = false;
        item.draft.rejected_reason = Some("struck at review".to_string());

        let deps = deps_with(IdentityMerger::new());
        commit(
            &store,
            ReviewedDraft {
                draft_id,
                items: vec![item],
                summary: "summary".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let stored = store.facts_for(draft_id).unwrap();
        assert!(!stored[0].active);
        assert_eq!(
            stored[0].rejected_reason.as_deref(),
            Some("struck at review")
        );
        assert!(store.active_facts(1).unwrap().is_empty());
    }

    #[test]
    fn edited_text_is_written_onto_the_stored_row() {
        let store = RecordingStore::new();
        let draft = FactDraft {
            category: FactCategory::OpenThread,
            subject: None,
            text: "original text".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (draft_id, ids) = seed_draft(&store, 1, std::slice::from_ref(&draft));

        let item = accepted_item(ids[0], FactCategory::OpenThread, "edited text");
        let deps = deps_with(IdentityMerger::new());
        commit(
            &store,
            ReviewedDraft {
                draft_id,
                items: vec![item],
                summary: "summary".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let stored = store.facts_for(draft_id).unwrap();
        assert_eq!(stored[0].text, "edited text");
    }

    #[test]
    fn replaces_supersedes_the_old_active_fact_of_the_same_category() {
        let store = RecordingStore::new();
        let old_draft = FactDraft {
            category: FactCategory::CompanionState,
            subject: Some(FactSubject::Companion),
            text: "is nervous".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (old_draft_id, old_ids) = seed_draft(&store, 1, std::slice::from_ref(&old_draft));
        let deps = deps_with(IdentityMerger::new());
        commit(
            &store,
            ReviewedDraft {
                draft_id: old_draft_id,
                items: vec![accepted_item(
                    old_ids[0],
                    FactCategory::CompanionState,
                    "is nervous",
                )],
                summary: "s1".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let new_draft = FactDraft {
            category: FactCategory::CompanionState,
            subject: Some(FactSubject::Companion),
            text: "is confident now".to_string(),
            quote_speaker: None,
            sources: vec![2],
            replaces: vec![old_ids[0]],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (new_draft_id, new_ids) =
            seed_draft_with_range(&store, 1, 11, 20, std::slice::from_ref(&new_draft));
        commit(
            &store,
            ReviewedDraft {
                draft_id: new_draft_id,
                items: vec![ReviewedItem {
                    fact_id: new_ids[0],
                    draft: new_draft,
                    accepted: true,
                }],
                summary: "s2".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let old_fact = store
            .facts_for(old_draft_id)
            .unwrap()
            .into_iter()
            .find(|f| f.id == old_ids[0])
            .unwrap();
        assert!(!old_fact.active);
        assert_eq!(old_fact.superseded_by, Some(new_ids[0]));

        let active = store.active_facts(1).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, new_ids[0]);
    }

    #[test]
    fn duplicate_rule_merges_sources_and_closes_the_drafts_own_row() {
        let store = RecordingStore::new();
        let first = FactDraft {
            category: FactCategory::Rule,
            subject: None,
            text: "Always knock first".to_string(),
            quote_speaker: Some("user".to_string()),
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (first_draft_id, first_ids) = seed_draft(&store, 1, std::slice::from_ref(&first));
        let deps = deps_with(IdentityMerger::new());
        commit(
            &store,
            ReviewedDraft {
                draft_id: first_draft_id,
                items: vec![ReviewedItem {
                    fact_id: first_ids[0],
                    draft: first,
                    accepted: true,
                }],
                summary: "s1".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let duplicate = FactDraft {
            category: FactCategory::Rule,
            subject: None,
            text: "always   knock FIRST".to_string(),
            quote_speaker: Some("user".to_string()),
            sources: vec![5],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (second_draft_id, second_ids) =
            seed_draft_with_range(&store, 1, 11, 20, std::slice::from_ref(&duplicate));
        commit(
            &store,
            ReviewedDraft {
                draft_id: second_draft_id,
                items: vec![ReviewedItem {
                    fact_id: second_ids[0],
                    draft: duplicate,
                    accepted: true,
                }],
                summary: "s2".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let duplicate_row = store
            .facts_for(second_draft_id)
            .unwrap()
            .into_iter()
            .find(|f| f.id == second_ids[0])
            .unwrap();
        assert!(!duplicate_row.active);
        assert_eq!(duplicate_row.superseded_by, Some(first_ids[0]));

        let existing_row = store
            .facts_for(first_draft_id)
            .unwrap()
            .into_iter()
            .find(|f| f.id == first_ids[0])
            .unwrap();
        assert_eq!(existing_row.sources, vec![1, 5]);

        let active = store.active_facts(1).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, first_ids[0]);
    }

    #[test]
    fn a_fact_id_not_on_the_draft_is_a_storage_error_and_writes_nothing() {
        let store = RecordingStore::new();
        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "a milestone".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (draft_id, ids) = seed_draft(&store, 1, std::slice::from_ref(&draft));

        let bogus_id = ids[0] + 999;
        let item = accepted_item(bogus_id, FactCategory::Milestone, "a milestone");
        let deps = deps_with(IdentityMerger::new());
        let err = commit(
            &store,
            ReviewedDraft {
                draft_id,
                items: vec![item],
                summary: "summary".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CommitError::Storage(rusqlite::Error::QueryReturnedNoRows)
        ));

        let checkpoint = store.get_checkpoint(draft_id).unwrap().unwrap();
        assert_eq!(checkpoint.status, CompactionStatus::Draft);
        let stored = store.facts_for(draft_id).unwrap();
        assert!(stored[0].active);
        assert_eq!(stored[0].text, "a milestone");
    }

    #[test]
    fn a_draft_with_no_raw_model_output_is_not_pending() {
        let store = RecordingStore::new();
        let draft_id = store
            .insert_draft(NewDraft {
                companion_id: 1,
                from_message_id: 1,
                through_message_id: 10,
                trigger: CompactionTrigger::Threshold,
                raw_model_output: None,
            })
            .unwrap();

        let deps = deps_with(IdentityMerger::new());
        let err = commit(
            &store,
            ReviewedDraft {
                draft_id,
                items: vec![],
                summary: "summary".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CommitError::DraftNotPending {
                status: CompactionStatus::Draft,
                ..
            }
        ));
    }

    /// #181 review finding: `Database::mark_stale_for_message_on` discards
    /// a pending draft an edit/delete falls inside
    /// (`discard_draft_containing_on`); `commit` must report that the same
    /// way it reports any other non-`Draft` checkpoint.
    #[test]
    fn commit_on_a_draft_discarded_by_an_in_range_edit_is_not_pending() {
        let store = RecordingStore::new();
        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "a milestone".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (draft_id, ids) = seed_draft(&store, 1, std::slice::from_ref(&draft));

        // Simulates `database.rs::mark_stale_for_message_on`'s
        // `discard_draft_containing_on` call: an edit inside the draft's
        // [1,10] range discards it before it is ever reviewed.
        store
            .transition_status(
                draft_id,
                CompactionStatus::Draft,
                CompactionStatus::Discarded,
            )
            .unwrap();

        let deps = deps_with(IdentityMerger::new());
        let err = commit(
            &store,
            ReviewedDraft {
                draft_id,
                items: vec![accepted_item(
                    ids[0],
                    FactCategory::Milestone,
                    "a milestone",
                )],
                summary: "summary".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            CommitError::DraftNotPending {
                status: CompactionStatus::Discarded,
                ..
            }
        ));
    }

    #[test]
    fn an_already_committed_draft_is_not_pending() {
        let store = RecordingStore::new();
        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "a milestone".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (draft_id, ids) = seed_draft(&store, 1, std::slice::from_ref(&draft));
        let deps = deps_with(IdentityMerger::new());
        commit(
            &store,
            ReviewedDraft {
                draft_id,
                items: vec![accepted_item(
                    ids[0],
                    FactCategory::Milestone,
                    "a milestone",
                )],
                summary: "s1".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let err = commit(
            &store,
            ReviewedDraft {
                draft_id,
                items: vec![],
                summary: "s2".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CommitError::DraftNotPending {
                status: CompactionStatus::Committed,
                ..
            }
        ));
    }

    #[test]
    fn commit_on_an_unknown_draft_is_draft_not_found() {
        let store = RecordingStore::new();
        let deps = deps_with(IdentityMerger::new());
        let err = commit(
            &store,
            ReviewedDraft {
                draft_id: 999,
                items: vec![],
                summary: "summary".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap_err();
        assert!(matches!(err, CommitError::DraftNotFound(999)));
    }

    #[test]
    fn under_budget_concatenation_skips_the_merger() {
        let store = RecordingStore::new();
        let first_draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "short".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (first_id, first_ids) = seed_draft(&store, 1, std::slice::from_ref(&first_draft));
        let merger = IdentityMerger::new();
        let deps = deps_with_ref(&merger);
        commit(
            &store,
            ReviewedDraft {
                draft_id: first_id,
                items: vec![accepted_item(
                    first_ids[0],
                    FactCategory::Milestone,
                    "short",
                )],
                summary: "short summary".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let (second_id, second_ids) =
            seed_draft_with_range(&store, 1, 11, 20, std::slice::from_ref(&first_draft));
        commit(
            &store,
            ReviewedDraft {
                draft_id: second_id,
                items: vec![accepted_item(
                    second_ids[0],
                    FactCategory::Milestone,
                    "short",
                )],
                summary: "second summary".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        assert_eq!(merger.calls.load(Ordering::SeqCst), 0);
    }

    fn deps_with_ref(merger: &dyn SummaryMerger) -> CommitDeps<'_> {
        CommitDeps {
            merger: Box::new(RefMerger { inner: merger }),
            observers: Vec::new(),
        }
    }

    struct RefMerger<'a> {
        inner: &'a dyn SummaryMerger,
    }

    impl<'a> SummaryMerger for RefMerger<'a> {
        fn merge(&self, rolling: &str, budget_tokens: usize) -> Result<String, MergeError> {
            self.inner.merge(rolling, budget_tokens)
        }
    }

    #[test]
    fn a_failing_merger_keeps_the_concatenation_and_sets_needs_merge() {
        let store = RecordingStore::new();
        let long_text = "x".repeat(20_000);
        let first_draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "milestone".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (first_id, first_ids) = seed_draft(&store, 1, std::slice::from_ref(&first_draft));
        let deps = deps_with(IdentityMerger::new());
        commit(
            &store,
            ReviewedDraft {
                draft_id: first_id,
                items: vec![accepted_item(
                    first_ids[0],
                    FactCategory::Milestone,
                    "milestone",
                )],
                summary: long_text.clone(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let (second_id, second_ids) =
            seed_draft_with_range(&store, 1, 11, 20, std::slice::from_ref(&first_draft));
        let failing_deps = deps_with(FailingMerger);
        let checkpoint = commit(
            &store,
            ReviewedDraft {
                draft_id: second_id,
                items: vec![accepted_item(
                    second_ids[0],
                    FactCategory::Milestone,
                    "milestone",
                )],
                summary: "second".to_string(),
            },
            &failing_deps,
            &budget(),
        )
        .unwrap();

        assert!(checkpoint.needs_merge);
        assert!(checkpoint.rolling_summary.unwrap().contains(&long_text));
    }

    #[test]
    fn discard_sets_discarded_and_leaves_fact_rows_untouched() {
        let store = RecordingStore::new();
        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "a milestone".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (draft_id, ids) = seed_draft(&store, 1, std::slice::from_ref(&draft));

        discard(&store, draft_id).unwrap();

        let checkpoint = store.get_checkpoint(draft_id).unwrap().unwrap();
        assert_eq!(checkpoint.status, CompactionStatus::Discarded);
        let stored = store.facts_for(draft_id).unwrap();
        assert_eq!(stored[0].id, ids[0]);
        assert!(stored[0].active);
        assert!(store.active_facts(1).unwrap().is_empty());
    }

    #[test]
    fn discarding_a_draft_that_is_not_pending_is_draft_not_pending() {
        let store = RecordingStore::new();
        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "a milestone".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (draft_id, ids) = seed_draft(&store, 1, std::slice::from_ref(&draft));
        let deps = deps_with(IdentityMerger::new());
        commit(
            &store,
            ReviewedDraft {
                draft_id,
                items: vec![accepted_item(
                    ids[0],
                    FactCategory::Milestone,
                    "a milestone",
                )],
                summary: "s".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let err = discard(&store, draft_id).unwrap_err();
        assert!(matches!(
            err,
            CommitError::DraftNotPending {
                status: CompactionStatus::Committed,
                ..
            }
        ));
        assert_eq!(
            store.get_checkpoint(draft_id).unwrap().unwrap().status,
            CompactionStatus::Committed
        );
    }

    /// Wraps a [`RecordingStore`] so `commit_checkpoint` simulates a
    /// concurrent discard landing between `commit`'s initial `Draft` read
    /// and the point where its own transactional gate would otherwise be
    /// the first thing to notice — every other method just delegates.
    struct RacyDiscardStore<'a> {
        inner: &'a RecordingStore,
    }

    impl CompactionStore for RacyDiscardStore<'_> {
        fn insert_draft(&self, draft: NewDraft) -> rusqlite::Result<i64> {
            self.inner.insert_draft(draft)
        }
        fn get_checkpoint(&self, id: i64) -> rusqlite::Result<Option<Checkpoint>> {
            self.inner.get_checkpoint(id)
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
            // The "concurrent" discard: lands after `commit`'s own read of
            // this draft as `Draft` (already past by the time this method
            // runs), before the transactional gate below would otherwise
            // be the first thing to see the mismatch.
            self.inner
                .update_status(record.draft_id, CompactionStatus::Discarded)
                .unwrap();
            self.inner.commit_checkpoint(record)
        }
    }

    #[test]
    fn a_concurrent_discard_between_commits_read_and_write_is_reported_as_draft_not_pending() {
        let inner = RecordingStore::new();
        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "a milestone".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (draft_id, ids) = seed_draft(&inner, 1, std::slice::from_ref(&draft));

        let observer = std::sync::Arc::new(RecordingObserver::default());
        let deps = CommitDeps {
            merger: Box::new(IdentityMerger::new()),
            observers: vec![Box::new(ArcObserver(observer.clone()))],
        };

        let store = RacyDiscardStore { inner: &inner };
        let err = commit(
            &store,
            ReviewedDraft {
                draft_id,
                items: vec![accepted_item(
                    ids[0],
                    FactCategory::Milestone,
                    "a milestone",
                )],
                summary: "s".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            CommitError::DraftNotPending {
                status: CompactionStatus::Discarded,
                ..
            }
        ));
        assert_eq!(
            inner.get_checkpoint(draft_id).unwrap().unwrap().status,
            CompactionStatus::Discarded
        );
        assert!(observer.calls.lock().unwrap().is_empty());
        let stored = inner.facts_for(draft_id).unwrap();
        assert_eq!(stored[0].text, "a milestone");
        assert!(stored[0].active);
    }

    #[test]
    fn discard_calls_no_observer() {
        let store = RecordingStore::new();
        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "a milestone".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (draft_id, _ids) = seed_draft(&store, 1, std::slice::from_ref(&draft));
        discard(&store, draft_id).unwrap();
        // discard takes no CommitDeps at all, so there is no way for it to
        // call an observer; this test documents that shape.
    }

    #[test]
    fn observer_receives_only_newly_active_facts_after_the_commit() {
        let store = RecordingStore::new();
        let accepted = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "accepted".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let rejected = FactDraft {
            rejected_reason: Some("struck at review".to_string()),
            text: "rejected".to_string(),
            ..accepted.clone()
        };
        let (draft_id, ids) = seed_draft(&store, 1, &[accepted.clone(), rejected.clone()]);

        let observer = std::sync::Arc::new(RecordingObserver::default());
        let observer_deps = observer.clone();
        let deps = CommitDeps {
            merger: Box::new(IdentityMerger::new()),
            observers: vec![Box::new(ArcObserver(observer_deps))],
        };

        commit(
            &store,
            ReviewedDraft {
                draft_id,
                items: vec![
                    ReviewedItem {
                        fact_id: ids[0],
                        draft: accepted,
                        accepted: true,
                    },
                    ReviewedItem {
                        fact_id: ids[1],
                        draft: rejected,
                        accepted: false,
                    },
                ],
                summary: "s".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let calls = observer.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        let (checkpoint, facts, _superseded) = &calls[0];
        assert_eq!(checkpoint.status, CompactionStatus::Committed);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].id, ids[0]);
    }

    struct ArcObserver(std::sync::Arc<RecordingObserver>);

    impl CommitObserver for ArcObserver {
        fn on_committed(
            &self,
            checkpoint: &Checkpoint,
            facts: &[Fact],
            superseded: &[i64],
        ) -> Result<(), String> {
            self.0.on_committed(checkpoint, facts, superseded)
        }
    }

    #[test]
    fn an_observer_error_does_not_change_the_ok_result() {
        let store = RecordingStore::new();
        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "a milestone".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (draft_id, ids) = seed_draft(&store, 1, std::slice::from_ref(&draft));

        let deps = CommitDeps {
            merger: Box::new(IdentityMerger::new()),
            observers: vec![Box::new(FailingObserver)],
        };

        let result = commit(
            &store,
            ReviewedDraft {
                draft_id,
                items: vec![accepted_item(
                    ids[0],
                    FactCategory::Milestone,
                    "a milestone",
                )],
                summary: "s".to_string(),
            },
            &deps,
            &budget(),
        );

        assert!(result.is_ok());
    }

    /// Wraps a [`RecordingStore`] so `facts_for` fails once `commit_checkpoint`
    /// has already succeeded — everything else, including `facts_for` called
    /// *before* the commit (`commit`'s own pre-commit read at the top of the
    /// function), delegates straight through. Mirrors `RacyDiscardStore`'s
    /// shape: only the methods needed to simulate the failure are special-
    /// cased, every other method just forwards.
    struct FailingFactsReadbackStore<'a> {
        inner: &'a RecordingStore,
        committed: std::cell::Cell<bool>,
    }

    impl CompactionStore for FailingFactsReadbackStore<'_> {
        fn insert_draft(&self, draft: NewDraft) -> rusqlite::Result<i64> {
            self.inner.insert_draft(draft)
        }
        fn get_checkpoint(&self, id: i64) -> rusqlite::Result<Option<Checkpoint>> {
            self.inner.get_checkpoint(id)
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
            if self.committed.get() {
                return Err(rusqlite::Error::InvalidQuery);
            }
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
            let checkpoint = self.inner.commit_checkpoint(record)?;
            self.committed.set(true);
            Ok(checkpoint)
        }
    }

    #[test]
    fn a_post_commit_facts_readback_failure_does_not_fail_the_already_successful_commit() {
        let inner = RecordingStore::new();
        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "a milestone".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (draft_id, ids) = seed_draft(&inner, 1, std::slice::from_ref(&draft));

        let observer = std::sync::Arc::new(RecordingObserver::default());
        let deps = CommitDeps {
            merger: Box::new(IdentityMerger::new()),
            observers: vec![Box::new(ArcObserver(observer.clone()))],
        };

        let store = FailingFactsReadbackStore {
            inner: &inner,
            committed: std::cell::Cell::new(false),
        };
        let checkpoint = commit(
            &store,
            ReviewedDraft {
                draft_id,
                items: vec![accepted_item(
                    ids[0],
                    FactCategory::Milestone,
                    "a milestone",
                )],
                summary: "s".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        assert_eq!(checkpoint.status, CompactionStatus::Committed);
        assert_eq!(
            inner.get_checkpoint(draft_id).unwrap().unwrap().status,
            CompactionStatus::Committed
        );
        assert_eq!(inner.compacted_through(1).unwrap(), Some(10));

        let calls = observer.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        let (_checkpoint, facts, _superseded) = &calls[0];
        assert!(facts.is_empty());
    }

    #[test]
    fn a_post_commit_facts_readback_failure_still_reports_superseded_ids_to_observers() {
        let inner = RecordingStore::new();
        let old_draft = FactDraft {
            category: FactCategory::CompanionState,
            subject: Some(FactSubject::Companion),
            text: "is nervous".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (old_draft_id, old_ids) = seed_draft(&inner, 1, std::slice::from_ref(&old_draft));
        commit(
            &inner,
            ReviewedDraft {
                draft_id: old_draft_id,
                items: vec![accepted_item(
                    old_ids[0],
                    FactCategory::CompanionState,
                    "is nervous",
                )],
                summary: "s1".to_string(),
            },
            &deps_with(IdentityMerger::new()),
            &budget(),
        )
        .unwrap();

        let new_draft = FactDraft {
            category: FactCategory::CompanionState,
            subject: Some(FactSubject::Companion),
            text: "is confident now".to_string(),
            quote_speaker: None,
            sources: vec![2],
            replaces: vec![old_ids[0]],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (new_draft_id, new_ids) =
            seed_draft_with_range(&inner, 1, 11, 20, std::slice::from_ref(&new_draft));

        let observer = std::sync::Arc::new(RecordingObserver::default());
        let deps = CommitDeps {
            merger: Box::new(IdentityMerger::new()),
            observers: vec![Box::new(ArcObserver(observer.clone()))],
        };
        let store = FailingFactsReadbackStore {
            inner: &inner,
            committed: std::cell::Cell::new(false),
        };
        commit(
            &store,
            ReviewedDraft {
                draft_id: new_draft_id,
                items: vec![ReviewedItem {
                    fact_id: new_ids[0],
                    draft: new_draft,
                    accepted: true,
                }],
                summary: "s2".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        let calls = observer.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        let (_checkpoint, facts, superseded) = &calls[0];
        assert!(facts.is_empty());
        assert_eq!(superseded, &vec![old_ids[0]]);
    }

    /// #181: a checkpoint whose range was marked `Stale` (an edit/delete
    /// landed inside it) gets retired the moment a fresh commit's range
    /// re-covers it — even though nothing here asked specifically for a
    /// re-compaction; any commit spanning a stale range heals it.
    #[test]
    fn commit_over_a_stale_range_retires_it() {
        let store = RecordingStore::new();
        let old_fact = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "old summary of 1-10".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (old_draft_id, old_ids) = seed_draft(&store, 1, std::slice::from_ref(&old_fact));
        let deps = deps_with(IdentityMerger::new());
        commit(
            &store,
            ReviewedDraft {
                draft_id: old_draft_id,
                items: vec![accepted_item(
                    old_ids[0],
                    FactCategory::Milestone,
                    "old summary of 1-10",
                )],
                summary: "s1".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        // An edit inside the committed range marks it Stale, exactly like
        // `database.rs::edit_message_on` does via `mark_stale_containing`.
        assert_eq!(store.mark_stale_containing(1, 5).unwrap(), 1);
        assert_eq!(
            store.get_checkpoint(old_draft_id).unwrap().unwrap().status,
            CompactionStatus::Stale
        );
        // Stale checkpoints still render until they are retired.
        assert_eq!(store.active_facts(1).unwrap().len(), 1);

        let new_fact = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "re-compacted summary of 1-10".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (new_draft_id, new_ids) = seed_draft(&store, 1, std::slice::from_ref(&new_fact));
        commit(
            &store,
            ReviewedDraft {
                draft_id: new_draft_id,
                items: vec![accepted_item(
                    new_ids[0],
                    FactCategory::Milestone,
                    "re-compacted summary of 1-10",
                )],
                summary: "s2".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        assert_eq!(
            store.get_checkpoint(old_draft_id).unwrap().unwrap().status,
            CompactionStatus::Discarded
        );
        let old_stored = store
            .facts_for(old_draft_id)
            .unwrap()
            .into_iter()
            .find(|f| f.id == old_ids[0])
            .unwrap();
        assert!(!old_stored.active);

        let active = store.active_facts(1).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, new_ids[0]);
        assert_eq!(active[0].text, "re-compacted summary of 1-10");
    }

    /// #208 review (esfisher): a failed draft never advances
    /// `compacted_through`, so a later commit whose range overlaps it must
    /// still retire it -- exactly like `commit_over_a_stale_range_retires_it`
    /// above, but for `Failed` instead of `Stale`, and through the real
    /// `commit()` entry point (not `retire_stale_within_on` directly), to
    /// prove `RecordingStore::commit_checkpoint`'s in-memory retirement
    /// mirrors the SQL predicate's `Failed` inclusion, not just the SQL side.
    #[test]
    fn commit_over_a_range_a_failed_draft_covers_retires_it() {
        let store = RecordingStore::new();
        let (failed_id, _) = seed_draft(&store, 1, &[]);
        store.fail_draft(failed_id, "boom").unwrap();
        assert_eq!(
            store.get_checkpoint(failed_id).unwrap().unwrap().status,
            CompactionStatus::Failed
        );

        let deps = deps_with(IdentityMerger::new());
        let new_fact = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "recovered summary of 1-10".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (new_draft_id, new_ids) = seed_draft(&store, 1, std::slice::from_ref(&new_fact));
        commit(
            &store,
            ReviewedDraft {
                draft_id: new_draft_id,
                items: vec![accepted_item(
                    new_ids[0],
                    FactCategory::Milestone,
                    "recovered summary of 1-10",
                )],
                summary: "s".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        assert_eq!(
            store.get_checkpoint(failed_id).unwrap().unwrap().status,
            CompactionStatus::Discarded
        );
    }

    /// #181 review finding: a re-compaction's range can re-cover more than
    /// just the stale checkpoint it was queued for — here it also re-covers
    /// B, an ordinary `Committed` checkpoint that was never marked stale.
    /// Both must retire, and the new checkpoint must not fold either one's
    /// summary forward: `prev_committed` is chosen by range
    /// (`latest_committed_before`), not by "highest id", so a checkpoint the
    /// new draft's own range re-covers is never treated as its predecessor.
    #[test]
    fn recompaction_over_a_stale_and_a_later_committed_checkpoint_retires_both_and_does_not_fold_the_stale_summary_forward(
    ) {
        let store = RecordingStore::new();
        let deps = deps_with(IdentityMerger::new());

        // A = [1,10], committed first.
        let fact_a = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "fact from A".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (a_id, a_ids) = seed_draft_with_range(&store, 1, 1, 10, std::slice::from_ref(&fact_a));
        commit(
            &store,
            ReviewedDraft {
                draft_id: a_id,
                items: vec![accepted_item(
                    a_ids[0],
                    FactCategory::Milestone,
                    "fact from A",
                )],
                summary: "summary A".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        // B = [11,20], committed second; its rolling_summary folds A's
        // summary forward, exactly as #175 always did before re-compaction
        // could start earlier than `compacted_through`.
        let fact_b = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "fact from B".to_string(),
            quote_speaker: None,
            sources: vec![11],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (b_id, b_ids) = seed_draft_with_range(&store, 1, 11, 20, std::slice::from_ref(&fact_b));
        commit(
            &store,
            ReviewedDraft {
                draft_id: b_id,
                items: vec![accepted_item(
                    b_ids[0],
                    FactCategory::Milestone,
                    "fact from B",
                )],
                summary: "summary B".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();
        assert_eq!(
            store
                .get_checkpoint(b_id)
                .unwrap()
                .unwrap()
                .rolling_summary
                .as_deref(),
            Some("summary A")
        );

        // An edit inside A's range marks it Stale.
        assert_eq!(store.mark_stale_containing(1, 5).unwrap(), 1);

        // A re-compaction spanning both A and B: C = [1, 20].
        let fact_c = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "fact from C".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let (c_id, c_ids) = seed_draft_with_range(&store, 1, 1, 20, std::slice::from_ref(&fact_c));
        let checkpoint_c = commit(
            &store,
            ReviewedDraft {
                draft_id: c_id,
                items: vec![accepted_item(
                    c_ids[0],
                    FactCategory::Milestone,
                    "fact from C",
                )],
                summary: "summary C".to_string(),
            },
            &deps,
            &budget(),
        )
        .unwrap();

        // C does not fold either A's or B's summary forward: nothing
        // precedes it — `latest_committed_before(company, from=1)` finds no
        // committed checkpoint ending before message 1.
        assert_eq!(checkpoint_c.rolling_summary.as_deref(), Some(""));

        assert_eq!(
            store.get_checkpoint(a_id).unwrap().unwrap().status,
            CompactionStatus::Discarded
        );
        assert_eq!(
            store.get_checkpoint(b_id).unwrap().unwrap().status,
            CompactionStatus::Discarded
        );

        let active = store.active_facts(1).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, c_ids[0]);
        assert_eq!(active[0].text, "fact from C");
    }
}
