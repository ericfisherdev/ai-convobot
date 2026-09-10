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
    /// to the draft.
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
/// runs the model.
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
    let prev_committed = store.latest_committed(draft.companion_id)?;

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

    let checkpoint = store.commit_checkpoint(record)?;

    Database::clear_message_cache();

    let new_active_facts: Vec<Fact> = store
        .facts_for(draft_id)?
        .into_iter()
        .filter(|f| f.active)
        .collect();

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
    store.update_status(draft_id, CompactionStatus::Discarded)?;
    Ok(())
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
        let draft_id = store
            .insert_draft(NewDraft {
                companion_id,
                from_message_id: 1,
                through_message_id: 10,
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
        let (second_id, second_ids) = seed_draft(&store, 1, std::slice::from_ref(&second_draft));
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
        let (new_draft_id, new_ids) = seed_draft(&store, 1, std::slice::from_ref(&new_draft));
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
        let (second_draft_id, second_ids) = seed_draft(&store, 1, std::slice::from_ref(&duplicate));
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

        let (second_id, second_ids) = seed_draft(&store, 1, std::slice::from_ref(&first_draft));
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

        let (second_id, second_ids) = seed_draft(&store, 1, std::slice::from_ref(&first_draft));
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
}
