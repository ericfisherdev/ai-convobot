//! The request side of commit (#179): turns a user's edits to a draft's
//! stored facts into the [`crate::compaction::commit::ReviewedDraft`]
//! `commit` (#175) takes, re-running #173's [`validate`] over every item
//! the review touches so an edit that breaks a rejection rule cannot sneak
//! past the validator that would have caught it at extraction time.
//!
//! [`CommitRequest`]/[`ItemReview`] are named apart from
//! [`crate::compaction::commit::ReviewedDraft`]/
//! [`crate::compaction::commit::ReviewedItem`] on purpose: those are #175's
//! *output* shape (one entry per stored fact, `fact_id` always populated);
//! these are the *wire* shape `POST /api/compaction/{id}/commit` accepts
//! (one entry per fact the user actually touched, `id` is the fact being
//! edited).

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::compaction::commit::{ReviewedDraft, ReviewedItem};
use crate::compaction::types::{Checkpoint, Fact, FactCategory, FactDraft};
use crate::compaction::validate::validate;
use crate::compaction::CitedMessage;

/// The reviewed reason a struck item's [`FactDraft::rejected_reason`] is
/// set to. `commit` (#175) trusts this value as given: it never re-derives
/// "struck" from `accepted` itself, only from `rejected_reason.is_none()`.
const STRUCK_AT_REVIEW: &str = "struck at review";

/// One item's review: `id` is the stored `Fact.id` (`CheckpointDetail.facts`
/// carries it, so the frontend never invents one). `text` edits a
/// non-quote item's `text`; `quote` edits a [`FactCategory::Rule`]/
/// [`FactCategory::KeyQuote`] item's verbatim quote (also `FactDraft.text`
/// under the hood, but kept as a separate wire field so the review UI can
/// label the two differently). Both are ignored when `accepted: false` — a
/// struck item is stored exactly as extracted.
#[derive(Debug, Clone, Deserialize)]
pub struct ItemReview {
    pub id: i64,
    pub accepted: bool,
    pub text: Option<String>,
    pub quote: Option<String>,
}

/// `POST /api/compaction/{id}/commit`'s request body. `summary` is
/// `Option` because most reviews leave the extracted summary as-is; an
/// omitted summary falls back to the draft's stored summary.
#[derive(Debug, Clone, Deserialize)]
pub struct CommitRequest {
    pub items: Vec<ItemReview>,
    pub summary: Option<String>,
}

/// One item [`apply_review`] rejected while re-validating an `accepted:
/// true` edit. `item_id` (not `id`) matches the field name the frontend's
/// `useCompaction().commit` reads off the `422` body.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RejectedItem {
    pub item_id: i64,
    pub reason: String,
}

/// Why [`apply_review`] could not build a [`ReviewedDraft`].
#[derive(Debug, Clone, PartialEq)]
pub enum ReviewError {
    /// `request` named a fact id that is not one of `draft`'s stored facts.
    UnknownItem(i64),
    /// One or more `accepted: true` items still failed validation after the
    /// requested edit; the draft is left untouched, nothing is committed.
    Rejected(Vec<RejectedItem>),
}

fn is_quote_category(category: FactCategory) -> bool {
    matches!(category, FactCategory::Rule | FactCategory::KeyQuote)
}

fn fact_to_draft(fact: &Fact) -> FactDraft {
    FactDraft {
        category: fact.category,
        subject: fact.subject.clone(),
        text: fact.text.clone(),
        quote_speaker: fact.quote_speaker.clone(),
        sources: fact.sources.clone(),
        replaces: fact.replaces.clone(),
        relation_to: fact.relation_to.clone(),
        relation: fact.relation.clone(),
        canon: fact.canon,
        rejected_reason: fact.rejected_reason.clone(),
    }
}

/// One stored fact carried through review: `revalidate` is set only for an
/// `accepted: true` review (an edit, or a plain "keep this"), the one case
/// [`validate`] is asked to re-judge; every other fact (untouched, or
/// struck) keeps exactly the verdict it already had.
struct Pending {
    fact_id: i64,
    draft: FactDraft,
    revalidate: bool,
}

/// Applies `request` onto `draft`'s stored `facts`, producing the
/// [`ReviewedDraft`] `commit` (#175) takes.
///
/// Every stored fact gets exactly one output item, in stored order,
/// regardless of whether the request touched it: an item with no matching
/// `request.items` entry keeps its stored `text`/`rejected_reason` exactly
/// as [`crate::compaction::store::CompactionStore::facts_for`] returned it.
/// An `accepted: false` item is stored inactive with
/// [`FactDraft::rejected_reason`] set to `"struck at review"`, regardless of
/// its stored verdict or any `text`/`quote` also sent for it. An `accepted:
/// true` item has its edit (if any) applied, `rejected_reason` cleared, and
/// is re-run through [`validate`] alongside every other `accepted: true`
/// item in this request — together, so [`validate`]'s within-batch
/// duplicate check sees the whole reconsidered set at once, the same way a
/// fresh extraction pass would. If any of those still carries a
/// `rejected_reason` afterward, the whole review fails with
/// [`ReviewError::Rejected`] and nothing in `draft` is changed.
///
/// `request.items` naming a fact id absent from `facts` is
/// [`ReviewError::UnknownItem`], checked before any other work.
pub fn apply_review(
    draft: &Checkpoint,
    facts: Vec<Fact>,
    request: CommitRequest,
    range: &[CitedMessage],
    active: &[Fact],
    is_canon: &dyn Fn(&str) -> bool,
) -> Result<ReviewedDraft, ReviewError> {
    let known_ids: HashSet<i64> = facts.iter().map(|f| f.id).collect();
    let mut reviews: HashMap<i64, ItemReview> = HashMap::with_capacity(request.items.len());
    for item in request.items {
        if !known_ids.contains(&item.id) {
            return Err(ReviewError::UnknownItem(item.id));
        }
        reviews.insert(item.id, item);
    }

    let mut pending: Vec<Pending> = Vec::with_capacity(facts.len());
    for fact in &facts {
        let mut fact_draft = fact_to_draft(fact);
        let revalidate = match reviews.remove(&fact.id) {
            None => false,
            Some(review) if !review.accepted => {
                fact_draft.rejected_reason = Some(STRUCK_AT_REVIEW.to_string());
                false
            }
            Some(review) => {
                if is_quote_category(fact.category) {
                    if let Some(quote) = review.quote {
                        fact_draft.text = quote;
                    }
                } else if let Some(text) = review.text {
                    fact_draft.text = text;
                }
                fact_draft.rejected_reason = None;
                true
            }
        };
        pending.push(Pending {
            fact_id: fact.id,
            draft: fact_draft,
            revalidate,
        });
    }

    let to_revalidate: Vec<FactDraft> = pending
        .iter()
        .filter(|p| p.revalidate)
        .map(|p| p.draft.clone())
        .collect();
    let mut revalidated = validate(to_revalidate, range, active, is_canon).into_iter();
    for p in pending.iter_mut() {
        if p.revalidate {
            p.draft = revalidated
                .next()
                .expect("one validated draft per revalidated item");
        }
    }

    let rejected: Vec<RejectedItem> = pending
        .iter()
        .filter(|p| p.revalidate)
        .filter_map(|p| {
            p.draft.rejected_reason.as_ref().map(|reason| RejectedItem {
                item_id: p.fact_id,
                reason: reason.clone(),
            })
        })
        .collect();
    if !rejected.is_empty() {
        return Err(ReviewError::Rejected(rejected));
    }

    let items = pending
        .into_iter()
        .map(|p| ReviewedItem {
            fact_id: p.fact_id,
            accepted: p.draft.rejected_reason.is_none(),
            draft: p.draft,
        })
        .collect();

    Ok(ReviewedDraft {
        draft_id: draft.id,
        items,
        summary: request
            .summary
            .unwrap_or_else(|| draft.summary.clone().unwrap_or_default()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::types::{CompactionStatus, CompactionTrigger, FactSubject};

    fn a_draft_checkpoint(id: i64) -> Checkpoint {
        Checkpoint {
            id,
            companion_id: 1,
            from_message_id: 1,
            through_message_id: 10,
            status: CompactionStatus::Draft,
            trigger: CompactionTrigger::Threshold,
            raw_model_output: Some("raw".to_string()),
            summary: Some("stored summary".to_string()),
            rolling_summary: None,
            attitude_ratings: None,
            needs_merge: false,
            created_at: "now".to_string(),
            committed_at: None,
        }
    }

    fn a_fact(id: i64, category: FactCategory, text: &str, sources: Vec<i32>) -> Fact {
        Fact {
            id,
            compaction_id: 1,
            category,
            subject: Some(FactSubject::User),
            text: text.to_string(),
            quote_speaker: None,
            sources,
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            active: true,
            superseded_by: None,
            rejected_reason: None,
        }
    }

    fn cited(id: i32, speaker_id: &str, content: &str) -> CitedMessage {
        CitedMessage {
            id,
            speaker_id: speaker_id.to_string(),
            content: content.to_string(),
        }
    }

    fn user_is_canon(speaker_id: &str) -> bool {
        speaker_id == "user"
    }

    fn accepted_review(id: i64) -> ItemReview {
        ItemReview {
            id,
            accepted: true,
            text: None,
            quote: None,
        }
    }

    #[test]
    fn an_untouched_accepted_item_passes_through_unchanged() {
        let draft = a_draft_checkpoint(1);
        let facts = vec![a_fact(
            10,
            FactCategory::Milestone,
            "moved in together",
            vec![1],
        )];
        let range = vec![cited(1, "user", "we moved in together")];
        let request = CommitRequest {
            items: vec![],
            summary: None,
        };

        let reviewed = apply_review(&draft, facts, request, &range, &[], &user_is_canon).unwrap();

        assert_eq!(reviewed.items.len(), 1);
        assert_eq!(reviewed.items[0].fact_id, 10);
        assert!(reviewed.items[0].accepted);
        assert!(reviewed.items[0].draft.rejected_reason.is_none());
    }

    #[test]
    fn every_emitted_item_matches_the_stored_fact_at_the_same_index() {
        let draft = a_draft_checkpoint(1);
        let facts = vec![
            a_fact(10, FactCategory::Milestone, "a", vec![1]),
            a_fact(11, FactCategory::OpenThread, "b", vec![1]),
            a_fact(12, FactCategory::Backstory, "c", vec![1]),
        ];
        let range = vec![cited(1, "companion", "irrelevant")];
        let request = CommitRequest {
            items: vec![],
            summary: None,
        };

        let reviewed = apply_review(&draft, facts, request, &range, &[], &user_is_canon).unwrap();

        assert_eq!(
            reviewed.items.iter().map(|i| i.fact_id).collect::<Vec<_>>(),
            vec![10, 11, 12]
        );
    }

    #[test]
    fn an_edited_rule_quote_that_no_longer_matches_its_source_is_rejected() {
        let draft = a_draft_checkpoint(1);
        let facts = vec![a_fact(
            10,
            FactCategory::Rule,
            "never go to the lake",
            vec![1],
        )];
        let range = vec![cited(1, "user", "never go to the lake alone")];
        let request = CommitRequest {
            items: vec![ItemReview {
                id: 10,
                accepted: true,
                text: None,
                quote: Some("never set foot near the lake".to_string()),
            }],
            summary: None,
        };

        let err = apply_review(&draft, facts, request, &range, &[], &user_is_canon).unwrap_err();

        match err {
            ReviewError::Rejected(items) => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0].item_id, 10);
                assert!(items[0].reason.contains("verbatim"));
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[test]
    fn an_id_not_on_the_draft_is_unknown_item() {
        let draft = a_draft_checkpoint(1);
        let facts = vec![a_fact(10, FactCategory::Milestone, "a", vec![1])];
        let range = vec![cited(1, "user", "a")];
        let request = CommitRequest {
            items: vec![accepted_review(999)],
            summary: None,
        };

        let err = apply_review(&draft, facts, request, &range, &[], &user_is_canon).unwrap_err();

        assert_eq!(err, ReviewError::UnknownItem(999));
    }

    #[test]
    fn a_struck_item_survives_as_inactive_with_the_struck_reason() {
        let draft = a_draft_checkpoint(1);
        let facts = vec![a_fact(10, FactCategory::Milestone, "a", vec![1])];
        let range = vec![cited(1, "user", "a")];
        let request = CommitRequest {
            items: vec![ItemReview {
                id: 10,
                accepted: false,
                text: Some("edited text should be ignored".to_string()),
                quote: None,
            }],
            summary: None,
        };

        let reviewed = apply_review(&draft, facts, request, &range, &[], &user_is_canon).unwrap();

        assert!(!reviewed.items[0].accepted);
        assert_eq!(
            reviewed.items[0].draft.rejected_reason.as_deref(),
            Some("struck at review")
        );
        assert_eq!(reviewed.items[0].draft.text, "a");
    }

    #[test]
    fn an_omitted_summary_falls_back_to_the_drafts_stored_summary() {
        let draft = a_draft_checkpoint(1);
        let facts = vec![];
        let request = CommitRequest {
            items: vec![],
            summary: None,
        };

        let reviewed = apply_review(&draft, facts, request, &[], &[], &user_is_canon).unwrap();

        assert_eq!(reviewed.summary, "stored summary");
    }
}
