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
use crate::compaction::contradiction::{
    Candidate, CandidateKey, Contradiction, StoredContradiction,
};
use crate::compaction::types::{Checkpoint, Fact, FactCategory, FactDraft};
use crate::compaction::validate::{validate, RejectReason};
use crate::compaction::CitedMessage;

/// The reviewed reason a struck item's [`FactDraft::rejected_reason`] is
/// set to. `commit` (#175) trusts this value as given: it never re-derives
/// "struck" from `accepted` itself, only from `rejected_reason.is_none()`.
const STRUCK_AT_REVIEW: &str = "struck at review";

/// Whether `reason` is one only `extract::to_fact_drafts` can set, because
/// deriving it needs the range's participant table and a review request
/// carries no such thing.
///
/// [`validate`] leaves a draft that already carries a reason alone, so
/// clearing one of these on an `accepted: true` review would file the item
/// with the category and subject the extractor could not determine —
/// `promote_fact_on` writes neither, so an unattributable `state` item would
/// go active with a `NULL` subject. Keeping the reason makes the accept come
/// back as [`ReviewError::Rejected`] naming the original reason instead.
/// Every other [`RejectReason`] is re-derivable by `validate`, and review
/// clears those so an edited item gets a fresh verdict.
fn is_set_only_at_extraction(reason: Option<&str>) -> bool {
    reason.is_some_and(|reason| {
        reason == RejectReason::UnknownSubject.to_string()
            || reason == RejectReason::PrincipalAsPerson.to_string()
    })
}

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

/// One item [`apply_review`] (or #219's commit-time re-check) rejected.
/// `item_id` (not `id`) matches the field name the frontend's
/// `useCompaction().commit` reads off the `422` body. `None` names the
/// checkpoint's summary rather than a fact — only the re-check can produce
/// that; `apply_review` itself never rejects the summary, so every
/// `RejectedItem` it builds carries `Some`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RejectedItem {
    pub item_id: Option<i64>,
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

/// `pub(crate)`: also reused by `multiplayer::joiner_compaction` (#186) to
/// turn a joiner's own stored facts into the `FactDraft`s its own-facts-only
/// `ReviewedDraft` carries, the same conversion `apply_review` uses for the
/// host's review request.
pub(crate) fn fact_to_draft(fact: &Fact) -> FactDraft {
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
    facts: &[Fact],
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
    for fact in facts {
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
                if !is_set_only_at_extraction(fact_draft.rejected_reason.as_deref()) {
                    fact_draft.rejected_reason = None;
                }
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
                item_id: Some(p.fact_id),
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

/// Which parts of a review differ from what extraction stored (#225): the
/// accepted fact ids whose reviewed `text` is not the stored `text`, and
/// whether the reviewed summary is not the stored summary. Built by
/// [`review_edits`]; a struck item can never appear in `fact_ids` since
/// [`apply_review`] ignores its `text`/`quote` and keeps the stored text.
#[derive(Debug, Clone, Default)]
pub struct ReviewEdits {
    pub summary: bool,
    pub fact_ids: HashSet<i64>,
}

/// Compares `reviewed` against what extraction stored (`draft`/`stored`) to
/// find the edits [`recheck_candidates`] must re-judge even when they were
/// never flagged (#225). Exact string comparison — a whitespace-only edit
/// costs one extra judge call, which is simpler and safer than normalising.
pub fn review_edits(draft: &Checkpoint, stored: &[Fact], reviewed: &ReviewedDraft) -> ReviewEdits {
    let stored_text: HashMap<i64, &str> = stored.iter().map(|f| (f.id, f.text.as_str())).collect();
    let fact_ids = reviewed
        .items
        .iter()
        .filter(|item| item.accepted)
        .filter(|item| stored_text.get(&item.fact_id) != Some(&item.draft.text.as_str()))
        .map(|item| item.fact_id)
        .collect();

    let stored_summary = draft.summary.clone().unwrap_or_default();
    ReviewEdits {
        summary: reviewed.summary != stored_summary,
        fact_ids,
    }
}

/// The candidates a commit must re-judge against the current covering
/// thoughts (#219, widened by #225): the reviewed summary, if `flagged`
/// names it (`fact_id: None`) or `edits.summary` is set, plus every
/// accepted item whose fact id `flagged` names or `edits.fact_ids` names. A
/// flagged item the user struck at review is not re-judged — it is already
/// inactive and cannot be committed either way. An item that is both
/// flagged and edited is pushed exactly once. Empty when nothing was
/// flagged and nothing was edited (the common case), so the caller makes no
/// model call.
///
/// Each returned `Candidate`'s text is the *reviewed* text — an edited
/// item's or summary's fix is exactly what gets judged, the same "the user
/// fixed the text" remedy the design doc describes — and its `key` is
/// unique only within this call's own slice (used by [`rejections_from`] to
/// map a verdict back); it carries no meaning to the caller beyond that.
pub fn recheck_candidates<'a>(
    reviewed: &'a ReviewedDraft,
    flagged: &[StoredContradiction],
    edits: &ReviewEdits,
) -> Vec<(Option<i64>, Candidate<'a>)> {
    let mut candidates = Vec::new();

    if edits.summary || flagged.iter().any(|row| row.fact_id.is_none()) {
        candidates.push((
            None,
            Candidate {
                key: CandidateKey::Summary,
                text: &reviewed.summary,
            },
        ));
    }

    let flagged_fact_ids: HashSet<i64> = flagged.iter().filter_map(|row| row.fact_id).collect();
    for item in &reviewed.items {
        if item.accepted
            && (flagged_fact_ids.contains(&item.fact_id) || edits.fact_ids.contains(&item.fact_id))
        {
            candidates.push((
                Some(item.fact_id),
                Candidate {
                    key: CandidateKey::Fact(candidates.len()),
                    text: &item.draft.text,
                },
            ));
        }
    }

    candidates
}

/// Maps [`recheck_candidates`]'s fresh verdicts back to [`RejectedItem`]s:
/// `item_id: None` for a contradicted summary, `Some(fact_id)` for a
/// contradicted fact. `keys` must be the exact slice `recheck_candidates`
/// returned (or built the same way) — a `Contradiction::candidate` this
/// module didn't hand out has nothing to map back to and is silently
/// dropped, which cannot happen when `found` comes from
/// `contradiction::check` run over `keys`' own candidates.
pub fn rejections_from(
    found: &[Contradiction],
    keys: &[(Option<i64>, Candidate<'_>)],
) -> Vec<RejectedItem> {
    found
        .iter()
        .filter_map(|hit| {
            keys.iter()
                .find(|(_, candidate)| candidate.key == hit.candidate)
                .map(|(item_id, _)| RejectedItem {
                    item_id: *item_id,
                    reason: RejectReason::ContradictsThought {
                        thought_id: hit.thought_id,
                    }
                    .to_string(),
                })
        })
        .collect()
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
            extraction_error: None,
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
    fn accepting_an_item_the_extractor_could_not_attribute_does_not_un_reject_it() {
        for reason in [
            RejectReason::UnknownSubject,
            RejectReason::PrincipalAsPerson,
        ] {
            let draft = a_draft_checkpoint(1);
            let mut fact = a_fact(10, FactCategory::CompanionState, "feels at home", vec![1]);
            fact.subject = None;
            fact.rejected_reason = Some(reason.to_string());
            let range = vec![cited(1, "user", "this place feels like home now")];
            let request = CommitRequest {
                items: vec![ItemReview {
                    id: 10,
                    accepted: true,
                    // An edit that repairs the *text* still cannot tell
                    // `validate` who the subject is: only the extractor had
                    // the participant table.
                    text: Some("Eric feels at home".to_string()),
                    quote: None,
                }],
                summary: None,
            };

            let result = apply_review(&draft, &[fact], request, &range, &[], &user_is_canon);

            match result {
                Err(ReviewError::Rejected(rejected)) => {
                    assert!(
                        rejected
                            .iter()
                            .any(|r| r.item_id == Some(10) && r.reason == reason.to_string()),
                        "`{reason}` should come back on fact 10 with its original reason, \
                         got {rejected:?}"
                    );
                }
                other => panic!("accepting a `{reason}` item should be rejected, got {other:?}"),
            }
        }
    }

    #[test]
    fn accepting_an_item_rejected_for_a_re_derivable_reason_still_clears_it() {
        let draft = a_draft_checkpoint(1);
        let mut fact = a_fact(10, FactCategory::KeyQuote, "not in any message", vec![1]);
        fact.quote_speaker = Some("user".to_string());
        fact.rejected_reason = Some(RejectReason::QuoteNotVerbatim.to_string());
        let range = vec![cited(1, "user", "we moved in together")];
        let request = CommitRequest {
            items: vec![ItemReview {
                id: 10,
                accepted: true,
                text: None,
                quote: Some("we moved in together".to_string()),
            }],
            summary: None,
        };

        let reviewed = apply_review(&draft, &[fact], request, &range, &[], &user_is_canon)
            .expect("an edited quote that is now verbatim should pass");

        assert!(reviewed.items[0].draft.rejected_reason.is_none());
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

        let reviewed = apply_review(&draft, &facts, request, &range, &[], &user_is_canon).unwrap();

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

        let reviewed = apply_review(&draft, &facts, request, &range, &[], &user_is_canon).unwrap();

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

        let err = apply_review(&draft, &facts, request, &range, &[], &user_is_canon).unwrap_err();

        match err {
            ReviewError::Rejected(items) => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0].item_id, Some(10));
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

        let err = apply_review(&draft, &facts, request, &range, &[], &user_is_canon).unwrap_err();

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

        let reviewed = apply_review(&draft, &facts, request, &range, &[], &user_is_canon).unwrap();

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

        let reviewed = apply_review(&draft, &facts, request, &[], &[], &user_is_canon).unwrap();

        assert_eq!(reviewed.summary, "stored summary");
    }

    // --- recheck_candidates / rejections_from (#219) ---

    fn a_reviewed_item(fact_id: i64, text: &str, accepted: bool) -> ReviewedItem {
        ReviewedItem {
            fact_id,
            accepted,
            draft: FactDraft {
                category: FactCategory::Milestone,
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
        }
    }

    fn a_stored_contradiction(fact_id: Option<i64>) -> StoredContradiction {
        StoredContradiction {
            fact_id,
            thought_id: 7,
            thought_text: "the companion's own note".to_string(),
            quote: "the conflicting words".to_string(),
        }
    }

    fn no_edits() -> ReviewEdits {
        ReviewEdits::default()
    }

    #[test]
    fn recheck_candidates_is_empty_when_nothing_was_flagged_and_nothing_was_edited() {
        let reviewed = ReviewedDraft {
            draft_id: 1,
            items: vec![a_reviewed_item(10, "a fact", true)],
            summary: "a summary".to_string(),
        };
        assert!(recheck_candidates(&reviewed, &[], &no_edits()).is_empty());
    }

    #[test]
    fn recheck_candidates_skips_a_flagged_item_struck_at_review() {
        let reviewed = ReviewedDraft {
            draft_id: 1,
            items: vec![a_reviewed_item(10, "a fact", false)],
            summary: "a summary".to_string(),
        };
        let flagged = vec![a_stored_contradiction(Some(10))];

        assert!(recheck_candidates(&reviewed, &flagged, &no_edits()).is_empty());
    }

    #[test]
    fn recheck_candidates_uses_the_reviewed_text_for_a_flagged_accepted_item() {
        let reviewed = ReviewedDraft {
            draft_id: 1,
            items: vec![a_reviewed_item(10, "the edited text", true)],
            summary: "a summary".to_string(),
        };
        let flagged = vec![a_stored_contradiction(Some(10))];

        let candidates = recheck_candidates(&reviewed, &flagged, &no_edits());

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, Some(10));
        assert_eq!(candidates[0].1.text, "the edited text");
    }

    #[test]
    fn recheck_candidates_includes_the_summary_when_it_was_flagged() {
        let reviewed = ReviewedDraft {
            draft_id: 1,
            items: vec![],
            summary: "the reviewed summary".to_string(),
        };
        let flagged = vec![a_stored_contradiction(None)];

        let candidates = recheck_candidates(&reviewed, &flagged, &no_edits());

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, None);
        assert_eq!(candidates[0].1.text, "the reviewed summary");
    }

    #[test]
    fn recheck_candidates_includes_an_edited_item_that_was_never_flagged() {
        let reviewed = ReviewedDraft {
            draft_id: 1,
            items: vec![a_reviewed_item(10, "the edited text", true)],
            summary: "a summary".to_string(),
        };
        let edits = ReviewEdits {
            summary: false,
            fact_ids: HashSet::from([10]),
        };

        let candidates = recheck_candidates(&reviewed, &[], &edits);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, Some(10));
        assert_eq!(candidates[0].1.text, "the edited text");
    }

    #[test]
    fn recheck_candidates_pushes_a_flagged_and_edited_item_once() {
        let reviewed = ReviewedDraft {
            draft_id: 1,
            items: vec![a_reviewed_item(10, "the edited text", true)],
            summary: "a summary".to_string(),
        };
        let flagged = vec![a_stored_contradiction(Some(10))];
        let edits = ReviewEdits {
            summary: false,
            fact_ids: HashSet::from([10]),
        };

        let candidates = recheck_candidates(&reviewed, &flagged, &edits);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, Some(10));
    }

    #[test]
    fn recheck_candidates_includes_an_edited_summary_that_was_never_flagged() {
        let reviewed = ReviewedDraft {
            draft_id: 1,
            items: vec![],
            summary: "the reviewed summary".to_string(),
        };
        let edits = ReviewEdits {
            summary: true,
            fact_ids: HashSet::new(),
        };

        let candidates = recheck_candidates(&reviewed, &[], &edits);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, None);
        assert_eq!(candidates[0].1.text, "the reviewed summary");
    }

    #[test]
    fn rejections_from_maps_a_fresh_verdict_back_to_its_fact_id() {
        let reviewed = ReviewedDraft {
            draft_id: 1,
            items: vec![a_reviewed_item(10, "the edited text", true)],
            summary: "a summary".to_string(),
        };
        let flagged = vec![a_stored_contradiction(Some(10))];
        let keys = recheck_candidates(&reviewed, &flagged, &no_edits());
        let found = vec![Contradiction {
            candidate: keys[0].1.key,
            thought_id: 42,
            thought_text: "the companion's own note".to_string(),
            quote: "the conflicting words".to_string(),
        }];

        let rejections = rejections_from(&found, &keys);

        assert_eq!(rejections.len(), 1);
        assert_eq!(rejections[0].item_id, Some(10));
        assert!(rejections[0].reason.contains("42"));
    }

    #[test]
    fn rejections_from_maps_a_summary_verdict_to_item_id_none() {
        let reviewed = ReviewedDraft {
            draft_id: 1,
            items: vec![],
            summary: "the reviewed summary".to_string(),
        };
        let flagged = vec![a_stored_contradiction(None)];
        let keys = recheck_candidates(&reviewed, &flagged, &no_edits());
        let found = vec![Contradiction {
            candidate: keys[0].1.key,
            thought_id: 3,
            thought_text: "the companion's own note".to_string(),
            quote: "the conflicting words".to_string(),
        }];

        let rejections = rejections_from(&found, &keys);

        assert_eq!(rejections.len(), 1);
        assert_eq!(rejections[0].item_id, None);
    }

    // --- review_edits (#225) ---

    #[test]
    fn review_edits_is_empty_for_an_untouched_review() {
        let draft = a_draft_checkpoint(1);
        let stored = vec![a_fact(10, FactCategory::Milestone, "a fact", vec![1])];
        let reviewed = ReviewedDraft {
            draft_id: 1,
            items: vec![a_reviewed_item(10, "a fact", true)],
            summary: "stored summary".to_string(),
        };

        let edits = review_edits(&draft, &stored, &reviewed);

        assert!(!edits.summary);
        assert!(edits.fact_ids.is_empty());
    }

    #[test]
    fn review_edits_names_an_accepted_item_whose_text_changed() {
        let draft = a_draft_checkpoint(1);
        let stored = vec![a_fact(10, FactCategory::Milestone, "a fact", vec![1])];
        let reviewed = ReviewedDraft {
            draft_id: 1,
            items: vec![a_reviewed_item(10, "a changed fact", true)],
            summary: "stored summary".to_string(),
        };

        let edits = review_edits(&draft, &stored, &reviewed);

        assert_eq!(edits.fact_ids, HashSet::from([10]));
    }

    #[test]
    fn review_edits_ignores_a_struck_item() {
        let draft = a_draft_checkpoint(1);
        let stored = vec![a_fact(10, FactCategory::Milestone, "a fact", vec![1])];
        let range = vec![cited(1, "user", "a fact")];
        let request = CommitRequest {
            items: vec![ItemReview {
                id: 10,
                accepted: false,
                // Ignored by `apply_review` for a struck item, so this must
                // not register as an edit either.
                text: Some("a changed fact".to_string()),
                quote: None,
            }],
            summary: None,
        };
        let reviewed = apply_review(&draft, &stored, request, &range, &[], &user_is_canon)
            .expect("striking an item never fails validation");

        let edits = review_edits(&draft, &stored, &reviewed);

        assert!(edits.fact_ids.is_empty());
    }

    #[test]
    fn review_edits_treats_an_omitted_or_identical_summary_as_unedited() {
        let draft = a_draft_checkpoint(1);
        for summary in [None, Some("stored summary".to_string())] {
            let reviewed = ReviewedDraft {
                draft_id: 1,
                items: vec![],
                summary: summary.unwrap_or_else(|| draft.summary.clone().unwrap_or_default()),
            };

            let edits = review_edits(&draft, &[], &reviewed);

            assert!(!edits.summary);
        }
    }

    #[test]
    fn review_edits_treats_a_changed_summary_as_edited() {
        let draft = a_draft_checkpoint(1);
        let reviewed = ReviewedDraft {
            draft_id: 1,
            items: vec![],
            summary: "a new summary".to_string(),
        };

        let edits = review_edits(&draft, &[], &reviewed);

        assert!(edits.summary);
    }
}
