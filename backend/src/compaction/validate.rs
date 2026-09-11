//! The canon validator (#173): [`validate`] runs every extracted
//! [`FactDraft`] through five rejection rules plus a `replaces`-filtering
//! step, then [`overlays_fit`] checks the accepted overlay/rule items
//! against a token budget. Pure, no `Database`, no I/O — the caller (#185's
//! runner) supplies the range, the active facts, and the canon predicate.
#![allow(dead_code)]

use crate::compaction::types::{Fact, FactCategory, FactDraft, FactSubject};
use crate::compaction::CitedMessage;
use crate::context_manager::ContextManager;
use std::fmt;

/// Longest an item's `text` may be, in whitespace-separated words, before
/// [`check_length`] rejects it.
pub const MAX_ITEM_WORDS: usize = 40;

/// Why [`validate`] rejected a draft. Stored (via `Display`) in
/// `compaction_facts.rejected_reason`; a typed enum rather than a bare
/// string so #180's review card and tests can match on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// The draft cited no source messages at all.
    NoSources,
    /// A cited source id is not in the compacted range.
    SourceOutOfRange(i32),
    /// A category that requires a human turn (`user_state`, `backstory`
    /// about the user, `people`) cited only advisory (non-canon) turns.
    NotCanon,
    /// A `rules`/`key_quotes` item's text is not a verbatim substring of
    /// any cited message.
    QuoteNotVerbatim,
    /// A `rules`/`key_quotes` item's quote is verbatim in a cited message,
    /// but not one said by the speaker (`user`/`companion`) the draft
    /// claims via `quote_speaker`.
    SpeakerMismatch,
    /// The item's `text` is longer than [`MAX_ITEM_WORDS`] words.
    TooLong { words: usize },
    /// The item's normalised text matches an active fact in the same
    /// category that it does not replace.
    Duplicate,
    /// The item named no participant of the range: a `state`/`backstory`
    /// item whose text does not begin with a participant's display name, or
    /// a quote/`people` item whose `speaker`/`relation_to` is not one of
    /// them. Set by `extract::to_fact_drafts` before [`validate`] runs,
    /// since only the mapping step knows the range's names; [`validate`]
    /// leaves such a draft alone rather than re-judging it.
    UnknownSubject,
    /// A `people` item naming one of the range's own participants. That
    /// array is for third parties; a principal filed as a person would
    /// become a duplicate identity alongside their own state facts. Like
    /// [`RejectReason::UnknownSubject`], set by `extract::to_fact_drafts`,
    /// which is where the range's names are known.
    PrincipalAsPerson,
    /// The item's text contradicts one of the companion's own curated
    /// running thoughts (#219) covering this checkpoint's range — the model
    /// judge in `contradiction::check` quoted the conflicting words. Never
    /// set by [`validate`] itself: only `extract::fill_draft`'s
    /// contradiction step has the judge's verdict and the range's thoughts.
    ContradictsThought { thought_id: i64 },
}

/// The fixed prefix [`RejectReason::ContradictsThought`]'s `Display` starts
/// with, so a caller holding only the stored string (never the enum) can
/// still recognise the reason by prefix — the same string-compare approach
/// `review.rs::is_set_only_at_extraction` uses for `UnknownSubject`/
/// `PrincipalAsPerson`.
pub const CONTRADICTS_THOUGHT_PREFIX: &str = "contradicts running thought";

impl fmt::Display for RejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RejectReason::NoSources => write!(f, "no sources cited"),
            RejectReason::SourceOutOfRange(id) => {
                write!(f, "source message {id} is outside the compacted range")
            }
            RejectReason::NotCanon => {
                write!(f, "not sourced from a canon (human) turn")
            }
            RejectReason::QuoteNotVerbatim => {
                write!(f, "quote is not verbatim in any cited message")
            }
            RejectReason::SpeakerMismatch => {
                write!(f, "quote is verbatim but attributed to the wrong speaker")
            }
            RejectReason::TooLong { words } => {
                write!(
                    f,
                    "item is {words} words, over the {MAX_ITEM_WORDS}-word limit"
                )
            }
            RejectReason::Duplicate => write!(f, "duplicates an active fact"),
            RejectReason::UnknownSubject => {
                write!(f, "item names no participant of the compacted range")
            }
            RejectReason::PrincipalAsPerson => {
                write!(f, "people item names a participant, not a third party")
            }
            RejectReason::ContradictsThought { thought_id } => {
                write!(f, "{CONTRADICTS_THOUGHT_PREFIX} {thought_id}")
            }
        }
    }
}

/// Collapses runs of whitespace to a single space and trims the ends.
/// Deliberately case-preserving: a quote is verbatim or it is not.
fn normalise_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A duplicate-detection key: whitespace-collapsed, lowercased, with
/// trailing `.`/`!`/`?` stripped.
fn normalise_key(s: &str) -> String {
    normalise_ws(s)
        .to_lowercase()
        .trim_end_matches(['.', '!', '?'])
        .to_string()
}

/// The message in `range` with the given `id`, if any.
fn find_message(range: &[CitedMessage], id: i32) -> Option<&CitedMessage> {
    range.iter().find(|m| m.id == id)
}

/// Whether at least one of `draft`'s sources is present in `range` and
/// satisfies `is_canon`. Ids not present in `range` are treated as
/// non-canon here; [`check_sources`] is what rejects them outright.
fn is_canon_sourced(
    draft: &FactDraft,
    range: &[CitedMessage],
    is_canon: &dyn Fn(&str) -> bool,
) -> bool {
    draft
        .sources
        .iter()
        .filter_map(|id| find_message(range, *id))
        .any(|m| is_canon(&m.speaker_id))
}

/// Rule 1: every source id must be present, and every cited id must exist
/// in `range`.
fn check_sources(draft: &FactDraft, range: &[CitedMessage]) -> Option<RejectReason> {
    if draft.sources.is_empty() {
        return Some(RejectReason::NoSources);
    }
    for &id in &draft.sources {
        if find_message(range, id).is_none() {
            return Some(RejectReason::SourceOutOfRange(id));
        }
    }
    None
}

/// Whether `draft`'s category is one the canon rule applies to: world facts
/// about the user, or a person, must trace back to a human turn.
fn requires_canon(draft: &FactDraft) -> bool {
    match draft.category {
        FactCategory::UserState | FactCategory::Person => true,
        FactCategory::Backstory => matches!(draft.subject, Some(FactSubject::User)),
        _ => false,
    }
}

/// Rule 2: the canon rule. Only applies to [`requires_canon`] categories.
fn check_canon(
    draft: &FactDraft,
    range: &[CitedMessage],
    is_canon: &dyn Fn(&str) -> bool,
) -> Option<RejectReason> {
    if !requires_canon(draft) {
        return None;
    }
    if is_canon_sourced(draft, range, is_canon) {
        None
    } else {
        Some(RejectReason::NotCanon)
    }
}

/// Rule 3: `rules`/`key_quotes` items must be a verbatim (whitespace- and
/// nothing-else-normalised) substring of at least one cited message, said by
/// the speaker the draft claims via `quote_speaker` (`"user"`/`"companion"`,
/// matched against `is_canon` — `"user"` must be canon, `"companion"` must
/// not). A draft with no `quote_speaker` claim (never produced by
/// `extract::to_fact_drafts`, which always sets one for these categories)
/// skips the speaker check rather than treating the absence as a claim.
fn check_verbatim(
    draft: &FactDraft,
    range: &[CitedMessage],
    is_canon: &dyn Fn(&str) -> bool,
) -> Option<RejectReason> {
    if !matches!(draft.category, FactCategory::Rule | FactCategory::KeyQuote) {
        return None;
    }
    let quote = normalise_ws(&draft.text);
    let cited: Vec<&CitedMessage> = draft
        .sources
        .iter()
        .filter_map(|id| find_message(range, *id))
        .collect();
    let text_matches = |m: &&CitedMessage| normalise_ws(&m.content).contains(&quote);

    if !cited.iter().any(text_matches) {
        return Some(RejectReason::QuoteNotVerbatim);
    }

    let expected_canon = match draft.quote_speaker.as_deref() {
        Some("user") => Some(true),
        Some("companion") => Some(false),
        _ => None,
    };
    match expected_canon {
        None => None,
        Some(expected_canon) => {
            let attributed_correctly = cited
                .iter()
                .any(|m| text_matches(m) && is_canon(&m.speaker_id) == expected_canon);
            if attributed_correctly {
                None
            } else {
                Some(RejectReason::SpeakerMismatch)
            }
        }
    }
}

/// Rule 4: `text` may not exceed [`MAX_ITEM_WORDS`] whitespace-separated
/// words.
fn check_length(draft: &FactDraft) -> Option<RejectReason> {
    let words = draft.text.split_whitespace().count();
    if words > MAX_ITEM_WORDS {
        Some(RejectReason::TooLong { words })
    } else {
        None
    }
}

/// Not a rejection rule: retains only `replaces` ids that name an active
/// fact of the same category, dropping (and logging) an unknown or
/// cross-category id. Runs before [`check_duplicate`] so an item that
/// replaces a fact is never counted as a duplicate of the fact it replaces.
fn filter_replaces(draft: &mut FactDraft, active: &[Fact]) {
    let category = draft.category;
    draft.replaces.retain(|id| {
        let keep = active.iter().any(|f| f.id == *id && f.category == category);
        if !keep {
            eprintln!(
                "compaction: draft `replaces` names fact {id}, which is not an active {category} fact; dropping"
            );
        }
        keep
    });
}

/// Rule 5 (checked last, after [`filter_replaces`]): `text`, normalised,
/// must not match an active fact in the same category that this draft does
/// not itself replace, and must not repeat a `(category, key)` already
/// accepted earlier in the same [`validate`] call — one extraction that
/// emits the same fact twice must not insert it twice.
fn check_duplicate(
    draft: &FactDraft,
    active: &[Fact],
    accepted_this_batch: &[(FactCategory, String)],
) -> Option<RejectReason> {
    let category = draft.category;
    let key = normalise_key(&draft.text);
    let duplicate_of_active = active.iter().any(|f| {
        f.category == category && !draft.replaces.contains(&f.id) && normalise_key(&f.text) == key
    });
    let duplicate_in_batch = accepted_this_batch
        .iter()
        .any(|(c, k)| *c == category && *k == key);
    if duplicate_of_active || duplicate_in_batch {
        Some(RejectReason::Duplicate)
    } else {
        None
    }
}

/// Runs every draft through the five rejection rules above, in order,
/// stopping at the first failure and recording its [`RejectReason`] (via
/// `Display`) on `rejected_reason`. Never drops an item: every input draft
/// comes back, rejected or not. Also sets `canon` on every draft (rejected
/// or not) to whether any of its sources is canon-sourced, matching the
/// design doc's `compaction_facts.canon` column ("1 if any source is a user
/// turn").
///
/// `is_canon` is evaluated on each cited message's `speaker_id`; solo/host
/// callers pass `|id| speakers.is_canon(id)`. Takes only the predicate, not
/// the whole `SpeakerInfo`, because that is all this needs.
pub fn validate(
    mut drafts: Vec<FactDraft>,
    range: &[CitedMessage],
    active: &[Fact],
    is_canon: &dyn Fn(&str) -> bool,
) -> Vec<FactDraft> {
    // `(category, normalised text)` of every draft accepted so far in this
    // call, so two drafts in the same extraction that assert the same fact
    // do not both get inserted (`check_duplicate` on its own only compares
    // against already-committed `active` facts).
    let mut accepted_this_batch: Vec<(FactCategory, String)> = Vec::new();

    for draft in drafts.iter_mut() {
        draft.canon = is_canon_sourced(draft, range, is_canon);

        // `extract::to_fact_drafts` rejects an item it could not attribute
        // to a participant. Such a draft has no subject to run the canon
        // rule or the speaker check against, so its reason stands as given.
        if draft.rejected_reason.is_some() {
            continue;
        }

        let rejection = check_sources(draft, range)
            .or_else(|| check_canon(draft, range, is_canon))
            .or_else(|| check_verbatim(draft, range, is_canon))
            .or_else(|| check_length(draft));

        if let Some(reason) = rejection {
            draft.rejected_reason = Some(reason.to_string());
            continue;
        }

        filter_replaces(draft, active);
        match check_duplicate(draft, active, &accepted_this_batch) {
            Some(reason) => draft.rejected_reason = Some(reason.to_string()),
            None => accepted_this_batch.push((draft.category, normalise_key(&draft.text))),
        }
    }
    drafts
}

/// Sums [`ContextManager::estimate_tokens`] over every accepted
/// `companion_state`/`user_state`/`rule` item (the categories rendered into
/// the overlay every turn) and checks the total against `budget_tokens`.
/// `Err(needed)` when they alone would exceed the budget; the runner (#185)
/// discards the whole draft in that case rather than truncating it.
pub fn overlays_fit(drafts: &[FactDraft], budget_tokens: usize) -> Result<(), usize> {
    let needed: usize = drafts
        .iter()
        .filter(|d| d.rejected_reason.is_none())
        .filter(|d| {
            matches!(
                d.category,
                FactCategory::CompanionState | FactCategory::UserState | FactCategory::Rule
            )
        })
        .map(|d| ContextManager::estimate_tokens(&d.text))
        .sum();
    if needed > budget_tokens {
        Err(needed)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::extract::to_fact_drafts;
    use crate::compaction::fixtures::fixture_participants;
    use crate::compaction::fixtures::{bad_draft, synthetic_range};

    fn user_is_canon(speaker_id: &str) -> bool {
        speaker_id == "user"
    }

    fn a_draft(category: FactCategory, text: &str, sources: Vec<i32>) -> FactDraft {
        FactDraft {
            category,
            subject: None,
            text: text.to_string(),
            quote_speaker: None,
            sources,
            replaces: Vec::new(),
            relation_to: None,
            relation: None,
            canon: false,
            rejected_reason: None,
        }
    }

    fn an_active_fact(id: i64, category: FactCategory, text: &str) -> Fact {
        Fact {
            id,
            compaction_id: 1,
            category,
            subject: None,
            text: text.to_string(),
            quote_speaker: None,
            sources: vec![46],
            replaces: Vec::new(),
            relation_to: None,
            relation: None,
            canon: true,
            active: true,
            superseded_by: None,
            rejected_reason: None,
        }
    }

    fn assert_rejected(draft: &FactDraft, expected: RejectReason) {
        assert_eq!(
            draft.rejected_reason.as_deref(),
            Some(expected.to_string().as_str())
        );
    }

    fn assert_accepted(draft: &FactDraft) {
        assert_eq!(draft.rejected_reason, None);
    }

    // --- Rule 1: sources ---

    #[test]
    fn no_sources_is_rejected_and_the_same_item_with_a_source_is_accepted() {
        let range = synthetic_range();
        let rejected = validate(
            vec![a_draft(FactCategory::Milestone, "left home", vec![])],
            &range,
            &[],
            &user_is_canon,
        );
        assert_rejected(&rejected[0], RejectReason::NoSources);

        let accepted = validate(
            vec![a_draft(FactCategory::Milestone, "left home", vec![46])],
            &range,
            &[],
            &user_is_canon,
        );
        assert_accepted(&accepted[0]);
    }

    #[test]
    fn an_out_of_range_source_is_rejected_and_an_in_range_one_is_accepted() {
        let range = synthetic_range();
        let rejected = validate(
            vec![a_draft(FactCategory::Milestone, "left home", vec![999])],
            &range,
            &[],
            &user_is_canon,
        );
        assert_rejected(&rejected[0], RejectReason::SourceOutOfRange(999));

        let accepted = validate(
            vec![a_draft(FactCategory::Milestone, "left home", vec![46])],
            &range,
            &[],
            &user_is_canon,
        );
        assert_accepted(&accepted[0]);
    }

    // --- Rule 2: canon ---

    #[test]
    fn a_user_state_item_sourced_only_from_char_is_rejected_and_with_a_user_source_is_accepted() {
        let range = synthetic_range();
        let rejected = validate(
            vec![a_draft(FactCategory::UserState, "feels at ease", vec![47])],
            &range,
            &[],
            &user_is_canon,
        );
        assert_rejected(&rejected[0], RejectReason::NotCanon);

        let accepted = validate(
            vec![a_draft(FactCategory::UserState, "feels at ease", vec![46])],
            &range,
            &[],
            &user_is_canon,
        );
        assert_accepted(&accepted[0]);
    }

    #[test]
    fn a_person_draft_cited_only_from_char_is_rejected_and_with_one_user_source_is_accepted() {
        let range = synthetic_range();
        let rejected = validate(
            vec![a_draft(FactCategory::Person, "Wren, a neighbor", vec![52])],
            &range,
            &[],
            &user_is_canon,
        );
        assert_rejected(&rejected[0], RejectReason::NotCanon);

        let accepted = validate(
            vec![a_draft(
                FactCategory::Person,
                "Wren, a neighbor",
                vec![51, 52],
            )],
            &range,
            &[],
            &user_is_canon,
        );
        assert_accepted(&accepted[0]);
    }

    #[test]
    fn a_predicate_that_treats_a_remote_bot_as_non_canon_rejects_a_draft_sourced_from_it() {
        // Stands in for #182's registry-backed policy (`ParticipantKind::RemoteBot`
        // is never canon) without depending on the registry itself.
        fn only_user_is_canon(speaker_id: &str) -> bool {
            speaker_id == "user"
        }
        let range = synthetic_range();
        let rejected = validate(
            vec![a_draft(FactCategory::UserState, "checked in", vec![58])],
            &range,
            &[],
            &only_user_is_canon,
        );
        assert_rejected(&rejected[0], RejectReason::NotCanon);
    }

    #[test]
    fn a_backstory_item_about_the_companion_does_not_require_canon() {
        let range = synthetic_range();
        let accepted = validate(
            vec![FactDraft {
                subject: Some(FactSubject::Companion),
                ..a_draft(
                    FactCategory::Backstory,
                    "has always loved lighthouses",
                    vec![47],
                )
            }],
            &range,
            &[],
            &user_is_canon,
        );
        assert_accepted(&accepted[0]);
    }

    // --- Rule 3: verbatim ---

    #[test]
    fn a_misquoted_rule_is_rejected_and_the_verbatim_quote_is_accepted() {
        let range = synthetic_range();
        let rejected = validate(
            vec![a_draft(
                FactCategory::Rule,
                "I swear I will never lie to you, whatever happens",
                vec![53],
            )],
            &range,
            &[],
            &user_is_canon,
        );
        assert_rejected(&rejected[0], RejectReason::QuoteNotVerbatim);

        let accepted = validate(
            vec![a_draft(
                FactCategory::Rule,
                "I promise I will never lie to you, no matter what happens.",
                vec![53],
            )],
            &range,
            &[],
            &user_is_canon,
        );
        assert_accepted(&accepted[0]);
    }

    #[test]
    fn a_verbatim_quote_attributed_to_the_wrong_speaker_is_rejected() {
        let range = synthetic_range();
        // Message 53 ("I promise I will never lie to you...") is a `user`
        // turn, but this draft claims a `companion` said it.
        let mut draft = a_draft(
            FactCategory::Rule,
            "I promise I will never lie to you, no matter what happens.",
            vec![53],
        );
        draft.quote_speaker = Some("companion".to_string());

        let rejected = validate(vec![draft], &range, &[], &user_is_canon);
        assert_rejected(&rejected[0], RejectReason::SpeakerMismatch);
    }

    #[test]
    fn a_verbatim_quote_attributed_to_the_right_speaker_is_accepted() {
        let range = synthetic_range();
        let mut draft = a_draft(
            FactCategory::Rule,
            "I promise I will never lie to you, no matter what happens.",
            vec![53],
        );
        draft.quote_speaker = Some("user".to_string());

        let accepted = validate(vec![draft], &range, &[], &user_is_canon);
        assert_accepted(&accepted[0]);
    }

    #[test]
    fn verbatim_check_ignores_extra_whitespace_but_not_case() {
        let range = synthetic_range();
        let accepted = validate(
            vec![a_draft(
                FactCategory::Rule,
                "I promise   I will never lie to you,\nno matter what happens.",
                vec![53],
            )],
            &range,
            &[],
            &user_is_canon,
        );
        assert_accepted(&accepted[0]);

        let rejected = validate(
            vec![a_draft(
                FactCategory::Rule,
                "I PROMISE I will never lie to you, no matter what happens.",
                vec![53],
            )],
            &range,
            &[],
            &user_is_canon,
        );
        assert_rejected(&rejected[0], RejectReason::QuoteNotVerbatim);
    }

    // --- Rule 4: length ---

    #[test]
    fn an_over_length_item_is_rejected_and_a_short_one_is_accepted() {
        let range = synthetic_range();
        let long_text = "word ".repeat(MAX_ITEM_WORDS + 1);
        let rejected = validate(
            vec![a_draft(FactCategory::Milestone, long_text.trim(), vec![46])],
            &range,
            &[],
            &user_is_canon,
        );
        assert_rejected(
            &rejected[0],
            RejectReason::TooLong {
                words: MAX_ITEM_WORDS + 1,
            },
        );

        let short_text = "word ".repeat(MAX_ITEM_WORDS);
        let accepted = validate(
            vec![a_draft(
                FactCategory::Milestone,
                short_text.trim(),
                vec![46],
            )],
            &range,
            &[],
            &user_is_canon,
        );
        assert_accepted(&accepted[0]);
    }

    // --- filter_replaces ---

    #[test]
    fn filter_replaces_keeps_active_same_category_ids_and_drops_unknown_and_cross_category_ones() {
        let range = synthetic_range();
        let active = vec![
            an_active_fact(1, FactCategory::CompanionState, "was cautious"),
            an_active_fact(2, FactCategory::UserState, "was anxious"),
        ];
        let mut draft = a_draft(FactCategory::CompanionState, "is now relaxed", vec![47]);
        draft.replaces = vec![1, 2, 999];

        let result = validate(vec![draft], &range, &active, &user_is_canon);

        assert_eq!(result[0].replaces, vec![1]);
        assert_accepted(&result[0]);
    }

    #[test]
    fn a_state_item_replacing_fact_f_is_not_flagged_as_fs_duplicate() {
        let range = synthetic_range();
        let active = vec![an_active_fact(
            1,
            FactCategory::CompanionState,
            "is cautious",
        )];
        let mut draft = a_draft(FactCategory::CompanionState, "is cautious", vec![47]);
        draft.replaces = vec![1];

        let result = validate(vec![draft], &range, &active, &user_is_canon);
        assert_accepted(&result[0]);
    }

    // --- Rule 6 (last): duplicate ---

    #[test]
    fn a_duplicate_of_an_active_fact_is_rejected_and_a_distinct_one_is_accepted() {
        let range = synthetic_range();
        let active = vec![an_active_fact(
            1,
            FactCategory::Milestone,
            "Left home for good.",
        )];

        let rejected = validate(
            vec![a_draft(
                FactCategory::Milestone,
                "left home for good",
                vec![46],
            )],
            &range,
            &active,
            &user_is_canon,
        );
        assert_rejected(&rejected[0], RejectReason::Duplicate);

        let accepted = validate(
            vec![a_draft(
                FactCategory::Milestone,
                "moved into the lighthouse",
                vec![46],
            )],
            &range,
            &active,
            &user_is_canon,
        );
        assert_accepted(&accepted[0]);
    }

    #[test]
    fn two_identical_drafts_in_one_validate_call_are_not_both_accepted() {
        let range = synthetic_range();
        let result = validate(
            vec![
                a_draft(FactCategory::Milestone, "left home for good", vec![46]),
                a_draft(FactCategory::Milestone, "left home for good", vec![46]),
            ],
            &range,
            &[],
            &user_is_canon,
        );
        assert_accepted(&result[0]);
        assert_rejected(&result[1], RejectReason::Duplicate);
    }

    // --- canon flag ---

    #[test]
    fn validate_sets_canon_true_when_any_source_is_a_human_turn() {
        let range = synthetic_range();
        let result = validate(
            vec![a_draft(
                FactCategory::CompanionState,
                "is relaxed",
                vec![46, 47],
            )],
            &range,
            &[],
            &user_is_canon,
        );
        assert!(result[0].canon);
    }

    #[test]
    fn validate_sets_canon_false_when_every_source_is_advisory() {
        let range = synthetic_range();
        let result = validate(
            vec![a_draft(
                FactCategory::CompanionState,
                "is relaxed",
                vec![47],
            )],
            &range,
            &[],
            &user_is_canon,
        );
        assert!(!result[0].canon);
    }

    // --- overall shape ---

    #[test]
    fn validate_returns_exactly_as_many_drafts_as_it_received() {
        let range = synthetic_range();
        let drafts = to_fact_drafts(&bad_draft(), &fixture_participants());
        let input_len = drafts.len();
        let result = validate(drafts, &range, &[], &user_is_canon);
        assert_eq!(result.len(), input_len);
    }

    #[test]
    fn bad_draft_fixture_exercises_all_four_deliberate_defects() {
        let range = synthetic_range();
        let drafts = to_fact_drafts(&bad_draft(), &fixture_participants());
        let result = validate(drafts, &range, &[], &user_is_canon);

        let by_category = |category: FactCategory| {
            result
                .iter()
                .find(|d| d.category == category)
                .unwrap_or_else(|| panic!("expected a {category} draft in bad_draft fixture"))
        };

        assert!(matches!(
            by_category(FactCategory::KeyQuote).rejected_reason.as_deref(),
            Some(s) if s.contains("999")
        ));
        assert_rejected(
            by_category(FactCategory::Rule),
            RejectReason::QuoteNotVerbatim,
        );
        assert_rejected(by_category(FactCategory::Person), RejectReason::NotCanon);
        assert!(matches!(
            by_category(FactCategory::Milestone).rejected_reason.as_deref(),
            Some(s) if s.contains("over the")
        ));

        // The rest of the fixture is valid and should be accepted.
        assert_accepted(by_category(FactCategory::CompanionState));
        assert_accepted(by_category(FactCategory::UserState));
        assert_accepted(by_category(FactCategory::Backstory));
        assert_accepted(by_category(FactCategory::OpenThread));
    }

    // --- overlays_fit ---

    #[test]
    fn overlays_fit_ok_under_budget_and_err_with_the_needed_total_over_it() {
        let drafts = vec![
            a_draft(FactCategory::CompanionState, "short", vec![46]),
            a_draft(FactCategory::UserState, "also short", vec![46]),
            // Not counted: wrong category.
            a_draft(
                FactCategory::Milestone,
                "a milestone that is not counted",
                vec![46],
            ),
        ];
        let needed: usize = drafts[..2]
            .iter()
            .map(|d| ContextManager::estimate_tokens(&d.text))
            .sum();

        assert!(overlays_fit(&drafts, needed).is_ok());
        assert_eq!(overlays_fit(&drafts, needed - 1), Err(needed));
    }

    #[test]
    fn overlays_fit_ignores_rejected_drafts() {
        let mut rejected = a_draft(FactCategory::Rule, "a rejected rule", vec![46]);
        rejected.rejected_reason = Some("test".to_string());
        assert_eq!(overlays_fit(&[rejected], 0), Ok(()));
    }

    #[test]
    fn contradicts_thought_display_starts_with_its_fixed_prefix_and_carries_the_thought_id() {
        let reason = RejectReason::ContradictsThought { thought_id: 42 };
        let rendered = reason.to_string();
        assert!(rendered.starts_with(CONTRADICTS_THOUGHT_PREFIX));
        assert!(rendered.contains("42"));
    }
}
