//! A joiner's own auto-extraction (#186): once the host's `ContinuityPayload`
//! shows it has compacted further than this joiner has locally caught up
//! to, this module queues and runs an extraction pass over the newly
//! compacted range, purely from this joiner's own transcript mirror, and
//! commits only what is safe for it to keep without host review.
//!
//! [`extraction_range`] and [`is_own_fact`] are the pure decision core, unit
//! tested with no model and no store. [`maybe_queue_extraction`] is the
//! impure dispatch [`remote_generation::LocalModelGeneration::try_handle`]
//! calls *after* a reply it generated has already released its own claim on
//! [`ACTIVE_TURN`] — never on the same claim a reply needed, and never
//! before that reply has been sent, so a `GenerateRequest` that advances
//! `compacted_through` still always gets its reply. It then claims
//! `ACTIVE_TURN` itself, independently, and hands the actual work to an
//! injected [`JoinerExtractionJob`] (mirroring
//! `remote_generation::RemoteGenerator`'s injectable seam), so this module's
//! own tests can observe "a job was queued" with a recording stub instead of
//! touching SQLite or a model. [`run_joiner_extraction`] is the production
//! job `remote_generation::LocalModelGeneration::with_local_model` builds:
//! insert a draft over the mirrored range, run
//! `compaction::extract::fill_draft`, then commit under [`is_own_fact`]'s
//! policy — `companion_state` and rules/key-quotes attributed to the
//! non-canon ("companion") voice, the closest the binary `user`/`companion`
//! extraction schema gets to "this joiner said it" (see [`is_own_fact`]'s
//! own doc comment for the multi-bot caveat). Every other item is stored
//! inactive with a reason, so nothing about the shared user/world ever
//! activates on a joiner without the host reviewing it.
//!
//! [`remote_generation`]: crate::multiplayer::remote_generation

use std::sync::Arc;

use crate::compaction::commit::{
    commit, CommitBudget, CommitDeps, CommitObserver, ReviewedDraft, ReviewedItem,
};
use crate::compaction::extract::{fill_draft, spawn_holding};
use crate::compaction::merge::LlmSummaryMerger;
use crate::compaction::registry_speakers::RegistrySpeakers;
use crate::compaction::review::fact_to_draft;
use crate::compaction::store::{CompactionStore, SqliteCompactionStore};
use crate::compaction::types::{CompactionTrigger, FactCategory, NewDraft};
use crate::compaction::CitedMessage;
use crate::context_manager::ContextManager;
use crate::database::{Database, Message};
use crate::llm::ResidentExtractor;
use crate::multiplayer::joiner::JoinerHandle;
use crate::multiplayer::protocol::ContinuityPayload;
use crate::participants::{ParticipantId, ParticipantRegistry};
use crate::turn_slot::ACTIVE_TURN;

/// Why a joiner-committed item that would otherwise be active is instead
/// stored inactive: everything except `companion_state` and this joiner's
/// own rules/key-quotes describes the shared user/world, which only the
/// host reviews.
const NOT_OWN_REASON: &str = "joiner auto-commit: category requires host review";

/// Whether this joiner's own compacted-through cursor (`local`, `None`
/// counting as `0`) is behind `payload.compacted_through`, and if so the
/// `[from, through]` range to extract over. Pure: no store, no clock.
///
/// - Advance: `local < payload.compacted_through` → `Some((local + 1,
///   payload.compacted_through))`.
/// - No advance: `local >= payload.compacted_through` → `None` (already
///   caught up, or the payload is stale/replayed).
/// - First payload: `local == None` is treated as `0`, so the very first
///   `ContinuityPayload` a joiner ever sees still queues an extraction from
///   message `1`.
pub(crate) fn extraction_range(
    local: Option<i32>,
    payload: &ContinuityPayload,
) -> Option<(i32, i32)> {
    let local = local.unwrap_or(0);
    if payload.compacted_through > local {
        Some((local + 1, payload.compacted_through))
    } else {
        None
    }
}

/// Whether a stored fact is safe for a joiner to auto-commit without host
/// review.
///
/// `companion_state` is unconditionally this joiner's own overlay: nothing
/// else in the chat could have produced it. A `rule`/`key_quote` counts as
/// "own speaker" when `quote_speaker` is the non-canon ("companion") token
/// — the closest the binary `user`/`companion` extraction schema
/// (`compaction::extract::Party`, `compaction::validate::check_verbatim`)
/// gets to attributing a line to a specific bot, since neither `Fact` nor
/// `FactDraft` records which participant actually spoke it. In a chat with
/// more than one bot this can misattribute another bot's rule/quote to this
/// joiner; tracked as a known limitation (#170) until extraction carries
/// per-participant attribution.
///
/// Every other category (`user_state`, `backstory`, `open_threads`,
/// `person`, and a `user`-attributed rule/key_quote) needs the host's
/// review: it describes the shared world/user, not this joiner's own state.
pub(crate) fn is_own_fact(category: FactCategory, quote_speaker: Option<&str>) -> bool {
    match category {
        FactCategory::CompanionState => true,
        FactCategory::Rule | FactCategory::KeyQuote => quote_speaker == Some("companion"),
        FactCategory::UserState
        | FactCategory::Milestone
        | FactCategory::Backstory
        | FactCategory::OpenThread
        | FactCategory::Person => false,
    }
}

/// One queued extraction: the already-filtered range (`from..=through`, in
/// the host's message id space) and everything the job needs to run without
/// reading `JoinerShared` itself, so the job stays free of the socket/lock
/// plumbing `maybe_queue_extraction` already handled.
pub(crate) struct JoinerExtractionRequest {
    pub companion_id: i32,
    pub self_id: ParticipantId,
    pub from: i32,
    pub through: i32,
    pub range: Vec<Message>,
    pub registry: ParticipantRegistry,
}

/// The seam a joiner's own auto-extraction job runs through — mirrors
/// `remote_generation::RemoteGenerator`'s injectable `Arc<dyn Fn>` shape, so
/// `maybe_queue_extraction`'s own dispatch tests (claim/queue/retry) can
/// inject a recording stub instead of touching SQLite or a model.
/// [`run_joiner_extraction`] is the production implementation, wired up by
/// `main.rs`'s startup code the same way `LocalModelGeneration::with_local_model`
/// is.
pub type JoinerExtractionJob = Arc<dyn Fn(JoinerExtractionRequest) + Send + Sync>;

/// Decides whether the most recent `GenerateRequest`'s payload
/// (`JoinerShared::last_continuity`, already mirrored there by
/// `joiner::serve` before a reply is even attempted) or a previously-queued
/// retry (`JoinerShared::pending_extraction`) advances this joiner's own
/// compacted-through cursor, and if so either spawns `job` under a freshly
/// claimed [`ACTIVE_TURN`] slot or — if the slot is already held elsewhere —
/// remembers the target `through` on `pending_extraction` for the next call
/// to retry.
///
/// Callers must never call this while still holding the reply's own claim
/// on [`ACTIVE_TURN`] for the same `GenerateRequest`: the sole production
/// caller, `remote_generation::LocalModelGeneration::try_handle`, calls it
/// from the reply-generation thread only *after* explicitly dropping that
/// thread's own [`crate::turn_slot::TurnGuard`], so extraction's claim here
/// is always independent of, and never racing, the reply this same frame
/// needed the slot for.
pub(crate) fn maybe_queue_extraction(handle: &JoinerHandle, job: &JoinerExtractionJob) {
    let (local, pending, last_continuity_through, snapshot, participants, companion_id, self_id) = {
        let shared = handle.read().unwrap_or_else(|p| p.into_inner());
        (
            shared.local_compacted_through,
            shared.pending_extraction,
            shared.last_continuity.as_ref().map(|p| p.compacted_through),
            shared.transcript.snapshot(),
            shared.participants.clone(),
            shared.companion_id,
            shared.participant_id.clone(),
        )
    };

    let target_through = match (last_continuity_through, pending) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    let Some(target_through) = target_through else {
        return;
    };

    let synthetic = ContinuityPayload {
        compacted_through: target_through,
        ..Default::default()
    };
    let Some((from, through)) = extraction_range(local, &synthetic) else {
        // Already caught up (or a stale/replayed payload): clear a stale
        // pending marker rather than retrying forever.
        let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
        shared.pending_extraction = None;
        return;
    };

    match ACTIVE_TURN.try_claim() {
        Some(guard) => {
            {
                let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
                shared.pending_extraction = None;
            }
            let range: Vec<Message> = snapshot
                .into_iter()
                .filter(|m| m.id >= from && m.id <= through)
                .collect();
            let registry = super::remote_generation::registry_from_participants(&participants);
            let request = JoinerExtractionRequest {
                companion_id,
                self_id,
                from,
                through,
                range,
                registry,
            };
            let job = Arc::clone(job);
            spawn_holding(guard, move || job(request));
        }
        None => {
            let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
            shared.pending_extraction = Some(target_through);
        }
    }
}

/// Reads this joiner's own locally-kept companion overlay: active
/// `companion_state` facts (rendered verbatim) plus `rule` facts attributed
/// to the non-canon ("companion") voice (see [`is_own_fact`]'s doc comment
/// on why that is the closest available "own speaker" signal), through the
/// same local [`CompactionStore`] every other local read uses — a joiner's
/// own store only ever holds its own companion's facts, so no extra key is
/// needed. Shared by `remote_generation::HostContinuity`'s two build sites
/// (a live reply, and `GET /api/debug/prompt`'s joiner branch), so they can
/// never render a different overlay from one another.
pub(crate) fn local_overlay(
    store: &dyn CompactionStore,
    companion_id: i32,
) -> rusqlite::Result<(Vec<String>, Vec<crate::compaction::context::QuoteLine>)> {
    use crate::compaction::context::{QuoteLine, QuoteSpeaker};

    let facts = store.active_facts(companion_id)?;
    let mut companion_state = Vec::new();
    let mut rules = Vec::new();
    for fact in facts {
        match fact.category {
            FactCategory::CompanionState => companion_state.push(fact.text),
            FactCategory::Rule if fact.quote_speaker.as_deref() == Some("companion") => {
                rules.push(QuoteLine {
                    speaker: QuoteSpeaker::Companion,
                    text: fact.text,
                })
            }
            _ => {}
        }
    }
    Ok((companion_state, rules))
}

/// The joiner's own [`CommitDeps`]: unlike `compaction::production_commit_deps`,
/// never registers `PersonsObserver` — a joiner's [`is_own_fact`] policy
/// already keeps every `Person` fact inactive (people are host-reviewed),
/// so nothing here should ever create a third-party row from a joiner's own
/// commit. Kept separate rather than filtering `production_commit_deps`'s
/// observer list after the fact, so a future observer added there is
/// opt-in for a joiner, not opt-out.
fn joiner_commit_deps(extractor: &dyn crate::llm::Extractor) -> CommitDeps<'_> {
    CommitDeps {
        merger: Box::new(LlmSummaryMerger { extractor }),
        observers: Vec::new() as Vec<Box<dyn CommitObserver>>,
    }
}

/// Discards `draft_id` (flips it from `Draft` to `Discarded`) and logs why,
/// used by every [`run_joiner_extraction`] failure branch that ran after
/// `insert_draft` already created the row. Without this, a failure here
/// would leave the row stuck in `Draft` status forever, and the *next*
/// retry over the same (still-uncaught-up) range would insert another draft
/// on top of it rather than replacing it — the row never becomes visible
/// anywhere (compaction routes 409 on a joiner), but it does accumulate,
/// unbounded, in this joiner's own database on every failed retry.
fn discard_or_log(
    store: &SqliteCompactionStore,
    draft_id: i64,
    self_id: &ParticipantId,
    why: &str,
) {
    match crate::compaction::commit::discard(store, draft_id) {
        Ok(()) => {}
        // Already past `Draft` status -- `fill_draft` itself already
        // discarded it for one of its own distinguishable error variants;
        // nothing further to do or report.
        Err(crate::compaction::commit::CommitError::DraftNotPending { .. }) => {}
        Err(e) => {
            eprintln!(
                "joiner compaction ({self_id}): failed to discard draft {draft_id} after {why}: {e}"
            );
        }
    }
}

/// The production [`JoinerExtractionJob`]: runs on the thread
/// [`maybe_queue_extraction`] spawned, holding [`crate::turn_slot::TurnGuard`]
/// for its whole duration. An empty `request.range` (a late joiner whose
/// mirror does not go back far enough — #170's known limitation) only
/// advances the local cursor, since there is nothing to extract. Every other
/// failure after `insert_draft` discards the draft it created
/// ([`discard_or_log`]) before returning, so a retry over the same range
/// never piles a second `Draft` row on top of one a prior attempt already
/// abandoned; the local cursor itself is left exactly where it was, so the
/// next `GenerateRequest` retries the same range from scratch.
pub(crate) fn run_joiner_extraction(handle: &JoinerHandle, request: JoinerExtractionRequest) {
    let JoinerExtractionRequest {
        companion_id,
        self_id,
        from,
        through,
        range,
        registry,
    } = request;

    let store = SqliteCompactionStore;

    if range.is_empty() {
        if let Err(e) = store.set_compacted_through(companion_id, Some(through)) {
            eprintln!(
                "joiner compaction ({self_id}): failed to advance compacted_through over an empty range: {e}"
            );
            return;
        }
        let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
        shared.local_compacted_through = Some(through);
        return;
    }

    let cited: Vec<CitedMessage> = range.iter().map(CitedMessage::from).collect();

    let draft_id = match store.insert_draft(NewDraft {
        companion_id,
        from_message_id: from,
        through_message_id: through,
        trigger: CompactionTrigger::JoinerSync,
        raw_model_output: None,
    }) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("joiner compaction ({self_id}): failed to queue a draft: {e}");
            return;
        }
    };

    let config = match Database::get_config() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("joiner compaction ({self_id}): failed to read config: {e}");
            discard_or_log(&store, draft_id, &self_id, "a config read failure");
            return;
        }
    };
    let context_manager = ContextManager::new(config);
    // #174 has not yet grown a dedicated extraction-overlay slice on
    // `TokenBudget`; a flat 15% of the chat model's total budget matches
    // `compaction::extract::run_extraction_job`'s own fallback.
    let overlay_budget_tokens = context_manager.token_budget.total * 15 / 100;

    let draft = match store.get_checkpoint(draft_id) {
        Ok(Some(draft)) => draft,
        Ok(None) => {
            eprintln!(
                "joiner compaction ({self_id}): draft {draft_id} vanished right after insert"
            );
            return;
        }
        Err(e) => {
            eprintln!("joiner compaction ({self_id}): failed to read draft {draft_id}: {e}");
            discard_or_log(&store, draft_id, &self_id, "a checkpoint read failure");
            return;
        }
    };

    let speakers = RegistrySpeakers(registry);
    if let Err(e) = fill_draft(
        &store,
        &ResidentExtractor,
        &draft,
        &cited,
        &speakers,
        overlay_budget_tokens,
    ) {
        eprintln!("joiner compaction ({self_id}): extraction failed for draft {draft_id}: {e}");
        // `fill_draft` already discards the draft itself for the error
        // variants it can distinguish (unparseable output, over-budget
        // overlays, an empty range); this is a defensive catch-all for any
        // other variant (e.g. the extractor model itself erroring) that
        // would otherwise leave the row stuck in `Draft` — `discard_or_log`
        // is a silent no-op when `fill_draft` already discarded it.
        discard_or_log(&store, draft_id, &self_id, "an extraction failure");
        return;
    }

    let stored_facts = match store.facts_for(draft_id) {
        Ok(facts) => facts,
        Err(e) => {
            eprintln!(
                "joiner compaction ({self_id}): failed to read draft {draft_id}'s facts: {e}"
            );
            discard_or_log(&store, draft_id, &self_id, "a facts read failure");
            return;
        }
    };
    let items: Vec<ReviewedItem> = stored_facts
        .into_iter()
        .map(|fact| {
            let mut draft = fact_to_draft(&fact);
            if draft.rejected_reason.is_none()
                && !is_own_fact(fact.category, fact.quote_speaker.as_deref())
            {
                draft.rejected_reason = Some(NOT_OWN_REASON.to_string());
            }
            let accepted = draft.rejected_reason.is_none();
            ReviewedItem {
                fact_id: fact.id,
                accepted,
                draft,
            }
        })
        .collect();

    // Re-read: `fill_draft` just set the checkpoint's `summary` via
    // `set_extraction_result`, which the `draft` binding above predates.
    let draft = match store.get_checkpoint(draft_id) {
        Ok(Some(draft)) => draft,
        Ok(None) => {
            eprintln!("joiner compaction ({self_id}): draft {draft_id} vanished after extraction");
            return;
        }
        Err(e) => {
            eprintln!("joiner compaction ({self_id}): failed to re-read draft {draft_id}: {e}");
            discard_or_log(
                &store,
                draft_id,
                &self_id,
                "a post-extraction checkpoint read failure",
            );
            return;
        }
    };
    let reviewed = ReviewedDraft {
        draft_id,
        items,
        summary: draft.summary.clone().unwrap_or_default(),
    };

    let user_view = match Database::get_user_data() {
        Ok(user) => user,
        Err(e) => {
            eprintln!("joiner compaction ({self_id}): failed to read user data: {e}");
            discard_or_log(&store, draft_id, &self_id, "a user-data read failure");
            return;
        }
    };
    let companion_view = match Database::get_companion_data() {
        Ok(companion) => companion,
        Err(e) => {
            eprintln!("joiner compaction ({self_id}): failed to read companion data: {e}");
            discard_or_log(&store, draft_id, &self_id, "a companion-data read failure");
            return;
        }
    };
    let budget = CommitBudget {
        compaction_slice_tokens: context_manager.compaction_token_budget,
        rolling_summary_tokens: context_manager.compaction_token_budget / 2,
        user_name: user_view.name,
        companion_name: companion_view.name,
    };

    let extractor = ResidentExtractor;
    let deps = joiner_commit_deps(&extractor);
    match commit(&store, reviewed, &deps, &budget) {
        Ok(_checkpoint) => {
            let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
            shared.local_compacted_through = Some(through);
        }
        Err(e) => {
            eprintln!("joiner compaction ({self_id}): failed to commit draft {draft_id}: {e}");
            // Without this, a failed commit leaves the draft stuck in
            // `Draft` status forever, and every retry over the same
            // (still-uncaught-up) range inserts another one on top of it
            // (PR #204 review finding).
            discard_or_log(&store, draft_id, &self_id, "a commit failure");
        }
    }
}

/// Test-only stand-in for [`JoinerExtractionJob`] that records nothing and
/// does nothing: for tests exercising `joiner::run`/`serve` end to end that
/// have no interest in compaction at all (mirrors
/// `multiplayer::joiner::tests::NoopGeneration`'s role for
/// `GenerateRequestHandler`).
#[cfg(test)]
pub(crate) fn noop_job() -> JoinerExtractionJob {
    Arc::new(|_request: JoinerExtractionRequest| {})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multiplayer::joiner::{JoinerIdentity, JoinerShared};
    use crate::multiplayer::protocol::ParticipantSummary;
    use std::sync::{Mutex, RwLock};

    // -- extraction_range --

    fn payload(compacted_through: i32) -> ContinuityPayload {
        ContinuityPayload {
            compacted_through,
            ..Default::default()
        }
    }

    #[test]
    fn extraction_range_advances_from_the_local_cursor_to_the_payloads_through() {
        assert_eq!(extraction_range(Some(5), &payload(9)), Some((6, 9)));
    }

    #[test]
    fn extraction_range_is_none_when_local_already_matches_or_leads() {
        assert_eq!(extraction_range(Some(9), &payload(9)), None);
        assert_eq!(extraction_range(Some(10), &payload(9)), None);
    }

    #[test]
    fn extraction_range_treats_a_first_ever_payload_as_starting_from_message_one() {
        assert_eq!(extraction_range(None, &payload(4)), Some((1, 4)));
    }

    // -- is_own_fact --

    #[test]
    fn companion_state_is_always_own() {
        assert!(is_own_fact(FactCategory::CompanionState, None));
        assert!(is_own_fact(FactCategory::CompanionState, Some("user")));
    }

    #[test]
    fn a_rule_or_key_quote_is_own_only_when_attributed_to_the_companion_voice() {
        assert!(is_own_fact(FactCategory::Rule, Some("companion")));
        assert!(is_own_fact(FactCategory::KeyQuote, Some("companion")));
        assert!(!is_own_fact(FactCategory::Rule, Some("user")));
        assert!(!is_own_fact(FactCategory::KeyQuote, Some("user")));
        assert!(!is_own_fact(FactCategory::Rule, None));
    }

    #[test]
    fn every_other_category_always_needs_host_review() {
        for category in [
            FactCategory::UserState,
            FactCategory::Milestone,
            FactCategory::Backstory,
            FactCategory::OpenThread,
            FactCategory::Person,
        ] {
            assert!(!is_own_fact(category, Some("companion")));
        }
    }

    // -- maybe_queue_extraction --

    fn test_identity() -> JoinerIdentity {
        JoinerIdentity {
            id: ParticipantId::parse("bot1").unwrap(),
            display_name: "Ada".to_string(),
            avatar: None,
            password: "hunter2".to_string(),
            host_address: "127.0.0.1:0".to_string(),
        }
    }

    fn handle_with(local_compacted_through: Option<i32>) -> JoinerHandle {
        let mut shared = JoinerShared::new(&test_identity(), 1, local_compacted_through);
        shared.transcript.replace(
            (1..=9)
                .map(|id| Message {
                    id,
                    ai: id % 2 == 0,
                    speaker_id: if id % 2 == 0 { "char" } else { "user" }.to_string(),
                    content: format!("message {id}"),
                    created_at: "2024-01-01T00:00:00Z".to_string(),
                })
                .collect(),
        );
        shared.participants = vec![ParticipantSummary {
            id: ParticipantId::parse("bot1").unwrap(),
            display_name: "Ada".to_string(),
            kind: crate::participants::ParticipantKind::RemoteBot,
            avatar_url: None,
            connected: true,
        }];
        Arc::new(RwLock::new(shared))
    }

    type RecordedRanges = Arc<Mutex<Vec<(i32, i32)>>>;

    fn recording_job() -> (JoinerExtractionJob, RecordedRanges) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&calls);
        let job: JoinerExtractionJob = Arc::new(move |request: JoinerExtractionRequest| {
            recorded
                .lock()
                .unwrap()
                .push((request.from, request.through));
        });
        (job, calls)
    }

    fn set_last_continuity(handle: &JoinerHandle, payload: Option<ContinuityPayload>) {
        handle
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .last_continuity = payload;
    }

    // Shares the process-wide `ACTIVE_TURN`, so every case below runs as one
    // `#[test]` function — the same convention
    // `remote_generation::tests::local_model_generation_claims_and_releases_the_shared_turn_slot`
    // documents for the same reason.
    #[test]
    fn maybe_queue_extraction_claims_spawns_queues_and_retries_correctly() {
        let handle = handle_with(None);
        let (job, calls) = recording_job();

        // A higher `compacted_through` than the local cursor (`None`, i.e.
        // 0): claims the slot and queues exactly one job over [1, 5].
        set_last_continuity(&handle, Some(payload(5)));
        maybe_queue_extraction(&handle, &job);
        // `spawn_holding` runs on its own thread; give it a moment, then
        // join by re-claiming the slot once it releases the guard.
        for _ in 0..100 {
            if ACTIVE_TURN.try_claim().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(*calls.lock().unwrap(), vec![(1, 5)]);
        assert_eq!(
            handle.read().unwrap().pending_extraction,
            None,
            "pending_extraction should have been cleared once the job was spawned"
        );

        // A lower `compacted_through` than the local cursor (simulating the
        // job having advanced it, the way the real production job does on
        // success) queues nothing further. Sleeps first so a wrongly-spawned
        // job — the regression this test exists to catch — has every chance
        // to run and record itself on its own thread before the assertion
        // below runs: checking immediately after `maybe_queue_extraction`
        // returns could observe the "nothing happened yet" state even for
        // buggy code that did wrongly spawn one (PR #204 review finding).
        handle.write().unwrap().local_compacted_through = Some(5);
        set_last_continuity(&handle, Some(payload(3)));
        maybe_queue_extraction(&handle, &job);
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert_eq!(
            *calls.lock().unwrap(),
            vec![(1, 5)],
            "a payload at or behind the local cursor must queue no job"
        );
        assert!(
            ACTIVE_TURN.try_claim().is_some(),
            "nothing should hold the turn slot when no job was queued"
        );

        // The slot claimed by someone else: the target is remembered as
        // `pending_extraction` instead of spawning, and retried once the
        // slot frees up.
        let outer_guard = ACTIVE_TURN.try_claim().expect("slot should be free");
        set_last_continuity(&handle, Some(payload(8)));
        maybe_queue_extraction(&handle, &job);
        assert_eq!(
            *calls.lock().unwrap(),
            vec![(1, 5)],
            "a job must never be spawned while the slot is held elsewhere"
        );
        assert_eq!(handle.read().unwrap().pending_extraction, Some(8));
        drop(outer_guard);

        // No fresh payload this time: `pending_extraction` alone drives the
        // retry.
        set_last_continuity(&handle, None);
        maybe_queue_extraction(&handle, &job);
        for _ in 0..100 {
            if ACTIVE_TURN.try_claim().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(
            *calls.lock().unwrap(),
            vec![(1, 5), (6, 8)],
            "the pending target should be retried once the slot is free again"
        );
        assert_eq!(handle.read().unwrap().pending_extraction, None);

        assert!(
            ACTIVE_TURN.try_claim().is_some(),
            "the slot should be free again once every spawned job has finished"
        );
    }

    // -- local_overlay --

    #[test]
    fn local_overlay_keeps_companion_state_and_own_voice_rules_only() {
        use crate::compaction::store::RecordingStore;
        use crate::compaction::types::{CompactionStatus, CompactionTrigger, FactDraft, NewDraft};

        let store = RecordingStore::new();
        let compaction_id = store
            .insert_draft(NewDraft {
                companion_id: 1,
                from_message_id: 1,
                through_message_id: 3,
                trigger: CompactionTrigger::JoinerSync,
                raw_model_output: None,
            })
            .unwrap();
        store
            .insert_facts(
                compaction_id,
                &[
                    FactDraft {
                        category: FactCategory::CompanionState,
                        subject: None,
                        text: "is cheerful".to_string(),
                        quote_speaker: None,
                        sources: vec![1],
                        replaces: vec![],
                        relation_to: None,
                        relation: None,
                        canon: true,
                        rejected_reason: None,
                    },
                    FactDraft {
                        category: FactCategory::Rule,
                        subject: None,
                        text: "always speaks formally".to_string(),
                        quote_speaker: Some("companion".to_string()),
                        sources: vec![1],
                        replaces: vec![],
                        relation_to: None,
                        relation: None,
                        canon: true,
                        rejected_reason: None,
                    },
                    FactDraft {
                        category: FactCategory::Rule,
                        subject: None,
                        text: "never mention the surprise".to_string(),
                        quote_speaker: Some("user".to_string()),
                        sources: vec![1],
                        replaces: vec![],
                        relation_to: None,
                        relation: None,
                        canon: true,
                        rejected_reason: None,
                    },
                    FactDraft {
                        category: FactCategory::UserState,
                        subject: None,
                        text: "loves cats".to_string(),
                        quote_speaker: None,
                        sources: vec![1],
                        replaces: vec![],
                        relation_to: None,
                        relation: None,
                        canon: false,
                        rejected_reason: None,
                    },
                ],
            )
            .unwrap();
        store
            .update_status(compaction_id, CompactionStatus::Committed)
            .unwrap();

        let (companion_state, rules) = local_overlay(&store, 1).unwrap();

        assert_eq!(companion_state, vec!["is cheerful".to_string()]);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].text, "always speaks formally");
    }
}
