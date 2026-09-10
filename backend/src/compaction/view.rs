//! HTTP response DTOs for the compaction routes (#179). Every handler in
//! `main.rs` serialises one of these, never a domain type from `types.rs`
//! directly: [`FactView`] flattens [`FactSubject`] to a bare string and
//! [`CheckpointDetail`] carries data ([`AttitudePreview`], `phase`) no
//! single domain struct has on its own.

use serde::{Deserialize, Serialize};

use crate::compaction::extract::AttitudeRatings;
use crate::compaction::types::{
    Checkpoint, CompactionStatus, CompactionTrigger, Fact, FactCategory, FactSubject,
};
use crate::database::CompanionAttitude;

/// `FactSubject::User`/`Companion` as their bare token; `Person(name)` as
/// the bare name, dropping the `person:` prefix `FactSubject`'s own
/// `Display` carries for storage — the view never needs to tell "a person
/// literally named `user`" apart from `FactSubject::User`, so it always
/// renders the friendlier bare form.
fn subject_to_string(subject: &FactSubject) -> String {
    match subject {
        FactSubject::User => "user".to_string(),
        FactSubject::Companion => "companion".to_string(),
        FactSubject::Person(name) => name.clone(),
    }
}

/// What phase a pending draft is in, derived from
/// `Checkpoint::raw_model_output`: `NULL` means extraction is still
/// running, set means it is ready for the user to review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DraftPhase {
    Extracting,
    Review,
}

/// `Some` only while `checkpoint.status == Draft`; every other status has
/// no phase.
fn phase_of(checkpoint: &Checkpoint) -> Option<DraftPhase> {
    if checkpoint.status != CompactionStatus::Draft {
        return None;
    }
    Some(if checkpoint.raw_model_output.is_some() {
        DraftPhase::Review
    } else {
        DraftPhase::Extracting
    })
}

/// One committed/discarded/stale checkpoint (or a pending draft, which also
/// gets one of these) as `GET /api/compaction` lists it.
#[derive(Debug, Clone, Serialize)]
pub struct CheckpointSummary {
    pub id: i64,
    pub from_message_id: i32,
    pub through_message_id: i32,
    pub status: CompactionStatus,
    pub trigger: CompactionTrigger,
    pub committed_at: Option<String>,
    pub needs_merge: bool,
}

impl From<&Checkpoint> for CheckpointSummary {
    fn from(checkpoint: &Checkpoint) -> Self {
        CheckpointSummary {
            id: checkpoint.id,
            from_message_id: checkpoint.from_message_id,
            through_message_id: checkpoint.through_message_id,
            status: checkpoint.status,
            trigger: checkpoint.trigger,
            committed_at: checkpoint.committed_at.clone(),
            needs_merge: checkpoint.needs_merge,
        }
    }
}

impl From<Checkpoint> for CheckpointSummary {
    fn from(checkpoint: Checkpoint) -> Self {
        CheckpointSummary::from(&checkpoint)
    }
}

/// The one `status = draft` row, as `GET /api/compaction`'s
/// `pending_draft` field.
#[derive(Debug, Clone, Serialize)]
pub struct PendingDraftSummary {
    pub id: i64,
    pub from_message_id: i32,
    pub through_message_id: i32,
    pub created_at: String,
    pub phase: DraftPhase,
}

impl From<&Checkpoint> for PendingDraftSummary {
    fn from(checkpoint: &Checkpoint) -> Self {
        PendingDraftSummary {
            id: checkpoint.id,
            from_message_id: checkpoint.from_message_id,
            through_message_id: checkpoint.through_message_id,
            created_at: checkpoint.created_at.clone(),
            // `phase_of` is `None` only for a non-`Draft` status, which
            // `pending_draft`/`get_draft` never return here.
            phase: phase_of(checkpoint).unwrap_or(DraftPhase::Extracting),
        }
    }
}

/// `GET /api/compaction`'s whole body.
#[derive(Debug, Clone, Serialize)]
pub struct CompactionListing {
    pub checkpoints: Vec<CheckpointSummary>,
    pub pending_draft: Option<PendingDraftSummary>,
}

/// One `compaction_facts` row, `subject`/`relation_to` flattened to bare
/// strings. Rejected facts are included (with `rejected_reason` set) so the
/// review card can show why an item did not make it in.
#[derive(Debug, Clone, Serialize)]
pub struct FactView {
    pub id: i64,
    pub category: FactCategory,
    pub subject: Option<String>,
    pub text: String,
    pub quote_speaker: Option<String>,
    pub sources: Vec<i32>,
    pub replaces: Vec<i64>,
    pub relation_to: Option<String>,
    pub relation: Option<String>,
    pub canon: bool,
    pub active: bool,
    pub superseded_by: Option<i64>,
    pub rejected_reason: Option<String>,
}

impl From<&Fact> for FactView {
    fn from(fact: &Fact) -> Self {
        FactView {
            id: fact.id,
            category: fact.category,
            subject: fact.subject.as_ref().map(subject_to_string),
            text: fact.text.clone(),
            quote_speaker: fact.quote_speaker.clone(),
            sources: fact.sources.clone(),
            replaces: fact.replaces.clone(),
            relation_to: fact.relation_to.as_ref().map(subject_to_string),
            relation: fact.relation.clone(),
            canon: fact.canon,
            active: fact.active,
            superseded_by: fact.superseded_by,
            rejected_reason: fact.rejected_reason.clone(),
        }
    }
}

/// The companion's attitude toward the user, three ways: `current` (live,
/// unaffected by this draft), `rated` (this draft's own narrative rating,
/// parsed from `Checkpoint::attitude_ratings`, `None` while still
/// extracting or if the model never emitted one), and `blended` (`current`
/// nudged toward `rated`; always `None` until #176 fills it in, in this
/// same `From` impl).
#[derive(Debug, Clone, Serialize)]
pub struct AttitudePreview {
    pub current: CompanionAttitude,
    pub rated: Option<AttitudeRatings>,
    pub blended: Option<CompanionAttitude>,
}

/// `GET /api/compaction/{id}`'s body: a [`CheckpointSummary`] plus
/// everything the review card needs that summary alone does not carry.
#[derive(Debug, Clone, Serialize)]
pub struct CheckpointDetail {
    #[serde(flatten)]
    pub summary: CheckpointSummary,
    pub phase: Option<DraftPhase>,
    pub summary_text: Option<String>,
    pub rolling_summary: Option<String>,
    pub facts: Vec<FactView>,
    pub attitude: AttitudePreview,
}

impl CheckpointDetail {
    /// Builds the detail view: `checkpoint`/`facts` come straight from the
    /// store, `current_attitude` is the companion's live attitude toward
    /// the user (`Database::get_attitude`, seeded if the row does not exist
    /// yet). `rated` is parsed from the checkpoint's own
    /// `attitude_ratings` JSON; a malformed or absent value is `None`
    /// rather than a hard error, since the attitude preview is advisory.
    pub fn new(
        checkpoint: &Checkpoint,
        facts: &[Fact],
        current_attitude: CompanionAttitude,
    ) -> Self {
        let rated = checkpoint
            .attitude_ratings
            .as_deref()
            .and_then(|raw| serde_json::from_str(raw).ok());
        CheckpointDetail {
            summary: CheckpointSummary::from(checkpoint),
            phase: phase_of(checkpoint),
            summary_text: checkpoint.summary.clone(),
            rolling_summary: checkpoint.rolling_summary.clone(),
            facts: facts.iter().map(FactView::from).collect(),
            attitude: AttitudePreview {
                current: current_attitude,
                rated,
                blended: None,
            },
        }
    }
}

/// `202` body for a freshly queued draft, from either
/// `POST /api/compaction/draft` or the streamed/non-streamed prompting
/// handlers' `compaction_draft_id`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct DraftQueued {
    pub draft_id: i64,
}

/// `POST /api/prompt`'s response body. Was a bare `text/plain` reply
/// before #179; the frontend never calls this route, so the shape change
/// is external-caller-only (see `docs/api_docs.md` section 6.1).
#[derive(Debug, Clone, Serialize)]
pub struct PromptResponse {
    pub reply: String,
    pub compaction_draft_id: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::types::FactSubject;

    fn a_checkpoint(status: CompactionStatus, raw_model_output: Option<&str>) -> Checkpoint {
        Checkpoint {
            id: 1,
            companion_id: 1,
            from_message_id: 1,
            through_message_id: 10,
            status,
            trigger: CompactionTrigger::Threshold,
            raw_model_output: raw_model_output.map(str::to_string),
            summary: Some("summary".to_string()),
            rolling_summary: None,
            attitude_ratings: None,
            needs_merge: false,
            created_at: "now".to_string(),
            committed_at: None,
        }
    }

    #[test]
    fn a_draft_with_no_raw_output_is_extracting() {
        let checkpoint = a_checkpoint(CompactionStatus::Draft, None);
        assert_eq!(phase_of(&checkpoint), Some(DraftPhase::Extracting));
    }

    #[test]
    fn a_draft_with_raw_output_is_ready_for_review() {
        let checkpoint = a_checkpoint(CompactionStatus::Draft, Some("raw"));
        assert_eq!(phase_of(&checkpoint), Some(DraftPhase::Review));
    }

    #[test]
    fn a_committed_checkpoint_has_no_phase() {
        let checkpoint = a_checkpoint(CompactionStatus::Committed, Some("raw"));
        assert_eq!(phase_of(&checkpoint), None);
    }

    #[test]
    fn person_subject_flattens_to_the_bare_name_without_the_person_prefix() {
        assert_eq!(
            subject_to_string(&FactSubject::Person("Ann".to_string())),
            "Ann"
        );
        assert_eq!(subject_to_string(&FactSubject::User), "user");
        assert_eq!(subject_to_string(&FactSubject::Companion), "companion");
    }
}
