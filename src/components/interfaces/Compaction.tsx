import { AttitudeData } from './AttitudeData';

// Mirrors the backend's `compaction::types::CompactionStatus`. `failed`
// (#208) is a checkpoint whose extraction pipeline itself errored (a
// model/store I/O failure) before it ever judged the draft's content --
// distinct from `discarded`, which covers both a user's deliberate
// rejection and `fill_draft`'s own content-driven discards (an empty
// range, unparseable output twice, an over-budget overlay).
export type CompactionStatus = 'draft' | 'committed' | 'discarded' | 'stale' | 'failed';

// Mirrors the backend's `compaction::types::CompactionTrigger`. `joiner_sync`
// (#186) never reaches the frontend: the compaction routes 409 on a joiner
// instance, so a checkpoint carrying this trigger is never listed here.
export type CompactionTrigger = 'threshold' | 'scene_break' | 'manual' | 'joiner_sync';

// Mirrors the backend's `compaction::types::FactCategory`.
export type FactCategory =
    | 'companion_state'
    | 'user_state'
    | 'milestone'
    | 'backstory'
    | 'open_thread'
    | 'rule'
    | 'key_quote'
    | 'person';

// Mirrors the backend's `compaction::view::DraftPhase`: `extracting` while
// the model is still running over a queued draft, `review` once it is ready
// for the user to review.
export type DraftPhase = 'extracting' | 'review';

// Mirrors the backend's `compaction::view::CheckpointSummary`.
// `extraction_error` (#208) is set only when `status === 'failed'`: why
// extraction never produced a draft to review.
export interface CheckpointSummary {
    id: number;
    from_message_id: number;
    through_message_id: number;
    status: CompactionStatus;
    trigger: CompactionTrigger;
    committed_at: string | null;
    needs_merge: boolean;
    extraction_error: string | null;
}

// Mirrors the backend's `compaction::view::PendingDraftSummary`.
export interface PendingDraftSummary {
    id: number;
    from_message_id: number;
    through_message_id: number;
    created_at: string;
    phase: DraftPhase;
}

// `GET /api/compaction`'s body, mirroring `compaction::view::CompactionListing`.
export interface CompactionListing {
    checkpoints: CheckpointSummary[];
    pending_draft: PendingDraftSummary | null;
}

// Mirrors the backend's `compaction::view::FactView`: `subject`/`relation_to`
// are already flattened to a bare string (`"user"`, `"companion"`, or a
// person's name) server-side.
export interface CompactionFact {
    id: number;
    category: FactCategory;
    subject: string | null;
    text: string;
    quote_speaker: string | null;
    sources: number[];
    replaces: number[];
    relation_to: string | null;
    relation: string | null;
    canon: boolean;
    active: boolean;
    superseded_by: number | null;
    rejected_reason: string | null;
}

// The eight `attitude_engine::AttitudeDimension` ratings a draft's own
// extraction pass emitted, mirroring the backend's
// `compaction::extract::AttitudeRatings` (0-100 scale, already clamped).
export interface AttitudeRatings {
    trust: number;
    love: number;
    fear: number;
    anger: number;
    joy: number;
    sorrow: number;
    suspicion: number;
    gratitude: number;
}

// Mirrors the backend's `compaction::view::AttitudePreview`: the companion's
// attitude toward the user, three ways. `rated` is `null` while a draft is
// still extracting or if the model never emitted one; `blended` is always
// `null` until a later issue fills it in.
export interface AttitudePreview {
    current: AttitudeData;
    rated: AttitudeRatings | null;
    blended: AttitudeData | null;
}

// One contradiction against a curated running thought (#219), mirroring the
// backend's `compaction::view::ContradictionView`. `fact_id: null` names the
// checkpoint's summary rather than one of `CheckpointDetail.facts`.
export interface ContradictionView {
    fact_id: number | null;
    thought_id: number;
    thought_text: string;
    quote: string;
}

// `GET /api/compaction/{id}`'s body, mirroring the backend's
// `compaction::view::CheckpointDetail` (its `#[serde(flatten)] summary`
// field flattens onto this same object).
export interface CheckpointDetail extends CheckpointSummary {
    phase: DraftPhase | null;
    summary_text: string | null;
    rolling_summary: string | null;
    facts: CompactionFact[];
    attitude: AttitudePreview;
    contradictions: ContradictionView[];
}

// One item's review, mirroring the backend's `compaction::review::ItemReview`
// (the wire shape `POST /api/compaction/{id}/commit` accepts, one entry per
// fact the user actually touched). `id` is the fact's id from
// `CheckpointDetail.facts`.
export interface ItemReview {
    id: number;
    accepted: boolean;
    text?: string;
    quote?: string;
}

// `POST /api/compaction/{id}/commit`'s request body, mirroring the backend's
// `compaction::review::CommitRequest`.
export interface CompactionDraftReview {
    items: ItemReview[];
    summary?: string;
}

// One item `apply_review` (or #219's commit-time re-check) rejected,
// mirroring the backend's `compaction::review::RejectedItem` — the `422`
// body shape. `item_id: null` names the checkpoint's summary rather than a
// fact; only the re-check produces that -- a plain `apply_review` rejection
// always carries a fact id.
export interface RejectedItem {
    item_id: number | null;
    reason: string;
}
