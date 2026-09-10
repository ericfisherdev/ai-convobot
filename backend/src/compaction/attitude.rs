//! Blends the extraction model's narrative rating of the relationship into
//! the lexicon scorer's running attitude values at commit time (#176), so
//! the attitude block stops contradicting the story a checkpoint just
//! summarised. Runs as a [`crate::compaction::commit::CommitObserver`], so a
//! failure here can never fail the commit that produced it.
//!
//! [`blended`]/[`blend`] are pure: no `Database`, no I/O, built on
//! [`crate::compaction::extract::AttitudeRatings`] (#173/#185's typed shape
//! for the model's `attitude` object — not redeclared here) and
//! [`crate::attitude_engine::AttitudeDimension::set_value`] (this issue's
//! addition, the mutable counterpart of `value_of`). [`AttitudeSink`] is the
//! persistence seam `AttitudeRecalibrator` runs the blend through:
//! `SqliteAttitudeSink::apply_blend` forwards to
//! `Database::apply_attitude_deltas_computed`, which reads the current
//! attitude row and calls back into [`blend`] *inside* its own write
//! transaction — never a separate read-then-blend-then-write across two
//! connections, which would let a concurrent writer's change land in the
//! gap and blend the rating against a value that is no longer current by
//! write time. `SqliteAttitudeSink::remember` shares
//! `Database::insert_attitude_memory` with the lexicon-scored path, so the
//! two paths converge on one attitude row and one `attitude_memories`
//! writer.

use crate::attitude_engine::{AttitudeDimension, DimensionDelta};
use crate::attitude_formatter::AttitudeFormatter;
use crate::compaction::commit::CommitObserver;
use crate::compaction::extract::AttitudeRatings;
use crate::compaction::types::{Checkpoint, Fact};
use crate::database::{AttitudeMemoryDraft, CompanionAttitude, ConfigView, Database};

/// Default blend weight when a config row predates this column (matches
/// `database.rs`'s `compaction_attitude_weight REAL DEFAULT 0.5`).
pub const DEFAULT_BLEND_WEIGHT: f32 = 0.5;

/// The eight rated dimensions, paired with their `attitude_engine`
/// counterpart. Walking this once here means [`blended`]/[`blend`] never
/// repeat a field-to-dimension match of their own.
fn rated_dimensions(ratings: &AttitudeRatings) -> [(AttitudeDimension, f32); 8] {
    [
        (AttitudeDimension::Trust, ratings.trust as f32),
        (AttitudeDimension::Love, ratings.love as f32),
        (AttitudeDimension::Fear, ratings.fear as f32),
        (AttitudeDimension::Anger, ratings.anger as f32),
        (AttitudeDimension::Joy, ratings.joy as f32),
        (AttitudeDimension::Sorrow, ratings.sorrow as f32),
        (AttitudeDimension::Suspicion, ratings.suspicion as f32),
        (AttitudeDimension::Gratitude, ratings.gratitude as f32),
    ]
}

/// Mirrors the `relationship_score` generated column's formula (the
/// `companion_attitudes` DDL in `database.rs`), so a value computed here on
/// a not-yet-persisted [`CompanionAttitude`] (as [`blended`] returns) always
/// matches what SQLite would compute once that row is actually written.
pub fn relationship_score_of(attitude: &CompanionAttitude) -> f32 {
    (attitude.attraction
        + attitude.trust
        + attitude.joy
        + attitude.respect
        + attitude.gratitude
        + attitude.empathy
        + attitude.love
        + attitude.lust
        + attitude.butterflies
        - attitude.fear
        - attitude.anger
        - attitude.sorrow
        - attitude.disgust
        - attitude.suspicion
        - attitude.jealousy
        - attitude.anxiety)
        / 16.0
}

/// What `current` would become if blended `weight` of the way toward
/// `ratings`, clamped to the `companion_attitudes` column bounds
/// (`-100..100`) the same way `Database::apply_attitude_deltas` clamps a
/// lexicon-scored turn. `weight` is clamped to `0.0..=1.0` first, so an
/// out-of-range config value degrades to the nearest valid blend rather than
/// overshooting. Only the eight rated dimensions move; every other
/// dimension is copied from `current` unchanged. `relationship_score` on the
/// result is recomputed via [`relationship_score_of`], so a caller showing
/// this as a preview (#179's `CheckpointDetail.attitude.blended`) sees the
/// tier it would actually land on.
pub fn blended(
    current: &CompanionAttitude,
    ratings: &AttitudeRatings,
    weight: f32,
) -> CompanionAttitude {
    let w = weight.clamp(0.0, 1.0);
    let mut result = current.clone();
    for (dimension, rated) in rated_dimensions(ratings) {
        let value = ((1.0 - w) * dimension.value_of(current) + w * rated).clamp(-100.0, 100.0);
        dimension.set_value(&mut result, value);
    }
    result.relationship_score = Some(relationship_score_of(&result));
    result
}

/// The deltas [`blended`] implies, one per rated dimension whose movement is
/// non-negligible (`f32::EPSILON`, not `AttitudeFormatter::CHANGE_THRESHOLD`
/// — that threshold is for what's worth *reporting*, not for what's worth
/// *applying*). `weight == 0.0` returns an empty vector. `current` should be
/// the value the caller is about to write against — `SqliteAttitudeSink`
/// calls this back from inside `Database::apply_attitude_deltas_computed`'s
/// own transaction, with the row that same transaction just read, so the
/// blend is never computed against a snapshot that could go stale before
/// the write lands.
pub fn blend(
    current: &CompanionAttitude,
    ratings: &AttitudeRatings,
    weight: f32,
) -> Vec<DimensionDelta> {
    let target = blended(current, ratings, weight);
    rated_dimensions(ratings)
        .into_iter()
        .filter_map(|(dimension, _)| {
            let delta = dimension.value_of(&target) - dimension.value_of(current);
            if delta.abs() < f32::EPSILON {
                None
            } else {
                Some(DimensionDelta { dimension, delta })
            }
        })
        .collect()
}

/// The persistence seam [`AttitudeRecalibrator`] runs its blend through,
/// mirroring `chat_turn::TurnStore`: lets the observer's commit-time logic
/// be unit-tested against an in-memory fake instead of
/// `companion_database.db`.
///
/// `apply_blend` takes `ratings`/`weight` rather than a precomputed
/// `Vec<DimensionDelta>` on purpose: the deltas must be derived from the
/// attitude row's value at write time, not from an earlier read on a
/// separate connection, so only an implementation that owns the write
/// transaction can compute them safely. `SqliteAttitudeSink` does that via
/// `Database::apply_attitude_deltas_computed`.
pub trait AttitudeSink {
    fn apply_blend(
        &self,
        companion_id: i32,
        user_id: i32,
        ratings: &AttitudeRatings,
        weight: f32,
    ) -> rusqlite::Result<Option<(CompanionAttitude, CompanionAttitude)>>;
    fn remember(
        &self,
        companion_id: i32,
        user_id: i32,
        draft: &AttitudeMemoryDraft,
    ) -> rusqlite::Result<()>;
}

/// The production [`AttitudeSink`]: forwards to `Database` associated
/// functions, always against `target_type = "user"` — ratings are the
/// companion's feelings toward the user only, matching `chat_turn::finish_turn`,
/// never a third-party target.
pub struct SqliteAttitudeSink;

impl AttitudeSink for SqliteAttitudeSink {
    fn apply_blend(
        &self,
        companion_id: i32,
        user_id: i32,
        ratings: &AttitudeRatings,
        weight: f32,
    ) -> rusqlite::Result<Option<(CompanionAttitude, CompanionAttitude)>> {
        // `compute` only runs after `apply_attitude_deltas_computed` has
        // read the row inside its own `Immediate` transaction, so `blend`
        // always sees the value it is about to write against — no window
        // for a concurrent writer to invalidate the numbers in between.
        Database::apply_attitude_deltas_computed(companion_id, user_id, "user", |current| {
            blend(current, ratings, weight)
        })
    }

    fn remember(
        &self,
        companion_id: i32,
        user_id: i32,
        draft: &AttitudeMemoryDraft,
    ) -> rusqlite::Result<()> {
        // No `message_context`: this memory isn't about anything the user
        // said in one turn, so `format_attitude_memories` renders no "(when
        // you said: ...)" suffix for it — just the `[moved: ...]` deltas
        // every other row already carries via `attitude_delta_json`.
        Database::insert_attitude_memory(companion_id, user_id, "user", draft, "")
    }
}

/// Recalibrates the companion's attitude toward the user at commit time,
/// blending the checkpoint's `attitude_ratings` into the current running
/// values via [`blend`]. Registered into [`crate::compaction::CommitDeps`]
/// by [`crate::compaction::production_commit_deps`].
///
/// Every failure path here is logged and swallowed (`on_committed` always
/// returns `Ok`): the checkpoint's facts are already durable by the time
/// this runs, and an attitude failure must never be mistaken for a failed
/// commit, the same rule `chat_turn::finish_turn` documents for the
/// lexicon-scored path.
pub struct AttitudeRecalibrator<S: AttitudeSink> {
    sink: S,
    user_id: i32,
    weight: f32,
}

impl<S: AttitudeSink> AttitudeRecalibrator<S> {
    pub fn new(sink: S, user_id: i32, weight: f32) -> Self {
        Self {
            sink,
            user_id,
            weight,
        }
    }
}

impl AttitudeRecalibrator<SqliteAttitudeSink> {
    /// The production constructor: `config.compaction_attitude_weight`
    /// (defaults to [`DEFAULT_BLEND_WEIGHT`] for a config row that predates
    /// the column, via `database.rs::read_config`) is the blend weight,
    /// `user_id` is the caller's — every `main.rs` handler passes the
    /// constant `1`, the same way handlers pass it to `PendingTurn::begin`.
    pub fn from_config(
        config: &ConfigView,
        user_id: i32,
    ) -> AttitudeRecalibrator<SqliteAttitudeSink> {
        AttitudeRecalibrator::new(
            SqliteAttitudeSink,
            user_id,
            config.compaction_attitude_weight,
        )
    }
}

impl<S: AttitudeSink> CommitObserver for AttitudeRecalibrator<S> {
    fn on_committed(
        &self,
        checkpoint: &Checkpoint,
        _facts: &[Fact],
        _superseded: &[i64],
    ) -> Result<(), String> {
        let Some(raw) = checkpoint
            .attitude_ratings
            .as_deref()
            .filter(|s| !s.is_empty())
        else {
            // A draft the model rated nothing on is not an error: older
            // checkpoints, or a chunked extraction whose merge dropped the
            // field, simply have nothing to recalibrate.
            return Ok(());
        };

        let ratings: AttitudeRatings = match serde_json::from_str(raw) {
            Ok(ratings) => ratings,
            Err(e) => {
                eprintln!(
                    "compaction attitude recalibration: checkpoint {} has unparseable attitude_ratings, skipping: {e}",
                    checkpoint.id
                );
                return Ok(());
            }
        };

        match self
            .sink
            .apply_blend(checkpoint.companion_id, self.user_id, &ratings, self.weight)
        {
            Ok(Some((previous, updated))) => {
                let formatter = AttitudeFormatter::new();
                if formatter.diff_attitudes(&previous, &updated).is_empty() {
                    // The blend, computed against the row's value at write
                    // time, moved nothing worth reporting (weight `0`, or
                    // the row already matched the rating) — nothing to log
                    // or remember.
                    return Ok(());
                }
                let changes = formatter.format_attitude_changes_for_console(&previous, &updated);
                if !changes.is_empty() {
                    println!("{changes}");
                }
                let draft =
                    crate::database::recalibration_memory_draft(&previous, &updated, checkpoint.id);
                if let Err(e) = self
                    .sink
                    .remember(checkpoint.companion_id, self.user_id, &draft)
                {
                    eprintln!("compaction attitude recalibration: failed to record memory: {e}");
                }
            }
            Ok(None) => {
                // No seeding here: `chat_turn::finish_turn` seeds the row on
                // the first turn, and a commit with no prior turn has
                // nothing to recalibrate.
                eprintln!(
                    "compaction attitude recalibration: no attitude row yet for companion {}, skipping",
                    checkpoint.companion_id
                );
            }
            Err(e) => {
                eprintln!("compaction attitude recalibration: failed to apply blend: {e}");
            }
        }

        Ok(())
    }
}

#[cfg(test)]
pub(crate) struct RecordingSink {
    current: std::sync::Mutex<Option<CompanionAttitude>>,
    pub(crate) applied: std::sync::Mutex<Vec<Vec<DimensionDelta>>>,
    pub(crate) remembered: std::sync::Mutex<Vec<AttitudeMemoryDraft>>,
    apply_fails: bool,
}

#[cfg(test)]
impl RecordingSink {
    pub(crate) fn new(current: Option<CompanionAttitude>) -> Self {
        Self {
            current: std::sync::Mutex::new(current),
            applied: std::sync::Mutex::new(Vec::new()),
            remembered: std::sync::Mutex::new(Vec::new()),
            apply_fails: false,
        }
    }

    pub(crate) fn failing_on_apply(current: Option<CompanionAttitude>) -> Self {
        Self {
            apply_fails: true,
            ..Self::new(current)
        }
    }
}

#[cfg(test)]
impl AttitudeSink for RecordingSink {
    fn apply_blend(
        &self,
        _companion_id: i32,
        _user_id: i32,
        ratings: &AttitudeRatings,
        weight: f32,
    ) -> rusqlite::Result<Option<(CompanionAttitude, CompanionAttitude)>> {
        if self.apply_fails {
            return Err(rusqlite::Error::QueryReturnedNoRows);
        }
        // Mirrors `Database::apply_attitude_deltas_computed`'s shape: read
        // `previous`, then derive the deltas from that same value, so the
        // fake behaves like the production seam it stands in for.
        let Some(previous) = self.current.lock().unwrap().clone() else {
            return Ok(None);
        };
        let deltas = blend(&previous, ratings, weight);
        let mut updated = previous.clone();
        for delta in &deltas {
            let value = (delta.dimension.value_of(&updated) + delta.delta).clamp(-100.0, 100.0);
            delta.dimension.set_value(&mut updated, value);
        }
        updated.relationship_score = Some(relationship_score_of(&updated));
        *self.current.lock().unwrap() = Some(updated.clone());
        self.applied.lock().unwrap().push(deltas);
        Ok(Some((previous, updated)))
    }

    fn remember(
        &self,
        _companion_id: i32,
        _user_id: i32,
        draft: &AttitudeMemoryDraft,
    ) -> rusqlite::Result<()> {
        self.remembered.lock().unwrap().push(draft.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::types::{CompactionStatus, CompactionTrigger};

    /// A `CompanionAttitude` with every dimension at `value`, mirroring
    /// `attitude_engine.rs`'s own `attitude_with` test helper.
    fn attitude_with(value: f32) -> CompanionAttitude {
        CompanionAttitude {
            id: None,
            companion_id: 1,
            target_id: 1,
            target_type: "user".to_string(),
            attraction: value,
            trust: value,
            fear: value,
            anger: value,
            joy: value,
            sorrow: value,
            disgust: value,
            surprise: value,
            curiosity: value,
            respect: value,
            suspicion: value,
            gratitude: value,
            jealousy: value,
            empathy: value,
            lust: value,
            love: value,
            anxiety: value,
            butterflies: value,
            submissiveness: value,
            dominance: value,
            relationship_score: Some(0.0),
            last_updated: "now".to_string(),
            created_at: "now".to_string(),
        }
    }

    fn ratings(value: i32) -> AttitudeRatings {
        serde_json::from_value(serde_json::json!({
            "trust": value,
            "love": value,
            "fear": value,
            "anger": value,
            "joy": value,
            "sorrow": value,
            "suspicion": value,
            "gratitude": value,
        }))
        .unwrap()
    }

    fn a_checkpoint(attitude_ratings: Option<String>) -> Checkpoint {
        Checkpoint {
            id: 1,
            companion_id: 1,
            from_message_id: 1,
            through_message_id: 10,
            status: CompactionStatus::Committed,
            trigger: CompactionTrigger::Threshold,
            raw_model_output: Some("raw".to_string()),
            summary: Some("summary".to_string()),
            rolling_summary: Some(String::new()),
            attitude_ratings,
            needs_merge: false,
            created_at: "now".to_string(),
            committed_at: Some("now".to_string()),
            extraction_error: None,
        }
    }

    #[test]
    fn blend_at_weight_zero_leaves_values_unchanged() {
        let current = attitude_with(20.0);
        let deltas = blend(&current, &ratings(90), 0.0);
        assert!(deltas.is_empty());
    }

    #[test]
    fn blend_at_weight_one_adopts_the_rating_for_every_rated_dimension() {
        let current = attitude_with(20.0);
        let target = blended(&current, &ratings(90), 1.0);
        for (dimension, _) in rated_dimensions(&ratings(90)) {
            assert_eq!(dimension.value_of(&target), 90.0);
        }
    }

    #[test]
    fn blend_at_weight_half_moves_halfway_to_the_rating() {
        let current = attitude_with(20.0);
        let target = blended(&current, &ratings(80), 0.5);
        assert_eq!(target.trust, 50.0);
        assert_eq!(target.love, 50.0);
    }

    #[test]
    fn blend_weight_above_one_clamps_to_one() {
        let current = attitude_with(20.0);
        let clamped = blended(&current, &ratings(90), 2.0);
        let uncapped = blended(&current, &ratings(90), 1.0);
        assert_eq!(clamped.trust, uncapped.trust);
    }

    #[test]
    fn blend_weight_below_zero_clamps_to_zero() {
        let current = attitude_with(20.0);
        let clamped = blended(&current, &ratings(90), -1.0);
        assert_eq!(clamped.trust, current.trust);
    }

    #[test]
    fn blend_never_overshoots_the_column_bounds() {
        let mut current = attitude_with(-100.0);
        current.trust = -100.0;
        let target = blended(&current, &ratings(100), 1.0);
        assert!(target.trust <= 100.0 && target.trust >= -100.0);
    }

    #[test]
    fn blend_from_a_ratings_json_with_only_one_dimension_still_moves_all_eight() {
        // `AttitudeRatings` always deserializes all eight fields (the model's
        // grammar requires them all); this documents that a caller cannot
        // partially rate the attitude block.
        let raw = r#"{"trust":70,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0}"#;
        let parsed: AttitudeRatings = serde_json::from_str(raw).unwrap();
        let current = attitude_with(0.0);
        let deltas = blend(&current, &parsed, 1.0);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].dimension, AttitudeDimension::Trust);
    }

    #[test]
    fn blend_ignores_unknown_json_keys() {
        let raw = r#"{"trust":10,"love":10,"fear":10,"anger":10,"joy":10,"sorrow":10,"suspicion":10,"gratitude":10,"mystery":99}"#;
        let parsed: AttitudeRatings = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.trust, 10);
    }

    #[test]
    fn blended_leaves_unrated_fields_identical_to_current() {
        let mut current = attitude_with(20.0);
        current.attraction = 33.0;
        current.respect = 44.0;
        let target = blended(&current, &ratings(90), 1.0);
        assert_eq!(target.attraction, 33.0);
        assert_eq!(target.respect, 44.0);
    }

    #[test]
    fn relationship_score_of_matches_the_generated_column_formula() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = crate::database::Database::open_at(dir.path().join("t.db")).unwrap();
        con.execute(
            "CREATE TABLE companion_attitudes (
                attraction REAL, trust REAL, fear REAL, anger REAL, joy REAL,
                sorrow REAL, disgust REAL, surprise REAL, curiosity REAL,
                respect REAL, suspicion REAL, gratitude REAL, jealousy REAL,
                empathy REAL, lust REAL, love REAL, anxiety REAL,
                butterflies REAL, submissiveness REAL, dominance REAL,
                relationship_score REAL GENERATED ALWAYS AS ((attraction + trust + joy + respect + gratitude + empathy + love + lust + butterflies - fear - anger - sorrow - disgust - suspicion - jealousy - anxiety) / 16.0) STORED
            )",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO companion_attitudes (
                attraction, trust, fear, anger, joy, sorrow, disgust, surprise,
                curiosity, respect, suspicion, gratitude, jealousy, empathy,
                lust, love, anxiety, butterflies, submissiveness, dominance
            ) VALUES (12, 34, 5, 6, 22, 3, 1, 2, 4, 18, 9, 15, 2, 7, 3, 41, 8, 6, 10, 11)",
            [],
        )
        .unwrap();
        let stored: f32 = con
            .query_row(
                "SELECT relationship_score FROM companion_attitudes",
                [],
                |row| row.get(0),
            )
            .unwrap();

        let mut attitude = attitude_with(0.0);
        attitude.attraction = 12.0;
        attitude.trust = 34.0;
        attitude.fear = 5.0;
        attitude.anger = 6.0;
        attitude.joy = 22.0;
        attitude.sorrow = 3.0;
        attitude.disgust = 1.0;
        attitude.surprise = 2.0;
        attitude.curiosity = 4.0;
        attitude.respect = 18.0;
        attitude.suspicion = 9.0;
        attitude.gratitude = 15.0;
        attitude.jealousy = 2.0;
        attitude.empathy = 7.0;
        attitude.lust = 3.0;
        attitude.love = 41.0;
        attitude.anxiety = 8.0;
        attitude.butterflies = 6.0;
        attitude.submissiveness = 10.0;
        attitude.dominance = 11.0;

        assert!((relationship_score_of(&attitude) - stored).abs() < f32::EPSILON);
    }

    #[test]
    fn recalibrator_applies_deltas_and_records_exactly_one_memory() {
        let sink = RecordingSink::new(Some(attitude_with(20.0)));
        let recalibrator = AttitudeRecalibrator::new(sink, 1, 1.0);
        let checkpoint = a_checkpoint(Some(serde_json::to_string(&ratings(90)).unwrap()));

        recalibrator
            .on_committed(&checkpoint, &[], &[])
            .expect("on_committed never fails");

        assert_eq!(recalibrator.sink.applied.lock().unwrap().len(), 1);
        let remembered = recalibrator.sink.remembered.lock().unwrap();
        assert_eq!(remembered.len(), 1);
        assert_eq!(remembered[0].memory_type, "NarrativeRecalibration");
        assert!(remembered[0]
            .description
            .contains(&checkpoint.id.to_string()));
    }

    #[test]
    fn recalibrator_applies_nothing_when_attitude_ratings_is_absent() {
        let sink = RecordingSink::new(Some(attitude_with(20.0)));
        let recalibrator = AttitudeRecalibrator::new(sink, 1, 1.0);
        let checkpoint = a_checkpoint(None);

        recalibrator.on_committed(&checkpoint, &[], &[]).unwrap();

        assert!(recalibrator.sink.applied.lock().unwrap().is_empty());
        assert!(recalibrator.sink.remembered.lock().unwrap().is_empty());
    }

    #[test]
    fn recalibrator_applies_nothing_when_no_attitude_row_exists_yet() {
        let sink = RecordingSink::new(None);
        let recalibrator = AttitudeRecalibrator::new(sink, 1, 1.0);
        let checkpoint = a_checkpoint(Some(serde_json::to_string(&ratings(90)).unwrap()));

        recalibrator.on_committed(&checkpoint, &[], &[]).unwrap();

        assert!(recalibrator.sink.applied.lock().unwrap().is_empty());
        assert!(recalibrator.sink.remembered.lock().unwrap().is_empty());
    }

    #[test]
    fn recalibrator_records_no_memory_when_the_sink_errors_on_apply() {
        let sink = RecordingSink::failing_on_apply(Some(attitude_with(20.0)));
        let recalibrator = AttitudeRecalibrator::new(sink, 1, 1.0);
        let checkpoint = a_checkpoint(Some(serde_json::to_string(&ratings(90)).unwrap()));

        let result = recalibrator.on_committed(&checkpoint, &[], &[]);

        assert!(result.is_ok());
        assert!(recalibrator.sink.remembered.lock().unwrap().is_empty());
    }

    #[test]
    fn recalibrator_applies_nothing_for_unparseable_attitude_ratings() {
        let sink = RecordingSink::new(Some(attitude_with(20.0)));
        let recalibrator = AttitudeRecalibrator::new(sink, 1, 1.0);
        let checkpoint = a_checkpoint(Some("not json".to_string()));

        recalibrator.on_committed(&checkpoint, &[], &[]).unwrap();

        assert!(recalibrator.sink.applied.lock().unwrap().is_empty());
    }
}
