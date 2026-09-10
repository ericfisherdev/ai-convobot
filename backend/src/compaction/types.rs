//! Enums and domain structs for conversation compaction (#171), kept out of
//! `database.rs` (already ~6000 lines) and out of `store.rs` so the pure
//! modules later issues add (`trigger`, `range`, `extract`, `validate`,
//! `commit`, `render`) can `use crate::compaction::types::*` without
//! pulling in rusqlite-backed code.
//!
//! Every plain enum here stores and serializes as the same lowercase
//! snake_case token, `FromSql`/`ToSql` written exactly like
//! `multiplayer::config::MultiplayerMode` (`ValueRef::Text` -> `from_utf8`
//! -> parse; unknown text -> `FromSqlError::OutOfRange(0)`; non-text ->
//! `FromSqlError::InvalidType`), so a SQLite column and a JSON field never
//! disagree about spelling. [`FactSubject`] carries data on one variant and
//! is written out by hand for the same reason.
//!
//! Nothing outside this module and `store.rs` constructs these types yet:
//! #172 (trigger), #173 (extraction/validation), #174 (rendering), #175
//! (commit), #179 (routes) and #180 (review UI) wire them in progressively.
//! Every shape is already exercised by the unit tests below, which is what
//! the #171 acceptance criteria ask for.
#![allow(dead_code)]

use rusqlite::types::{FromSql, FromSqlError, ToSqlOutput, ValueRef};
use rusqlite::ToSql;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// A checkpoint's lifecycle. `Draft` is pending review; `Committed` facts
/// and summary are live and rendered into the prompt; `Discarded` is a
/// draft the user rejected (or one `extract.rs`'s `fill_draft` rejected on
/// its own — an empty range, unparseable model output twice in a row, or an
/// over-budget overlay: all extraction ran to completion and produced a
/// definitive negative outcome); `Stale` is a committed checkpoint
/// superseded by a re-run. `Failed` (#208) is different from all of those:
/// the extraction pipeline itself never finished evaluating the draft's
/// content — a model/store I/O error (load failure, OOM, a truncated
/// completion, a lost database write) — so nothing about the draft's
/// content was ever judged. Kept distinct from `Discarded` so the API/UI
/// can tell "extraction ran and rejected everything" apart from "extraction
/// crashed", and so a "Retry" affordance stays meaningful (retrying an
/// infrastructure fault makes sense; retrying a fully-evaluated-and-
/// rejected draft would just reproduce the same rejection).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionStatus {
    Draft,
    Committed,
    Discarded,
    Stale,
    Failed,
}

impl fmt::Display for CompactionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CompactionStatus::Draft => "draft",
            CompactionStatus::Committed => "committed",
            CompactionStatus::Discarded => "discarded",
            CompactionStatus::Stale => "stale",
            CompactionStatus::Failed => "failed",
        };
        write!(f, "{s}")
    }
}

impl FromStr for CompactionStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "draft" => Ok(CompactionStatus::Draft),
            "committed" => Ok(CompactionStatus::Committed),
            "discarded" => Ok(CompactionStatus::Discarded),
            "stale" => Ok(CompactionStatus::Stale),
            "failed" => Ok(CompactionStatus::Failed),
            _ => Err(s.to_string()),
        }
    }
}

impl FromSql for CompactionStatus {
    fn column_result(value: ValueRef<'_>) -> Result<Self, FromSqlError> {
        match value {
            ValueRef::Text(i) => match std::str::from_utf8(i) {
                Ok(s) => s.parse().map_err(|_| FromSqlError::OutOfRange(0)),
                Err(e) => Err(FromSqlError::Other(Box::new(e))),
            },
            _ => Err(FromSqlError::InvalidType),
        }
    }
}

impl ToSql for CompactionStatus {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.to_string()))
    }
}

/// What caused a checkpoint's draft to be created. `JoinerSync` (#186) is
/// the one variant never queued by `hook.rs`/`main.rs::compaction_draft`:
/// it marks a draft `multiplayer::joiner_compaction`'s own auto-extraction
/// job queued locally on a joiner, following the host's `ContinuityPayload`
/// forward rather than a local threshold/scene-break/manual trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionTrigger {
    Threshold,
    SceneBreak,
    Manual,
    JoinerSync,
}

impl fmt::Display for CompactionTrigger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CompactionTrigger::Threshold => "threshold",
            CompactionTrigger::SceneBreak => "scene_break",
            CompactionTrigger::Manual => "manual",
            CompactionTrigger::JoinerSync => "joiner_sync",
        };
        write!(f, "{s}")
    }
}

impl FromStr for CompactionTrigger {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "threshold" => Ok(CompactionTrigger::Threshold),
            "scene_break" => Ok(CompactionTrigger::SceneBreak),
            "manual" => Ok(CompactionTrigger::Manual),
            "joiner_sync" => Ok(CompactionTrigger::JoinerSync),
            _ => Err(s.to_string()),
        }
    }
}

impl FromSql for CompactionTrigger {
    fn column_result(value: ValueRef<'_>) -> Result<Self, FromSqlError> {
        match value {
            ValueRef::Text(i) => match std::str::from_utf8(i) {
                Ok(s) => s.parse().map_err(|_| FromSqlError::OutOfRange(0)),
                Err(e) => Err(FromSqlError::Other(Box::new(e))),
            },
            _ => Err(FromSqlError::InvalidType),
        }
    }
}

impl ToSql for CompactionTrigger {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.to_string()))
    }
}

/// What kind of fact one `compaction_facts` row records. `Person` rows are
/// the only ones that ever carry `relation_to`/`relation`; `CompanionState`
/// and `UserState` rows are the only ones a model's extraction can point
/// `replaces` at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactCategory {
    CompanionState,
    UserState,
    Milestone,
    Backstory,
    OpenThread,
    Rule,
    KeyQuote,
    Person,
}

impl fmt::Display for FactCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FactCategory::CompanionState => "companion_state",
            FactCategory::UserState => "user_state",
            FactCategory::Milestone => "milestone",
            FactCategory::Backstory => "backstory",
            FactCategory::OpenThread => "open_thread",
            FactCategory::Rule => "rule",
            FactCategory::KeyQuote => "key_quote",
            FactCategory::Person => "person",
        };
        write!(f, "{s}")
    }
}

impl FromStr for FactCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "companion_state" => Ok(FactCategory::CompanionState),
            "user_state" => Ok(FactCategory::UserState),
            "milestone" => Ok(FactCategory::Milestone),
            "backstory" => Ok(FactCategory::Backstory),
            "open_thread" => Ok(FactCategory::OpenThread),
            "rule" => Ok(FactCategory::Rule),
            "key_quote" => Ok(FactCategory::KeyQuote),
            "person" => Ok(FactCategory::Person),
            _ => Err(s.to_string()),
        }
    }
}

impl FromSql for FactCategory {
    fn column_result(value: ValueRef<'_>) -> Result<Self, FromSqlError> {
        match value {
            ValueRef::Text(i) => match std::str::from_utf8(i) {
                Ok(s) => s.parse().map_err(|_| FromSqlError::OutOfRange(0)),
                Err(e) => Err(FromSqlError::Other(Box::new(e))),
            },
            _ => Err(FromSqlError::InvalidType),
        }
    }
}

impl ToSql for FactCategory {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.to_string()))
    }
}

/// Who or what a fact (or a `Person` fact's `relation_to`) is about.
///
/// `Person(String)` stores and serializes as `person:<name>` rather than a
/// bare name: a deliberate deviation from the design doc (which allows a
/// bare person name) so a person literally named "user" can never be
/// confused with [`FactSubject::User`]. A stored value that is neither
/// `user`, `companion`, nor `person:`-prefixed is an error, never a silent
/// fallback. Not `Copy`, unlike the plain enums above, since it owns a
/// `String`.
///
/// `Serialize`/`Deserialize` are hand-written through `Display`/`FromStr`
/// rather than derived: a derive would encode `Person("Ann")` externally
/// tagged as `{"person":"Ann"}`, disagreeing with the `person:Ann` every
/// other representation (`Display`, `FromStr`, `ToSql`, `FromSql`) uses.
/// Going through the same string form everywhere means JSON and SQLite can
/// never disagree about a `FactSubject` value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactSubject {
    User,
    Companion,
    Person(String),
}

impl Serialize for FactSubject {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for FactSubject {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Prefix a stored/wire `FactSubject::Person` value carries before the
/// name, e.g. `person:Ann`.
const PERSON_PREFIX: &str = "person:";

impl fmt::Display for FactSubject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FactSubject::User => write!(f, "user"),
            FactSubject::Companion => write!(f, "companion"),
            FactSubject::Person(name) => write!(f, "{PERSON_PREFIX}{name}"),
        }
    }
}

impl FromStr for FactSubject {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "user" => Ok(FactSubject::User),
            "companion" => Ok(FactSubject::Companion),
            _ => match s.strip_prefix(PERSON_PREFIX) {
                Some(name) => Ok(FactSubject::Person(name.to_string())),
                None => Err(s.to_string()),
            },
        }
    }
}

impl FromSql for FactSubject {
    fn column_result(value: ValueRef<'_>) -> Result<Self, FromSqlError> {
        match value {
            ValueRef::Text(i) => match std::str::from_utf8(i) {
                Ok(s) => s.parse().map_err(|_| FromSqlError::OutOfRange(0)),
                Err(e) => Err(FromSqlError::Other(Box::new(e))),
            },
            _ => Err(FromSqlError::InvalidType),
        }
    }
}

impl ToSql for FactSubject {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.to_string()))
    }
}

/// One `compactions` row: a checkpoint spanning `[from_message_id,
/// through_message_id]`.
///
/// Message ids (`from_message_id`, `through_message_id`) are `i32` to
/// match `Message::id`; `id` is the checkpoint's own `i64` row id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: i64,
    pub companion_id: i32,
    pub from_message_id: i32,
    pub through_message_id: i32,
    pub status: CompactionStatus,
    pub trigger: CompactionTrigger,
    pub raw_model_output: Option<String>,
    pub summary: Option<String>,
    pub rolling_summary: Option<String>,
    /// Raw JSON; #176 gives this a typed shape.
    pub attitude_ratings: Option<String>,
    /// Set when the summary-merge model was unavailable at commit time
    /// (#175).
    pub needs_merge: bool,
    pub created_at: String,
    pub committed_at: Option<String>,
    /// Why extraction failed, set only when `status == Failed` (#208).
    /// `Display`-formatted text from the `DraftError`/store error that
    /// `extract::fail_pending_draft` recorded, surfaced verbatim through
    /// `GET /api/compaction/{id}` so the UI can explain itself.
    pub extraction_error: Option<String>,
}

/// What [`crate::compaction::store::CompactionStore::insert_draft`] takes:
/// status is always [`CompactionStatus::Draft`] and `created_at` is always
/// `get_current_date()`, so neither is part of this shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewDraft {
    pub companion_id: i32,
    pub from_message_id: i32,
    pub through_message_id: i32,
    pub trigger: CompactionTrigger,
    pub raw_model_output: Option<String>,
}

/// One `compaction_facts` row.
///
/// `sources` (message ids, `i32` to match `Message::id`) and `replaces`
/// (prior fact ids, `i64` to match [`Checkpoint::id`]/this row's own `id`)
/// are each stored as a JSON array via `serde_json`, mapping a malformed
/// stored array to `rusqlite::Error::FromSqlConversionFailure` rather than
/// an empty vec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub id: i64,
    pub compaction_id: i64,
    pub category: FactCategory,
    pub subject: Option<FactSubject>,
    pub text: String,
    pub quote_speaker: Option<String>,
    pub sources: Vec<i32>,
    /// Ids of prior facts this item updates, as emitted by #173's
    /// extraction schema on `companion_state`/`user_state` items (always
    /// empty for every other category). This is the model's *claim*, kept
    /// verbatim on the row: #179's review carries it back into a
    /// [`FactDraft`] and #175's commit turns it into `superseded_by` via
    /// `supersede`. This store never acts on it by itself.
    pub replaces: Vec<i64>,
    /// Only set on `Person` facts, from #173's `people[]` items: the
    /// [`FactSubject`] (`User`/`Companion`) this person relates to. `None`
    /// for every other category.
    pub relation_to: Option<FactSubject>,
    /// Only set on `Person` facts: the free-text relationship, e.g.
    /// `"sister"`. `None` for every other category.
    pub relation: Option<String>,
    pub canon: bool,
    pub active: bool,
    pub superseded_by: Option<i64>,
    pub rejected_reason: Option<String>,
}

/// The insert shape for one fact: no `id`, `compaction_id` (the caller
/// passes it separately to
/// [`crate::compaction::store::CompactionStore::insert_facts`]), no
/// `active` (derived as `rejected_reason.is_none()` at insert time, so a
/// rejected item is always stored inactive), no `superseded_by` (only
/// `supersede` ever sets it). `replaces`, `relation_to` and `relation` are
/// stored exactly as given, with no validation here — #173's validator is
/// what filters `replaces`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactDraft {
    pub category: FactCategory,
    pub subject: Option<FactSubject>,
    pub text: String,
    pub quote_speaker: Option<String>,
    pub sources: Vec<i32>,
    pub replaces: Vec<i64>,
    pub relation_to: Option<FactSubject>,
    pub relation: Option<String>,
    pub canon: bool,
    pub rejected_reason: Option<String>,
}

/// One `pinned_messages` row: a message that stays in the prompt
/// regardless of what a checkpoint compacts over it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pin {
    pub message_id: i32,
    pub pinned_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trips one enum value through `ToSql`/`FromSql` and asserts it
    /// comes back unchanged. `ToSql::to_sql` returns a borrowed
    /// `ToSqlOutput`; `column_result` wants a `ValueRef`, so this goes
    /// through the `Text`/`Blob` shape `ToSqlOutput::from(String)`
    /// actually produces.
    fn assert_sql_round_trips<T>(value: T)
    where
        T: ToSql + FromSql + PartialEq + std::fmt::Debug,
    {
        let sql = value.to_sql().expect("to_sql should succeed");
        let owned = match sql {
            ToSqlOutput::Borrowed(v) => v.into(),
            ToSqlOutput::Owned(v) => v,
            _ => panic!("unexpected ToSqlOutput variant"),
        };
        let value_ref = ValueRef::from(&owned);
        let round_tripped = T::column_result(value_ref).expect("column_result should succeed");
        assert_eq!(value, round_tripped);
    }

    fn assert_json_round_trips<T>(value: T)
    where
        T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
    {
        let json = serde_json::to_string(&value).expect("serialize should succeed");
        let round_tripped: T = serde_json::from_str(&json).expect("deserialize should succeed");
        assert_eq!(value, round_tripped);
    }

    #[test]
    fn compaction_status_variants_round_trip_through_sql_and_json() {
        for variant in [
            CompactionStatus::Draft,
            CompactionStatus::Committed,
            CompactionStatus::Discarded,
            CompactionStatus::Stale,
            CompactionStatus::Failed,
        ] {
            assert_sql_round_trips(variant);
            assert_json_round_trips(variant);
        }
    }

    #[test]
    fn compaction_trigger_variants_round_trip_through_sql_and_json() {
        for variant in [
            CompactionTrigger::Threshold,
            CompactionTrigger::SceneBreak,
            CompactionTrigger::Manual,
            CompactionTrigger::JoinerSync,
        ] {
            assert_sql_round_trips(variant);
            assert_json_round_trips(variant);
        }
    }

    #[test]
    fn fact_category_variants_round_trip_through_sql_and_json() {
        for variant in [
            FactCategory::CompanionState,
            FactCategory::UserState,
            FactCategory::Milestone,
            FactCategory::Backstory,
            FactCategory::OpenThread,
            FactCategory::Rule,
            FactCategory::KeyQuote,
            FactCategory::Person,
        ] {
            assert_sql_round_trips(variant);
            assert_json_round_trips(variant);
        }
    }

    #[test]
    fn fact_subject_variants_round_trip_through_sql_and_json() {
        for variant in [
            FactSubject::User,
            FactSubject::Companion,
            FactSubject::Person("Ann".to_string()),
        ] {
            assert_sql_round_trips(variant.clone());
            assert_json_round_trips(variant);
        }
    }

    #[test]
    fn fact_subject_person_stores_with_the_person_prefix() {
        let subject = FactSubject::Person("Ann".to_string());
        assert_eq!(subject.to_string(), "person:Ann");
        assert_eq!(
            "person:Ann".parse::<FactSubject>().unwrap(),
            FactSubject::Person("Ann".to_string())
        );
    }

    #[test]
    fn fact_subject_serializes_as_its_string_form_not_an_externally_tagged_object() {
        // Guards against a derived `Serialize`/`Deserialize`, which would
        // encode `Person` as `{"person":"Ann"}` instead of the `person:Ann`
        // string every other representation uses.
        let json = serde_json::to_string(&FactSubject::Person("Ann".to_string())).unwrap();
        assert_eq!(json, "\"person:Ann\"");
        assert_eq!(json, serde_json::to_string("person:Ann").unwrap());

        assert_eq!(
            serde_json::to_string(&FactSubject::User).unwrap(),
            "\"user\""
        );
        assert_eq!(
            serde_json::to_string(&FactSubject::Companion).unwrap(),
            "\"companion\""
        );
    }

    #[test]
    fn unknown_stored_tokens_are_errors_not_defaults() {
        assert!("pending".parse::<CompactionStatus>().is_err());
        assert!("scenebreak".parse::<CompactionTrigger>().is_err());
        // Missing the `person:` prefix must not be mistaken for a keyword.
        assert!("person".parse::<FactSubject>().is_err());
        assert!("ann".parse::<FactSubject>().is_err());

        let err = CompactionStatus::column_result(ValueRef::Text(b"pending"))
            .expect_err("unexpected stored token must be an error");
        assert!(matches!(err, FromSqlError::OutOfRange(0)));

        let err = CompactionStatus::column_result(ValueRef::Integer(1))
            .expect_err("non-text column must be an error");
        assert!(matches!(err, FromSqlError::InvalidType));
    }

    #[test]
    fn fact_replaces_round_trips_through_json_empty_and_non_empty() {
        let base = Fact {
            id: 1,
            compaction_id: 1,
            category: FactCategory::CompanionState,
            subject: Some(FactSubject::Companion),
            text: "text".to_string(),
            quote_speaker: None,
            sources: vec![1, 2],
            replaces: vec![3, 4],
            relation_to: None,
            relation: None,
            canon: true,
            active: true,
            superseded_by: None,
            rejected_reason: None,
        };
        assert_json_round_trips(base.clone());

        let empty = Fact {
            replaces: vec![],
            ..base
        };
        assert_json_round_trips(empty);
    }
}
