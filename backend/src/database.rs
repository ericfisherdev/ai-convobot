use chrono::{DateTime, Local};
use rusqlite::types::{FromSql, FromSqlError, ToSqlOutput, ValueRef};
use rusqlite::{
    params, Connection, Error, OptionalExtension, Result, ToSql, Transaction, TransactionBehavior,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::character_card::CharacterCard;
use crate::multiplayer::config::{MultiplayerConfig, MultiplayerMode};

/// Reserved speaker id for the human participant. #126's `ParticipantId`
/// reserved IDs must reuse this constant, not redefine it.
pub const USER_SPEAKER_ID: &str = "user";
/// Reserved speaker id for the (single, solo-mode) companion. #126's
/// `ParticipantId` reserved IDs must reuse this constant, not redefine it.
pub const CHAR_SPEAKER_ID: &str = "char";
/// Speaker id for a system-generated notice (e.g. a remote speaker that did
/// not respond in time). Not a chat participant, so it never appears in a
/// `ParticipantRegistry`; #131's `ParticipantId::SYSTEM` reuses this
/// constant, not redefine it.
pub const SYSTEM_SPEAKER_ID: &str = "system";

/// Derives the legacy `ai` flag from a `speaker_id`. The single source of
/// truth for that derivation, used by both the row mapper and every insert,
/// so the two can never disagree.
///
/// Note for #134/#135: under this rule a future `system` speaker reads as
/// `ai: true`; if that is wrong for them, the derivation lives in exactly
/// this one function.
pub fn is_ai_speaker(speaker_id: &str) -> bool {
    speaker_id != USER_SPEAKER_ID
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Message {
    pub id: i32,
    pub ai: bool,
    pub speaker_id: String,
    pub content: String,
    pub created_at: String,
}

/// Column list shared by every query that reads a full message row, kept
/// alongside `message_from_row` so the two can never drift apart.
const MESSAGE_COLUMNS: &str = "id, speaker_id, content, created_at";

/// Maps a `SELECT {MESSAGE_COLUMNS} FROM messages ...` row to a `Message`.
/// Shared by every query that reads that exact column list. `ai` is always
/// derived from `speaker_id`, never read from its own column.
fn message_from_row(row: &rusqlite::Row) -> Result<Message> {
    let speaker_id: String = row.get(1)?;
    Ok(Message {
        id: row.get(0)?,
        ai: is_ai_speaker(&speaker_id),
        speaker_id,
        content: row.get(2)?,
        created_at: row.get(3)?,
    })
}

/// Outcome of `Database::pop_latest_bot_reply`.
#[derive(Debug, PartialEq)]
pub enum PoppedReply {
    /// The trailing bot reply was deleted; `speaker_id` is who said it,
    /// `message_id` is the deleted row's id (for
    /// `regenerate_prompt`'s `ServerFrame::MessageRemoved` broadcast, #135
    /// part 2), and `user_turn` is the newest user turn before it.
    /// `user_turn` is not necessarily the reply's immediate predecessor: a
    /// mention follow-up (#131/#132) can put another bot's reply — or this
    /// bot's own earlier one — in between.
    Removed {
        speaker_id: String,
        message_id: i32,
        user_turn: Message,
    },
    /// Nothing was deleted: the conversation is empty, its newest row is not
    /// a bot reply (a user message or a system notice), or there is no user
    /// turn anywhere before it to regenerate from.
    NothingToRegenerate,
    /// The trailing reply's owner (`speaker_id`) is not connected, so
    /// nothing was deleted. Checked inside the same transaction as the
    /// delete (`pop_latest_bot_reply_on`'s `owner_ready` predicate), so a
    /// direct `POST /api/message` cannot land between "checked offline" and
    /// "deleted" and make this report stale.
    OwnerUnavailable { speaker_id: String },
}

pub fn get_current_date() -> String {
    let local: DateTime<Local> = Local::now();
    local.format("%A %d.%m.%Y %H:%M").to_string()
}

pub fn contains_time_question(text: &str) -> bool {
    let time_related_keywords = [
        "time",
        "date",
        "hour",
        "day",
        "month",
        "year",
        "minute",
        "second",
        "morning",
        "afternoon",
        "evening",
        "night",
    ];
    for keyword in &time_related_keywords {
        if text.contains(keyword) {
            return true;
        }
    }
    false
}

/// The domain type every internal caller inserts. `speaker_id` is the only
/// source of truth; `ai` is never carried here (it is derived at read time
/// by `message_from_row`/`is_ai_speaker`).
#[derive(Serialize, Deserialize)]
pub struct NewMessage {
    pub speaker_id: String,
    pub content: String,
}

impl NewMessage {
    pub fn new(speaker_id: impl Into<String>, content: impl Into<String>) -> Self {
        NewMessage {
            speaker_id: speaker_id.into(),
            content: content.into(),
        }
    }

    pub fn from_user(content: impl Into<String>) -> Self {
        NewMessage::new(USER_SPEAKER_ID, content)
    }
}

/// Error returned by `resolve_speaker` when a `POST /api/message` body's
/// `ai`/`speaker_id` fields cannot be resolved to a single speaker.
#[derive(Debug, PartialEq, Eq)]
pub enum SpeakerResolveError {
    /// Both `ai` and `speaker_id` were given, and they disagree.
    Conflict,
    /// Neither field was given, or `speaker_id` was empty.
    Missing,
}

impl std::fmt::Display for SpeakerResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpeakerResolveError::Conflict => {
                write!(f, "`ai` and `speaker_id` disagree on who sent this message")
            }
            SpeakerResolveError::Missing => {
                write!(f, "either `speaker_id` or `ai` must be given")
            }
        }
    }
}

/// Resolves the legacy `ai` flag and the new `speaker_id` field into a
/// single speaker id.
///
/// - `speaker_id` present and non-empty wins; if `ai` is also present it
///   must agree with `is_ai_speaker`, otherwise `Err(Conflict)`.
/// - only `ai` present: `true` becomes `CHAR_SPEAKER_ID`, `false` becomes
///   `USER_SPEAKER_ID`.
/// - neither, or an empty `speaker_id`: `Err(Missing)`.
fn resolve_speaker(
    ai: Option<bool>,
    speaker_id: Option<String>,
) -> std::result::Result<String, SpeakerResolveError> {
    match (ai, speaker_id) {
        (_, Some(speaker_id)) if speaker_id.is_empty() => Err(SpeakerResolveError::Missing),
        (ai, Some(speaker_id)) => {
            if let Some(ai) = ai {
                if ai != is_ai_speaker(&speaker_id) {
                    return Err(SpeakerResolveError::Conflict);
                }
            }
            Ok(speaker_id)
        }
        (Some(ai), None) => Ok(if ai {
            CHAR_SPEAKER_ID.to_string()
        } else {
            USER_SPEAKER_ID.to_string()
        }),
        (None, None) => Err(SpeakerResolveError::Missing),
    }
}

/// Body accepted by `POST /api/message`. Kept separate from `NewMessage` so
/// the `ai`/`speaker_id` compatibility shim does not leak into internal
/// callers, which always build a `NewMessage` directly.
#[derive(Deserialize)]
pub struct NewMessageRequest {
    #[serde(default)]
    pub ai: Option<bool>,
    #[serde(default)]
    pub speaker_id: Option<String>,
    pub content: String,
}

impl TryFrom<NewMessageRequest> for NewMessage {
    type Error = SpeakerResolveError;

    fn try_from(value: NewMessageRequest) -> std::result::Result<Self, Self::Error> {
        let speaker_id = resolve_speaker(value.ai, value.speaker_id)?;
        Ok(NewMessage::new(speaker_id, value.content))
    }
}

/// Body accepted by `PUT /api/message/{id}`. Deliberately carries no role
/// flag: an edit changes text only, never who a message is attributed to.
/// `serde` ignores unknown fields by default, so a client still sending the
/// old `ai` field (e.g. one built from `docs/api_docs.md` before this fix)
/// keeps working; the flag is silently dropped instead of rejected.
#[derive(Serialize, Deserialize)]
pub struct MessageEdit {
    pub content: String,
}

#[derive(Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Companion {
    pub id: i32,
    pub name: String,
    pub persona: String,
    pub example_dialogue: String,
    pub first_message: String,
    pub long_term_mem: usize,
    pub short_term_mem: usize,
    pub roleplay: bool,
    pub dialogue_tuning: bool,
    pub avatar_path: String,
}

#[derive(Serialize, Deserialize)]
pub struct CompanionView {
    pub name: String,
    pub persona: String,
    pub example_dialogue: String,
    pub first_message: String,
    pub long_term_mem: usize,
    pub short_term_mem: usize,
    pub roleplay: bool,
    pub dialogue_tuning: bool,
    pub avatar_path: String,
}

#[derive(Serialize, Deserialize)]
#[allow(dead_code)]
pub struct User {
    pub id: i32,
    pub name: String,
    pub persona: String,
}

#[derive(Serialize, Deserialize)]
pub struct UserView {
    pub name: String,
    pub persona: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CompanionAttitude {
    pub id: Option<i32>,
    pub companion_id: i32,
    pub target_id: i32,
    pub target_type: String,
    pub attraction: f32,
    pub trust: f32,
    pub fear: f32,
    pub anger: f32,
    pub joy: f32,
    pub sorrow: f32,
    pub disgust: f32,
    pub surprise: f32,
    pub curiosity: f32,
    pub respect: f32,
    pub suspicion: f32,
    pub gratitude: f32,
    pub jealousy: f32,
    pub empathy: f32,
    pub lust: f32,
    pub love: f32,
    pub anxiety: f32,
    pub butterflies: f32,
    pub submissiveness: f32,
    pub dominance: f32,
    pub relationship_score: Option<f32>,
    pub last_updated: String,
    pub created_at: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThirdPartyIndividual {
    pub id: Option<i32>,
    pub name: String,
    pub relationship_to_user: Option<String>,
    pub relationship_to_companion: Option<String>,
    pub occupation: Option<String>,
    pub personality_traits: Option<String>,
    pub physical_description: Option<String>,
    pub first_mentioned: String,
    pub last_mentioned: Option<String>,
    pub mention_count: i32,
    pub importance_score: f32,
    pub created_at: String,
    pub updated_at: String,
    /// Whether the heuristic detector or compaction's `PersonsObserver`
    /// (#177) created/last trusted this row.
    pub source: PersonSource,
}

/// What [`Database::upsert_compaction_person`] takes: one canon-validated
/// `Person` fact (or several merged by `compaction::persons::plan_upserts`),
/// ready to become or update a `third_party_individuals` row with
/// `source = 'compaction'`.
#[derive(Debug, Clone, PartialEq)]
pub struct PersonUpsert {
    pub name: String,
    pub relationship_to_user: Option<String>,
    pub relationship_to_companion: Option<String>,
    /// How many message sources this upsert accounts for; added onto the
    /// row's existing `mention_count` on an update, or used as the initial
    /// `mention_count` on insert.
    pub mentions: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThirdPartyMemory {
    pub id: Option<i32>,
    pub third_party_id: i32,
    pub companion_id: i32,
    pub memory_type: String,
    pub content: String,
    pub importance: f32,
    pub emotional_valence: f32,
    pub created_at: String,
    pub context_message_id: Option<i32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThirdPartyInteraction {
    pub id: Option<i32>,
    pub third_party_id: i32,
    pub companion_id: i32,
    pub interaction_type: String,
    pub description: String,
    pub planned_date: Option<String>,
    pub actual_date: Option<String>,
    pub outcome: Option<String>,
    pub impact_on_relationship: f32,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub enum Device {
    CPU,
    GPU,
    Metal,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CPU => write!(f, "CPU"),
            Device::GPU => write!(f, "GPU"),
            Device::Metal => write!(f, "Metal"),
        }
    }
}

impl FromSql for Device {
    fn column_result(value: ValueRef<'_>) -> Result<Self, FromSqlError> {
        match value {
            ValueRef::Text(i) => match std::str::from_utf8(i) {
                Ok(s) => match s {
                    "CPU" => Ok(Device::CPU),
                    "GPU" => Ok(Device::GPU),
                    "Metal" => Ok(Device::Metal),
                    _ => Err(FromSqlError::OutOfRange(0)),
                },
                Err(e) => Err(FromSqlError::Other(Box::new(e))),
            },
            _ => Err(FromSqlError::InvalidType),
        }
    }
}

impl ToSql for Device {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        match self {
            Device::CPU => Ok(ToSqlOutput::from("CPU")),
            Device::GPU => Ok(ToSqlOutput::from("GPU")),
            Device::Metal => Ok(ToSqlOutput::from("Metal")),
        }
    }
}

#[derive(PartialEq, Serialize, Deserialize, Clone)]
pub enum PromptTemplate {
    /// Render the prompt with the chat template embedded in the GGUF file.
    /// Falls back to `Default` when the model does not carry one.
    Auto,
    Default,
    Llama2,
    Mistral,
}

impl FromSql for PromptTemplate {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> Result<Self, FromSqlError> {
        match value {
            ValueRef::Text(i) => match std::str::from_utf8(i) {
                Ok(s) => match s {
                    "Auto" => Ok(PromptTemplate::Auto),
                    "Default" => Ok(PromptTemplate::Default),
                    "Llama2" => Ok(PromptTemplate::Llama2),
                    "Mistral" => Ok(PromptTemplate::Mistral),
                    _ => Err(FromSqlError::OutOfRange(0)),
                },
                Err(e) => Err(FromSqlError::Other(Box::new(e))),
            },
            _ => Err(FromSqlError::InvalidType),
        }
    }
}

impl ToSql for PromptTemplate {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        match self {
            PromptTemplate::Auto => Ok(ToSqlOutput::from("Auto")),
            PromptTemplate::Default => Ok(ToSqlOutput::from("Default")),
            PromptTemplate::Llama2 => Ok(ToSqlOutput::from("Llama2")),
            PromptTemplate::Mistral => Ok(ToSqlOutput::from("Mistral")),
        }
    }
}

/// Where a `third_party_individuals` row came from (#177): `Heuristic` is
/// the pre-compaction pronoun/capitalised-word detector
/// (`detect_new_persons_in_message`), `Compaction` is a canon-validated
/// `Person` fact promoted at commit time
/// (`compaction::persons::PersonsObserver`). Same `FromSql`/`ToSql` shape as
/// [`PromptTemplate`] above: an unknown stored value is an error, never a
/// silent default.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersonSource {
    Heuristic,
    Compaction,
}

impl FromSql for PersonSource {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> Result<Self, FromSqlError> {
        match value {
            ValueRef::Text(i) => match std::str::from_utf8(i) {
                Ok(s) => match s {
                    "heuristic" => Ok(PersonSource::Heuristic),
                    "compaction" => Ok(PersonSource::Compaction),
                    _ => Err(FromSqlError::OutOfRange(0)),
                },
                Err(e) => Err(FromSqlError::Other(Box::new(e))),
            },
            _ => Err(FromSqlError::InvalidType),
        }
    }
}

impl ToSql for PersonSource {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        match self {
            PersonSource::Heuristic => Ok(ToSqlOutput::from("heuristic")),
            PersonSource::Compaction => Ok(ToSqlOutput::from("compaction")),
        }
    }
}

/*
struct Config {
    id: i32,
    device: Device,
    llm_model_path: String,
    gpu_layers: usize,
    prompt_template: PromptTemplate
}
*/

#[derive(Serialize, Deserialize, Clone)]
pub struct ConfigView {
    pub device: Device,
    pub llm_model_path: String,
    pub gpu_layers: usize,
    pub prompt_template: PromptTemplate,
    pub context_window_size: usize,
    pub max_response_tokens: usize,
    pub enable_dynamic_context: bool,
    pub vram_limit_gb: usize,
    pub dynamic_gpu_allocation: bool,
    pub gpu_safety_margin: f32,
    pub min_free_vram_mb: u64,
    pub enable_hybrid_context: bool,
    pub max_system_ram_usage_gb: usize,
    pub context_expansion_strategy: String,
    pub ram_safety_margin_gb: usize,
    pub multiplayer_mode: MultiplayerMode,
    pub multiplayer_host_address: String,
    pub multiplayer_participant_id: String,
    pub mention_followup_depth: u8,
    pub remote_generation_timeout_secs: u64,
    /// Derived from `multiplayer_password`: whether a host password is
    /// currently stored. What `GET /api/config` sends instead of the
    /// password itself.
    pub multiplayer_password_set: bool,
    /// The shared HMAC secret #129/#130 use to authenticate host/joiner
    /// connections. Stored and kept in plaintext: unlike a login password,
    /// it must be recoverable so both sides of the connection can present
    /// it. `#[serde(skip)]` (not `skip_serializing`, so `ConfigView` still
    /// derives `Deserialize`) keeps it out of every JSON response;
    /// `multiplayer_password_set` is what callers see instead.
    ///
    /// Read by `multiplayer::host::SqliteHostConfig::host_password` (#129);
    /// #130 reads it too, for the joiner's own HMAC proof.
    #[serde(skip)]
    pub multiplayer_password: String,
    /// Token budget a companion's recent-message window must exceed before
    /// compaction (#172) triggers a draft. `None` means "derive at runtime
    /// from `TokenBudget::recent_messages`" rather than a fixed number.
    pub compact_threshold_tokens: Option<usize>,
    /// Fewest uncompacted messages compaction (#172) will ever fire on,
    /// regardless of token count.
    pub compact_min_messages: usize,
    /// Model used for compaction's summarisation/extraction passes. `None`
    /// means "use `llm_model_path`".
    pub compaction_model_path: Option<String>,
    /// Whether #173's extraction pass should also run the heuristic
    /// person-detection path already used elsewhere in the codebase.
    pub heuristic_person_detection: bool,
    /// How far a compaction commit's `AttitudeRecalibrator` (#176) blends
    /// the companion's running attitude toward the extraction model's
    /// narrative rating: `0.0` keeps the running values untouched, `1.0`
    /// adopts the rating outright. Clamped to `0.0..=1.0` on write.
    pub compaction_attitude_weight: f32,
}

#[derive(Serialize, Deserialize)]
pub struct ConfigModify {
    pub device: String,
    pub llm_model_path: String,
    pub gpu_layers: usize,
    pub prompt_template: String,
    pub context_window_size: usize,
    pub max_response_tokens: usize,
    pub enable_dynamic_context: bool,
    pub vram_limit_gb: usize,
    pub dynamic_gpu_allocation: bool,
    pub gpu_safety_margin: f32,
    pub min_free_vram_mb: u64,
    pub enable_hybrid_context: bool,
    pub max_system_ram_usage_gb: usize,
    pub context_expansion_strategy: String,
    pub ram_safety_margin_gb: usize,
    #[serde(default = "default_multiplayer_mode")]
    pub multiplayer_mode: String,
    #[serde(default)]
    pub multiplayer_host_address: String,
    #[serde(default)]
    pub multiplayer_participant_id: String,
    #[serde(default = "default_mention_followup_depth")]
    pub mention_followup_depth: u8,
    #[serde(default = "default_remote_generation_timeout_secs")]
    pub remote_generation_timeout_secs: u64,
    /// Write-only: `None` or `Some("")` leaves the stored password
    /// unchanged, so the frontend never has to resend it on every save.
    #[serde(default)]
    pub multiplayer_password: Option<String>,
    #[serde(default)]
    pub compact_threshold_tokens: Option<usize>,
    #[serde(default = "default_compact_min_messages")]
    pub compact_min_messages: usize,
    #[serde(default)]
    pub compaction_model_path: Option<String>,
    #[serde(default = "default_heuristic_person_detection")]
    pub heuristic_person_detection: bool,
    #[serde(default = "default_compaction_attitude_weight")]
    pub compaction_attitude_weight: f32,
}

fn default_multiplayer_mode() -> String {
    "solo".to_string()
}

fn default_mention_followup_depth() -> u8 {
    1
}

fn default_remote_generation_timeout_secs() -> u64 {
    120
}

fn default_compact_min_messages() -> usize {
    8
}

fn default_heuristic_person_detection() -> bool {
    false
}

fn default_compaction_attitude_weight() -> f32 {
    0.5
}

/// The one way `Database::write_config` (#128) can reject a `PUT
/// /api/config` request: `Invalid` names an already-user-facing message
/// (bad device/template/multiplayer value), surfaced by `main.rs::config_post`
/// as a 400. `Database` wraps any lower-level `rusqlite` failure and is
/// surfaced as the existing 500.
#[derive(Debug)]
pub enum ConfigChangeError {
    Invalid(String),
    Database(rusqlite::Error),
}

impl std::fmt::Display for ConfigChangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigChangeError::Invalid(msg) => write!(f, "{}", msg),
            ConfigChangeError::Database(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for ConfigChangeError {}

impl From<rusqlite::Error> for ConfigChangeError {
    fn from(e: rusqlite::Error) -> Self {
        ConfigChangeError::Database(e)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AttitudeMemory {
    pub id: Option<i32>,
    pub companion_id: i32,
    pub target_id: i32,
    pub target_type: String,
    pub memory_type: String,
    pub description: String,
    pub priority_score: f32,
    pub attitude_delta_json: String,
    pub impact_score: f32,
    pub message_context: String,
    pub created_at: String,
}

/// Per-dimension movement between two `CompanionAttitude` snapshots, persisted
/// as `attitude_memories.attitude_delta_json`.
///
/// Every field defaults, so rows written before a dimension existed still
/// deserialize (the six trailing dimensions were added after the first rows
/// could have been written).
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct AttitudeDelta {
    #[serde(default)]
    pub attraction: f32,
    #[serde(default)]
    pub trust: f32,
    #[serde(default)]
    pub fear: f32,
    #[serde(default)]
    pub anger: f32,
    #[serde(default)]
    pub joy: f32,
    #[serde(default)]
    pub sorrow: f32,
    #[serde(default)]
    pub disgust: f32,
    #[serde(default)]
    pub surprise: f32,
    #[serde(default)]
    pub curiosity: f32,
    #[serde(default)]
    pub respect: f32,
    #[serde(default)]
    pub suspicion: f32,
    #[serde(default)]
    pub gratitude: f32,
    #[serde(default)]
    pub jealousy: f32,
    #[serde(default)]
    pub empathy: f32,
    #[serde(default)]
    pub lust: f32,
    #[serde(default)]
    pub love: f32,
    #[serde(default)]
    pub anxiety: f32,
    #[serde(default)]
    pub butterflies: f32,
    #[serde(default)]
    pub submissiveness: f32,
    #[serde(default)]
    pub dominance: f32,
}

/// One attitude shift judged worth remembering, before it becomes a row.
#[derive(Debug, Clone)]
pub struct AttitudeMemoryDraft {
    pub memory_type: String,
    pub description: String,
    pub priority_score: f32,
    pub impact_score: f32,
    pub delta: AttitudeDelta,
}

fn calculate_attitude_delta(
    previous: &CompanionAttitude,
    new: &CompanionAttitude,
) -> AttitudeDelta {
    AttitudeDelta {
        attraction: new.attraction - previous.attraction,
        trust: new.trust - previous.trust,
        fear: new.fear - previous.fear,
        anger: new.anger - previous.anger,
        joy: new.joy - previous.joy,
        sorrow: new.sorrow - previous.sorrow,
        disgust: new.disgust - previous.disgust,
        surprise: new.surprise - previous.surprise,
        curiosity: new.curiosity - previous.curiosity,
        respect: new.respect - previous.respect,
        suspicion: new.suspicion - previous.suspicion,
        gratitude: new.gratitude - previous.gratitude,
        jealousy: new.jealousy - previous.jealousy,
        empathy: new.empathy - previous.empathy,
        lust: new.lust - previous.lust,
        love: new.love - previous.love,
        anxiety: new.anxiety - previous.anxiety,
        butterflies: new.butterflies - previous.butterflies,
        submissiveness: new.submissiveness - previous.submissiveness,
        dominance: new.dominance - previous.dominance,
    }
}

fn calculate_impact_score(delta: &AttitudeDelta) -> f32 {
    // Calculate weighted Euclidean distance in attitude space
    let dimensions = [
        ("attraction", delta.attraction, 1.2), // High weight for relationship-defining emotions
        ("trust", delta.trust, 1.5),
        ("fear", delta.fear, 1.1),
        ("anger", delta.anger, 1.3),
        ("joy", delta.joy, 1.0),
        ("sorrow", delta.sorrow, 1.0),
        ("disgust", delta.disgust, 1.1),
        ("surprise", delta.surprise, 0.8), // Lower weight for transient emotions
        ("curiosity", delta.curiosity, 0.9),
        ("respect", delta.respect, 1.4),
        ("suspicion", delta.suspicion, 1.2),
        ("gratitude", delta.gratitude, 1.1),
        ("jealousy", delta.jealousy, 1.3),
        ("empathy", delta.empathy, 1.2),
        ("lust", delta.lust, 1.1),
        ("love", delta.love, 1.5), // Relationship-defining, weighted like trust
        ("anxiety", delta.anxiety, 1.1),
        ("butterflies", delta.butterflies, 0.9),
        ("submissiveness", delta.submissiveness, 1.0),
        ("dominance", delta.dominance, 1.0),
    ];

    let mut weighted_sum = 0.0;
    for (_, value, weight) in dimensions.iter() {
        weighted_sum += (value * weight).powi(2);
    }

    weighted_sum.sqrt()
}

fn classify_memory_type(delta: &AttitudeDelta, impact_score: f32) -> String {
    // Classify based on dominant changes and impact
    if delta.trust > 15.0 && delta.attraction > 10.0 {
        "BondingMoment".to_string()
    } else if delta.trust < -20.0 || delta.anger > 20.0 {
        "Betrayal".to_string()
    } else if delta.attraction > 20.0 {
        "AttractionSpike".to_string()
    } else if delta.fear > 15.0 && delta.suspicion > 10.0 {
        "ThreatDetection".to_string()
    } else if delta.respect > 15.0 {
        "RespectGained".to_string()
    } else if delta.respect < -15.0 {
        "RespectLost".to_string()
    } else if delta.anger > 15.0 {
        "ConflictMoment".to_string()
    } else if delta.joy > 15.0 && delta.gratitude > 10.0 {
        "JoyfulMemory".to_string()
    } else if delta.sorrow > 15.0 {
        "SadMoment".to_string()
    } else if impact_score > 25.0 {
        "PowerShift".to_string()
    } else {
        "SignificantChange".to_string()
    }
}

fn calculate_priority_score(delta: &AttitudeDelta, impact_score: f32, memory_type: &str) -> f32 {
    let recency_weight = 0.25;
    let impact_weight = 0.4;
    let type_weight = 0.2;
    let relevance_weight = 0.15;

    // Base scores
    let recency_score = 100.0; // Recent changes get max recency
    let impact_normalized = (impact_score / 50.0).min(100.0); // Normalize to 0-100

    let type_score = match memory_type {
        "BondingMoment" | "Betrayal" => 95.0,
        "PowerShift" | "AttractionSpike" => 90.0,
        "NarrativeRecalibration" => 90.0,
        "ThreatDetection" | "ConflictMoment" => 85.0,
        "RespectGained" | "RespectLost" => 80.0,
        "JoyfulMemory" | "SadMoment" => 70.0,
        _ => 60.0,
    };

    // Relevance based on relationship-critical dimensions
    let critical_changes = delta.trust.abs() + delta.attraction.abs() + delta.respect.abs();
    let relevance_score = (critical_changes / 30.0 * 100.0).min(100.0);

    recency_score * recency_weight
        + impact_normalized * impact_weight
        + type_score * type_weight
        + relevance_score * relevance_weight
}

/// Impact score a turn has to clear before it is worth remembering.
pub const SIGNIFICANT_IMPACT_THRESHOLD: f32 = 10.0;

/// Largest number of `attitude_memories` rows kept per companion.
pub const MAX_ATTITUDE_MEMORIES_PER_COMPANION: usize = 100;

/// Judges whether one attitude shift is worth remembering.
///
/// Pure and database-free, so the threshold and the classification can be
/// tested without SQLite. Returns `None` when the movement is below
/// `SIGNIFICANT_IMPACT_THRESHOLD`.
pub fn evaluate_attitude_shift(
    previous: &CompanionAttitude,
    new: &CompanionAttitude,
) -> Option<AttitudeMemoryDraft> {
    let delta = calculate_attitude_delta(previous, new);
    let impact_score = calculate_impact_score(&delta);

    if impact_score <= SIGNIFICANT_IMPACT_THRESHOLD {
        return None;
    }

    let memory_type = classify_memory_type(&delta, impact_score);
    let priority_score = calculate_priority_score(&delta, impact_score, &memory_type);
    let description = generate_memory_description(&memory_type, &delta, impact_score);

    Some(AttitudeMemoryDraft {
        memory_type,
        description,
        priority_score,
        impact_score,
        delta,
    })
}

/// Builds the `attitude_memories` draft for one compaction commit's
/// narrative recalibration ([`crate::compaction::attitude::AttitudeRecalibrator`]),
/// with no [`SIGNIFICANT_IMPACT_THRESHOLD`] gate: a checkpoint the extraction
/// model rated is always worth remembering, however small the resulting
/// move. `"NarrativeRecalibration"` scores `90.0` in
/// `calculate_priority_score`'s `type_score` match, ahead of the lexicon
/// scorer's uncategorised `"SignificantChange"` default (`60.0`), so it
/// survives `prune_attitude_memories` alongside turn-scored memories of
/// similar impact.
pub fn recalibration_memory_draft(
    previous: &CompanionAttitude,
    new: &CompanionAttitude,
    checkpoint_id: i64,
) -> AttitudeMemoryDraft {
    let delta = calculate_attitude_delta(previous, new);
    let impact_score = calculate_impact_score(&delta);
    let memory_type = "NarrativeRecalibration".to_string();
    let priority_score = calculate_priority_score(&delta, impact_score, &memory_type);
    // No dates or clock times (compaction's "user turns canon" rule): what
    // matters is that the feelings moved to match the story, not when. No
    // `{{char}}`/`{{user}}` placeholder either, matching every other
    // `generate_memory_description` arm above: `format_attitude_memories`
    // renders `description` verbatim into the prompt, with nothing left to
    // substitute those tokens the way `insert_companion_greeting` does for
    // the opening message.
    let description =
        format!("Attitude recalibrated to match the story so far (checkpoint {checkpoint_id})");

    AttitudeMemoryDraft {
        memory_type,
        description,
        priority_score,
        impact_score,
        delta,
    }
}

fn generate_memory_description(
    memory_type: &str,
    delta: &AttitudeDelta,
    impact_score: f32,
) -> String {
    match memory_type {
        "BondingMoment" => format!("A bonding moment occurred (trust +{:.1}, attraction +{:.1}) with significant relationship impact", delta.trust, delta.attraction),
        "Betrayal" => format!("Trust was broken (trust {:.1}, anger +{:.1}) creating lasting negative impact", delta.trust, delta.anger),
        "AttractionSpike" => format!("Strong attraction developed (+{:.1}) indicating romantic/personal interest", delta.attraction),
        "ThreatDetection" => format!("Threat response triggered (fear +{:.1}, suspicion +{:.1}) affecting security perception", delta.fear, delta.suspicion),
        "PowerShift" => format!("Significant power dynamic change detected (impact score: {:.1})", impact_score),
        "ConflictMoment" => format!("Conflict arose (anger +{:.1}) potentially damaging relationship", delta.anger),
        "RespectGained" => format!("Respect significantly increased (+{:.1}) enhancing relationship status", delta.respect),
        "RespectLost" => format!("Respect was lost ({:.1}) diminishing relationship quality", delta.respect),
        "JoyfulMemory" => format!("Joyful experience shared (joy +{:.1}, gratitude +{:.1})", delta.joy, delta.gratitude),
        "SadMoment" => format!("Sadness experienced together (sorrow +{:.1}) creating emotional bond", delta.sorrow),
        _ => format!("Significant attitude change detected (impact: {:.1})", impact_score),
    }
}

type MessageCache = Arc<Mutex<HashMap<String, (Vec<Message>, Instant)>>>;

// Database query cache for performance optimization
lazy_static::lazy_static! {
    static ref MESSAGE_CACHE: MessageCache = Arc::new(Mutex::new(HashMap::new()));
}

/// Reads one `companion_attitudes` row on the given connection (or transaction,
/// via its `Deref<Target = Connection>`).
///
/// Uses `.optional()?` rather than `.ok()` so a real SQL failure (e.g.
/// `SQLITE_BUSY`) propagates as `Err` instead of being indistinguishable from
/// "no row exists" — callers (like `finish_turn`'s seed-on-missing-row path)
/// rely on `Ok(None)` meaning the row is actually absent.
fn read_attitude_row(
    con: &Connection,
    companion_id: i32,
    target_id: i32,
    target_type: &str,
) -> Result<Option<CompanionAttitude>> {
    let mut stmt = con.prepare(
        "SELECT id, companion_id, target_id, target_type, attraction, trust, fear, anger,
                joy, sorrow, disgust, surprise, curiosity, respect, suspicion,
                gratitude, jealousy, empathy, lust, love, anxiety, butterflies,
                submissiveness, dominance, relationship_score, last_updated, created_at
         FROM companion_attitudes
         WHERE companion_id = ? AND target_id = ? AND target_type = ?",
    )?;

    stmt.query_row(params![companion_id, target_id, target_type], |row| {
        Ok(CompanionAttitude {
            id: Some(row.get(0)?),
            companion_id: row.get(1)?,
            target_id: row.get(2)?,
            target_type: row.get(3)?,
            attraction: row.get(4)?,
            trust: row.get(5)?,
            fear: row.get(6)?,
            anger: row.get(7)?,
            joy: row.get(8)?,
            sorrow: row.get(9)?,
            disgust: row.get(10)?,
            surprise: row.get(11)?,
            curiosity: row.get(12)?,
            respect: row.get(13)?,
            suspicion: row.get(14)?,
            gratitude: row.get(15)?,
            jealousy: row.get(16)?,
            empathy: row.get(17)?,
            lust: row.get(18)?,
            love: row.get(19)?,
            anxiety: row.get(20)?,
            butterflies: row.get(21)?,
            submissiveness: row.get(22)?,
            dominance: row.get(23)?,
            relationship_score: row.get(24)?,
            last_updated: row.get(25)?,
            created_at: row.get(26)?,
        })
    })
    .optional()
}

pub struct Database {}

impl Database {
    /// Opens the shared companion database with the pragmas every caller
    /// needs: a five-second busy timeout so concurrent access waits instead
    /// of racing rusqlite's default (`sqlite3_busy_timeout(db, 5000)`, which
    /// its own docs mark as "subject to change"), and WAL journaling so
    /// readers (HTTP handlers) and the writer (the generation thread) stop
    /// excluding each other the way the default rollback journal does.
    /// `synchronous=NORMAL` is safe with WAL (durable across crashes, may
    /// lose only the last transactions on power loss) and must be set on
    /// every connection since it is not persisted like `journal_mode` is.
    ///
    /// `PRAGMA foreign_keys` is on: `attitude_memories` used to declare a
    /// foreign key to a `companions` table that did not exist (the table is
    /// named `companion`), which would have broken every attitude-memory
    /// insert with enforcement on. That typo is fixed and `init()` runs a
    /// rebuild migration for databases created before the fix (#110).
    pub fn open() -> Result<Connection> {
        Self::open_at(crate::paths::db_path())
    }

    pub(crate) fn open_at(path: impl AsRef<Path>) -> Result<Connection> {
        let con = Connection::open(path)?;
        con.busy_timeout(Duration::from_secs(5))?;
        con.pragma_update(None, "journal_mode", "WAL")?;
        con.pragma_update(None, "synchronous", "NORMAL")?;
        con.pragma_update(None, "foreign_keys", true)?;
        Ok(con)
    }

    pub fn clear_message_cache() {
        if let Ok(mut cache) = MESSAGE_CACHE.lock() {
            cache.clear();
        }
    }
}

/// The `messages` DDL, shared by `init` and the test fixtures so the column
/// list only exists once. `speaker_id` defaults to `''`: `migrate_messages_speaker_id`
/// backfills it from `ai` on databases created before this column existed.
pub(crate) fn messages_ddl() -> &'static str {
    "CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ai BOOLEAN,
        speaker_id TEXT NOT NULL DEFAULT '',
        content TEXT,
        created_at TEXT
    )"
}

/// Inserts the companion's opening greeting, with `{{char}}`/`{{user}}`
/// resolved to the current companion and user names. Shared by `init`'s
/// first-run seed and `erase_messages`, which both build this exact row.
fn insert_companion_greeting(con: &Connection) -> Result<()> {
    struct CompanionReturn {
        name: String,
        first_message: String,
    }
    let companion_data = con.query_row("SELECT name, first_message FROM companion", [], |row| {
        Ok(CompanionReturn {
            name: row.get(0)?,
            first_message: row.get(1)?,
        })
    })?;
    let user_name: String = con.query_row("SELECT name, persona FROM user LIMIT 1", [], |row| {
        row.get(0)
    })?;
    con.execute(
        "INSERT INTO messages (ai, speaker_id, content, created_at) VALUES (1, ?, ?, ?)",
        [
            CHAR_SPEAKER_ID,
            &companion_data
                .first_message
                .replace("{{char}}", &companion_data.name)
                .replace("{{user}}", &user_name),
            &get_current_date(),
        ],
    )?;
    Ok(())
}

/// Testable half of `Database::get_companion_id`. Single-companion schema,
/// so the only companion row is always the one being asked about.
fn get_companion_id_on(con: &Connection) -> Result<i32> {
    let mut stmt = con.prepare("SELECT id FROM companion LIMIT 1")?;
    let row = stmt.query_row([], |row| row.get(0))?;
    Ok(row)
}

/// Flips `message_id`'s containing checkpoint to `Stale` if it falls inside
/// a `Committed` one, and discards a pending `Draft` checkpoint containing
/// it outright (#181), on the caller's own connection/transaction so it
/// composes with `edit_message_on`/`delete_message_on`/
/// `pop_latest_bot_reply_on`'s own statement.
///
/// The draft check runs *before* the `compacted_through IS NULL`
/// short-circuit below: a chat's very first draft is pending while
/// `compacted_through` is still `NULL` (nothing has committed yet), so
/// short-circuiting on that would let an edit/delete under a pending draft's
/// nose go unnoticed. Discarding it here — rather than merely marking it
/// stale, which only `Committed` rows support — closes both windows a
/// draft's content can go wrong under: extraction still in flight over the
/// old text fails `DraftNotPending` when it tries to write its result
/// (`fill_draft`/`commit` both re-check the status), and a pending review
/// card simply disappears (the hook re-queues on the next round).
///
/// The `Committed`-checkpoint check *does* skip entirely once
/// `compacted_through` is `NULL`, so a chat that never triggered compaction
/// pays one cheap `SELECT` per edit/delete and nothing else — the
/// `id <= compacted_through` bound the original plan also checked here is
/// redundant with `mark_stale_containing_on`'s own `from_message_id <= id
/// <= through_message_id` predicate, so only the `NULL` short-circuit is
/// kept.
fn mark_stale_for_message_on(con: &Connection, message_id: i32) -> Result<()> {
    let companion_id = get_companion_id_on(con)?;
    crate::compaction::store::discard_draft_containing_on(con, companion_id, message_id)?;
    if crate::compaction::store::compacted_through_on(con, companion_id)?.is_none() {
        return Ok(());
    }
    crate::compaction::store::mark_stale_containing_on(con, companion_id, message_id)?;
    Ok(())
}

/// The `attitude_memories` DDL, shared by the table creation path and the
/// rebuild migration below so the column list only exists once. `target_id`
/// stays unconstrained: it is polymorphic on `target_type` (a `user` or
/// `third_party_individuals` row id) and cannot be a single foreign key.
fn attitude_memories_ddl(table_name: &str) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            companion_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            target_type TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            description TEXT NOT NULL,
            priority_score REAL NOT NULL,
            attitude_delta_json TEXT NOT NULL,
            impact_score REAL NOT NULL,
            message_context TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(companion_id) REFERENCES companion(id) ON DELETE CASCADE
        )"
    )
}

impl Database {
    pub fn init() -> Result<usize> {
        let con = Self::open()?;
        con.execute(messages_ddl(), [])?;
        // Must run before the greeting seed below, which names the
        // speaker_id column: a database created before this column existed
        // needs it backfilled first.
        Database::migrate_messages_speaker_id(&con)?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS companion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                persona TEXT,
                example_dialogue TEXT,
                first_message TEXT,
                long_term_mem INTEGER,
                short_term_mem INTEGER,
                roleplay BOOLEAN,
                dialogue_tuning BOOLEAN,
                avatar_path TEXT,
                compacted_through INTEGER
            )",
            [],
        )?;
        Database::migrate_companion_compacted_through(&con)?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                persona TEXT,
                avatar_path TEXT
            )",
            [],
        )?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device TEXT,
                llm_model_path TEXT,
                gpu_layers INTEGER,
                prompt_template TEXT,
                context_window_size INTEGER DEFAULT 2048,
                max_response_tokens INTEGER DEFAULT 512,
                enable_dynamic_context BOOLEAN DEFAULT true,
                vram_limit_gb INTEGER DEFAULT 4,
                dynamic_gpu_allocation BOOLEAN DEFAULT true,
                gpu_safety_margin REAL DEFAULT 0.8,
                min_free_vram_mb INTEGER DEFAULT 512,
                enable_hybrid_context BOOLEAN DEFAULT true,
                max_system_ram_usage_gb INTEGER DEFAULT 8,
                context_expansion_strategy TEXT DEFAULT 'balanced',
                ram_safety_margin_gb INTEGER DEFAULT 2,
                multiplayer_mode TEXT DEFAULT 'solo',
                multiplayer_password TEXT DEFAULT '',
                multiplayer_host_address TEXT DEFAULT '',
                multiplayer_participant_id TEXT DEFAULT '',
                mention_followup_depth INTEGER DEFAULT 1,
                remote_generation_timeout_secs INTEGER DEFAULT 120,
                compact_threshold_tokens INTEGER,
                compact_min_messages INTEGER DEFAULT 8,
                compaction_model_path TEXT,
                heuristic_person_detection BOOLEAN DEFAULT false,
                compaction_attitude_weight REAL DEFAULT 0.5
            )",
            [],
        )?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS companion_attitudes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                companion_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                target_type TEXT NOT NULL CHECK(target_type IN ('user', 'third_party')),
                attraction REAL DEFAULT 0 CHECK(attraction >= -100 AND attraction <= 100),
                trust REAL DEFAULT 0 CHECK(trust >= -100 AND trust <= 100),
                fear REAL DEFAULT 0 CHECK(fear >= -100 AND fear <= 100),
                anger REAL DEFAULT 0 CHECK(anger >= -100 AND anger <= 100),
                joy REAL DEFAULT 0 CHECK(joy >= -100 AND joy <= 100),
                sorrow REAL DEFAULT 0 CHECK(sorrow >= -100 AND sorrow <= 100),
                disgust REAL DEFAULT 0 CHECK(disgust >= -100 AND disgust <= 100),
                surprise REAL DEFAULT 0 CHECK(surprise >= -100 AND surprise <= 100),
                curiosity REAL DEFAULT 0 CHECK(curiosity >= -100 AND curiosity <= 100),
                respect REAL DEFAULT 0 CHECK(respect >= -100 AND respect <= 100),
                suspicion REAL DEFAULT 0 CHECK(suspicion >= -100 AND suspicion <= 100),
                gratitude REAL DEFAULT 0 CHECK(gratitude >= -100 AND gratitude <= 100),
                jealousy REAL DEFAULT 0 CHECK(jealousy >= -100 AND jealousy <= 100),
                empathy REAL DEFAULT 0 CHECK(empathy >= -100 AND empathy <= 100),
                lust REAL DEFAULT 0 CHECK(lust >= -100 AND lust <= 100),
                love REAL DEFAULT 0 CHECK(love >= -100 AND love <= 100),
                anxiety REAL DEFAULT 0 CHECK(anxiety >= -100 AND anxiety <= 100),
                butterflies REAL DEFAULT 0 CHECK(butterflies >= -100 AND butterflies <= 100),
                submissiveness REAL DEFAULT 0 CHECK(submissiveness >= -100 AND submissiveness <= 100),
                dominance REAL DEFAULT 0 CHECK(dominance >= -100 AND dominance <= 100),
                relationship_score REAL GENERATED ALWAYS AS ((attraction + trust + joy + respect + gratitude + empathy + love + lust + butterflies - fear - anger - sorrow - disgust - suspicion - jealousy - anxiety) / 16.0) STORED,
                last_updated TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (companion_id) REFERENCES companion(id) ON DELETE CASCADE,
                UNIQUE(companion_id, target_id, target_type)
            )", []
        )?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS attitude_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attitude_id INTEGER NOT NULL,
                interaction_count INTEGER DEFAULT 0,
                positive_interactions INTEGER DEFAULT 0,
                negative_interactions INTEGER DEFAULT 0,
                neutral_interactions INTEGER DEFAULT 0,
                last_significant_event TEXT,
                relationship_status TEXT DEFAULT 'neutral' CHECK(relationship_status IN ('hostile', 'unfriendly', 'neutral', 'friendly', 'close', 'intimate')),
                notes TEXT,
                FOREIGN KEY (attitude_id) REFERENCES companion_attitudes(id) ON DELETE CASCADE
            )", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_companion_attitudes_companion ON companion_attitudes(companion_id)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_companion_attitudes_target ON companion_attitudes(target_id, target_type)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_companion_attitudes_compound ON companion_attitudes(companion_id, target_id, target_type)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_order ON messages(id DESC)",
            [],
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)",
            [],
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_companion_attitudes_relationship ON companion_attitudes(relationship_score)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_attitude_metadata_attitude ON attitude_metadata(attitude_id)", []
        )?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS third_party_individuals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                relationship_to_user TEXT,
                relationship_to_companion TEXT,
                occupation TEXT,
                personality_traits TEXT,
                physical_description TEXT,
                first_mentioned TEXT NOT NULL,
                last_mentioned TEXT,
                mention_count INTEGER DEFAULT 1,
                importance_score REAL DEFAULT 0.5 CHECK(importance_score >= 0 AND importance_score <= 1),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'heuristic'
            )", []
        )?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS third_party_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                third_party_id INTEGER NOT NULL,
                companion_id INTEGER NOT NULL,
                memory_type TEXT CHECK(memory_type IN ('fact', 'event', 'opinion', 'relationship_change')),
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5 CHECK(importance >= 0 AND importance <= 1),
                emotional_valence REAL DEFAULT 0 CHECK(emotional_valence >= -1 AND emotional_valence <= 1),
                created_at TEXT NOT NULL,
                context_message_id INTEGER,
                FOREIGN KEY (third_party_id) REFERENCES third_party_individuals(id) ON DELETE CASCADE,
                FOREIGN KEY (companion_id) REFERENCES companion(id) ON DELETE CASCADE,
                FOREIGN KEY (context_message_id) REFERENCES messages(id) ON DELETE SET NULL
            )", []
        )?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS third_party_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                third_party_id INTEGER NOT NULL,
                companion_id INTEGER NOT NULL,
                interaction_type TEXT CHECK(interaction_type IN ('planned', 'ongoing', 'completed', 'cancelled')),
                description TEXT NOT NULL,
                planned_date TEXT,
                actual_date TEXT,
                outcome TEXT,
                impact_on_relationship REAL DEFAULT 0 CHECK(impact_on_relationship >= -100 AND impact_on_relationship <= 100),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (third_party_id) REFERENCES third_party_individuals(id) ON DELETE CASCADE,
                FOREIGN KEY (companion_id) REFERENCES companion(id) ON DELETE CASCADE
            )", []
        )?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS third_party_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_party_id INTEGER NOT NULL,
                to_party_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 0.5 CHECK(strength >= 0 AND strength <= 1),
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (from_party_id) REFERENCES third_party_individuals(id) ON DELETE CASCADE,
                FOREIGN KEY (to_party_id) REFERENCES third_party_individuals(id) ON DELETE CASCADE,
                UNIQUE(from_party_id, to_party_id)
            )", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_name ON third_party_individuals(name)",
            [],
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_importance ON third_party_individuals(importance_score DESC, mention_count DESC)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_memories_party ON third_party_memories(third_party_id)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_memories_companion ON third_party_memories(companion_id)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_memories_importance ON third_party_memories(importance DESC, created_at DESC)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_interactions_party ON third_party_interactions(third_party_id)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_interactions_companion ON third_party_interactions(companion_id)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_interactions_date ON third_party_interactions(companion_id, COALESCE(actual_date, planned_date) DESC)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_interactions_type ON third_party_interactions(companion_id, interaction_type, planned_date ASC)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_relationships ON third_party_relationships(from_party_id, to_party_id)", []
        )?;

        // A row created before #177 predates the `source` column; the
        // migration backfills it to `'heuristic'`, which is correct since
        // nothing but this feature writes `'compaction'`.
        Database::migrate_third_party_individuals_table(&con)?;

        if Database::is_table_empty("companion", &con)? {
            con.execute(
                "INSERT INTO companion (name, persona, example_dialogue, first_message, long_term_mem, short_term_mem, roleplay, dialogue_tuning, avatar_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    "Assistant",
                    "{{char}} is an artificial intelligence chatbot designed to help {{user}}. {{char}} is an artificial intelligence created in ai-companion backend",
                    "{{user}}: What is ai-companion?\n{{char}}: AI Companion is a open-source project, wrote in Rust, Typescript and React, that aims to provide users with their own personal AI chatbot on their computer. It allows users to engage in friendly and natural conversations with their AI, creating a unique and personalized experience. This software can also be used as a backend or API for other projects that require a personalised AI chatbot. Very light size, simple installation, simple configuration, quick cold start and ease of use are some of the strengths of AI Companion in comparison to other similar projects.\n{{user}}: Can you tell me about the creator of ai-companion?\n{{char}}: the creator of the ai-companion program is 'Hubert Kasperek', he is a young programmer from Poland who is mostly interested in web development and computer science concepts, he has account on GitHub under nickname \"Hukasx0\"",
                    "Hello {{user}}, how can i help you today?",
                    "2",
                    "5",
                    "1",
                    "1",
                    "/assets/companion_avatar-4rust.jpg"
                ]
            )?;
        }
        if Database::is_table_empty("user", &con)? {
            con.execute(
                "INSERT INTO user (name, persona, avatar_path) VALUES (?, ?, ?)",
                [
                    "User",
                    "{{user}} is chatting with {{char}} using ai-companion web user interface",
                    "/assets/user_avatar-4rust.jpg",
                ],
            )?;
        }
        if Database::is_table_empty("messages", &con)? {
            insert_companion_greeting(&con)?;
        }
        if Database::is_table_empty("config", &con)? {
            con.execute(
                "INSERT INTO config (device, llm_model_path, gpu_layers, prompt_template, context_window_size, max_response_tokens, enable_dynamic_context, vram_limit_gb, dynamic_gpu_allocation, gpu_safety_margin, min_free_vram_mb, enable_hybrid_context, max_system_ram_usage_gb, context_expansion_strategy, ram_safety_margin_gb) VALUES (?, ?, 20, ?, 2048, 512, true, 4, true, 0.8, 512, true, 8, 'balanced', 2)",
                [
                    &Device::CPU as &dyn ToSql,
                    &"path/to/your/gguf/model.gguf",
                    &PromptTemplate::Auto as &dyn ToSql
                ]
            )?;
        }

        // Initialize attitude memories table
        Database::create_attitude_memories_table(&con)?;

        // Rebuild attitude_memories if it still targets the nonexistent
        // `companions` table (#110), before foreign key enforcement runs
        // against it.
        Database::migrate_attitude_memories_foreign_key(&con)?;

        // Migrate config table to add new context window fields if they don't exist
        Database::migrate_config_table(&con)?;

        // Compaction tables (#171): facts reference `companion` and
        // `messages`, both already created above.
        crate::compaction::store::create_tables(&con)?;

        // Migrate companion_attitudes table to add new attitude dimensions if they don't exist
        Database::migrate_companion_attitudes_table(&con)?;

        // Create inference performance metrics table
        con.execute(
            "CREATE TABLE IF NOT EXISTS inference_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_path TEXT NOT NULL,
                gpu_layers INTEGER NOT NULL,
                device_type TEXT DEFAULT 'CPU',
                tokens_per_second REAL NOT NULL,
                time_to_first_token REAL NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_time REAL GENERATED ALWAYS AS (output_tokens / tokens_per_second + time_to_first_token) STORED,
                created_at TEXT NOT NULL
            )", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_inference_metrics_config ON inference_metrics(model_path, gpu_layers, created_at DESC)",
            [],
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_inference_metrics_performance ON inference_metrics(tokens_per_second DESC, created_at DESC)",
            [],
        )?;

        // Create llm_directories table for managing model scan directories
        con.execute(
            "CREATE TABLE IF NOT EXISTS llm_directories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            )",
            [],
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_directories_path ON llm_directories(path)",
            [],
        )?;

        Ok(0)
    }

    pub fn is_table_empty(table_name: &str, con: &Connection) -> Result<bool> {
        let mut stmt = con.prepare(&format!("SELECT COUNT(*) FROM {}", table_name))?;
        let mut rows = stmt.query([])?;
        let count: i64 = rows.next()?.unwrap().get(0)?;
        Ok(count == 0)
    }

    pub fn get_x_messages(x: usize, index: usize) -> Result<Vec<Message>> {
        let cache_key = format!("messages:{}:{}", x, index);

        // Check cache first
        if let Ok(cache) = MESSAGE_CACHE.lock() {
            if let Some((messages, timestamp)) = cache.get(&cache_key) {
                // Cache for 2 minutes for message queries
                if timestamp.elapsed() < Duration::from_secs(120) {
                    return Ok(messages.clone());
                }
            }
        }

        let con = Self::open()?;
        let mut stmt = con.prepare(&format!(
            "SELECT {MESSAGE_COLUMNS} FROM messages ORDER BY id DESC LIMIT ? OFFSET ?"
        ))?;
        let rows = stmt.query_map([x, index], message_from_row)?;
        let mut messages = Vec::new();
        for row in rows {
            messages.push(row?);
        }
        let result: Vec<Message> = messages.into_iter().rev().collect();

        // Cache the results
        if let Ok(mut cache) = MESSAGE_CACHE.lock() {
            // Limit cache size
            if cache.len() > 50 {
                cache.clear();
            }
            cache.insert(cache_key, (result.clone(), Instant::now()));
        }

        Ok(result)
    }

    /// The oldest-first `limit` messages with `id > after` (`None` behaves
    /// as `Some(0)`, since message ids start at 1) — the same shape
    /// `get_x_messages` returns, but anchored on `compacted_through`
    /// (#174's `TranscriptSource::recent_messages`) instead of a plain
    /// offset from the end. Same `MESSAGE_CACHE` pattern as `get_x_messages`;
    /// `clear_message_cache` already wipes every key, this one included.
    pub fn get_x_messages_after(after: Option<i32>, limit: usize) -> Result<Vec<Message>> {
        let cache_key = format!("messages_after:{:?}:{}", after, limit);

        if let Ok(cache) = MESSAGE_CACHE.lock() {
            if let Some((messages, timestamp)) = cache.get(&cache_key) {
                if timestamp.elapsed() < Duration::from_secs(120) {
                    return Ok(messages.clone());
                }
            }
        }

        let con = Self::open()?;
        let result = Self::get_x_messages_after_on(&con, after, limit)?;

        if let Ok(mut cache) = MESSAGE_CACHE.lock() {
            if cache.len() > 50 {
                cache.clear();
            }
            cache.insert(cache_key, (result.clone(), Instant::now()));
        }

        Ok(result)
    }

    /// Testable half of `get_x_messages_after`, taking a caller-provided
    /// connection so tests can point it at a `TempDir`-backed database
    /// instead of the hardwired `paths::db_path()`.
    fn get_x_messages_after_on(
        con: &Connection,
        after: Option<i32>,
        limit: usize,
    ) -> Result<Vec<Message>> {
        let mut stmt = con.prepare(&format!(
            "SELECT {MESSAGE_COLUMNS} FROM messages WHERE id > ?1 ORDER BY id DESC LIMIT ?2"
        ))?;
        let rows = stmt.query_map(params![after.unwrap_or(0), limit], message_from_row)?;
        let mut messages = Vec::new();
        for row in rows {
            messages.push(row?);
        }
        Ok(messages.into_iter().rev().collect())
    }

    pub fn get_total_message_count() -> Result<usize> {
        let con = Self::open()?;
        let count: i64 = con.query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Every message with `id > after_id`, oldest first — the uncompacted
    /// tail compaction's hook (#172) reads once per round, right after the
    /// round's own inserts. Unlike `get_x_messages`, this never goes through
    /// `MESSAGE_CACHE`: it runs at most once per round, not once per page
    /// request, so the cache would only add staleness risk for no benefit.
    pub fn get_messages_after(after_id: i32) -> Result<Vec<Message>> {
        let con = Self::open()?;
        Self::get_messages_after_on(&con, after_id)
    }

    /// Testable half of `get_messages_after`, taking a caller-provided
    /// connection.
    fn get_messages_after_on(con: &Connection, after_id: i32) -> Result<Vec<Message>> {
        let mut stmt = con.prepare(&format!(
            "SELECT {MESSAGE_COLUMNS} FROM messages WHERE id > ? ORDER BY id ASC"
        ))?;
        let rows = stmt.query_map(params![after_id], message_from_row)?;
        rows.collect()
    }

    pub fn get_companion_data() -> Result<CompanionView> {
        let con = Self::open()?;
        let mut stmt = con.prepare("SELECT name, persona, example_dialogue, first_message, long_term_mem, short_term_mem, roleplay, dialogue_tuning, avatar_path FROM companion LIMIT 1")?;
        let row = stmt.query_row([], |row| {
            Ok(CompanionView {
                name: row.get(0)?,
                persona: row.get(1)?,
                example_dialogue: row.get(2)?,
                first_message: row.get(3)?,
                long_term_mem: row.get(4)?,
                short_term_mem: row.get(5)?,
                roleplay: row.get(6)?,
                dialogue_tuning: row.get(7)?,
                avatar_path: row.get(8)?,
            })
        })?;
        Ok(row)
    }

    pub fn get_companion_id() -> Result<i32> {
        let con = Self::open()?;
        get_companion_id_on(&con)
    }

    pub fn get_companion_card_data() -> Result<CharacterCard> {
        let con = Self::open()?;
        let mut stmt = con.prepare(
            "SELECT name, persona, first_message, example_dialogue FROM companion LIMIT 1",
        )?;
        let row = stmt.query_row([], |row| {
            Ok(CharacterCard {
                name: row.get(0)?,
                description: row.get(1)?,
                first_mes: row.get(2)?,
                mes_example: row.get(3)?,
            })
        })?;
        Ok(row)
    }

    pub fn get_user_data() -> Result<UserView> {
        let con = Self::open()?;
        let mut stmt = con.prepare("SELECT name, persona FROM user LIMIT 1")?;
        let row: UserView = stmt.query_row([], |row| {
            Ok(UserView {
                name: row.get(0)?,
                persona: row.get(1)?,
            })
        })?;
        Ok(row)
    }

    pub fn get_message(id: i32) -> Result<Message> {
        let con = Self::open()?;
        let mut stmt = con.prepare(&format!(
            "SELECT {MESSAGE_COLUMNS} FROM messages WHERE id = ?"
        ))?;
        let row = stmt.query_row([id], message_from_row)?;
        Ok(row)
    }

    /// Every message with `from_message_id <= id <= through_message_id`,
    /// ordered by id. What a `compactions` row's range actually covers;
    /// callers convert each row to `compaction::MessageRef` via `From`,
    /// which drops `created_at` before it reaches the pure compaction
    /// modules.
    #[allow(dead_code)] // wired up by #172's range.rs
    pub fn get_messages_between(
        from_message_id: i32,
        through_message_id: i32,
    ) -> Result<Vec<Message>> {
        let con = Self::open()?;
        Self::get_messages_between_on(&con, from_message_id, through_message_id)
    }

    /// Testable half of `get_messages_between`, taking a caller-provided
    /// connection so tests can point it at a `TempDir`-backed database
    /// instead of the hardwired `paths::db_path()`, mirroring
    /// `insert_message_on`.
    #[allow(dead_code)] // wired up by #172's range.rs
    fn get_messages_between_on(
        con: &Connection,
        from_message_id: i32,
        through_message_id: i32,
    ) -> Result<Vec<Message>> {
        let mut stmt = con.prepare(&format!(
            "SELECT {MESSAGE_COLUMNS} FROM messages WHERE id >= ? AND id <= ? ORDER BY id"
        ))?;
        let rows = stmt.query_map([from_message_id, through_message_id], message_from_row)?;
        rows.collect()
    }

    /// Inserts `message` and returns the new row's id — what #131's
    /// `TurnStore::insert_reply` puts on `PersistedReply::message_id`.
    pub fn insert_message(message: NewMessage) -> Result<i32, Error> {
        let con = Self::open()?;
        Self::insert_message_on(&con, message)
    }

    /// Testable half of `insert_message`, taking a caller-provided connection
    /// so tests can point it at a `TempDir`-backed database instead of the
    /// hardwired `paths::db_path()`, mirroring `pop_latest_bot_reply_on`.
    fn insert_message_on(con: &Connection, message: NewMessage) -> Result<i32, Error> {
        con.execute(
            "INSERT INTO messages (ai, speaker_id, content, created_at) VALUES (?, ?, ?, ?)",
            params![
                is_ai_speaker(&message.speaker_id),
                message.speaker_id,
                message.content,
                get_current_date()
            ],
        )?;
        let id = con.last_insert_rowid() as i32;

        // Clear message cache when new message is inserted
        Database::clear_message_cache();

        Ok(id)
    }

    /// Updates a message's text without ever touching its role. Deliberately
    /// takes `MessageEdit`, not `NewMessage`: the request type has no `ai`
    /// field, so there is nothing here that could flip a companion reply
    /// into a user message (or vice versa) through an edit.
    pub fn edit_message(id: i32, edit: MessageEdit) -> Result<(), Error> {
        let mut con = Self::open()?;
        Self::edit_message_on(&mut con, id, edit)
    }

    /// Testable half of `edit_message`, taking a caller-provided connection
    /// so tests can point it at a `TempDir`-backed database instead of the
    /// hardwired `paths::db_path()`, mirroring `pop_latest_bot_reply_on`.
    /// Runs the edit and #181's stale-checkpoint check
    /// (`mark_stale_for_message_on`) inside one `Immediate` transaction, so
    /// editing history under a committed checkpoint can never leave that
    /// checkpoint `Committed` while the edit itself is durable, or vice
    /// versa. Clears the message cache after commit (not inside `open()`)
    /// so the cache-invalidation test can exercise it without touching the
    /// real `paths::db_path()`.
    fn edit_message_on(con: &mut Connection, id: i32, edit: MessageEdit) -> Result<(), Error> {
        let tx = con.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            params![edit.content, id],
        )?;
        mark_stale_for_message_on(&tx, id)?;
        tx.commit()?;

        Database::clear_message_cache();

        Ok(())
    }

    pub fn delete_message(id: i32) -> Result<(), Error> {
        let mut con = Self::open()?;
        Self::delete_message_on(&mut con, id)
    }

    /// Testable half of `delete_message`, mirroring `edit_message_on`:
    /// deletes the message and runs #181's stale-checkpoint check in one
    /// `Immediate` transaction. A pin on `id` cascades away with the
    /// message row itself (`pinned_messages.message_id REFERENCES
    /// messages(id) ON DELETE CASCADE`, and `foreign_keys = ON` on every
    /// connection `Database::open_at` returns) rather than needing an
    /// explicit `DELETE FROM pinned_messages` first.
    fn delete_message_on(con: &mut Connection, id: i32) -> Result<(), Error> {
        let tx = con.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM messages WHERE id = ?", [id])?;
        mark_stale_for_message_on(&tx, id)?;
        tx.commit()?;

        Database::clear_message_cache();

        Ok(())
    }

    /// Removes the trailing bot reply so a regenerate can re-prompt from the
    /// user turn it answered, without ever deleting a user message.
    ///
    /// `owner_ready(speaker_id)` decides whether the reply's owner can
    /// actually regenerate it right now (the host companion is always
    /// ready; a remote bot is ready only while it is connected); see
    /// `pop_latest_bot_reply_on` for why the check has to run inside the
    /// same transaction as the delete.
    pub fn pop_latest_bot_reply(owner_ready: impl FnOnce(&str) -> bool) -> Result<PoppedReply> {
        let mut con = Self::open()?;
        Self::pop_latest_bot_reply_on(&mut con, owner_ready)
    }

    /// Testable half of `pop_latest_bot_reply`, taking a caller-provided
    /// connection so tests can point it at a `TempDir`-backed database
    /// instead of the hardwired `paths::db_path()`.
    ///
    /// Runs the "is the newest row a bot reply with a preceding user turn"
    /// check, `owner_ready`, the delete, and #181's stale-checkpoint check
    /// inside one `IMMEDIATE` transaction, so a concurrent
    /// `DELETE /api/message/{id}` or `POST /api/message` (neither of which
    /// is covered by the turn slot) cannot interleave between the check and
    /// the delete, and a remote bot cannot go offline between "checked
    /// ready" and "deleted". The stale check matters here too, not just in
    /// `edit_message_on`/`delete_message_on`: with `short_term_mem: 0` (not
    /// validated by `edit_companion`) `select_range` can include the
    /// newest message in a committed checkpoint's range, and this is the
    /// only path that deletes that specific row.
    fn pop_latest_bot_reply_on(
        con: &mut Connection,
        owner_ready: impl FnOnce(&str) -> bool,
    ) -> Result<PoppedReply> {
        let tx = con.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let latest: Option<Message> = tx
            .query_row(
                &format!("SELECT {MESSAGE_COLUMNS} FROM messages ORDER BY id DESC LIMIT 1"),
                [],
                message_from_row,
            )
            .optional()?;
        // Poppable when the trailing row is neither the user's own message
        // nor a system notice — i.e. some bot's reply — regardless of what
        // row precedes it: with rounds (#131/#132) the newest reply can
        // follow another bot's reply rather than the user's.
        let reply = match latest {
            Some(message)
                if message.speaker_id != USER_SPEAKER_ID
                    && message.speaker_id != SYSTEM_SPEAKER_ID =>
            {
                message
            }
            _ => return Ok(PoppedReply::NothingToRegenerate),
        };

        // Anchored on the newest row with `speaker_id = 'user'` before the
        // reply, not on its immediate predecessor: that predecessor can be
        // another bot's reply (or this bot's own earlier one) from the same
        // round. This keeps the original invariant in its real form — the
        // anchor can never be bot content mistaken for the user's message —
        // without requiring the user's turn to be immediately adjacent.
        let user_turn: Option<Message> = tx
            .query_row(
                &format!(
                    "SELECT {MESSAGE_COLUMNS} FROM messages WHERE id < ? AND speaker_id = ? ORDER BY id DESC LIMIT 1"
                ),
                params![reply.id, USER_SPEAKER_ID],
                message_from_row,
            )
            .optional()?;
        let Some(user_turn) = user_turn else {
            return Ok(PoppedReply::NothingToRegenerate);
        };

        if !owner_ready(&reply.speaker_id) {
            return Ok(PoppedReply::OwnerUnavailable {
                speaker_id: reply.speaker_id,
            });
        }

        let message_id = reply.id;
        tx.execute("DELETE FROM messages WHERE id = ?", [message_id])?;
        mark_stale_for_message_on(&tx, message_id)?;
        tx.commit()?;

        // Cleared after commit (not inside `open()`) so the invalidation is
        // observable from a test that points this function at a TempDir.
        Database::clear_message_cache();

        Ok(PoppedReply::Removed {
            speaker_id: reply.speaker_id,
            message_id,
            user_turn,
        })
    }

    pub fn erase_messages() -> Result<(), Error> {
        let mut con = Self::open()?;
        Self::erase_messages_on(&mut con)
    }

    /// Testable half of `erase_messages`, mirroring `edit_message_on`:
    /// resets every compaction table (#181's `clear_all_on` — every
    /// checkpoint, fact, and pin, plus `compacted_through` back to `NULL`),
    /// deletes every message and every compaction-sourced person (#177 —
    /// they belong to the story just deleted; a heuristic row is left alone,
    /// the cleanup endpoint owns those), and reseeds the greeting, all
    /// inside one `Immediate` transaction.
    fn erase_messages_on(con: &mut Connection) -> Result<(), Error> {
        let tx = con.transaction_with_behavior(TransactionBehavior::Immediate)?;
        crate::compaction::store::clear_all_on(&tx)?;
        tx.execute("DELETE FROM messages", [])?;
        Self::delete_compaction_persons_in(&tx)?;
        insert_companion_greeting(&tx)?;
        tx.commit()?;

        Database::clear_message_cache();

        Ok(())
    }

    pub fn edit_companion(companion: CompanionView) -> Result<(), Error> {
        let con = Self::open()?;
        con.execute(
            &format!("UPDATE companion SET name = ?, persona = ?, example_dialogue = ?, first_message = ?, long_term_mem = {}, short_term_mem = {}, roleplay = {}, dialogue_tuning = {}, avatar_path = ?", companion.long_term_mem, companion.short_term_mem, companion.roleplay, companion.dialogue_tuning),
            [
                &companion.name,
                &companion.persona,
                &companion.example_dialogue,
                &companion.first_message,
                &companion.avatar_path,
            ]
        )?;
        Ok(())
    }

    pub fn import_character_json(companion: CharacterCard) -> Result<(), Error> {
        let con = Self::open()?;
        con.execute(
            "UPDATE companion SET name = ?, persona = ?, example_dialogue = ?, first_message = ?",
            [
                &companion.name,
                &companion.description,
                &companion.mes_example,
                &companion.first_mes,
            ],
        )?;
        Ok(())
    }

    pub fn import_character_card(companion: CharacterCard, image_path: &str) -> Result<(), Error> {
        let con = Self::open()?;
        con.execute(
            "UPDATE companion SET name = ?, persona = ?, example_dialogue = ?, first_message = ?, avatar_path = ?",
            [
                &companion.name,
                &companion.description,
                &companion.mes_example,
                &companion.first_mes,
                image_path
            ]
        )?;
        Ok(())
    }

    pub fn change_companion_avatar(avatar_path: &str) -> Result<(), Error> {
        let con = Self::open()?;
        con.execute("UPDATE companion SET avatar_path = ?", [avatar_path])?;
        Ok(())
    }

    pub fn edit_user(user: UserView) -> Result<(), Error> {
        let con = Self::open()?;
        con.execute(
            "UPDATE user SET name = ?, persona = ?",
            [&user.name, &user.persona],
        )?;
        Ok(())
    }

    pub fn get_config() -> Result<ConfigView> {
        let con = Self::open()?;
        Self::read_config(&con)
    }

    /// The `SELECT`/row-mapping half of `get_config`, split out so tests can
    /// run it against a temp-file `Connection` instead of the real database
    /// path (matches how `migrate_config_table` already takes a
    /// connection).
    fn read_config(con: &Connection) -> Result<ConfigView> {
        let mut stmt = con.prepare("SELECT device, llm_model_path, gpu_layers, prompt_template, context_window_size, max_response_tokens, enable_dynamic_context, vram_limit_gb, dynamic_gpu_allocation, gpu_safety_margin, min_free_vram_mb, enable_hybrid_context, max_system_ram_usage_gb, context_expansion_strategy, ram_safety_margin_gb, multiplayer_mode, multiplayer_password, multiplayer_host_address, multiplayer_participant_id, mention_followup_depth, remote_generation_timeout_secs, compact_threshold_tokens, compact_min_messages, compaction_model_path, heuristic_person_detection, compaction_attitude_weight FROM config LIMIT 1")?;
        let row = stmt.query_row([], |row| {
            let multiplayer_password: String =
                row.get::<_, Option<String>>(16)?.unwrap_or_default();
            let compaction_model_path: Option<String> = row.get::<_, Option<String>>(23)?;
            Ok(ConfigView {
                device: row.get(0)?,
                llm_model_path: row.get(1)?,
                gpu_layers: row.get(2)?,
                prompt_template: row.get(3)?,
                context_window_size: row.get::<_, Option<usize>>(4)?.unwrap_or(2048),
                max_response_tokens: row.get::<_, Option<usize>>(5)?.unwrap_or(512),
                enable_dynamic_context: row.get::<_, Option<bool>>(6)?.unwrap_or(true),
                vram_limit_gb: row.get::<_, Option<usize>>(7)?.unwrap_or(4),
                dynamic_gpu_allocation: row.get::<_, Option<bool>>(8)?.unwrap_or(true),
                gpu_safety_margin: row.get::<_, Option<f32>>(9)?.unwrap_or(0.8),
                min_free_vram_mb: row.get::<_, Option<u64>>(10)?.unwrap_or(512),
                enable_hybrid_context: row.get::<_, Option<bool>>(11)?.unwrap_or(true),
                max_system_ram_usage_gb: row.get::<_, Option<usize>>(12)?.unwrap_or(8),
                context_expansion_strategy: row
                    .get::<_, Option<String>>(13)?
                    .unwrap_or("balanced".to_string()),
                ram_safety_margin_gb: row.get::<_, Option<usize>>(14)?.unwrap_or(2),
                multiplayer_mode: row
                    .get::<_, Option<MultiplayerMode>>(15)?
                    .unwrap_or_default(),
                multiplayer_password_set: !multiplayer_password.is_empty(),
                multiplayer_password,
                multiplayer_host_address: row.get::<_, Option<String>>(17)?.unwrap_or_default(),
                multiplayer_participant_id: row.get::<_, Option<String>>(18)?.unwrap_or_default(),
                mention_followup_depth: row.get::<_, Option<u8>>(19)?.unwrap_or(1),
                remote_generation_timeout_secs: row.get::<_, Option<u64>>(20)?.unwrap_or(120),
                compact_threshold_tokens: row.get::<_, Option<usize>>(21)?,
                compact_min_messages: row.get::<_, Option<usize>>(22)?.unwrap_or(8),
                // Empty string and NULL both read as "use llm_model_path".
                compaction_model_path: compaction_model_path.filter(|s| !s.is_empty()),
                heuristic_person_detection: row.get::<_, Option<bool>>(24)?.unwrap_or(false),
                compaction_attitude_weight: row.get::<_, Option<f32>>(25)?.unwrap_or(0.5),
            })
        })?;
        Ok(row)
    }

    pub fn change_config(config: ConfigModify) -> Result<(), ConfigChangeError> {
        let con = Self::open()?;
        Self::write_config(&con, config)
    }

    /// The validation/`UPDATE` half of `change_config`, split out so tests
    /// can run it against a temp-file `Connection`.
    ///
    /// Validation order: device, then prompt template, then the
    /// multiplayer fields as one unit via `MultiplayerConfig::parse`, then
    /// (only once every field parses) the host-mode-needs-a-password rule,
    /// which needs the currently stored password. The main `UPDATE` and the
    /// password `UPDATE` run in one transaction so a crash between the two
    /// can never leave a password write half-applied; the password
    /// statement only runs when the caller supplied a non-empty
    /// `multiplayer_password`, so an empty or absent one never clears the
    /// stored value.
    fn write_config(con: &Connection, config: ConfigModify) -> Result<(), ConfigChangeError> {
        let device = match config.device.as_str() {
            "CPU" => Device::CPU,
            "GPU" => Device::GPU,
            "Metal" => Device::Metal,
            _ => {
                return Err(ConfigChangeError::Invalid(
                    "Invalid device type".to_string(),
                ))
            }
        };

        let prompt_template = match config.prompt_template.as_str() {
            "Auto" => PromptTemplate::Auto,
            "Default" => PromptTemplate::Default,
            "Llama2" => PromptTemplate::Llama2,
            "Mistral" => PromptTemplate::Mistral,
            _ => {
                return Err(ConfigChangeError::Invalid(
                    "Invalid prompt template type".to_string(),
                ))
            }
        };

        let multiplayer = MultiplayerConfig::parse(
            &config.multiplayer_mode,
            &config.multiplayer_host_address,
            &config.multiplayer_participant_id,
            config.mention_followup_depth,
            config.remote_generation_timeout_secs,
        )
        .map_err(|e| ConfigChangeError::Invalid(e.to_string()))?;

        let stored_password: String = con
            .query_row(
                "SELECT multiplayer_password FROM config LIMIT 1",
                [],
                |row| row.get::<_, Option<String>>(0),
            )?
            .unwrap_or_default();
        let incoming_password = config.multiplayer_password.clone().unwrap_or_default();
        if multiplayer.mode == MultiplayerMode::Host
            && stored_password.is_empty()
            && incoming_password.is_empty()
        {
            return Err(ConfigChangeError::Invalid(
                "host mode requires a password".to_string(),
            ));
        }

        if config.compact_min_messages < 2 {
            return Err(ConfigChangeError::Invalid(
                "compact_min_messages must be at least 2".to_string(),
            ));
        }
        if let Some(threshold) = config.compact_threshold_tokens {
            if threshold < 256 {
                return Err(ConfigChangeError::Invalid(
                    "compact_threshold_tokens must be at least 256".to_string(),
                ));
            }
        }
        if !(0.0..=1.0).contains(&config.compaction_attitude_weight) {
            return Err(ConfigChangeError::Invalid(
                "compaction_attitude_weight must be within 0..=1".to_string(),
            ));
        }

        let tx = con.unchecked_transaction()?;
        tx.execute(
            "UPDATE config SET device = ?, llm_model_path = ?, gpu_layers = ?, prompt_template = ?, context_window_size = ?, max_response_tokens = ?, enable_dynamic_context = ?, vram_limit_gb = ?, dynamic_gpu_allocation = ?, gpu_safety_margin = ?, min_free_vram_mb = ?, enable_hybrid_context = ?, max_system_ram_usage_gb = ?, context_expansion_strategy = ?, ram_safety_margin_gb = ?, multiplayer_mode = ?, multiplayer_host_address = ?, multiplayer_participant_id = ?, mention_followup_depth = ?, remote_generation_timeout_secs = ?, compact_threshold_tokens = ?, compact_min_messages = ?, compaction_model_path = ?, heuristic_person_detection = ?, compaction_attitude_weight = ?",
            params![
                &device as &dyn ToSql,
                &config.llm_model_path,
                &config.gpu_layers,
                &prompt_template as &dyn ToSql,
                &config.context_window_size,
                &config.max_response_tokens,
                &config.enable_dynamic_context,
                &config.vram_limit_gb,
                &config.dynamic_gpu_allocation,
                &config.gpu_safety_margin,
                &config.min_free_vram_mb,
                &config.enable_hybrid_context,
                &config.max_system_ram_usage_gb,
                &config.context_expansion_strategy,
                &config.ram_safety_margin_gb,
                &multiplayer.mode as &dyn ToSql,
                &config.multiplayer_host_address,
                &config.multiplayer_participant_id,
                &config.mention_followup_depth,
                &config.remote_generation_timeout_secs,
                &config.compact_threshold_tokens,
                &config.compact_min_messages,
                &config.compaction_model_path,
                &config.heuristic_person_detection,
                &config.compaction_attitude_weight,
            ],
        )?;

        if !incoming_password.is_empty() {
            tx.execute(
                "UPDATE config SET multiplayer_password = ?",
                params![&incoming_password],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    pub fn create_or_update_attitude(
        companion_id: i32,
        target_id: i32,
        target_type: &str,
        attitude: &CompanionAttitude,
    ) -> Result<i32> {
        let con = Self::open()?;
        let current_time = get_current_date();

        let existing_id: Option<i32> = con.query_row(
            "SELECT id FROM companion_attitudes WHERE companion_id = ? AND target_id = ? AND target_type = ?",
            params![companion_id, target_id, target_type],
            |row| row.get(0)
        ).ok();

        if let Some(id) = existing_id {
            con.execute(
                "UPDATE companion_attitudes SET
                    attraction = ?, trust = ?, fear = ?, anger = ?, joy = ?, sorrow = ?,
                    disgust = ?, surprise = ?, curiosity = ?, respect = ?, suspicion = ?,
                    gratitude = ?, jealousy = ?, empathy = ?, lust = ?, love = ?,
                    anxiety = ?, butterflies = ?, submissiveness = ?, dominance = ?, last_updated = ?
                WHERE id = ?",
                params![
                    attitude.attraction,
                    attitude.trust,
                    attitude.fear,
                    attitude.anger,
                    attitude.joy,
                    attitude.sorrow,
                    attitude.disgust,
                    attitude.surprise,
                    attitude.curiosity,
                    attitude.respect,
                    attitude.suspicion,
                    attitude.gratitude,
                    attitude.jealousy,
                    attitude.empathy,
                    attitude.lust,
                    attitude.love,
                    attitude.anxiety,
                    attitude.butterflies,
                    attitude.submissiveness,
                    attitude.dominance,
                    current_time,
                    id
                ],
            )?;
            Ok(id)
        } else {
            con.execute(
                "INSERT INTO companion_attitudes (
                    companion_id, target_id, target_type, attraction, trust, fear, anger,
                    joy, sorrow, disgust, surprise, curiosity, respect, suspicion,
                    gratitude, jealousy, empathy, lust, love, anxiety, butterflies,
                    submissiveness, dominance, last_updated, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    companion_id,
                    target_id,
                    target_type,
                    attitude.attraction,
                    attitude.trust,
                    attitude.fear,
                    attitude.anger,
                    attitude.joy,
                    attitude.sorrow,
                    attitude.disgust,
                    attitude.surprise,
                    attitude.curiosity,
                    attitude.respect,
                    attitude.suspicion,
                    attitude.gratitude,
                    attitude.jealousy,
                    attitude.empathy,
                    attitude.lust,
                    attitude.love,
                    attitude.anxiety,
                    attitude.butterflies,
                    attitude.submissiveness,
                    attitude.dominance,
                    current_time,
                    current_time
                ],
            )?;
            Ok(con.last_insert_rowid() as i32)
        }
    }

    pub fn get_attitude(
        companion_id: i32,
        target_id: i32,
        target_type: &str,
    ) -> Result<Option<CompanionAttitude>> {
        let con = Self::open()?;
        read_attitude_row(&con, companion_id, target_id, target_type)
    }

    pub fn update_attitude_dimension(
        companion_id: i32,
        target_id: i32,
        target_type: &str,
        dimension: &str,
        delta: f32,
    ) -> Result<()> {
        // No attitude memory is recorded here: a manual edit through
        // `PUT /api/attitude/dimension` is not something the companion lived
        // through. Memories are recorded per conversation turn, from
        // `finish_turn` in `main.rs`.
        let con = Self::open()?;
        let current_time = get_current_date();

        let query = format!(
            "UPDATE companion_attitudes
             SET {} = MAX(-100, MIN(100, {} + ?)), last_updated = ?
             WHERE companion_id = ? AND target_id = ? AND target_type = ?",
            dimension, dimension
        );

        con.execute(
            &query,
            params![delta, current_time, companion_id, target_id, target_type],
        )?;

        Ok(())
    }

    /// Applies a batch of engine-derived deltas to one attitude row in a single
    /// transaction and returns the (previous, current) pair.
    ///
    /// Returns `Ok(None)` when no row exists yet for `(companion_id, target_id,
    /// target_type)` — the caller must seed one (see `default_user_attitude`
    /// and `create_initial_user_attitude`) before deltas can land anywhere,
    /// since an `UPDATE` against a missing row silently changes zero rows.
    ///
    /// Unlike `update_attitude_dimension`, this does not call
    /// `detect_attitude_change`: turning the returned pair into an attitude
    /// memory is the caller's responsibility (see `finish_turn` in `main.rs`).
    pub fn apply_attitude_deltas(
        companion_id: i32,
        target_id: i32,
        target_type: &str,
        deltas: &[crate::attitude_engine::DimensionDelta],
    ) -> Result<Option<(CompanionAttitude, CompanionAttitude)>> {
        Self::apply_attitude_deltas_computed(companion_id, target_id, target_type, |_previous| {
            deltas.to_vec()
        })
    }

    /// Like `apply_attitude_deltas`, but `compute` derives the deltas from
    /// `previous` *after* it is read inside this function's own `Immediate`
    /// transaction, rather than a caller handing in numbers already fixed
    /// against an earlier, separately-read snapshot.
    ///
    /// A caller that reads the current attitude on one connection, computes
    /// deltas from it, then applies them on another (or on this same
    /// function, but after its own transaction has already opened) leaves a
    /// window where a concurrent writer's change lands in between: the
    /// applied numbers end up relative to a value that is no longer current
    /// by write time, silently breaking whatever semantics produced them
    /// (e.g. `compaction::attitude::blend`'s "blend weight of the way toward
    /// the rating" — the same read-then-write shape #175's `commit`/
    /// `discard` closed by re-checking status inside the transaction that
    /// writes it). Passing `compute` instead of a precomputed slice makes
    /// that race structurally impossible: `compute` only ever sees the
    /// `previous` this same transaction is about to write against.
    pub fn apply_attitude_deltas_computed(
        companion_id: i32,
        target_id: i32,
        target_type: &str,
        compute: impl FnOnce(&CompanionAttitude) -> Vec<crate::attitude_engine::DimensionDelta>,
    ) -> Result<Option<(CompanionAttitude, CompanionAttitude)>> {
        let mut con = Self::open()?;
        let current_time = get_current_date();

        // Immediate acquires the write lock before the first read below, so
        // the previous/current snapshots and the writes between them are one
        // continuous critical section: no concurrent writer's changes can
        // land between "previous" and "current" and be misattributed to this
        // turn's diff.
        let tx = con.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let previous = match read_attitude_row(&tx, companion_id, target_id, target_type)? {
            Some(attitude) => attitude,
            None => return Ok(None),
        };

        let deltas = compute(&previous);
        if deltas.is_empty() {
            return Ok(Some((previous.clone(), previous)));
        }

        for delta in &deltas {
            let column = delta.dimension.column();
            let query = format!(
                "UPDATE companion_attitudes
                 SET {} = MAX(-100, MIN(100, {} + ?)), last_updated = ?
                 WHERE companion_id = ? AND target_id = ? AND target_type = ?",
                column, column
            );
            tx.execute(
                &query,
                params![
                    delta.delta,
                    current_time,
                    companion_id,
                    target_id,
                    target_type
                ],
            )?;
        }

        let current = read_attitude_row(&tx, companion_id, target_id, target_type)?
            .unwrap_or_else(|| previous.clone());

        tx.commit()?;

        Ok(Some((previous, current)))
    }

    pub fn get_all_companion_attitudes(companion_id: i32) -> Result<Vec<CompanionAttitude>> {
        let con = Self::open()?;
        let mut stmt = con.prepare(
            "SELECT id, companion_id, target_id, target_type, attraction, trust, fear, anger,
                    joy, sorrow, disgust, surprise, curiosity, respect, suspicion,
                    gratitude, jealousy, empathy, lust, love, anxiety, butterflies,
                    submissiveness, dominance, relationship_score, last_updated, created_at
             FROM companion_attitudes
             WHERE companion_id = ?
             ORDER BY relationship_score DESC",
        )?;

        let attitudes = stmt.query_map([&companion_id], |row| {
            Ok(CompanionAttitude {
                id: Some(row.get(0)?),
                companion_id: row.get(1)?,
                target_id: row.get(2)?,
                target_type: row.get(3)?,
                attraction: row.get(4)?,
                trust: row.get(5)?,
                fear: row.get(6)?,
                anger: row.get(7)?,
                joy: row.get(8)?,
                sorrow: row.get(9)?,
                disgust: row.get(10)?,
                surprise: row.get(11)?,
                curiosity: row.get(12)?,
                respect: row.get(13)?,
                suspicion: row.get(14)?,
                gratitude: row.get(15)?,
                jealousy: row.get(16)?,
                empathy: row.get(17)?,
                lust: row.get(18)?,
                love: row.get(19)?,
                anxiety: row.get(20)?,
                butterflies: row.get(21)?,
                submissiveness: row.get(22)?,
                dominance: row.get(23)?,
                relationship_score: row.get(24)?,
                last_updated: row.get(25)?,
                created_at: row.get(26)?,
            })
        })?;

        let mut result = Vec::new();
        for attitude in attitudes {
            result.push(attitude?);
        }

        Ok(result)
    }

    #[allow(dead_code)]
    pub fn update_attitude_metadata(
        attitude_id: i32,
        interaction_type: &str,
        event: Option<&str>,
    ) -> Result<()> {
        let con = Self::open()?;

        let field = match interaction_type {
            "positive" => "positive_interactions",
            "negative" => "negative_interactions",
            "neutral" => "neutral_interactions",
            _ => {
                return Err(Error::InvalidParameterName(
                    "Invalid interaction type".to_string(),
                ))
            }
        };

        let query = format!(
            "UPDATE attitude_metadata
             SET interaction_count = interaction_count + 1, {} = {} + 1, last_significant_event = COALESCE(?, last_significant_event)
             WHERE attitude_id = ?",
            field, field
        );

        con.execute(&query, params![event, attitude_id])?;

        Ok(())
    }

    pub fn clear_companion_attitudes(companion_id: i32) -> Result<()> {
        let con = Self::open()?;
        con.execute(
            "DELETE FROM companion_attitudes WHERE companion_id = ?",
            params![companion_id],
        )?;
        Ok(())
    }

    /// Unadjusted starting point for a new `companion_attitudes` row, before
    /// persona adjustment. Pure (no connection), so callers that only need a
    /// decay baseline can use it without touching SQLite.
    pub fn default_user_attitude(companion_id: i32, user_id: i32) -> CompanionAttitude {
        CompanionAttitude {
            id: None,
            companion_id,
            target_id: user_id,
            target_type: "user".to_string(),
            attraction: 50.0,
            trust: 45.0,
            fear: 5.0,
            anger: 5.0,
            joy: 40.0,
            sorrow: 10.0,
            disgust: 5.0,
            surprise: 30.0,
            curiosity: 60.0,
            respect: 40.0,
            suspicion: 15.0,
            gratitude: 20.0,
            jealousy: 10.0,
            empathy: 50.0,
            lust: 25.0,
            love: 30.0,
            anxiety: 20.0,
            butterflies: 15.0,
            submissiveness: 30.0,
            dominance: 35.0,
            relationship_score: Some(0.0),
            last_updated: get_current_date(),
            created_at: get_current_date(),
        }
    }

    pub fn create_initial_user_attitude(
        companion_id: i32,
        user_id: i32,
        companion_persona: &str,
    ) -> Result<i32> {
        let base_attitude = Database::default_user_attitude(companion_id, user_id);
        let adjusted_attitude =
            Database::adjust_attitude_for_persona(&base_attitude, companion_persona);
        Database::create_or_update_attitude(companion_id, user_id, "user", &adjusted_attitude)
    }

    /// Inserts a persona-adjusted default attitude row for a user target only
    /// if one does not already exist, relying on the table's
    /// `UNIQUE(companion_id, target_id, target_type)` constraint to no-op
    /// instead of overwriting.
    ///
    /// This is the seed path `finish_turn` uses when `get_attitude` reports no
    /// row: unlike `create_initial_user_attitude` (still used by the
    /// `/api/attitude/clear` REST handler, which legitimately wants to reset
    /// an existing row via `create_or_update_attitude`'s UPDATE fallback), this
    /// never falls back to an UPDATE, so a wrongly inferred "row absent"
    /// premise can never wipe accumulated attitude state.
    pub fn seed_missing_user_attitude(
        companion_id: i32,
        user_id: i32,
        companion_persona: &str,
    ) -> Result<()> {
        let base_attitude = Database::default_user_attitude(companion_id, user_id);
        let attitude = Database::adjust_attitude_for_persona(&base_attitude, companion_persona);
        let con = Self::open()?;
        con.execute(
            "INSERT OR IGNORE INTO companion_attitudes (
                companion_id, target_id, target_type, attraction, trust, fear, anger,
                joy, sorrow, disgust, surprise, curiosity, respect, suspicion,
                gratitude, jealousy, empathy, lust, love, anxiety, butterflies,
                submissiveness, dominance, last_updated, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                companion_id,
                user_id,
                "user",
                attitude.attraction,
                attitude.trust,
                attitude.fear,
                attitude.anger,
                attitude.joy,
                attitude.sorrow,
                attitude.disgust,
                attitude.surprise,
                attitude.curiosity,
                attitude.respect,
                attitude.suspicion,
                attitude.gratitude,
                attitude.jealousy,
                attitude.empathy,
                attitude.lust,
                attitude.love,
                attitude.anxiety,
                attitude.butterflies,
                attitude.submissiveness,
                attitude.dominance,
                attitude.last_updated,
                attitude.created_at,
            ],
        )?;
        Ok(())
    }

    pub fn adjust_attitude_for_persona(
        base_attitude: &CompanionAttitude,
        persona: &str,
    ) -> CompanionAttitude {
        let mut attitude = base_attitude.clone();
        let persona_lower = persona.to_lowercase();

        if persona_lower.contains("shy") || persona_lower.contains("introverted") {
            attitude.curiosity -= 10.0;
            attitude.anxiety += 15.0;
            attitude.trust -= 10.0;
            attitude.submissiveness += 10.0;
        }

        if persona_lower.contains("confident") || persona_lower.contains("outgoing") {
            attitude.curiosity += 15.0;
            attitude.anxiety -= 10.0;
            attitude.dominance += 10.0;
            attitude.attraction += 5.0;
        }

        if persona_lower.contains("friendly") || persona_lower.contains("warm") {
            attitude.joy += 15.0;
            attitude.empathy += 10.0;
            attitude.trust += 10.0;
            attitude.gratitude += 10.0;
        }

        if persona_lower.contains("cold") || persona_lower.contains("distant") {
            attitude.joy -= 10.0;
            attitude.empathy -= 15.0;
            attitude.trust -= 15.0;
            attitude.suspicion += 10.0;
        }

        if persona_lower.contains("flirty") || persona_lower.contains("seductive") {
            attitude.attraction += 15.0;
            attitude.lust += 20.0;
            attitude.butterflies += 10.0;
        }

        if persona_lower.contains("aggressive") || persona_lower.contains("dominant") {
            attitude.dominance += 15.0;
            attitude.anger += 10.0;
            attitude.submissiveness -= 10.0;
        }

        if persona_lower.contains("submissive") || persona_lower.contains("obedient") {
            attitude.submissiveness += 15.0;
            attitude.dominance -= 10.0;
            attitude.respect += 10.0;
        }

        if persona_lower.contains("curious") || persona_lower.contains("inquisitive") {
            attitude.curiosity += 20.0;
            attitude.surprise += 10.0;
        }

        attitude.attraction = attitude.attraction.clamp(0.0, 100.0);
        attitude.trust = attitude.trust.clamp(0.0, 100.0);
        attitude.fear = attitude.fear.clamp(0.0, 100.0);
        attitude.anger = attitude.anger.clamp(0.0, 100.0);
        attitude.joy = attitude.joy.clamp(0.0, 100.0);
        attitude.sorrow = attitude.sorrow.clamp(0.0, 100.0);
        attitude.disgust = attitude.disgust.clamp(0.0, 100.0);
        attitude.surprise = attitude.surprise.clamp(0.0, 100.0);
        attitude.curiosity = attitude.curiosity.clamp(0.0, 100.0);
        attitude.respect = attitude.respect.clamp(0.0, 100.0);
        attitude.suspicion = attitude.suspicion.clamp(0.0, 100.0);
        attitude.gratitude = attitude.gratitude.clamp(0.0, 100.0);
        attitude.jealousy = attitude.jealousy.clamp(0.0, 100.0);
        attitude.empathy = attitude.empathy.clamp(0.0, 100.0);
        attitude.lust = attitude.lust.clamp(0.0, 100.0);
        attitude.love = attitude.love.clamp(0.0, 100.0);
        attitude.anxiety = attitude.anxiety.clamp(0.0, 100.0);
        attitude.butterflies = attitude.butterflies.clamp(0.0, 100.0);
        attitude.submissiveness = attitude.submissiveness.clamp(0.0, 100.0);
        attitude.dominance = attitude.dominance.clamp(0.0, 100.0);

        attitude
    }

    pub fn create_or_update_third_party(
        name: &str,
        initial_data: Option<ThirdPartyIndividual>,
    ) -> Result<i32> {
        let con = Self::open()?;
        let current_time = get_current_date();

        let existing_id: Option<i32> = con
            .query_row(
                "SELECT id FROM third_party_individuals WHERE name = ?",
                [name],
                |row| row.get(0),
            )
            .ok();

        if let Some(id) = existing_id {
            if let Some(data) = initial_data {
                con.execute(
                    "UPDATE third_party_individuals SET
                        relationship_to_user = COALESCE(?, relationship_to_user),
                        relationship_to_companion = COALESCE(?, relationship_to_companion),
                        occupation = COALESCE(?, occupation),
                        personality_traits = COALESCE(?, personality_traits),
                        physical_description = COALESCE(?, physical_description),
                        last_mentioned = ?,
                        mention_count = mention_count + 1,
                        updated_at = ?
                    WHERE id = ?",
                    params![
                        data.relationship_to_user,
                        data.relationship_to_companion,
                        data.occupation,
                        data.personality_traits,
                        data.physical_description,
                        Some(current_time.clone()),
                        Some(current_time),
                        id
                    ],
                )?;
            } else {
                con.execute(
                    "UPDATE third_party_individuals SET
                        last_mentioned = ?, mention_count = mention_count + 1, updated_at = ?
                    WHERE id = ?",
                    params![&current_time, &current_time, &id],
                )?;
            }
            Ok(id)
        } else {
            let data = initial_data.unwrap_or(ThirdPartyIndividual {
                id: None,
                name: name.to_string(),
                relationship_to_user: None,
                relationship_to_companion: None,
                occupation: None,
                personality_traits: None,
                physical_description: None,
                first_mentioned: current_time.clone(),
                last_mentioned: None,
                mention_count: 1,
                importance_score: 0.5,
                created_at: current_time.clone(),
                updated_at: current_time.clone(),
                source: PersonSource::Heuristic,
            });

            con.execute(
                "INSERT INTO third_party_individuals (
                    name, relationship_to_user, relationship_to_companion, occupation,
                    personality_traits, physical_description, first_mentioned,
                    mention_count, importance_score, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    data.name,
                    data.relationship_to_user
                        .as_ref()
                        .unwrap_or(&"".to_string()),
                    data.relationship_to_companion
                        .as_ref()
                        .unwrap_or(&"".to_string()),
                    data.occupation,
                    data.personality_traits,
                    data.physical_description,
                    data.first_mentioned,
                    data.mention_count,
                    data.importance_score,
                    data.created_at,
                    data.updated_at
                ],
            )?;
            Ok(con.last_insert_rowid() as i32)
        }
    }

    /// Creates or updates a `third_party_individuals` row from a
    /// canon-validated `Person` fact (#177), always leaving `source =
    /// 'compaction'` — a canon-validated fact overrides a heuristic origin,
    /// so the row becomes trusted even if a heuristic row already existed
    /// under a different case (`alice` vs `Alice`; the lookup below is
    /// `COLLATE NOCASE`, unlike the `UNIQUE` index on `name`, which is
    /// case-sensitive and would not have merged the two on its own).
    pub fn upsert_compaction_person(p: &PersonUpsert) -> Result<i32> {
        let con = Self::open()?;
        Self::upsert_compaction_person_in(&con, p)
    }

    /// The connection-taking half of [`Database::upsert_compaction_person`],
    /// split out so tests can run it against `Database::open_at(tempdir)`.
    ///
    /// The name lookup and the insert/update run inside one `IMMEDIATE`
    /// transaction (started before the read, matching
    /// `migrate_messages_speaker_id`'s and `apply_attitude_deltas`'s shape),
    /// so "no row named this, case-insensitively" is a premise this
    /// transaction's own write lock enforces, not a fact read on one
    /// autocommit statement and assumed still true on the next. Without it,
    /// two commits racing to upsert the same newly-introduced person could
    /// both see "no existing row" and both `INSERT`, colliding on the
    /// `UNIQUE(name)` index for an exact-case duplicate or silently
    /// duplicating a differently-cased one.
    pub(crate) fn upsert_compaction_person_in(con: &Connection, p: &PersonUpsert) -> Result<i32> {
        let tx = Transaction::new_unchecked(con, TransactionBehavior::Immediate)?;
        let current_time = get_current_date();

        let existing: Option<(i32, i32)> = tx
            .query_row(
                "SELECT id, mention_count FROM third_party_individuals WHERE name = ? COLLATE NOCASE",
                [&p.name],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        let result_id = if let Some((id, existing_mentions)) = existing {
            let total_mentions = existing_mentions + p.mentions;
            let importance = crate::compaction::persons::importance_from_mentions(total_mentions);
            tx.execute(
                "UPDATE third_party_individuals SET
                    relationship_to_user = COALESCE(?, relationship_to_user),
                    relationship_to_companion = COALESCE(?, relationship_to_companion),
                    mention_count = mention_count + ?,
                    importance_score = ?,
                    last_mentioned = ?,
                    updated_at = ?,
                    source = 'compaction'
                WHERE id = ?",
                params![
                    p.relationship_to_user,
                    p.relationship_to_companion,
                    p.mentions,
                    importance,
                    current_time,
                    current_time,
                    id
                ],
            )?;
            id
        } else {
            let importance = crate::compaction::persons::importance_from_mentions(p.mentions);
            tx.execute(
                "INSERT INTO third_party_individuals (
                    name, relationship_to_user, relationship_to_companion,
                    first_mentioned, mention_count, importance_score,
                    created_at, updated_at, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'compaction')",
                params![
                    p.name,
                    p.relationship_to_user.as_deref().unwrap_or(""),
                    p.relationship_to_companion.as_deref().unwrap_or(""),
                    current_time,
                    p.mentions,
                    importance,
                    current_time,
                    current_time,
                ],
            )?;
            tx.last_insert_rowid() as i32
        };

        tx.commit()?;
        Ok(result_id)
    }

    /// Deletes one `third_party_individuals` row and everything that
    /// references it (attitude, memories), shared by
    /// [`Database::cleanup_invalid_third_parties_in`] (which used to repeat
    /// this three-statement delete twice) and
    /// [`Database::delete_compaction_persons_in`].
    fn delete_third_party_in(con: &Connection, id: i32) -> Result<()> {
        con.execute(
            "DELETE FROM companion_attitudes WHERE target_id = ? AND target_type = 'third_party'",
            params![id],
        )?;
        con.execute(
            "DELETE FROM third_party_memories WHERE third_party_id = ?",
            params![id],
        )?;
        con.execute(
            "DELETE FROM third_party_individuals WHERE id = ?",
            params![id],
        )?;
        Ok(())
    }

    /// Removes every compaction-sourced person. [`Database::erase_messages`]
    /// calls [`Database::delete_compaction_persons_in`] directly (on its own
    /// already-open connection) rather than this wrapper; kept `pub` for API
    /// symmetry with every other `Database` associated function, the same
    /// reason `get_third_party_memories` below keeps `#[allow(dead_code)]`
    /// rather than being removed.
    #[allow(dead_code)]
    pub fn delete_compaction_persons() -> Result<usize> {
        let con = Self::open()?;
        Self::delete_compaction_persons_in(&con)
    }

    /// The connection-taking half of [`Database::delete_compaction_persons`],
    /// split out so tests can run it against `Database::open_at(tempdir)`
    /// and so [`Database::erase_messages`] can call it on the same
    /// connection as the rest of a clear-chat operation.
    pub(crate) fn delete_compaction_persons_in(con: &Connection) -> Result<usize> {
        let ids: Vec<i32> = con
            .prepare("SELECT id FROM third_party_individuals WHERE source = 'compaction'")?
            .query_map([], |row| row.get::<_, i32>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        for id in &ids {
            Self::delete_third_party_in(con, *id)?;
        }

        Ok(ids.len())
    }

    pub fn add_third_party_memory(
        third_party_id: i32,
        companion_id: i32,
        memory: &ThirdPartyMemory,
    ) -> Result<i32> {
        let con = Self::open()?;
        let current_time = get_current_date();

        con.execute(
            "INSERT INTO third_party_memories (
                third_party_id, companion_id, memory_type, content,
                importance, emotional_valence, created_at, context_message_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                third_party_id,
                companion_id,
                memory.memory_type,
                memory.content,
                memory.importance,
                memory.emotional_valence,
                current_time,
                memory.context_message_id
            ],
        )?;

        Ok(con.last_insert_rowid() as i32)
    }

    pub fn plan_third_party_interaction(interaction: &ThirdPartyInteraction) -> Result<i32> {
        let con = Self::open()?;
        let current_time = get_current_date();

        con.execute(
            "INSERT INTO third_party_interactions (
                third_party_id, companion_id, interaction_type, description,
                planned_date, impact_on_relationship, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                interaction.third_party_id,
                interaction.companion_id,
                interaction.interaction_type,
                interaction.description,
                interaction.planned_date,
                interaction.impact_on_relationship,
                current_time,
                current_time
            ],
        )?;

        Ok(con.last_insert_rowid() as i32)
    }

    pub fn get_planned_interactions(
        companion_id: i32,
        limit: Option<usize>,
    ) -> Result<Vec<ThirdPartyInteraction>> {
        let con = Self::open()?;
        let query = if let Some(limit) = limit {
            format!(
                "SELECT id, third_party_id, companion_id, interaction_type, description,
                        planned_date, actual_date, outcome, impact_on_relationship,
                        created_at, updated_at
                 FROM third_party_interactions
                 WHERE companion_id = ? AND interaction_type = 'planned'
                 ORDER BY planned_date ASC
                 LIMIT {}",
                limit
            )
        } else {
            "SELECT id, third_party_id, companion_id, interaction_type, description,
                    planned_date, actual_date, outcome, impact_on_relationship,
                    created_at, updated_at
             FROM third_party_interactions
             WHERE companion_id = ? AND interaction_type = 'planned'
             ORDER BY planned_date ASC"
                .to_string()
        };

        let mut stmt = con.prepare(&query)?;
        let interactions = stmt.query_map([&companion_id], |row| {
            Ok(ThirdPartyInteraction {
                id: Some(row.get(0)?),
                third_party_id: row.get(1)?,
                companion_id: row.get(2)?,
                interaction_type: row.get(3)?,
                description: row.get(4)?,
                planned_date: row.get(5)?,
                actual_date: row.get(6)?,
                outcome: row.get(7)?,
                impact_on_relationship: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
            })
        })?;

        let mut result = Vec::new();
        for interaction in interactions {
            result.push(interaction?);
        }

        Ok(result)
    }

    pub fn complete_interaction(interaction_id: i32, outcome: &str, impact: f32) -> Result<()> {
        let con = Self::open()?;
        let current_time = get_current_date();

        con.execute(
            "UPDATE third_party_interactions
             SET interaction_type = 'completed',
                 actual_date = ?,
                 outcome = ?,
                 impact_on_relationship = ?,
                 updated_at = ?
             WHERE id = ?",
            params![current_time, outcome, impact, current_time, interaction_id],
        )?;

        Ok(())
    }

    pub fn get_interaction_history(
        companion_id: i32,
        third_party_id: i32,
    ) -> Result<Vec<ThirdPartyInteraction>> {
        let con = Self::open()?;
        let mut stmt = con.prepare(
            "SELECT id, third_party_id, companion_id, interaction_type, description,
                    planned_date, actual_date, outcome, impact_on_relationship,
                    created_at, updated_at
             FROM third_party_interactions
             WHERE companion_id = ? AND third_party_id = ?
             ORDER BY COALESCE(actual_date, planned_date) DESC",
        )?;

        let interactions = stmt.query_map(params![companion_id, third_party_id], |row| {
            Ok(ThirdPartyInteraction {
                id: Some(row.get(0)?),
                third_party_id: row.get(1)?,
                companion_id: row.get(2)?,
                interaction_type: row.get(3)?,
                description: row.get(4)?,
                planned_date: row.get(5)?,
                actual_date: row.get(6)?,
                outcome: row.get(7)?,
                impact_on_relationship: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
            })
        })?;

        let mut result = Vec::new();
        for interaction in interactions {
            result.push(interaction?);
        }

        Ok(result)
    }

    pub fn get_third_party_by_name(name: &str) -> Result<Option<ThirdPartyIndividual>> {
        let con = Self::open()?;
        let mut stmt = con.prepare(
            "SELECT id, name, relationship_to_user, relationship_to_companion, occupation,
                    personality_traits, physical_description, first_mentioned, last_mentioned,
                    mention_count, importance_score, created_at, updated_at, source
             FROM third_party_individuals WHERE name = ?",
        )?;

        let individual = stmt
            .query_row([name], |row| {
                Ok(ThirdPartyIndividual {
                    id: Some(row.get(0)?),
                    name: row.get(1)?,
                    relationship_to_user: row.get(2)?,
                    relationship_to_companion: row.get(3)?,
                    occupation: row.get(4)?,
                    personality_traits: row.get(5)?,
                    physical_description: row.get(6)?,
                    first_mentioned: row.get(7)?,
                    last_mentioned: row.get(8)?,
                    mention_count: row.get(9)?,
                    importance_score: row.get(10)?,
                    created_at: row.get(11)?,
                    updated_at: row.get(12)?,
                    source: row.get(13)?,
                })
            })
            .ok();

        Ok(individual)
    }

    pub fn get_all_third_party_individuals() -> Result<Vec<ThirdPartyIndividual>> {
        let con = Self::open()?;
        let mut stmt = con.prepare(
            "SELECT id, name, relationship_to_user, relationship_to_companion, occupation,
                    personality_traits, physical_description, first_mentioned, last_mentioned,
                    mention_count, importance_score, created_at, updated_at, source
             FROM third_party_individuals
             ORDER BY importance_score DESC, mention_count DESC",
        )?;

        let individuals = stmt.query_map([], |row| {
            Ok(ThirdPartyIndividual {
                id: Some(row.get(0)?),
                name: row.get(1)?,
                relationship_to_user: row.get(2)?,
                relationship_to_companion: row.get(3)?,
                occupation: row.get(4)?,
                personality_traits: row.get(5)?,
                physical_description: row.get(6)?,
                first_mentioned: row.get(7)?,
                last_mentioned: row.get(8)?,
                mention_count: row.get(9)?,
                importance_score: row.get(10)?,
                created_at: row.get(11)?,
                updated_at: row.get(12)?,
                source: row.get(13)?,
            })
        })?;

        let mut result = Vec::new();
        for individual in individuals {
            result.push(individual?);
        }

        Ok(result)
    }

    #[allow(dead_code)]
    pub fn get_third_party_memories(
        third_party_id: i32,
        limit: Option<usize>,
    ) -> Result<Vec<ThirdPartyMemory>> {
        let con = Self::open()?;
        let query = if let Some(limit) = limit {
            format!(
                "SELECT id, third_party_id, companion_id, memory_type, content,
                        importance, emotional_valence, created_at, context_message_id
                 FROM third_party_memories
                 WHERE third_party_id = ?
                 ORDER BY importance DESC, created_at DESC
                 LIMIT {}",
                limit
            )
        } else {
            "SELECT id, third_party_id, companion_id, memory_type, content,
                    importance, emotional_valence, created_at, context_message_id
             FROM third_party_memories
             WHERE third_party_id = ?
             ORDER BY importance DESC, created_at DESC"
                .to_string()
        };

        let mut stmt = con.prepare(&query)?;
        let memories = stmt.query_map([&third_party_id], |row| {
            Ok(ThirdPartyMemory {
                id: Some(row.get(0)?),
                third_party_id: row.get(1)?,
                companion_id: row.get(2)?,
                memory_type: row.get(3)?,
                content: row.get(4)?,
                importance: row.get(5)?,
                emotional_valence: row.get(6)?,
                created_at: row.get(7)?,
                context_message_id: row.get(8)?,
            })
        })?;

        let mut result = Vec::new();
        for memory in memories {
            result.push(memory?);
        }

        Ok(result)
    }

    #[allow(dead_code)]
    pub fn update_third_party_importance(third_party_id: i32, new_importance: f32) -> Result<()> {
        let con = Self::open()?;
        let current_time = get_current_date();

        con.execute(
            "UPDATE third_party_individuals
             SET importance_score = ?, updated_at = ?
             WHERE id = ?",
            params![&new_importance, &current_time, &third_party_id],
        )?;

        Ok(())
    }

    // Attitude Change Detection System

    pub fn create_attitude_memories_table(con: &Connection) -> Result<()> {
        con.execute(&attitude_memories_ddl("attitude_memories"), [])?;

        // Create index for priority queries
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_attitude_memories_priority
             ON attitude_memories(companion_id, priority_score DESC)",
            [],
        )?;

        Ok(())
    }

    /// Rebuilds `attitude_memories` if it still carries the old `companions`
    /// (nonexistent) foreign key target, so `PRAGMA foreign_keys=ON` does not
    /// break inserts against databases created before that typo was fixed.
    /// Fresh and already-migrated databases detect a match on `companion` and
    /// return immediately.
    ///
    /// Precondition: no transaction may be open on `con`. `PRAGMA
    /// foreign_keys` is silently ignored while a transaction is open, so this
    /// must run right after the connection is opened.
    pub fn migrate_attitude_memories_foreign_key(con: &Connection) -> Result<()> {
        debug_assert!(con.is_autocommit());

        let mut stmt = con.prepare("PRAGMA foreign_key_list(attitude_memories)")?;
        let targets = stmt
            .query_map([], |row| row.get::<_, String>(2))?
            .collect::<Result<Vec<_>>>()?;
        drop(stmt);

        if targets.iter().all(|table| table == "companion") {
            return Ok(());
        }

        // Must precede BEGIN: SQLite ignores PRAGMA foreign_keys inside a
        // transaction, and turning it off first stops DROP TABLE below from
        // running an implicit FK-checked DELETE.
        con.pragma_update(None, "foreign_keys", false)?;

        let tx = con.unchecked_transaction()?;
        tx.execute(&attitude_memories_ddl("attitude_memories_new"), [])?;
        tx.execute(
            "INSERT INTO attitude_memories_new (
                id, companion_id, target_id, target_type, memory_type, description,
                priority_score, attitude_delta_json, impact_score, message_context, created_at
            )
            SELECT id, companion_id, target_id, target_type, memory_type, description,
                   priority_score, attitude_delta_json, impact_score, message_context, created_at
            FROM attitude_memories",
            [],
        )?;
        tx.execute("DROP TABLE attitude_memories", [])?;
        tx.execute(
            "ALTER TABLE attitude_memories_new RENAME TO attitude_memories",
            [],
        )?;
        tx.execute(
            "CREATE INDEX IF NOT EXISTS idx_attitude_memories_priority
             ON attitude_memories(companion_id, priority_score DESC)",
            [],
        )?;

        let orphan_count = {
            let mut check_stmt = tx.prepare("PRAGMA foreign_key_check(attitude_memories)")?;
            let rows = check_stmt.query_map([], |_| Ok(()))?;
            rows.count()
        };
        if orphan_count > 0 {
            println!(
                "warning: attitude_memories has {orphan_count} row(s) whose \
                 companion_id no longer matches a companion; keeping them as-is"
            );
        }

        tx.commit()?;
        con.pragma_update(None, "foreign_keys", true)?;

        Ok(())
    }

    /// Persists an attitude shift as a memory when it is significant enough.
    ///
    /// Detection itself lives in `evaluate_attitude_shift`; this only calls
    /// `insert_attitude_memory` when a shift clears the threshold. Callers
    /// pass the whole turn's before/after pair, so one turn produces at most
    /// one memory. `message_context` should be an excerpt of what the user
    /// said, so the memory records why the feelings moved.
    pub fn detect_attitude_change(
        companion_id: i32,
        target_id: i32,
        target_type: &str,
        previous_attitude: &CompanionAttitude,
        new_attitude: &CompanionAttitude,
        message_context: Option<&str>,
    ) -> Result<()> {
        let Some(draft) = evaluate_attitude_shift(previous_attitude, new_attitude) else {
            return Ok(());
        };

        Database::insert_attitude_memory(
            companion_id,
            target_id,
            target_type,
            &draft,
            message_context.unwrap_or(""),
        )
    }

    /// Writes one attitude-shift memory row and prunes back to
    /// `MAX_ATTITUDE_MEMORIES_PER_COMPANION`. The single writer both
    /// `detect_attitude_change` (lexicon-scored turns, gated by
    /// `evaluate_attitude_shift`'s significance threshold) and
    /// `compaction::attitude::AttitudeRecalibrator` (narrative recalibration
    /// at commit, no gate) call.
    ///
    /// # Errors
    /// Returns `rusqlite::Error::SqliteFailure` on a foreign-key violation
    /// (`companion_id` names no `companion` row, with `PRAGMA foreign_keys`
    /// on) or any other statement open/execute failure.
    pub fn insert_attitude_memory(
        companion_id: i32,
        target_id: i32,
        target_type: &str,
        draft: &AttitudeMemoryDraft,
        message_context: &str,
    ) -> Result<()> {
        let con = Self::open()?;
        Self::insert_attitude_memory_on(
            &con,
            companion_id,
            target_id,
            target_type,
            draft,
            message_context,
        )?;

        // The only writers maintain the bound, so the table cannot grow
        // without limit however many turns are played or checkpoints
        // committed.
        Database::prune_attitude_memories(companion_id, MAX_ATTITUDE_MEMORIES_PER_COMPANION)?;

        Ok(())
    }

    /// Testable half of `insert_attitude_memory`, taking a caller-provided
    /// connection so tests can point it at a `TempDir`-backed database
    /// instead of the hardwired `paths::db_path()`. Does not prune: tests
    /// exercise that separately via `prune_attitude_memories_on`.
    fn insert_attitude_memory_on(
        con: &Connection,
        companion_id: i32,
        target_id: i32,
        target_type: &str,
        draft: &AttitudeMemoryDraft,
        message_context: &str,
    ) -> Result<()> {
        let attitude_delta_json = serde_json::to_string(&draft.delta).unwrap_or_default();
        let current_time = get_current_date();

        con.execute(
            "INSERT INTO attitude_memories (
                companion_id, target_id, target_type, memory_type, description,
                priority_score, attitude_delta_json, impact_score, message_context, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                companion_id,
                target_id,
                target_type,
                draft.memory_type,
                draft.description,
                draft.priority_score,
                attitude_delta_json,
                draft.impact_score,
                message_context,
                current_time
            ],
        )?;

        Ok(())
    }

    /// Drops all but the `keep` highest-priority memories for one companion.
    ///
    /// Ties on `priority_score` break on `id`, not `created_at`: the latter
    /// holds `get_current_date()` strings ("Friday 05.09.2026 14:30"), which
    /// sort by weekday name rather than chronologically. The rowid reflects
    /// insert order, so it is the real recency tie-break.
    ///
    /// Returns the number of rows deleted.
    pub fn prune_attitude_memories(companion_id: i32, keep: usize) -> Result<usize> {
        let con = Self::open()?;
        Self::prune_attitude_memories_on(&con, companion_id, keep)
    }

    /// Testable half of `prune_attitude_memories`, taking a caller-provided
    /// connection so tests can point it at a `TempDir`-backed database
    /// instead of the hardwired `paths::db_path()`.
    fn prune_attitude_memories_on(
        con: &Connection,
        companion_id: i32,
        keep: usize,
    ) -> Result<usize> {
        con.execute(
            "DELETE FROM attitude_memories
             WHERE companion_id = ?1
               AND id NOT IN (
                   SELECT id FROM attitude_memories
                   WHERE companion_id = ?1
                   ORDER BY priority_score DESC, id DESC
                   LIMIT ?2
               )",
            params![companion_id, keep],
        )
    }

    /// Highest-priority memories for one companion, across every target type.
    pub fn get_priority_attitude_memories(
        companion_id: i32,
        limit: usize,
    ) -> Result<Vec<AttitudeMemory>> {
        Self::query_priority_attitude_memories(companion_id, None, limit)
    }

    /// Highest-priority memories for one companion and one target type.
    ///
    /// The filter belongs in the query rather than after it: limiting across
    /// every target type first lets third-party rows occupy all the slots and
    /// starve out user memories that would have qualified.
    pub fn get_priority_attitude_memories_for_target(
        companion_id: i32,
        target_type: &str,
        limit: usize,
    ) -> Result<Vec<AttitudeMemory>> {
        Self::query_priority_attitude_memories(companion_id, Some(target_type), limit)
    }

    fn query_priority_attitude_memories(
        companion_id: i32,
        target_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<AttitudeMemory>> {
        let con = Self::open()?;
        let base = "SELECT id, companion_id, target_id, target_type, memory_type, description,
                    priority_score, attitude_delta_json, impact_score, message_context, created_at
             FROM attitude_memories
             WHERE companion_id = ?";
        let query = match target_type {
            Some(_) => format!("{} AND target_type = ?\n             ORDER BY priority_score DESC, id DESC\n             LIMIT ?", base),
            None => format!("{}\n             ORDER BY priority_score DESC, id DESC\n             LIMIT ?", base),
        };
        let mut stmt = con.prepare(&query)?;

        let row_to_memory = |row: &rusqlite::Row| {
            Ok(AttitudeMemory {
                id: row.get(0)?,
                companion_id: row.get(1)?,
                target_id: row.get(2)?,
                target_type: row.get(3)?,
                memory_type: row.get(4)?,
                description: row.get(5)?,
                priority_score: row.get(6)?,
                attitude_delta_json: row.get(7)?,
                impact_score: row.get(8)?,
                message_context: row.get(9)?,
                created_at: row.get(10)?,
            })
        };

        let memories = match target_type {
            Some(target_type) => {
                stmt.query_map(params![companion_id, target_type, limit], row_to_memory)?
            }
            None => stmt.query_map(params![companion_id, limit], row_to_memory)?,
        };

        let mut result = Vec::new();
        for memory in memories {
            result.push(memory?);
        }

        Ok(result)
    }

    // Automatic Person Detection System

    pub fn detect_new_persons_in_message(
        message: &str,
        companion_id: i32,
        excluded_names: &[String],
    ) -> Result<Vec<i32>> {
        let detected_names =
            Database::drop_excluded_names(Database::extract_person_names(message), excluded_names);
        let mut new_person_ids = Vec::new();

        // Get user name to filter it out from third party detection
        let user_name = match Database::get_user_data() {
            Ok(user) => Some(user.name.to_lowercase()),
            Err(_) => None,
        };

        for name in detected_names {
            // Skip if this is the user's own name
            if let Some(ref user_name) = user_name {
                if name.to_lowercase() == *user_name {
                    continue;
                }
            }

            // Check if person already exists
            if Database::get_third_party_by_name(&name)?.is_none() {
                // Create new third-party individual with context-based initial data
                let initial_data = Database::analyze_context_for_person(&name, message);
                let person_id = Database::create_or_update_third_party(&name, Some(initial_data))?;

                // Initialize attitude tracking with context-based values
                let mut initial_attitude =
                    Database::generate_initial_attitudes(&name, message, companion_id);
                initial_attitude.target_id = person_id;
                Database::create_or_update_attitude(
                    companion_id,
                    person_id,
                    "third_party",
                    &initial_attitude,
                )?;

                new_person_ids.push(person_id);

                // Add initial memory about this person
                let memory = ThirdPartyMemory {
                    id: None,
                    third_party_id: person_id,
                    companion_id,
                    memory_type: "fact".to_string(),
                    content: format!("First mentioned: {}", message.trim()),
                    importance: 0.6,
                    emotional_valence: 0.0,
                    created_at: get_current_date(),
                    context_message_id: None,
                };
                Database::add_third_party_memory(person_id, companion_id, &memory)?;
            } else {
                // Update mention count for existing person
                Database::create_or_update_third_party(&name, None)?;
            }
        }

        Ok(new_person_ids)
    }

    pub fn cleanup_duplicate_third_parties() -> Result<i32> {
        let con = Self::open()?;
        let mut cleaned_count = 0;

        // Find all duplicate names (case-insensitive)
        let mut stmt = con.prepare(
            "
            SELECT LOWER(name) as lower_name, COUNT(*) as count
            FROM third_party_individuals
            GROUP BY LOWER(name)
            HAVING COUNT(*) > 1
        ",
        )?;

        let duplicate_names: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        for lower_name in duplicate_names {
            // Get all instances of this name
            let mut instances_stmt = con.prepare(
                "
                SELECT id, name, relationship_to_user, relationship_to_companion, occupation,
                       personality_traits, physical_description, first_mentioned, last_mentioned,
                       mention_count, importance_score, created_at, updated_at, source
                FROM third_party_individuals
                WHERE LOWER(name) = ?
                ORDER BY created_at ASC
            ",
            )?;

            let instances: Vec<ThirdPartyIndividual> = instances_stmt
                .query_map([&lower_name], |row| {
                    Ok(ThirdPartyIndividual {
                        id: Some(row.get(0)?),
                        name: row.get(1)?,
                        relationship_to_user: row.get(2)?,
                        relationship_to_companion: row.get(3)?,
                        occupation: row.get(4)?,
                        personality_traits: row.get(5)?,
                        physical_description: row.get(6)?,
                        first_mentioned: row.get(7)?,
                        last_mentioned: row.get(8)?,
                        mention_count: row.get(9)?,
                        importance_score: row.get(10)?,
                        created_at: row.get(11)?,
                        updated_at: row.get(12)?,
                        source: row.get(13)?,
                    })
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            if instances.len() > 1 {
                // Keep the first instance, merge data from others
                let keep_id = instances[0].id.unwrap();
                let mut total_mentions = 0;
                let mut max_importance = 0.0;
                let mut earliest_first_mentioned = instances[0].first_mentioned.clone();
                let mut latest_last_mentioned = instances[0].last_mentioned.clone();

                // Collect data from all instances
                for instance in &instances {
                    total_mentions += instance.mention_count;
                    if instance.importance_score > max_importance {
                        max_importance = instance.importance_score;
                    }
                    if instance.first_mentioned < earliest_first_mentioned {
                        earliest_first_mentioned = instance.first_mentioned.clone();
                    }
                    if let Some(ref last) = instance.last_mentioned {
                        if latest_last_mentioned.is_none()
                            || last > latest_last_mentioned.as_ref().unwrap()
                        {
                            latest_last_mentioned = Some(last.clone());
                        }
                    }
                }

                // Update the kept instance with merged data
                con.execute(
                    "
                    UPDATE third_party_individuals SET
                        mention_count = ?,
                        importance_score = ?,
                        first_mentioned = ?,
                        last_mentioned = ?,
                        updated_at = ?
                    WHERE id = ?
                ",
                    params![
                        total_mentions,
                        max_importance,
                        earliest_first_mentioned,
                        latest_last_mentioned,
                        get_current_date(),
                        keep_id
                    ],
                )?;

                // Update attitudes to point to the kept instance
                for instance in &instances[1..] {
                    if let Some(delete_id) = instance.id {
                        con.execute(
                            "
                            UPDATE companion_attitudes SET target_id = ?
                            WHERE target_id = ? AND target_type = 'third_party'
                        ",
                            params![keep_id, delete_id],
                        )?;

                        // Update memories to point to the kept instance
                        con.execute(
                            "
                            UPDATE third_party_memories SET third_party_id = ?
                            WHERE third_party_id = ?
                        ",
                            params![keep_id, delete_id],
                        )?;

                        // Delete the duplicate instance
                        con.execute(
                            "DELETE FROM third_party_individuals WHERE id = ?",
                            [delete_id],
                        )?;
                        cleaned_count += 1;
                    }
                }
            }
        }

        Ok(cleaned_count)
    }

    pub fn cleanup_invalid_third_parties() -> Result<i32> {
        let con = Self::open()?;
        Self::cleanup_invalid_third_parties_in(&con)
    }

    /// The connection-taking half of [`Database::cleanup_invalid_third_parties`],
    /// split out so tests can run it against `Database::open_at(tempdir)`.
    ///
    /// Every candidate is restricted to `source = 'heuristic'`: a
    /// compaction-sourced row (#177) is canon-validated at extraction time
    /// and is never a candidate here, whatever its name — clearing those out
    /// is `Database::delete_compaction_persons`'s job, run from
    /// `erase_messages` instead.
    fn cleanup_invalid_third_parties_in(con: &Connection) -> Result<i32> {
        let mut cleaned_count = 0;

        // List of invalid names that should be removed
        let invalid_names = [
            // Body parts
            "hand",
            "hands",
            "shoulder",
            "shoulders",
            "head",
            "heads",
            "arm",
            "arms",
            "leg",
            "legs",
            "foot",
            "feet",
            "eye",
            "eyes",
            "ear",
            "ears",
            "nose",
            "mouth",
            "face",
            "hair",
            "neck",
            "back",
            "chest",
            "stomach",
            "knee",
            "knees",
            "elbow",
            "elbows",
            "finger",
            "fingers",
            "thumb",
            "thumbs",
            "toe",
            "toes",
            // Common objects
            "class",
            "classes",
            "book",
            "books",
            "table",
            "tables",
            "chair",
            "chairs",
            "door",
            "doors",
            "window",
            "windows",
            "desk",
            "desks",
            "computer",
            "computers",
            "phone",
            "phones",
            "car",
            "cars",
            "house",
            "houses",
            "room",
            "rooms",
            // Abstract concepts
            "should",
            "could",
            "would",
            "thing",
            "things",
            "stuff",
            "matter",
            "matters",
            "way",
            "ways",
            "time",
            "times",
            "place",
            "places",
            "work",
            "works",
            // Common verbs/actions
            "walk",
            "walks",
            "talk",
            "talks",
            "look",
            "looks",
            "feel",
            "feels",
            "want",
            "wants",
            "need",
            "needs",
            "use",
            "uses",
            "make",
            "makes",
        ];

        for invalid_name in &invalid_names {
            // Find and delete invalid third parties
            let mut stmt = con.prepare(
                "
                SELECT id FROM third_party_individuals
                WHERE LOWER(name) = LOWER(?) AND source = 'heuristic'
            ",
            )?;

            let ids: Vec<i32> = stmt
                .query_map([invalid_name], |row| row.get::<_, i32>(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            for id in ids {
                Self::delete_third_party_in(con, id)?;
                cleaned_count += 1;
                println!("Removed invalid third party: {} (id: {})", invalid_name, id);
            }
        }

        // Also check for entries that don't look like proper names, or are
        // junk (a pronoun, a stop word, or too short). #177 deliberately
        // does *not* also flag a bare `relationship_to_companion =
        // 'newly_mentioned'` with nothing else filled in: `analyze_context_for_person`
        // sets that value on every heuristically detected person
        // unconditionally, real or not, and `extract_occupation`/
        // `extract_personality_traits`/etc. commonly find nothing to fill
        // in even for a genuine person mentioned in passing — a rule keyed
        // on that combination would delete correctly detected people, not
        // just junk.
        let mut stmt =
            con.prepare("SELECT id, name FROM third_party_individuals WHERE source = 'heuristic'")?;

        let entries: Vec<(i32, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        for (id, name) in entries {
            let looks_wrong = !Database::is_likely_person_name(&name)
                || !name.chars().next().unwrap_or('a').is_uppercase();
            let is_junk = crate::compaction::persons::is_junk_person_name(&name);

            if looks_wrong || is_junk {
                Self::delete_third_party_in(con, id)?;
                cleaned_count += 1;
                println!("Removed invalid third party: {} (id: {})", name, id);
            }
        }

        if cleaned_count > 0 {
            println!("Cleaned up {} invalid third party entries", cleaned_count);
        } else {
            println!("No invalid third party entries found");
        }

        Ok(cleaned_count)
    }

    fn extract_person_names(text: &str) -> Vec<String> {
        let mut names = Vec::new();

        // Keep original text for proper name detection (with capitalization)
        let text_original = text;
        let _text_lower = text.to_lowercase();

        // More specific patterns for person references
        // Note: These patterns now focus on clearer indicators of person names
        let patterns = [
            // Family relationships with names
            r"(?i)(my|our|their|his|her) (friend|colleague|boss|manager|teacher|doctor|neighbor|brother|sister|mother|father|mom|dad|parent|cousin|uncle|aunt|grandmother|grandfather|grandma|grandpa) ([A-Z][a-z]+)",
            // Names with clear person indicators
            r"(?i)(talked to|spoke with|met|saw|visited|called|texted|emailed) ([A-Z][a-z]+)",
            r"(?i)([A-Z][a-z]+) (called|texted|emailed|visited|invited|asked|told|said)",
            // Professional titles with names
            r"(?i)(dr\.|mr\.|mrs\.|ms\.|prof\.|professor) ([A-Z][a-z]+)",
            // Names in possessive contexts
            r"(?i)([A-Z][a-z]+)'s (house|place|car|office|room|family|friend|work)",
            // Names with relationship descriptors
            r"(?i)(friend|colleague|neighbor) ([A-Z][a-z]+)",
            r"(?i)([A-Z][a-z]+) is my (friend|colleague|boss|teacher|doctor|neighbor)",
            // Proper names (capitalized) that appear independently
            // Only match if preceded/followed by clear context
            r"(?i)(with|and|or|met|saw|told|asked) ([A-Z][a-z]{2,})\b",
            r"\b([A-Z][a-z]{2,}) (and I|and me|said|told|asked|mentioned|arrived|left|came|went)",
        ];

        // Process patterns on original text to preserve capitalization
        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for cap in re.captures_iter(text_original) {
                    // Try to get the name from the capture group
                    // Usually it's the last capturing group
                    for i in (1..cap.len()).rev() {
                        if let Some(name_match) = cap.get(i) {
                            let potential_name = name_match.as_str().trim();

                            // Check if this looks like a proper name (starts with capital)
                            if !potential_name.is_empty()
                                && potential_name.chars().next().unwrap().is_uppercase()
                                && Database::is_likely_person_name(potential_name)
                                && Database::is_proper_name_context(potential_name, text_original)
                            {
                                names.push(potential_name.to_string());
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Also check for standalone capitalized words that are likely names
        // But only if they appear in a clear person context
        let words: Vec<&str> = text_original.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphabetic());

            // Check if it's a capitalized word
            if clean_word.len() > 2
                && clean_word.chars().next().unwrap().is_uppercase()
                && clean_word.chars().skip(1).all(|c| c.is_lowercase())
                && Database::is_likely_person_name(clean_word)
            {
                // Check surrounding context for person indicators
                let has_person_context = (i > 0
                    && Database::is_person_indicator(&words[i - 1].to_lowercase()))
                    || (i < words.len() - 1
                        && Database::is_person_indicator(&words[i + 1].to_lowercase()));

                if has_person_context {
                    names.push(clean_word.to_string());
                }
            }
        }

        // Remove duplicates and validate
        names.sort();
        names.dedup();
        names
            .into_iter()
            .filter(|name| {
                !Database::is_common_word(name) && name.chars().next().unwrap().is_uppercase()
            })
            .collect()
    }

    fn is_likely_person_name(name: &str) -> bool {
        let name_lower = name.to_lowercase();

        // Filter out common non-name words (shared with #177's
        // `compaction::persons::is_junk_person_name`, so the two lists can
        // never drift apart).
        let non_names = crate::compaction::persons::NON_NAME_WORDS;

        // Check if in non-names list
        if non_names.contains(&name_lower.as_str()) {
            return false;
        }

        // Filter out words with certain suffixes that are unlikely to be names
        if name_lower.ends_with("ing")
            || name_lower.ends_with("tion")
            || name_lower.ends_with("sion")
            || name_lower.ends_with("ness")
            || name_lower.ends_with("ment")
            || name_lower.ends_with("ity")
            || name_lower.ends_with("ance")
            || name_lower.ends_with("ence")
            || name_lower.ends_with("ship")
            || name_lower.ends_with("hood")
            || name_lower.ends_with("dom")
            || name_lower.ends_with("ism")
            || name_lower.ends_with("ist")
            || name_lower.ends_with("able")
            || name_lower.ends_with("ible")
            || name_lower.ends_with("ful")
            || name_lower.ends_with("less")
            || name_lower.ends_with("ous")
            || name_lower.ends_with("ive")
            || name_lower.ends_with("ly")
        {
            return false;
        }

        // Basic validation: length and character checks
        name.len() > 2
            && name.len() < 20  // Most names are shorter than 20 characters
            && name.chars().all(|c| c.is_alphabetic() || c == '\'' || c == '-')
    }

    fn is_common_word(name: &str) -> bool {
        let common_words = [
            "User",
            "Assistant",
            "System",
            "Admin",
            "Anonymous",
            "Guest",
            "Bot",
            "AI",
            "Computer",
            "Machine",
            "Program",
            "Software",
            "App",
            "Website",
        ];
        common_words.contains(&name)
    }

    fn capitalize_name(name: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = true;

        for c in name.chars() {
            if c.is_alphabetic() {
                if capitalize_next {
                    result.push(c.to_uppercase().next().unwrap_or(c));
                    capitalize_next = false;
                } else {
                    result.push(c.to_lowercase().next().unwrap_or(c));
                }
            } else {
                result.push(c);
                if c == ' ' || c == '-' || c == '\'' {
                    capitalize_next = true;
                }
            }
        }

        result
    }

    fn is_proper_name_context(name: &str, text: &str) -> bool {
        // Check if the name appears in a context that suggests it's a person
        // This helps filter out words that might be capitalized for other reasons

        let name_lower = name.to_lowercase();
        let text_lower = text.to_lowercase();

        // Check for possessive forms
        if text.contains(&format!("{}'s", name)) || text.contains(&format!("{}' ", name)) {
            return true;
        }

        // Check for titles before the name
        let titles = ["mr.", "mrs.", "ms.", "dr.", "prof.", "professor"];
        for title in &titles {
            if text_lower.contains(&format!("{} {}", title, name_lower)) {
                return true;
            }
        }

        // Check for person-related verbs around the name
        let person_verbs = [
            "said", "told", "asked", "called", "visited", "met", "saw", "knows", "likes",
        ];
        for verb in &person_verbs {
            if text_lower.contains(&format!("{} {}", name_lower, verb))
                || text_lower.contains(&format!("{} {}", verb, name_lower))
            {
                return true;
            }
        }

        // If none of the above, be conservative
        true // We'll rely on other filters to catch non-names
    }

    fn is_person_indicator(word: &str) -> bool {
        // Words that often appear before or after person names
        let indicators = [
            "with",
            "and",
            "met",
            "saw",
            "told",
            "asked",
            "called",
            "visited",
            "friend",
            "colleague",
            "neighbor",
            "brother",
            "sister",
            "mother",
            "father",
            "uncle",
            "aunt",
            "cousin",
            "boss",
            "teacher",
            "doctor",
            "said",
            "says",
            "thinks",
            "believes",
            "wants",
            "needs",
            "likes",
            "loves",
            "hates",
        ];

        indicators.contains(&word.trim_matches(|c: char| !c.is_alphabetic()))
    }

    fn analyze_context_for_person(name: &str, message: &str) -> ThirdPartyIndividual {
        let current_time = get_current_date();
        let relationship_to_user = Database::extract_relationship_to_user(name, message);
        let occupation = Database::extract_occupation(name, message);
        let personality_traits = Database::extract_personality_traits(name, message);

        let importance_score = Database::calculate_person_importance(name, message);

        ThirdPartyIndividual {
            id: None,
            name: name.to_string(),
            relationship_to_user,
            relationship_to_companion: Some("newly_mentioned".to_string()),
            occupation,
            personality_traits,
            physical_description: None,
            first_mentioned: current_time.clone(),
            last_mentioned: None,
            mention_count: 1,
            importance_score,
            created_at: current_time.clone(),
            updated_at: current_time,
            source: PersonSource::Heuristic,
        }
    }

    fn extract_relationship_to_user(name: &str, message: &str) -> Option<String> {
        let text = message.to_lowercase();
        let name_lower = name.to_lowercase();

        // Look for relationship keywords near the name
        let relationships = [
            ("friend", "friend"),
            ("best friend", "best friend"),
            ("colleague", "colleague"),
            ("coworker", "colleague"),
            ("boss", "boss"),
            ("manager", "manager"),
            ("teacher", "teacher"),
            ("professor", "teacher"),
            ("doctor", "doctor"),
            ("neighbor", "neighbor"),
            ("brother", "brother"),
            ("sister", "sister"),
            ("mother", "mother"),
            ("father", "father"),
            ("mom", "mother"),
            ("dad", "father"),
            ("parent", "parent"),
            ("cousin", "cousin"),
            ("uncle", "uncle"),
            ("aunt", "aunt"),
            ("boyfriend", "boyfriend"),
            ("girlfriend", "girlfriend"),
            ("partner", "partner"),
            ("spouse", "spouse"),
            ("husband", "husband"),
            ("wife", "wife"),
        ];

        for (keyword, relationship) in &relationships {
            if text.contains(&format!("my {} {}", keyword, name_lower))
                || text.contains(&format!("{} is my {}", name_lower, keyword))
                || text.contains(&format!("my {}", keyword))
            {
                return Some(relationship.to_string());
            }
        }

        None
    }

    fn extract_occupation(name: &str, message: &str) -> Option<String> {
        let text = message.to_lowercase();
        let name_lower = name.to_lowercase();

        let occupations = [
            "doctor",
            "teacher",
            "engineer",
            "lawyer",
            "nurse",
            "manager",
            "developer",
            "programmer",
            "designer",
            "artist",
            "writer",
            "accountant",
            "consultant",
            "analyst",
            "researcher",
            "scientist",
            "professor",
            "student",
            "chef",
            "mechanic",
            "electrician",
            "plumber",
            "carpenter",
            "architect",
            "pharmacist",
        ];

        for occupation in &occupations {
            if text.contains(&format!("{} is a {}", name_lower, occupation))
                || text.contains(&format!("{} works as", name_lower))
                || text.contains(&format!("dr. {}", name_lower))
                || text.contains(&format!("professor {}", name_lower))
            {
                return Some(occupation.to_string());
            }
        }

        None
    }

    fn extract_personality_traits(name: &str, message: &str) -> Option<String> {
        let text = message.to_lowercase();
        let name_lower = name.to_lowercase();

        let traits = [
            "kind",
            "nice",
            "friendly",
            "helpful",
            "smart",
            "intelligent",
            "funny",
            "serious",
            "quiet",
            "loud",
            "outgoing",
            "shy",
            "confident",
            "nervous",
            "patient",
            "impatient",
            "generous",
            "selfish",
            "honest",
            "dishonest",
            "reliable",
            "unreliable",
            "creative",
            "logical",
            "emotional",
            "calm",
        ];

        let mut found_traits = Vec::new();
        for trait_word in &traits {
            if text.contains(&format!("{} is {}", name_lower, trait_word))
                || text.contains(&format!("{} seems {}", name_lower, trait_word))
                || text.contains(&format!("very {} {}", trait_word, name_lower))
            {
                found_traits.push(trait_word.to_string());
            }
        }

        if found_traits.is_empty() {
            None
        } else {
            Some(found_traits.join(", "))
        }
    }

    fn calculate_person_importance(name: &str, message: &str) -> f32 {
        let mut importance = 0.5; // Base importance
        let text = message.to_lowercase();
        let name_lower = name.to_lowercase();

        // Increase importance based on relationship closeness
        if text.contains("best friend") || text.contains("family") {
            importance += 0.3;
        } else if text.contains("friend")
            || text.contains("colleague")
            || text.contains("boss")
            || text.contains("manager")
        {
            importance += 0.2;
        }

        // Increase importance based on emotional context
        let emotional_words = [
            "love", "hate", "angry", "happy", "sad", "excited", "worried",
        ];
        for word in &emotional_words {
            if text.contains(word) {
                importance += 0.1;
                break;
            }
        }

        // Increase importance if mentioned multiple times in the same message
        let mention_count = text.matches(&name_lower).count();
        if mention_count > 1 {
            importance += 0.1 * (mention_count - 1) as f32;
        }

        // Cap at 1.0
        importance.min(1.0)
    }

    fn generate_initial_attitudes(
        name: &str,
        message: &str,
        companion_id: i32,
    ) -> CompanionAttitude {
        let current_time = get_current_date();
        let text = message.to_lowercase();

        // Base neutral attitudes
        let mut attitude = CompanionAttitude {
            id: None,
            companion_id,
            target_id: 0, // Will be set by caller
            target_type: "third_party".to_string(),
            attraction: 0.0,
            trust: 5.0,
            fear: 0.0,
            anger: 0.0,
            joy: 0.0,
            sorrow: 0.0,
            disgust: 0.0,
            surprise: 15.0,  // New person = some surprise
            curiosity: 20.0, // New person = high curiosity
            respect: 10.0,
            suspicion: 5.0, // Slight initial caution
            gratitude: 0.0,
            jealousy: 0.0,
            empathy: 10.0,
            lust: 0.0,
            love: 0.0,
            anxiety: 0.0,
            butterflies: 0.0,
            submissiveness: 0.0,
            dominance: 0.0,
            relationship_score: None,
            last_updated: current_time.clone(),
            created_at: current_time,
        };

        // Adjust based on relationship context
        if let Some(relationship) = Database::extract_relationship_to_user(name, message) {
            match relationship.as_str() {
                "friend" | "best friend" => {
                    attitude.trust += 15.0;
                    attitude.joy += 10.0;
                    attitude.respect += 10.0;
                    attitude.suspicion -= 5.0;
                }
                "family" | "brother" | "sister" | "mother" | "father" => {
                    attitude.trust += 20.0;
                    attitude.joy += 15.0;
                    attitude.respect += 15.0;
                    attitude.empathy += 10.0;
                    attitude.suspicion = 0.0;
                }
                "boss" | "manager" => {
                    attitude.respect += 20.0;
                    attitude.fear += 10.0;
                    attitude.curiosity += 10.0;
                }
                "colleague" | "coworker" => {
                    attitude.trust += 10.0;
                    attitude.respect += 10.0;
                }
                _ => {}
            }
        }

        // Adjust based on emotional context in the message
        if text.contains("love") || text.contains("adore") {
            attitude.attraction += 15.0;
            attitude.joy += 20.0;
        } else if text.contains("hate") || text.contains("dislike") {
            attitude.anger += 15.0;
            attitude.disgust += 10.0;
            attitude.trust -= 10.0;
        } else if text.contains("worried") || text.contains("concerned") {
            attitude.fear += 10.0;
            attitude.empathy += 10.0;
        } else if text.contains("excited") || text.contains("happy") {
            attitude.joy += 15.0;
            attitude.curiosity += 10.0;
        }

        // Clamp all values to valid range
        Database::clamp_attitude_values(&mut attitude);
        attitude
    }

    fn clamp_attitude_values(attitude: &mut CompanionAttitude) {
        attitude.attraction = attitude.attraction.clamp(-100.0, 100.0);
        attitude.trust = attitude.trust.clamp(-100.0, 100.0);
        attitude.fear = attitude.fear.clamp(-100.0, 100.0);
        attitude.anger = attitude.anger.clamp(-100.0, 100.0);
        attitude.joy = attitude.joy.clamp(-100.0, 100.0);
        attitude.sorrow = attitude.sorrow.clamp(-100.0, 100.0);
        attitude.disgust = attitude.disgust.clamp(-100.0, 100.0);
        attitude.surprise = attitude.surprise.clamp(-100.0, 100.0);
        attitude.curiosity = attitude.curiosity.clamp(-100.0, 100.0);
        attitude.respect = attitude.respect.clamp(-100.0, 100.0);
        attitude.suspicion = attitude.suspicion.clamp(-100.0, 100.0);
        attitude.gratitude = attitude.gratitude.clamp(-100.0, 100.0);
        attitude.jealousy = attitude.jealousy.clamp(-100.0, 100.0);
        attitude.empathy = attitude.empathy.clamp(-100.0, 100.0);
    }

    // Companion Interaction Tracking System

    pub fn generate_interaction_outcome(interaction_id: i32) -> Result<String> {
        let con = Self::open()?;

        // Get the interaction details
        let interaction: ThirdPartyInteraction = con.query_row(
            "SELECT id, third_party_id, companion_id, interaction_type, description,
                    planned_date, actual_date, outcome, impact_on_relationship,
                    created_at, updated_at
             FROM third_party_interactions WHERE id = ?",
            [&interaction_id],
            |row| {
                Ok(ThirdPartyInteraction {
                    id: Some(row.get(0)?),
                    third_party_id: row.get(1)?,
                    companion_id: row.get(2)?,
                    interaction_type: row.get(3)?,
                    description: row.get(4)?,
                    planned_date: row.get(5)?,
                    actual_date: row.get(6)?,
                    outcome: row.get(7)?,
                    impact_on_relationship: row.get(8)?,
                    created_at: row.get(9)?,
                    updated_at: row.get(10)?,
                })
            },
        )?;

        // Get the companion's attitude toward this third party
        let attitude = Database::get_attitude(
            interaction.companion_id,
            interaction.third_party_id,
            "third_party",
        )?
        .ok_or_else(|| Error::QueryReturnedNoRows)?;

        // Get third party details
        let third_party = Database::get_third_party_by_id(interaction.third_party_id)?
            .ok_or_else(|| Error::QueryReturnedNoRows)?;

        // Generate outcome based on attitude and interaction type
        let outcome = Database::create_realistic_outcome(&interaction, &attitude, &third_party);

        // Calculate impact on relationship
        let impact = Database::calculate_interaction_impact(&interaction, &attitude);

        // Complete the interaction with the generated outcome
        Database::complete_interaction(interaction_id, &outcome, impact)?;

        // Update attitudes based on the interaction
        Database::update_attitude_from_interaction(
            interaction.companion_id,
            interaction.third_party_id,
            &interaction.description,
            impact,
        )?;

        Ok(outcome)
    }

    fn create_realistic_outcome(
        interaction: &ThirdPartyInteraction,
        attitude: &CompanionAttitude,
        third_party: &ThirdPartyIndividual,
    ) -> String {
        let relationship_quality = attitude.relationship_score.unwrap_or(0.0);
        let interaction_desc = &interaction.description;
        let person_name = &third_party.name;

        // Generate outcome based on relationship quality and interaction type
        if interaction_desc.contains("meet")
            || interaction_desc.contains("coffee")
            || interaction_desc.contains("lunch")
        {
            if relationship_quality > 50.0 {
                format!("Had a wonderful time with {}! We talked about various topics and really enjoyed each other's company. {} seemed happy and we made plans to meet again soon.", person_name, person_name)
            } else if relationship_quality > 0.0 {
                format!("Met with {} as planned. The conversation was pleasant enough, though there were a few awkward moments. {} was friendly but seemed a bit distracted.", person_name, person_name)
            } else {
                format!("The meeting with {} was tense. We struggled to find common ground and the conversation felt forced. {} left early citing other commitments.", person_name, person_name)
            }
        } else if interaction_desc.contains("call") || interaction_desc.contains("phone") {
            if relationship_quality > 30.0 {
                format!("Had a great phone conversation with {}. We caught up on recent events and shared some laughs. The call lasted longer than expected because we were enjoying the chat.", person_name)
            } else if relationship_quality > -20.0 {
                format!("Spoke with {} on the phone briefly. The conversation was polite but somewhat formal. We covered the necessary topics and ended the call.", person_name)
            } else {
                format!("The phone call with {} was brief and uncomfortable. We barely exchanged pleasantries before {} had to go.", person_name, person_name)
            }
        } else if interaction_desc.contains("help") || interaction_desc.contains("assist") {
            if attitude.trust > 50.0 && attitude.gratitude > 30.0 {
                format!("{} was incredibly grateful for my help! They thanked me multiple times and offered to return the favor anytime. This really strengthened our bond.", person_name)
            } else if attitude.trust > 0.0 {
                format!("{} appreciated the help, though they seemed a bit hesitant to accept it at first. In the end, everything worked out well.", person_name)
            } else {
                format!("{} reluctantly accepted my help but didn't seem very appreciative. There was an underlying tension throughout the interaction.", person_name)
            }
        } else if interaction_desc.contains("party")
            || interaction_desc.contains("event")
            || interaction_desc.contains("gathering")
        {
            if attitude.joy > 40.0 && relationship_quality > 20.0 {
                format!("The event with {} was fantastic! We had a great time, met interesting people, and {} introduced me to several of their friends. Definitely a night to remember!", person_name, person_name)
            } else if relationship_quality > -10.0 {
                format!("Attended the event with {}. It was decent - the venue was nice and there were some interesting moments, though {} and I didn't interact as much as expected.", person_name, person_name)
            } else {
                format!("The event with {} was awkward. We barely spoke and {} spent most of the time with other people. I left early.", person_name, person_name)
            }
        } else {
            // Generic interaction outcome
            if relationship_quality > 40.0 {
                format!("The interaction with {} went very well! Everything proceeded smoothly and we both seemed to enjoy it. Our relationship feels stronger.", person_name)
            } else if relationship_quality > -20.0 {
                format!("Completed the planned activity with {}. It was fine, nothing particularly memorable but no issues either.", person_name)
            } else {
                format!("The interaction with {} was difficult. There were several uncomfortable moments and neither of us seemed happy with how things went.", person_name)
            }
        }
    }

    fn calculate_interaction_impact(
        interaction: &ThirdPartyInteraction,
        attitude: &CompanionAttitude,
    ) -> f32 {
        let base_relationship = attitude.relationship_score.unwrap_or(0.0);

        // Positive interactions have more impact when relationship is already good
        let impact = if interaction.description.contains("fun")
            || interaction.description.contains("enjoy")
            || interaction.description.contains("great")
        {
            5.0 + (base_relationship * 0.1)
        }
        // Helping interactions build trust and gratitude
        else if interaction.description.contains("help")
            || interaction.description.contains("assist")
            || interaction.description.contains("support")
        {
            8.0 + (attitude.trust * 0.05)
        }
        // Conflict reduces relationship quality
        else if interaction.description.contains("argue")
            || interaction.description.contains("fight")
            || interaction.description.contains("disagree")
        {
            -10.0 - (attitude.anger * 0.1)
        }
        // Casual interactions have mild impact
        else if interaction.description.contains("meet")
            || interaction.description.contains("talk")
            || interaction.description.contains("chat")
        {
            2.0 * (1.0 + base_relationship / 100.0)
        }
        // Professional interactions are neutral to positive
        else if interaction.description.contains("work")
            || interaction.description.contains("project")
            || interaction.description.contains("business")
        {
            1.0 + (attitude.respect * 0.02)
        } else {
            // Default small positive impact
            1.0
        };

        // Clamp impact to reasonable range
        impact.clamp(-25.0, 25.0)
    }

    fn update_attitude_from_interaction(
        companion_id: i32,
        third_party_id: i32,
        description: &str,
        impact: f32,
    ) -> Result<()> {
        // Determine which dimensions to update based on interaction description
        let mut updates: Vec<(&str, f32)> = Vec::new();

        if impact > 0.0 {
            // Positive interaction
            if description.contains("fun")
                || description.contains("laugh")
                || description.contains("enjoy")
            {
                updates.push(("joy", impact * 0.8));
                updates.push(("attraction", impact * 0.3));
            }
            if description.contains("help")
                || description.contains("support")
                || description.contains("assist")
            {
                updates.push(("gratitude", impact * 1.2));
                updates.push(("trust", impact * 0.6));
            }
            if description.contains("deep")
                || description.contains("meaningful")
                || description.contains("understand")
            {
                updates.push(("empathy", impact * 0.7));
                updates.push(("respect", impact * 0.5));
            }
            // Reduce negative emotions
            updates.push(("suspicion", -impact * 0.3));
            updates.push(("fear", -impact * 0.2));
        } else {
            // Negative interaction
            if description.contains("argue")
                || description.contains("fight")
                || description.contains("conflict")
            {
                updates.push(("anger", -impact * 0.8));
                updates.push(("trust", impact * 0.5));
            }
            if description.contains("disappoint")
                || description.contains("letdown")
                || description.contains("fail")
            {
                updates.push(("sorrow", -impact * 0.6));
                updates.push(("respect", impact * 0.4));
            }
            if description.contains("lie")
                || description.contains("betray")
                || description.contains("deceive")
            {
                updates.push(("suspicion", -impact * 1.5));
                updates.push(("trust", impact * 2.0));
                updates.push(("disgust", -impact * 0.7));
            }
            // Reduce positive emotions
            updates.push(("joy", impact * 0.4));
            updates.push(("attraction", impact * 0.3));
        }

        // Apply all updates
        for (dimension, delta) in updates {
            Database::update_attitude_dimension(
                companion_id,
                third_party_id,
                "third_party",
                dimension,
                delta,
            )?;
        }

        Ok(())
    }

    pub fn get_third_party_by_id(id: i32) -> Result<Option<ThirdPartyIndividual>> {
        let con = Self::open()?;
        let mut stmt = con.prepare(
            "SELECT id, name, relationship_to_user, relationship_to_companion, occupation,
                    personality_traits, physical_description, first_mentioned, last_mentioned,
                    mention_count, importance_score, created_at, updated_at, source
             FROM third_party_individuals WHERE id = ?",
        )?;

        let individual = stmt
            .query_row([&id], |row| {
                Ok(ThirdPartyIndividual {
                    id: Some(row.get(0)?),
                    name: row.get(1)?,
                    relationship_to_user: row.get(2)?,
                    relationship_to_companion: row.get(3)?,
                    occupation: row.get(4)?,
                    personality_traits: row.get(5)?,
                    physical_description: row.get(6)?,
                    first_mentioned: row.get(7)?,
                    last_mentioned: row.get(8)?,
                    mention_count: row.get(9)?,
                    importance_score: row.get(10)?,
                    created_at: row.get(11)?,
                    updated_at: row.get(12)?,
                    source: row.get(13)?,
                })
            })
            .ok();

        Ok(individual)
    }

    pub fn detect_interaction_request(
        message: &str,
        companion_id: i32,
    ) -> Result<Option<ThirdPartyInteraction>> {
        let message_lower = message.to_lowercase();

        // Check if user is asking about past interactions
        if message_lower.contains("did you")
            || message_lower.contains("have you")
            || message_lower.contains("what happened")
            || message_lower.contains("how did")
            || message_lower.contains("tell me about")
        {
            // Extract person name from the message
            if let Some(person_name) = Database::extract_person_from_query(message) {
                if let Some(third_party) = Database::get_third_party_by_name(&person_name)? {
                    // Check for recent interactions
                    let history =
                        Database::get_interaction_history(companion_id, third_party.id.unwrap())?;
                    if !history.is_empty() {
                        return Ok(Some(history[0].clone()));
                    }

                    // Check for planned interactions that might have occurred
                    let planned = Database::get_planned_interactions(companion_id, Some(5))?;
                    for interaction in planned {
                        if interaction.third_party_id == third_party.id.unwrap() {
                            // Generate outcome for this interaction
                            let _outcome =
                                Database::generate_interaction_outcome(interaction.id.unwrap())?;
                            return Database::get_interaction_by_id(interaction.id.unwrap());
                        }
                    }
                }
            }
        }

        // Check if user is planning future interaction
        if message_lower.contains("plan to")
            || message_lower.contains("going to")
            || message_lower.contains("will meet")
            || message_lower.contains("scheduled")
        {
            if let Some(person_name) = Database::extract_person_from_query(message) {
                if let Some(third_party) = Database::get_third_party_by_name(&person_name)? {
                    let interaction = ThirdPartyInteraction {
                        id: None,
                        third_party_id: third_party.id.unwrap(),
                        companion_id,
                        interaction_type: "planned".to_string(),
                        description: Database::extract_interaction_description(
                            message,
                            &person_name,
                        ),
                        planned_date: Some(Database::extract_planned_date(message)),
                        actual_date: None,
                        outcome: None,
                        impact_on_relationship: 0.0,
                        created_at: get_current_date(),
                        updated_at: get_current_date(),
                    };

                    let interaction_id = Database::plan_third_party_interaction(&interaction)?;
                    return Database::get_interaction_by_id(interaction_id);
                }
            }
        }

        Ok(None)
    }

    fn extract_person_from_query(message: &str) -> Option<String> {
        // Try to find person names mentioned in the query
        let message_lower = message.to_lowercase();

        // Look for patterns like "with [Name]", "to [Name]", "about [Name]"
        let patterns = [
            r"with\s+(\w+)",
            r"to\s+(\w+)",
            r"about\s+(\w+)",
            r"see\s+(\w+)",
            r"meet\s+(\w+)",
            r"call\s+(\w+)",
            r"visit\s+(\w+)",
        ];

        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                if let Some(cap) = re.captures(&message_lower) {
                    if let Some(name_match) = cap.get(1) {
                        let name = name_match.as_str();
                        if name.len() > 2 && !Database::is_common_word(name) {
                            return Some(Database::capitalize_name(name));
                        }
                    }
                }
            }
        }

        None
    }

    fn extract_interaction_description(message: &str, person_name: &str) -> String {
        let message_lower = message.to_lowercase();
        let _name_lower = person_name.to_lowercase();

        // Extract the core activity from the message
        if message_lower.contains("coffee") {
            format!("Have coffee with {}", person_name)
        } else if message_lower.contains("lunch") {
            format!("Have lunch with {}", person_name)
        } else if message_lower.contains("dinner") {
            format!("Have dinner with {}", person_name)
        } else if message_lower.contains("meet") {
            format!("Meet with {}", person_name)
        } else if message_lower.contains("call") || message_lower.contains("phone") {
            format!("Phone call with {}", person_name)
        } else if message_lower.contains("help") {
            format!("Help {} with something", person_name)
        } else if message_lower.contains("party") || message_lower.contains("event") {
            format!("Attend event with {}", person_name)
        } else if message_lower.contains("work") || message_lower.contains("project") {
            format!("Work on project with {}", person_name)
        } else if message_lower.contains("visit") {
            format!("Visit {}", person_name)
        } else {
            format!("Interact with {}", person_name)
        }
    }

    fn extract_planned_date(message: &str) -> String {
        let message_lower = message.to_lowercase();

        if message_lower.contains("tomorrow") {
            "tomorrow".to_string()
        } else if message_lower.contains("today") {
            "today".to_string()
        } else if message_lower.contains("tonight") {
            "tonight".to_string()
        } else if message_lower.contains("this weekend") {
            "this weekend".to_string()
        } else if message_lower.contains("next week") {
            "next week".to_string()
        } else if message_lower.contains("monday") {
            "Monday".to_string()
        } else if message_lower.contains("tuesday") {
            "Tuesday".to_string()
        } else if message_lower.contains("wednesday") {
            "Wednesday".to_string()
        } else if message_lower.contains("thursday") {
            "Thursday".to_string()
        } else if message_lower.contains("friday") {
            "Friday".to_string()
        } else if message_lower.contains("saturday") {
            "Saturday".to_string()
        } else if message_lower.contains("sunday") {
            "Sunday".to_string()
        } else {
            "soon".to_string()
        }
    }

    pub fn get_interaction_by_id(id: i32) -> Result<Option<ThirdPartyInteraction>> {
        let con = Self::open()?;
        let mut stmt = con.prepare(
            "SELECT id, third_party_id, companion_id, interaction_type, description,
                    planned_date, actual_date, outcome, impact_on_relationship,
                    created_at, updated_at
             FROM third_party_interactions WHERE id = ?",
        )?;

        let interaction = stmt
            .query_row([&id], |row| {
                Ok(ThirdPartyInteraction {
                    id: Some(row.get(0)?),
                    third_party_id: row.get(1)?,
                    companion_id: row.get(2)?,
                    interaction_type: row.get(3)?,
                    description: row.get(4)?,
                    planned_date: row.get(5)?,
                    actual_date: row.get(6)?,
                    outcome: row.get(7)?,
                    impact_on_relationship: row.get(8)?,
                    created_at: row.get(9)?,
                    updated_at: row.get(10)?,
                })
            })
            .ok();

        Ok(interaction)
    }

    /// Backfills `messages.speaker_id` on a database created before that
    /// column existed. `speaker_id TEXT NOT NULL DEFAULT ''` (SQLite serves
    /// that default to every pre-existing row at read time, per the
    /// `ALTER TABLE ADD COLUMN` docs) is what makes the follow-up `UPDATE`
    /// necessary.
    ///
    /// The existence check, the `ALTER TABLE` and the backfill all run
    /// inside one `IMMEDIATE` transaction (started before the check, not
    /// after it) so two `init()` calls racing on the same database file
    /// cannot both observe "column missing" and both try to add it: the
    /// second to acquire the write lock sees the first's committed column
    /// and returns early instead of erroring on a duplicate `ALTER TABLE`.
    /// That transaction also guarantees no row can be left at `''` if the
    /// process crashes mid-migration. Idempotent: the `PRAGMA table_info`
    /// check short-circuits on an already-migrated database, and the
    /// `UPDATE`'s `WHERE speaker_id = ''` would no-op even if it ran again.
    pub fn migrate_messages_speaker_id(con: &Connection) -> Result<()> {
        let tx = Transaction::new_unchecked(con, TransactionBehavior::Immediate)?;

        let mut stmt = tx.prepare("PRAGMA table_info(messages)")?;
        let has_speaker_id = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>>>()?
            .iter()
            .any(|name| name == "speaker_id");
        drop(stmt);

        if has_speaker_id {
            return tx.commit();
        }

        tx.execute(
            "ALTER TABLE messages ADD COLUMN speaker_id TEXT NOT NULL DEFAULT ''",
            [],
        )?;
        tx.execute(
            &format!(
                "UPDATE messages SET speaker_id = CASE WHEN ai = 1 THEN '{CHAR_SPEAKER_ID}' ELSE '{USER_SPEAKER_ID}' END WHERE speaker_id = ''"
            ),
            [],
        )?;
        tx.commit()?;

        Ok(())
    }

    /// Adds the `compacted_through` column (#171) to a `companion` table
    /// that predates conversation compaction. `NULL` is the correct initial
    /// state ("never compacted"), so unlike `migrate_messages_speaker_id`
    /// there is no backfill, just the `ALTER TABLE`. Idempotent, via the
    /// same `PRAGMA table_info` check.
    pub fn migrate_companion_compacted_through(con: &Connection) -> Result<()> {
        // `IMMEDIATE`, before the check, for the same reason
        // `migrate_messages_speaker_id` above needs it: without it, two
        // `init()` calls against the same database file can both observe
        // "column missing" before either runs its `ALTER TABLE`, and the
        // second then fails with a duplicate-column error instead of
        // blocking (via the busy timeout) and seeing the column already
        // there.
        let tx = Transaction::new_unchecked(con, TransactionBehavior::Immediate)?;

        let mut stmt = tx.prepare("PRAGMA table_info(companion)")?;
        let has_compacted_through = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>>>()?
            .iter()
            .any(|name| name == "compacted_through");
        drop(stmt);

        if has_compacted_through {
            return tx.commit();
        }

        tx.execute(
            "ALTER TABLE companion ADD COLUMN compacted_through INTEGER",
            [],
        )?;
        tx.commit()
    }

    /// Adds every `config` column introduced after the original four
    /// (`device`, `llm_model_path`, `gpu_layers`, `prompt_template`) to a
    /// database that predates it. Table-driven rather than one `has_*` bool
    /// per column (that grew unwieldy past #128's six new multiplayer
    /// columns): reads the existing column set once via `PRAGMA
    /// table_info`, then runs each `ALTER TABLE` whose column is absent.
    /// Idempotent, like the per-bool version it replaces. Includes
    /// `dynamic_gpu_allocation`, `gpu_safety_margin`, and `min_free_vram_mb`,
    /// which the previous per-bool version omitted entirely (a database
    /// missing those three columns could migrate "successfully" and then
    /// fail `read_config`/`write_config` with a "no such column" error).
    /// Backfills `third_party_individuals.source` on a database created
    /// before #177, matching the `PRAGMA table_info` guard `migrate_config_table`
    /// already uses. Existing rows read back as `'heuristic'` (the column's
    /// own `DEFAULT`), which is correct: nothing but `upsert_compaction_person`
    /// ever writes `'compaction'`.
    pub fn migrate_third_party_individuals_table(con: &Connection) -> Result<()> {
        let mut stmt = con.prepare("PRAGMA table_info(third_party_individuals)")?;
        let existing: HashSet<String> = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<_>>()?;
        drop(stmt);

        if !existing.contains("source") {
            con.execute(
                "ALTER TABLE third_party_individuals ADD COLUMN source TEXT NOT NULL DEFAULT 'heuristic'",
                [],
            )?;
        }
        Ok(())
    }

    pub fn migrate_config_table(con: &Connection) -> Result<()> {
        const COLUMNS: &[(&str, &str)] = &[
            (
                "context_window_size",
                "ALTER TABLE config ADD COLUMN context_window_size INTEGER DEFAULT 2048",
            ),
            (
                "max_response_tokens",
                "ALTER TABLE config ADD COLUMN max_response_tokens INTEGER DEFAULT 512",
            ),
            (
                "enable_dynamic_context",
                "ALTER TABLE config ADD COLUMN enable_dynamic_context BOOLEAN DEFAULT true",
            ),
            (
                "vram_limit_gb",
                "ALTER TABLE config ADD COLUMN vram_limit_gb INTEGER DEFAULT 4",
            ),
            (
                "dynamic_gpu_allocation",
                "ALTER TABLE config ADD COLUMN dynamic_gpu_allocation BOOLEAN DEFAULT true",
            ),
            (
                "gpu_safety_margin",
                "ALTER TABLE config ADD COLUMN gpu_safety_margin REAL DEFAULT 0.8",
            ),
            (
                "min_free_vram_mb",
                "ALTER TABLE config ADD COLUMN min_free_vram_mb INTEGER DEFAULT 512",
            ),
            (
                "enable_hybrid_context",
                "ALTER TABLE config ADD COLUMN enable_hybrid_context BOOLEAN DEFAULT true",
            ),
            (
                "max_system_ram_usage_gb",
                "ALTER TABLE config ADD COLUMN max_system_ram_usage_gb INTEGER DEFAULT 8",
            ),
            (
                "context_expansion_strategy",
                "ALTER TABLE config ADD COLUMN context_expansion_strategy TEXT DEFAULT 'balanced'",
            ),
            (
                "ram_safety_margin_gb",
                "ALTER TABLE config ADD COLUMN ram_safety_margin_gb INTEGER DEFAULT 2",
            ),
            (
                "multiplayer_mode",
                "ALTER TABLE config ADD COLUMN multiplayer_mode TEXT DEFAULT 'solo'",
            ),
            (
                "multiplayer_password",
                "ALTER TABLE config ADD COLUMN multiplayer_password TEXT DEFAULT ''",
            ),
            (
                "multiplayer_host_address",
                "ALTER TABLE config ADD COLUMN multiplayer_host_address TEXT DEFAULT ''",
            ),
            (
                "multiplayer_participant_id",
                "ALTER TABLE config ADD COLUMN multiplayer_participant_id TEXT DEFAULT ''",
            ),
            (
                "mention_followup_depth",
                "ALTER TABLE config ADD COLUMN mention_followup_depth INTEGER DEFAULT 1",
            ),
            (
                "remote_generation_timeout_secs",
                "ALTER TABLE config ADD COLUMN remote_generation_timeout_secs INTEGER DEFAULT 120",
            ),
            (
                "compact_threshold_tokens",
                "ALTER TABLE config ADD COLUMN compact_threshold_tokens INTEGER",
            ),
            (
                "compact_min_messages",
                "ALTER TABLE config ADD COLUMN compact_min_messages INTEGER DEFAULT 8",
            ),
            (
                "compaction_model_path",
                "ALTER TABLE config ADD COLUMN compaction_model_path TEXT",
            ),
            (
                "heuristic_person_detection",
                "ALTER TABLE config ADD COLUMN heuristic_person_detection BOOLEAN DEFAULT false",
            ),
            (
                "compaction_attitude_weight",
                "ALTER TABLE config ADD COLUMN compaction_attitude_weight REAL DEFAULT 0.5",
            ),
        ];

        let mut stmt = con.prepare("PRAGMA table_info(config)")?;
        let existing: HashSet<String> = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<_>>()?;
        drop(stmt);

        for (name, ddl) in COLUMNS {
            if !existing.contains(*name) {
                con.execute(ddl, [])?;
            }
        }

        Ok(())
    }

    pub fn migrate_companion_attitudes_table(con: &Connection) -> Result<()> {
        // Check if new attitude columns exist and add them if they don't
        let mut has_lust = false;
        let mut has_love = false;
        let mut has_anxiety = false;
        let mut has_butterflies = false;
        let mut has_submissiveness = false;
        let mut has_dominance = false;

        // Check existing columns
        let mut stmt = con.prepare("PRAGMA table_info(companion_attitudes)")?;
        let rows = stmt.query_map([], |row| {
            let column_name: String = row.get(1)?;
            Ok(column_name)
        })?;

        for row in rows {
            let column_name = row?;
            match column_name.as_str() {
                "lust" => has_lust = true,
                "love" => has_love = true,
                "anxiety" => has_anxiety = true,
                "butterflies" => has_butterflies = true,
                "submissiveness" => has_submissiveness = true,
                "dominance" => has_dominance = true,
                _ => {}
            }
        }

        // Add missing columns
        if !has_lust {
            con.execute("ALTER TABLE companion_attitudes ADD COLUMN lust REAL DEFAULT 0 CHECK(lust >= -100 AND lust <= 100)", [])?;
        }
        if !has_love {
            con.execute("ALTER TABLE companion_attitudes ADD COLUMN love REAL DEFAULT 0 CHECK(love >= -100 AND love <= 100)", [])?;
        }
        if !has_anxiety {
            con.execute("ALTER TABLE companion_attitudes ADD COLUMN anxiety REAL DEFAULT 0 CHECK(anxiety >= -100 AND anxiety <= 100)", [])?;
        }
        if !has_butterflies {
            con.execute("ALTER TABLE companion_attitudes ADD COLUMN butterflies REAL DEFAULT 0 CHECK(butterflies >= -100 AND butterflies <= 100)", [])?;
        }
        if !has_submissiveness {
            con.execute("ALTER TABLE companion_attitudes ADD COLUMN submissiveness REAL DEFAULT 0 CHECK(submissiveness >= -100 AND submissiveness <= 100)", [])?;
        }
        if !has_dominance {
            con.execute("ALTER TABLE companion_attitudes ADD COLUMN dominance REAL DEFAULT 0 CHECK(dominance >= -100 AND dominance <= 100)", [])?;
        }

        // Update the relationship_score calculation in the database by dropping the generated column and recreating it
        // Note: SQLite doesn't support modifying generated columns directly
        if !has_lust
            || !has_love
            || !has_anxiety
            || !has_butterflies
            || !has_submissiveness
            || !has_dominance
        {
            // The relationship_score column will be recalculated automatically with the new formula
            // when the table structure is updated
        }

        Ok(())
    }

    /// Check for third-party mentions in message and track them, returning console output
    pub fn track_third_party_mentions(message: &str, excluded_names: &[String]) -> Result<String> {
        let mut console_output = Vec::new();

        // Get all existing third parties to check for mentions
        let third_parties = Database::get_all_third_party_individuals()?;

        let message_lower = message.to_lowercase();
        let con = Self::open()?;

        for party in &third_parties {
            if Database::names_same_participant(&party.name, excluded_names) {
                continue;
            }
            let name_lower = party.name.to_lowercase();

            // Check if this person is mentioned in the message
            if message_lower.contains(&name_lower) {
                // Update mention count and last_mentioned
                let current_time = get_current_date();

                con.execute(
                    "UPDATE third_party_individuals
                     SET mention_count = mention_count + 1,
                         last_mentioned = ?,
                         updated_at = ?
                     WHERE id = ?",
                    params![current_time, current_time, party.id.unwrap()],
                )?;

                let new_count = party.mention_count + 1;
                let suffix = match new_count {
                    1 => "st",
                    2 => "nd",
                    3 => "rd",
                    _ => "th",
                };

                console_output.push(format!(
                    "👥 {} mentioned for the {}{} time",
                    party.name, new_count, suffix
                ));
            }
        }

        // Also check for new person names that might not be in the database yet
        // This is a simplified detection - in practice you might want more sophisticated NER
        // `extract_potential_names` requires an uppercase first letter, so it
        // must see the original casing, not `message_lower`.
        let potential_names = Database::drop_excluded_names(
            Database::extract_potential_names(message),
            excluded_names,
        );
        for potential_name in potential_names {
            // Check if this is a new person (not in database)
            if Database::get_third_party_by_name(&potential_name)?.is_none() {
                // This could be a new person mention - for now just report it
                // The actual person creation is handled by detect_new_persons_in_message
                console_output.push(format!(
                    "👥 {} mentioned for the 1st time (new person detected)",
                    potential_name
                ));
            }
        }

        Ok(console_output.join("\n"))
    }

    /// Extract potential person names from message (simplified approach)
    fn extract_potential_names(message: &str) -> Vec<String> {
        let mut names = Vec::new();
        let words: Vec<&str> = message.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            // Look for capitalized words that might be names
            if word.chars().next().unwrap_or('a').is_uppercase()
                && word.len() > 2
                && word.chars().all(|c| c.is_alphabetic())
            {
                // Skip common non-name words
                let skip_words = [
                    "the", "and", "but", "for", "nor", "yet", "so", "at", "by", "in", "of", "on",
                    "to", "up", "as", "is", "it", "or", "be", "do", "go", "he", "if", "me", "my",
                    "no", "we", "I",
                ];

                if !skip_words.contains(&word.to_lowercase().as_str()) {
                    // Check if the next word might be a last name
                    if i + 1 < words.len() {
                        let next_word = words[i + 1];
                        if next_word.chars().next().unwrap_or('a').is_uppercase()
                            && next_word.len() > 2
                            && next_word.chars().all(|c| c.is_alphabetic())
                        {
                            names.push(format!("{} {}", word, next_word));
                        } else {
                            names.push(word.to_string());
                        }
                    } else {
                        names.push(word.to_string());
                    }
                }
            }
        }

        names
    }

    /// Drops any name in `names` that names the same participant as an entry
    /// in `excluded`, so a chat participant's own display name (the user's,
    /// the host companion's, or a joined bot's) is never treated as a third
    /// party mentioned in the conversation.
    ///
    /// A name matches when it case-insensitively equals a full excluded
    /// display name, or when it is a single word that is itself one word of
    /// a multi-word excluded name: `extract_person_names`'s patterns capture
    /// only the first token of "Mary Jane" as "Mary", so a bare word-level
    /// match is needed to still exclude her.
    fn drop_excluded_names(names: Vec<String>, excluded: &[String]) -> Vec<String> {
        names
            .into_iter()
            .filter(|name| !Database::names_same_participant(name, excluded))
            .collect()
    }

    /// Whether `name` refers to the same participant as one of `excluded`'s
    /// display names. See `drop_excluded_names` for the matching rule.
    fn names_same_participant(name: &str, excluded: &[String]) -> bool {
        let name_lower = name.to_lowercase();
        excluded.iter().any(|excl| {
            let excl_lower = excl.to_lowercase();
            excl_lower == name_lower || excl_lower.split_whitespace().any(|word| word == name_lower)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn open_at_enables_wal_on_a_fresh_database() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("t.db");

        let con = Database::open_at(&db_path).unwrap();
        let mode: String = con
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .unwrap();
        assert_eq!(mode, "wal");
        drop(con);

        // WAL is persisted in the file itself, so a fresh connection to the
        // same path reports it too, not just the connection that set it.
        let con2 = Database::open_at(&db_path).unwrap();
        let mode2: String = con2
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .unwrap();
        assert_eq!(mode2, "wal");
    }

    /// Old (pre-#125) `messages` shape, with no `speaker_id` column. Used
    /// only by the migration tests, which need to start from what a
    /// database created before this issue actually looks like.
    fn create_legacy_messages_table(con: &Connection) {
        con.execute(
            "CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ai BOOLEAN,
                content TEXT,
                created_at TEXT
            )",
            [],
        )
        .unwrap();
    }

    fn create_messages_table(con: &Connection) {
        con.execute(messages_ddl(), []).unwrap();
    }

    /// The three compaction tables (#181's acceptance tests below need real
    /// checkpoint/fact/pin rows, not just the `companion`/`messages` tables
    /// `mark_stale_for_message_on` itself touches), reusing #171's own DDL
    /// function rather than duplicating it.
    fn create_compaction_tables(con: &Connection) {
        crate::compaction::store::create_tables(con).unwrap();
    }

    /// Seeds one `compactions` row spanning `[from, through]` with the given
    /// `status`, returning its id.
    fn insert_checkpoint_row(
        con: &Connection,
        from: i32,
        through: i32,
        status: crate::compaction::types::CompactionStatus,
    ) -> i64 {
        con.execute(
            "INSERT INTO compactions (companion_id, from_message_id, through_message_id, status, trigger, created_at) VALUES (1, ?, ?, ?, 'threshold', ?)",
            params![from, through, &status as &dyn ToSql, get_current_date()],
        )
        .unwrap();
        con.last_insert_rowid()
    }

    /// A minimal `companion` row (#181): `edit_message_on`/`delete_message_on`
    /// look up the companion id and `compacted_through` on every call via
    /// `mark_stale_for_message_on`, so any test exercising them needs this
    /// table even when it has nothing to do with compaction itself.
    /// `compacted_through` starts `NULL`, matching a companion that has
    /// never been compacted.
    fn create_companion_table(con: &Connection) {
        con.execute(
            "CREATE TABLE IF NOT EXISTS companion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                persona TEXT,
                example_dialogue TEXT,
                first_message TEXT,
                long_term_mem INTEGER,
                short_term_mem INTEGER,
                roleplay BOOLEAN,
                dialogue_tuning BOOLEAN,
                avatar_path TEXT,
                compacted_through INTEGER
            )",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO companion (id, name, persona, example_dialogue, first_message, long_term_mem, short_term_mem, roleplay, dialogue_tuning, avatar_path) VALUES (1, 'Test', '', '', 'hi {{user}}', 0, 0, 0, 0, '')",
            [],
        )
        .unwrap();
    }

    /// A minimal `user` row: `insert_companion_greeting` (called by
    /// `erase_messages_on`) reads it to resolve `{{user}}` in the greeting.
    fn create_user_table(con: &Connection) {
        con.execute(
            "CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                persona TEXT,
                avatar_path TEXT
            )",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO user (id, name, persona, avatar_path) VALUES (1, 'Eric', '', '')",
            [],
        )
        .unwrap();
    }

    /// Matches the post-#125 schema (a `speaker_id` column, `ai` derived
    /// from it): every test in this module inserts by `speaker_id` rather
    /// than a bare `ai` flag, so a bot id like `"bot1"` can be seeded too.
    fn insert_message_row(con: &Connection, speaker_id: &str, content: &str) {
        con.execute(
            "INSERT INTO messages (ai, speaker_id, content, created_at) VALUES (?, ?, ?, ?)",
            params![
                is_ai_speaker(speaker_id),
                speaker_id,
                content,
                get_current_date()
            ],
        )
        .unwrap();
    }

    /// `owner_ready` for a test that never exercises the offline path: every
    /// speaker is ready.
    fn always_ready(_speaker_id: &str) -> bool {
        true
    }

    #[test]
    fn pop_latest_bot_reply_removes_the_trailing_bot_reply_and_anchors_on_the_user_turn() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "hi");
        insert_message_row(&con, CHAR_SPEAKER_ID, "hello");
        insert_message_row(&con, "bot1", "hi from bot1");

        let result = Database::pop_latest_bot_reply_on(&mut con, always_ready).unwrap();
        match result {
            PoppedReply::Removed {
                speaker_id,
                message_id,
                user_turn,
            } => {
                assert_eq!(speaker_id, "bot1");
                assert_eq!(message_id, 3);
                assert_eq!(user_turn.content, "hi");
            }
            other => panic!("expected bot1's reply to be removed, got {:?}", other),
        }

        let count: i64 = con
            .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn pop_latest_bot_reply_anchors_on_the_user_turn_two_positions_back_after_a_mention_followup() {
        // A mention follow-up (#131/#132) puts `char`'s reply between the
        // user's turn and `bot1`'s: the anchor must still be the user row,
        // not `char`'s reply immediately preceding `bot1`.
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "hi @bot1");
        insert_message_row(&con, CHAR_SPEAKER_ID, "sure, @bot1 go ahead");
        insert_message_row(&con, "bot1", "hi from bot1");

        let result = Database::pop_latest_bot_reply_on(&mut con, always_ready).unwrap();
        match result {
            PoppedReply::Removed {
                speaker_id,
                user_turn,
                ..
            } => {
                assert_eq!(speaker_id, "bot1");
                assert_eq!(user_turn.content, "hi @bot1");
            }
            _ => panic!("expected bot1's reply to be removed"),
        }
    }

    #[test]
    fn pop_latest_bot_reply_leaves_a_trailing_system_notice_alone() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "hi");
        insert_message_row(&con, SYSTEM_SPEAKER_ID, "bot1 did not respond");

        let result = Database::pop_latest_bot_reply_on(&mut con, always_ready).unwrap();
        assert!(matches!(result, PoppedReply::NothingToRegenerate));

        let count: i64 = con
            .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn get_messages_after_on_returns_messages_strictly_after_the_given_id_in_ascending_order() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "one");
        insert_message_row(&con, CHAR_SPEAKER_ID, "two");
        insert_message_row(&con, USER_SPEAKER_ID, "three");

        let messages = Database::get_messages_after_on(&con, 1).unwrap();
        assert_eq!(
            messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            vec!["two", "three"],
            "the boundary message (id 1) itself must be excluded"
        );

        assert!(Database::get_messages_after_on(&con, 3).unwrap().is_empty());
        assert_eq!(Database::get_messages_after_on(&con, 0).unwrap().len(), 3);
    }

    #[test]
    fn pop_latest_bot_reply_leaves_a_trailing_user_message_alone() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        insert_message_row(&con, CHAR_SPEAKER_ID, "hello");
        insert_message_row(&con, USER_SPEAKER_ID, "how are you?");

        let result = Database::pop_latest_bot_reply_on(&mut con, always_ready).unwrap();
        assert!(matches!(result, PoppedReply::NothingToRegenerate));

        let count: i64 = con
            .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn pop_latest_bot_reply_refuses_when_the_reply_has_no_preceding_user_turn() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        insert_message_row(&con, CHAR_SPEAKER_ID, "first message");

        let result = Database::pop_latest_bot_reply_on(&mut con, always_ready).unwrap();
        assert!(matches!(result, PoppedReply::NothingToRegenerate));

        let count: i64 = con
            .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn pop_latest_bot_reply_anchors_on_the_latest_user_turn_not_the_preceding_bot_row() {
        // Pre-#135 this refused to pop because the *immediate predecessor*
        // was another bot message; the real invariant is "the anchor is a
        // user row somewhere before the reply", which this conversation
        // still satisfies, so the pop now succeeds and anchors two rows back.
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "hi");
        insert_message_row(&con, CHAR_SPEAKER_ID, "first reply");
        insert_message_row(&con, CHAR_SPEAKER_ID, "second reply, inserted directly");

        let result = Database::pop_latest_bot_reply_on(&mut con, always_ready).unwrap();
        match result {
            PoppedReply::Removed {
                speaker_id,
                user_turn,
                ..
            } => {
                assert_eq!(speaker_id, CHAR_SPEAKER_ID);
                assert_eq!(user_turn.content, "hi");
            }
            _ => panic!("expected the trailing char reply to be removed"),
        }

        let count: i64 = con
            .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn pop_latest_bot_reply_reports_owner_unavailable_and_deletes_nothing() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "hi");
        insert_message_row(&con, "bot1", "hi from bot1");

        let result = Database::pop_latest_bot_reply_on(&mut con, |_speaker_id| false).unwrap();
        assert_eq!(
            result,
            PoppedReply::OwnerUnavailable {
                speaker_id: "bot1".to_string()
            }
        );

        let count: i64 = con
            .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn get_x_messages_after_returns_the_oldest_first_tail_beyond_the_cutoff() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "one");
        insert_message_row(&con, CHAR_SPEAKER_ID, "two");
        insert_message_row(&con, USER_SPEAKER_ID, "three");
        insert_message_row(&con, CHAR_SPEAKER_ID, "four");

        let after_two = Database::get_x_messages_after_on(&con, Some(2), 10).unwrap();
        assert_eq!(
            after_two
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            vec!["three", "four"]
        );

        let no_cutoff = Database::get_x_messages_after_on(&con, None, 10).unwrap();
        assert_eq!(no_cutoff.len(), 4);
    }

    #[test]
    fn get_x_messages_after_respects_the_limit() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        for i in 1..=5 {
            insert_message_row(&con, USER_SPEAKER_ID, &format!("msg {i}"));
        }

        let limited = Database::get_x_messages_after_on(&con, None, 2).unwrap();
        assert_eq!(
            limited
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            vec!["msg 4", "msg 5"]
        );
    }

    #[test]
    fn pop_latest_bot_reply_invalidates_the_message_cache() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "hi");
        insert_message_row(&con, CHAR_SPEAKER_ID, "hello");

        let cache_key = "messages:50:0".to_string();
        {
            let mut cache = MESSAGE_CACHE.lock().unwrap();
            cache.insert(cache_key.clone(), (Vec::new(), Instant::now()));
        }

        Database::pop_latest_bot_reply_on(&mut con, always_ready).unwrap();

        let cache = MESSAGE_CACHE.lock().unwrap();
        assert!(!cache.contains_key(&cache_key));
    }

    /// #181 review finding: `pop_latest_bot_reply_on` is a third
    /// message-delete path (alongside `edit_message_on`/`delete_message_on`)
    /// that can remove a message inside a committed checkpoint's range —
    /// with `short_term_mem: 0` (not validated by `edit_companion`), the
    /// range `select_range` returns can include the very row a regenerate
    /// then pops. It must mark that checkpoint `Stale` exactly like the
    /// other two paths do.
    #[test]
    fn pop_latest_bot_reply_marks_a_covering_committed_checkpoint_stale() {
        use crate::compaction::types::CompactionStatus;

        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "hi");
        insert_message_row(&con, CHAR_SPEAKER_ID, "hello");
        let checkpoint = insert_checkpoint_row(&con, 1, 2, CompactionStatus::Committed);
        con.execute(
            "UPDATE companion SET compacted_through = 2 WHERE id = 1",
            [],
        )
        .unwrap();

        let result = Database::pop_latest_bot_reply_on(&mut con, always_ready).unwrap();
        assert!(matches!(result, PoppedReply::Removed { message_id: 2, .. }));

        assert_eq!(checkpoint_status(&con, checkpoint), CompactionStatus::Stale);
    }

    #[test]
    fn edit_message_keeps_an_ai_reply_marked_as_ai() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        insert_message_row(&con, CHAR_SPEAKER_ID, "hello");

        Database::edit_message_on(
            &mut con,
            1,
            MessageEdit {
                content: "hello, edited".to_string(),
            },
        )
        .unwrap();

        let (ai, content): (bool, String) = con
            .query_row("SELECT ai, content FROM messages WHERE id = 1", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap();
        assert!(ai);
        assert_eq!(content, "hello, edited");
    }

    #[test]
    fn edit_message_keeps_a_remote_bot_reply_attributed_to_its_speaker() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        insert_message_row(&con, "bot1", "hi from bot1");

        Database::edit_message_on(
            &mut con,
            1,
            MessageEdit {
                content: "hi from bot1, edited".to_string(),
            },
        )
        .unwrap();

        let (speaker_id, content): (String, String) = con
            .query_row(
                "SELECT speaker_id, content FROM messages WHERE id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(speaker_id, "bot1");
        assert_eq!(content, "hi from bot1, edited");
    }

    #[test]
    fn edit_message_keeps_a_user_message_marked_as_user() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "hi");

        Database::edit_message_on(
            &mut con,
            1,
            MessageEdit {
                content: "hi, edited".to_string(),
            },
        )
        .unwrap();

        let (ai, content): (bool, String) = con
            .query_row("SELECT ai, content FROM messages WHERE id = 1", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap();
        assert!(!ai);
        assert_eq!(content, "hi, edited");
    }

    #[test]
    fn edit_message_leaves_other_rows_untouched() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "hi");
        insert_message_row(&con, CHAR_SPEAKER_ID, "hello");

        Database::edit_message_on(
            &mut con,
            1,
            MessageEdit {
                content: "hi, edited".to_string(),
            },
        )
        .unwrap();

        let (ai, content): (bool, String) = con
            .query_row("SELECT ai, content FROM messages WHERE id = 2", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap();
        assert!(ai);
        assert_eq!(content, "hello");
    }

    #[test]
    fn edit_message_invalidates_the_message_cache() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        insert_message_row(&con, CHAR_SPEAKER_ID, "hello");

        // Unique to this test (not "messages:50:0", which the
        // pop_latest_bot_reply cache test also uses): MESSAGE_CACHE is
        // process-global, so a shared key can be reinserted by a parallel
        // test between this test's clear and its assertion, making the
        // assertion flaky.
        let cache_key = "edit-message-cache-invalidation".to_string();
        {
            let mut cache = MESSAGE_CACHE.lock().unwrap();
            cache.insert(cache_key.clone(), (Vec::new(), Instant::now()));
        }

        Database::edit_message_on(
            &mut con,
            1,
            MessageEdit {
                content: "hello, edited".to_string(),
            },
        )
        .unwrap();

        let cache = MESSAGE_CACHE.lock().unwrap();
        assert!(!cache.contains_key(&cache_key));
    }

    /// Reads back one `compactions` row's status, for the #181 tests below.
    fn checkpoint_status(con: &Connection, id: i64) -> crate::compaction::types::CompactionStatus {
        con.query_row("SELECT status FROM compactions WHERE id = ?", [id], |row| {
            row.get(0)
        })
        .unwrap()
    }

    #[test]
    fn edit_inside_committed_range_marks_only_that_checkpoint_stale() {
        use crate::compaction::types::CompactionStatus;

        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        for i in 1..=20 {
            insert_message_row(&con, USER_SPEAKER_ID, &format!("msg {i}"));
        }
        let first = insert_checkpoint_row(&con, 1, 10, CompactionStatus::Committed);
        let second = insert_checkpoint_row(&con, 11, 20, CompactionStatus::Committed);
        con.execute(
            "UPDATE companion SET compacted_through = 20 WHERE id = 1",
            [],
        )
        .unwrap();

        Database::edit_message_on(
            &mut con,
            5,
            MessageEdit {
                content: "edited".to_string(),
            },
        )
        .unwrap();

        assert_eq!(checkpoint_status(&con, first), CompactionStatus::Stale);
        assert_eq!(checkpoint_status(&con, second), CompactionStatus::Committed);
    }

    #[test]
    fn edit_after_compacted_through_leaves_checkpoints_committed() {
        use crate::compaction::types::CompactionStatus;

        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        for i in 1..=15 {
            insert_message_row(&con, USER_SPEAKER_ID, &format!("msg {i}"));
        }
        let checkpoint = insert_checkpoint_row(&con, 1, 10, CompactionStatus::Committed);
        con.execute(
            "UPDATE companion SET compacted_through = 10 WHERE id = 1",
            [],
        )
        .unwrap();

        // Id 12 is in the uncompacted tail, past every checkpoint's range.
        Database::edit_message_on(
            &mut con,
            12,
            MessageEdit {
                content: "edited".to_string(),
            },
        )
        .unwrap();

        assert_eq!(
            checkpoint_status(&con, checkpoint),
            CompactionStatus::Committed
        );
    }

    #[test]
    fn edit_on_never_compacted_chat_touches_no_compaction_rows() {
        use crate::compaction::types::CompactionStatus;

        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "hi");
        // A `Committed` checkpoint row exists but `compacted_through` is
        // still NULL, as if seeded out of band: `discard_draft_containing_on`
        // runs unconditionally (it only ever matches `Draft` rows, so it is
        // a no-op here) but the NULL short-circuit means
        // `mark_stale_containing_on` must never even query this row, let
        // alone flip it.
        let checkpoint = insert_checkpoint_row(&con, 1, 1, CompactionStatus::Committed);

        Database::edit_message_on(
            &mut con,
            1,
            MessageEdit {
                content: "edited".to_string(),
            },
        )
        .unwrap();

        assert_eq!(
            checkpoint_status(&con, checkpoint),
            CompactionStatus::Committed
        );
    }

    #[test]
    fn delete_pinned_message_removes_the_pin_and_marks_stale() {
        use crate::compaction::types::CompactionStatus;

        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        for i in 1..=5 {
            insert_message_row(&con, USER_SPEAKER_ID, &format!("msg {i}"));
        }
        let checkpoint = insert_checkpoint_row(&con, 1, 5, CompactionStatus::Committed);
        con.execute(
            "UPDATE companion SET compacted_through = 5 WHERE id = 1",
            [],
        )
        .unwrap();
        crate::compaction::store::pin_on(&con, 3).unwrap();

        // Must not fail with a foreign-key constraint error even though
        // `foreign_keys = ON`: the pin cascades away with the message.
        Database::delete_message_on(&mut con, 3).unwrap();

        let pin_count: i64 = con
            .query_row("SELECT COUNT(*) FROM pinned_messages", [], |row| row.get(0))
            .unwrap();
        assert_eq!(pin_count, 0);
        assert_eq!(checkpoint_status(&con, checkpoint), CompactionStatus::Stale);
    }

    /// #181 review finding: `mark_stale_containing_on` only ever matches
    /// `Committed` rows, so a pending `Draft` checkpoint whose range an edit
    /// falls inside was left completely untouched — committing it later
    /// would produce a `Committed` checkpoint describing pre-edit content.
    /// `discard_draft_containing_on` closes that gap.
    #[test]
    fn edit_inside_a_pending_drafts_range_discards_it_and_leaves_a_committed_sibling_untouched() {
        use crate::compaction::types::CompactionStatus;

        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        for i in 1..=15 {
            insert_message_row(&con, USER_SPEAKER_ID, &format!("msg {i}"));
        }
        let committed = insert_checkpoint_row(&con, 1, 10, CompactionStatus::Committed);
        con.execute(
            "UPDATE companion SET compacted_through = 10 WHERE id = 1",
            [],
        )
        .unwrap();
        let pending_draft = insert_checkpoint_row(&con, 11, 15, CompactionStatus::Draft);

        Database::edit_message_on(
            &mut con,
            13,
            MessageEdit {
                content: "edited".to_string(),
            },
        )
        .unwrap();

        assert_eq!(
            checkpoint_status(&con, pending_draft),
            CompactionStatus::Discarded
        );
        assert_eq!(
            checkpoint_status(&con, committed),
            CompactionStatus::Committed
        );
    }

    #[test]
    fn delete_inside_a_pending_drafts_range_discards_it() {
        use crate::compaction::types::CompactionStatus;

        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        for i in 1..=15 {
            insert_message_row(&con, USER_SPEAKER_ID, &format!("msg {i}"));
        }
        let pending_draft = insert_checkpoint_row(&con, 11, 15, CompactionStatus::Draft);

        Database::delete_message_on(&mut con, 13).unwrap();

        assert_eq!(
            checkpoint_status(&con, pending_draft),
            CompactionStatus::Discarded
        );
    }

    /// The draft check must run *before* the `compacted_through IS NULL`
    /// short-circuit: a chat's very first draft is pending while
    /// `compacted_through` is still `NULL` (nothing has committed yet).
    #[test]
    fn edit_inside_the_chats_very_first_pending_draft_discards_it_even_though_compacted_through_is_still_null(
    ) {
        use crate::compaction::types::CompactionStatus;

        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_compaction_tables(&con);
        for i in 1..=5 {
            insert_message_row(&con, USER_SPEAKER_ID, &format!("msg {i}"));
        }
        let pending_draft = insert_checkpoint_row(&con, 1, 5, CompactionStatus::Draft);

        Database::edit_message_on(
            &mut con,
            3,
            MessageEdit {
                content: "edited".to_string(),
            },
        )
        .unwrap();

        assert_eq!(
            checkpoint_status(&con, pending_draft),
            CompactionStatus::Discarded
        );
    }

    #[test]
    fn erase_messages_clears_every_compaction_table_and_resets_compacted_through() {
        use crate::compaction::types::CompactionStatus;

        let dir = tempfile::TempDir::new().unwrap();
        let mut con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        create_companion_table(&con);
        create_user_table(&con);
        create_compaction_tables(&con);
        // #177's `delete_compaction_persons_in` runs inside this same
        // transaction now, so its table needs to exist here too.
        create_third_party_tables(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "hi");
        let checkpoint = insert_checkpoint_row(&con, 1, 1, CompactionStatus::Committed);
        con.execute(
            "INSERT INTO compaction_facts (compaction_id, category, subject, text, quote_speaker, sources, replaces, canon, active) VALUES (?, 'milestone', NULL, 'a fact', NULL, '[1]', '[]', 1, 1)",
            [checkpoint],
        )
        .unwrap();
        crate::compaction::store::pin_on(&con, 1).unwrap();
        con.execute(
            "UPDATE companion SET compacted_through = 1 WHERE id = 1",
            [],
        )
        .unwrap();

        Database::erase_messages_on(&mut con).unwrap();

        let checkpoint_count: i64 = con
            .query_row("SELECT COUNT(*) FROM compactions", [], |row| row.get(0))
            .unwrap();
        let fact_count: i64 = con
            .query_row("SELECT COUNT(*) FROM compaction_facts", [], |row| {
                row.get(0)
            })
            .unwrap();
        let pin_count: i64 = con
            .query_row("SELECT COUNT(*) FROM pinned_messages", [], |row| row.get(0))
            .unwrap();
        assert_eq!(checkpoint_count, 0);
        assert_eq!(fact_count, 0);
        assert_eq!(pin_count, 0);

        let compacted_through: Option<i32> = con
            .query_row(
                "SELECT compacted_through FROM companion WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(compacted_through, None);

        // The greeting is still inserted.
        let message_count: i64 = con
            .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
            .unwrap();
        assert_eq!(message_count, 1);
    }

    #[test]
    fn concurrent_read_succeeds_while_a_write_transaction_is_held() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("t.db");

        let setup = Database::open_at(&db_path).unwrap();
        setup
            .execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", [])
            .unwrap();
        setup.execute("INSERT INTO t (id) VALUES (1)", []).unwrap();

        let barrier = Arc::new(Barrier::new(2));
        let writer_barrier = Arc::clone(&barrier);
        let commit_gate = Arc::new(Barrier::new(2));
        let writer_commit_gate = Arc::clone(&commit_gate);
        let writer_path = db_path.clone();

        let writer = thread::spawn(move || {
            let mut con = Database::open_at(&writer_path).unwrap();
            let tx = con
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .unwrap();
            // This second row stays uncommitted while the reader below runs,
            // so the reader's snapshot must not include it.
            tx.execute("INSERT INTO t (id) VALUES (2)", []).unwrap();
            writer_barrier.wait();
            // Commit only after the reader has taken and asserted its
            // snapshot, so the test cannot race the scheduler.
            writer_commit_gate.wait();
            tx.commit().unwrap();
        });

        barrier.wait();
        let reader = Database::open_at(&db_path).unwrap();
        // WAL readers see a snapshot as of the start of their read: the
        // pre-existing committed row, not the writer's uncommitted insert.
        let count: i64 = reader
            .query_row("SELECT COUNT(*) FROM t", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
        commit_gate.wait();

        writer.join().unwrap();
    }

    #[test]
    fn open_sets_a_busy_timeout_so_a_second_writer_waits_instead_of_failing() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("t.db");

        let setup = Database::open_at(&db_path).unwrap();
        setup
            .execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", [])
            .unwrap();

        let barrier = Arc::new(Barrier::new(2));
        let writer_barrier = Arc::clone(&barrier);
        let writer_path = db_path.clone();

        let writer = thread::spawn(move || {
            let mut con = Database::open_at(&writer_path).unwrap();
            let tx = con
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .unwrap();
            writer_barrier.wait();
            thread::sleep(Duration::from_millis(100));
            tx.commit().unwrap();
        });

        barrier.wait();
        let mut second = Database::open_at(&db_path).unwrap();
        let tx2 = second.transaction_with_behavior(TransactionBehavior::Immediate);
        assert!(tx2.is_ok(), "busy handler should wait rather than fail");
        tx2.unwrap().commit().unwrap();

        writer.join().unwrap();
    }

    #[test]
    fn test_get_current_date() {
        let date = get_current_date();
        assert!(!date.is_empty());
        assert!(date.len() > 10);
    }

    #[test]
    fn test_contains_time_question() {
        assert!(contains_time_question("What time is it?"));
        assert!(contains_time_question("What's the date today?"));
        assert!(contains_time_question("It's morning here"));
        assert!(!contains_time_question("How are you doing?"));
        assert!(!contains_time_question("Tell me a story"));
    }

    #[test]
    fn test_message_struct() {
        let message = Message {
            id: 1,
            ai: true,
            speaker_id: CHAR_SPEAKER_ID.to_string(),
            content: "Hello world".to_string(),
            created_at: "2024-01-15 10:00".to_string(),
        };

        assert_eq!(message.id, 1);
        assert!(message.ai);
        assert_eq!(message.content, "Hello world");
    }

    #[test]
    fn test_new_message_struct() {
        let new_message = NewMessage::from_user("User message");

        assert!(!is_ai_speaker(&new_message.speaker_id));
        assert_eq!(new_message.content, "User message");
    }

    #[test]
    fn drop_excluded_names_is_case_insensitive_and_keeps_unmatched_names() {
        let names = vec!["Bob".to_string(), "bob".to_string(), "Carol".to_string()];
        let excluded = vec!["Bob".to_string()];
        assert_eq!(
            Database::drop_excluded_names(names, &excluded),
            vec!["Carol".to_string()]
        );
    }

    #[test]
    fn drop_excluded_names_folds_case_beyond_ascii() {
        let names = vec!["zoë".to_string(), "Carol".to_string()];
        let excluded = vec!["Zoë".to_string()];
        assert_eq!(
            Database::drop_excluded_names(names, &excluded),
            vec!["Carol".to_string()]
        );
    }

    /// `extract_person_names`'s patterns capture only the first token of a
    /// multi-word name ("Mary" from "met Mary Jane"), so `drop_excluded_names`
    /// must still recognise that lone "Mary" as the excluded participant
    /// "Mary Jane" — the #127 review finding this pins down. Both callers
    /// of `drop_excluded_names` (`detect_new_persons_in_message`,
    /// `track_third_party_mentions`) share this logic, so a test at this
    /// level covers both; neither caller has existing test coverage of its
    /// own to extend (both require a live `Database` connection this file's
    /// other name-extraction tests deliberately avoid).
    #[test]
    fn drop_excluded_names_matches_a_single_word_of_a_multi_word_excluded_name() {
        let names = vec!["Mary".to_string(), "Carol".to_string()];
        let excluded = vec!["Mary Jane".to_string()];
        assert_eq!(
            Database::drop_excluded_names(names, &excluded),
            vec!["Carol".to_string()]
        );
    }

    #[test]
    fn test_person_name_extraction() {
        // Test valid person names with clear context
        let names = Database::extract_person_names(
            "I met with John and Sarah yesterday. John said he likes the project.",
        );
        assert!(names.contains(&"John".to_string()));
        assert!(names.contains(&"Sarah".to_string()));

        // Test with relationship indicators
        let names2 =
            Database::extract_person_names("My friend Alex called me. Dr. Smith visited today.");
        assert!(names2.contains(&"Alex".to_string()));
        assert!(names2.contains(&"Smith".to_string()));

        // Test empty string
        let names3 = Database::extract_person_names("The weather is nice today.");
        assert!(names3.is_empty());

        // Test that body parts are NOT extracted
        let names4 = Database::extract_person_names(
            "Put your hand on your shoulder. The class starts at 9.",
        );
        assert!(!names4.contains(&"Hand".to_string()));
        assert!(!names4.contains(&"Shoulder".to_string()));
        assert!(!names4.contains(&"Class".to_string()));

        // Test that objects are NOT extracted
        let names5 =
            Database::extract_person_names("The door is open. The table has a book on it.");
        assert!(!names5.contains(&"Door".to_string()));
        assert!(!names5.contains(&"Table".to_string()));
        assert!(!names5.contains(&"Book".to_string()));
    }

    #[test]
    fn test_is_likely_person_name() {
        // Valid person names
        assert!(Database::is_likely_person_name("John"));
        assert!(Database::is_likely_person_name("Mary-Jane"));
        assert!(Database::is_likely_person_name("O'Connor"));
        assert!(Database::is_likely_person_name("Sarah"));
        assert!(Database::is_likely_person_name("Michael"));

        // Common words that should be filtered
        assert!(!Database::is_likely_person_name("the"));
        assert!(!Database::is_likely_person_name("and"));
        assert!(!Database::is_likely_person_name("if"));
        assert!(!Database::is_likely_person_name("a"));

        // Body parts that should be filtered
        assert!(!Database::is_likely_person_name("hand"));
        assert!(!Database::is_likely_person_name("shoulder"));
        assert!(!Database::is_likely_person_name("head"));
        assert!(!Database::is_likely_person_name("arm"));
        assert!(!Database::is_likely_person_name("leg"));

        // Objects that should be filtered
        assert!(!Database::is_likely_person_name("class"));
        assert!(!Database::is_likely_person_name("table"));
        assert!(!Database::is_likely_person_name("door"));
        assert!(!Database::is_likely_person_name("book"));
        assert!(!Database::is_likely_person_name("computer"));

        // Words with non-name suffixes
        assert!(!Database::is_likely_person_name("walking"));
        assert!(!Database::is_likely_person_name("creation"));
        assert!(!Database::is_likely_person_name("happiness"));
        assert!(!Database::is_likely_person_name("movement"));
        assert!(!Database::is_likely_person_name("quickly"));
    }

    #[test]
    fn test_is_proper_name_context() {
        // Test possessive forms
        assert!(Database::is_proper_name_context(
            "John",
            "John's car is red"
        ));
        assert!(Database::is_proper_name_context(
            "Sarah",
            "Sarah's house is nearby"
        ));

        // Test with titles
        assert!(Database::is_proper_name_context(
            "Smith",
            "Dr. Smith arrived"
        ));
        assert!(Database::is_proper_name_context(
            "Johnson",
            "Mrs. Johnson called"
        ));

        // Test with person-related verbs
        assert!(Database::is_proper_name_context("Alex", "Alex said hello"));
        assert!(Database::is_proper_name_context(
            "Maria",
            "I met Maria yesterday"
        ));
    }

    #[test]
    fn test_is_person_indicator() {
        // Words that indicate person context
        assert!(Database::is_person_indicator("met"));
        assert!(Database::is_person_indicator("friend"));
        assert!(Database::is_person_indicator("told"));
        assert!(Database::is_person_indicator("colleague"));

        // Words that don't indicate person context
        assert!(!Database::is_person_indicator("table"));
        assert!(!Database::is_person_indicator("quickly"));
        assert!(!Database::is_person_indicator("blue"));
    }

    #[test]
    fn test_capitalize_name() {
        assert_eq!(Database::capitalize_name("john"), "John");
        assert_eq!(Database::capitalize_name("mary-jane"), "Mary-Jane");
        assert_eq!(Database::capitalize_name("o'connor"), "O'Connor");
        assert_eq!(Database::capitalize_name("jean-luc"), "Jean-Luc");
    }

    /// Minimal `companion` table with one row, since `init()` is hard-wired
    /// to `paths::db_path()` and these tests run against a `TempDir` instead.
    fn create_companion_row(con: &Connection) {
        con.execute(
            "CREATE TABLE companion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT
            )",
            [],
        )
        .unwrap();
        con.execute("INSERT INTO companion (id, name) VALUES (1, 'Test')", [])
            .unwrap();
    }

    /// The pre-#110 `attitude_memories` DDL, literal `REFERENCES
    /// companions(id)` typo included, so the migration is exercised against
    /// what real databases actually contain.
    fn create_legacy_attitude_memories_table(con: &Connection) {
        con.execute(
            "CREATE TABLE attitude_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                companion_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                target_type TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                description TEXT NOT NULL,
                priority_score REAL NOT NULL,
                attitude_delta_json TEXT NOT NULL,
                impact_score REAL NOT NULL,
                message_context TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(companion_id) REFERENCES companions(id)
            )",
            [],
        )
        .unwrap();
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_attitude_memories_priority
             ON attitude_memories(companion_id, priority_score DESC)",
            [],
        )
        .unwrap();
    }

    fn insert_attitude_memory_row(con: &Connection, id: i32, companion_id: i32, description: &str) {
        con.execute(
            "INSERT INTO attitude_memories (
                id, companion_id, target_id, target_type, memory_type, description,
                priority_score, attitude_delta_json, impact_score, message_context, created_at
            ) VALUES (?, ?, 1, 'user', 'shift', ?, 0.5, '{}', 0.5, NULL, '2024-01-01')",
            params![id, companion_id, description],
        )
        .unwrap();
    }

    #[test]
    fn open_at_enables_foreign_keys() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();

        let enabled: i64 = con
            .pragma_query_value(None, "foreign_keys", |row| row.get(0))
            .unwrap();
        assert_eq!(enabled, 1);
    }

    #[test]
    fn legacy_attitude_memories_foreign_key_is_rebuilt() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();

        // Setting up the legacy schema needs foreign keys off: the old FK
        // target table (`companions`) never existed.
        con.pragma_update(None, "foreign_keys", false).unwrap();
        create_companion_row(&con);
        create_legacy_attitude_memories_table(&con);
        insert_attitude_memory_row(&con, 1, 1, "first");
        insert_attitude_memory_row(&con, 2, 1, "second");
        con.pragma_update(None, "foreign_keys", true).unwrap();

        Database::migrate_attitude_memories_foreign_key(&con).unwrap();

        let mut fk_stmt = con
            .prepare("PRAGMA foreign_key_list(attitude_memories)")
            .unwrap();
        let fks: Vec<(String, String)> = fk_stmt
            .query_map([], |row| Ok((row.get(2)?, row.get(6)?)))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();
        drop(fk_stmt);
        assert_eq!(fks, vec![("companion".to_string(), "CASCADE".to_string())]);

        let mut select_stmt = con
            .prepare("SELECT id, description FROM attitude_memories ORDER BY id")
            .unwrap();
        let rows: Vec<(i32, String)> = select_stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();
        drop(select_stmt);
        assert_eq!(
            rows,
            vec![(1, "first".to_string()), (2, "second".to_string())]
        );

        let index_count: i64 = con
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'index' AND name = 'idx_attitude_memories_priority'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(index_count, 1);

        let fk_enabled: i64 = con
            .pragma_query_value(None, "foreign_keys", |row| row.get(0))
            .unwrap();
        assert_eq!(fk_enabled, 1);
    }

    #[test]
    fn migration_keeps_orphan_rows() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();

        con.pragma_update(None, "foreign_keys", false).unwrap();
        create_companion_row(&con);
        create_legacy_attitude_memories_table(&con);
        insert_attitude_memory_row(&con, 1, 99, "orphan");
        con.pragma_update(None, "foreign_keys", true).unwrap();

        Database::migrate_attitude_memories_foreign_key(&con).unwrap();

        let count: i64 = con
            .query_row(
                "SELECT COUNT(*) FROM attitude_memories WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "existing rows must survive the migration");
    }

    #[test]
    fn corrected_attitude_memories_table_is_left_alone() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_companion_row(&con);
        Database::create_attitude_memories_table(&con).unwrap();

        Database::migrate_attitude_memories_foreign_key(&con).unwrap();

        let mut stmt = con
            .prepare("PRAGMA foreign_key_list(attitude_memories)")
            .unwrap();
        let targets: Vec<String> = stmt
            .query_map([], |row| row.get(2))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();
        assert_eq!(targets, vec!["companion".to_string()]);
    }

    #[test]
    fn attitude_memory_insert_succeeds_with_foreign_keys_on() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_companion_row(&con);
        Database::create_attitude_memories_table(&con).unwrap();

        let result = con.execute(
            "INSERT INTO attitude_memories (
                companion_id, target_id, target_type, memory_type, description,
                priority_score, attitude_delta_json, impact_score, message_context, created_at
            ) VALUES (1, 1, 'user', 'shift', 'test', 0.5, '{}', 0.5, NULL, '2024-01-01')",
            [],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn orphan_attitude_memory_insert_is_rejected() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_companion_row(&con);
        Database::create_attitude_memories_table(&con).unwrap();

        let result = con.execute(
            "INSERT INTO attitude_memories (
                companion_id, target_id, target_type, memory_type, description,
                priority_score, attitude_delta_json, impact_score, message_context, created_at
            ) VALUES (42, 1, 'user', 'shift', 'test', 0.5, '{}', 0.5, NULL, '2024-01-01')",
            [],
        );

        match result {
            Err(Error::SqliteFailure(e, _)) => {
                assert_eq!(e.code, rusqlite::ErrorCode::ConstraintViolation);
            }
            other => panic!("expected a foreign key constraint violation, got {other:?}"),
        }
    }

    /// A minimal `AttitudeMemoryDraft` a test only needs to override a field
    /// or two of, mirroring `simple_tests.rs`'s `memory_fixture` for
    /// `AttitudeMemory` rows.
    fn draft_fixture(memory_type: &str, priority_score: f32) -> AttitudeMemoryDraft {
        AttitudeMemoryDraft {
            memory_type: memory_type.to_string(),
            description: format!("a {memory_type} memory"),
            priority_score,
            impact_score: 20.0,
            delta: AttitudeDelta {
                attraction: 0.0,
                trust: 5.0,
                fear: 0.0,
                anger: 0.0,
                joy: 0.0,
                sorrow: 0.0,
                disgust: 0.0,
                surprise: 0.0,
                curiosity: 0.0,
                respect: 0.0,
                suspicion: 0.0,
                gratitude: 0.0,
                jealousy: 0.0,
                empathy: 0.0,
                lust: 0.0,
                love: 0.0,
                anxiety: 0.0,
                butterflies: 0.0,
                submissiveness: 0.0,
                dominance: 0.0,
            },
        }
    }

    #[test]
    fn insert_attitude_memory_then_prune_still_caps_at_the_configured_limit() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_companion_row(&con);
        Database::create_attitude_memories_table(&con).unwrap();

        for i in 0..5 {
            Database::insert_attitude_memory_on(
                &con,
                1,
                1,
                "user",
                &draft_fixture("SignificantChange", i as f32),
                "",
            )
            .unwrap();
        }
        // The insert helper already prunes after every write; asking for a
        // stricter cap here exercises the same statement `prune_attitude_memories`
        // itself runs, against the rows this test just inserted.
        Database::prune_attitude_memories_on(&con, 1, 2).unwrap();

        let count: usize = con
            .query_row(
                "SELECT COUNT(*) FROM attitude_memories WHERE companion_id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 2);

        // The two highest-priority rows (priority 3 and 4) are the ones kept.
        let mut stmt = con
            .prepare("SELECT priority_score FROM attitude_memories WHERE companion_id = 1 ORDER BY priority_score DESC")
            .unwrap();
        let kept: Vec<f32> = stmt
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();
        assert_eq!(kept, vec![4.0, 3.0]);
    }

    #[test]
    fn recalibration_memory_draft_outranks_a_significant_change_of_equal_impact() {
        let previous = attitude_fixture();
        let mut new = previous.clone();
        // A lone, moderate trust move: too small for any of
        // `classify_memory_type`'s named buckets, so `evaluate_attitude_shift`
        // falls through to its uncategorised `"SignificantChange"` default —
        // the actual "equal impact" comparison this test names.
        new.trust += 12.0;

        let recalibration = recalibration_memory_draft(&previous, &new, 42);
        let turn_scored =
            evaluate_attitude_shift(&previous, &new).expect("shift is significant enough");

        assert_eq!(turn_scored.memory_type, "SignificantChange");
        assert_eq!(recalibration.memory_type, "NarrativeRecalibration");
        assert!(recalibration.description.contains("42"));
        assert_eq!(recalibration.impact_score, turn_scored.impact_score);
        assert!(recalibration.priority_score > turn_scored.priority_score);
    }

    #[test]
    fn recalibration_memory_draft_has_no_significance_gate() {
        let previous = attitude_fixture();
        let mut new = previous.clone();
        new.trust += 0.5; // Well under `SIGNIFICANT_IMPACT_THRESHOLD`.

        assert!(evaluate_attitude_shift(&previous, &new).is_none());
        // `recalibration_memory_draft` has no threshold to clear: a narrative
        // rating is always worth remembering, however small the move.
        let draft = recalibration_memory_draft(&previous, &new, 7);
        assert_eq!(draft.memory_type, "NarrativeRecalibration");
    }

    /// A `CompanionAttitude` every field of which is neutral, for tests that
    /// only care about one or two dimensions' movement. Mirrors
    /// `simple_tests.rs`'s `attitude_fixture`.
    fn attitude_fixture() -> CompanionAttitude {
        CompanionAttitude {
            id: Some(1),
            companion_id: 1,
            target_id: 1,
            target_type: "user".to_string(),
            attraction: 0.0,
            trust: 0.0,
            fear: 0.0,
            anger: 0.0,
            joy: 0.0,
            sorrow: 0.0,
            disgust: 0.0,
            surprise: 0.0,
            curiosity: 0.0,
            respect: 0.0,
            suspicion: 0.0,
            gratitude: 0.0,
            jealousy: 0.0,
            empathy: 0.0,
            lust: 0.0,
            love: 0.0,
            anxiety: 0.0,
            butterflies: 0.0,
            submissiveness: 0.0,
            dominance: 0.0,
            relationship_score: Some(0.0),
            last_updated: "now".to_string(),
            created_at: "now".to_string(),
        }
    }

    #[test]
    fn migrate_messages_speaker_id_backfills_legacy_rows() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_legacy_messages_table(&con);
        con.execute(
            "INSERT INTO messages (ai, content, created_at) VALUES (1, 'hi from char', ?)",
            [get_current_date()],
        )
        .unwrap();
        con.execute(
            "INSERT INTO messages (ai, content, created_at) VALUES (0, 'hi from user', ?)",
            [get_current_date()],
        )
        .unwrap();

        Database::migrate_messages_speaker_id(&con).unwrap();

        let mut stmt = con
            .prepare("SELECT ai, speaker_id FROM messages ORDER BY id")
            .unwrap();
        let rows: Vec<(bool, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();
        assert_eq!(
            rows,
            vec![
                (true, CHAR_SPEAKER_ID.to_string()),
                (false, USER_SPEAKER_ID.to_string()),
            ]
        );
        assert!(rows.iter().all(|(_, speaker_id)| !speaker_id.is_empty()));
    }

    #[test]
    fn migrate_messages_speaker_id_is_idempotent_on_a_migrated_table() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_legacy_messages_table(&con);
        con.execute(
            "INSERT INTO messages (ai, content, created_at) VALUES (1, 'hi', ?)",
            [get_current_date()],
        )
        .unwrap();

        Database::migrate_messages_speaker_id(&con).unwrap();
        Database::migrate_messages_speaker_id(&con).unwrap();

        let speaker_id: String = con
            .query_row("SELECT speaker_id FROM messages WHERE id = 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(speaker_id, CHAR_SPEAKER_ID);
    }

    #[test]
    fn migrate_messages_speaker_id_run_concurrently_by_two_connections_does_not_error() {
        // Both connections open the column-missing table before either
        // starts migrating. Without the fix, the second to reach `ALTER
        // TABLE` fails with a duplicate-column error instead of blocking
        // on the first (via the busy timeout) and then seeing the column
        // already there.
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("t.db");
        let setup = Database::open_at(&db_path).unwrap();
        create_legacy_messages_table(&setup);
        setup
            .execute(
                "INSERT INTO messages (ai, content, created_at) VALUES (1, 'hi', ?)",
                [get_current_date()],
            )
            .unwrap();
        drop(setup);

        let barrier = Arc::new(Barrier::new(2));
        let other_barrier = Arc::clone(&barrier);
        let other_path = db_path.clone();

        let other = thread::spawn(move || {
            let con = Database::open_at(&other_path).unwrap();
            other_barrier.wait();
            Database::migrate_messages_speaker_id(&con)
        });

        let con = Database::open_at(&db_path).unwrap();
        barrier.wait();
        let result = Database::migrate_messages_speaker_id(&con);

        assert!(result.is_ok(), "expected Ok, got {result:?}");
        assert!(
            other.join().unwrap().is_ok(),
            "the concurrent migration call should also succeed, not hit a duplicate column error"
        );

        let speaker_id: String = con
            .query_row("SELECT speaker_id FROM messages WHERE id = 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(speaker_id, CHAR_SPEAKER_ID);
    }

    #[test]
    fn insert_message_on_writes_ai_in_sync_with_speaker_id() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);

        Database::insert_message_on(&con, NewMessage::from_user("hi")).unwrap();
        Database::insert_message_on(&con, NewMessage::new("bot1", "hello")).unwrap();

        let mut stmt = con
            .prepare("SELECT ai, speaker_id FROM messages ORDER BY id")
            .unwrap();
        let rows: Vec<(bool, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();
        assert_eq!(
            rows,
            vec![(false, "user".to_string()), (true, "bot1".to_string())]
        );
    }

    #[test]
    fn get_messages_between_on_is_inclusive_and_ordered_by_id() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        for i in 1..=5 {
            insert_message_row(&con, USER_SPEAKER_ID, &format!("message {i}"));
        }

        let messages = Database::get_messages_between_on(&con, 2, 4).unwrap();
        let ids: Vec<i32> = messages.iter().map(|m| m.id).collect();
        assert_eq!(ids, vec![2, 3, 4]);
    }

    #[test]
    fn get_messages_between_on_an_empty_range_is_an_empty_vec() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_messages_table(&con);
        insert_message_row(&con, USER_SPEAKER_ID, "hi");

        assert!(Database::get_messages_between_on(&con, 100, 200)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn resolve_speaker_only_ai_true_gives_char() {
        assert_eq!(
            resolve_speaker(Some(true), None),
            Ok(CHAR_SPEAKER_ID.to_string())
        );
    }

    #[test]
    fn resolve_speaker_only_ai_false_gives_user() {
        assert_eq!(
            resolve_speaker(Some(false), None),
            Ok(USER_SPEAKER_ID.to_string())
        );
    }

    #[test]
    fn resolve_speaker_speaker_id_alone_is_kept_verbatim() {
        assert_eq!(
            resolve_speaker(None, Some("bot1".to_string())),
            Ok("bot1".to_string())
        );
    }

    #[test]
    fn resolve_speaker_agreeing_pair_is_fine() {
        assert_eq!(
            resolve_speaker(Some(true), Some(CHAR_SPEAKER_ID.to_string())),
            Ok(CHAR_SPEAKER_ID.to_string())
        );
        assert_eq!(
            resolve_speaker(Some(false), Some(USER_SPEAKER_ID.to_string())),
            Ok(USER_SPEAKER_ID.to_string())
        );
    }

    #[test]
    fn resolve_speaker_disagreeing_pair_errors() {
        assert_eq!(
            resolve_speaker(Some(false), Some(CHAR_SPEAKER_ID.to_string())),
            Err(SpeakerResolveError::Conflict)
        );
        assert_eq!(
            resolve_speaker(Some(true), Some(USER_SPEAKER_ID.to_string())),
            Err(SpeakerResolveError::Conflict)
        );
    }

    #[test]
    fn resolve_speaker_neither_field_errors() {
        assert_eq!(
            resolve_speaker(None, None),
            Err(SpeakerResolveError::Missing)
        );
    }

    #[test]
    fn resolve_speaker_empty_speaker_id_errors() {
        assert_eq!(
            resolve_speaker(None, Some(String::new())),
            Err(SpeakerResolveError::Missing)
        );
    }

    #[test]
    fn resolve_speaker_empty_speaker_id_errors_even_with_a_legacy_ai_fallback() {
        // An explicitly empty `speaker_id` must not fall through to the
        // legacy `ai`-only branch, for either value of `ai`.
        assert_eq!(
            resolve_speaker(Some(true), Some(String::new())),
            Err(SpeakerResolveError::Missing)
        );
        assert_eq!(
            resolve_speaker(Some(false), Some(String::new())),
            Err(SpeakerResolveError::Missing)
        );
    }

    #[test]
    fn new_message_request_with_only_ai_deserializes() {
        let request: NewMessageRequest =
            serde_json::from_str(r#"{"ai": true, "content": "hi"}"#).unwrap();
        let message = NewMessage::try_from(request).unwrap();
        assert_eq!(message.speaker_id, CHAR_SPEAKER_ID);
        assert_eq!(message.content, "hi");
    }

    // --- config / multiplayer (#128) ---

    /// The four original `config` columns, matching what `init()` created
    /// before #128 (and before the seven columns #94/#101/etc. added
    /// earlier still) - what `migrate_config_table` needs to backfill on an
    /// old database.
    fn create_legacy_config_table(con: &Connection) {
        con.execute(
            "CREATE TABLE IF NOT EXISTS config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device TEXT,
                llm_model_path TEXT,
                gpu_layers INTEGER,
                prompt_template TEXT
            )",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO config (device, llm_model_path, gpu_layers, prompt_template) VALUES ('CPU', '', 0, 'Auto')",
            [],
        )
        .unwrap();
    }

    /// The full, current `config` DDL (mirrors `init()`), plus a single
    /// seed row, for tests that exercise `read_config`/`write_config`
    /// directly without going through `migrate_config_table`.
    fn create_config_table(con: &Connection) {
        con.execute(
            "CREATE TABLE IF NOT EXISTS config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device TEXT,
                llm_model_path TEXT,
                gpu_layers INTEGER,
                prompt_template TEXT,
                context_window_size INTEGER DEFAULT 2048,
                max_response_tokens INTEGER DEFAULT 512,
                enable_dynamic_context BOOLEAN DEFAULT true,
                vram_limit_gb INTEGER DEFAULT 4,
                dynamic_gpu_allocation BOOLEAN DEFAULT true,
                gpu_safety_margin REAL DEFAULT 0.8,
                min_free_vram_mb INTEGER DEFAULT 512,
                enable_hybrid_context BOOLEAN DEFAULT true,
                max_system_ram_usage_gb INTEGER DEFAULT 8,
                context_expansion_strategy TEXT DEFAULT 'balanced',
                ram_safety_margin_gb INTEGER DEFAULT 2,
                multiplayer_mode TEXT DEFAULT 'solo',
                multiplayer_password TEXT DEFAULT '',
                multiplayer_host_address TEXT DEFAULT '',
                multiplayer_participant_id TEXT DEFAULT '',
                mention_followup_depth INTEGER DEFAULT 1,
                remote_generation_timeout_secs INTEGER DEFAULT 120,
                compact_threshold_tokens INTEGER,
                compact_min_messages INTEGER DEFAULT 8,
                compaction_model_path TEXT,
                heuristic_person_detection BOOLEAN DEFAULT false,
                compaction_attitude_weight REAL DEFAULT 0.5
            )",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO config (device, llm_model_path, gpu_layers, prompt_template) VALUES ('CPU', '', 0, 'Auto')",
            [],
        )
        .unwrap();
    }

    /// A `ConfigModify` that passes every validation rule, for tests to
    /// mutate the one or two fields they care about.
    fn valid_config_modify() -> ConfigModify {
        ConfigModify {
            device: "CPU".to_string(),
            llm_model_path: String::new(),
            gpu_layers: 0,
            prompt_template: "Auto".to_string(),
            context_window_size: 2048,
            max_response_tokens: 512,
            enable_dynamic_context: true,
            vram_limit_gb: 4,
            dynamic_gpu_allocation: true,
            gpu_safety_margin: 0.8,
            min_free_vram_mb: 512,
            enable_hybrid_context: true,
            max_system_ram_usage_gb: 8,
            context_expansion_strategy: "balanced".to_string(),
            ram_safety_margin_gb: 2,
            multiplayer_mode: "solo".to_string(),
            multiplayer_host_address: String::new(),
            multiplayer_participant_id: String::new(),
            mention_followup_depth: 1,
            remote_generation_timeout_secs: 120,
            multiplayer_password: None,
            compact_threshold_tokens: None,
            compact_min_messages: 8,
            compaction_model_path: None,
            heuristic_person_detection: true,
            compaction_attitude_weight: 0.5,
        }
    }

    #[test]
    fn migrate_config_table_adds_all_new_columns_and_is_idempotent() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_legacy_config_table(&con);

        Database::migrate_config_table(&con).unwrap();
        // Running it again on an already-migrated table must still be Ok.
        Database::migrate_config_table(&con).unwrap();

        let mut stmt = con.prepare("PRAGMA table_info(config)").unwrap();
        let columns: HashSet<String> = stmt
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<Result<_>>()
            .unwrap();
        drop(stmt);

        for column in [
            "context_window_size",
            "max_response_tokens",
            "enable_dynamic_context",
            "vram_limit_gb",
            "dynamic_gpu_allocation",
            "gpu_safety_margin",
            "min_free_vram_mb",
            "enable_hybrid_context",
            "max_system_ram_usage_gb",
            "context_expansion_strategy",
            "ram_safety_margin_gb",
            "multiplayer_mode",
            "multiplayer_password",
            "multiplayer_host_address",
            "multiplayer_participant_id",
            "mention_followup_depth",
            "remote_generation_timeout_secs",
            "compact_threshold_tokens",
            "compact_min_messages",
            "compaction_model_path",
            "heuristic_person_detection",
            "compaction_attitude_weight",
        ] {
            assert!(columns.contains(column), "missing column {column}");
        }
    }

    #[test]
    fn write_config_then_read_config_round_trips_multiplayer_fields() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_config_table(&con);

        let mut modify = valid_config_modify();
        modify.multiplayer_mode = "joiner".to_string();
        modify.multiplayer_host_address = "host:3000".to_string();
        modify.multiplayer_participant_id = "bot1".to_string();
        modify.multiplayer_password = Some("secret".to_string());
        Database::write_config(&con, modify).unwrap();

        let view = Database::read_config(&con).unwrap();
        assert_eq!(view.multiplayer_mode, MultiplayerMode::Joiner);
        assert_eq!(view.multiplayer_host_address, "host:3000");
        assert_eq!(view.multiplayer_participant_id, "bot1");
        assert!(view.multiplayer_password_set);
        assert_eq!(view.multiplayer_password, "secret");
    }

    #[test]
    fn write_config_with_an_empty_password_keeps_the_stored_one() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_config_table(&con);

        let mut first = valid_config_modify();
        first.multiplayer_mode = "host".to_string();
        first.multiplayer_password = Some("secret".to_string());
        Database::write_config(&con, first).unwrap();

        let mut second = valid_config_modify();
        second.multiplayer_mode = "host".to_string();
        second.multiplayer_password = Some(String::new());
        Database::write_config(&con, second).unwrap();

        let view = Database::read_config(&con).unwrap();
        assert_eq!(view.multiplayer_password, "secret");
        assert!(view.multiplayer_password_set);
    }

    #[test]
    fn write_config_rejects_joiner_with_empty_host_address() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_config_table(&con);

        let mut modify = valid_config_modify();
        modify.multiplayer_mode = "joiner".to_string();
        modify.multiplayer_participant_id = "bot1".to_string();

        let err = Database::write_config(&con, modify).unwrap_err();
        assert!(matches!(err, ConfigChangeError::Invalid(_)));
    }

    #[test]
    fn write_config_rejects_host_mode_with_no_password_stored_or_supplied() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_config_table(&con);

        let mut modify = valid_config_modify();
        modify.multiplayer_mode = "host".to_string();

        let err = Database::write_config(&con, modify).unwrap_err();
        assert!(matches!(err, ConfigChangeError::Invalid(ref msg) if msg.contains("password")));
    }

    #[test]
    fn write_config_then_read_config_round_trips_compaction_fields() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_config_table(&con);

        let mut modify = valid_config_modify();
        modify.compact_threshold_tokens = Some(4096);
        modify.compact_min_messages = 12;
        modify.compaction_model_path = Some("/models/compact.gguf".to_string());
        modify.heuristic_person_detection = false;
        Database::write_config(&con, modify).unwrap();

        let view = Database::read_config(&con).unwrap();
        assert_eq!(view.compact_threshold_tokens, Some(4096));
        assert_eq!(view.compact_min_messages, 12);
        assert_eq!(
            view.compaction_model_path,
            Some("/models/compact.gguf".to_string())
        );
        assert!(!view.heuristic_person_detection);

        // None round-trips through NULL back to None, not a stored empty
        // string or a default.
        let mut reset = valid_config_modify();
        reset.compact_threshold_tokens = None;
        reset.compaction_model_path = None;
        Database::write_config(&con, reset).unwrap();
        let view = Database::read_config(&con).unwrap();
        assert_eq!(view.compact_threshold_tokens, None);
        assert_eq!(view.compaction_model_path, None);
    }

    #[test]
    fn write_config_rejects_compact_min_messages_below_two() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_config_table(&con);

        let mut modify = valid_config_modify();
        modify.compact_min_messages = 1;

        let err = Database::write_config(&con, modify).unwrap_err();
        assert!(
            matches!(err, ConfigChangeError::Invalid(ref msg) if msg.contains("compact_min_messages"))
        );
    }

    #[test]
    fn write_config_rejects_compact_threshold_tokens_below_256() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_config_table(&con);

        let mut modify = valid_config_modify();
        modify.compact_threshold_tokens = Some(100);

        let err = Database::write_config(&con, modify).unwrap_err();
        assert!(
            matches!(err, ConfigChangeError::Invalid(ref msg) if msg.contains("compact_threshold_tokens"))
        );
    }

    #[test]
    fn write_config_then_read_config_round_trips_compaction_attitude_weight() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_config_table(&con);

        let mut modify = valid_config_modify();
        modify.compaction_attitude_weight = 0.75;
        Database::write_config(&con, modify).unwrap();

        let view = Database::read_config(&con).unwrap();
        assert_eq!(view.compaction_attitude_weight, 0.75);
    }

    #[test]
    fn write_config_rejects_compaction_attitude_weight_above_one() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_config_table(&con);

        let mut modify = valid_config_modify();
        modify.compaction_attitude_weight = 1.5;

        let err = Database::write_config(&con, modify).unwrap_err();
        assert!(
            matches!(err, ConfigChangeError::Invalid(ref msg) if msg.contains("compaction_attitude_weight"))
        );
    }

    #[test]
    fn write_config_rejects_compaction_attitude_weight_below_zero() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_config_table(&con);

        let mut modify = valid_config_modify();
        modify.compaction_attitude_weight = -0.1;

        let err = Database::write_config(&con, modify).unwrap_err();
        assert!(
            matches!(err, ConfigChangeError::Invalid(ref msg) if msg.contains("compaction_attitude_weight"))
        );
    }

    #[test]
    fn get_config_defaults_compaction_attitude_weight_for_a_pre_176_row() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_legacy_config_table(&con);
        Database::migrate_config_table(&con).unwrap();
        // A migrated row's new column reads back as its `ALTER TABLE ...
        // DEFAULT` (SQLite backfills existing rows with it), so this also
        // stands in for a config row written before this column existed.
        con.execute("UPDATE config SET compaction_attitude_weight = NULL", [])
            .unwrap();

        let view = Database::read_config(&con).unwrap();
        assert_eq!(view.compaction_attitude_weight, 0.5);
    }

    /// The pre-compaction `companion` DDL (no `compacted_through`), matching
    /// what `init()` created before #171.
    fn create_legacy_companion_table(con: &Connection) {
        con.execute(
            "CREATE TABLE IF NOT EXISTS companion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                persona TEXT,
                example_dialogue TEXT,
                first_message TEXT,
                long_term_mem INTEGER,
                short_term_mem INTEGER,
                roleplay BOOLEAN,
                dialogue_tuning BOOLEAN,
                avatar_path TEXT
            )",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO companion (name, persona, example_dialogue, first_message, long_term_mem, short_term_mem, roleplay, dialogue_tuning, avatar_path) VALUES ('Assistant', '', '', '', 2, 5, 1, 1, '')",
            [],
        )
        .unwrap();
    }

    #[test]
    fn migrate_companion_compacted_through_is_idempotent_and_keeps_rows() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_legacy_companion_table(&con);

        Database::migrate_companion_compacted_through(&con).unwrap();
        // Running it again on an already-migrated table must still be Ok.
        Database::migrate_companion_compacted_through(&con).unwrap();

        let mut stmt = con.prepare("PRAGMA table_info(companion)").unwrap();
        let columns: HashSet<String> = stmt
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<Result<_>>()
            .unwrap();
        drop(stmt);
        assert!(columns.contains("compacted_through"));

        let (name, compacted_through): (String, Option<i32>) = con
            .query_row(
                "SELECT name, compacted_through FROM companion LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(name, "Assistant");
        assert_eq!(compacted_through, None);
    }

    #[test]
    fn migrate_companion_compacted_through_run_concurrently_by_two_connections_does_not_error() {
        // Mirrors `migrate_messages_speaker_id_run_concurrently_by_two_connections_does_not_error`:
        // both connections open the column-missing table before either
        // starts migrating, so without the `IMMEDIATE` transaction the
        // second to reach `ALTER TABLE` would fail with a duplicate-column
        // error instead of blocking on the first and then seeing the
        // column already there.
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("t.db");
        let setup = Database::open_at(&db_path).unwrap();
        create_legacy_companion_table(&setup);
        drop(setup);

        let barrier = Arc::new(Barrier::new(2));
        let other_barrier = Arc::clone(&barrier);
        let other_path = db_path.clone();

        let other = thread::spawn(move || {
            let con = Database::open_at(&other_path).unwrap();
            other_barrier.wait();
            Database::migrate_companion_compacted_through(&con)
        });

        let con = Database::open_at(&db_path).unwrap();
        barrier.wait();
        let result = Database::migrate_companion_compacted_through(&con);

        assert!(result.is_ok(), "expected Ok, got {result:?}");
        assert!(
            other.join().unwrap().is_ok(),
            "the concurrent migration call should also succeed, not hit a duplicate column error"
        );

        let compacted_through: Option<i32> = con
            .query_row(
                "SELECT compacted_through FROM companion LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(compacted_through, None);
    }

    // --- #177: `source` column, compaction person upsert/delete, cleanup ---

    /// The `third_party_individuals`/`third_party_memories`/`companion_attitudes`
    /// DDL these tests need, trimmed to the columns #177's functions touch.
    /// Mirrors how `compaction::store`'s own `fresh_db` builds a minimal
    /// schema by hand rather than calling `Database::init`'s hardwired path.
    fn create_third_party_tables(con: &Connection) {
        con.execute(
            "CREATE TABLE IF NOT EXISTS third_party_individuals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                relationship_to_user TEXT,
                relationship_to_companion TEXT,
                occupation TEXT,
                personality_traits TEXT,
                physical_description TEXT,
                first_mentioned TEXT NOT NULL,
                last_mentioned TEXT,
                mention_count INTEGER DEFAULT 1,
                importance_score REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'heuristic'
            )",
            [],
        )
        .unwrap();
        con.execute(
            "CREATE TABLE IF NOT EXISTS third_party_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                third_party_id INTEGER NOT NULL,
                companion_id INTEGER NOT NULL,
                memory_type TEXT,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                emotional_valence REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                context_message_id INTEGER
            )",
            [],
        )
        .unwrap();
        con.execute(
            "CREATE TABLE IF NOT EXISTS companion_attitudes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                companion_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                target_type TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                created_at TEXT NOT NULL
            )",
            [],
        )
        .unwrap();
    }

    fn insert_heuristic_third_party(con: &Connection, name: &str) -> i32 {
        con.execute(
            "INSERT INTO third_party_individuals (
                name, relationship_to_user, relationship_to_companion, first_mentioned,
                mention_count, importance_score, created_at, updated_at, source
            ) VALUES (?, '', '', ?, 1, 0.5, ?, ?, 'heuristic')",
            params![
                name,
                get_current_date(),
                get_current_date(),
                get_current_date()
            ],
        )
        .unwrap();
        con.last_insert_rowid() as i32
    }

    #[test]
    fn migrate_third_party_individuals_table_backfills_source_as_heuristic() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        // Legacy shape: no `source` column at all.
        con.execute(
            "CREATE TABLE third_party_individuals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                relationship_to_user TEXT,
                relationship_to_companion TEXT,
                occupation TEXT,
                personality_traits TEXT,
                physical_description TEXT,
                first_mentioned TEXT NOT NULL,
                last_mentioned TEXT,
                mention_count INTEGER DEFAULT 1,
                importance_score REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO third_party_individuals (name, first_mentioned, created_at, updated_at) VALUES ('Old', 'now', 'now', 'now')",
            [],
        )
        .unwrap();

        Database::migrate_third_party_individuals_table(&con).unwrap();
        // Idempotent.
        Database::migrate_third_party_individuals_table(&con).unwrap();

        let source: String = con
            .query_row(
                "SELECT source FROM third_party_individuals WHERE name = 'Old'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(source, "heuristic");
    }

    #[test]
    fn upsert_compaction_person_inserts_a_new_row_with_compaction_source() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_third_party_tables(&con);

        let id = Database::upsert_compaction_person_in(
            &con,
            &PersonUpsert {
                name: "Alice".to_string(),
                relationship_to_user: Some("sister".to_string()),
                relationship_to_companion: None,
                mentions: 2,
            },
        )
        .unwrap();

        let (name, relationship_to_user, mention_count, source): (String, String, i32, PersonSource) = con
            .query_row(
                "SELECT name, relationship_to_user, mention_count, source FROM third_party_individuals WHERE id = ?",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(name, "Alice");
        assert_eq!(relationship_to_user, "sister");
        assert_eq!(mention_count, 2);
        assert_eq!(source, PersonSource::Compaction);
    }

    #[test]
    fn upsert_compaction_person_merges_case_insensitively_and_flips_a_heuristic_row() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_third_party_tables(&con);
        let id = insert_heuristic_third_party(&con, "alice");

        let returned_id = Database::upsert_compaction_person_in(
            &con,
            &PersonUpsert {
                name: "Alice".to_string(),
                relationship_to_user: Some("sister".to_string()),
                relationship_to_companion: None,
                mentions: 3,
            },
        )
        .unwrap();

        assert_eq!(returned_id, id);
        let (mention_count, source): (i32, PersonSource) = con
            .query_row(
                "SELECT mention_count, source FROM third_party_individuals WHERE id = ?",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        // 1 (seeded) + 3 (this upsert).
        assert_eq!(mention_count, 4);
        assert_eq!(source, PersonSource::Compaction);

        // Only one row exists for the two different-case names.
        let count: i64 = con
            .query_row("SELECT COUNT(*) FROM third_party_individuals", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn delete_compaction_persons_in_removes_only_compaction_rows_and_their_attitudes() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_third_party_tables(&con);
        let heuristic_id = insert_heuristic_third_party(&con, "Heuristic");
        let compaction_id = Database::upsert_compaction_person_in(
            &con,
            &PersonUpsert {
                name: "Compacted".to_string(),
                relationship_to_user: None,
                relationship_to_companion: None,
                mentions: 1,
            },
        )
        .unwrap();
        con.execute(
            "INSERT INTO companion_attitudes (companion_id, target_id, target_type, last_updated, created_at) VALUES (1, ?, 'third_party', 'now', 'now')",
            [compaction_id],
        )
        .unwrap();

        let removed = Database::delete_compaction_persons_in(&con).unwrap();

        assert_eq!(removed, 1);
        let remaining: Vec<i32> = con
            .prepare("SELECT id FROM third_party_individuals")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(remaining, vec![heuristic_id]);
        let attitude_count: i64 = con
            .query_row(
                "SELECT COUNT(*) FROM companion_attitudes WHERE target_id = ?",
                [compaction_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(attitude_count, 0);
    }

    #[test]
    fn cleanup_invalid_third_parties_removes_pronoun_and_stopword_rows_but_keeps_compaction() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_third_party_tables(&con);

        // A different pronoun than the compaction row below uses, so the
        // two never collide on the `name` unique index / the
        // `COLLATE NOCASE` upsert lookup.
        let pronoun_id = insert_heuristic_third_party(&con, "You");
        let stopword_id = insert_heuristic_third_party(&con, "Table");
        let short_id = insert_heuristic_third_party(&con, "Ab");
        con.execute(
            "UPDATE third_party_individuals SET relationship_to_companion = 'newly_mentioned' WHERE id IN (?, ?, ?)",
            params![pronoun_id, stopword_id, short_id],
        )
        .unwrap();

        let real_id = insert_heuristic_third_party(&con, "Alice");
        con.execute(
            "UPDATE third_party_individuals SET relationship_to_companion = 'newly_mentioned', occupation = 'teacher' WHERE id = ?",
            [real_id],
        )
        .unwrap();

        // A compaction-sourced row that happens to be named after a
        // pronoun: cleanup must keep it, since it is canon-validated, not a
        // heuristic guess.
        let compaction_id = Database::upsert_compaction_person_in(
            &con,
            &PersonUpsert {
                name: "Her".to_string(),
                relationship_to_user: None,
                relationship_to_companion: None,
                mentions: 1,
            },
        )
        .unwrap();

        let cleaned = Database::cleanup_invalid_third_parties_in(&con).unwrap();

        assert_eq!(cleaned, 3);
        let mut remaining: Vec<i32> = con
            .prepare("SELECT id FROM third_party_individuals")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        remaining.sort();
        let mut expected = vec![real_id, compaction_id];
        expected.sort();
        assert_eq!(remaining, expected);
    }

    /// Regression pin: `analyze_context_for_person` sets
    /// `relationship_to_companion = 'newly_mentioned'` on *every*
    /// heuristically detected person, real or not, and the extraction
    /// helpers behind occupation/personality/physical-description commonly
    /// find nothing even for a genuine person mentioned only in passing.
    /// Cleanup must not treat "nothing extra was extracted" as "this row is
    /// junk" — a well-formed capitalised name with a bare `newly_mentioned`
    /// value and no other detail is still a real, legitimately detected
    /// person and must survive.
    #[test]
    fn cleanup_invalid_third_parties_keeps_a_real_name_with_nothing_but_the_auto_fill() {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        create_third_party_tables(&con);

        let bare_real_id = insert_heuristic_third_party(&con, "Marcus");
        con.execute(
            "UPDATE third_party_individuals SET relationship_to_companion = 'newly_mentioned' WHERE id = ?",
            [bare_real_id],
        )
        .unwrap();

        let cleaned = Database::cleanup_invalid_third_parties_in(&con).unwrap();

        assert_eq!(cleaned, 0);
        let remaining: i64 = con
            .query_row(
                "SELECT COUNT(*) FROM third_party_individuals WHERE id = ?",
                [bare_real_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remaining, 1);
    }
}
