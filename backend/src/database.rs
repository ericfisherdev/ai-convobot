use chrono::{DateTime, Local};
use rusqlite::types::{FromSql, FromSqlError, ToSqlOutput, ValueRef};
use rusqlite::{params, Connection, Error, Result, ToSql};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::character_card::CharacterCard;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub id: i32,
    pub ai: bool,
    pub content: String,
    pub created_at: String,
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

#[derive(Serialize, Deserialize)]
pub struct NewMessage {
    pub ai: bool,
    pub content: String,
}

#[derive(Serialize, Deserialize)]
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
pub struct AttitudeMetadata {
    pub id: Option<i32>,
    pub attitude_id: i32,
    pub interaction_count: i32,
    pub positive_interactions: i32,
    pub negative_interactions: i32,
    pub neutral_interactions: i32,
    pub last_significant_event: Option<String>,
    pub relationship_status: String,
    pub notes: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AttitudeUpdate {
    pub attraction: Option<f32>,
    pub trust: Option<f32>,
    pub fear: Option<f32>,
    pub anger: Option<f32>,
    pub joy: Option<f32>,
    pub sorrow: Option<f32>,
    pub disgust: Option<f32>,
    pub surprise: Option<f32>,
    pub curiosity: Option<f32>,
    pub respect: Option<f32>,
    pub suspicion: Option<f32>,
    pub gratitude: Option<f32>,
    pub jealousy: Option<f32>,
    pub empathy: Option<f32>,
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThirdPartyRelationship {
    pub id: Option<i32>,
    pub from_party_id: i32,
    pub to_party_id: i32,
    pub relationship_type: String,
    pub strength: f32,
    pub description: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(PartialEq, Serialize, Deserialize, Clone)]
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
    Default,
    Llama2,
    Mistral,
}

impl FromSql for PromptTemplate {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> Result<Self, FromSqlError> {
        match value {
            ValueRef::Text(i) => match std::str::from_utf8(i) {
                Ok(s) => match s {
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
            PromptTemplate::Default => Ok(ToSqlOutput::from("Default")),
            PromptTemplate::Llama2 => Ok(ToSqlOutput::from("Llama2")),
            PromptTemplate::Mistral => Ok(ToSqlOutput::from("Mistral")),
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AttitudeDelta {
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

// Database query cache for performance optimization
lazy_static::lazy_static! {
    static ref DB_CACHE: Arc<Mutex<HashMap<String, (String, Instant)>>> = Arc::new(Mutex::new(HashMap::new()));
    static ref MESSAGE_CACHE: Arc<Mutex<HashMap<String, (Vec<Message>, Instant)>>> = Arc::new(Mutex::new(HashMap::new()));
}

pub struct Database {}

impl Database {
    pub fn clear_message_cache() {
        if let Ok(mut cache) = MESSAGE_CACHE.lock() {
            cache.clear();
        }
    }

    pub fn clear_db_cache() {
        if let Ok(mut cache) = DB_CACHE.lock() {
            cache.clear();
        }
    }
}

impl Database {
    pub fn new() -> Result<usize> {
        let con = Connection::open("companion_database.db")?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ai BOOLEAN,
                content TEXT,
                created_at TEXT
            )",
            [],
        )?;
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
        )?;
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
                ram_safety_margin_gb INTEGER DEFAULT 2
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
                updated_at TEXT NOT NULL
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
        if Database::is_table_empty("companion", &con)? {
            con.execute(
                "INSERT INTO companion (name, persona, example_dialogue, first_message, long_term_mem, short_term_mem, roleplay, dialogue_tuning, avatar_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                &[
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
                &[
                    "User",
                    "{{user}} is chatting with {{char}} using ai-companion web user interface",
                    "/assets/user_avatar-4rust.jpg",
                ],
            )?;
        }
        if Database::is_table_empty("messages", &con)? {
            struct CompanionReturn {
                name: String,
                first_message: String,
            }
            let companion_data =
                con.query_row("SELECT name, first_message FROM companion", [], |row| {
                    Ok(CompanionReturn {
                        name: row.get(0)?,
                        first_message: row.get(1)?,
                    })
                })?;
            let user_name: String =
                con.query_row("SELECT name, persona FROM user LIMIT 1", [], |row| {
                    Ok(row.get(0)?)
                })?;
            con.execute(
                "INSERT INTO messages (ai, content, created_at) VALUES (?, ?, ?)",
                &[
                    "1",
                    &companion_data
                        .first_message
                        .replace("{{char}}", &companion_data.name)
                        .replace("{{user}}", &user_name),
                    &get_current_date(),
                ],
            )?;
        }
        if Database::is_table_empty("config", &con)? {
            con.execute(
                "INSERT INTO config (device, llm_model_path, gpu_layers, prompt_template, context_window_size, max_response_tokens, enable_dynamic_context, vram_limit_gb, dynamic_gpu_allocation, gpu_safety_margin, min_free_vram_mb, enable_hybrid_context, max_system_ram_usage_gb, context_expansion_strategy, ram_safety_margin_gb) VALUES (?, ?, 20, ?, 2048, 512, true, 4, true, 0.8, 512, true, 8, 'balanced', 2)",
                &[
                    &Device::CPU as &dyn ToSql,
                    &"path/to/your/gguf/model.gguf",
                    &PromptTemplate::Default as &dyn ToSql
                ]
            )?;
        }

        // Initialize attitude memories table
        Database::create_attitude_memories_table()?;

        // Migrate config table to add new context window fields if they don't exist
        Database::migrate_config_table(&con)?;

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
            )", []
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

    /* pub fn get_messages() -> Result<Vec<Message>> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare("SELECT id, ai, content, created_at FROM messages")?;
        let rows = stmt.query_map([], |row| {
            Ok(Message {
                id: row.get(0)?,
                ai: row.get(1)?,
                content: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        let mut messages = Vec::new();
        for row in rows {
            messages.push(row?);
        }
        Ok(messages)
    } */

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

        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, ai, content, created_at FROM messages ORDER BY id DESC LIMIT ? OFFSET ?",
        )?;
        let rows = stmt.query_map([x, index], |row| {
            Ok(Message {
                id: row.get(0)?,
                ai: row.get(1)?,
                content: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
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

    pub fn get_total_message_count() -> Result<usize> {
        let con = Connection::open("companion_database.db")?;
        let count: i64 = con.query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    pub fn get_latest_message() -> Result<Message> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con
            .prepare("SELECT id, ai, content, created_at FROM messages ORDER BY id DESC LIMIT 1")?;
        let row = stmt.query_row([], |row| {
            Ok(Message {
                id: row.get(0)?,
                ai: row.get(1)?,
                content: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        Ok(row)
    }

    pub fn get_companion_data() -> Result<CompanionView> {
        let con = Connection::open("companion_database.db")?;
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

    pub fn get_companion_card_data() -> Result<CharacterCard> {
        let con = Connection::open("companion_database.db")?;
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
        let con = Connection::open("companion_database.db")?;
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
        let con = Connection::open("companion_database.db")?;
        let mut stmt =
            con.prepare("SELECT id, ai, content, created_at FROM messages WHERE id = ?")?;
        let row = stmt.query_row([id], |row| {
            Ok(Message {
                id: row.get(0)?,
                ai: row.get(1)?,
                content: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        Ok(row)
    }

    pub fn insert_message(message: NewMessage) -> Result<(), Error> {
        let con = Connection::open("companion_database.db")?;
        con.execute(
            &format!(
                "INSERT INTO messages (ai, content, created_at) VALUES ({}, ?, ?)",
                message.ai
            ),
            &[&message.content, &get_current_date()],
        )?;

        // Clear message cache when new message is inserted
        Database::clear_message_cache();

        Ok(())
    }

    pub fn edit_message(id: i32, message: NewMessage) -> Result<(), Error> {
        let con = Connection::open("companion_database.db")?;
        con.execute(
            &format!(
                "UPDATE messages SET ai = {}, content = ? WHERE id = ?",
                message.ai
            ),
            &[&message.content, &id.to_string()],
        )?;

        // Clear message cache when message is edited
        Database::clear_message_cache();

        Ok(())
    }

    pub fn delete_message(id: i32) -> Result<(), Error> {
        let con = Connection::open("companion_database.db")?;
        con.execute("DELETE FROM messages WHERE id = ?", [id])?;

        // Clear message cache when message is deleted
        Database::clear_message_cache();

        Ok(())
    }

    pub fn delete_latest_message() -> Result<(), rusqlite::Error> {
        let con = Connection::open("companion_database.db")?;
        let last_message_id: i32 = con.query_row(
            "SELECT id FROM messages ORDER BY id DESC LIMIT 1",
            [],
            |row| row.get(0),
        )?;
        con.execute("DELETE FROM messages WHERE id = ?", [last_message_id])?;
        Ok(())
    }

    pub fn erase_messages() -> Result<(), Error> {
        let con = Connection::open("companion_database.db")?;
        con.execute("DELETE FROM messages", [])?;

        // Clear message cache when all messages are erased
        Database::clear_message_cache();
        struct CompanionReturn {
            name: String,
            first_message: String,
        }
        let companion_data =
            con.query_row("SELECT name, first_message FROM companion", [], |row| {
                Ok(CompanionReturn {
                    name: row.get(0)?,
                    first_message: row.get(1)?,
                })
            })?;
        let user_name: String =
            con.query_row("SELECT name, persona FROM user LIMIT 1", [], |row| {
                Ok(row.get(0)?)
            })?;
        con.execute(
            "INSERT INTO messages (ai, content, created_at) VALUES (?, ?, ?)",
            &[
                "1",
                &companion_data
                    .first_message
                    .replace("{{char}}", &companion_data.name)
                    .replace("{{user}}", &user_name),
                &get_current_date(),
            ],
        )?;
        Ok(())
    }

    pub fn edit_companion(companion: CompanionView) -> Result<(), Error> {
        let con = Connection::open("companion_database.db")?;
        con.execute(
            &format!("UPDATE companion SET name = ?, persona = ?, example_dialogue = ?, first_message = ?, long_term_mem = {}, short_term_mem = {}, roleplay = {}, dialogue_tuning = {}, avatar_path = ?", companion.long_term_mem, companion.short_term_mem, companion.roleplay, companion.dialogue_tuning),
            &[
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
        let con = Connection::open("companion_database.db")?;
        con.execute(
            "UPDATE companion SET name = ?, persona = ?, example_dialogue = ?, first_message = ?",
            &[
                &companion.name,
                &companion.description,
                &companion.mes_example,
                &companion.first_mes,
            ],
        )?;
        Ok(())
    }

    pub fn import_character_card(companion: CharacterCard, image_path: &str) -> Result<(), Error> {
        let con = Connection::open("companion_database.db")?;
        con.execute(
            "UPDATE companion SET name = ?, persona = ?, example_dialogue = ?, first_message = ?, avatar_path = ?",
            &[
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
        let con = Connection::open("companion_database.db")?;
        con.execute("UPDATE companion SET avatar_path = ?", &[avatar_path])?;
        Ok(())
    }

    pub fn edit_user(user: UserView) -> Result<(), Error> {
        let con = Connection::open("companion_database.db")?;
        con.execute(
            "UPDATE user SET name = ?, persona = ?",
            &[&user.name, &user.persona],
        )?;
        Ok(())
    }

    pub fn get_config() -> Result<ConfigView> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare("SELECT device, llm_model_path, gpu_layers, prompt_template, context_window_size, max_response_tokens, enable_dynamic_context, vram_limit_gb, dynamic_gpu_allocation, gpu_safety_margin, min_free_vram_mb, enable_hybrid_context, max_system_ram_usage_gb, context_expansion_strategy, ram_safety_margin_gb FROM config LIMIT 1")?;
        let row = stmt.query_row([], |row| {
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
                context_expansion_strategy: row.get::<_, Option<String>>(13)?.unwrap_or("balanced".to_string()),
                ram_safety_margin_gb: row.get::<_, Option<usize>>(14)?.unwrap_or(2),
            })
        })?;
        Ok(row)
    }

    pub fn change_config(config: ConfigModify) -> Result<(), Error> {
        let device = match config.device.as_str() {
            "CPU" => Device::CPU,
            "GPU" => Device::GPU,
            "Metal" => Device::Metal,
            _ => {
                return Err(rusqlite::Error::InvalidParameterName(
                    "Invalid device type".to_string(),
                ))
            }
        };

        let prompt_template = match config.prompt_template.as_str() {
            "Default" => PromptTemplate::Default,
            "Llama2" => PromptTemplate::Llama2,
            "Mistral" => PromptTemplate::Mistral,
            _ => {
                return Err(rusqlite::Error::InvalidParameterName(
                    "Invalid prompt template type".to_string(),
                ))
            }
        };

        let con = Connection::open("companion_database.db")?;
        con.execute(
            "UPDATE config SET device = ?, llm_model_path = ?, gpu_layers = ?, prompt_template = ?, context_window_size = ?, max_response_tokens = ?, enable_dynamic_context = ?, vram_limit_gb = ?, dynamic_gpu_allocation = ?, gpu_safety_margin = ?, min_free_vram_mb = ?, enable_hybrid_context = ?, max_system_ram_usage_gb = ?, context_expansion_strategy = ?, ram_safety_margin_gb = ?",
            &[
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
            ]
        )?;
        Ok(())
    }

    pub fn create_or_update_attitude(
        companion_id: i32,
        target_id: i32,
        target_type: &str,
        attitude: &CompanionAttitude,
    ) -> Result<i32> {
        let con = Connection::open("companion_database.db")?;
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
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, companion_id, target_id, target_type, attraction, trust, fear, anger,
                    joy, sorrow, disgust, surprise, curiosity, respect, suspicion,
                    gratitude, jealousy, empathy, lust, love, anxiety, butterflies,
                    submissiveness, dominance, relationship_score, last_updated, created_at
             FROM companion_attitudes
             WHERE companion_id = ? AND target_id = ? AND target_type = ?",
        )?;

        let attitude = stmt
            .query_row(params![companion_id, target_id, target_type], |row| {
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
            .ok();

        Ok(attitude)
    }

    pub fn update_attitude_dimension(
        companion_id: i32,
        target_id: i32,
        target_type: &str,
        dimension: &str,
        delta: f32,
    ) -> Result<()> {
        // Get the attitude before the change for comparison
        let previous_attitude = Database::get_attitude(companion_id, target_id, target_type)?;

        let con = Connection::open("companion_database.db")?;
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

        // Get the attitude after the change and check for significant changes
        if let Some(previous) = previous_attitude {
            if let Some(new_attitude) =
                Database::get_attitude(companion_id, target_id, target_type)?
            {
                // Trigger change detection - pass None for message context since we don't have it here
                Database::detect_attitude_change(
                    companion_id,
                    target_id,
                    target_type,
                    &previous,
                    &new_attitude,
                    None,
                )?;
            }
        }

        Ok(())
    }

    pub fn get_all_companion_attitudes(companion_id: i32) -> Result<Vec<CompanionAttitude>> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, companion_id, target_id, target_type, attraction, trust, fear, anger,
                    joy, sorrow, disgust, surprise, curiosity, respect, suspicion,
                    gratitude, jealousy, empathy, lust, love, anxiety, butterflies,
                    submissiveness, dominance, relationship_score, last_updated, created_at
             FROM companion_attitudes
             WHERE companion_id = ?
             ORDER BY relationship_score DESC",
        )?;

        let attitudes = stmt.query_map(&[&companion_id], |row| {
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

    pub fn update_attitude_metadata(
        attitude_id: i32,
        interaction_type: &str,
        event: Option<&str>,
    ) -> Result<()> {
        let con = Connection::open("companion_database.db")?;

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
        let con = Connection::open("companion_database.db")?;
        con.execute(
            "DELETE FROM companion_attitudes WHERE companion_id = ?",
            params![companion_id],
        )?;
        Ok(())
    }

    pub fn create_initial_user_attitude(companion_id: i32, user_id: i32, companion_persona: &str) -> Result<i32> {
        let base_attitude = CompanionAttitude {
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
        };

        let adjusted_attitude = Database::adjust_attitude_for_persona(&base_attitude, companion_persona);
        Database::create_or_update_attitude(companion_id, user_id, "user", &adjusted_attitude)
    }

    pub fn adjust_attitude_for_persona(base_attitude: &CompanionAttitude, persona: &str) -> CompanionAttitude {
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
        let con = Connection::open("companion_database.db")?;
        let current_time = get_current_date();

        let existing_id: Option<i32> = con
            .query_row(
                "SELECT id FROM third_party_individuals WHERE name = ?",
                &[name],
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

    pub fn add_third_party_memory(
        third_party_id: i32,
        companion_id: i32,
        memory: &ThirdPartyMemory,
    ) -> Result<i32> {
        let con = Connection::open("companion_database.db")?;
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
        let con = Connection::open("companion_database.db")?;
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
        let con = Connection::open("companion_database.db")?;
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
        let interactions = stmt.query_map(&[&companion_id], |row| {
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
        let con = Connection::open("companion_database.db")?;
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
        let con = Connection::open("companion_database.db")?;
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
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, name, relationship_to_user, relationship_to_companion, occupation,
                    personality_traits, physical_description, first_mentioned, last_mentioned,
                    mention_count, importance_score, created_at, updated_at
             FROM third_party_individuals WHERE name = ?",
        )?;

        let individual = stmt
            .query_row(&[name], |row| {
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
                })
            })
            .ok();

        Ok(individual)
    }

    pub fn get_all_third_party_individuals() -> Result<Vec<ThirdPartyIndividual>> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, name, relationship_to_user, relationship_to_companion, occupation,
                    personality_traits, physical_description, first_mentioned, last_mentioned,
                    mention_count, importance_score, created_at, updated_at
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
            })
        })?;

        let mut result = Vec::new();
        for individual in individuals {
            result.push(individual?);
        }

        Ok(result)
    }

    pub fn get_third_party_memories(
        third_party_id: i32,
        limit: Option<usize>,
    ) -> Result<Vec<ThirdPartyMemory>> {
        let con = Connection::open("companion_database.db")?;
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
        let memories = stmt.query_map(&[&third_party_id], |row| {
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

    pub fn update_third_party_importance(third_party_id: i32, new_importance: f32) -> Result<()> {
        let con = Connection::open("companion_database.db")?;
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

    pub fn create_attitude_memories_table() -> Result<()> {
        let con = Connection::open("companion_database.db")?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS attitude_memories (
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
        )?;

        // Create index for priority queries
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_attitude_memories_priority 
             ON attitude_memories(companion_id, priority_score DESC)",
            [],
        )?;

        Ok(())
    }

    pub fn detect_attitude_change(
        companion_id: i32,
        target_id: i32,
        target_type: &str,
        previous_attitude: &CompanionAttitude,
        new_attitude: &CompanionAttitude,
        message_context: Option<&str>,
    ) -> Result<()> {
        let delta = calculate_attitude_delta(previous_attitude, new_attitude);
        let impact_score = calculate_impact_score(&delta);

        if impact_score > 10.0 {
            // Threshold for significant changes
            let memory_type = classify_memory_type(&delta, impact_score);
            let priority_score = calculate_priority_score(&delta, impact_score, &memory_type);

            let description = generate_memory_description(&memory_type, &delta, impact_score);
            let attitude_delta_json = serde_json::to_string(&delta).unwrap_or_default();

            let con = Connection::open("companion_database.db")?;
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
                    memory_type,
                    description,
                    priority_score,
                    attitude_delta_json,
                    impact_score,
                    message_context.unwrap_or(""),
                    current_time
                ],
            )?;
        }

        Ok(())
    }

    pub fn get_priority_attitude_memories(
        companion_id: i32,
        limit: usize,
    ) -> Result<Vec<AttitudeMemory>> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, companion_id, target_id, target_type, memory_type, description,
                    priority_score, attitude_delta_json, impact_score, message_context, created_at
             FROM attitude_memories 
             WHERE companion_id = ?
             ORDER BY priority_score DESC
             LIMIT ?",
        )?;

        let memories = stmt.query_map(params![companion_id, limit], |row| {
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
        })?;

        let mut result = Vec::new();
        for memory in memories {
            result.push(memory?);
        }

        Ok(result)
    }

    // Automatic Person Detection System

    pub fn detect_new_persons_in_message(message: &str, companion_id: i32) -> Result<Vec<i32>> {
        let detected_names = Database::extract_person_names(message);
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
        let con = Connection::open("companion_database.db")?;
        let mut cleaned_count = 0;

        // Find all duplicate names (case-insensitive)
        let mut stmt = con.prepare("
            SELECT LOWER(name) as lower_name, COUNT(*) as count 
            FROM third_party_individuals 
            GROUP BY LOWER(name) 
            HAVING COUNT(*) > 1
        ")?;

        let duplicate_names: Vec<String> = stmt.query_map([], |row| {
            Ok(row.get::<_, String>(0)?)
        })?.collect::<std::result::Result<Vec<_>, _>>()?;

        for lower_name in duplicate_names {
            // Get all instances of this name
            let mut instances_stmt = con.prepare("
                SELECT id, name, relationship_to_user, relationship_to_companion, occupation,
                       personality_traits, physical_description, first_mentioned, last_mentioned,
                       mention_count, importance_score, created_at, updated_at
                FROM third_party_individuals 
                WHERE LOWER(name) = ? 
                ORDER BY created_at ASC
            ")?;

            let instances: Vec<ThirdPartyIndividual> = instances_stmt.query_map([&lower_name], |row| {
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
                })
            })?.collect::<std::result::Result<Vec<_>, _>>()?;

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
                        if latest_last_mentioned.is_none() || last > latest_last_mentioned.as_ref().unwrap() {
                            latest_last_mentioned = Some(last.clone());
                        }
                    }
                }

                // Update the kept instance with merged data
                con.execute("
                    UPDATE third_party_individuals SET 
                        mention_count = ?,
                        importance_score = ?,
                        first_mentioned = ?,
                        last_mentioned = ?,
                        updated_at = ?
                    WHERE id = ?
                ", params![
                    total_mentions,
                    max_importance,
                    earliest_first_mentioned,
                    latest_last_mentioned,
                    get_current_date(),
                    keep_id
                ])?;

                // Update attitudes to point to the kept instance
                for instance in &instances[1..] {
                    if let Some(delete_id) = instance.id {
                        con.execute("
                            UPDATE companion_attitudes SET target_id = ? 
                            WHERE target_id = ? AND target_type = 'third_party'
                        ", params![keep_id, delete_id])?;

                        // Update memories to point to the kept instance  
                        con.execute("
                            UPDATE third_party_memories SET third_party_id = ?
                            WHERE third_party_id = ?
                        ", params![keep_id, delete_id])?;

                        // Delete the duplicate instance
                        con.execute("DELETE FROM third_party_individuals WHERE id = ?", [delete_id])?;
                        cleaned_count += 1;
                    }
                }
            }
        }

        Ok(cleaned_count)
    }

    pub fn cleanup_invalid_third_parties() -> Result<i32> {
        let con = Connection::open("companion_database.db")?;
        let mut cleaned_count = 0;
        
        // List of invalid names that should be removed
        let invalid_names = [
            // Body parts
            "hand", "hands", "shoulder", "shoulders", "head", "heads", "arm", "arms",
            "leg", "legs", "foot", "feet", "eye", "eyes", "ear", "ears", "nose", "mouth",
            "face", "hair", "neck", "back", "chest", "stomach", "knee", "knees", "elbow",
            "elbows", "finger", "fingers", "thumb", "thumbs", "toe", "toes",
            
            // Common objects
            "class", "classes", "book", "books", "table", "tables", "chair", "chairs",
            "door", "doors", "window", "windows", "desk", "desks", "computer", "computers",
            "phone", "phones", "car", "cars", "house", "houses", "room", "rooms",
            
            // Abstract concepts
            "should", "could", "would", "thing", "things", "stuff", "matter", "matters",
            "way", "ways", "time", "times", "place", "places", "work", "works",
            
            // Common verbs/actions
            "walk", "walks", "talk", "talks", "look", "looks", "feel", "feels",
            "want", "wants", "need", "needs", "use", "uses", "make", "makes",
        ];
        
        for invalid_name in &invalid_names {
            // Find and delete invalid third parties
            let mut stmt = con.prepare("
                SELECT id FROM third_party_individuals 
                WHERE LOWER(name) = LOWER(?)
            ")?;
            
            let ids: Vec<i32> = stmt.query_map([invalid_name], |row| {
                Ok(row.get::<_, i32>(0)?)
            })?.collect::<std::result::Result<Vec<_>, _>>()?;
            
            for id in ids {
                // Delete associated attitudes
                con.execute(
                    "DELETE FROM companion_attitudes WHERE target_id = ? AND target_type = 'third_party'",
                    params![id]
                )?;
                
                // Delete associated memories
                con.execute(
                    "DELETE FROM third_party_memories WHERE third_party_id = ?",
                    params![id]
                )?;
                
                // Delete the third party record
                con.execute(
                    "DELETE FROM third_party_individuals WHERE id = ?",
                    params![id]
                )?;
                
                cleaned_count += 1;
                println!("Removed invalid third party: {} (id: {})", invalid_name, id);
            }
        }
        
        // Also check for entries that don't look like proper names
        let mut stmt = con.prepare("
            SELECT id, name FROM third_party_individuals
        ")?;
        
        let entries: Vec<(i32, String)> = stmt.query_map([], |row| {
            Ok((row.get::<_, i32>(0)?, row.get::<_, String>(1)?))
        })?.collect::<std::result::Result<Vec<_>, _>>()?;
        
        for (id, name) in entries {
            // Check if this is likely NOT a person name
            if !Database::is_likely_person_name(&name) || 
               !name.chars().next().unwrap_or('a').is_uppercase() {
                // Delete associated attitudes
                con.execute(
                    "DELETE FROM companion_attitudes WHERE target_id = ? AND target_type = 'third_party'",
                    params![id]
                )?;
                
                // Delete associated memories
                con.execute(
                    "DELETE FROM third_party_memories WHERE third_party_id = ?",
                    params![id]
                )?;
                
                // Delete the third party record
                con.execute(
                    "DELETE FROM third_party_individuals WHERE id = ?",
                    params![id]
                )?;
                
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
        let text_lower = text.to_lowercase();

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
                            if potential_name.len() > 0 
                                && potential_name.chars().next().unwrap().is_uppercase()
                                && Database::is_likely_person_name(potential_name) 
                                && Database::is_proper_name_context(potential_name, text_original) {
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
                && Database::is_likely_person_name(clean_word) {
                
                // Check surrounding context for person indicators
                let has_person_context = 
                    (i > 0 && Database::is_person_indicator(&words[i-1].to_lowercase())) ||
                    (i < words.len() - 1 && Database::is_person_indicator(&words[i+1].to_lowercase()));
                
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
            .filter(|name| !Database::is_common_word(name) && name.chars().next().unwrap().is_uppercase())
            .collect()
    }

    fn is_likely_person_name(name: &str) -> bool {
        let name_lower = name.to_lowercase();
        
        // Filter out common non-name words
        let non_names = [
            // Original words
            "the", "and", "or", "but", "if", "when", "where", "what", "who", "how", "why",
            "this", "that", "these", "those", "here", "there", "now", "then",
            "today", "tomorrow", "yesterday", "said", "told", "asked", "mentioned", "think", "know",
            
            // Body parts
            "hand", "hands", "shoulder", "shoulders", "head", "heads", "arm", "arms", 
            "leg", "legs", "foot", "feet", "eye", "eyes", "ear", "ears", "nose", "mouth",
            "face", "hair", "neck", "back", "chest", "stomach", "knee", "knees", "elbow", 
            "elbows", "finger", "fingers", "thumb", "thumbs", "toe", "toes", "ankle", "ankles",
            "wrist", "wrists", "hip", "hips", "body", "skin", "bone", "bones", "muscle", "muscles",
            
            // Common objects
            "class", "classes", "book", "books", "table", "tables", "chair", "chairs",
            "door", "doors", "window", "windows", "desk", "desks", "computer", "computers",
            "phone", "phones", "car", "cars", "house", "houses", "room", "rooms",
            "wall", "walls", "floor", "floors", "ceiling", "ceilings", "roof", "roofs",
            "street", "streets", "road", "roads", "building", "buildings", "office", "offices",
            
            // Abstract concepts and common words
            "should", "could", "would", "must", "might", "may", "can", "will", "shall",
            "thing", "things", "stuff", "matter", "matters", "way", "ways", "time", "times",
            "place", "places", "work", "works", "play", "plays", "run", "runs", "walk", "walks",
            "talk", "talks", "look", "looks", "feel", "feels", "want", "wants", "need", "needs",
            "use", "uses", "make", "makes", "take", "takes", "give", "gives", "get", "gets",
            "keep", "keeps", "let", "lets", "help", "helps", "show", "shows", "try", "tries",
            
            // Nature and environment
            "tree", "trees", "plant", "plants", "flower", "flowers", "grass", "ground",
            "sky", "sun", "moon", "star", "stars", "cloud", "clouds", "rain", "snow",
            "wind", "air", "water", "fire", "earth", "stone", "stones", "rock", "rocks",
            
            // Common activities/states
            "sleep", "wake", "eat", "drink", "sit", "stand", "lie", "move", "stop", "start",
            "end", "begin", "open", "close", "break", "fix", "clean", "wash", "dry", "cut",
            
            // Pronouns and determiners
            "it", "its", "them", "their", "theirs", "some", "any", "all", "each", "every",
            "few", "many", "much", "more", "most", "less", "least", "other", "another",
            "such", "own", "same", "different", "various", "several", "both", "either", "neither",
        ];

        // Check if in non-names list
        if non_names.contains(&name_lower.as_str()) {
            return false;
        }
        
        // Filter out words with certain suffixes that are unlikely to be names
        if name_lower.ends_with("ing") || 
           name_lower.ends_with("tion") || 
           name_lower.ends_with("sion") ||
           name_lower.ends_with("ness") ||
           name_lower.ends_with("ment") || 
           name_lower.ends_with("ity") ||
           name_lower.ends_with("ance") ||
           name_lower.ends_with("ence") ||
           name_lower.ends_with("ship") ||
           name_lower.ends_with("hood") ||
           name_lower.ends_with("dom") ||
           name_lower.ends_with("ism") ||
           name_lower.ends_with("ist") ||
           name_lower.ends_with("able") ||
           name_lower.ends_with("ible") ||
           name_lower.ends_with("ful") ||
           name_lower.ends_with("less") ||
           name_lower.ends_with("ous") ||
           name_lower.ends_with("ive") ||
           name_lower.ends_with("ly") {
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
        let person_verbs = ["said", "told", "asked", "called", "visited", "met", "saw", "knows", "likes"];
        for verb in &person_verbs {
            if text_lower.contains(&format!("{} {}", name_lower, verb)) ||
               text_lower.contains(&format!("{} {}", verb, name_lower)) {
                return true;
            }
        }
        
        // If none of the above, be conservative
        true // We'll rely on other filters to catch non-names
    }
    
    fn is_person_indicator(word: &str) -> bool {
        // Words that often appear before or after person names
        let indicators = [
            "with", "and", "met", "saw", "told", "asked", "called", "visited",
            "friend", "colleague", "neighbor", "brother", "sister", "mother", "father",
            "uncle", "aunt", "cousin", "boss", "teacher", "doctor", "said", "says",
            "thinks", "believes", "wants", "needs", "likes", "loves", "hates"
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
        } else if text.contains("friend") || text.contains("colleague") {
            importance += 0.2;
        } else if text.contains("boss") || text.contains("manager") {
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
        attitude.attraction = attitude.attraction.max(-100.0).min(100.0);
        attitude.trust = attitude.trust.max(-100.0).min(100.0);
        attitude.fear = attitude.fear.max(-100.0).min(100.0);
        attitude.anger = attitude.anger.max(-100.0).min(100.0);
        attitude.joy = attitude.joy.max(-100.0).min(100.0);
        attitude.sorrow = attitude.sorrow.max(-100.0).min(100.0);
        attitude.disgust = attitude.disgust.max(-100.0).min(100.0);
        attitude.surprise = attitude.surprise.max(-100.0).min(100.0);
        attitude.curiosity = attitude.curiosity.max(-100.0).min(100.0);
        attitude.respect = attitude.respect.max(-100.0).min(100.0);
        attitude.suspicion = attitude.suspicion.max(-100.0).min(100.0);
        attitude.gratitude = attitude.gratitude.max(-100.0).min(100.0);
        attitude.jealousy = attitude.jealousy.max(-100.0).min(100.0);
        attitude.empathy = attitude.empathy.max(-100.0).min(100.0);
    }

    // Companion Interaction Tracking System

    pub fn generate_interaction_outcome(interaction_id: i32) -> Result<String> {
        let con = Connection::open("companion_database.db")?;

        // Get the interaction details
        let interaction: ThirdPartyInteraction = con.query_row(
            "SELECT id, third_party_id, companion_id, interaction_type, description,
                    planned_date, actual_date, outcome, impact_on_relationship,
                    created_at, updated_at
             FROM third_party_interactions WHERE id = ?",
            &[&interaction_id],
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
        let mut impact = 0.0;

        // Positive interactions have more impact when relationship is already good
        if interaction.description.contains("fun")
            || interaction.description.contains("enjoy")
            || interaction.description.contains("great")
        {
            impact = 5.0 + (base_relationship * 0.1);
        }
        // Helping interactions build trust and gratitude
        else if interaction.description.contains("help")
            || interaction.description.contains("assist")
            || interaction.description.contains("support")
        {
            impact = 8.0 + (attitude.trust * 0.05);
        }
        // Conflict reduces relationship quality
        else if interaction.description.contains("argue")
            || interaction.description.contains("fight")
            || interaction.description.contains("disagree")
        {
            impact = -10.0 - (attitude.anger * 0.1);
        }
        // Casual interactions have mild impact
        else if interaction.description.contains("meet")
            || interaction.description.contains("talk")
            || interaction.description.contains("chat")
        {
            impact = 2.0 * (1.0 + base_relationship / 100.0);
        }
        // Professional interactions are neutral to positive
        else if interaction.description.contains("work")
            || interaction.description.contains("project")
            || interaction.description.contains("business")
        {
            impact = 1.0 + (attitude.respect * 0.02);
        } else {
            // Default small positive impact
            impact = 1.0;
        }

        // Clamp impact to reasonable range
        impact.max(-25.0).min(25.0)
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
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, name, relationship_to_user, relationship_to_companion, occupation,
                    personality_traits, physical_description, first_mentioned, last_mentioned,
                    mention_count, importance_score, created_at, updated_at
             FROM third_party_individuals WHERE id = ?",
        )?;

        let individual = stmt
            .query_row(&[&id], |row| {
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
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, third_party_id, companion_id, interaction_type, description,
                    planned_date, actual_date, outcome, impact_on_relationship,
                    created_at, updated_at
             FROM third_party_interactions WHERE id = ?",
        )?;

        let interaction = stmt
            .query_row(&[&id], |row| {
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

    pub fn migrate_config_table(con: &Connection) -> Result<()> {
        // Check if new columns exist and add them if they don't
        let mut has_context_window = false;
        let mut has_max_response = false;
        let mut has_dynamic_context = false;
        let mut has_vram_limit = false;
        let mut has_hybrid_context = false;
        let mut has_max_system_ram = false;
        let mut has_context_strategy = false;
        let mut has_ram_safety_margin = false;

        // Check existing columns
        let mut stmt = con.prepare("PRAGMA table_info(config)")?;
        let rows = stmt.query_map([], |row| {
            let column_name: String = row.get(1)?;
            Ok(column_name)
        })?;

        for row in rows {
            let column_name = row?;
            match column_name.as_str() {
                "context_window_size" => has_context_window = true,
                "max_response_tokens" => has_max_response = true,
                "enable_dynamic_context" => has_dynamic_context = true,
                "vram_limit_gb" => has_vram_limit = true,
                "enable_hybrid_context" => has_hybrid_context = true,
                "max_system_ram_usage_gb" => has_max_system_ram = true,
                "context_expansion_strategy" => has_context_strategy = true,
                "ram_safety_margin_gb" => has_ram_safety_margin = true,
                _ => {}
            }
        }

        // Add missing columns with default values
        if !has_context_window {
            con.execute(
                "ALTER TABLE config ADD COLUMN context_window_size INTEGER DEFAULT 2048",
                [],
            )?;
        }
        if !has_max_response {
            con.execute(
                "ALTER TABLE config ADD COLUMN max_response_tokens INTEGER DEFAULT 512",
                [],
            )?;
        }
        if !has_dynamic_context {
            con.execute(
                "ALTER TABLE config ADD COLUMN enable_dynamic_context BOOLEAN DEFAULT true",
                [],
            )?;
        }
        if !has_vram_limit {
            con.execute(
                "ALTER TABLE config ADD COLUMN vram_limit_gb INTEGER DEFAULT 4",
                [],
            )?;
        }
        if !has_hybrid_context {
            con.execute(
                "ALTER TABLE config ADD COLUMN enable_hybrid_context BOOLEAN DEFAULT true",
                [],
            )?;
        }
        if !has_max_system_ram {
            con.execute(
                "ALTER TABLE config ADD COLUMN max_system_ram_usage_gb INTEGER DEFAULT 8",
                [],
            )?;
        }
        if !has_context_strategy {
            con.execute(
                "ALTER TABLE config ADD COLUMN context_expansion_strategy TEXT DEFAULT 'balanced'",
                [],
            )?;
        }
        if !has_ram_safety_margin {
            con.execute(
                "ALTER TABLE config ADD COLUMN ram_safety_margin_gb INTEGER DEFAULT 2",
                [],
            )?;
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
        if !has_lust || !has_love || !has_anxiety || !has_butterflies || !has_submissiveness || !has_dominance {
            // The relationship_score column will be recalculated automatically with the new formula
            // when the table structure is updated
        }

        Ok(())
    }

    /// Check for third-party mentions in message and track them, returning console output
    pub fn track_third_party_mentions(message: &str) -> Result<String> {
        let mut console_output = Vec::new();
        
        // Get all existing third parties to check for mentions
        let third_parties = Database::get_all_third_party_individuals()?;
        
        let message_lower = message.to_lowercase();
        
        for party in &third_parties {
            let name_lower = party.name.to_lowercase();
            
            // Check if this person is mentioned in the message
            if message_lower.contains(&name_lower) {
                // Update mention count and last_mentioned
                let con = Connection::open("companion_database.db")?;
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
        let potential_names = Database::extract_potential_names(&message_lower);
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
                && word.chars().all(|c| c.is_alphabetic()) {
                
                // Skip common non-name words
                let skip_words = [
                    "the", "and", "but", "for", "nor", "yet", "so", "at", "by", "in", "of", "on", "to", "up", "as", "is", "it", "or", "be", "do", "go", "he", "if", "me", "my", "no", "we", "I"
                ];
                
                if !skip_words.contains(&word.to_lowercase().as_str()) {
                    // Check if the next word might be a last name
                    if i + 1 < words.len() {
                        let next_word = words[i + 1];
                        if next_word.chars().next().unwrap_or('a').is_uppercase() 
                            && next_word.len() > 2 
                            && next_word.chars().all(|c| c.is_alphabetic()) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
            content: "Hello world".to_string(),
            created_at: "2024-01-15 10:00".to_string(),
        };

        assert_eq!(message.id, 1);
        assert!(message.ai);
        assert_eq!(message.content, "Hello world");
    }

    #[test]
    fn test_new_message_struct() {
        let new_message = NewMessage {
            ai: false,
            content: "User message".to_string(),
        };

        assert!(!new_message.ai);
        assert_eq!(new_message.content, "User message");
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
        let names2 = Database::extract_person_names(
            "My friend Alex called me. Dr. Smith visited today.",
        );
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
        let names5 = Database::extract_person_names(
            "The door is open. The table has a book on it.",
        );
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
        assert!(Database::is_proper_name_context("John", "John's car is red"));
        assert!(Database::is_proper_name_context("Sarah", "Sarah's house is nearby"));
        
        // Test with titles
        assert!(Database::is_proper_name_context("Smith", "Dr. Smith arrived"));
        assert!(Database::is_proper_name_context("Johnson", "Mrs. Johnson called"));
        
        // Test with person-related verbs
        assert!(Database::is_proper_name_context("Alex", "Alex said hello"));
        assert!(Database::is_proper_name_context("Maria", "I met Maria yesterday"));
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
}
