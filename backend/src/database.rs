use rusqlite::{Connection, Error, Result, ToSql, params};
use rusqlite::types::{FromSql, FromSqlError, ValueRef, ToSqlOutput};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Local};

use crate::character_card::CharacterCard;


#[derive(Serialize, Deserialize, Clone)]
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
    let time_related_keywords = ["time", "date", "hour", "day", "month", "year", "minute", "second", "morning", "afternoon", "evening", "night"];
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

impl FromSql for Device {
    fn column_result(value: ValueRef<'_>) -> Result<Self, FromSqlError> {
        match value {
            ValueRef::Text(i) => {
                match std::str::from_utf8(i) {
                    Ok(s) => {
                        match s {
                            "CPU" => Ok(Device::CPU),
                            "GPU" => Ok(Device::GPU),
                            "Metal" => Ok(Device::Metal),
                            _ => Err(FromSqlError::OutOfRange(0)),
                        }
                    }
                    Err(e) => Err(FromSqlError::Other(Box::new(e))),
                }
            }
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
    Mistral
}

impl FromSql for PromptTemplate {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> Result<Self, FromSqlError> {
        match value {
            ValueRef::Text(i) => {
                match std::str::from_utf8(i) {
                    Ok(s) => {
                        match s {
                            "Default" => Ok(PromptTemplate::Default),
                            "Llama2" => Ok(PromptTemplate::Llama2),
                            "Mistral" => Ok(PromptTemplate::Mistral),
                            _ => Err(FromSqlError::OutOfRange(0)),
                        }
                    }
                    Err(e) => Err(FromSqlError::Other(Box::new(e))),
                }
            }
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

fn calculate_attitude_delta(previous: &CompanionAttitude, new: &CompanionAttitude) -> AttitudeDelta {
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
        ("attraction", delta.attraction, 1.2),  // High weight for relationship-defining emotions
        ("trust", delta.trust, 1.5),
        ("fear", delta.fear, 1.1),
        ("anger", delta.anger, 1.3),
        ("joy", delta.joy, 1.0),
        ("sorrow", delta.sorrow, 1.0),
        ("disgust", delta.disgust, 1.1),
        ("surprise", delta.surprise, 0.8),    // Lower weight for transient emotions
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

fn generate_memory_description(memory_type: &str, delta: &AttitudeDelta, impact_score: f32) -> String {
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

pub struct Database {}

impl Database {
    pub fn new() -> Result<usize> {
        let con = Connection::open("companion_database.db")?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ai BOOLEAN,
                content TEXT,
                created_at TEXT
            )", []
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
            )", []
        )?;
        con.execute(
            "CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                persona TEXT,
                avatar_path TEXT
            )", []
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
                vram_limit_gb INTEGER DEFAULT 4
            )", []
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
                relationship_score REAL GENERATED ALWAYS AS ((attraction + trust + joy + respect + gratitude + empathy - fear - anger - sorrow - disgust - suspicion - jealousy) / 12.0) STORED,
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
            "CREATE INDEX IF NOT EXISTS idx_third_party_name ON third_party_individuals(name)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_memories_party ON third_party_memories(third_party_id)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_memories_companion ON third_party_memories(companion_id)", []
        )?;
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_third_party_interactions_party ON third_party_interactions(third_party_id)", []
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
                    "/assets/user_avatar-4rust.jpg"
                ]
            )?;
        }
        if Database::is_table_empty("messages", &con)? {
            struct CompanionReturn {
                name: String,
                first_message: String
            }
            let companion_data = con.query_row("SELECT name, first_message FROM companion", [], |row| {
                Ok(CompanionReturn {
                    name: row.get(0)?,
                    first_message: row.get(1)?
                   }
                )
            })?;
            let user_name: String = con.query_row("SELECT name, persona FROM user LIMIT 1", [], |row| {
                Ok(row.get(0)?)
            })?;
            con.execute(
                "INSERT INTO messages (ai, content, created_at) VALUES (?, ?, ?)",
                &[
                    "1",
                    &companion_data.first_message.replace("{{char}}", &companion_data.name).replace("{{user}}", &user_name),
                    &get_current_date()
                ]
            )?;
        }
        if Database::is_table_empty("config", &con)? {
            con.execute(
                "INSERT INTO config (device, llm_model_path, gpu_layers, prompt_template, context_window_size, max_response_tokens, enable_dynamic_context, vram_limit_gb) VALUES (?, ?, 20, ?, 2048, 512, true, 4)",
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
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare("SELECT id, ai, content, created_at FROM messages ORDER BY id DESC LIMIT ? OFFSET ?")?;
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
        Ok(messages.into_iter().rev().collect())
    }
    
    pub fn get_total_message_count() -> Result<usize> {
        let con = Connection::open("companion_database.db")?;
        let count: i64 = con.query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    pub fn get_latest_message() -> Result<Message> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare("SELECT id, ai, content, created_at FROM messages ORDER BY id DESC LIMIT 1")?;
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
        let mut stmt = con.prepare("SELECT name, persona, first_message, example_dialogue FROM companion LIMIT 1")?;
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
        let mut stmt = con.prepare("SELECT id, ai, content, created_at FROM messages WHERE id = ?")?;
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
            &format!("INSERT INTO messages (ai, content, created_at) VALUES ({}, ?, ?)", message.ai),
            &[
                &message.content,
                &get_current_date()
            ]
        )?;
        Ok(())
    }

    pub fn edit_message(id: i32, message: NewMessage) -> Result<(), Error> {
        let con = Connection::open("companion_database.db")?;
        con.execute(
            &format!("UPDATE messages SET ai = {}, content = ? WHERE id = ?", message.ai),
            &[
                &message.content,
                &id.to_string()
            ]
        )?;
        Ok(())
    }

    pub fn delete_message(id: i32) -> Result<(), Error> {
        let con = Connection::open("companion_database.db")?;
        con.execute(
            "DELETE FROM messages WHERE id = ?",
            [id],
        )?;
        Ok(())
    }

    pub fn delete_latest_message() -> Result<(), rusqlite::Error> {
        let con = Connection::open("companion_database.db")?;
        let last_message_id: i32 = con.query_row(
            "SELECT id FROM messages ORDER BY id DESC LIMIT 1",
            [],
            |row| row.get(0)
        )?;
        con.execute(
            "DELETE FROM messages WHERE id = ?",
            [last_message_id]
        )?;
        Ok(())
    }

    pub fn erase_messages() -> Result<(), Error> {
        let con = Connection::open("companion_database.db")?;
        con.execute(
            "DELETE FROM messages",
            []
        )?;
        struct CompanionReturn {
            name: String,
            first_message: String
        }
        let companion_data = con.query_row("SELECT name, first_message FROM companion", [], |row| {
            Ok(CompanionReturn {
                name: row.get(0)?,
                first_message: row.get(1)?
               }
            )
        })?;
        let user_name: String = con.query_row("SELECT name, persona FROM user LIMIT 1", [], |row| {
            Ok(row.get(0)?)
        })?;
        con.execute(
            "INSERT INTO messages (ai, content, created_at) VALUES (?, ?, ?)",
            &[
                "1",
                &companion_data.first_message.replace("{{char}}", &companion_data.name).replace("{{user}}", &user_name),
                &get_current_date()
            ]
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
                &companion.first_mes
            ]
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
        con.execute(
            "UPDATE companion SET avatar_path = ?",
            &[
                avatar_path,
            ]
        )?;
        Ok(())
    }

    pub fn edit_user(user: UserView) -> Result<(), Error> {
        let con = Connection::open("companion_database.db")?;
        con.execute(
            "UPDATE user SET name = ?, persona = ?",
            &[
                &user.name,
                &user.persona,
            ]
        )?;
        Ok(())
    }

    pub fn get_config() -> Result<ConfigView> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare("SELECT device, llm_model_path, gpu_layers, prompt_template, context_window_size, max_response_tokens, enable_dynamic_context, vram_limit_gb FROM config LIMIT 1")?;
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
            })
        })?;
        Ok(row)
    }

    pub fn change_config(config: ConfigModify) -> Result<(), Error> {
        let device = match config.device.as_str() {
            "CPU" => Device::CPU,
            "GPU" => Device::GPU,
            "Metal" => Device::Metal,
            _ => return Err(rusqlite::Error::InvalidParameterName("Invalid device type".to_string())),
        };
    
        let prompt_template = match config.prompt_template.as_str() {
            "Default" => PromptTemplate::Default,
            "Llama2" => PromptTemplate::Llama2,
            "Mistral" => PromptTemplate::Mistral,
            _ => return Err(rusqlite::Error::InvalidParameterName("Invalid prompt template type".to_string())),
        };
    
        let con = Connection::open("companion_database.db")?;
        con.execute(
            "UPDATE config SET device = ?, llm_model_path = ?, gpu_layers = ?, prompt_template = ?, context_window_size = ?, max_response_tokens = ?, enable_dynamic_context = ?, vram_limit_gb = ?",
            &[
                &device as &dyn ToSql,
                &config.llm_model_path,
                &config.gpu_layers,
                &prompt_template as &dyn ToSql,
                &config.context_window_size,
                &config.max_response_tokens,
                &config.enable_dynamic_context,
                &config.vram_limit_gb,
            ]
        )?;
        Ok(())
    }

    pub fn create_or_update_attitude(companion_id: i32, target_id: i32, target_type: &str, attitude: &CompanionAttitude) -> Result<i32> {
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
                    gratitude = ?, jealousy = ?, empathy = ?, last_updated = ?
                WHERE id = ?",
                params![
                    attitude.attraction, attitude.trust, attitude.fear, attitude.anger,
                    attitude.joy, attitude.sorrow, attitude.disgust, attitude.surprise,
                    attitude.curiosity, attitude.respect, attitude.suspicion,
                    attitude.gratitude, attitude.jealousy, attitude.empathy,
                    current_time, id
                ]
            )?;
            Ok(id)
        } else {
            con.execute(
                "INSERT INTO companion_attitudes (
                    companion_id, target_id, target_type, attraction, trust, fear, anger,
                    joy, sorrow, disgust, surprise, curiosity, respect, suspicion,
                    gratitude, jealousy, empathy, last_updated, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    companion_id, target_id, target_type,
                    attitude.attraction, attitude.trust, attitude.fear, attitude.anger,
                    attitude.joy, attitude.sorrow, attitude.disgust, attitude.surprise,
                    attitude.curiosity, attitude.respect, attitude.suspicion,
                    attitude.gratitude, attitude.jealousy, attitude.empathy,
                    current_time, current_time
                ]
            )?;
            Ok(con.last_insert_rowid() as i32)
        }
    }

    pub fn get_attitude(companion_id: i32, target_id: i32, target_type: &str) -> Result<Option<CompanionAttitude>> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, companion_id, target_id, target_type, attraction, trust, fear, anger,
                    joy, sorrow, disgust, surprise, curiosity, respect, suspicion,
                    gratitude, jealousy, empathy, relationship_score, last_updated, created_at
             FROM companion_attitudes
             WHERE companion_id = ? AND target_id = ? AND target_type = ?"
        )?;
        
        let attitude = stmt.query_row(params![companion_id, target_id, target_type], |row| {
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
                relationship_score: row.get(18)?,
                last_updated: row.get(19)?,
                created_at: row.get(20)?,
            })
        }).ok();
        
        Ok(attitude)
    }

    pub fn update_attitude_dimension(companion_id: i32, target_id: i32, target_type: &str, dimension: &str, delta: f32) -> Result<()> {
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
            params![delta, current_time, companion_id, target_id, target_type]
        )?;
        
        // Get the attitude after the change and check for significant changes
        if let Some(previous) = previous_attitude {
            if let Some(new_attitude) = Database::get_attitude(companion_id, target_id, target_type)? {
                // Trigger change detection - pass None for message context since we don't have it here
                Database::detect_attitude_change(companion_id, target_id, target_type, &previous, &new_attitude, None)?;
            }
        }
        
        Ok(())
    }

    pub fn get_all_companion_attitudes(companion_id: i32) -> Result<Vec<CompanionAttitude>> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, companion_id, target_id, target_type, attraction, trust, fear, anger,
                    joy, sorrow, disgust, surprise, curiosity, respect, suspicion,
                    gratitude, jealousy, empathy, relationship_score, last_updated, created_at
             FROM companion_attitudes
             WHERE companion_id = ?
             ORDER BY relationship_score DESC"
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
                relationship_score: row.get(18)?,
                last_updated: row.get(19)?,
                created_at: row.get(20)?,
            })
        })?;
        
        let mut result = Vec::new();
        for attitude in attitudes {
            result.push(attitude?);
        }
        
        Ok(result)
    }

    pub fn update_attitude_metadata(attitude_id: i32, interaction_type: &str, event: Option<&str>) -> Result<()> {
        let con = Connection::open("companion_database.db")?;
        
        let field = match interaction_type {
            "positive" => "positive_interactions",
            "negative" => "negative_interactions",
            "neutral" => "neutral_interactions",
            _ => return Err(Error::InvalidParameterName("Invalid interaction type".to_string())),
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

    pub fn create_or_update_third_party(name: &str, initial_data: Option<ThirdPartyIndividual>) -> Result<i32> {
        let con = Connection::open("companion_database.db")?;
        let current_time = get_current_date();
        
        let existing_id: Option<i32> = con.query_row(
            "SELECT id FROM third_party_individuals WHERE name = ?",
            &[name],
            |row| row.get(0)
        ).ok();
        
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
                        data.relationship_to_user, data.relationship_to_companion,
                        data.occupation, data.personality_traits, data.physical_description,
                        Some(current_time.clone()), Some(current_time), id
                    ]
                )?;
            } else {
                con.execute(
                    "UPDATE third_party_individuals SET 
                        last_mentioned = ?, mention_count = mention_count + 1, updated_at = ?
                    WHERE id = ?",
                    params![&current_time, &current_time, &id]
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
                    data.name, data.relationship_to_user.as_ref().unwrap_or(&"".to_string()), data.relationship_to_companion.as_ref().unwrap_or(&"".to_string()),
                    data.occupation, data.personality_traits, data.physical_description,
                    data.first_mentioned, data.mention_count, data.importance_score,
                    data.created_at, data.updated_at
                ]
            )?;
            Ok(con.last_insert_rowid() as i32)
        }
    }

    pub fn add_third_party_memory(third_party_id: i32, companion_id: i32, memory: &ThirdPartyMemory) -> Result<i32> {
        let con = Connection::open("companion_database.db")?;
        let current_time = get_current_date();
        
        con.execute(
            "INSERT INTO third_party_memories (
                third_party_id, companion_id, memory_type, content,
                importance, emotional_valence, created_at, context_message_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                third_party_id, companion_id, memory.memory_type, memory.content,
                memory.importance, memory.emotional_valence, current_time,
                memory.context_message_id
            ]
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
                interaction.third_party_id, interaction.companion_id,
                interaction.interaction_type, interaction.description,
                interaction.planned_date, interaction.impact_on_relationship,
                current_time, current_time
            ]
        )?;
        
        Ok(con.last_insert_rowid() as i32)
    }
    
    pub fn get_planned_interactions(companion_id: i32, limit: Option<usize>) -> Result<Vec<ThirdPartyInteraction>> {
        let con = Connection::open("companion_database.db")?;
        let query = if let Some(limit) = limit {
            format!(
                "SELECT id, third_party_id, companion_id, interaction_type, description,
                        planned_date, actual_date, outcome, impact_on_relationship,
                        created_at, updated_at
                 FROM third_party_interactions
                 WHERE companion_id = ? AND interaction_type = 'planned'
                 ORDER BY planned_date ASC
                 LIMIT {}", limit
            )
        } else {
            "SELECT id, third_party_id, companion_id, interaction_type, description,
                    planned_date, actual_date, outcome, impact_on_relationship,
                    created_at, updated_at
             FROM third_party_interactions
             WHERE companion_id = ? AND interaction_type = 'planned'
             ORDER BY planned_date ASC".to_string()
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
            params![current_time, outcome, impact, current_time, interaction_id]
        )?;
        
        Ok(())
    }
    
    pub fn get_interaction_history(companion_id: i32, third_party_id: i32) -> Result<Vec<ThirdPartyInteraction>> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, third_party_id, companion_id, interaction_type, description,
                    planned_date, actual_date, outcome, impact_on_relationship,
                    created_at, updated_at
             FROM third_party_interactions
             WHERE companion_id = ? AND third_party_id = ?
             ORDER BY COALESCE(actual_date, planned_date) DESC"
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
             FROM third_party_individuals WHERE name = ?"
        )?;
        
        let individual = stmt.query_row(&[name], |row| {
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
        }).ok();
        
        Ok(individual)
    }

    pub fn get_all_third_party_individuals() -> Result<Vec<ThirdPartyIndividual>> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, name, relationship_to_user, relationship_to_companion, occupation,
                    personality_traits, physical_description, first_mentioned, last_mentioned,
                    mention_count, importance_score, created_at, updated_at
             FROM third_party_individuals 
             ORDER BY importance_score DESC, mention_count DESC"
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

    pub fn get_third_party_memories(third_party_id: i32, limit: Option<usize>) -> Result<Vec<ThirdPartyMemory>> {
        let con = Connection::open("companion_database.db")?;
        let query = if let Some(limit) = limit {
            format!(
                "SELECT id, third_party_id, companion_id, memory_type, content,
                        importance, emotional_valence, created_at, context_message_id
                 FROM third_party_memories
                 WHERE third_party_id = ?
                 ORDER BY importance DESC, created_at DESC
                 LIMIT {}", limit
            )
        } else {
            "SELECT id, third_party_id, companion_id, memory_type, content,
                    importance, emotional_valence, created_at, context_message_id
             FROM third_party_memories
             WHERE third_party_id = ?
             ORDER BY importance DESC, created_at DESC".to_string()
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
            params![&new_importance, &current_time, &third_party_id]
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

    pub fn detect_attitude_change(companion_id: i32, target_id: i32, target_type: &str, 
                                 previous_attitude: &CompanionAttitude, new_attitude: &CompanionAttitude,
                                 message_context: Option<&str>) -> Result<()> {
        let delta = calculate_attitude_delta(previous_attitude, new_attitude);
        let impact_score = calculate_impact_score(&delta);
        
        if impact_score > 10.0 { // Threshold for significant changes
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
                    companion_id, target_id, target_type, memory_type, description,
                    priority_score, attitude_delta_json, impact_score, 
                    message_context.unwrap_or(""), current_time
                ]
            )?;
        }
        
        Ok(())
    }

    pub fn get_priority_attitude_memories(companion_id: i32, limit: usize) -> Result<Vec<AttitudeMemory>> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, companion_id, target_id, target_type, memory_type, description,
                    priority_score, attitude_delta_json, impact_score, message_context, created_at
             FROM attitude_memories 
             WHERE companion_id = ?
             ORDER BY priority_score DESC
             LIMIT ?"
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
        
        for name in detected_names {
            // Check if person already exists
            if Database::get_third_party_by_name(&name)?.is_none() {
                // Create new third-party individual with context-based initial data
                let initial_data = Database::analyze_context_for_person(&name, message);
                let person_id = Database::create_or_update_third_party(&name, Some(initial_data))?;
                
                // Initialize attitude tracking with context-based values
                let mut initial_attitude = Database::generate_initial_attitudes(&name, message, companion_id);
                initial_attitude.target_id = person_id;
                Database::create_or_update_attitude(companion_id, person_id, "third_party", &initial_attitude)?;
                
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
                if let Some(_existing) = Database::get_third_party_by_name(&name)? {
                    Database::create_or_update_third_party(&name, None)?;
                }
            }
        }
        
        Ok(new_person_ids)
    }
    
    fn extract_person_names(text: &str) -> Vec<String> {
        let mut names = Vec::new();
        let text = text.to_lowercase();
        
        // Common patterns for person references
        let patterns = [
            // Direct name mentions
            r"(my|the|a) (friend|colleague|boss|manager|teacher|doctor|neighbor|brother|sister|mother|father|parent|cousin|uncle|aunt) (\w+)",
            r"(\w+) (said|told|asked|mentioned|thinks|believes|wants|needs|likes|dislikes)",
            r"(with|from|to|about|for) (\w+) (yesterday|today|tomorrow|last|next)",
            r"(\w+) (is|was|will be|has|had|does|did|can|could|should|would)",
            // Relationship indicators
            r"(my|his|her) (\w+)",
            // Conversation indicators  
            r"(\w+) (and I|and me)",
            r"(I and|me and) (\w+)",
            // Common name patterns (capitalize first letter)
            r"\b([A-Z][a-z]{2,})\b",
        ];
        
        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for cap in re.captures_iter(&text) {
                    if let Some(name_match) = cap.get(cap.len() - 1) {
                        let name = name_match.as_str().trim();
                        if Database::is_likely_person_name(name) {
                            names.push(Database::capitalize_name(name));
                        }
                    }
                }
            }
        }
        
        // Remove duplicates and common words
        names.sort();
        names.dedup();
        names.into_iter()
            .filter(|name| !Database::is_common_word(name))
            .collect()
    }
    
    fn is_likely_person_name(name: &str) -> bool {
        // Filter out common non-name words
        let non_names = ["the", "and", "or", "but", "if", "when", "where", "what", "who", "how", "why",
                        "this", "that", "these", "those", "here", "there", "now", "then", "today",
                        "tomorrow", "yesterday", "said", "told", "asked", "mentioned", "think", "know"];
        
        !non_names.contains(&name.to_lowercase().as_str()) && 
        name.len() > 2 && 
        name.chars().all(|c| c.is_alphabetic() || c == '\'' || c == '-')
    }
    
    fn is_common_word(name: &str) -> bool {
        let common_words = ["User", "Assistant", "System", "Admin", "Anonymous", "Guest", "Bot", 
                           "AI", "Computer", "Machine", "Program", "Software", "App", "Website"];
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
    
    fn analyze_context_for_person(name: &str, message: &str) -> ThirdPartyIndividual {
        let current_time = get_current_date();
        let relationship_to_user = Database::extract_relationship_to_user(name, message);
        let occupation = Database::extract_occupation(name, message);
        let personality_traits = Database::extract_personality_traits(name, message);
        
        let importance_score = Database::calculate_person_importance(name, message);
        
        ThirdPartyIndividual {
            id: None,
            name: name.to_string(),
            relationship_to_user: relationship_to_user,
            relationship_to_companion: Some("newly_mentioned".to_string()),
            occupation: occupation,
            personality_traits: personality_traits,
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
            if text.contains(&format!("my {} {}", keyword, name_lower)) ||
               text.contains(&format!("{} is my {}", name_lower, keyword)) ||
               text.contains(&format!("my {}", keyword)) {
                return Some(relationship.to_string());
            }
        }
        
        None
    }
    
    fn extract_occupation(name: &str, message: &str) -> Option<String> {
        let text = message.to_lowercase();
        let name_lower = name.to_lowercase();
        
        let occupations = [
            "doctor", "teacher", "engineer", "lawyer", "nurse", "manager", "developer",
            "programmer", "designer", "artist", "writer", "accountant", "consultant",
            "analyst", "researcher", "scientist", "professor", "student", "chef",
            "mechanic", "electrician", "plumber", "carpenter", "architect", "pharmacist",
        ];
        
        for occupation in &occupations {
            if text.contains(&format!("{} is a {}", name_lower, occupation)) ||
               text.contains(&format!("{} works as", name_lower)) ||
               text.contains(&format!("dr. {}", name_lower)) ||
               text.contains(&format!("professor {}", name_lower)) {
                return Some(occupation.to_string());
            }
        }
        
        None
    }
    
    fn extract_personality_traits(name: &str, message: &str) -> Option<String> {
        let text = message.to_lowercase();
        let name_lower = name.to_lowercase();
        
        let traits = [
            "kind", "nice", "friendly", "helpful", "smart", "intelligent", "funny",
            "serious", "quiet", "loud", "outgoing", "shy", "confident", "nervous",
            "patient", "impatient", "generous", "selfish", "honest", "dishonest",
            "reliable", "unreliable", "creative", "logical", "emotional", "calm",
        ];
        
        let mut found_traits = Vec::new();
        for trait_word in &traits {
            if text.contains(&format!("{} is {}", name_lower, trait_word)) ||
               text.contains(&format!("{} seems {}", name_lower, trait_word)) ||
               text.contains(&format!("very {} {}", trait_word, name_lower)) {
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
        let emotional_words = ["love", "hate", "angry", "happy", "sad", "excited", "worried"];
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
    
    fn generate_initial_attitudes(name: &str, message: &str, companion_id: i32) -> CompanionAttitude {
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
            surprise: 15.0, // New person = some surprise
            curiosity: 20.0, // New person = high curiosity
            respect: 10.0,
            suspicion: 5.0, // Slight initial caution
            gratitude: 0.0,
            jealousy: 0.0,
            empathy: 10.0,
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
            }
        )?;
        
        // Get the companion's attitude toward this third party
        let attitude = Database::get_attitude(interaction.companion_id, interaction.third_party_id, "third_party")?
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
            impact
        )?;
        
        Ok(outcome)
    }
    
    fn create_realistic_outcome(
        interaction: &ThirdPartyInteraction, 
        attitude: &CompanionAttitude,
        third_party: &ThirdPartyIndividual
    ) -> String {
        let relationship_quality = attitude.relationship_score.unwrap_or(0.0);
        let interaction_desc = &interaction.description;
        let person_name = &third_party.name;
        
        // Generate outcome based on relationship quality and interaction type
        if interaction_desc.contains("meet") || interaction_desc.contains("coffee") || interaction_desc.contains("lunch") {
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
        } else if interaction_desc.contains("party") || interaction_desc.contains("event") || interaction_desc.contains("gathering") {
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
    
    fn calculate_interaction_impact(interaction: &ThirdPartyInteraction, attitude: &CompanionAttitude) -> f32 {
        let base_relationship = attitude.relationship_score.unwrap_or(0.0);
        let mut impact = 0.0;
        
        // Positive interactions have more impact when relationship is already good
        if interaction.description.contains("fun") || interaction.description.contains("enjoy") || interaction.description.contains("great") {
            impact = 5.0 + (base_relationship * 0.1);
        }
        // Helping interactions build trust and gratitude
        else if interaction.description.contains("help") || interaction.description.contains("assist") || interaction.description.contains("support") {
            impact = 8.0 + (attitude.trust * 0.05);
        }
        // Conflict reduces relationship quality
        else if interaction.description.contains("argue") || interaction.description.contains("fight") || interaction.description.contains("disagree") {
            impact = -10.0 - (attitude.anger * 0.1);
        }
        // Casual interactions have mild impact
        else if interaction.description.contains("meet") || interaction.description.contains("talk") || interaction.description.contains("chat") {
            impact = 2.0 * (1.0 + base_relationship / 100.0);
        }
        // Professional interactions are neutral to positive
        else if interaction.description.contains("work") || interaction.description.contains("project") || interaction.description.contains("business") {
            impact = 1.0 + (attitude.respect * 0.02);
        }
        else {
            // Default small positive impact
            impact = 1.0;
        }
        
        // Clamp impact to reasonable range
        impact.max(-25.0).min(25.0)
    }
    
    fn update_attitude_from_interaction(companion_id: i32, third_party_id: i32, description: &str, impact: f32) -> Result<()> {
        // Determine which dimensions to update based on interaction description
        let mut updates: Vec<(&str, f32)> = Vec::new();
        
        if impact > 0.0 {
            // Positive interaction
            if description.contains("fun") || description.contains("laugh") || description.contains("enjoy") {
                updates.push(("joy", impact * 0.8));
                updates.push(("attraction", impact * 0.3));
            }
            if description.contains("help") || description.contains("support") || description.contains("assist") {
                updates.push(("gratitude", impact * 1.2));
                updates.push(("trust", impact * 0.6));
            }
            if description.contains("deep") || description.contains("meaningful") || description.contains("understand") {
                updates.push(("empathy", impact * 0.7));
                updates.push(("respect", impact * 0.5));
            }
            // Reduce negative emotions
            updates.push(("suspicion", -impact * 0.3));
            updates.push(("fear", -impact * 0.2));
        } else {
            // Negative interaction
            if description.contains("argue") || description.contains("fight") || description.contains("conflict") {
                updates.push(("anger", -impact * 0.8));
                updates.push(("trust", impact * 0.5));
            }
            if description.contains("disappoint") || description.contains("letdown") || description.contains("fail") {
                updates.push(("sorrow", -impact * 0.6));
                updates.push(("respect", impact * 0.4));
            }
            if description.contains("lie") || description.contains("betray") || description.contains("deceive") {
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
            Database::update_attitude_dimension(companion_id, third_party_id, "third_party", dimension, delta)?;
        }
        
        Ok(())
    }
    
    pub fn get_third_party_by_id(id: i32) -> Result<Option<ThirdPartyIndividual>> {
        let con = Connection::open("companion_database.db")?;
        let mut stmt = con.prepare(
            "SELECT id, name, relationship_to_user, relationship_to_companion, occupation,
                    personality_traits, physical_description, first_mentioned, last_mentioned,
                    mention_count, importance_score, created_at, updated_at
             FROM third_party_individuals WHERE id = ?"
        )?;
        
        let individual = stmt.query_row(&[&id], |row| {
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
        }).ok();
        
        Ok(individual)
    }
    
    pub fn detect_interaction_request(message: &str, companion_id: i32) -> Result<Option<ThirdPartyInteraction>> {
        let message_lower = message.to_lowercase();
        
        // Check if user is asking about past interactions
        if message_lower.contains("did you") || message_lower.contains("have you") || 
           message_lower.contains("what happened") || message_lower.contains("how did") ||
           message_lower.contains("tell me about") {
            
            // Extract person name from the message
            if let Some(person_name) = Database::extract_person_from_query(message) {
                if let Some(third_party) = Database::get_third_party_by_name(&person_name)? {
                    // Check for recent interactions
                    let history = Database::get_interaction_history(companion_id, third_party.id.unwrap())?;
                    if !history.is_empty() {
                        return Ok(Some(history[0].clone()));
                    }
                    
                    // Check for planned interactions that might have occurred
                    let planned = Database::get_planned_interactions(companion_id, Some(5))?;
                    for interaction in planned {
                        if interaction.third_party_id == third_party.id.unwrap() {
                            // Generate outcome for this interaction
                            let _outcome = Database::generate_interaction_outcome(interaction.id.unwrap())?;
                            return Database::get_interaction_by_id(interaction.id.unwrap());
                        }
                    }
                }
            }
        }
        
        // Check if user is planning future interaction
        if message_lower.contains("plan to") || message_lower.contains("going to") ||
           message_lower.contains("will meet") || message_lower.contains("scheduled") {
            
            if let Some(person_name) = Database::extract_person_from_query(message) {
                if let Some(third_party) = Database::get_third_party_by_name(&person_name)? {
                    let interaction = ThirdPartyInteraction {
                        id: None,
                        third_party_id: third_party.id.unwrap(),
                        companion_id,
                        interaction_type: "planned".to_string(),
                        description: Database::extract_interaction_description(message, &person_name),
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
             FROM third_party_interactions WHERE id = ?"
        )?;
        
        let interaction = stmt.query_row(&[&id], |row| {
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
        }).ok();
        
        Ok(interaction)
    }
    
    pub fn migrate_config_table(con: &Connection) -> Result<()> {
        // Check if new columns exist and add them if they don't
        let mut has_context_window = false;
        let mut has_max_response = false;
        let mut has_dynamic_context = false;
        let mut has_vram_limit = false;
        
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
                _ => {}
            }
        }
        
        // Add missing columns with default values
        if !has_context_window {
            con.execute("ALTER TABLE config ADD COLUMN context_window_size INTEGER DEFAULT 2048", [])?;
        }
        if !has_max_response {
            con.execute("ALTER TABLE config ADD COLUMN max_response_tokens INTEGER DEFAULT 512", [])?;
        }
        if !has_dynamic_context {
            con.execute("ALTER TABLE config ADD COLUMN enable_dynamic_context BOOLEAN DEFAULT true", [])?;
        }
        if !has_vram_limit {
            con.execute("ALTER TABLE config ADD COLUMN vram_limit_gb INTEGER DEFAULT 4", [])?;
        }
        
        Ok(())
    }
}
