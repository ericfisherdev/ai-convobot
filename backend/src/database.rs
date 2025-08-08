use rusqlite::{Connection, Error, Result, ToSql};
use rusqlite::types::{FromSql, FromSqlError, ValueRef, ToSqlOutput};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Local};

use crate::character_card::CharacterCard;


#[derive(Serialize, Deserialize)]
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

#[derive(PartialEq, Serialize, Deserialize)]
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

#[derive(PartialEq, Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
pub struct ConfigView {
    pub device: Device,
    pub llm_model_path: String,
    pub gpu_layers: usize,
    pub prompt_template: PromptTemplate
}

#[derive(Serialize, Deserialize)]
pub struct ConfigModify {
    pub device: String,
    pub llm_model_path: String,
    pub gpu_layers: usize,
    pub prompt_template: String
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
                prompt_template TEXT
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
                "INSERT INTO config (device, llm_model_path, gpu_layers, prompt_template) VALUES (?, ?, 20, ?)",
                &[
                    &Device::CPU as &dyn ToSql,
                    &"path/to/your/gguf/model.gguf",
                    &PromptTemplate::Default as &dyn ToSql
                ]
            )?;
        } 
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
        let mut stmt = con.prepare("SELECT device, llm_model_path, gpu_layers, prompt_template FROM config LIMIT 1")?;
        let row = stmt.query_row([], |row| {
            Ok(ConfigView {
                device: row.get(0)?,
                llm_model_path: row.get(1)?,
                gpu_layers: row.get(2)?,
                prompt_template: row.get(3)?
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
            "UPDATE config SET device = ?, llm_model_path = ?, gpu_layers = ?, prompt_template = ?",
            &[
                &device as &dyn ToSql,
                &config.llm_model_path,
                &config.gpu_layers,
                &prompt_template as &dyn ToSql,
            ]
        )?;
        Ok(())
    }

    pub fn create_or_update_attitude(companion_id: i32, target_id: i32, target_type: &str, attitude: &CompanionAttitude) -> Result<i32> {
        let con = Connection::open("companion_database.db")?;
        let current_time = get_current_date();
        
        let existing_id: Option<i32> = con.query_row(
            "SELECT id FROM companion_attitudes WHERE companion_id = ? AND target_id = ? AND target_type = ?",
            &[&companion_id, &target_id, target_type],
            |row| row.get(0)
        ).ok();
        
        if let Some(id) = existing_id {
            con.execute(
                "UPDATE companion_attitudes SET 
                    attraction = ?, trust = ?, fear = ?, anger = ?, joy = ?, sorrow = ?,
                    disgust = ?, surprise = ?, curiosity = ?, respect = ?, suspicion = ?,
                    gratitude = ?, jealousy = ?, empathy = ?, last_updated = ?
                WHERE id = ?",
                &[
                    &attitude.attraction, &attitude.trust, &attitude.fear, &attitude.anger,
                    &attitude.joy, &attitude.sorrow, &attitude.disgust, &attitude.surprise,
                    &attitude.curiosity, &attitude.respect, &attitude.suspicion,
                    &attitude.gratitude, &attitude.jealousy, &attitude.empathy,
                    &current_time, &id
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
                &[
                    &companion_id, &target_id, target_type,
                    &attitude.attraction, &attitude.trust, &attitude.fear, &attitude.anger,
                    &attitude.joy, &attitude.sorrow, &attitude.disgust, &attitude.surprise,
                    &attitude.curiosity, &attitude.respect, &attitude.suspicion,
                    &attitude.gratitude, &attitude.jealousy, &attitude.empathy,
                    &current_time, &current_time
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
        
        let attitude = stmt.query_row(&[&companion_id, &target_id, target_type], |row| {
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
            &[&delta, &current_time, &companion_id, &target_id, target_type]
        )?;
        
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
        
        con.execute(&query, &[&event, &attitude_id])?;
        
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
                    &[
                        &data.relationship_to_user, &data.relationship_to_companion,
                        &data.occupation, &data.personality_traits, &data.physical_description,
                        &current_time, &current_time, &id
                    ]
                )?;
            } else {
                con.execute(
                    "UPDATE third_party_individuals SET 
                        last_mentioned = ?, mention_count = mention_count + 1, updated_at = ?
                    WHERE id = ?",
                    &[&current_time, &current_time, &id]
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
                &[
                    &data.name, &data.relationship_to_user, &data.relationship_to_companion,
                    &data.occupation, &data.personality_traits, &data.physical_description,
                    &data.first_mentioned, &data.mention_count, &data.importance_score,
                    &data.created_at, &data.updated_at
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
            &[
                &third_party_id, &companion_id, &memory.memory_type, &memory.content,
                &memory.importance, &memory.emotional_valence, &current_time,
                &memory.context_message_id
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
            &[
                &interaction.third_party_id, &interaction.companion_id,
                &interaction.interaction_type, &interaction.description,
                &interaction.planned_date, &interaction.impact_on_relationship,
                &current_time, &current_time
            ]
        )?;
        
        Ok(con.last_insert_rowid() as i32)
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
            &[&new_importance, &current_time, &third_party_id]
        )?;
        
        Ok(())
    }
}
