//! Integration tests for #171: a database built with the pre-compaction
//! schema migrates cleanly on the very first startup with the compaction
//! code paths compiled in (no data loss, every new table/column present),
//! and a fresh database starts with the documented compaction config
//! defaults and persists changes to them.
//!
//! Builds its own legacy-shaped SQLite file at test time
//! (`create_legacy_db` below) rather than committing one: runtime `*.db`
//! files hold real chat text and are gitignored.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use rusqlite::{params, Connection};
use serde_json::{json, Value};

mod common;
use common::spawn_on_a_free_port;

fn instance_command(port: u16, data_dir: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ai-companion"));
    command
        .env("COMPANION_HOST", "127.0.0.1")
        .env("COMPANION_PORT", port.to_string())
        .env("COMPANION_DATA_DIR", data_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

fn get_json(agent: &ureq::Agent, url: &str) -> Value {
    agent
        .get(url)
        .call()
        .unwrap_or_else(|e| panic!("GET {url} failed at the transport level: {e}"))
        .body_mut()
        .read_json()
        .unwrap_or_else(|e| panic!("GET {url} did not return valid JSON: {e}"))
}

fn put_config(agent: &ureq::Agent, url: &str, config: &Value) -> u16 {
    agent
        .put(url)
        .send_json(config)
        .unwrap_or_else(|e| panic!("PUT {url} failed at the transport level: {e}"))
        .status()
        .as_u16()
}

/// The `messages`/`companion`/`config` shape `Database::init` produced
/// before #125 (`speaker_id`) and #171 (`compacted_through`, and the four
/// new `config` columns): no `speaker_id` column, a `companion` with no
/// `compacted_through`, and the original four-column `config`. Seeds a
/// handful of synthetic rows (no real chat text) so the migration
/// assertions below have something to preserve.
fn create_legacy_db(path: &Path) {
    let con = Connection::open(path).expect("failed to create the legacy database file");

    con.execute(
        "CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ai BOOLEAN,
            content TEXT,
            created_at TEXT
        )",
        [],
    )
    .unwrap();
    for (ai, content) in [(0, "hello"), (1, "hi there"), (0, "how are you")] {
        con.execute(
            "INSERT INTO messages (ai, content, created_at) VALUES (?, ?, '2024-01-01')",
            params![ai, content],
        )
        .unwrap();
    }

    con.execute(
        "CREATE TABLE companion (
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
        "INSERT INTO companion (name, persona, example_dialogue, first_message, long_term_mem, short_term_mem, roleplay, dialogue_tuning, avatar_path) VALUES ('Assistant', 'persona', 'example', 'Hello {{user}}', 2, 5, 1, 1, '')",
        [],
    )
    .unwrap();

    con.execute(
        "CREATE TABLE config (
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

#[test]
fn legacy_database_migrates_without_data_loss() {
    let data_dir = tempfile::tempdir().expect("failed to create the data-dir temp dir");
    create_legacy_db(&data_dir.path().join("companion_database.db"));

    // `spawn_on_a_free_port` only returns once the process is listening,
    // which happens after `main()`'s `init_storage()` (and therefore every
    // migration) has already run to completion.
    let (_port, _addr, guard) = spawn_on_a_free_port(
        |port| instance_command(port, data_dir.path()),
        Duration::from_secs(10),
    );
    drop(guard);

    let con = Connection::open(data_dir.path().join("companion_database.db"))
        .expect("failed to reopen the migrated database");

    let message_count: i64 = con
        .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
        .unwrap();
    assert_eq!(message_count, 3, "seeded messages must survive migration");

    let companion_count: i64 = con
        .query_row("SELECT COUNT(*) FROM companion", [], |row| row.get(0))
        .unwrap();
    assert_eq!(
        companion_count, 1,
        "seeded companion row must survive migration"
    );

    let config_count: i64 = con
        .query_row("SELECT COUNT(*) FROM config", [], |row| row.get(0))
        .unwrap();
    assert_eq!(config_count, 1, "seeded config row must survive migration");

    let compacted_through: Option<i32> = con
        .query_row(
            "SELECT compacted_through FROM companion LIMIT 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(compacted_through, None);

    for table in ["compactions", "compaction_facts", "pinned_messages"] {
        let exists: i64 = con
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
                [table],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(exists, 1, "expected table {table} to exist after migration");
    }

    let mut stmt = con.prepare("PRAGMA foreign_key_check").unwrap();
    let violations: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(
        violations.is_empty(),
        "expected no foreign key violations, got {violations:?}"
    );
}

#[test]
fn fresh_database_has_default_compaction_config_and_persists_updates() {
    let data_dir = tempfile::tempdir().expect("failed to create the data-dir temp dir");

    let (_port, addr, _guard) = spawn_on_a_free_port(
        |port| instance_command(port, data_dir.path()),
        Duration::from_secs(10),
    );

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();
    let config_url = format!("http://{addr}/api/config");

    let config = get_json(&agent, &config_url);
    assert_eq!(config["compact_threshold_tokens"], Value::Null);
    assert_eq!(config["compact_min_messages"], json!(8));
    assert_eq!(config["compaction_model_path"], Value::Null);
    assert_eq!(config["heuristic_person_detection"], json!(true));

    let mut updated = config;
    updated["compact_min_messages"] = json!(12);
    updated["heuristic_person_detection"] = json!(false);
    assert_eq!(put_config(&agent, &config_url, &updated), 200);

    let config = get_json(&agent, &config_url);
    assert_eq!(config["compact_min_messages"], json!(12));
    assert_eq!(config["heuristic_person_detection"], json!(false));
}
