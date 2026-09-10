//! Integration test for #174: `GET /api/debug/prompt` honours
//! `compacted_through` end to end — the assembled history starts strictly
//! after it, exactly the invariant `TranscriptSource::recent_messages` is
//! meant to hold everywhere a turn's prompt is built.

use std::process::{Command, Stdio};
use std::time::Duration;

use rusqlite::Connection;
use serde_json::{json, Value};

mod common;
use common::spawn_on_a_free_port;

fn post_message(agent: &ureq::Agent, addr: &str, speaker_id: &str, content: &str) {
    let status = agent
        .post(format!("http://{addr}/api/message"))
        .send_json(json!({ "speaker_id": speaker_id, "content": content }))
        .unwrap_or_else(|e| panic!("POST /api/message failed at the transport level: {e}"))
        .status();
    assert!(status.is_success(), "POST /api/message returned {status}");
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

#[test]
fn debug_prompt_history_starts_strictly_after_compacted_through() {
    let data_dir = tempfile::tempdir().expect("failed to create the data-dir temp dir");

    let (_port, addr, _guard) = spawn_on_a_free_port(
        |port| {
            let mut command = Command::new(env!("CARGO_BIN_EXE_ai-companion"));
            command
                .env("COMPANION_HOST", "127.0.0.1")
                .env("COMPANION_PORT", port.to_string())
                .env("COMPANION_DATA_DIR", data_dir.path())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            command
        },
        Duration::from_secs(10),
    );
    let addr = addr.to_string();

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();

    for i in 1..=12 {
        post_message(&agent, &addr, "user", &format!("message {i}"));
    }

    // Set directly in the database rather than through a route: #175 (the
    // commit path that would set this through the API) is a sibling issue,
    // not yet available to this test.
    let cutoff: i64 = 6;
    {
        let con = Connection::open(data_dir.path().join("companion_database.db"))
            .expect("failed to open the companion database directly");
        con.execute(
            "UPDATE companion SET compacted_through = ?1 WHERE id = 1",
            rusqlite::params![cutoff],
        )
        .expect("failed to set compacted_through directly");
    }

    let body = get_json(&agent, &format!("http://{addr}/api/debug/prompt"));

    assert_eq!(
        body["compacted_through"],
        json!(cutoff),
        "unexpected /api/debug/prompt body: {body}"
    );
    let managed_messages = body["managed_messages"]
        .as_array()
        .expect("managed_messages should be a JSON array");
    assert!(
        !managed_messages.is_empty(),
        "expected some messages to remain after the cutoff"
    );
    for message in managed_messages {
        let id = message["id"]
            .as_i64()
            .expect("message id should be an integer");
        assert!(
            id > cutoff,
            "message id {id} should be greater than compacted_through {cutoff}"
        );
    }
}
