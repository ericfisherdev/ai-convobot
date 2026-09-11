//! Route-level integration test for #217: the running-thoughts read/write
//! HTTP surface (`GET`/`PATCH`/`DELETE /api/thoughts`,
//! `POST /api/thoughts/regenerate`) against a real server process and a
//! real SQLite file. There is no insert route and no model runs in CI, so
//! thoughts are seeded directly against the server's own database file with
//! `rusqlite::Connection::open`, the same pattern
//! `compaction_routes.rs`'s stale-checkpoint test uses.

use std::net::SocketAddr;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

mod common;
use common::{spawn_on_a_free_port, wait_until_listening, ChildGuard};

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

/// Respawns an instance on a port already secured by an earlier
/// [`spawn_on_a_free_port`] call, mirroring `multiplayer_modes.rs`'s helper
/// of the same name: used here after a `PUT /api/config` mode change, which
/// (like every other config field `main()`'s own startup wiring reads once)
/// only takes effect on the next restart.
fn respawn_instance(port: u16, addr: SocketAddr, data_dir: &Path) -> ChildGuard {
    let child = instance_command(port, data_dir)
        .spawn()
        .expect("failed to spawn the ai-companion binary");
    let guard = ChildGuard(child);
    wait_until_listening(addr, Duration::from_secs(10));
    guard
}

fn post_message(agent: &ureq::Agent, addr: &str, speaker_id: &str, content: &str) -> i32 {
    let response = agent
        .post(format!("http://{addr}/api/message"))
        .send_json(json!({ "speaker_id": speaker_id, "content": content }))
        .unwrap_or_else(|e| panic!("POST /api/message failed at the transport level: {e}"));
    assert!(
        response.status().is_success(),
        "POST /api/message returned {}",
        response.status()
    );
    let body = get_json(agent, &format!("http://{addr}/api/message?limit=1"));
    body["messages"][0]["id"]
        .as_i64()
        .expect("the newest message should carry an id") as i32
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

fn put_config(agent: &ureq::Agent, addr: &str, config: &Value) {
    let response = agent
        .put(format!("http://{addr}/api/config"))
        .send_json(config)
        .unwrap_or_else(|e| panic!("PUT /api/config failed at the transport level: {e}"));
    assert!(
        response.status().is_success(),
        "PUT /api/config returned {}",
        response.status()
    );
}

/// Flips `running_thoughts_enabled` on, round-tripping the rest of `GET
/// /api/config`'s body straight back through `PUT` (see
/// `compaction_routes.rs::set_invalid_model_path`'s identical rationale for
/// why: `ConfigModify` ignores unknown fields, so this needs no field-by-
/// field translation).
fn enable_running_thoughts(agent: &ureq::Agent, addr: &str) {
    let mut config = get_json(agent, &format!("http://{addr}/api/config"));
    config["running_thoughts_enabled"] = json!(true);
    put_config(agent, addr, &config);
}

/// Points `llm_model_path` at a path that does not exist, so a
/// `POST /api/thoughts/regenerate` call fails at generation without a real
/// GGUF loaded. Mirrors `compaction_routes.rs::set_invalid_model_path`
/// exactly (including leaving `compaction_model_path` unset).
fn set_invalid_model_path(agent: &ureq::Agent, addr: &str) {
    let mut config = get_json(agent, &format!("http://{addr}/api/config"));
    config["llm_model_path"] = json!("/definitely/does/not/exist.gguf");
    put_config(agent, addr, &config);
}

/// Seeds one `running_thoughts` row directly against the server's own
/// database file and returns its id.
fn seed_thought(
    con: &rusqlite::Connection,
    companion_id: i32,
    speaker_id: &str,
    from_message_id: i32,
    through_message_id: i32,
    text: &str,
) -> i64 {
    con.execute(
        "INSERT INTO running_thoughts (companion_id, speaker_id, from_message_id, through_message_id, text, edited, created_at) VALUES (?, ?, ?, ?, ?, 0, 'now')",
        rusqlite::params![companion_id, speaker_id, from_message_id, through_message_id, text],
    )
    .expect("failed to seed a running_thoughts row");
    con.last_insert_rowid()
}

fn seeded_companion_id(con: &rusqlite::Connection) -> i32 {
    con.query_row("SELECT id FROM companion LIMIT 1", [], |row| row.get(0))
        .expect("the server should have seeded a companion row at startup")
}

/// Reads an SSE response body to its end (the connection closes once the
/// terminal chunk has gone out) and parses each `data: ` record as JSON, in
/// order.
fn read_sse_events(response: ureq::http::Response<ureq::Body>) -> Vec<Value> {
    let body = response
        .into_body()
        .read_to_string()
        .expect("SSE body should be readable to its end");
    body.split("\n\n")
        .map(|record| record.trim())
        .filter(|record| !record.is_empty())
        .map(|record| {
            let payload = record.strip_prefix("data: ").unwrap_or(record);
            serde_json::from_str(payload)
                .unwrap_or_else(|e| panic!("SSE record was not valid JSON ({e}): {payload}"))
        })
        .collect()
}

/// Polls `check` every 100ms until it returns `true` or `deadline` elapses.
fn wait_until(deadline: Duration, mut check: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    loop {
        if check() {
            return true;
        }
        if start.elapsed() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn list_edit_and_delete_work_end_to_end() {
    let data_dir = tempfile::tempdir().expect("failed to create the data-dir temp dir");

    let (_port, addr, _guard) = spawn_on_a_free_port(
        |port| instance_command(port, data_dir.path()),
        Duration::from_secs(10),
    );
    let addr = addr.to_string();

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();

    let db_path = data_dir.path().join("companion_database.db");
    let (first_id, second_id, third_id) = {
        let con = rusqlite::Connection::open(&db_path)
            .expect("failed to open the server's own database file");
        let companion_id = seeded_companion_id(&con);
        let first = seed_thought(&con, companion_id, "char", 1, 2, "first note");
        let second = seed_thought(&con, companion_id, "char", 3, 4, "second note");
        let third = seed_thought(&con, companion_id, "user", 5, 6, "third note");
        (first, second, third)
    };

    // `GET` returns every thought, oldest first.
    let listing = get_json(&agent, &format!("http://{addr}/api/thoughts"));
    let ids: Vec<i64> = listing["thoughts"]
        .as_array()
        .expect("thoughts should be a JSON array")
        .iter()
        .map(|t| t["id"].as_i64().unwrap())
        .collect();
    assert_eq!(ids, vec![first_id, second_id, third_id]);

    // `PATCH` rewrites the text and flips `edited`.
    let patch_response = agent
        .patch(format!("http://{addr}/api/thoughts/{second_id}"))
        .send_json(json!({ "text": "the user's own words" }))
        .unwrap_or_else(|e| {
            panic!("PATCH /api/thoughts/{{id}} failed at the transport level: {e}")
        });
    assert_eq!(patch_response.status(), 200);
    let patched: Value = patch_response
        .into_body()
        .read_json()
        .expect("PATCH response should be valid JSON");
    assert_eq!(patched["text"], "the user's own words");
    assert_eq!(patched["edited"], true);

    let listing_after_patch = get_json(&agent, &format!("http://{addr}/api/thoughts"));
    let second_after_patch = listing_after_patch["thoughts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_i64() == Some(second_id))
        .expect("the edited thought should still be listed");
    assert_eq!(second_after_patch["text"], "the user's own words");
    assert_eq!(second_after_patch["edited"], true);

    // `PATCH` on an unknown id is `404`.
    let unknown_patch = agent
        .patch(format!("http://{addr}/api/thoughts/999999"))
        .send_json(json!({ "text": "whatever" }))
        .unwrap_or_else(|e| {
            panic!("PATCH /api/thoughts/{{id}} failed at the transport level: {e}")
        });
    assert_eq!(unknown_patch.status(), 404);

    // Empty text (after trimming) is `422`.
    let empty_patch = agent
        .patch(format!("http://{addr}/api/thoughts/{first_id}"))
        .send_json(json!({ "text": "   " }))
        .unwrap_or_else(|e| {
            panic!("PATCH /api/thoughts/{{id}} failed at the transport level: {e}")
        });
    assert_eq!(empty_patch.status(), 422);
    let empty_patch_body: Value = empty_patch
        .into_body()
        .read_json()
        .expect("422 body should be valid JSON");
    assert_eq!(empty_patch_body["reason"], "text must not be empty");

    // `DELETE` removes the row and reports `200`; deleting it again is `404`.
    let delete_response = agent
        .delete(format!("http://{addr}/api/thoughts/{third_id}"))
        .call()
        .unwrap_or_else(|e| {
            panic!("DELETE /api/thoughts/{{id}} failed at the transport level: {e}")
        });
    assert_eq!(delete_response.status(), 200);

    let listing_after_delete = get_json(&agent, &format!("http://{addr}/api/thoughts"));
    let ids_after_delete: Vec<i64> = listing_after_delete["thoughts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["id"].as_i64().unwrap())
        .collect();
    assert_eq!(ids_after_delete, vec![first_id, second_id]);

    let redelete_response = agent
        .delete(format!("http://{addr}/api/thoughts/{third_id}"))
        .call()
        .unwrap_or_else(|e| {
            panic!("DELETE /api/thoughts/{{id}} failed at the transport level: {e}")
        });
    assert_eq!(redelete_response.status(), 404);
}

#[test]
fn editing_or_deleting_a_chat_message_leaves_every_thought_untouched() {
    let data_dir = tempfile::tempdir().expect("failed to create the data-dir temp dir");

    let (_port, addr, _guard) = spawn_on_a_free_port(
        |port| instance_command(port, data_dir.path()),
        Duration::from_secs(10),
    );
    let addr = addr.to_string();

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();

    let first_id = post_message(&agent, &addr, "user", "hello there");
    let second_id = post_message(&agent, &addr, "user", "how are you");
    let third_id = post_message(&agent, &addr, "user", "goodbye");

    let db_path = data_dir.path().join("companion_database.db");
    {
        let con = rusqlite::Connection::open(&db_path)
            .expect("failed to open the server's own database file");
        let companion_id = seeded_companion_id(&con);
        seed_thought(
            &con,
            companion_id,
            "char",
            first_id,
            third_id,
            "a note about the whole exchange",
        );
    }

    let before = get_json(&agent, &format!("http://{addr}/api/thoughts"));

    // Editing and deleting messages inside the thought's own range.
    let edit_response = agent
        .put(format!("http://{addr}/api/message/{second_id}"))
        .send_json(json!({ "content": "how are you doing" }))
        .unwrap_or_else(|e| panic!("PUT /api/message/{{id}} failed at the transport level: {e}"));
    assert!(
        edit_response.status().is_success(),
        "PUT /api/message/{{id}} returned {}",
        edit_response.status()
    );

    let delete_response = agent
        .delete(format!("http://{addr}/api/message/{third_id}"))
        .call()
        .unwrap_or_else(|e| {
            panic!("DELETE /api/message/{{id}} failed at the transport level: {e}")
        });
    assert!(
        delete_response.status().is_success(),
        "DELETE /api/message/{{id}} returned {}",
        delete_response.status()
    );

    let after = get_json(&agent, &format!("http://{addr}/api/thoughts"));
    assert_eq!(
        before, after,
        "editing/deleting a chat message must never change existing running thoughts"
    );
}

#[test]
fn regenerate_without_a_model_restores_the_originals_and_ends_with_an_error_chunk() {
    let data_dir = tempfile::tempdir().expect("failed to create the data-dir temp dir");

    let (_port, addr, _guard) = spawn_on_a_free_port(
        |port| instance_command(port, data_dir.path()),
        Duration::from_secs(10),
    );
    let addr = addr.to_string();

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();

    enable_running_thoughts(&agent, &addr);
    set_invalid_model_path(&agent, &addr);

    let first_id = post_message(&agent, &addr, "user", "message one");
    let second_id = post_message(&agent, &addr, "user", "message two");
    let third_id = post_message(&agent, &addr, "user", "message three");

    let db_path = data_dir.path().join("companion_database.db");
    let earlier_id = {
        let con = rusqlite::Connection::open(&db_path)
            .expect("failed to open the server's own database file");
        let companion_id = seeded_companion_id(&con);
        let earlier = seed_thought(&con, companion_id, "char", first_id, first_id, "round one");
        seed_thought(
            &con,
            companion_id,
            "char",
            second_id,
            second_id,
            "round two",
        );
        seed_thought(
            &con,
            companion_id,
            "char",
            third_id,
            third_id,
            "round three",
        );
        earlier
    };

    let response = agent
        .post(format!("http://{addr}/api/thoughts/regenerate"))
        .send_json(json!({ "from_message_id": second_id }))
        .unwrap_or_else(|e| {
            panic!("POST /api/thoughts/regenerate failed at the transport level: {e}")
        });
    assert_eq!(response.status(), 200);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .map(|v| v.to_str().unwrap_or("")),
        Some("text/event-stream")
    );

    let events = read_sse_events(response);
    let terminal = events
        .last()
        .expect("the stream should carry at least one event");
    assert_eq!(terminal["event"], "error");
    assert!(terminal["is_complete"].as_bool().unwrap_or(false));

    let after = get_json(&agent, &format!("http://{addr}/api/thoughts"));
    let thoughts = after["thoughts"]
        .as_array()
        .expect("thoughts should be an array");
    assert_eq!(thoughts.len(), 3);

    let earlier = thoughts
        .iter()
        .find(|t| t["id"].as_i64() == Some(earlier_id))
        .expect("the untouched earlier thought should keep its own id");
    assert_eq!(earlier["text"], "round one");

    assert!(
        thoughts.iter().any(|t| t["text"] == "round two"),
        "round two's original text should have been restored"
    );
    assert!(
        thoughts.iter().any(|t| t["text"] == "round three"),
        "round three's original text should have been restored"
    );
}

#[test]
fn regenerate_with_nothing_after_the_given_message_reports_it() {
    let data_dir = tempfile::tempdir().expect("failed to create the data-dir temp dir");

    let (_port, addr, _guard) = spawn_on_a_free_port(
        |port| instance_command(port, data_dir.path()),
        Duration::from_secs(10),
    );
    let addr = addr.to_string();

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();

    enable_running_thoughts(&agent, &addr);

    let response = agent
        .post(format!("http://{addr}/api/thoughts/regenerate"))
        .send_json(json!({ "from_message_id": 1 }))
        .unwrap_or_else(|e| {
            panic!("POST /api/thoughts/regenerate failed at the transport level: {e}")
        });
    assert_eq!(response.status(), 200);

    let events = read_sse_events(response);
    let terminal = events
        .last()
        .expect("the stream should carry at least one event");
    assert_eq!(terminal["event"], "error");
    assert!(
        terminal["error"]
            .as_str()
            .unwrap_or("")
            .contains("nothing to regenerate"),
        "terminal chunk should say there is nothing to regenerate, got: {terminal}"
    );

    let listing = get_json(&agent, &format!("http://{addr}/api/thoughts"));
    assert_eq!(listing["thoughts"], json!([]));
}

#[test]
fn thought_routes_serve_the_local_table_in_joiner_mode() {
    let data_dir = tempfile::tempdir().expect("failed to create the data-dir temp dir");

    let (port, addr, _initial) = spawn_on_a_free_port(
        |port| instance_command(port, data_dir.path()),
        Duration::from_secs(10),
    );

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();

    {
        let mut config = get_json(&agent, &format!("http://{addr}/api/config"));
        config["multiplayer_mode"] = json!("joiner");
        // Never actually reached: nothing here depends on a live connection
        // to a host, only on this instance's own local table and mirror.
        config["multiplayer_host_address"] = json!("127.0.0.1:1");
        config["multiplayer_participant_id"] = json!("bot1");
        config["multiplayer_password"] = json!("test-secret");
        config["running_thoughts_enabled"] = json!(true);
        put_config(&agent, &addr.to_string(), &config);
    }
    drop(_initial);
    let _guard = respawn_instance(port, addr, data_dir.path());

    let addr = addr.to_string();

    // Wait until the joiner-mode startup wiring has actually taken effect
    // (`GET /api/multiplayer/status` reports its role) before touching the
    // thoughts routes, the same readiness check `multiplayer_modes.rs` uses.
    let became_joiner = wait_until(Duration::from_secs(10), || {
        get_json(&agent, &format!("http://{addr}/api/multiplayer/status"))["mode"] == "joiner"
    });
    assert!(became_joiner, "instance never reported joiner mode");

    // Unlike the compaction routes, `GET /api/thoughts` stays readable (and
    // empty) on a joiner rather than 409ing.
    let empty_listing = get_json(&agent, &format!("http://{addr}/api/thoughts"));
    assert_eq!(empty_listing["thoughts"], json!([]));

    // A seeded row round-trips through PATCH/DELETE exactly as it does in
    // solo/host mode.
    let db_path = data_dir.path().join("companion_database.db");
    let seeded_id = {
        let con = rusqlite::Connection::open(&db_path)
            .expect("failed to open the server's own database file");
        let companion_id = seeded_companion_id(&con);
        seed_thought(&con, companion_id, "bot1", 1, 2, "a joiner's own note")
    };

    let patch_response = agent
        .patch(format!("http://{addr}/api/thoughts/{seeded_id}"))
        .send_json(json!({ "text": "an edited note" }))
        .unwrap_or_else(|e| {
            panic!("PATCH /api/thoughts/{{id}} failed at the transport level: {e}")
        });
    assert_eq!(patch_response.status(), 200);

    let delete_response = agent
        .delete(format!("http://{addr}/api/thoughts/{seeded_id}"))
        .call()
        .unwrap_or_else(|e| {
            panic!("DELETE /api/thoughts/{{id}} failed at the transport level: {e}")
        });
    assert_eq!(delete_response.status(), 200);

    // Regenerating over an empty mirror (no host ever connected) ends with
    // the same nothing-to-regenerate error chunk solo/host mode gets.
    let regenerate_response = agent
        .post(format!("http://{addr}/api/thoughts/regenerate"))
        .send_json(json!({ "from_message_id": 1 }))
        .unwrap_or_else(|e| {
            panic!("POST /api/thoughts/regenerate failed at the transport level: {e}")
        });
    assert_eq!(regenerate_response.status(), 200);
    let events = read_sse_events(regenerate_response);
    let terminal = events
        .last()
        .expect("the stream should carry at least one event");
    assert_eq!(terminal["event"], "error");
    assert!(
        terminal["error"]
            .as_str()
            .unwrap_or("")
            .contains("nothing to regenerate"),
        "terminal chunk should say there is nothing to regenerate, got: {terminal}"
    );

    // Pinning that the two route groups differ on purpose: the compaction
    // routes stay host-only and still 409 here.
    let compaction_status = agent
        .get(format!("http://{addr}/api/compaction"))
        .call()
        .unwrap_or_else(|e| panic!("GET /api/compaction failed at the transport level: {e}"))
        .status();
    assert_eq!(compaction_status, 409);
}
