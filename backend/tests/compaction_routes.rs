//! Integration test for #179's HTTP surface: the manual trigger's `409`
//! reason, an empty `GET /api/compaction` listing on a fresh chat, and the
//! pin routes end to end (`GET /api/message` reporting `pinned`, and a
//! pinned message's content showing up in `GET /api/debug/prompt`'s
//! rendered `pins` block). A *successful* `202` extraction still needs a
//! real GGUF and is verified manually, per this issue's plan — but #208's
//! failure path needs no model at all: this binary's own default
//! `llm_model_path` (the literal placeholder `path/to/your/gguf/model.gguf`)
//! fails to load immediately, so `an_extraction_failure_...` below exercises
//! the real `run_extraction_job`/`fail_pending_draft` wiring end to end
//! without one.

use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::{json, Value};

mod common;
use common::spawn_on_a_free_port;

fn post_message(agent: &ureq::Agent, addr: &str, speaker_id: &str, content: &str) -> i64 {
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
        .expect("the newest message should carry an id")
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
fn manual_trigger_listing_and_pins_work_end_to_end_on_a_short_chat() {
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

    // Three messages is well under the default `compact_min_messages` (8),
    // so the manual trigger must refuse rather than queue a draft.
    let mut last_id = 0;
    for i in 1..=3 {
        last_id = post_message(&agent, &addr, "user", &format!("message {i}"));
    }

    let draft_response = agent
        .post(format!("http://{addr}/api/compaction/draft"))
        .send_empty()
        .unwrap_or_else(|e| {
            panic!("POST /api/compaction/draft failed at the transport level: {e}")
        });
    assert_eq!(draft_response.status(), 409);
    let draft_body = draft_response
        .into_body()
        .read_to_string()
        .expect("409 body should be readable text");
    assert!(
        draft_body.contains("compact_min_messages")
            || draft_body.contains("compaction needs at least"),
        "409 body should name the minimum message requirement, got: {draft_body}"
    );

    // No draft was queued, so the listing is empty.
    let listing = get_json(&agent, &format!("http://{addr}/api/compaction"));
    assert_eq!(listing["checkpoints"], json!([]));
    assert_eq!(listing["pending_draft"], Value::Null);

    // Pinning a real message id makes it show up as `pinned: true` and in
    // the rendered prompt's `pins` block.
    let pin_status = agent
        .post(format!("http://{addr}/api/message/{last_id}/pin"))
        .send_empty()
        .unwrap_or_else(|e| {
            panic!("POST /api/message/{{id}}/pin failed at the transport level: {e}")
        })
        .status();
    assert!(pin_status.is_success(), "pin returned {pin_status}");

    let messages = get_json(&agent, &format!("http://{addr}/api/message"));
    let pinned_message = messages["messages"]
        .as_array()
        .expect("messages should be a JSON array")
        .iter()
        .find(|m| m["id"].as_i64() == Some(last_id))
        .expect("the pinned message should be in the page");
    assert_eq!(pinned_message["pinned"], json!(true));

    let debug_prompt = get_json(&agent, &format!("http://{addr}/api/debug/prompt"));
    let pins_block = debug_prompt["compaction"]["pins"]
        .as_str()
        .expect("compaction.pins should be a string");
    assert!(
        pins_block.contains("message 3"),
        "pins block should contain the pinned message's content, got: {pins_block}"
    );

    // Unpinning the same message clears `pinned` and drops it from the
    // rendered prompt's `pins` block, so the round trip is exercised on
    // both legs, not just the pin.
    let unpin_status = agent
        .delete(format!("http://{addr}/api/message/{last_id}/pin"))
        .call()
        .unwrap_or_else(|e| {
            panic!("DELETE /api/message/{{id}}/pin failed at the transport level: {e}")
        })
        .status();
    assert!(unpin_status.is_success(), "unpin returned {unpin_status}");

    let messages_after_unpin = get_json(&agent, &format!("http://{addr}/api/message"));
    let unpinned_message = messages_after_unpin["messages"]
        .as_array()
        .expect("messages should be a JSON array")
        .iter()
        .find(|m| m["id"].as_i64() == Some(last_id))
        .expect("the unpinned message should still be in the page");
    assert_eq!(unpinned_message["pinned"], json!(false));

    let debug_prompt_after_unpin = get_json(&agent, &format!("http://{addr}/api/debug/prompt"));
    let pins_block_after_unpin = debug_prompt_after_unpin["compaction"]["pins"]
        .as_str()
        .expect("compaction.pins should be a string");
    assert!(
        !pins_block_after_unpin.contains("message 3"),
        "pins block should no longer contain the unpinned message's content, got: {pins_block_after_unpin}"
    );

    // Pinning an id that names no message is `404`.
    let unknown_status = agent
        .post(format!("http://{addr}/api/message/999999/pin"))
        .send_empty()
        .unwrap_or_else(|e| {
            panic!("POST /api/message/{{id}}/pin failed at the transport level: {e}")
        })
        .status();
    assert_eq!(unknown_status, 404);

    // Unpinning an id that names no message is also `404`.
    let unknown_unpin_status = agent
        .delete(format!("http://{addr}/api/message/999999/pin"))
        .call()
        .unwrap_or_else(|e| {
            panic!("DELETE /api/message/{{id}}/pin failed at the transport level: {e}")
        })
        .status();
    assert_eq!(unknown_unpin_status, 404);
}

/// #181's `{"from_stale": true}` body through the real
/// `POST /api/compaction/draft` route (folded into #179's handler on
/// rebase), not just `select_recompaction_range`/`oldest_stale_from` in
/// isolation: the `409` when there is nothing stale to re-compact, and a
/// `202` that queues a draft starting at the stale checkpoint's own
/// `from_message_id` once one exists. Seeds the `Stale` checkpoint row
/// directly against the server's own SQLite file — a real one requires a
/// full extract-then-commit cycle, which needs a GGUF model, the same
/// limitation the manual-trigger `202` path already has (see this file's
/// header comment).
#[test]
fn from_stale_true_queues_a_recompaction_draft_over_the_stale_checkpoints_range() {
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

    // The default `short_term_mem` (5, seeded by `Database::init`) needs at
    // least a few messages past the stale checkpoint's end for
    // `select_recompaction_range` to find anything to re-compact. 20, not
    // 15: the stale checkpoint below starts at message 6, not message 1,
    // specifically so the two branches inside the handler cannot produce
    // the same range by coincidence — see the comment there.
    for i in 1..=20 {
        post_message(&agent, &addr, "user", &format!("message {i}"));
    }

    // No checkpoint exists at all yet, let alone a stale one: `409` through
    // the real route, not just `oldest_stale_from` returning `None` in a
    // unit test.
    let no_stale_response = agent
        .post(format!("http://{addr}/api/compaction/draft"))
        .send_json(json!({ "from_stale": true }))
        .unwrap_or_else(|e| {
            panic!("POST /api/compaction/draft failed at the transport level: {e}")
        });
    assert_eq!(no_stale_response.status(), 409);
    let no_stale_body = no_stale_response
        .into_body()
        .read_to_string()
        .expect("409 body should be readable text");
    assert!(
        no_stale_body.contains("no stale checkpoint"),
        "409 body should say there is no stale checkpoint, got: {no_stale_body}"
    );

    // Seed a `Stale` checkpoint directly against the server's own database
    // file: a real one requires a committed checkpoint plus an edit inside
    // its range, which this binary cannot reach without a loaded model.
    //
    // Its `from_message_id` is 6, deliberately not 1: on a fresh chat with
    // `compacted_through` still `NULL`, the *ordinary* manual-trigger path
    // (`select_range`) also starts at message 1 (the first eligible
    // message), so a stale range starting at 1 would make the `202`
    // assertions below pass identically whether or not the `from_stale`
    // branch actually ran — a test that cannot fail when the feature is
    // skipped is not coverage of it. Starting the stale range at 6 means
    // only `select_recompaction_range` (which anchors on
    // `oldest_stale_from`, not the first eligible message) can produce
    // `from_message_id: 6`; the manual path can only ever produce `1` here.
    let db_path = data_dir.path().join("companion_database.db");
    {
        let con = rusqlite::Connection::open(&db_path)
            .expect("failed to open the server's own database file");
        let companion_id: i32 = con
            .query_row("SELECT id FROM companion LIMIT 1", [], |row| row.get(0))
            .expect("the server should have seeded a companion row at startup");
        con.execute(
            "INSERT INTO compactions (companion_id, from_message_id, through_message_id, status, trigger, created_at) VALUES (?, 6, 10, 'stale', 'threshold', 'now')",
            [companion_id],
        )
        .expect("failed to seed a stale checkpoint row");
    }

    let queued_response = agent
        .post(format!("http://{addr}/api/compaction/draft"))
        .send_json(json!({ "from_stale": true }))
        .unwrap_or_else(|e| {
            panic!("POST /api/compaction/draft failed at the transport level: {e}")
        });
    assert_eq!(
        queued_response.status(),
        202,
        "expected the stale checkpoint's range to queue a draft"
    );
    let queued_body: Value = queued_response
        .into_body()
        .read_json()
        .expect("202 body should be valid JSON");
    let draft_id = queued_body["draft_id"]
        .as_i64()
        .expect("202 body should carry draft_id");

    // Visible through the listing route too, starting at the stale
    // checkpoint's own `from_message_id` (6) — not message 1, which is
    // where the ordinary manual-trigger path would have started instead
    // (see the seeding comment above for why that distinction is the
    // whole point of this assertion).
    let listing = get_json(&agent, &format!("http://{addr}/api/compaction"));
    assert_eq!(listing["pending_draft"]["id"].as_i64(), Some(draft_id));
    assert_eq!(
        listing["pending_draft"]["from_message_id"].as_i64(),
        Some(6)
    );
}

/// #208: an extraction that fails must leave the checkpoint in an explicit
/// terminal state, expose the reason over the API, and unblock the next
/// trigger — never stuck in `Draft`/`extracting` forever. Runs the real
/// `POST /api/compaction/draft` -> background `spawn_extraction` ->
/// `run_extraction_job` pipeline end to end, no mocks or seeded rows:
/// this binary's default `llm_model_path` (`path/to/your/gguf/model.gguf`,
/// see `database.rs`'s `Database::init` seeding) names a file that does not
/// exist and `compaction_model_path` is unset, so `ResidentExtractor` falls
/// back to it and `LlamaModel::load_from_file` fails immediately with a
/// clean `std::io::Error` — the same `DraftError::Model` class of failure
/// #207's grammar bug produced before it was fixed, reached here without
/// needing a real GGUF.
///
/// Before #208, this test would time out waiting for the status to leave
/// `draft` (the checkpoint was never touched again after the failed
/// extraction), and the final re-trigger assertion would fail with `409
/// AlreadyPending` instead of `202`.
#[test]
fn an_extraction_failure_without_a_configured_model_reaches_failed_and_unblocks_the_trigger() {
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

    // Default `short_term_mem` is 5, default `compact_min_messages` is 8
    // (see the `from_stale` test above for the same arithmetic): 20 user
    // messages clears `20 - 5 >= 8` with room to spare.
    for i in 1..=20 {
        post_message(&agent, &addr, "user", &format!("message {i}"));
    }

    let draft_response = agent
        .post(format!("http://{addr}/api/compaction/draft"))
        .send_empty()
        .unwrap_or_else(|e| {
            panic!("POST /api/compaction/draft failed at the transport level: {e}")
        });
    assert_eq!(
        draft_response.status(),
        202,
        "expected enough uncompacted messages to queue a draft"
    );
    let queued_body: Value = draft_response
        .into_body()
        .read_json()
        .expect("202 body should be valid JSON");
    let draft_id = queued_body["draft_id"]
        .as_i64()
        .expect("202 body should carry draft_id");

    // Poll until the checkpoint leaves `draft` status. No real model load
    // is attempted (the configured path does not exist), so this resolves
    // almost immediately; the generous deadline is headroom for a loaded
    // CI box, not for anything resembling real inference.
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut detail;
    loop {
        detail = get_json(&agent, &format!("http://{addr}/api/compaction/{draft_id}"));
        if detail["status"] != json!("draft") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "extraction never left Draft status within the deadline; last detail: {detail}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    assert_eq!(
        detail["status"],
        json!("failed"),
        "a model-load failure should reach the Failed terminal state, got: {detail}"
    );
    let error = detail["extraction_error"]
        .as_str()
        .expect("extraction_error should be a readable string once status is failed");
    assert!(
        !error.is_empty(),
        "extraction_error should explain the failure, not just mark it"
    );

    // The reason is readable from the listing route too, not just the
    // per-checkpoint detail route.
    let listing = get_json(&agent, &format!("http://{addr}/api/compaction"));
    assert_eq!(
        listing["pending_draft"],
        Value::Null,
        "a failed draft must not still read as pending"
    );
    let summary = listing["checkpoints"]
        .as_array()
        .expect("checkpoints should be a JSON array")
        .iter()
        .find(|c| c["id"].as_i64() == Some(draft_id))
        .expect("the failed checkpoint should still be listed");
    assert_eq!(summary["status"], json!("failed"));
    assert_eq!(summary["extraction_error"].as_str(), Some(error));

    // The core of #208's bug: a failed draft must not block the next
    // trigger. Without the fix this 409s with "a draft is already pending".
    let second_response = agent
        .post(format!("http://{addr}/api/compaction/draft"))
        .send_empty()
        .unwrap_or_else(|e| {
            panic!("POST /api/compaction/draft failed at the transport level: {e}")
        });
    assert_eq!(
        second_response.status(),
        202,
        "a failed draft must not block a fresh trigger"
    );
}
