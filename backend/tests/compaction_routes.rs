//! Integration test for #179's HTTP surface: the manual trigger's `409`
//! reason, an empty `GET /api/compaction` listing on a fresh chat, and the
//! pin routes end to end (`GET /api/message` reporting `pinned`, and a
//! pinned message's content showing up in `GET /api/debug/prompt`'s
//! rendered `pins` block). The `202` extraction path needs a real GGUF and
//! is verified manually, per this issue's plan.

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
