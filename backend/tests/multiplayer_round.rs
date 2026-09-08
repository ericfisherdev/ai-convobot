//! End-to-end test for #154: a real `ai-companion` process, in `host`
//! multiplayer mode, running a round against a `tokio-tungstenite` client
//! standing in for a joiner over the real `/api/multiplayer/ws` socket.
//!
//! Like `tests/env_config.rs`, this spawns the compiled binary rather than
//! calling into the crate directly: `main.rs` has no `[lib]` target, and
//! `paths::init`/`Database::open()` are process-wide globals that can only
//! be initialised once, so a real HTTP+WS round trip against a real
//! (temp-dir) database has to run out of process.
//!
//! The round is driven by `@bot1 hi` rather than a bare `hi`: `plan_round`
//! (#132) resolves an `@mention` of a single bot to a plan that never asks
//! the host companion (`char`) to speak at all (`round.rs`'s
//! `a_user_mention_of_one_bot_runs_only_it_and_finish_returns_none`), so
//! this test needs no GGUF model loaded — only the joiner's own (fake, this
//! test's own WS client) reply is ever generated.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};
use hmac::{Hmac, KeyInit, Mac};
use serde_json::{json, Value};
use sha2::Sha256;
use tokio_tungstenite::tungstenite::Message as WsMessage;

type HmacSha256 = Hmac<Sha256>;

/// Kills and reaps the wrapped child on drop — see `tests/env_config.rs`'s
/// identical guard for why this matters even when a test only reads state.
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind an ephemeral port");
    listener
        .local_addr()
        .expect("failed to read the ephemeral port's local address")
        .port()
}

fn wait_until_listening(addr: SocketAddr, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
            return;
        }
        if Instant::now() >= deadline {
            panic!("server at {addr} did not start listening within {timeout:?}");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// A minimal blocking HTTP/1.1 client good enough for this test's three
/// calls (`GET`/`PUT /api/config`, `GET /api/message`, `POST /api/prompt`):
/// no external HTTP client crate is a dependency of this crate, and adding
/// one just for this test is not worth it. `Connection: close` sidesteps
/// keep-alive/chunked-encoding handling entirely — read to EOF, split once
/// on the blank line that ends the headers.
fn http_request(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut stream = TcpStream::connect(addr).expect("failed to connect to the test server");
    // Generous: this call may be the one that blocks on the whole round
    // (`POST /api/prompt`), which this test's own WS client is answering
    // concurrently, but a hung protocol implementation should fail the test
    // rather than hang the suite forever.
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();

    let body = body.unwrap_or("");
    let mut request =
        format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n");
    if !body.is_empty() {
        request.push_str("Content-Type: application/json\r\n");
    }
    request.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    request.push_str(body);

    stream
        .write_all(request.as_bytes())
        .expect("failed to write the HTTP request");
    stream.shutdown(std::net::Shutdown::Write).ok();

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .expect("failed to read the HTTP response");
    let raw = String::from_utf8_lossy(&raw).into_owned();

    let mut parts = raw.splitn(2, "\r\n\r\n");
    let head = parts.next().unwrap_or("");
    let body = parts.next().unwrap_or("").to_string();
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    (status, body)
}

/// The HMAC-SHA256 join proof `handshake::join_proof` computes, reproduced
/// here since this test cannot import the crate's internals (no `[lib]`
/// target) — see `multiplayer::handshake`'s own known-answer-vector test
/// for the construction this mirrors: HMAC over `nonce || id`, keyed with
/// the host password.
fn join_proof(password: &str, nonce: &[u8], id: &str) -> String {
    let mut mac =
        HmacSha256::new_from_slice(password.as_bytes()).expect("HMAC accepts a key of any length");
    mac.update(nonce);
    mac.update(id.as_bytes());
    BASE64.encode(mac.finalize().into_bytes())
}

#[tokio::test]
async fn a_joiner_answers_a_generate_request_and_its_reply_is_persisted_and_broadcast() {
    let port = free_port();
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let data_dir = tempfile::tempdir().expect("failed to create the data-dir temp dir");

    let child = Command::new(env!("CARGO_BIN_EXE_ai-companion"))
        .env("COMPANION_HOST", "127.0.0.1")
        .env("COMPANION_PORT", port.to_string())
        .env("COMPANION_DATA_DIR", data_dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn the ai-companion binary");
    let _guard = ChildGuard(child);
    wait_until_listening(addr, Duration::from_secs(10));

    // Switch to `host` mode with a password, over the same `PUT
    // /api/config` a real deployment would use: read the current config,
    // flip the multiplayer fields, and write it back whole (`ConfigModify`
    // has no `#[serde(default)]` on most fields, so a partial body would
    // reject every other setting back to its type's default).
    let password = "hunter2-test-password";
    let (status, body) = http_request(addr, "GET", "/api/config", None);
    assert_eq!(status, 200, "GET /api/config failed: {body}");
    let mut config: Value = serde_json::from_str(&body).expect("config should be valid JSON");
    config["multiplayer_mode"] = json!("host");
    config["multiplayer_password"] = json!(password);
    // The minimum `MultiplayerConfig::parse` accepts (#128): keeps this
    // test's own failure mode fast if the round never completes, instead of
    // waiting out the default 120s.
    config["remote_generation_timeout_secs"] = json!(5);
    let (status, body) = http_request(
        addr,
        "PUT",
        "/api/config",
        Some(&serde_json::to_string(&config).unwrap()),
    );
    assert_eq!(status, 200, "PUT /api/config failed: {body}");

    // Connect the joiner over the real socket and complete the handshake —
    // the wire shapes here are pinned by `multiplayer::protocol`'s own
    // `join_has_the_exact_wire_shape_a_joiner_depends_on` test.
    let (mut ws, _) =
        tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/api/multiplayer/ws"))
            .await
            .expect("failed to connect to the multiplayer socket");

    let challenge = recv_json(&mut ws).await;
    assert_eq!(challenge["type"], "challenge");
    let nonce = BASE64
        .decode(challenge["nonce"].as_str().unwrap())
        .expect("nonce should be valid base64");

    let proof = join_proof(password, &nonce, "bot1");
    send_json(
        &mut ws,
        &json!({
            "type": "join",
            "protocol_version": 1,
            "id": "bot1",
            "display_name": "Bot One",
            "avatar": null,
            "proof": proof,
        }),
    )
    .await;

    let joined = recv_json(&mut ws).await;
    assert_eq!(joined["type"], "joined");
    assert_eq!(joined["self_id"], "bot1");

    // `POST /api/prompt` runs the whole round on the calling worker thread
    // and does not return until it settles, so it has to run concurrently
    // with this test driving the joiner's side of the socket.
    let prompt_addr = addr;
    let prompt_handle = tokio::task::spawn_blocking(move || {
        http_request(
            prompt_addr,
            "POST",
            "/api/prompt",
            Some(r#"{"prompt":"@bot1 hi"}"#),
        )
    });

    // The user's turn, broadcast to every joiner before any speaker
    // generates (#154's `run_round` broadcasts it first).
    let user_turn = recv_json(&mut ws).await;
    assert_eq!(user_turn["type"], "message");
    assert_eq!(user_turn["speaker_id"], "user");
    assert_eq!(user_turn["content"], "@bot1 hi");

    // `char` is never asked to speak in an `@bot1`-only round, so the next
    // frame is bot1's own `GenerateRequest`.
    let generate_request = recv_json(&mut ws).await;
    assert_eq!(generate_request["type"], "generate_request");
    let round_id = generate_request["round_id"].as_u64().unwrap();

    send_json(
        &mut ws,
        &json!({"type": "token", "round_id": round_id, "text": "hel"}),
    )
    .await;
    send_json(
        &mut ws,
        &json!({"type": "token", "round_id": round_id, "text": "lo"}),
    )
    .await;
    send_json(
        &mut ws,
        &json!({"type": "reply_complete", "round_id": round_id, "text": "hello from bot1"}),
    )
    .await;

    // The persisted reply, broadcast back out to every joiner (including
    // the one that just produced it).
    let reply_broadcast = recv_json(&mut ws).await;
    assert_eq!(reply_broadcast["type"], "message");
    assert_eq!(reply_broadcast["speaker_id"], "bot1");
    assert_eq!(reply_broadcast["content"], "hello from bot1");
    let message_id = reply_broadcast["id"].as_i64().unwrap();

    // A mention-filtered round with no host reply: `prompt_message` returns
    // 204, not the 200-with-body a `char` reply would get.
    let (status, body) = prompt_handle
        .await
        .expect("the /api/prompt request should not panic");
    assert_eq!(status, 204, "POST /api/prompt body: {body}");

    // The reply landed in the host's own database under the same id the
    // broadcast carried.
    let (status, body) = http_request(addr, "GET", "/api/message?limit=10&start_index=0", None);
    assert_eq!(status, 200, "GET /api/message failed: {body}");
    let page: Value = serde_json::from_str(&body).expect("message page should be valid JSON");
    let messages = page["messages"]
        .as_array()
        .expect("message page should carry a messages array");
    let persisted = messages
        .iter()
        .find(|m| m["id"].as_i64() == Some(message_id))
        .unwrap_or_else(|| panic!("no persisted message with id {message_id} in {messages:?}"));
    assert_eq!(persisted["speaker_id"], "bot1");
    assert_eq!(persisted["content"], "hello from bot1");
}

async fn recv_json(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> Value {
    let deadline = tokio::time::sleep(Duration::from_secs(10));
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            frame = ws.next() => {
                match frame.expect("stream ended").expect("ws error") {
                    WsMessage::Text(text) => return serde_json::from_str(&text).expect("frame should be valid JSON"),
                    WsMessage::Ping(_) | WsMessage::Pong(_) => continue,
                    other => panic!("expected a text frame, got {:?}", other),
                }
            }
            _ = &mut deadline => panic!("timed out waiting for a WS frame"),
        }
    }
}

async fn send_json(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    value: &Value,
) {
    ws.send(WsMessage::Text(value.to_string().into()))
        .await
        .expect("failed to send a WS frame");
}
