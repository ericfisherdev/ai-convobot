//! #136: a real host `HttpServer` and a real joiner `joiner::run`, talking
//! over an actual loopback TCP socket, drive one round through #131's
//! `round::run_round` with #154's real, socket-backed `SocketRemoteGenerator`
//! — the one seam #131's own tests (`main.rs`'s `stream_turn_tests`) only
//! ever exercise with a scripted fake.
//!
//! No model, no `Database::open()`, and no `paths::init` (see its own doc
//! comment on why a unit test must never call it): the host's own reply and
//! the joiner's own reply are both cheap stub closures, and the round runs
//! against a `RecordingStore` rather than SQLite. This is what makes an
//! in-crate `#[actix_web::test]` the right level for this: `main.rs`'s SSE
//! handler hardwires `SqliteTurnStore` and `llm::prompt_streaming`, and the
//! process-wide data dir (`paths::init`) cannot be pointed at a temp
//! directory from inside a unit-test process. `backend/tests/multiplayer_modes.rs`
//! is the complementary process-level test: two real binaries, covering the
//! HTTP layer and `main()`'s own wiring, with no model and no round.
//!
//! `run_round` runs on the calling (blocking) thread by design (see its own
//! doc comment), and `SocketRemoteGenerator::generate` blocks on a
//! synchronous channel while it waits for the joiner's reply — so every
//! round in this file runs inside `tokio::task::spawn_blocking`, exactly as
//! both prompting handlers in `main.rs` run it inside `web::block`. Calling
//! it directly on the test's own single-threaded runtime would starve the
//! joiner's own tasks (spawned on that same runtime) of the chance to ever
//! answer.

use std::io;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use actix_web::{web, App, HttpServer};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::chat_turn::{PendingTurn, RecordingStore, TurnStore};
use crate::compaction::context::{QuoteLine, QuoteSpeaker};
use crate::database::{CompanionAttitude, Message};
use crate::llm::PromptSpeakers;
use crate::multiplayer::handshake;
use crate::multiplayer::host::{multiplayer_ws, HostConfigSource, HostSettings};
use crate::multiplayer::join_throttle::JoinThrottle;
use crate::multiplayer::joiner::{
    GenerateRequestHandler, JoinerHandle, JoinerIdentity, JoinerShared,
};
use crate::multiplayer::protocol::{ClientFrame, ContinuityPayload, ServerFrame, PROTOCOL_VERSION};
use crate::multiplayer::remote_bots::RemoteBots;
use crate::multiplayer::remote_generation::{LocalModelGeneration, RemoteGenerator};
use crate::multiplayer::remote_generator::SocketRemoteGenerator;
use crate::multiplayer::round::{plan_round, run_round, RoundOutcome, RoundPlan, RoundSink};
use crate::multiplayer::routing::RoutingPolicy;
use crate::participants::ParticipantId;
use crate::participants::ParticipantRegistry;
use crate::turn_slot::TurnSlot;

/// The fixed companion/user ids every round in this file scores against —
/// the same "Default user ID" every prompting handler in `main.rs` uses.
const COMPANION_ID: i32 = 1;
const USER_ID: i32 = 1;

/// A [`HostConfigSource`] that reports a fixed password, always in `Host`
/// mode — this file never exercises a mode change, only the join handshake
/// and a round.
struct StubHostConfig(String);

impl HostConfigSource for StubHostConfig {
    fn host_password(&self) -> Result<Option<String>, String> {
        Ok(Some(self.0.clone()))
    }
}

/// A running host: the `HttpServer` itself, bound to an ephemeral port, plus
/// the same `web::Data` clones its workers hold — so the round-driving side
/// of a test observes exactly the state the socket handler mutates, the same
/// pattern `host.rs`'s own `socket_tests::Harness` uses.
struct HostHandle {
    port: u16,
    server_handle: actix_web::dev::ServerHandle,
    registry: web::Data<RwLock<ParticipantRegistry>>,
    remote_bots: web::Data<RemoteBots>,
}

impl HostHandle {
    async fn start(password: &str) -> Self {
        let registry = web::Data::new(RwLock::new(ParticipantRegistry::solo("Alice", "Bob", None)));
        let remote_bots = web::Data::new(RemoteBots::new());
        let throttle = web::Data::new(JoinThrottle::new(5, Duration::from_secs(600)));
        let settings = web::Data::new(HostSettings::default());
        let host_config: web::Data<Arc<dyn HostConfigSource>> = web::Data::new(Arc::new(
            StubHostConfig(password.to_string()),
        )
            as Arc<dyn HostConfigSource>);

        let (r, rb, th, se, hc) = (
            registry.clone(),
            remote_bots.clone(),
            throttle.clone(),
            settings.clone(),
            host_config.clone(),
        );
        let server = HttpServer::new(move || {
            App::new()
                .app_data(r.clone())
                .app_data(rb.clone())
                .app_data(th.clone())
                .app_data(se.clone())
                .app_data(hc.clone())
                .service(multiplayer_ws)
                .service(crate::multiplayer_participants)
        })
        .workers(1)
        .bind(("127.0.0.1", 0))
        .expect("bind an ephemeral port");
        let port = server.addrs()[0].port();
        let server = server.run();
        let server_handle = server.handle();
        actix_web::rt::spawn(server);

        HostHandle {
            port,
            server_handle,
            registry,
            remote_bots,
        }
    }

    async fn stop(self) {
        self.server_handle.stop(true).await;
    }
}

/// Connects a joiner to `port` under `id`, generating every reply from
/// `generator`. Returns the shared status handle and the reconnect-loop
/// task (`joiner::run`); aborting the task is how a test drops the
/// connection, since dropping the handle alone leaves the socket open.
fn spawn_joiner(
    port: u16,
    id: &str,
    display_name: &str,
    password: &str,
    generator: RemoteGenerator,
) -> (JoinerHandle, tokio::task::JoinHandle<()>) {
    let identity = JoinerIdentity {
        id: ParticipantId::parse(id).expect("valid participant id"),
        display_name: display_name.to_string(),
        avatar: None,
        password: password.to_string(),
        host_address: format!("127.0.0.1:{port}"),
    };
    let handle: JoinerHandle = Arc::new(RwLock::new(JoinerShared::new(
        &identity,
        COMPANION_ID,
        None,
    )));
    let generation: Arc<dyn GenerateRequestHandler> = Arc::new(LocalModelGeneration::new(
        COMPANION_ID,
        identity.id.clone(),
        handle.clone(),
        generator,
    ));
    let extraction = crate::multiplayer::joiner_compaction::noop_job();
    let task = tokio::spawn(crate::multiplayer::joiner::run(
        handle.clone(),
        identity,
        generation,
        extraction,
    ));
    (handle, task)
}

/// A [`RemoteGenerator`] that ignores the transcript it is handed and emits
/// a fixed token sequence, for a joiner's own generation.
fn stub_generator(tokens: &'static [&'static str], reply: &'static str) -> RemoteGenerator {
    Arc::new(
        move |_transcript: &[Message],
              _speakers: &PromptSpeakers,
              on_token: &mut dyn FnMut(&str)| {
            for token in tokens {
                on_token(token);
            }
            Ok(reply.to_string())
        },
    )
}

/// Polls `check` every 50ms until it returns `true` or `deadline` elapses.
async fn wait_until(deadline: Duration, check: impl Fn() -> bool) -> bool {
    let start = tokio::time::Instant::now();
    loop {
        if check() {
            return true;
        }
        if start.elapsed() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_bot_connected(host: &HostHandle, id: &ParticipantId) {
    let connected = wait_until(Duration::from_secs(5), || {
        host.registry
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(id)
            .is_some()
            && host.remote_bots.connected_ids().contains(id)
    })
    .await;
    assert!(connected, "{id} never joined within 5s");
}

async fn wait_for_bot_disconnected(host: &HostHandle, id: &ParticipantId) {
    let disconnected = wait_until(Duration::from_secs(5), || {
        !host.remote_bots.connected_ids().contains(id)
    })
    .await;
    assert!(disconnected, "{id} still connected after 5s");
}

/// One event a [`RecordingSink`] observed, stripped of the parts a test
/// does not need to assert on (message ids, `created_at`).
#[derive(Debug, Clone, PartialEq, Eq)]
enum SinkEvent {
    ReplyStarted(ParticipantId),
    Token(ParticipantId, String),
    ReplyComplete(ParticipantId, String),
    SpeakerSkipped(ParticipantId),
    RoundComplete,
}

#[derive(Default)]
struct RecordingSink {
    events: Vec<SinkEvent>,
}

impl RoundSink for RecordingSink {
    fn reply_started(&mut self, speaker: &ParticipantId) {
        self.events.push(SinkEvent::ReplyStarted(speaker.clone()));
    }

    fn token(&mut self, speaker: &ParticipantId, text: &str) {
        self.events
            .push(SinkEvent::Token(speaker.clone(), text.to_string()));
    }

    fn reply_complete(&mut self, reply: &crate::chat_turn::PersistedReply) {
        self.events.push(SinkEvent::ReplyComplete(
            reply.speaker_id.clone(),
            reply.text.clone(),
        ));
    }

    fn speaker_skipped(
        &mut self,
        speaker: &ParticipantId,
        _notice: &crate::chat_turn::PersistedReply,
    ) {
        self.events.push(SinkEvent::SpeakerSkipped(speaker.clone()));
    }

    fn round_complete(&mut self, _attitude: Option<&(CompanionAttitude, CompanionAttitude)>) {
        self.events.push(SinkEvent::RoundComplete);
    }
}

/// Runs one round on a blocking thread (see the module doc for why), over
/// the real socket-backed [`SocketRemoteGenerator`] and a fresh
/// [`RecordingStore`], broadcasting through `host`'s real `RemoteBots` so a
/// connected joiner's transcript mirror actually receives every frame.
async fn run_one_round(
    slot: &'static TurnSlot,
    host: &HostHandle,
    prompt: &str,
    plan: RoundPlan,
    host_tokens: &'static [&'static str],
    host_reply: &'static str,
    timeout: Duration,
) -> (RoundOutcome, RecordingStore, RecordingSink) {
    let guard = slot.try_claim().expect("round slot should be free");
    let registry_snapshot = host
        .registry
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    let policy = RoutingPolicy {
        max_followup_depth: 1,
    };
    let remotes = SocketRemoteGenerator::new(host.remote_bots.clone());
    let remote_bots_for_broadcast = host.remote_bots.clone();
    let prompt_owned = prompt.to_string();

    tokio::task::spawn_blocking(move || {
        let store = RecordingStore::new(None);
        let pending = PendingTurn::begin(
            &guard,
            &store,
            COMPANION_ID,
            USER_ID,
            prompt_owned,
            registry_snapshot.clone(),
        )
        .expect("insert the user turn");
        let broadcast = move |frame: ServerFrame| remote_bots_for_broadcast.broadcast(frame, None);
        let mut host_gen =
            move |_generation_prompt: &str, on_token: &mut dyn FnMut(&str)| -> io::Result<String> {
                for token in host_tokens {
                    on_token(token);
                }
                Ok(host_reply.to_string())
            };
        let mut sink = RecordingSink::default();
        let outcome = run_round(
            guard,
            pending,
            plan,
            &store,
            &registry_snapshot,
            &policy,
            &mut host_gen,
            &remotes,
            &broadcast,
            timeout,
            &mut sink,
        )
        .expect("the round should complete");
        (outcome, store, sink)
    })
    .await
    .expect("the blocking round task should not panic")
}

#[actix_web::test]
async fn a_full_round_runs_over_a_real_socket_between_a_host_and_a_joiner() {
    let host = HostHandle::start("test-secret").await;
    let bot1 = ParticipantId::parse("bot1").expect("valid id");

    let (joiner_handle, _joiner_task) = spawn_joiner(
        host.port,
        "bot1",
        "Ada",
        "test-secret",
        stub_generator(&["Hi ", "there"], "Hi there"),
    );
    wait_for_bot_connected(&host, &bot1).await;

    // #131/#132: the plan the user's un-mentioned message builds is the
    // host companion first, then every joined bot in join order.
    let registry_snapshot = host
        .registry
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    let policy = RoutingPolicy {
        max_followup_depth: 1,
    };
    let plan = plan_round("hello everyone", &registry_snapshot, &policy);
    assert_eq!(
        plan.pending_ids(),
        vec![ParticipantId::CHAR, bot1.clone()],
        "plan_round should run the host companion, then bot1"
    );

    static SLOT: TurnSlot = TurnSlot::new();
    let (outcome, store, sink) = run_one_round(
        &SLOT,
        &host,
        "hello everyone",
        plan,
        &["Hello ", "from host"],
        "Hello from host",
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(
        sink.events,
        vec![
            SinkEvent::ReplyStarted(ParticipantId::CHAR),
            SinkEvent::Token(ParticipantId::CHAR, "Hello ".to_string()),
            SinkEvent::Token(ParticipantId::CHAR, "from host".to_string()),
            SinkEvent::ReplyComplete(ParticipantId::CHAR, "Hello from host".to_string()),
            SinkEvent::ReplyStarted(bot1.clone()),
            SinkEvent::Token(bot1.clone(), "Hi ".to_string()),
            SinkEvent::Token(bot1.clone(), "there".to_string()),
            SinkEvent::ReplyComplete(bot1.clone(), "Hi there".to_string()),
            SinkEvent::RoundComplete,
        ]
    );
    assert_eq!(
        outcome.host_reply.map(|reply| reply.text),
        Some("Hello from host".to_string())
    );

    let rows: Vec<(String, String)> = store
        .transcript_tail(10)
        .expect("recording store never fails")
        .into_iter()
        .map(|m| (m.speaker_id, m.content))
        .collect();
    assert_eq!(
        rows,
        vec![
            ("user".to_string(), "hello everyone".to_string()),
            ("char".to_string(), "Hello from host".to_string()),
            ("bot1".to_string(), "Hi there".to_string()),
        ]
    );

    // #154: the joiner's own transcript mirror is fed by the `Message`
    // broadcast `run_round` sends for char's reply before bot1's own
    // `GenerateRequest` is even issued (see `round.rs::run_round`'s
    // per-speaker loop), not only by what bot1 generated itself.
    let mirror_has_char_reply = wait_until(Duration::from_secs(2), || {
        joiner_handle
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .transcript
            .snapshot()
            .iter()
            .any(|m| m.speaker_id == "char" && m.content == "Hello from host")
    })
    .await;
    assert!(
        mirror_has_char_reply,
        "the joiner's transcript mirror never saw char's broadcast reply"
    );

    assert!(
        SLOT.try_claim().is_some(),
        "the turn slot should be free again once run_round has returned"
    );

    host.stop().await;
}

#[actix_web::test]
async fn a_disconnected_bot_is_skipped_and_the_round_still_completes() {
    let host = HostHandle::start("test-secret").await;
    let bot1 = ParticipantId::parse("bot1").expect("valid id");

    let (_joiner_handle, joiner_task) = spawn_joiner(
        host.port,
        "bot1",
        "Ada",
        "test-secret",
        stub_generator(&["never sent"], "never sent"),
    );
    wait_for_bot_connected(&host, &bot1).await;

    // Real disconnect: aborting the reconnect-loop task drops its socket,
    // which the host notices and unregisters — exercising the real
    // `SocketRemoteGenerator::generate` `Offline` path (`RemoteBots::send`
    // failing), not a scripted fake like #131's own tests use.
    joiner_task.abort();
    wait_for_bot_disconnected(&host, &bot1).await;

    // Scheduled directly (`RoundPlan::from_speakers`, like #131's own
    // tests) rather than through `plan_round`: `host.rs::unregister` also
    // drops bot1 from the shared registry on disconnect, so a plan built
    // from the live registry would no longer include it at all — this test
    // is about a round that *tries* to reach a bot that just went offline,
    // not about routing.
    let plan = RoundPlan::from_speakers([ParticipantId::CHAR, bot1.clone()]);

    static SLOT: TurnSlot = TurnSlot::new();
    let (_outcome, store, sink) = run_one_round(
        &SLOT,
        &host,
        "hello again",
        plan,
        &["Hi "],
        "Hi",
        Duration::from_secs(1),
    )
    .await;

    assert!(
        sink.events
            .contains(&SinkEvent::SpeakerSkipped(bot1.clone())),
        "expected bot1 to be reported skipped: {:?}",
        sink.events
    );
    assert_eq!(
        sink.events.last(),
        Some(&SinkEvent::RoundComplete),
        "the round should still finish after a skipped speaker"
    );

    let notice = store
        .transcript_tail(10)
        .expect("recording store never fails")
        .into_iter()
        .find(|m| m.speaker_id == "system");
    assert_eq!(
        notice.map(|m| m.content),
        Some("bot1 did not respond".to_string())
    );

    host.stop().await;
}

/// A raw joiner socket, deliberately not `joiner::run`: #182's
/// `ContinuityPayload` has no `GenerateRequestHandler`-level counterpart
/// yet (#186's job), so a test that needs to see the raw
/// `ServerFrame::GenerateRequest` — payload and all — has to read frames
/// straight off the socket instead of going through `joiner::run`'s own
/// per-round dispatch.
type RawJoinerSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Connects to `port` and completes the join handshake for `id`, the same
/// wire sequence `joiner::run`'s own `handshake` function drives.
async fn connect_raw_joiner(port: u16, id: &str, password: &str) -> RawJoinerSocket {
    let (mut ws, _) =
        tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/api/multiplayer/ws"))
            .await
            .expect("failed to connect to the multiplayer socket");

    let nonce = match recv_server_frame(&mut ws).await {
        ServerFrame::Challenge { nonce, .. } => nonce,
        other => panic!("expected a Challenge frame, got {:?}", other),
    };
    let nonce_bytes: [u8; handshake::NONCE_BYTES] = handshake::decode(&nonce)
        .and_then(|bytes| bytes.try_into().ok())
        .expect("Challenge nonce should decode to NONCE_BYTES bytes");
    let participant_id = ParticipantId::parse(id).expect("valid participant id");
    let proof = handshake::join_proof(password, &nonce_bytes, &participant_id);

    send_client_frame(
        &mut ws,
        &ClientFrame::Join {
            protocol_version: PROTOCOL_VERSION,
            id: participant_id,
            display_name: "Raw Joiner".to_string(),
            avatar: None,
            proof: handshake::encode(&proof),
        },
    )
    .await;

    match recv_server_frame(&mut ws).await {
        ServerFrame::Joined { .. } => {}
        other => panic!("expected a Joined frame, got {:?}", other),
    }

    ws
}

async fn send_client_frame(ws: &mut RawJoinerSocket, frame: &ClientFrame) {
    let text = serde_json::to_string(frame).expect("ClientFrame always serializes");
    ws.send(WsMessage::text(text))
        .await
        .expect("failed to send a frame to the host");
}

async fn recv_server_frame(ws: &mut RawJoinerSocket) -> ServerFrame {
    loop {
        match ws.next().await {
            Some(Ok(WsMessage::Text(text))) => {
                return serde_json::from_str(&text)
                    .unwrap_or_else(|e| panic!("unparsable frame {text:?}: {e}"));
            }
            Some(Ok(WsMessage::Ping(_))) | Some(Ok(WsMessage::Pong(_))) => continue,
            other => panic!(
                "unexpected websocket event while waiting for a frame: {:?}",
                other
            ),
        }
    }
}

/// Reads frames until a `GenerateRequest` arrives, discarding every
/// `Message` broadcast along the way (the user's turn, and — on a second
/// round over the same socket — the previous round's own reply broadcast
/// back to its sender). Returns the request's `round_id`, transcript and
/// continuity payload so the caller can both assert on them and answer.
async fn wait_for_generate_request(
    ws: &mut RawJoinerSocket,
) -> (u64, Vec<Message>, Option<ContinuityPayload>) {
    loop {
        match recv_server_frame(ws).await {
            ServerFrame::Message(_) => continue,
            ServerFrame::GenerateRequest {
                round_id,
                transcript,
                continuity,
            } => return (round_id, transcript, continuity),
            other => panic!(
                "unexpected frame while waiting for a GenerateRequest: {:?}",
                other
            ),
        }
    }
}

/// #182: a committed checkpoint ships its `ContinuityPayload` on every
/// `GenerateRequest` and trims the transcript to messages after
/// `compacted_through`; a companion that has never been compacted (or was
/// reset) carries neither, unchanged from before #182.
#[actix_web::test]
async fn a_committed_checkpoint_ships_continuity_and_a_trimmed_transcript_to_the_joiner() {
    let host = HostHandle::start("test-secret").await;
    let bot1 = ParticipantId::parse("bot1").expect("valid id");

    let mut raw = connect_raw_joiner(host.port, "bot1", "test-secret").await;
    wait_for_bot_connected(&host, &bot1).await;

    static SLOT: TurnSlot = TurnSlot::new();
    let policy = RoutingPolicy {
        max_followup_depth: 1,
    };

    // --- Round 1: a committed checkpoint through id 2 ---
    let guard = SLOT.try_claim().expect("slot should be free");
    let registry_snapshot = host
        .registry
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    let remotes = SocketRemoteGenerator::new(host.remote_bots.clone());
    let remote_bots_for_broadcast = host.remote_bots.clone();
    let payload = ContinuityPayload {
        compacted_through: 2,
        rules: vec![QuoteLine {
            speaker: QuoteSpeaker::User,
            text: "never call me Bob".to_string(),
        }],
        ..Default::default()
    };
    let payload_for_round = payload.clone();
    let round_registry = registry_snapshot.clone();
    let round_policy = policy;

    let round_task = tokio::task::spawn_blocking(move || {
        let store = RecordingStore::new(None);
        // Seeds ids 1-4; `compacted_through: 2` should drop the first two.
        store.insert_reply(&ParticipantId::USER, "one").unwrap();
        store.insert_reply(&ParticipantId::CHAR, "two").unwrap();
        store.insert_reply(&ParticipantId::USER, "three").unwrap();
        store.insert_reply(&ParticipantId::CHAR, "four").unwrap();
        store.set_continuity(Some(payload_for_round));

        let pending = PendingTurn::begin(
            &guard,
            &store,
            COMPANION_ID,
            USER_ID,
            "@bot1 hi".to_string(),
            round_registry.clone(),
        )
        .expect("insert the user turn");
        let plan = RoundPlan::from_speakers([ParticipantId::parse("bot1").unwrap()]);
        let broadcast = move |frame: ServerFrame| remote_bots_for_broadcast.broadcast(frame, None);
        let mut sink = RecordingSink::default();
        run_round(
            guard,
            pending,
            plan,
            &store,
            &round_registry,
            &round_policy,
            &mut |_prompt: &str, _on_token: &mut dyn FnMut(&str)| -> io::Result<String> {
                panic!("char should never be asked to speak in this test")
            },
            &remotes,
            &broadcast,
            Duration::from_secs(5),
            &mut sink,
        )
        .expect("the round should complete")
    });

    let (round_id, transcript, continuity) = wait_for_generate_request(&mut raw).await;
    send_client_frame(
        &mut raw,
        &ClientFrame::ReplyComplete {
            round_id,
            text: "hi from bot1".to_string(),
        },
    )
    .await;
    round_task.await.expect("the round task should not panic");

    assert_eq!(
        continuity,
        Some(payload),
        "the checkpoint's continuity payload should ride along unchanged"
    );
    assert!(
        transcript.iter().all(|m| m.id > 2),
        "no message at or before compacted_through should reach the joiner: {:?}",
        transcript
    );

    // --- Round 2: no compaction ever happened, so no payload and the full tail ---
    let guard = SLOT.try_claim().expect("slot should be free again");
    let remotes = SocketRemoteGenerator::new(host.remote_bots.clone());
    let remote_bots_for_broadcast = host.remote_bots.clone();
    let round_registry = registry_snapshot.clone();
    let round_policy = policy;

    let round_task = tokio::task::spawn_blocking(move || {
        // `RecordingStore::new` defaults `continuity` to `None`.
        let store = RecordingStore::new(None);
        store.insert_reply(&ParticipantId::USER, "one").unwrap();
        store.insert_reply(&ParticipantId::CHAR, "two").unwrap();

        let pending = PendingTurn::begin(
            &guard,
            &store,
            COMPANION_ID,
            USER_ID,
            "@bot1 hi again".to_string(),
            round_registry.clone(),
        )
        .expect("insert the user turn");
        let plan = RoundPlan::from_speakers([ParticipantId::parse("bot1").unwrap()]);
        let broadcast = move |frame: ServerFrame| remote_bots_for_broadcast.broadcast(frame, None);
        let mut sink = RecordingSink::default();
        run_round(
            guard,
            pending,
            plan,
            &store,
            &round_registry,
            &round_policy,
            &mut |_prompt: &str, _on_token: &mut dyn FnMut(&str)| -> io::Result<String> {
                panic!("char should never be asked to speak in this test")
            },
            &remotes,
            &broadcast,
            Duration::from_secs(5),
            &mut sink,
        )
        .expect("the round should complete")
    });

    let (round_id, transcript, continuity) = wait_for_generate_request(&mut raw).await;
    send_client_frame(
        &mut raw,
        &ClientFrame::ReplyComplete {
            round_id,
            text: "hi again from bot1".to_string(),
        },
    )
    .await;
    round_task.await.expect("the round task should not panic");

    assert_eq!(
        continuity, None,
        "a companion that has never been compacted must carry no payload"
    );
    assert_eq!(
        transcript
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>(),
        vec!["one", "two", "@bot1 hi again"],
        "with no compaction, the tail is the full, untrimmed transcript"
    );

    host.stop().await;
}
