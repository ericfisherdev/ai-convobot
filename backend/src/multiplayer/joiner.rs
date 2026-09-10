//! The joiner's connection state machine: connect out to a host, complete
//! the HMAC handshake, mirror its transcript, serve heartbeats and
//! `GenerateRequest`s, and reconnect (with backoff) when the connection
//! drops for any reason other than an explicit `Rejected`.
//!
//! `host.rs` is the host's counterpart; this module is deliberately its own
//! file (not folded into `host.rs`) because the two run in different
//! processes and share nothing but `protocol.rs` and `handshake.rs`.

use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use futures_util::{SinkExt, Stream, StreamExt};
use serde::Serialize;
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{connect_async, tungstenite};

use crate::database::{CompanionView, ConfigView, Message};
use crate::multiplayer::backoff::ReconnectBackoff;
use crate::multiplayer::handshake;
use crate::multiplayer::joiner_compaction::{maybe_queue_extraction, JoinerExtractionJob};
use crate::multiplayer::protocol::{
    AvatarUpload, ClientFrame, ContinuityPayload, ParticipantSummary, RejectReason, ServerFrame,
    PROTOCOL_VERSION,
};
use crate::multiplayer::remote_transcript::RemoteTranscript;
use crate::participants::ParticipantId;
use crate::paths;

/// How long a single connect attempt (TCP + WebSocket upgrade) is given
/// before it counts as a failure.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// How long, from the moment the socket is open, the handshake (`Challenge`
/// then `Joined`/`Rejected`) has to finish.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// How often the writer task flushes the socket when it has nothing of its
/// own to send. `tokio-tungstenite` queues an automatic `Pong` for an
/// incoming `Ping` inside the shared connection state, but only a
/// `Sink::flush`/`send` on the write half actually puts it on the wire; the
/// read half polled by `serve` cannot do that itself once the stream is
/// split. Well under `HostSettings::default()`'s `heartbeat_interval` (15s)
/// so a queued `Pong` is never at risk of missing the host's
/// `missed_pongs_before_drop` window.
const IDLE_FLUSH_INTERVAL: Duration = Duration::from_secs(5);

/// The joiner's connection state, as reported by `GET /api/multiplayer/status`.
///
/// Tagged on the wire as `"state"` so the status endpoint can flatten it
/// directly alongside the fields `JoinerShared` adds (`mode`, `attempts`,
/// `host_address`, `participant_id`, `participants`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JoinerState {
    /// Not currently connected. `last_error` is the reason the previous
    /// attempt ended, or `None` on the very first attempt.
    Disconnected { last_error: Option<String> },
    /// A connection attempt (including the handshake) is in progress.
    Connecting,
    /// Admitted by the host and serving its transcript/heartbeats.
    Connected,
    /// The host refused the handshake. Terminal: [`run`] does not retry.
    Rejected { reason: String },
}

/// State shared between [`run`] and the HTTP handlers that read it
/// (`GET /api/message`, `GET /api/multiplayer/status`, the joiner-mode 409
/// guards). Locks are only ever held for a field read or a short write,
/// never across an `.await`, so a plain [`std::sync::RwLock`] is correct
/// here (the same convention `database.rs`'s `MESSAGE_CACHE` uses), not
/// `tokio::sync::RwLock`.
pub struct JoinerShared {
    pub state: JoinerState,
    pub transcript: RemoteTranscript,
    pub participants: Vec<ParticipantSummary>,
    pub attempts: u32,
    /// Copied from [`JoinerIdentity`] at construction so the status endpoint
    /// does not need a second handle just to report them; both are fixed
    /// for the process's lifetime (changing them requires a restart, see
    /// `main.rs`'s startup wiring).
    pub host_address: String,
    pub participant_id: ParticipantId,
    /// This joiner's own companion id (`Database::get_companion_id()` at
    /// startup, #186): every local `CompactionStore` read/write
    /// `multiplayer::joiner_compaction` does is scoped to this id, the same
    /// way every other `Database` call in solo/host mode is scoped to the
    /// single local companion row.
    pub companion_id: i32,
    /// This joiner's own compacted-through cursor, in the host's message id
    /// space (mirrors `compaction::store::CompactionStore::compacted_through`
    /// for `companion_id`, cached here so the per-frame extraction decision
    /// in `joiner_compaction::maybe_queue_extraction` never blocks on a
    /// SQLite read). Seeded from the store at startup, updated after every
    /// successful auto-commit (`joiner_compaction::run_joiner_extraction`).
    pub local_compacted_through: Option<i32>,
    /// The highest `ContinuityPayload::compacted_through`
    /// `maybe_queue_extraction` saw while [`crate::turn_slot::ACTIVE_TURN`]
    /// was claimed by an in-flight reply; retried on the next
    /// `GenerateRequest`. `None` when nothing is waiting.
    pub pending_extraction: Option<i32>,
    /// The continuity payload from the most recent `GenerateRequest`,
    /// mirrored here so `GET /api/debug/prompt` can reproduce the same
    /// prompt a live reply would have rendered (#186).
    pub last_continuity: Option<ContinuityPayload>,
}

impl JoinerShared {
    pub fn new(
        identity: &JoinerIdentity,
        companion_id: i32,
        local_compacted_through: Option<i32>,
    ) -> Self {
        JoinerShared {
            state: JoinerState::Disconnected { last_error: None },
            transcript: RemoteTranscript::new(),
            participants: Vec::new(),
            attempts: 0,
            host_address: identity.host_address.clone(),
            participant_id: identity.id.clone(),
            companion_id,
            local_compacted_through,
            pending_extraction: None,
            last_continuity: None,
        }
    }
}

/// The shared handle every joiner-mode HTTP handler and [`run`] hold a
/// clone of.
pub type JoinerHandle = Arc<RwLock<JoinerShared>>;

/// This instance's own identity, built once at startup from the persisted
/// config and companion data. Immutable for the process's lifetime: a
/// config change to the multiplayer fields takes effect on the next
/// restart, not live.
#[derive(Clone)]
pub struct JoinerIdentity {
    pub id: ParticipantId,
    pub display_name: String,
    pub avatar: Option<AvatarUpload>,
    pub password: String,
    pub host_address: String,
}

impl JoinerIdentity {
    /// # Errors
    /// A human-readable message when `config.multiplayer_participant_id` is
    /// not a valid [`ParticipantId`] or `config.multiplayer_host_address` is
    /// empty. `main()` treats either as a startup error, the same as an
    /// `init_storage` failure.
    pub fn from_config(config: &ConfigView, companion: &CompanionView) -> Result<Self, String> {
        let id = ParticipantId::parse(&config.multiplayer_participant_id)
            .map_err(|e| format!("invalid multiplayer participant id: {e}"))?;
        let host_address = config.multiplayer_host_address.trim();
        if host_address.is_empty() {
            return Err("joiner mode requires a host address".to_string());
        }
        Ok(JoinerIdentity {
            id,
            display_name: companion.name.clone(),
            avatar: read_avatar_upload(),
            password: config.multiplayer_password.clone(),
            host_address: host_address.to_string(),
        })
    }
}

/// Reads and base64-encodes the on-disk companion avatar
/// (`paths::avatar_path()`, always stored as PNG regardless of the format
/// it was uploaded in — see `write_companion_avatar` in `main.rs`). `None`
/// when no avatar has ever been set; that is not an error, `Join.avatar` is
/// optional.
fn read_avatar_upload() -> Option<AvatarUpload> {
    let bytes = std::fs::read(paths::avatar_path()).ok()?;
    Some(AvatarUpload {
        mime: "image/png".to_string(),
        data_base64: handshake::encode(&bytes),
    })
}

/// The seam a joiner's reply is generated through. The production
/// implementation, [`LocalModelGeneration`] (`multiplayer::remote_generation`,
/// #153), runs the joiner's own model on the transcript the host supplied.
///
/// Never `.await`ed from the read loop in [`connect_and_serve`], so a
/// heartbeat is still answered while generation runs; `tx` is how an
/// implementation replies, whether synchronously or (as
/// [`LocalModelGeneration`] does) from a spawned thread.
///
/// [`LocalModelGeneration`]: crate::multiplayer::remote_generation::LocalModelGeneration
pub trait GenerateRequestHandler: Send + Sync {
    fn handle(&self, round_id: u64, transcript: Vec<Message>, tx: UnboundedSender<ClientFrame>);
}

/// Everything that can end a connection attempt before or during
/// [`connect_and_serve`].
#[derive(Debug)]
enum ConnectError {
    Timeout,
    Connect(tungstenite::Error),
    /// A frame arrived that was not what the handshake expected at that
    /// point (malformed JSON, or a frame variant out of sequence).
    Protocol(String),
    /// The socket closed, or its read half ended, before or during the
    /// handshake or the serve loop.
    ConnectionClosed,
    /// The host refused the `Join`. Terminal: [`run`] stops retrying.
    Rejected(RejectReason),
    /// A WebSocket protocol error surfaced while serving.
    Stream(String),
}

impl fmt::Display for ConnectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConnectError::Timeout => write!(f, "timed out"),
            ConnectError::Connect(e) => write!(f, "could not connect: {e}"),
            ConnectError::Protocol(reason) => write!(f, "protocol error: {reason}"),
            ConnectError::ConnectionClosed => write!(f, "connection closed"),
            ConnectError::Rejected(reason) => write!(f, "{}", describe_rejection(reason)),
            ConnectError::Stream(e) => write!(f, "connection error: {e}"),
        }
    }
}

/// A human-readable rendering of [`RejectReason`], surfaced through
/// `GET /api/multiplayer/status`'s `reason` field.
fn describe_rejection(reason: &RejectReason) -> String {
    match reason {
        RejectReason::UnsupportedProtocol => {
            "the host rejected this connection: unsupported protocol version".to_string()
        }
        RejectReason::BadProof => "the host rejected this connection: wrong password".to_string(),
        RejectReason::NoHostPassword => {
            "the host rejected this connection: the host has no password set".to_string()
        }
        RejectReason::DuplicateId => {
            "the host rejected this connection: a participant with this id is already connected"
                .to_string()
        }
        RejectReason::ReservedId => {
            "the host rejected this connection: this participant id is reserved".to_string()
        }
        RejectReason::InvalidAvatar(reason) => {
            format!("the host rejected this connection: invalid avatar ({reason})")
        }
        RejectReason::JoinTimeout => {
            "the host rejected this connection: join timed out".to_string()
        }
    }
}

/// The outer reconnect loop. Runs until the host issues an explicit
/// `Rejected` (a wrong password or a duplicate id never fixes itself by
/// retrying, and would only feed `host.rs`'s per-address join throttle),
/// or forever otherwise.
pub async fn run(
    handle: JoinerHandle,
    identity: JoinerIdentity,
    generation: Arc<dyn GenerateRequestHandler>,
    extraction: JoinerExtractionJob,
) {
    let mut backoff = ReconnectBackoff::new();
    loop {
        {
            let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
            shared.state = JoinerState::Connecting;
            shared.attempts += 1;
        }

        match connect_and_serve(&handle, &identity, &generation, &extraction, &mut backoff).await {
            Err(ConnectError::Rejected(reason)) => {
                let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
                shared.state = JoinerState::Rejected {
                    reason: describe_rejection(&reason),
                };
                return;
            }
            Err(e) => {
                let last_error = Some(e.to_string());
                let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
                shared.state = JoinerState::Disconnected { last_error };
            }
            Ok(()) => {
                let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
                shared.state = JoinerState::Disconnected { last_error: None };
            }
        }

        tokio::time::sleep(backoff.delay()).await;
    }
}

/// One full connection attempt: connect, handshake, then serve until the
/// connection ends. `backoff` is reset the moment the handshake succeeds
/// (`Connected`), not merely on a successful TCP connect, so a `Rejected`
/// or a mid-handshake failure still backs off.
async fn connect_and_serve(
    handle: &JoinerHandle,
    identity: &JoinerIdentity,
    generation: &Arc<dyn GenerateRequestHandler>,
    extraction: &JoinerExtractionJob,
    backoff: &mut ReconnectBackoff,
) -> Result<(), ConnectError> {
    let url = format!("ws://{}/api/multiplayer/ws", identity.host_address);
    let ws_stream = tokio::time::timeout(CONNECT_TIMEOUT, connect_async(&url))
        .await
        .map_err(|_| ConnectError::Timeout)?
        .map_err(ConnectError::Connect)?
        .0;

    let (mut write, mut read) = ws_stream.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<ClientFrame>();

    // Drains outbound frames into the socket. Unbounded on purpose: a
    // `GenerateRequestHandler` (#153) must be able to push `Token` frames
    // without ever awaiting a full channel. Also flushes on
    // `IDLE_FLUSH_INTERVAL` even with nothing queued, so an automatic `Pong`
    // queued by a `Ping` the read half saw does not sit unsent for the rest
    // of an idle connection (see `IDLE_FLUSH_INTERVAL`'s doc comment).
    let writer_task = tokio::spawn(async move {
        let mut idle_flush = tokio::time::interval(IDLE_FLUSH_INTERVAL);
        idle_flush.tick().await; // the first tick fires immediately; consume it
        loop {
            tokio::select! {
                frame = rx.recv() => {
                    let Some(frame) = frame else { break; };
                    let text = match serde_json::to_string(&frame) {
                        Ok(text) => text,
                        Err(e) => {
                            eprintln!("joiner: failed to serialise {:?}: {}", frame, e);
                            continue;
                        }
                    };
                    if write.send(WsMessage::text(text)).await.is_err() {
                        break;
                    }
                }
                _ = idle_flush.tick() => {
                    if write.flush().await.is_err() {
                        break;
                    }
                }
            }
        }
    });

    let handshake_result = handshake(&mut read, &tx, identity).await;
    let joined = match handshake_result {
        Ok(joined) => joined,
        Err(e) => {
            drop(tx);
            let _ = writer_task.await;
            return Err(e);
        }
    };

    {
        let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
        shared.state = JoinerState::Connected;
        shared.transcript.replace(joined.transcript);
        shared.participants = joined.participants;
    }
    backoff.reset();

    let result = serve(handle, &mut read, generation, extraction, &tx).await;

    drop(tx);
    let _ = writer_task.await;
    result
}

/// What a successful [`handshake`] produced.
struct Joined {
    participants: Vec<ParticipantSummary>,
    transcript: Vec<Message>,
}

/// Waits for `Challenge`, sends `Join` with the computed proof, then waits
/// for `Joined` or `Rejected`. `read` is the same `SplitStream` the
/// production socket uses; the fake-host tests in this module's `#[cfg(test)]`
/// exercise this exact code path against a real (non-actix) WebSocket
/// server, so there is no separate mock-friendly abstraction to keep in
/// sync with the wire format.
async fn handshake(
    read: &mut (impl Stream<Item = Result<WsMessage, tungstenite::Error>> + Unpin),
    tx: &UnboundedSender<ClientFrame>,
    identity: &JoinerIdentity,
) -> Result<Joined, ConnectError> {
    let deadline = tokio::time::Instant::now() + HANDSHAKE_TIMEOUT;

    let nonce = loop {
        match next_frame(read, deadline).await? {
            Some(ServerFrame::Challenge { nonce, .. }) => break nonce,
            Some(_) => {
                return Err(ConnectError::Protocol(
                    "expected Challenge first".to_string(),
                ))
            }
            None => continue,
        }
    };

    let nonce_bytes: [u8; handshake::NONCE_BYTES] = handshake::decode(&nonce)
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| ConnectError::Protocol("Challenge nonce was not valid".to_string()))?;

    let proof = handshake::join_proof(&identity.password, &nonce_bytes, &identity.id);
    let join = ClientFrame::Join {
        protocol_version: PROTOCOL_VERSION,
        id: identity.id.clone(),
        display_name: identity.display_name.clone(),
        avatar: identity.avatar.clone(),
        proof: handshake::encode(&proof),
    };
    tx.send(join).map_err(|_| ConnectError::ConnectionClosed)?;

    loop {
        match next_frame(read, deadline).await? {
            Some(ServerFrame::Joined {
                participants,
                transcript,
                ..
            }) => {
                return Ok(Joined {
                    participants,
                    transcript,
                })
            }
            Some(ServerFrame::Rejected { reason }) => return Err(ConnectError::Rejected(reason)),
            Some(_) => {
                return Err(ConnectError::Protocol(
                    "expected Joined or Rejected".to_string(),
                ))
            }
            None => continue,
        }
    }
}

/// Reads the next `ServerFrame` from `read`, up to `deadline`. `Ok(None)`
/// for a control frame the caller should just keep waiting past (`Ping`/
/// `Pong`); every other outcome (timeout, close, a stream error, or an
/// unparsable text frame) is an [`ConnectError`].
async fn next_frame(
    read: &mut (impl Stream<Item = Result<WsMessage, tungstenite::Error>> + Unpin),
    deadline: tokio::time::Instant,
) -> Result<Option<ServerFrame>, ConnectError> {
    let next = tokio::time::timeout_at(deadline, read.next())
        .await
        .map_err(|_| ConnectError::Timeout)?;
    match next {
        Some(Ok(WsMessage::Text(text))) => serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| ConnectError::Protocol(format!("unparsable frame: {e}"))),
        Some(Ok(WsMessage::Ping(_))) | Some(Ok(WsMessage::Pong(_))) => Ok(None),
        Some(Ok(WsMessage::Close(_))) | None => Err(ConnectError::ConnectionClosed),
        Some(Ok(_)) => Ok(None),
        Some(Err(e)) => Err(ConnectError::Stream(e.to_string())),
    }
}

/// The post-handshake loop: applies every `ServerFrame` to `handle`'s
/// mirror until the connection ends. Never awaits `generation.handle`, so a
/// heartbeat (native WebSocket ping/pong, answered by `tokio-tungstenite`
/// itself while this loop keeps `read.next()` polled) is still answered
/// while generation runs.
async fn serve(
    handle: &JoinerHandle,
    read: &mut (impl Stream<Item = Result<WsMessage, tungstenite::Error>> + Unpin),
    generation: &Arc<dyn GenerateRequestHandler>,
    extraction: &JoinerExtractionJob,
    tx: &UnboundedSender<ClientFrame>,
) -> Result<(), ConnectError> {
    loop {
        let next = read.next().await;
        match next {
            Some(Ok(WsMessage::Text(text))) => match serde_json::from_str::<ServerFrame>(&text) {
                Ok(ServerFrame::Message(message)) => {
                    let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
                    shared.transcript.push(message);
                }
                Ok(ServerFrame::MessageEdited { message }) => {
                    let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
                    shared.transcript.replace_message(message);
                }
                Ok(ServerFrame::MessageRemoved { id }) => {
                    let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
                    shared.transcript.remove(id);
                }
                Ok(ServerFrame::ParticipantJoined(summary)) => {
                    let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
                    if !shared.participants.iter().any(|p| p.id == summary.id) {
                        shared.participants.push(summary);
                    }
                }
                Ok(ServerFrame::ParticipantLeft { id }) => {
                    let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
                    shared.participants.retain(|p| p.id != id);
                }
                Ok(ServerFrame::GenerateRequest {
                    round_id,
                    transcript,
                    continuity,
                }) => {
                    // A `None` payload never clears an already-known one:
                    // once the host has ever compacted, `continuity` is
                    // `Some` on every request (`ContinuityPayload`'s own
                    // doc comment); a stray `None` would only ever be a
                    // protocol anomaly, not a real "compaction was undone".
                    if continuity.is_some() {
                        let mut shared = handle.write().unwrap_or_else(|p| p.into_inner());
                        shared.last_continuity = continuity.clone();
                    }
                    maybe_queue_extraction(handle, extraction, continuity.as_ref());
                    generation.handle(round_id, transcript, tx.clone());
                }
                Ok(other) => {
                    eprintln!("joiner: unexpected frame after join: {:?}", other);
                }
                Err(e) => {
                    eprintln!("joiner: dropping unparsable frame: {}", e);
                }
            },
            Some(Ok(WsMessage::Close(_))) | None => return Ok(()),
            // Ping/Pong/Binary: no application-level handling needed
            // (heartbeats are answered by `tokio-tungstenite` itself).
            Some(Ok(_)) => {}
            Some(Err(e)) => return Err(ConnectError::Stream(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;
    use tokio_tungstenite::{accept_async, WebSocketStream};

    type FakeHostStream = WebSocketStream<tokio::net::TcpStream>;

    fn identity(host_address: String) -> JoinerIdentity {
        JoinerIdentity {
            id: ParticipantId::parse("bot1").unwrap(),
            display_name: "Ada".to_string(),
            avatar: None,
            password: "hunter2".to_string(),
            host_address,
        }
    }

    fn sample_message(id: i32) -> Message {
        Message {
            id,
            ai: false,
            speaker_id: "user".to_string(),
            content: format!("message {id}"),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    /// A minimal fake host: accepts one connection at a time, drives one
    /// closure per accepted connection. Not built on `host.rs`/actix, per
    /// the plan: these tests exercise the joiner without depending on the
    /// host's actix-ws handler.
    async fn spawn_fake_host() -> (String, TcpListener) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        (format!("127.0.0.1:{}", addr.port()), listener)
    }

    async fn accept_one(listener: &TcpListener) -> FakeHostStream {
        let (stream, _) = listener.accept().await.unwrap();
        accept_async(stream).await.unwrap()
    }

    async fn send_frame(ws: &mut FakeHostStream, frame: &ServerFrame) {
        ws.send(WsMessage::text(serde_json::to_string(frame).unwrap()))
            .await
            .unwrap();
    }

    async fn recv_client_frame(ws: &mut FakeHostStream) -> ClientFrame {
        loop {
            match ws.next().await.expect("stream ended").unwrap() {
                WsMessage::Text(text) => return serde_json::from_str(&text).unwrap(),
                WsMessage::Ping(_) | WsMessage::Pong(_) => continue,
                other => panic!("expected a text frame, got {:?}", other),
            }
        }
    }

    fn nonce_and_proof(identity: &JoinerIdentity) -> ([u8; handshake::NONCE_BYTES], String) {
        let nonce = [7u8; handshake::NONCE_BYTES];
        let proof = handshake::join_proof(&identity.password, &nonce, &identity.id);
        (nonce, handshake::encode(&proof))
    }

    struct NoopGeneration;
    impl GenerateRequestHandler for NoopGeneration {
        fn handle(
            &self,
            _round_id: u64,
            _transcript: Vec<Message>,
            _tx: UnboundedSender<ClientFrame>,
        ) {
        }
    }

    #[tokio::test]
    async fn handshake_succeeds_and_seeds_the_transcript_mirror() {
        let (host_address, listener) = spawn_fake_host().await;
        let identity = identity(host_address);
        let handle: JoinerHandle = Arc::new(RwLock::new(JoinerShared::new(&identity, 1, None)));

        let server = tokio::spawn({
            let identity = identity.clone();
            async move {
                let mut ws = accept_one(&listener).await;
                let (nonce, expected_proof) = nonce_and_proof(&identity);
                send_frame(
                    &mut ws,
                    &ServerFrame::Challenge {
                        protocol_version: PROTOCOL_VERSION,
                        nonce: handshake::encode(&nonce),
                    },
                )
                .await;

                let ClientFrame::Join { proof, id, .. } = recv_client_frame(&mut ws).await else {
                    panic!("expected Join");
                };
                assert_eq!(proof, expected_proof);
                assert_eq!(id, identity.id);

                send_frame(
                    &mut ws,
                    &ServerFrame::Joined {
                        self_id: identity.id.clone(),
                        participants: vec![],
                        transcript: vec![sample_message(1), sample_message(2)],
                    },
                )
                .await;
                // Closes the connection so `connect_and_serve`'s serve loop
                // ends instead of blocking on a socket the server side
                // deliberately keeps open.
                ws.close(None).await.ok();
            }
        });

        let mut backoff = ReconnectBackoff::new();
        let generation: Arc<dyn GenerateRequestHandler> = Arc::new(NoopGeneration);
        let extraction = crate::multiplayer::joiner_compaction::noop_job();
        // The server closes right after `Joined`, so `connect_and_serve`
        // returns as soon as the serve loop sees the close. Only `run`'s
        // outer loop moves the state back to `Disconnected` on that `Ok`
        // return; called directly like this, `connect_and_serve` leaves the
        // state at whatever the handshake last set it to (`Connected`).
        let _ = connect_and_serve(&handle, &identity, &generation, &extraction, &mut backoff).await;
        server.await.unwrap();

        let shared = handle.read().unwrap();
        assert_eq!(shared.state, JoinerState::Connected);
        let (page, total, _) = shared.transcript.page(0, 15);
        assert_eq!(total, 2);
        assert_eq!(page, vec![sample_message(1), sample_message(2)]);
    }

    #[tokio::test]
    async fn rejected_handshake_stops_retrying_with_no_second_attempt() {
        let (host_address, listener) = spawn_fake_host().await;
        let identity = identity(host_address);
        let handle: JoinerHandle = Arc::new(RwLock::new(JoinerShared::new(&identity, 1, None)));

        let server = tokio::spawn(async move {
            let mut ws = accept_one(&listener).await;
            send_frame(
                &mut ws,
                &ServerFrame::Challenge {
                    protocol_version: PROTOCOL_VERSION,
                    nonce: handshake::encode(&[0u8; handshake::NONCE_BYTES]),
                },
            )
            .await;
            let _join = recv_client_frame(&mut ws).await;
            send_frame(
                &mut ws,
                &ServerFrame::Rejected {
                    reason: RejectReason::DuplicateId,
                },
            )
            .await;
            // If `run` retried after `Rejected`, it would land here; block
            // instead of accepting so a regression makes `run` hang rather
            // than silently succeeding on a second connection.
            std::future::pending::<()>().await;
        });

        let generation: Arc<dyn GenerateRequestHandler> = Arc::new(NoopGeneration);
        let extraction = crate::multiplayer::joiner_compaction::noop_job();
        // Bounded so a retry regression fails this test (`run` never
        // returning) instead of hanging the suite.
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            run(handle.clone(), identity, generation, extraction),
        )
        .await;
        server.abort();
        result.expect("run kept retrying after a Rejected handshake");

        let shared = handle.read().unwrap();
        assert!(matches!(shared.state, JoinerState::Rejected { .. }));
        assert_eq!(shared.attempts, 1);
    }

    #[tokio::test]
    async fn a_closed_connection_reconnects_and_reaches_connected_again() {
        let (host_address, listener) = spawn_fake_host().await;
        let identity = identity(host_address);
        let handle: JoinerHandle = Arc::new(RwLock::new(JoinerShared::new(&identity, 1, None)));

        let server = tokio::spawn({
            let identity = identity.clone();
            async move {
                for attempt in 0..2 {
                    let mut ws = accept_one(&listener).await;
                    let (nonce, _) = nonce_and_proof(&identity);
                    send_frame(
                        &mut ws,
                        &ServerFrame::Challenge {
                            protocol_version: PROTOCOL_VERSION,
                            nonce: handshake::encode(&nonce),
                        },
                    )
                    .await;
                    let _join = recv_client_frame(&mut ws).await;
                    send_frame(
                        &mut ws,
                        &ServerFrame::Joined {
                            self_id: identity.id.clone(),
                            participants: vec![],
                            transcript: vec![],
                        },
                    )
                    .await;
                    if attempt == 0 {
                        ws.close(None).await.ok();
                    } else {
                        // Keep the second connection open until the test
                        // aborts this task, so `read.next()` on the client
                        // side has nothing to error on before the assertion
                        // below runs.
                        std::future::pending::<()>().await;
                    }
                }
            }
        });

        let generation: Arc<dyn GenerateRequestHandler> = Arc::new(NoopGeneration);
        let extraction = crate::multiplayer::joiner_compaction::noop_job();
        let run_handle = handle.clone();
        tokio::spawn(run(run_handle, identity, generation, extraction));

        // `ReconnectBackoff::INITIAL` (1s) delays the second attempt, so
        // this polls rather than sleeping a single fixed duration up
        // front, the same convention `env_config.rs`'s `wait_until_listening`
        // uses for a startup race.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if handle.read().unwrap().state == JoinerState::Connected {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "joiner did not reconnect within 5s"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let shared = handle.read().unwrap();
        assert_eq!(shared.state, JoinerState::Connected);
        assert!(shared.attempts >= 2);
        drop(shared);

        server.abort();
    }

    /// A fixed-reply [`GenerateRequestHandler`]: unlike [`NoopGeneration`],
    /// it actually answers, so this module's own tests can pin the wire
    /// shape of a `GenerateRequest`'s reply without depending on
    /// `multiplayer::remote_generation::LocalModelGeneration` (#153), which
    /// needs a companion database and a turn slot this module's fake host
    /// does not set up.
    struct FixedReplyGeneration;
    impl GenerateRequestHandler for FixedReplyGeneration {
        fn handle(
            &self,
            round_id: u64,
            _transcript: Vec<Message>,
            tx: UnboundedSender<ClientFrame>,
        ) {
            let _ = tx.send(ClientFrame::ReplyFailed {
                round_id,
                reason: "stub failure".to_string(),
            });
        }
    }

    #[tokio::test]
    async fn generate_request_reaches_the_injected_handler() {
        let (host_address, listener) = spawn_fake_host().await;
        let identity = identity(host_address);
        let handle: JoinerHandle = Arc::new(RwLock::new(JoinerShared::new(&identity, 1, None)));

        let server = tokio::spawn({
            let identity = identity.clone();
            async move {
                let mut ws = accept_one(&listener).await;
                let (nonce, _) = nonce_and_proof(&identity);
                send_frame(
                    &mut ws,
                    &ServerFrame::Challenge {
                        protocol_version: PROTOCOL_VERSION,
                        nonce: handshake::encode(&nonce),
                    },
                )
                .await;
                let _join = recv_client_frame(&mut ws).await;
                send_frame(
                    &mut ws,
                    &ServerFrame::Joined {
                        self_id: identity.id.clone(),
                        participants: vec![],
                        transcript: vec![],
                    },
                )
                .await;
                send_frame(
                    &mut ws,
                    &ServerFrame::GenerateRequest {
                        round_id: 42,
                        transcript: vec![sample_message(1)],
                        continuity: None,
                    },
                )
                .await;

                let reply = recv_client_frame(&mut ws).await;
                assert_eq!(
                    reply,
                    ClientFrame::ReplyFailed {
                        round_id: 42,
                        reason: "stub failure".to_string(),
                    }
                );
            }
        });

        let mut backoff = ReconnectBackoff::new();
        let generation: Arc<dyn GenerateRequestHandler> = Arc::new(FixedReplyGeneration);
        let extraction = crate::multiplayer::joiner_compaction::noop_job();
        let _ = connect_and_serve(&handle, &identity, &generation, &extraction, &mut backoff).await;
        server.await.unwrap();
    }

    #[tokio::test]
    async fn last_continuity_is_set_from_a_payload_and_left_untouched_by_a_frame_without_one() {
        let (host_address, listener) = spawn_fake_host().await;
        let identity = identity(host_address);
        let handle: JoinerHandle = Arc::new(RwLock::new(JoinerShared::new(&identity, 1, None)));

        let payload = ContinuityPayload {
            compacted_through: 3,
            ..Default::default()
        };
        let payload_for_server = payload.clone();

        let server = tokio::spawn({
            let identity = identity.clone();
            async move {
                let mut ws = accept_one(&listener).await;
                let (nonce, _) = nonce_and_proof(&identity);
                send_frame(
                    &mut ws,
                    &ServerFrame::Challenge {
                        protocol_version: PROTOCOL_VERSION,
                        nonce: handshake::encode(&nonce),
                    },
                )
                .await;
                let _join = recv_client_frame(&mut ws).await;
                send_frame(
                    &mut ws,
                    &ServerFrame::Joined {
                        self_id: identity.id.clone(),
                        participants: vec![],
                        transcript: vec![],
                    },
                )
                .await;
                send_frame(
                    &mut ws,
                    &ServerFrame::GenerateRequest {
                        round_id: 1,
                        transcript: vec![],
                        continuity: Some(payload_for_server),
                    },
                )
                .await;
                send_frame(
                    &mut ws,
                    &ServerFrame::GenerateRequest {
                        round_id: 2,
                        transcript: vec![],
                        continuity: None,
                    },
                )
                .await;
                ws.close(None).await.ok();
            }
        });

        let mut backoff = ReconnectBackoff::new();
        let generation: Arc<dyn GenerateRequestHandler> = Arc::new(NoopGeneration);
        let extraction = crate::multiplayer::joiner_compaction::noop_job();
        let _ = connect_and_serve(&handle, &identity, &generation, &extraction, &mut backoff).await;
        server.await.unwrap();

        assert_eq!(
            handle.read().unwrap().last_continuity,
            Some(payload),
            "a GenerateRequest without a payload must never clear an already-known one"
        );
    }

    #[tokio::test]
    async fn message_edited_and_message_removed_frames_update_the_transcript_mirror() {
        let (host_address, listener) = spawn_fake_host().await;
        let identity = identity(host_address);
        let handle: JoinerHandle = Arc::new(RwLock::new(JoinerShared::new(&identity, 1, None)));

        let server = tokio::spawn({
            let identity = identity.clone();
            async move {
                let mut ws = accept_one(&listener).await;
                let (nonce, _) = nonce_and_proof(&identity);
                send_frame(
                    &mut ws,
                    &ServerFrame::Challenge {
                        protocol_version: PROTOCOL_VERSION,
                        nonce: handshake::encode(&nonce),
                    },
                )
                .await;
                let _join = recv_client_frame(&mut ws).await;
                send_frame(
                    &mut ws,
                    &ServerFrame::Joined {
                        self_id: identity.id.clone(),
                        participants: vec![],
                        transcript: vec![sample_message(1), sample_message(2)],
                    },
                )
                .await;

                let mut edited = sample_message(1);
                edited.content = "edited content".to_string();
                send_frame(&mut ws, &ServerFrame::MessageEdited { message: edited }).await;
                send_frame(&mut ws, &ServerFrame::MessageRemoved { id: 2 }).await;

                ws.close(None).await.ok();
            }
        });

        let mut backoff = ReconnectBackoff::new();
        let generation: Arc<dyn GenerateRequestHandler> = Arc::new(NoopGeneration);
        let extraction = crate::multiplayer::joiner_compaction::noop_job();
        let _ = connect_and_serve(&handle, &identity, &generation, &extraction, &mut backoff).await;
        server.await.unwrap();

        let shared = handle.read().unwrap();
        let (page, total, _) = shared.transcript.page(0, 15);
        assert_eq!(total, 1, "message 2 should have been removed");
        assert_eq!(page[0].content, "edited content");
    }

    #[test]
    fn read_avatar_upload_is_none_when_no_file_is_present() {
        // `paths::avatar_path()` resolves under the uninitialised `.`
        // directory in unit tests (see `paths.rs`'s test module doc), which
        // never has an `assets/avatar.png` checked in.
        assert!(read_avatar_upload().is_none());
    }

    #[test]
    fn describe_rejection_names_the_reason_for_every_variant() {
        for reason in [
            RejectReason::UnsupportedProtocol,
            RejectReason::BadProof,
            RejectReason::NoHostPassword,
            RejectReason::DuplicateId,
            RejectReason::ReservedId,
            RejectReason::JoinTimeout,
        ] {
            assert!(!describe_rejection(&reason).is_empty());
        }
        assert!(
            describe_rejection(&RejectReason::InvalidAvatar("too large".to_string()))
                .contains("too large")
        );
    }
}
