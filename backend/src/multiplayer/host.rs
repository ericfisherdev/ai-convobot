//! The `/api/multiplayer/ws` handler and the connection lifecycle it drives:
//! challenge, authenticate, admit, serve (heartbeats + inbound routing),
//! unregister.
//!
//! `HostConfigSource` is the seam that lets [`multiplayer_ws`] and
//! [`host_password_or_404`] be tested without SQLite, mirroring the
//! `TurnStore`/`SqliteTurnStore` seam in `chat_turn.rs`.

use std::net::IpAddr;
use std::sync::{Arc, Once, RwLock};
use std::time::{Duration, Instant};

use actix_web::{get, web, HttpRequest, HttpResponse};
use actix_ws::{AggregatedMessage, CloseCode, CloseReason};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::database::Database;
use crate::multiplayer::avatar::{self};
use crate::multiplayer::config::MultiplayerMode;
use crate::multiplayer::handshake::{self};
use crate::multiplayer::join_throttle::JoinThrottle;
use crate::multiplayer::protocol::{
    AvatarUpload, ClientFrame, ParticipantSummary, RejectReason, ServerFrame, PROTOCOL_VERSION,
};
use crate::multiplayer::remote_bots::{AlreadyConnected, RemoteBots};
use crate::participants::{
    AvatarRef, Participant, ParticipantError, ParticipantId, ParticipantKind, ParticipantRegistry,
};

/// The seam over `Database::get_config()` this handler and its tests use
/// instead of calling SQLite directly.
pub trait HostConfigSource: Send + Sync {
    /// `Ok(None)` when `multiplayer_mode != Host`; `Ok(Some(password))`
    /// otherwise (an empty string means "host mode is on but no password is
    /// set yet", handled by [`run_connection`], not here).
    fn host_password(&self) -> Result<Option<String>, String>;
}

/// The production [`HostConfigSource`], reading the live config on every
/// call (the mode and password can change at runtime via `PUT /api/config`).
pub struct SqliteHostConfig;

impl HostConfigSource for SqliteHostConfig {
    fn host_password(&self) -> Result<Option<String>, String> {
        let config = Database::get_config().map_err(|e| e.to_string())?;
        if config.multiplayer_mode != MultiplayerMode::Host {
            return Ok(None);
        }
        Ok(Some(config.multiplayer_password))
    }
}

/// Tunables for the connection lifecycle, overridden in tests to keep them
/// fast.
#[derive(Debug, Clone)]
pub struct HostSettings {
    pub join_timeout: Duration,
    pub heartbeat_interval: Duration,
    pub missed_pongs_before_drop: u32,
    pub max_join_frame_bytes: usize,
}

impl Default for HostSettings {
    fn default() -> Self {
        HostSettings {
            join_timeout: Duration::from_secs(10),
            heartbeat_interval: Duration::from_secs(15),
            missed_pongs_before_drop: 2,
            max_join_frame_bytes: 4 * 1024 * 1024,
        }
    }
}

static NO_PEER_IP_WARNED: Once = Once::new();
static EMPTY_PASSWORD_WARNED: Once = Once::new();

/// `req.peer_addr()`, falling back to the proxy-aware
/// `connection_info().realip_remote_addr()`. `None` means "could not
/// determine an address to throttle by", logged once by the caller.
fn peer_ip(req: &HttpRequest) -> Option<IpAddr> {
    if let Some(addr) = req.peer_addr() {
        return Some(addr.ip());
    }
    let info = req.connection_info();
    let candidate = info.realip_remote_addr()?;
    candidate.parse::<IpAddr>().ok().or_else(|| {
        candidate
            .parse::<std::net::SocketAddr>()
            .ok()
            .map(|a| a.ip())
    })
}

/// Reads the current host password via `host_config`, off the request
/// thread.
///
/// # Errors
/// `Ok(Err(response))` for the `404` a non-`Host` mode returns, or the
/// `500` a config read failure returns; both are ready-to-send responses so
/// callers do not have to build their own.
#[allow(clippy::result_large_err)] // see off_worker's doc comment in main.rs
async fn host_password_or_404(
    host_config: &Arc<dyn HostConfigSource>,
) -> Result<String, HttpResponse> {
    let config_source = host_config.clone();
    let password = crate::off_worker("failed to read multiplayer host config", move || {
        config_source.host_password()
    })
    .await?;
    password.ok_or_else(|| {
        HttpResponse::NotFound()
            .json(serde_json::json!({ "error": "multiplayer host mode is not enabled" }))
    })
}

/// Gates `GET /api/multiplayer/participants` and the avatar route on `Host`
/// mode, discarding the password `host_password_or_404` reads: those two
/// routes only need to know the mode, not the secret.
#[allow(clippy::result_large_err)] // see off_worker's doc comment in main.rs
pub async fn require_host_mode(
    host_config: &Arc<dyn HostConfigSource>,
) -> Result<(), HttpResponse> {
    host_password_or_404(host_config).await.map(|_| ())
}

/// Builds the `ParticipantSummary` the frontend and the `Joined`/
/// `ParticipantJoined` frames use. `connected` is always `true` for
/// `Human`/`HostBot` (they are the process itself); for `RemoteBot` it
/// reflects whether `remote_bots` currently holds a live connection for
/// that id. `avatar_url` is read straight from the registry's own
/// `Participant::avatar`, which already holds the right URL per kind (the
/// companion's `avatar_path` for `char`, none for `user`, and
/// `/api/multiplayer/participants/{id}/avatar` for a `RemoteBot` admitted
/// with an avatar).
pub fn participant_summary(p: &Participant, remote_bots: &RemoteBots) -> ParticipantSummary {
    let connected = match p.kind {
        ParticipantKind::Human | ParticipantKind::HostBot => true,
        ParticipantKind::RemoteBot => remote_bots.connected_ids().contains(&p.id),
    };
    ParticipantSummary {
        id: p.id.clone(),
        display_name: p.display_name.clone(),
        kind: p.kind.clone(),
        avatar_url: p.avatar.as_ref().map(|a| a.to_string()),
        connected,
    }
}

/// `GET /api/multiplayer/ws`: the join handshake and connection upgrade.
/// Registered unconditionally (the mode is a runtime-toggleable config
/// field); gated per-request through [`host_password_or_404`], so a `solo`
/// or `joiner` instance answers `404` here, matching every other
/// multiplayer route.
#[get("/api/multiplayer/ws")]
pub async fn multiplayer_ws(
    req: HttpRequest,
    body: web::Payload,
    host_config: web::Data<Arc<dyn HostConfigSource>>,
    throttle: web::Data<JoinThrottle>,
    settings: web::Data<HostSettings>,
    remote_bots: web::Data<RemoteBots>,
    participants: web::Data<RwLock<ParticipantRegistry>>,
) -> actix_web::Result<HttpResponse> {
    let peer_ip = peer_ip(&req);
    if peer_ip.is_none() {
        NO_PEER_IP_WARNED.call_once(|| {
            eprintln!(
                "multiplayer: could not determine a peer address to throttle joins by; joins from this connection type will not be rate-limited"
            );
        });
    }

    let password = match host_password_or_404(host_config.get_ref()).await {
        Ok(password) => password,
        Err(response) => return Ok(response),
    };

    // Reserved atomically with the block check (`try_reserve`, not a
    // separate is_blocked + record_failure pair) so a burst of concurrent
    // connections from one address cannot all pass the check before any of
    // them is counted — see the module doc on `join_throttle`.
    if let Some(ip) = peer_ip {
        if !throttle.try_reserve(ip, Instant::now()) {
            return Ok(HttpResponse::TooManyRequests()
                .json(serde_json::json!({ "error": "too many failed joins" })));
        }
    }

    let (response, session, msg_stream) = match actix_ws::handle(&req, body) {
        Ok(upgraded) => upgraded,
        Err(e) => {
            // The upgrade itself failed before a connection task exists to
            // release the reservation on any of its own exit paths, so
            // release it here instead of leaking it for the rest of the
            // window.
            if let Some(ip) = peer_ip {
                throttle.release(ip, Instant::now());
            }
            return Err(e);
        }
    };
    actix_web::rt::spawn(run_connection(
        session,
        msg_stream,
        password,
        peer_ip,
        remote_bots.into_inner(),
        throttle.into_inner(),
        settings.into_inner(),
        participants.into_inner(),
    ));

    Ok(response)
}

/// What [`await_join`] produced.
enum AwaitJoinOutcome {
    /// A `Join` frame at the expected protocol version, within the timeout.
    Frame(ClientFrame),
    /// A frame arrived that warrants a `Rejected` response before closing.
    Reject(RejectReason),
    /// The connection ended (closed, or a stream error) before any usable
    /// frame arrived; nothing left to reply to.
    GiveUp,
}

/// Waits up to `timeout` (one absolute deadline for the whole wait, not
/// reset per message) for a `Join` at [`PROTOCOL_VERSION`], replying to any
/// `Ping` received first and ignoring any `Pong` — `AggregatedMessageStream`
/// surfaces both as plain messages and does not answer `Ping` itself, and a
/// standards-compliant client may send one before `Join`. Everything else
/// the plan calls "not a `Join` at protocol version 1" — malformed JSON, a
/// future frame variant, or a version mismatch — becomes
/// [`RejectReason::UnsupportedProtocol`]; running out the deadline becomes
/// [`RejectReason::JoinTimeout`].
async fn await_join(
    session: &mut actix_ws::Session,
    stream: &mut actix_ws::AggregatedMessageStream,
    timeout: Duration,
) -> AwaitJoinOutcome {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let Ok(next) = tokio::time::timeout_at(deadline, stream.recv()).await else {
            return AwaitJoinOutcome::Reject(RejectReason::JoinTimeout);
        };
        let Some(Ok(msg)) = next else {
            return AwaitJoinOutcome::GiveUp;
        };
        match msg {
            AggregatedMessage::Text(text) => {
                return match serde_json::from_str::<ClientFrame>(&text) {
                    Ok(ClientFrame::Join {
                        protocol_version, ..
                    }) if protocol_version != PROTOCOL_VERSION => {
                        AwaitJoinOutcome::Reject(RejectReason::UnsupportedProtocol)
                    }
                    Ok(frame @ ClientFrame::Join { .. }) => AwaitJoinOutcome::Frame(frame),
                    // Any other frame before authentication (e.g. a #130
                    // joiner's `ReplyFailed` sent out of sequence) is not a
                    // `Join` at protocol version 1 either.
                    Ok(_) | Err(_) => AwaitJoinOutcome::Reject(RejectReason::UnsupportedProtocol),
                };
            }
            AggregatedMessage::Ping(bytes) => {
                if session.pong(&bytes).await.is_err() {
                    return AwaitJoinOutcome::GiveUp;
                }
            }
            AggregatedMessage::Pong(_) => {}
            AggregatedMessage::Close(_) => return AwaitJoinOutcome::GiveUp,
            AggregatedMessage::Binary(_) => {
                return AwaitJoinOutcome::Reject(RejectReason::UnsupportedProtocol);
            }
        }
    }
}

/// What [`admit`] produced on success: the receiver [`serve`] drains into
/// the socket, and the `Joined` frame to send back.
struct Admitted {
    outbound_rx: UnboundedReceiver<ServerFrame>,
    joined: ServerFrame,
}

enum AdmitOutcome {
    // Boxed: `ServerFrame::GenerateRequest` carries an optional #182
    // `ContinuityPayload`, which makes `ServerFrame` (and so `Admitted`,
    // which embeds one) large enough that clippy's `large_enum_variant`
    // flags the bare, unboxed form here.
    Admitted(Box<Admitted>),
    Rejected(RejectReason),
}

/// Validates the avatar (if any), inserts the participant into the shared
/// registry, registers it with `remote_bots`, stores the avatar, and builds
/// the `Joined` frame (participant snapshot plus the last 50 messages).
///
/// Order matters. Avatar *validation* (pure, no I/O) happens first so an
/// invalid upload is rejected before anything is written anywhere. Avatar
/// *storage* happens last, only after the registry insert and
/// `remote_bots.register` both succeed: storing first would let an upload
/// for an already-taken or reserved id overwrite that id's file on disk
/// before the `Duplicate`/`Reserved` check ever ran. If storage then fails,
/// the registry entry and `remote_bots` registration are rolled back and
/// the join is rejected, so a participant is never admitted with an
/// avatar URL for bytes that were never actually written; `Participant::avatar`
/// is set only from *this* join's own successful store, never derived from
/// `avatar::find_stored_avatar`, which is not scoped to this join and could
/// still find an older file after a failed write.
async fn admit(
    id: &ParticipantId,
    display_name: String,
    avatar_upload: Option<AvatarUpload>,
    participants: &RwLock<ParticipantRegistry>,
    remote_bots: &RemoteBots,
) -> AdmitOutcome {
    let validated_avatar = match avatar_upload {
        None => None,
        Some(upload) => match avatar::validate_avatar(&upload) {
            Ok(validated) => Some(validated),
            Err(e) => return AdmitOutcome::Rejected(RejectReason::InvalidAvatar(e.to_string())),
        },
    };

    let participant = Participant {
        id: id.clone(),
        display_name: display_name.clone(),
        kind: ParticipantKind::RemoteBot,
        avatar: None,
    };

    {
        let mut registry = participants.write().unwrap_or_else(|p| p.into_inner());
        if let Err(e) = registry.insert(participant) {
            return AdmitOutcome::Rejected(match e {
                ParticipantError::Reserved(_) => RejectReason::ReservedId,
                _ => RejectReason::DuplicateId,
            });
        }
    }

    let outbound_rx = match remote_bots.register(id.clone()) {
        Ok(rx) => rx,
        Err(AlreadyConnected) => {
            let mut registry = participants.write().unwrap_or_else(|p| p.into_inner());
            let _ = registry.remove(id);
            return AdmitOutcome::Rejected(RejectReason::DuplicateId);
        }
    };

    if let Some(validated) = validated_avatar {
        match avatar::store_participant_avatar(id, &validated) {
            Ok(_) => {
                let avatar_ref =
                    AvatarRef::new(format!("/api/multiplayer/participants/{}/avatar", id));
                let mut registry = participants.write().unwrap_or_else(|p| p.into_inner());
                let _ = registry.rename(id, &display_name, Some(avatar_ref));
            }
            Err(e) => {
                eprintln!("multiplayer: failed to store avatar for {}: {}", id, e);
                avatar::remove_stored_avatar(id);
                remote_bots.unregister(id);
                let mut registry = participants.write().unwrap_or_else(|p| p.into_inner());
                let _ = registry.remove(id);
                return AdmitOutcome::Rejected(RejectReason::InvalidAvatar(
                    "failed to store avatar, check host logs".to_string(),
                ));
            }
        }
    }

    let participant_summaries = {
        let registry = participants.read().unwrap_or_else(|p| p.into_inner());
        registry
            .iter()
            .map(|p| participant_summary(p, remote_bots))
            .collect::<Vec<_>>()
    };

    let transcript = match web::block(|| Database::get_x_messages(50, 0)).await {
        Ok(Ok(messages)) => messages,
        Ok(Err(e)) => {
            eprintln!("multiplayer: failed to load transcript for {}: {}", id, e);
            Vec::new()
        }
        Err(e) => {
            eprintln!(
                "multiplayer: blocking task failed loading transcript for {}: {}",
                id, e
            );
            Vec::new()
        }
    };

    AdmitOutcome::Admitted(Box::new(Admitted {
        outbound_rx,
        joined: ServerFrame::Joined {
            self_id: id.clone(),
            participants: participant_summaries,
            transcript,
        },
    }))
}

/// Serialises `frame` and sends it, logging (rather than propagating) a
/// serialisation failure: every `ServerFrame` this module builds is made of
/// plain owned data, so a failure here would be a bug, not a runtime
/// condition callers need to branch on.
async fn send_frame(
    session: &mut actix_ws::Session,
    frame: &ServerFrame,
) -> Result<(), actix_ws::Closed> {
    let json = serde_json::to_string(frame).unwrap_or_else(|e| {
        eprintln!("multiplayer: failed to serialise {:?}: {}", frame, e);
        "{}".to_string()
    });
    session.text(json).await
}

fn policy_close_reason() -> CloseReason {
    CloseReason {
        code: CloseCode::Policy,
        description: None,
    }
}

/// Sends `Rejected { reason }` then closes with [`CloseCode::Policy`].
/// Errors from either step are ignored: the connection is being torn down
/// either way.
async fn reject_and_close(session: &mut actix_ws::Session, reason: RejectReason) {
    let _ = send_frame(session, &ServerFrame::Rejected { reason }).await;
    let _ = session.clone().close(Some(policy_close_reason())).await;
}

/// The post-join loop: reads inbound frames (routed to `RemoteBots` for
/// #131's round protocol), drains `outbound_rx` into the socket, and runs
/// the heartbeat. Returns when the connection should end, for any reason
/// (client close, protocol error, too many missed pongs, or the outbound
/// channel closing because `remote_bots` dropped this peer's sender).
async fn serve(
    session: &mut actix_ws::Session,
    stream: &mut actix_ws::AggregatedMessageStream,
    id: &ParticipantId,
    mut outbound_rx: UnboundedReceiver<ServerFrame>,
    remote_bots: &RemoteBots,
    settings: &HostSettings,
) {
    let mut heartbeat = tokio::time::interval(settings.heartbeat_interval);
    // The first tick fires immediately; consume it so the connection is not
    // pinged the instant it joins.
    heartbeat.tick().await;
    let mut missed_pongs: u32 = 0;

    loop {
        tokio::select! {
            incoming = stream.recv() => {
                match incoming {
                    Some(Ok(AggregatedMessage::Pong(_))) => {
                        missed_pongs = 0;
                    }
                    Some(Ok(AggregatedMessage::Ping(bytes))) => {
                        if session.pong(&bytes).await.is_err() {
                            return;
                        }
                    }
                    Some(Ok(AggregatedMessage::Text(text))) => {
                        match serde_json::from_str::<ClientFrame>(&text) {
                            Ok(frame) => remote_bots.route_inbound(id, frame),
                            Err(e) => {
                                eprintln!("multiplayer: dropping unparsable frame from {}: {}", id, e);
                            }
                        }
                    }
                    Some(Ok(AggregatedMessage::Binary(_))) => {
                        // No binary frame is defined by the protocol; ignore.
                    }
                    Some(Ok(AggregatedMessage::Close(_))) | None => return,
                    Some(Err(e)) => {
                        eprintln!("multiplayer: protocol error from {}: {}", id, e);
                        return;
                    }
                }
            }
            outbound = outbound_rx.recv() => {
                match outbound {
                    Some(frame) => {
                        if send_frame(session, &frame).await.is_err() {
                            return;
                        }
                    }
                    None => return,
                }
            }
            _ = heartbeat.tick() => {
                if missed_pongs >= settings.missed_pongs_before_drop {
                    return;
                }
                missed_pongs += 1;
                if session.ping(b"").await.is_err() {
                    return;
                }
            }
        }
    }
}

/// Removes `id` from both the connection table and the shared registry,
/// broadcasts `ParticipantLeft`, and closes the session. Called on every
/// exit path of an admitted connection (see [`run_connection`]), so it
/// never leaves a disconnected participant behind in the registry.
async fn unregister(
    id: &ParticipantId,
    session: actix_ws::Session,
    remote_bots: &RemoteBots,
    participants: &RwLock<ParticipantRegistry>,
) {
    remote_bots.unregister(id);
    {
        let mut registry = participants.write().unwrap_or_else(|p| p.into_inner());
        let _ = registry.remove(id);
    }
    remote_bots.broadcast(ServerFrame::ParticipantLeft { id: id.clone() }, None);
    println!("{} left", id);
    let _ = session.close(None).await;
}

/// The whole connection lifecycle for one upgraded socket, spawned by
/// [`multiplayer_ws`] and running until the connection ends. Structured as
/// straight-line `await`s rather than early returns from a shared
/// `unregister` call: every rejection path before admission closes the
/// session directly (nothing to unregister yet); every path after
/// admission falls through to the single `unregister` call at the end, so
/// it always runs exactly once for an admitted participant.
#[allow(clippy::too_many_arguments)]
async fn run_connection(
    mut session: actix_ws::Session,
    msg_stream: actix_ws::MessageStream,
    password: String,
    peer_ip: Option<IpAddr>,
    remote_bots: Arc<RemoteBots>,
    throttle: Arc<JoinThrottle>,
    settings: Arc<HostSettings>,
    participants: Arc<RwLock<ParticipantRegistry>>,
) {
    let mut stream = msg_stream
        .max_frame_size(settings.max_join_frame_bytes)
        .aggregate_continuations()
        .max_continuation_size(settings.max_join_frame_bytes);

    let nonce = handshake::new_nonce();
    let challenge = ServerFrame::Challenge {
        protocol_version: PROTOCOL_VERSION,
        nonce: handshake::encode(&nonce),
    };
    if send_frame(&mut session, &challenge).await.is_err() {
        // Never got as far as reading a proof, so this was not a password
        // guess — release the reservation `multiplayer_ws` made rather
        // than leave it counted for the rest of the window.
        if let Some(ip) = peer_ip {
            throttle.release(ip, Instant::now());
        }
        return;
    }

    let join_frame = match await_join(&mut session, &mut stream, settings.join_timeout).await {
        AwaitJoinOutcome::Frame(frame) => frame,
        AwaitJoinOutcome::Reject(reason) => {
            // Not a bad password guess (a timeout, a malformed frame, or an
            // unsupported protocol version), so release the reservation
            // rather than let it count toward the throttle.
            if let Some(ip) = peer_ip {
                throttle.release(ip, Instant::now());
            }
            reject_and_close(&mut session, reason).await;
            return;
        }
        AwaitJoinOutcome::GiveUp => {
            if let Some(ip) = peer_ip {
                throttle.release(ip, Instant::now());
            }
            return;
        }
    };

    // `await_join` only ever returns `AwaitJoinOutcome::Frame` for a `Join`
    // (see its match arms above); every other `ClientFrame` variant becomes
    // a `Reject` there instead, which already returned above.
    let ClientFrame::Join {
        id,
        display_name,
        avatar,
        proof,
        ..
    } = join_frame
    else {
        unreachable!("await_join only returns AwaitJoinOutcome::Frame for a Join");
    };

    if password.is_empty() {
        EMPTY_PASSWORD_WARNED.call_once(|| {
            eprintln!(
                "multiplayer: refusing joins because host mode has no password set; set one in the settings dialog"
            );
        });
        if let Some(ip) = peer_ip {
            throttle.release(ip, Instant::now());
        }
        reject_and_close(&mut session, RejectReason::NoHostPassword).await;
        return;
    }

    if !handshake::verify_join_proof(&password, &nonce, &id, &proof) {
        // Deliberately not released: a wrong password proof is exactly the
        // kind of attempt this throttle exists to keep counted for the
        // rest of the window.
        reject_and_close(&mut session, RejectReason::BadProof).await;
        return;
    }

    // The proof was valid, so this attempt is done consuming throttle
    // budget regardless of what `admit` decides next (`DuplicateId`,
    // `ReservedId`, and an invalid avatar are not password guesses).
    if let Some(ip) = peer_ip {
        throttle.release(ip, Instant::now());
    }

    let admitted = match admit(&id, display_name, avatar, &participants, &remote_bots).await {
        AdmitOutcome::Admitted(admitted) => admitted,
        AdmitOutcome::Rejected(reason) => {
            reject_and_close(&mut session, reason).await;
            return;
        }
    };

    if send_frame(&mut session, &admitted.joined).await.is_err() {
        unregister(&id, session, &remote_bots, &participants).await;
        return;
    }

    let self_summary = {
        let registry = participants.read().unwrap_or_else(|p| p.into_inner());
        registry
            .get(&id)
            .map(|p| participant_summary(p, &remote_bots))
    };
    if let Some(summary) = self_summary {
        remote_bots.broadcast(ServerFrame::ParticipantJoined(summary), Some(&id));
    }
    match peer_ip {
        Some(ip) => println!("{} joined from {}", id, ip),
        None => println!("{} joined", id),
    }

    serve(
        &mut session,
        &mut stream,
        &id,
        admitted.outbound_rx,
        &remote_bots,
        &settings,
    )
    .await;

    unregister(&id, session, &remote_bots, &participants).await;
}

#[cfg(test)]
mod socket_tests {
    use super::*;
    use actix_web::{test, App};
    use base64::Engine as _;
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode as TCloseCode;
    use tokio_tungstenite::tungstenite::Message as WsMessage;
    use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

    type WsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

    struct StubHostConfig(Option<String>);

    impl HostConfigSource for StubHostConfig {
        fn host_password(&self) -> Result<Option<String>, String> {
            Ok(self.0.clone())
        }
    }

    /// Everything one test needs: a bound, running server for the socket
    /// itself, plus the same shared `web::Data` handles wired into a
    /// separate `test::init_service` app for the plain HTTP routes — the
    /// two apps share state because `web::Data` clones share the
    /// underlying `Arc`.
    struct Harness {
        port: u16,
        server_handle: actix_web::dev::ServerHandle,
        participants: web::Data<RwLock<ParticipantRegistry>>,
        remote_bots: web::Data<RemoteBots>,
        throttle: web::Data<JoinThrottle>,
    }

    impl Harness {
        async fn start(password: Option<&str>) -> Self {
            Self::start_with_settings(password, fast_settings()).await
        }

        async fn start_with_settings(password: Option<&str>, settings: HostSettings) -> Self {
            let participants =
                web::Data::new(RwLock::new(ParticipantRegistry::solo("Alice", "Bob", None)));
            let remote_bots = web::Data::new(RemoteBots::new());
            let throttle = web::Data::new(JoinThrottle::new(5, Duration::from_secs(600)));
            let settings = web::Data::new(settings);
            let host_config: web::Data<Arc<dyn HostConfigSource>> =
                web::Data::new(Arc::new(StubHostConfig(password.map(str::to_string)))
                    as Arc<dyn HostConfigSource>);

            let (p, rb, th, se, hc) = (
                participants.clone(),
                remote_bots.clone(),
                throttle.clone(),
                settings.clone(),
                host_config.clone(),
            );
            let server = actix_web::HttpServer::new(move || {
                App::new()
                    .app_data(p.clone())
                    .app_data(rb.clone())
                    .app_data(th.clone())
                    .app_data(se.clone())
                    .app_data(hc.clone())
                    .service(multiplayer_ws)
                    .service(crate::multiplayer_participants)
                    .service(crate::multiplayer_participant_avatar)
            })
            .workers(1)
            .bind("127.0.0.1:0")
            .unwrap();
            let port = server.addrs()[0].port();
            let server = server.run();
            let server_handle = server.handle();
            actix_web::rt::spawn(server);

            Harness {
                port,
                server_handle,
                participants,
                remote_bots,
                throttle,
            }
        }

        async fn connect(&self) -> WsStream {
            let (ws, _) = connect_async(format!("ws://127.0.0.1:{}/api/multiplayer/ws", self.port))
                .await
                .unwrap();
            ws
        }

        async fn upgrade_status(&self) -> Option<u16> {
            match connect_async(format!("ws://127.0.0.1:{}/api/multiplayer/ws", self.port)).await {
                Ok(_) => None,
                Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
                    Some(response.status().as_u16())
                }
                Err(e) => panic!("unexpected connect error: {}", e),
            }
        }

        async fn call_get_participants(&self) -> serde_json::Value {
            let (p, rb, th) = (
                self.participants.clone(),
                self.remote_bots.clone(),
                self.throttle.clone(),
            );
            let host_config: web::Data<Arc<dyn HostConfigSource>> =
                web::Data::new(Arc::new(StubHostConfig(Some("irrelevant".to_string())))
                    as Arc<dyn HostConfigSource>);
            let settings = web::Data::new(fast_settings());
            let app = test::init_service(
                App::new()
                    .app_data(p)
                    .app_data(rb)
                    .app_data(th)
                    .app_data(settings)
                    .app_data(host_config)
                    .service(crate::multiplayer_participants),
            )
            .await;
            let req = test::TestRequest::get()
                .uri("/api/multiplayer/participants")
                .to_request();
            test::call_and_read_body_json(&app, req).await
        }

        async fn stop(self) {
            self.server_handle.stop(true).await;
        }
    }

    fn fast_settings() -> HostSettings {
        HostSettings {
            join_timeout: Duration::from_millis(300),
            heartbeat_interval: Duration::from_millis(200),
            missed_pongs_before_drop: 2,
            max_join_frame_bytes: 4 * 1024 * 1024,
        }
    }

    async fn recv_frame(ws: &mut WsStream) -> ServerFrame {
        loop {
            match ws.next().await.expect("stream ended").expect("ws error") {
                WsMessage::Text(text) => return serde_json::from_str(&text).unwrap(),
                WsMessage::Ping(_) | WsMessage::Pong(_) => continue,
                other => panic!("expected a text frame, got {:?}", other),
            }
        }
    }

    async fn await_challenge(ws: &mut WsStream) -> [u8; handshake::NONCE_BYTES] {
        match recv_frame(ws).await {
            ServerFrame::Challenge { nonce, .. } => {
                handshake::decode(&nonce).unwrap().try_into().unwrap()
            }
            other => panic!("expected Challenge, got {:?}", other),
        }
    }

    async fn send_join(
        ws: &mut WsStream,
        password: &str,
        nonce: &[u8; handshake::NONCE_BYTES],
        id: &str,
        avatar: Option<AvatarUpload>,
    ) {
        let participant_id = ParticipantId::parse(id).unwrap();
        let proof = handshake::join_proof(password, nonce, &participant_id);
        let frame = ClientFrame::Join {
            protocol_version: PROTOCOL_VERSION,
            id: participant_id,
            display_name: id.to_string(),
            avatar,
            proof: handshake::encode(&proof),
        };
        ws.send(WsMessage::Text(
            serde_json::to_string(&frame).unwrap().into(),
        ))
        .await
        .unwrap();
    }

    async fn send_join_with_wrong_proof(ws: &mut WsStream, id: &str) {
        let frame = ClientFrame::Join {
            protocol_version: PROTOCOL_VERSION,
            id: ParticipantId::parse(id).unwrap(),
            display_name: id.to_string(),
            avatar: None,
            proof: "wrong".to_string(),
        };
        ws.send(WsMessage::Text(
            serde_json::to_string(&frame).unwrap().into(),
        ))
        .await
        .unwrap();
    }

    #[actix_web::test]
    async fn valid_proof_completes_the_handshake_and_lists_the_participant() {
        let harness = Harness::start(Some("hunter2")).await;
        let mut ws = harness.connect().await;
        let nonce = await_challenge(&mut ws).await;

        let png = AvatarUpload {
            mime: "image/png".to_string(),
            data_base64: base64::engine::general_purpose::STANDARD
                .encode([0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, b'x']),
        };
        send_join(&mut ws, "hunter2", &nonce, "bot1", Some(png)).await;

        match recv_frame(&mut ws).await {
            ServerFrame::Joined { self_id, .. } => {
                assert_eq!(self_id, ParticipantId::parse("bot1").unwrap())
            }
            other => panic!("expected Joined, got {:?}", other),
        }

        assert!(harness
            .participants
            .read()
            .unwrap()
            .get(&ParticipantId::parse("bot1").unwrap())
            .is_some());

        let body = harness.call_get_participants().await;
        let rows = body.as_array().unwrap();
        let bot1 = rows
            .iter()
            .find(|r| r["id"] == "bot1")
            .expect("bot1 listed");
        assert_eq!(bot1["connected"], true);
        assert_eq!(
            bot1["avatar_url"],
            "/api/multiplayer/participants/bot1/avatar"
        );

        harness.stop().await;
    }

    #[actix_web::test]
    async fn wrong_proof_is_rejected_and_closes_with_policy_violation() {
        let harness = Harness::start(Some("hunter2")).await;
        let mut ws = harness.connect().await;
        let _nonce = await_challenge(&mut ws).await;
        send_join_with_wrong_proof(&mut ws, "bot1").await;

        match recv_frame(&mut ws).await {
            ServerFrame::Rejected {
                reason: RejectReason::BadProof,
            } => {}
            other => panic!("expected Rejected(bad_proof), got {:?}", other),
        }

        match ws.next().await.unwrap().unwrap() {
            WsMessage::Close(Some(frame)) => assert_eq!(frame.code, TCloseCode::Policy),
            other => panic!("expected a policy-violation close, got {:?}", other),
        }

        assert!(harness
            .participants
            .read()
            .unwrap()
            .get(&ParticipantId::parse("bot1").unwrap())
            .is_none());

        harness.stop().await;
    }

    #[actix_web::test]
    async fn five_failed_joins_throttle_the_sixth_upgrade_attempt() {
        let harness = Harness::start(Some("hunter2")).await;
        for _ in 0..5 {
            let mut ws = harness.connect().await;
            let _nonce = await_challenge(&mut ws).await;
            send_join_with_wrong_proof(&mut ws, "bot1").await;
            let _ = recv_frame(&mut ws).await;
        }

        assert_eq!(harness.upgrade_status().await, Some(429));

        harness.stop().await;
    }

    #[actix_web::test]
    async fn duplicate_id_is_rejected() {
        let harness = Harness::start(Some("hunter2")).await;

        let mut first = harness.connect().await;
        let nonce1 = await_challenge(&mut first).await;
        send_join(&mut first, "hunter2", &nonce1, "bot2", None).await;
        let _ = recv_frame(&mut first).await; // Joined

        let mut second = harness.connect().await;
        let nonce2 = await_challenge(&mut second).await;
        send_join(&mut second, "hunter2", &nonce2, "bot2", None).await;

        match recv_frame(&mut second).await {
            ServerFrame::Rejected {
                reason: RejectReason::DuplicateId,
            } => {}
            other => panic!("expected Rejected(duplicate_id), got {:?}", other),
        }

        harness.stop().await;
    }

    #[actix_web::test]
    async fn dropping_the_socket_removes_the_participant_and_notifies_others() {
        let harness = Harness::start(Some("hunter2")).await;

        let mut watcher = harness.connect().await;
        let nonce = await_challenge(&mut watcher).await;
        send_join(&mut watcher, "hunter2", &nonce, "bot3", None).await;
        let _ = recv_frame(&mut watcher).await; // Joined

        let mut leaver = harness.connect().await;
        let nonce = await_challenge(&mut leaver).await;
        send_join(&mut leaver, "hunter2", &nonce, "bot4", None).await;
        let _ = recv_frame(&mut leaver).await; // Joined
        let _ = recv_frame(&mut watcher).await; // ParticipantJoined(bot4)

        drop(leaver);

        match recv_frame(&mut watcher).await {
            ServerFrame::ParticipantLeft { id } => {
                assert_eq!(id, ParticipantId::parse("bot4").unwrap())
            }
            other => panic!("expected ParticipantLeft, got {:?}", other),
        }

        assert!(harness
            .participants
            .read()
            .unwrap()
            .get(&ParticipantId::parse("bot4").unwrap())
            .is_none());

        harness.stop().await;
    }

    #[actix_web::test]
    async fn no_host_password_configured_answers_404() {
        let harness = Harness::start(None).await;
        assert_eq!(harness.upgrade_status().await, Some(404));
        harness.stop().await;
    }

    #[actix_web::test]
    async fn oversized_avatar_is_rejected() {
        let harness = Harness::start(Some("hunter2")).await;
        let mut ws = harness.connect().await;
        let nonce = await_challenge(&mut ws).await;

        let mut bytes = vec![0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        bytes.resize(avatar::MAX_AVATAR_BYTES + 1, 0);
        let oversized = AvatarUpload {
            mime: "image/png".to_string(),
            data_base64: base64::engine::general_purpose::STANDARD.encode(&bytes),
        };
        send_join(&mut ws, "hunter2", &nonce, "bot5", Some(oversized)).await;

        match recv_frame(&mut ws).await {
            ServerFrame::Rejected {
                reason: RejectReason::InvalidAvatar(_),
            } => {}
            other => panic!("expected Rejected(invalid_avatar), got {:?}", other),
        }

        harness.stop().await;
    }
}
