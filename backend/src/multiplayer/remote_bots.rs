//! The shared map of connected remote bots (joiners), and the channels
//! `host.rs`'s connection tasks and #131's round orchestrator use to talk to
//! each other without either side ever handing out an `actix_ws::Session`.
//!
//! #131's round orchestrator runs on a blocking thread (like `stream_turn`
//! today) and `Session::text` is async, so `RemoteBots` never hands out a
//! `Session`; each connection task in `host.rs` owns its own `Session` and
//! drains an outbound channel into it. `UnboundedSender::send` never blocks
//! and never awaits, so every method here is a plain, synchronous `fn`
//! callable from a `web::block` thread; no `.await` is ever taken while
//! holding either `Mutex`.

use std::collections::HashMap;
use std::sync::mpsc as std_mpsc;
use std::sync::Mutex;
use std::time::Instant;

use tokio::sync::mpsc::{self as tokio_mpsc, UnboundedReceiver, UnboundedSender};

use crate::multiplayer::protocol::{ClientFrame, ServerFrame};
use crate::participants::ParticipantId;

/// One connected remote bot's outbound channel, plus when it joined
/// (currently unused beyond bookkeeping; kept for #131's round-order and
/// diagnostics needs).
struct RemoteBotHandle {
    outbound: UnboundedSender<ServerFrame>,
    #[allow(dead_code)] // read by future diagnostics/round-order logic (#131)
    joined_at: Instant,
}

/// `register` was called for an id that is already connected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlreadyConnected;

/// `send` targeted an id that is not currently connected (never joined, or
/// its receiver/task has gone away).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotConnected;

/// The shared connection table. One instance per process, held in
/// `web::Data`.
#[derive(Default)]
pub struct RemoteBots {
    peers: Mutex<HashMap<ParticipantId, RemoteBotHandle>>,
    rounds: Mutex<HashMap<u64, std_mpsc::Sender<(ParticipantId, ClientFrame)>>>,
}

impl RemoteBots {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `id` as connected and returns the receiver its connection
    /// task should drain into `Session::text`.
    ///
    /// # Errors
    /// [`AlreadyConnected`] if `id` is already registered. `host.rs` maps
    /// this to [`crate::multiplayer::protocol::RejectReason::DuplicateId`].
    pub fn register(
        &self,
        id: ParticipantId,
    ) -> Result<UnboundedReceiver<ServerFrame>, AlreadyConnected> {
        let mut peers = self.peers.lock().unwrap_or_else(|p| p.into_inner());
        if peers.contains_key(&id) {
            return Err(AlreadyConnected);
        }
        let (tx, rx) = tokio_mpsc::unbounded_channel();
        peers.insert(
            id,
            RemoteBotHandle {
                outbound: tx,
                joined_at: Instant::now(),
            },
        );
        Ok(rx)
    }

    /// Removes `id` from the connection table. A no-op if it was not
    /// present (e.g. called twice on a race between a heartbeat timeout and
    /// a client-initiated close).
    pub fn unregister(&self, id: &ParticipantId) {
        let mut peers = self.peers.lock().unwrap_or_else(|p| p.into_inner());
        peers.remove(id);
    }

    /// The ids currently connected, in no particular order.
    pub fn connected_ids(&self) -> Vec<ParticipantId> {
        let peers = self.peers.lock().unwrap_or_else(|p| p.into_inner());
        peers.keys().cloned().collect()
    }

    /// Sends `frame` to exactly `id`.
    ///
    /// # Errors
    /// [`NotConnected`] if `id` is not registered, or its receiving end has
    /// been dropped (the connection task has already exited).
    pub fn send(&self, id: &ParticipantId, frame: ServerFrame) -> Result<(), NotConnected> {
        let peers = self.peers.lock().unwrap_or_else(|p| p.into_inner());
        let handle = peers.get(id).ok_or(NotConnected)?;
        handle.outbound.send(frame).map_err(|_| NotConnected)
    }

    /// Sends `frame` to every connected id except `except`. A peer whose
    /// receiver has been dropped is silently skipped, not treated as an
    /// error: a broadcast has no single target to fail for.
    pub fn broadcast(&self, frame: ServerFrame, except: Option<&ParticipantId>) {
        let peers = self.peers.lock().unwrap_or_else(|p| p.into_inner());
        for (id, handle) in peers.iter() {
            if Some(id) == except {
                continue;
            }
            let _ = handle.outbound.send(frame.clone());
        }
    }

    /// Opens a channel for round `round_id`: #154's `SocketRemoteGenerator`
    /// reads from the returned receiver (typically with `recv_timeout`)
    /// while [`RemoteBots::route_inbound`] forwards frames whose
    /// [`ClientFrame::round_id`] matches.
    pub fn subscribe_round(
        &self,
        round_id: u64,
    ) -> std_mpsc::Receiver<(ParticipantId, ClientFrame)> {
        let (tx, rx) = std_mpsc::channel();
        let mut rounds = self.rounds.lock().unwrap_or_else(|p| p.into_inner());
        rounds.insert(round_id, tx);
        rx
    }

    /// Closes round `round_id`'s inbound channel. Any frame that arrives
    /// after this is dropped by `route_inbound` like any other unsubscribed
    /// round.
    pub fn unsubscribe_round(&self, round_id: u64) {
        let mut rounds = self.rounds.lock().unwrap_or_else(|p| p.into_inner());
        rounds.remove(&round_id);
    }

    /// Routes an inbound frame from a connection task to whichever round
    /// subscribed to it. A frame with no round id (every `ClientFrame`
    /// variant in this issue), for an unsubscribed round, or whose receiver
    /// has been dropped, is logged and dropped rather than causing an
    /// error: the sender (a joiner) cannot be told routing failed without a
    /// wire frame this issue does not define.
    pub fn route_inbound(&self, from: &ParticipantId, frame: ClientFrame) {
        let Some(round_id) = frame.round_id() else {
            eprintln!("multiplayer: dropping frame from {} with no round id", from);
            return;
        };
        let rounds = self.rounds.lock().unwrap_or_else(|p| p.into_inner());
        match rounds.get(&round_id) {
            Some(sender) if sender.send((from.clone(), frame)).is_ok() => {}
            _ => {
                eprintln!(
                    "multiplayer: dropping frame from {} for unsubscribed round {}",
                    from, round_id
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> ParticipantId {
        ParticipantId::parse(s).unwrap()
    }

    fn sample_summary(id: ParticipantId) -> crate::multiplayer::protocol::ParticipantSummary {
        crate::multiplayer::protocol::ParticipantSummary {
            id,
            display_name: "Ada".to_string(),
            kind: crate::participants::ParticipantKind::RemoteBot,
            avatar_url: None,
            connected: true,
        }
    }

    #[test]
    fn register_then_duplicate_register_fails() {
        let bots = RemoteBots::new();
        let _rx = bots.register(id("bot1")).unwrap();
        assert_eq!(bots.register(id("bot1")).unwrap_err(), AlreadyConnected);
    }

    #[test]
    fn unregister_allows_a_fresh_register() {
        let bots = RemoteBots::new();
        let _rx = bots.register(id("bot1")).unwrap();
        bots.unregister(&id("bot1"));
        assert!(bots.register(id("bot1")).is_ok());
    }

    #[test]
    fn send_to_unknown_id_is_not_connected() {
        let bots = RemoteBots::new();
        let frame = ServerFrame::ParticipantJoined(sample_summary(id("bot1")));
        assert_eq!(bots.send(&id("bot1"), frame).unwrap_err(), NotConnected);
    }

    #[test]
    fn send_delivers_to_the_registered_receiver() {
        let bots = RemoteBots::new();
        let mut rx = bots.register(id("bot1")).unwrap();
        let frame = ServerFrame::ParticipantJoined(sample_summary(id("bot1")));
        bots.send(&id("bot1"), frame.clone()).unwrap();
        assert_eq!(rx.try_recv().unwrap(), frame);
    }

    #[test]
    fn broadcast_skips_the_excepted_peer() {
        let bots = RemoteBots::new();
        let mut rx1 = bots.register(id("bot1")).unwrap();
        let mut rx2 = bots.register(id("bot2")).unwrap();
        let frame = ServerFrame::ParticipantLeft { id: id("bot3") };
        bots.broadcast(frame.clone(), Some(&id("bot1")));
        assert!(rx1.try_recv().is_err());
        assert_eq!(rx2.try_recv().unwrap(), frame);
    }

    #[test]
    fn connected_ids_reflects_register_and_unregister() {
        let bots = RemoteBots::new();
        let _rx = bots.register(id("bot1")).unwrap();
        assert_eq!(bots.connected_ids(), vec![id("bot1")]);
        bots.unregister(&id("bot1"));
        assert!(bots.connected_ids().is_empty());
    }

    #[test]
    fn subscribe_and_unsubscribe_round_do_not_panic() {
        // No `ClientFrame` variant carries a round id yet (added by #131),
        // so the delivering case is exercised there; this covers the
        // plumbing that exists today.
        let bots = RemoteBots::new();
        let rx = bots.subscribe_round(1);
        drop(rx);
        bots.unsubscribe_round(1);
    }

    #[test]
    fn route_inbound_drops_a_frame_with_no_round_id() {
        let bots = RemoteBots::new();
        let rx = bots.subscribe_round(1);
        let frame = ClientFrame::Join {
            protocol_version: crate::multiplayer::protocol::PROTOCOL_VERSION,
            id: id("bot1"),
            display_name: "Ada".to_string(),
            avatar: None,
            proof: "x".to_string(),
        };
        bots.route_inbound(&id("bot1"), frame);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn route_inbound_drops_when_unsubscribed() {
        let bots = RemoteBots::new();
        bots.unsubscribe_round(1);
        // No panic, no receiver to deliver to.
        let frame = ClientFrame::Join {
            protocol_version: crate::multiplayer::protocol::PROTOCOL_VERSION,
            id: id("bot1"),
            display_name: "Ada".to_string(),
            avatar: None,
            proof: "x".to_string(),
        };
        bots.route_inbound(&id("bot1"), frame);
    }
}
