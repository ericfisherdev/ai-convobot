//! Shared helpers for the integration tests that spawn a real
//! `ai-companion` process (`env_config.rs`, `multiplayer_round.rs`,
//! `multiplayer_modes.rs`): the crate is bin-only (no `[lib]` target), so
//! these tests can only drive the binary from the outside, over HTTP and
//! `env!("CARGO_BIN_EXE_ai-companion")`, the same pattern `env_config.rs`
//! established first.
//!
//! Per Cargo's own convention for code shared between integration test
//! binaries, this lives under `tests/common/` rather than `tests/`: a
//! `.rs` file directly under `tests/` is compiled as its own standalone
//! test binary (with its own, usually-empty `main`), which is not what a
//! `mod common;` shared by other test files wants.

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

/// Kills and reaps the wrapped child on drop, so a failed assertion in a
/// test body (which unwinds past any explicit cleanup) can never leak a
/// still-listening `ai-companion` process behind it.
pub struct ChildGuard(pub Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Binds an ephemeral port and immediately releases it, so the caller has a
/// port number to hand a not-yet-spawned child. This alone is a genuine
/// TOCTOU race (the port is free between this returning and the child's own
/// `bind`, so another process can occasionally win it first); callers that
/// actually spawn a child on the returned port should go through
/// [`spawn_on_a_free_port`] instead, which retries the allocation when that
/// happens rather than trusting a single number.
fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind an ephemeral port");
    listener
        .local_addr()
        .expect("failed to read the ephemeral port's local address")
        .port()
}

/// Polls `addr` until a connection succeeds or `timeout` elapses, so a
/// respawn on an already-secured port (`multiplayer_modes.rs`'s
/// `respawn_instance`) does not race the server's startup work (schema
/// init, tantivy index open) with a fixed sleep. `#[allow(dead_code)]`
/// because `mod common;` is compiled fresh per integration-test binary, and
/// not every one of them needs this helper on top of
/// [`spawn_on_a_free_port`]'s own wait.
#[allow(dead_code)]
pub fn wait_until_listening(addr: SocketAddr, timeout: Duration) {
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

/// How many times [`spawn_on_a_free_port`] retries a lost port-allocation
/// race before giving up.
const MAX_PORT_ALLOCATION_ATTEMPTS: u32 = 5;

/// Allocates a free port, spawns the child `command_for` builds for it, and
/// waits for it to start listening — retrying with a freshly allocated port
/// (up to [`MAX_PORT_ALLOCATION_ATTEMPTS`] times) if the child exits before
/// it ever starts listening.
///
/// [`free_port`] necessarily releases its listener before the child can
/// bind the same port (the child is a separate process; there is no way to
/// hand it an already-bound socket without process-level fd inheritance
/// support `main.rs` does not have), so a competing process can occasionally
/// win that port first. When that happens the child's own `bind()` fails
/// and it exits early — detected here via `Child::try_wait` — rather than
/// this ever mistaking a *different* process's listener on the same port
/// for a successful start; a bind failure could otherwise be misread as
/// success if something else on the machine answers plain TCP connects on
/// the same port.
pub fn spawn_on_a_free_port(
    mut command_for: impl FnMut(u16) -> Command,
    timeout: Duration,
) -> (u16, SocketAddr, ChildGuard) {
    for attempt in 1..=MAX_PORT_ALLOCATION_ATTEMPTS {
        let port = free_port();
        let addr = SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, port));
        let mut child = command_for(port)
            .spawn()
            .expect("failed to spawn the ai-companion binary");

        let deadline = Instant::now() + timeout;
        loop {
            if let Ok(Some(_status)) = child.try_wait() {
                // Exited before it ever listened — most likely lost the
                // race for this port. Reap it and try a fresh one.
                break;
            }
            if TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
                return (port, addr, ChildGuard(child));
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "attempt {attempt}/{MAX_PORT_ALLOCATION_ATTEMPTS}: server on port {port} did not start listening within {timeout:?}"
                );
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    panic!(
        "failed to start the ai-companion binary after {MAX_PORT_ALLOCATION_ATTEMPTS} port-allocation attempts (kept losing the race for a free port)"
    );
}
