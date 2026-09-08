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
use std::process::Child;
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

/// Binds an ephemeral port and immediately releases it, so the spawned
/// server has a free port to bind that will not collide with anything else
/// running on the machine (including a dev server on the default 3000).
pub fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind an ephemeral port");
    listener
        .local_addr()
        .expect("failed to read the ephemeral port's local address")
        .port()
}

/// Polls `addr` until a connection succeeds or `timeout` elapses, so the
/// test does not race the server's startup work (schema init, tantivy index
/// open) with a fixed sleep.
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
