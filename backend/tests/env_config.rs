//! Integration tests for #107: `main()` reads `COMPANION_HOST`,
//! `COMPANION_PORT`, and `COMPANION_DATA_DIR` instead of hardcoding
//! `0.0.0.0:3000` and resolving every storage path against the working
//! directory.
//!
//! Unlike `tests/startup_failure.rs` (whose processes fail before `bind()`
//! and exit on their own), the happy-path tests here spawn a server that
//! keeps running, so each one uses `ChildGuard` to make sure the process is
//! always killed and reaped, even if an assertion panics.

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Kills and reaps the wrapped child on drop, so a failed assertion in a
/// test body (which unwinds past any explicit cleanup) can never leak a
/// still-listening `ai-companion` process behind it.
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Binds an ephemeral port and immediately releases it, so the spawned
/// server has a free port to bind that will not collide with anything else
/// running on the machine (including a dev server on the default 3000).
fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind an ephemeral port");
    listener
        .local_addr()
        .expect("failed to read the ephemeral port's local address")
        .port()
}

/// Polls `addr` until a connection succeeds or `timeout` elapses, so the
/// test does not race the server's startup work (schema init, tantivy index
/// open) with a fixed sleep.
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

#[test]
fn honours_port_and_data_dir() {
    let port = free_port();
    let cwd_dir = tempfile::tempdir().expect("failed to create the working-directory temp dir");
    let data_root = tempfile::tempdir().expect("failed to create the data-dir temp dir");
    // Nested and not yet created, to prove `create_dir_all` rather than a
    // bare `create_dir`.
    let data_dir = data_root.path().join("nested").join("data");

    let child = Command::new(env!("CARGO_BIN_EXE_ai-companion"))
        .current_dir(cwd_dir.path())
        .env("COMPANION_HOST", "127.0.0.1")
        .env("COMPANION_PORT", port.to_string())
        .env("COMPANION_DATA_DIR", &data_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn the ai-companion binary");
    let _guard = ChildGuard(child);

    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    wait_until_listening(addr, Duration::from_secs(10));

    assert!(
        data_dir.join("companion_database.db").exists(),
        "expected the database to be created under COMPANION_DATA_DIR"
    );
    assert!(
        data_dir.join("longterm_memory").is_dir(),
        "expected the long-term memory index to be created under COMPANION_DATA_DIR"
    );
    assert!(
        data_dir.join("assets").is_dir(),
        "expected the assets directory to be created under COMPANION_DATA_DIR"
    );

    assert!(
        is_empty_dir(cwd_dir.path()),
        "nothing should have been written to the working directory when COMPANION_DATA_DIR is set"
    );
}

#[test]
fn defaults_to_the_working_directory() {
    let port = free_port();
    let cwd_dir = tempfile::tempdir().expect("failed to create the working-directory temp dir");

    let child = Command::new(env!("CARGO_BIN_EXE_ai-companion"))
        .current_dir(cwd_dir.path())
        .env("COMPANION_HOST", "127.0.0.1")
        .env("COMPANION_PORT", port.to_string())
        .env_remove("COMPANION_DATA_DIR")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn the ai-companion binary");
    let _guard = ChildGuard(child);

    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    wait_until_listening(addr, Duration::from_secs(10));

    assert!(
        cwd_dir.path().join("companion_database.db").exists(),
        "expected the database to be created in the working directory by default"
    );
    assert!(
        cwd_dir.path().join("longterm_memory").is_dir(),
        "expected the long-term memory index to be created in the working directory by default"
    );
}

#[test]
fn exits_non_zero_on_an_invalid_port() {
    let cwd_dir = tempfile::tempdir().expect("failed to create the working-directory temp dir");

    let output = Command::new(env!("CARGO_BIN_EXE_ai-companion"))
        .current_dir(cwd_dir.path())
        .env("COMPANION_PORT", "abc")
        .output()
        .expect("failed to spawn the ai-companion binary");

    assert!(
        !output.status.success(),
        "expected a non-zero exit status, got {:?}; stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("COMPANION_PORT"),
        "stderr did not name COMPANION_PORT: {stderr}"
    );
}

fn is_empty_dir(dir: &Path) -> bool {
    match std::fs::read_dir(dir) {
        Ok(mut entries) => entries.next().is_none(),
        Err(_) => false,
    }
}
