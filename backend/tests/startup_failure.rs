//! Integration tests for the fail-fast startup behaviour added for #98:
//! `main()` used to warn and keep running when the database or long-term
//! memory index could not be opened, serving 500s from a half-initialised
//! process. It now exits non-zero before binding a port.
//!
//! Startup never loads a model or touches the network before init_storage,
//! so these are safe to spawn in CI: both processes fail before `bind()`,
//! so there is no port-3000 conflict with a dev server and no model weight
//! download.

use std::process::Command;

/// `companion_database.db` as a directory (rather than a regular file) makes
/// SQLite fail to open it with "unable to open database file" on every
/// platform, without depending on the test running as a non-root user (a
/// `chmod 555` parent directory is still writable by root, which would make
/// that approach flaky under CI/Docker containers that run as root).
#[test]
fn exits_non_zero_when_the_database_path_cannot_be_opened() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    std::fs::create_dir(temp_dir.path().join("companion_database.db"))
        .expect("failed to create directory standing in for the database file");

    let output = Command::new(env!("CARGO_BIN_EXE_ai-companion"))
        .current_dir(temp_dir.path())
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
        stderr.contains("companion_database.db"),
        "stderr did not name the database path: {stderr}"
    );
    assert!(
        stderr.contains("unable to open database file"),
        "stderr did not include the sqlite cause: {stderr}"
    );
}

/// `longterm_memory` as a regular file (rather than a directory) makes
/// tantivy fail to open its index. No database file is present, so
/// `Database::init()` succeeds first and the failure is isolated to the
/// long-term memory step.
#[test]
fn exits_non_zero_when_the_long_term_memory_dir_is_a_file() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    std::fs::write(temp_dir.path().join("longterm_memory"), b"not a directory")
        .expect("failed to create a file standing in for the longterm_memory directory");

    let output = Command::new(env!("CARGO_BIN_EXE_ai-companion"))
        .current_dir(temp_dir.path())
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
        stderr.contains("longterm_memory"),
        "stderr did not name the longterm_memory path: {stderr}"
    );
}
