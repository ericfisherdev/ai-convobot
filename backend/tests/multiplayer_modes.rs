//! Process-level test for #136: two real `ai-companion` binaries, one
//! configured as a `host` and one as a `joiner`, covering `main()`'s own
//! multiplayer startup wiring (`JoinerIdentity::from_config`, spawning
//! `multiplayer::joiner::run`, registering the `/api/multiplayer/*` routes)
//! end to end with no model loaded.
//!
//! `backend/src/multiplayer/two_instance_tests.rs` covers the round
//! orchestrator itself (`run_round` over a real socket) at the in-crate
//! level, with a fake `TurnStore`; `tests/multiplayer_round.rs` covers a
//! round against one real host process with a scripted WS client standing
//! in for the joiner. Neither exercises a *second real binary* actually
//! starting up in `Joiner` mode, which is what this file is for.

use std::net::SocketAddr;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

mod common;
use common::{spawn_on_a_free_port, wait_until_listening, ChildGuard};

/// Builds (without spawning) the command for one instance on `port` against
/// `data_dir`, so both the initial, race-safe spawn (via
/// [`spawn_on_a_free_port`]) and a later respawn on the same, already-secured
/// port can share it.
fn instance_command(port: u16, data_dir: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ai-companion"));
    command
        .env("COMPANION_HOST", "127.0.0.1")
        .env("COMPANION_PORT", port.to_string())
        .env("COMPANION_DATA_DIR", data_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

/// Respawns an instance on a port already secured by an earlier
/// [`spawn_on_a_free_port`] call (used after killing the previous process
/// to pick up a saved `multiplayer_mode` change) — no fresh port allocation
/// here, so no new race to lose.
fn respawn_instance(port: u16, addr: SocketAddr, data_dir: &Path) -> ChildGuard {
    let child = instance_command(port, data_dir)
        .spawn()
        .expect("failed to spawn the ai-companion binary");
    wait_until_listening(addr, Duration::from_secs(10));
    ChildGuard(child)
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

fn status_of(agent: &ureq::Agent, url: &str) -> u16 {
    agent
        .get(url)
        .call()
        .unwrap_or_else(|e| panic!("GET {url} failed at the transport level: {e}"))
        .status()
        .as_u16()
}

fn put_config(agent: &ureq::Agent, addr: SocketAddr, config: &Value) -> u16 {
    agent
        .put(format!("http://{addr}/api/config"))
        .send_json(config)
        .unwrap_or_else(|e| panic!("PUT /api/config on {addr} failed at the transport level: {e}"))
        .status()
        .as_u16()
}

/// Polls `check` every 100ms until it returns `true` or `deadline` elapses.
fn wait_until(deadline: Duration, mut check: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    loop {
        if check() {
            return true;
        }
        if start.elapsed() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn a_joiner_appears_in_the_hosts_participant_list_and_disappears_when_it_leaves() {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();

    let host_data = tempfile::tempdir().expect("failed to create the host data-dir temp dir");
    let joiner_data = tempfile::tempdir().expect("failed to create the joiner data-dir temp dir");

    // `PUT /api/config` saves a `multiplayer_mode` change but it only takes
    // effect on the next restart (`main.rs`'s `config_post`: the joiner's
    // identity and the host's `app_data` are both built once at startup).
    // Each instance is spawned once in its default `solo` role just to save
    // its new role, then killed (the block's guard drops at its end) and
    // respawned on the same, already-secured port below.
    let password = "test-multiplayer-secret";

    let (host_port, host_addr, _initial) = spawn_on_a_free_port(
        |port| instance_command(port, host_data.path()),
        Duration::from_secs(10),
    );
    {
        let mut host_config = get_json(&agent, &format!("http://{host_addr}/api/config"));
        host_config["multiplayer_mode"] = json!("host");
        host_config["multiplayer_password"] = json!(password);
        assert_eq!(put_config(&agent, host_addr, &host_config), 200);
    }
    drop(_initial);
    let host_guard = respawn_instance(host_port, host_addr, host_data.path());

    let (joiner_port, joiner_addr, _initial) = spawn_on_a_free_port(
        |port| instance_command(port, joiner_data.path()),
        Duration::from_secs(10),
    );
    {
        let mut joiner_config = get_json(&agent, &format!("http://{joiner_addr}/api/config"));
        joiner_config["multiplayer_mode"] = json!("joiner");
        joiner_config["multiplayer_host_address"] = json!(format!("127.0.0.1:{host_port}"));
        joiner_config["multiplayer_participant_id"] = json!("bot1");
        joiner_config["multiplayer_password"] = json!(password);
        assert_eq!(put_config(&agent, joiner_addr, &joiner_config), 200);
    }
    drop(_initial);
    let joiner_guard = respawn_instance(joiner_port, joiner_addr, joiner_data.path());

    // The joiner connects on its own at startup; poll the host until it
    // shows up connected, with the companion name a fresh joiner instance
    // is seeded with ("Assistant", `database.rs`'s default `companion` row).
    let participants_url = format!("http://{host_addr}/api/multiplayer/participants");
    let joined = wait_until(Duration::from_secs(10), || {
        let participants = get_json(&agent, &participants_url);
        participants
            .as_array()
            .map(|rows| {
                rows.iter().any(|p| {
                    p["id"] == "bot1" && p["connected"] == true && p["display_name"] == "Assistant"
                })
            })
            .unwrap_or(false)
    });
    assert!(
        joined,
        "bot1 never appeared connected in the host's participant list"
    );

    let status = get_json(
        &agent,
        &format!("http://{joiner_addr}/api/multiplayer/status"),
    );
    assert_eq!(status["mode"], "joiner");
    assert_eq!(status["state"], "connected");

    // The joiner is not in `host` mode, so its own `/api/multiplayer/ws`
    // never upgrades.
    assert_eq!(
        status_of(&agent, &format!("http://{joiner_addr}/api/multiplayer/ws")),
        404
    );

    // No avatar was uploaded as part of the join.
    assert_eq!(
        status_of(
            &agent,
            &format!("http://{host_addr}/api/multiplayer/participants/bot1/avatar")
        ),
        404
    );

    // Killing the joiner process closes its socket immediately; the host
    // notices the close and drops it from the participant list.
    drop(joiner_guard);
    let left = wait_until(Duration::from_secs(10), || {
        let participants = get_json(&agent, &participants_url);
        participants
            .as_array()
            .map(|rows| {
                !rows
                    .iter()
                    .any(|p| p["id"] == "bot1" && p["connected"] == true)
            })
            .unwrap_or(true)
    });
    assert!(
        left,
        "bot1 was still listed as connected after its process was killed"
    );

    drop(host_guard);
}
