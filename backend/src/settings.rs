//! Server configuration read from environment variables (#107).
//!
//! `main()` used to hardcode `0.0.0.0:3000` and resolve every data path
//! against the process's working directory, which the Docker images'
//! `COMPANION_HOST`/`COMPANION_PORT` variables and the `/app/data` volume
//! could never actually change. `from_env` is the single place that reads
//! `COMPANION_HOST`, `COMPANION_PORT`, and `COMPANION_DATA_DIR`; `paths.rs`
//! turns the resulting `data_dir` into concrete file paths.

use std::fmt;

/// Server configuration derived from `COMPANION_HOST`, `COMPANION_PORT`, and
/// `COMPANION_DATA_DIR`. `data_dir` is not yet resolved against the working
/// directory; `main()` does that before calling `paths::init`.
#[derive(Debug)]
pub struct Settings {
    pub host: String,
    pub port: u16,
    pub data_dir: std::path::PathBuf,
}

/// The one way `from_env`/`parse` can fail: `COMPANION_PORT` was set to a
/// value that is not a `u16` in `1..=65535`. Every other input has a sane
/// default, so this is the only startup condition `main` needs to name to
/// the user before exiting non-zero.
#[derive(Debug, PartialEq, Eq)]
pub struct SettingsError {
    value: String,
    reason: &'static str,
}

impl fmt::Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "COMPANION_PORT must be an integer in 1..=65535 ({}, got {:?})",
            self.reason, self.value
        )
    }
}

impl std::error::Error for SettingsError {}

/// Reads `COMPANION_HOST`, `COMPANION_PORT`, and `COMPANION_DATA_DIR` from
/// the process environment and validates them.
///
/// # Errors
///
/// Returns [`SettingsError`] when `COMPANION_PORT` is set to a value that
/// does not parse as a `u16` or that parses to `0` (not a valid TCP port).
/// `COMPANION_HOST` and `COMPANION_DATA_DIR` never fail validation: any
/// non-empty string is accepted and left for `HttpServer::bind` /
/// `fs::create_dir_all` to reject if it turns out to be unusable.
pub fn from_env() -> Result<Settings, SettingsError> {
    parse(
        std::env::var("COMPANION_HOST").ok(),
        std::env::var("COMPANION_PORT").ok(),
        std::env::var("COMPANION_DATA_DIR").ok(),
    )
}

/// A trimmed-empty value is treated the same as an unset variable, so
/// `COMPANION_PORT=""` (e.g. from an empty compose interpolation) falls back
/// to the default instead of failing to parse as a `u16`.
fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.trim().is_empty())
}

/// Pure env-parsing logic, split out from `from_env` so unit tests call it
/// directly instead of mutating `std::env` (which races other tests running
/// in parallel on the same process-wide state, as the comment on
/// `configured_workers_reads_the_env_var` notes).
fn parse(
    host: Option<String>,
    port: Option<String>,
    data_dir: Option<String>,
) -> Result<Settings, SettingsError> {
    let host = non_empty(host).unwrap_or_else(|| "0.0.0.0".to_string());

    let port = match non_empty(port) {
        None => 3000,
        Some(value) => {
            let trimmed = value.trim();
            let parsed: u16 = trimmed.parse().map_err(|_| SettingsError {
                value: value.clone(),
                reason: "not an integer",
            })?;
            if parsed == 0 {
                return Err(SettingsError {
                    value,
                    reason: "0 is not a valid port",
                });
            }
            parsed
        }
    };

    let data_dir = non_empty(data_dir)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    Ok(Settings {
        host,
        port,
        data_dir,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_unset_gives_the_documented_defaults() {
        let settings = parse(None, None, None).expect("defaults should always parse");
        assert_eq!(settings.host, "0.0.0.0");
        assert_eq!(settings.port, 3000);
        assert_eq!(settings.data_dir, std::path::PathBuf::from("."));
    }

    #[test]
    fn host_and_data_dir_round_trip() {
        let settings = parse(
            Some("127.0.0.1".to_string()),
            None,
            Some("/tmp/x".to_string()),
        )
        .expect("valid input should parse");
        assert_eq!(settings.host, "127.0.0.1");
        assert_eq!(settings.data_dir, std::path::PathBuf::from("/tmp/x"));
    }

    #[test]
    fn a_valid_port_string_parses() {
        let settings =
            parse(None, Some("3100".to_string()), None).expect("a valid port string should parse");
        assert_eq!(settings.port, 3100);
    }

    #[test]
    fn a_non_numeric_port_is_an_error() {
        let err = parse(None, Some("abc".to_string()), None)
            .expect_err("a non-numeric port must be rejected");
        assert!(err.to_string().contains("COMPANION_PORT"));
        assert!(err.to_string().contains("abc"));
    }

    #[test]
    fn port_zero_is_an_error() {
        parse(None, Some("0".to_string()), None).expect_err("port 0 must be rejected");
    }

    #[test]
    fn a_port_above_u16_range_is_an_error() {
        parse(None, Some("70000".to_string()), None)
            .expect_err("a port above 65535 must be rejected");
    }

    #[test]
    fn a_blank_port_is_treated_as_unset() {
        let settings =
            parse(None, Some("   ".to_string()), None).expect("a blank port should fall back");
        assert_eq!(settings.port, 3000);
    }

    #[test]
    fn a_blank_host_is_treated_as_unset() {
        let settings =
            parse(Some("  ".to_string()), None, None).expect("a blank host should fall back");
        assert_eq!(settings.host, "0.0.0.0");
    }

    #[test]
    fn a_blank_data_dir_is_treated_as_unset() {
        let settings =
            parse(None, None, Some(" ".to_string())).expect("a blank data dir should fall back");
        assert_eq!(settings.data_dir, std::path::PathBuf::from("."));
    }
}
