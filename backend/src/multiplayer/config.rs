//! The multiplayer network-role config: `solo` / `host` / `joiner`, the
//! host's shared password, and the joiner's target host, participant id and
//! password.
//!
//! Pure and DB-free, unit-tested on its own (like `participants.rs`).
//! `database.rs` is the only caller: `MultiplayerConfig::parse` is what
//! `Database::write_config` (#128) runs before persisting a `PUT
//! /api/config` request, and `MultiplayerMode` is what the `config` table's
//! `multiplayer_mode` column round-trips through.

use rusqlite::types::{FromSql, FromSqlError, ToSqlOutput, ValueRef};
use std::fmt;
use std::str::FromStr;

use crate::participants::ParticipantId;

/// The instance's role in a multiplayer chat. `Solo` (the default) never
/// talks to another instance; `Host` accepts joiner connections; `Joiner`
/// connects out to a host at startup (#130).
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MultiplayerMode {
    #[default]
    Solo,
    Host,
    Joiner,
}

impl fmt::Display for MultiplayerMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MultiplayerMode::Solo => "solo",
            MultiplayerMode::Host => "host",
            MultiplayerMode::Joiner => "joiner",
        };
        write!(f, "{}", s)
    }
}

impl FromStr for MultiplayerMode {
    type Err = MultiplayerConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "solo" => Ok(MultiplayerMode::Solo),
            "host" => Ok(MultiplayerMode::Host),
            "joiner" => Ok(MultiplayerMode::Joiner),
            _ => Err(MultiplayerConfigError::UnknownMode(s.to_string())),
        }
    }
}

impl FromSql for MultiplayerMode {
    fn column_result(value: ValueRef<'_>) -> Result<Self, FromSqlError> {
        match value {
            ValueRef::Text(i) => match std::str::from_utf8(i) {
                Ok(s) => match s {
                    "solo" => Ok(MultiplayerMode::Solo),
                    "host" => Ok(MultiplayerMode::Host),
                    "joiner" => Ok(MultiplayerMode::Joiner),
                    _ => Err(FromSqlError::OutOfRange(0)),
                },
                Err(e) => Err(FromSqlError::Other(Box::new(e))),
            },
            _ => Err(FromSqlError::InvalidType),
        }
    }
}

impl rusqlite::ToSql for MultiplayerMode {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.to_string()))
    }
}

/// A validated `host:port` string, e.g. `192.168.0.20:3000`. What #130
/// interpolates into `ws://{host_address}/api/multiplayer/ws`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostAddress(String);

impl HostAddress {
    /// Trims `value`, then requires: no scheme (`://`), no path (`/`), and a
    /// trailing `:port` where `port` parses as a `u16` in `1..=65535`.
    ///
    /// # Errors
    /// See [`HostAddressError`] for which rule produces which variant.
    pub fn parse(value: &str) -> Result<Self, HostAddressError> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(HostAddressError::Empty);
        }
        if trimmed.contains("://") {
            return Err(HostAddressError::HasScheme);
        }
        if trimmed.contains('/') {
            return Err(HostAddressError::HasPath);
        }
        let (host, port_str) = trimmed
            .rsplit_once(':')
            .ok_or(HostAddressError::MissingPort)?;
        if host.is_empty() {
            return Err(HostAddressError::MissingPort);
        }
        let port: u16 = port_str
            .parse()
            .map_err(|_| HostAddressError::InvalidPort(port_str.to_string()))?;
        if port == 0 {
            return Err(HostAddressError::InvalidPort(port_str.to_string()));
        }
        Ok(HostAddress(trimmed.to_string()))
    }

    #[allow(dead_code)] // wired up by #130: ws://{host_address}/api/multiplayer/ws
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Everything that can go wrong parsing a [`HostAddress`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostAddressError {
    Empty,
    HasScheme,
    HasPath,
    MissingPort,
    InvalidPort(String),
}

impl fmt::Display for HostAddressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostAddressError::Empty => write!(f, "host address must not be empty"),
            HostAddressError::HasScheme => {
                write!(
                    f,
                    "host address must not include a scheme (e.g. \"http://\")"
                )
            }
            HostAddressError::HasPath => write!(f, "host address must not include a path"),
            HostAddressError::MissingPort => {
                write!(f, "host address must include a trailing :port")
            }
            HostAddressError::InvalidPort(port) => {
                write!(f, "{:?} is not a valid port (expected 1-65535)", port)
            }
        }
    }
}

impl std::error::Error for HostAddressError {}

/// The full multiplayer config: role plus the fields each role needs.
///
/// `Joiner` requires both `host_address` and `participant_id`; `Solo` and
/// `Host` only validate those two fields when the caller supplied a
/// non-empty string (the user may fill them in before switching to
/// `joiner`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiplayerConfig {
    pub mode: MultiplayerMode,
    pub host_address: Option<HostAddress>,
    pub participant_id: Option<ParticipantId>,
    pub mention_followup_depth: u8,
    pub remote_generation_timeout_secs: u64,
}

impl MultiplayerConfig {
    /// Validates and assembles a [`MultiplayerConfig`] from the raw strings
    /// `PUT /api/config` supplies.
    ///
    /// # Errors
    /// - [`MultiplayerConfigError::UnknownMode`] if `mode` is not `"solo"`,
    ///   `"host"`, or `"joiner"`.
    /// - [`MultiplayerConfigError::MissingHostAddress`] /
    ///   [`MultiplayerConfigError::InvalidHostAddress`] if `mode` is
    ///   `"joiner"` and `host_address` is empty or fails
    ///   [`HostAddress::parse`]; a non-`"joiner"` mode only returns
    ///   `InvalidHostAddress` when `host_address` is non-empty and invalid.
    /// - [`MultiplayerConfigError::MissingParticipantId`] /
    ///   [`MultiplayerConfigError::InvalidParticipantId`]: the same rule,
    ///   for `participant_id` against [`ParticipantId::parse`].
    /// - [`MultiplayerConfigError::MentionDepthOutOfRange`] if
    ///   `mention_followup_depth` is not in `0..=10`.
    /// - [`MultiplayerConfigError::TimeoutOutOfRange`] if
    ///   `remote_generation_timeout_secs` is not in `5..=3600`.
    pub fn parse(
        mode: &str,
        host_address: &str,
        participant_id: &str,
        mention_followup_depth: u8,
        remote_generation_timeout_secs: u64,
    ) -> Result<Self, MultiplayerConfigError> {
        let mode: MultiplayerMode = mode.parse()?;

        if mention_followup_depth > 10 {
            return Err(MultiplayerConfigError::MentionDepthOutOfRange(
                mention_followup_depth,
            ));
        }
        if !(5..=3600).contains(&remote_generation_timeout_secs) {
            return Err(MultiplayerConfigError::TimeoutOutOfRange(
                remote_generation_timeout_secs,
            ));
        }

        let host_address = Self::parse_host_address(mode, host_address)?;
        let participant_id = Self::parse_participant_id(mode, participant_id)?;

        Ok(MultiplayerConfig {
            mode,
            host_address,
            participant_id,
            mention_followup_depth,
            remote_generation_timeout_secs,
        })
    }

    fn parse_host_address(
        mode: MultiplayerMode,
        host_address: &str,
    ) -> Result<Option<HostAddress>, MultiplayerConfigError> {
        let trimmed = host_address.trim();
        if trimmed.is_empty() {
            return if mode == MultiplayerMode::Joiner {
                Err(MultiplayerConfigError::MissingHostAddress)
            } else {
                Ok(None)
            };
        }
        HostAddress::parse(trimmed)
            .map(Some)
            .map_err(|e| MultiplayerConfigError::InvalidHostAddress(e.to_string()))
    }

    fn parse_participant_id(
        mode: MultiplayerMode,
        participant_id: &str,
    ) -> Result<Option<ParticipantId>, MultiplayerConfigError> {
        let trimmed = participant_id.trim();
        if trimmed.is_empty() {
            return if mode == MultiplayerMode::Joiner {
                Err(MultiplayerConfigError::MissingParticipantId)
            } else {
                Ok(None)
            };
        }
        ParticipantId::parse(trimmed)
            .map(Some)
            .map_err(|_| MultiplayerConfigError::InvalidParticipantId(trimmed.to_string()))
    }
}

/// Everything that can go wrong assembling a [`MultiplayerConfig`]. The repo
/// convention (documented on [`MultiplayerConfig::parse`]) is to name, per
/// method, which rule returns which variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiplayerConfigError {
    UnknownMode(String),
    MissingHostAddress,
    InvalidHostAddress(String),
    MissingParticipantId,
    /// The pattern text in the message is asserted on verbatim by the
    /// acceptance test for issue #128: "a-z, 0-9 and _, starting with a
    /// letter, at most 16 characters".
    InvalidParticipantId(String),
    MentionDepthOutOfRange(u8),
    TimeoutOutOfRange(u64),
}

impl fmt::Display for MultiplayerConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiplayerConfigError::UnknownMode(mode) => write!(
                f,
                "{:?} is not a valid multiplayer mode (expected \"solo\", \"host\", or \"joiner\")",
                mode
            ),
            MultiplayerConfigError::MissingHostAddress => {
                write!(f, "joiner mode requires a host address")
            }
            MultiplayerConfigError::InvalidHostAddress(reason) => {
                write!(f, "invalid host address: {}", reason)
            }
            MultiplayerConfigError::MissingParticipantId => {
                write!(f, "joiner mode requires a participant id")
            }
            MultiplayerConfigError::InvalidParticipantId(value) => write!(
                f,
                "{:?} is not a valid participant id (expected a-z, 0-9 and _, starting with a letter, at most 16 characters)",
                value
            ),
            MultiplayerConfigError::MentionDepthOutOfRange(depth) => write!(
                f,
                "mention_followup_depth must be between 0 and 10 (got {})",
                depth
            ),
            MultiplayerConfigError::TimeoutOutOfRange(secs) => write!(
                f,
                "remote_generation_timeout_secs must be between 5 and 3600 (got {})",
                secs
            ),
        }
    }
}

impl std::error::Error for MultiplayerConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    // --- MultiplayerMode ---

    #[test]
    fn mode_round_trips_through_display_and_from_str() {
        for mode in [
            MultiplayerMode::Solo,
            MultiplayerMode::Host,
            MultiplayerMode::Joiner,
        ] {
            assert_eq!(mode.to_string().parse::<MultiplayerMode>().unwrap(), mode);
        }
    }

    #[test]
    fn mode_from_str_rejects_unknown_strings() {
        assert_eq!(
            "Solo".parse::<MultiplayerMode>().unwrap_err(),
            MultiplayerConfigError::UnknownMode("Solo".to_string())
        );
    }

    #[test]
    fn mode_to_sql_produces_the_lowercase_text() {
        for (mode, text) in [
            (MultiplayerMode::Solo, "solo"),
            (MultiplayerMode::Host, "host"),
            (MultiplayerMode::Joiner, "joiner"),
        ] {
            match rusqlite::ToSql::to_sql(&mode).unwrap() {
                ToSqlOutput::Owned(rusqlite::types::Value::Text(s)) => assert_eq!(s, text),
                other => panic!("unexpected ToSqlOutput: {:?}", other),
            }
        }
    }

    #[test]
    fn mode_from_sql_round_trips_known_text() {
        for (text, mode) in [
            ("solo", MultiplayerMode::Solo),
            ("host", MultiplayerMode::Host),
            ("joiner", MultiplayerMode::Joiner),
        ] {
            assert_eq!(
                MultiplayerMode::column_result(ValueRef::Text(text.as_bytes())).unwrap(),
                mode
            );
        }
    }

    #[test]
    fn mode_from_sql_rejects_unknown_text() {
        let err = MultiplayerMode::column_result(ValueRef::Text(b"bogus")).unwrap_err();
        assert!(matches!(err, FromSqlError::OutOfRange(0)));
    }

    #[test]
    fn mode_default_is_solo() {
        assert_eq!(MultiplayerMode::default(), MultiplayerMode::Solo);
    }

    // --- HostAddress ---

    #[test]
    fn host_address_accepts_plain_host_port() {
        assert_eq!(
            HostAddress::parse("192.168.0.20:3000").unwrap().as_str(),
            "192.168.0.20:3000"
        );
        assert_eq!(
            HostAddress::parse("host:3000").unwrap().as_str(),
            "host:3000"
        );
    }

    #[test]
    fn host_address_rejects_empty() {
        assert_eq!(HostAddress::parse("").unwrap_err(), HostAddressError::Empty);
        assert_eq!(
            HostAddress::parse("   ").unwrap_err(),
            HostAddressError::Empty
        );
    }

    #[test]
    fn host_address_rejects_a_scheme() {
        assert_eq!(
            HostAddress::parse("http://x:1").unwrap_err(),
            HostAddressError::HasScheme
        );
    }

    #[test]
    fn host_address_rejects_a_zero_port() {
        assert_eq!(
            HostAddress::parse("x:0").unwrap_err(),
            HostAddressError::InvalidPort("0".to_string())
        );
    }

    #[test]
    fn host_address_rejects_an_out_of_range_port() {
        assert_eq!(
            HostAddress::parse("x:70000").unwrap_err(),
            HostAddressError::InvalidPort("70000".to_string())
        );
    }

    #[test]
    fn host_address_rejects_a_missing_port() {
        assert_eq!(
            HostAddress::parse("x").unwrap_err(),
            HostAddressError::MissingPort
        );
    }

    #[test]
    fn host_address_rejects_a_path() {
        assert_eq!(
            HostAddress::parse("host:3000/api").unwrap_err(),
            HostAddressError::HasPath
        );
    }

    // --- MultiplayerConfig::parse ---

    #[test]
    fn parse_rejects_joiner_with_empty_host() {
        assert_eq!(
            MultiplayerConfig::parse("joiner", "", "bot1", 1, 120).unwrap_err(),
            MultiplayerConfigError::MissingHostAddress
        );
    }

    #[test]
    fn parse_rejects_joiner_with_empty_id() {
        assert_eq!(
            MultiplayerConfig::parse("joiner", "host:3000", "", 1, 120).unwrap_err(),
            MultiplayerConfigError::MissingParticipantId
        );
    }

    #[test]
    fn parse_rejects_an_invalid_id_naming_the_pattern() {
        let err = MultiplayerConfig::parse("joiner", "host:3000", "Bot-1", 1, 120).unwrap_err();
        assert!(err
            .to_string()
            .contains("a-z, 0-9 and _, starting with a letter, at most 16 characters"));
    }

    #[test]
    fn parse_accepts_solo_with_empty_strings() {
        let config = MultiplayerConfig::parse("solo", "", "", 1, 120).unwrap();
        assert_eq!(config.mode, MultiplayerMode::Solo);
        assert_eq!(config.host_address, None);
        assert_eq!(config.participant_id, None);
    }

    #[test]
    fn parse_rejects_depth_above_ten() {
        assert_eq!(
            MultiplayerConfig::parse("solo", "", "", 11, 120).unwrap_err(),
            MultiplayerConfigError::MentionDepthOutOfRange(11)
        );
    }

    #[test]
    fn parse_rejects_timeout_below_five() {
        assert_eq!(
            MultiplayerConfig::parse("solo", "", "", 1, 4).unwrap_err(),
            MultiplayerConfigError::TimeoutOutOfRange(4)
        );
    }
}
