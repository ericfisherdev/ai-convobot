//! The joiner's reconnect backoff: how long `joiner::run` waits before
//! retrying a connection that ended without an explicit `Rejected`.
//!
//! Pure, no I/O, unit-tested on its own (like `handshake.rs`).

use std::time::Duration;

/// Doubles from [`ReconnectBackoff::INITIAL`] up to
/// [`ReconnectBackoff::MAX`] on every [`ReconnectBackoff::delay`] call, and
/// resets to `INITIAL` once a connection succeeds
/// ([`ReconnectBackoff::reset`]). A fresh connection failure right after a
/// long-lived one should not have to wait 30 seconds before its first
/// retry, which is why `run` resets this only on a successful `Joined`, not
/// merely on a successful TCP connect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconnectBackoff {
    next: Duration,
}

impl ReconnectBackoff {
    /// The delay before the first retry.
    pub const INITIAL: Duration = Duration::from_secs(1);
    /// The delay [`ReconnectBackoff::delay`] never exceeds.
    pub const MAX: Duration = Duration::from_secs(30);

    pub fn new() -> Self {
        ReconnectBackoff {
            next: Self::INITIAL,
        }
    }

    /// Returns the delay to wait before the next retry, then doubles the
    /// internal counter (capped at [`ReconnectBackoff::MAX`]) for the retry
    /// after that.
    pub fn delay(&mut self) -> Duration {
        let current = self.next;
        self.next = (self.next * 2).min(Self::MAX);
        current
    }

    /// Returns to [`ReconnectBackoff::INITIAL`]. Called after a connection
    /// reaches `Connected`, so a brief blip does not leave the joiner
    /// waiting up to 30 seconds for its next reconnect.
    pub fn reset(&mut self) {
        self.next = Self::INITIAL;
    }
}

impl Default for ReconnectBackoff {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_doubles_and_caps_at_max() {
        let mut backoff = ReconnectBackoff::new();
        let expected = [1, 2, 4, 8, 16, 30, 30];
        for seconds in expected {
            assert_eq!(backoff.delay(), Duration::from_secs(seconds));
        }
    }

    #[test]
    fn reset_returns_to_initial() {
        let mut backoff = ReconnectBackoff::new();
        backoff.delay();
        backoff.delay();
        backoff.delay();
        backoff.reset();
        assert_eq!(backoff.delay(), ReconnectBackoff::INITIAL);
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(ReconnectBackoff::default(), ReconnectBackoff::new());
    }
}
