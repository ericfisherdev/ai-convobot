//! Per-address rate limit on failed join attempts.
//!
//! Pure, with an injected clock (`std::time::Instant`, passed in by the
//! caller rather than read internally) so tests never have to sleep.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

struct Failures {
    count: u32,
    window_started: Instant,
}

/// Blocks an address after too many failed joins within a sliding window.
///
/// A window that has expired resets the count on the next call for that
/// address (in both [`JoinThrottle::is_blocked`] and
/// [`JoinThrottle::record_failure`]), and an expired entry is dropped from
/// the map in `record_failure`, so the map cannot grow without bound from
/// one repeatedly-retrying address.
pub struct JoinThrottle {
    max_failures: u32,
    window: Duration,
    inner: Mutex<HashMap<IpAddr, Failures>>,
}

impl JoinThrottle {
    pub fn new(max_failures: u32, window: Duration) -> Self {
        JoinThrottle {
            max_failures,
            window,
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Whether `ip` currently has at least `max_failures` failures within
    /// the current window, as of `now`.
    pub fn is_blocked(&self, ip: IpAddr, now: Instant) -> bool {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        match inner.get(&ip) {
            Some(failures) => {
                now.saturating_duration_since(failures.window_started) < self.window
                    && failures.count >= self.max_failures
            }
            None => false,
        }
    }

    /// Records a failed join attempt from `ip` at `now`, starting a fresh
    /// window if none is active or the previous one expired. Also sweeps
    /// every other entry whose window has expired, so a burst of distinct
    /// failing addresses does not leak memory.
    pub fn record_failure(&self, ip: IpAddr, now: Instant) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.retain(|addr, failures| {
            *addr == ip || now.saturating_duration_since(failures.window_started) < self.window
        });
        match inner.get_mut(&ip) {
            Some(failures)
                if now.saturating_duration_since(failures.window_started) < self.window =>
            {
                failures.count += 1;
            }
            _ => {
                inner.insert(
                    ip,
                    Failures {
                        count: 1,
                        window_started: now,
                    },
                );
            }
        }
    }

    /// Clears any recorded failures for `ip`, e.g. after a successful join.
    pub fn clear(&self, ip: IpAddr) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.remove(&ip);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(n: u8) -> IpAddr {
        IpAddr::from([127, 0, 0, n])
    }

    #[test]
    fn fifth_failure_blocks_but_not_the_fourth() {
        let throttle = JoinThrottle::new(5, Duration::from_secs(600));
        let now = Instant::now();
        for _ in 0..4 {
            throttle.record_failure(ip(1), now);
        }
        assert!(!throttle.is_blocked(ip(1), now));
        throttle.record_failure(ip(1), now);
        assert!(throttle.is_blocked(ip(1), now));
    }

    #[test]
    fn block_expires_after_the_window() {
        let throttle = JoinThrottle::new(1, Duration::from_secs(600));
        let now = Instant::now();
        throttle.record_failure(ip(1), now);
        assert!(throttle.is_blocked(ip(1), now));
        let later = now + Duration::from_secs(601);
        assert!(!throttle.is_blocked(ip(1), later));
    }

    #[test]
    fn success_clears_recorded_failures() {
        let throttle = JoinThrottle::new(1, Duration::from_secs(600));
        let now = Instant::now();
        throttle.record_failure(ip(1), now);
        assert!(throttle.is_blocked(ip(1), now));
        throttle.clear(ip(1));
        assert!(!throttle.is_blocked(ip(1), now));
    }

    #[test]
    fn two_addresses_are_independent() {
        let throttle = JoinThrottle::new(1, Duration::from_secs(600));
        let now = Instant::now();
        throttle.record_failure(ip(1), now);
        assert!(throttle.is_blocked(ip(1), now));
        assert!(!throttle.is_blocked(ip(2), now));
    }

    #[test]
    fn a_failure_after_the_window_expires_starts_a_fresh_window_instead_of_accumulating() {
        let throttle = JoinThrottle::new(2, Duration::from_secs(600));
        let now = Instant::now();
        throttle.record_failure(ip(1), now);
        let later = now + Duration::from_secs(601);
        throttle.record_failure(ip(1), later);
        // A fresh window with one failure, not two accumulated across
        // windows, so this must not be blocked yet.
        assert!(!throttle.is_blocked(ip(1), later));
    }
}
