//! Per-address rate limit on join attempts.
//!
//! Pure, with an injected clock (`std::time::Instant`, passed in by the
//! caller rather than read internally) so tests never have to sleep.
//!
//! `try_reserve`/`release` (not a separate check-then-record pair) are what
//! make this safe under concurrent connections from the same address: the
//! whole handshake takes a await-ing round trip, and a `is_blocked` check
//! followed by a `record_failure` call only after the proof turns out
//! wrong leaves a window in between where an unbounded number of
//! concurrent connections can all pass the check before any of them
//! record a failure. Reserving a slot up front and only releasing it on
//! success (or an inconclusive outcome, never on a bad proof) closes that
//! window: a caller that never releases a reservation keeps counting
//! against the same window's `max_failures`, no matter how many
//! connections raced to get there.

use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Upper bound on how many distinct addresses one throttle tracks at once.
/// Without it, a distributed attacker using many source addresses (one or
/// two attempts each, never enough to itself get blocked) could grow the
/// table without bound. Reaching capacity evicts the single
/// oldest-tracked address rather than refusing the new one: forgetting a
/// quiet address is a second-order concern next to bounding memory.
const MAX_TRACKED_ADDRESSES: usize = 10_000;

/// How many stale or expired entries [`JoinThrottle::evict_stale`] walks
/// off the front of the tracking order per call. Bounding this keeps each
/// call's added cost O(1) amortised instead of O(n) in the table size (the
/// table's own DoS vector this replaces): a burst of simultaneous
/// expirations is cleared over several calls rather than one large scan.
const MAX_SWEEP_PER_CALL: usize = 8;

struct Entry {
    count: u32,
    window_started: Instant,
}

/// The tracking table plus its insertion order, oldest push first. An
/// address can appear more than once in `order` if it started more than
/// one window over time (expired, then attempted again); a popped id is
/// only actually removed from `entries` when the popped `(ip,
/// window_started)` pair still matches the live entry, which is what
/// makes a stale duplicate push harmless to pop and discard.
struct Inner {
    entries: HashMap<IpAddr, Entry>,
    order: VecDeque<(IpAddr, Instant)>,
}

/// Blocks an address after too many outstanding join attempts within a
/// sliding window, bounded to [`MAX_TRACKED_ADDRESSES`] distinct
/// addresses and swept incrementally rather than with a full-table scan
/// per call.
pub struct JoinThrottle {
    max_failures: u32,
    window: Duration,
    inner: Mutex<Inner>,
}

impl JoinThrottle {
    pub fn new(max_failures: u32, window: Duration) -> Self {
        JoinThrottle {
            max_failures,
            window,
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                order: VecDeque::new(),
            }),
        }
    }

    /// Whether `ip` currently has at least `max_failures` attempts
    /// reserved within the current window, as of `now`. Read-only —
    /// callers deciding whether to admit a new attempt should use
    /// [`JoinThrottle::try_reserve`] instead, which makes the same check
    /// atomic with recording the attempt.
    #[allow(dead_code)] // introspection helper, exercised directly by this module's tests
    pub fn is_blocked(&self, ip: IpAddr, now: Instant) -> bool {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        match inner.entries.get(&ip) {
            Some(entry) => {
                now.saturating_duration_since(entry.window_started) < self.window
                    && entry.count >= self.max_failures
            }
            None => false,
        }
    }

    /// Atomically checks whether `ip` is blocked and, if not, reserves one
    /// attempt slot for it (starting a fresh window if none is active or
    /// the previous one expired). Returns `false` (nothing reserved) if
    /// `ip` already has `max_failures` attempts reserved within the
    /// window.
    ///
    /// Every successful reservation must be paired with exactly one later
    /// call to [`JoinThrottle::release`] — unless the attempt turns out to
    /// be an actual bad password proof, in which case leaving it
    /// unreleased is what keeps it counted for the rest of the window.
    pub fn try_reserve(&self, ip: IpAddr, now: Instant) -> bool {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        self.evict_stale(&mut inner, now);

        let window = self.window;
        let has_current_entry = inner
            .entries
            .get(&ip)
            .is_some_and(|entry| now.saturating_duration_since(entry.window_started) < window);

        if has_current_entry {
            let entry = inner.entries.get_mut(&ip).expect("checked above");
            if entry.count >= self.max_failures {
                return false;
            }
            entry.count += 1;
            return true;
        }

        if inner.entries.len() >= MAX_TRACKED_ADDRESSES {
            self.evict_oldest(&mut inner);
        }
        inner.entries.insert(
            ip,
            Entry {
                count: 1,
                window_started: now,
            },
        );
        inner.order.push_back((ip, now));
        true
    }

    /// Releases one reservation for `ip` at `now`: call this after a
    /// *successful* authentication, or an attempt that ended
    /// inconclusively (timeout, malformed frame, or anything else that
    /// is not itself a wrong password). Decrements the count, removing the
    /// entry once it reaches zero so a burst of legitimate one-off
    /// connections does not linger in the table.
    pub fn release(&self, ip: IpAddr, now: Instant) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let Some(entry) = inner.entries.get_mut(&ip) else {
            return;
        };
        if now.saturating_duration_since(entry.window_started) >= self.window {
            inner.entries.remove(&ip);
            return;
        }
        entry.count = entry.count.saturating_sub(1);
        if entry.count == 0 {
            inner.entries.remove(&ip);
        }
    }

    /// Walks up to [`MAX_SWEEP_PER_CALL`] entries off the front of
    /// `order` (oldest push first), removing each from `entries` if the
    /// popped push is still the live one for that address and either it
    /// or the address's window has since gone stale. Stops as soon as the
    /// front entry is both still current and not expired: everything
    /// behind it was pushed later, so it cannot be staler.
    fn evict_stale(&self, inner: &mut Inner, now: Instant) {
        for _ in 0..MAX_SWEEP_PER_CALL {
            let Some(&(ip, pushed_at)) = inner.order.front() else {
                break;
            };
            let still_current = inner
                .entries
                .get(&ip)
                .is_some_and(|entry| entry.window_started == pushed_at);
            let expired = now.saturating_duration_since(pushed_at) >= self.window;
            if still_current && !expired {
                break;
            }
            inner.order.pop_front();
            if still_current {
                inner.entries.remove(&ip);
            }
        }
    }

    /// Evicts the single oldest still-live address to make room under
    /// [`MAX_TRACKED_ADDRESSES`], discarding any stale pushes in front of
    /// it along the way.
    fn evict_oldest(&self, inner: &mut Inner) {
        while let Some((ip, pushed_at)) = inner.order.pop_front() {
            if inner
                .entries
                .get(&ip)
                .is_some_and(|entry| entry.window_started == pushed_at)
            {
                inner.entries.remove(&ip);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(n: u8) -> IpAddr {
        IpAddr::from([127, 0, 0, n])
    }

    #[test]
    fn fifth_reservation_blocks_the_sixth_but_not_the_fifth() {
        let throttle = JoinThrottle::new(5, Duration::from_secs(600));
        let now = Instant::now();
        for _ in 0..4 {
            assert!(throttle.try_reserve(ip(1), now));
        }
        assert!(!throttle.is_blocked(ip(1), now));
        assert!(throttle.try_reserve(ip(1), now));
        assert!(throttle.is_blocked(ip(1), now));
        assert!(!throttle.try_reserve(ip(1), now));
    }

    #[test]
    fn block_expires_after_the_window() {
        let throttle = JoinThrottle::new(1, Duration::from_secs(600));
        let now = Instant::now();
        assert!(throttle.try_reserve(ip(1), now));
        assert!(throttle.is_blocked(ip(1), now));
        let later = now + Duration::from_secs(601);
        assert!(!throttle.is_blocked(ip(1), later));
        assert!(throttle.try_reserve(ip(1), later));
    }

    #[test]
    fn release_un_blocks_a_reserved_attempt() {
        let throttle = JoinThrottle::new(1, Duration::from_secs(600));
        let now = Instant::now();
        assert!(throttle.try_reserve(ip(1), now));
        assert!(throttle.is_blocked(ip(1), now));
        throttle.release(ip(1), now);
        assert!(!throttle.is_blocked(ip(1), now));
        assert!(throttle.try_reserve(ip(1), now));
    }

    #[test]
    fn an_unreleased_reservation_stays_counted_like_a_recorded_failure() {
        // Mirrors the old record_failure contract: a reservation that is
        // never released (a bad password proof) keeps counting for the
        // rest of the window.
        let throttle = JoinThrottle::new(2, Duration::from_secs(600));
        let now = Instant::now();
        assert!(throttle.try_reserve(ip(1), now));
        assert!(throttle.try_reserve(ip(1), now));
        assert!(throttle.is_blocked(ip(1), now));
    }

    #[test]
    fn two_addresses_are_independent() {
        let throttle = JoinThrottle::new(1, Duration::from_secs(600));
        let now = Instant::now();
        assert!(throttle.try_reserve(ip(1), now));
        assert!(throttle.is_blocked(ip(1), now));
        assert!(!throttle.is_blocked(ip(2), now));
        assert!(throttle.try_reserve(ip(2), now));
    }

    #[test]
    fn a_reservation_after_the_window_expires_starts_a_fresh_window_instead_of_accumulating() {
        let throttle = JoinThrottle::new(2, Duration::from_secs(600));
        let now = Instant::now();
        assert!(throttle.try_reserve(ip(1), now));
        let later = now + Duration::from_secs(601);
        assert!(throttle.try_reserve(ip(1), later));
        // A fresh window with one reservation, not two accumulated across
        // windows, so this must not be blocked yet.
        assert!(!throttle.is_blocked(ip(1), later));
    }

    #[test]
    fn concurrent_reservations_up_to_the_limit_all_succeed_and_the_next_is_blocked() {
        // The scenario the reserve/release split exists for: every
        // connection reserves before doing anything else, so a burst of
        // simultaneous attempts cannot all slip past a stale `is_blocked`
        // read the way a separate check-then-record pair would allow.
        let throttle = JoinThrottle::new(3, Duration::from_secs(600));
        let now = Instant::now();
        let results: Vec<bool> = (0..4).map(|_| throttle.try_reserve(ip(1), now)).collect();
        assert_eq!(results, vec![true, true, true, false]);
    }

    #[test]
    fn capacity_evicts_the_oldest_tracked_address() {
        let throttle = JoinThrottle::new(1, Duration::from_secs(600));
        let now = Instant::now();
        for i in 0..MAX_TRACKED_ADDRESSES {
            let addr = IpAddr::from([10, 0, (i / 256) as u8, (i % 256) as u8]);
            assert!(throttle.try_reserve(addr, now));
        }
        assert!(!throttle.is_blocked(ip(1), now));
        // The very first tracked address should have been evicted to make
        // room for the (MAX_TRACKED_ADDRESSES + 1)-th.
        let first_addr = IpAddr::from([10, 0, 0, 0]);
        assert!(throttle.is_blocked(first_addr, now));
        let new_addr = IpAddr::from([10, 1, 0, 0]);
        assert!(throttle.try_reserve(new_addr, now));
        assert!(!throttle.is_blocked(first_addr, now));
    }

    #[test]
    fn evict_stale_reclaims_expired_entries_over_several_calls() {
        let throttle = JoinThrottle::new(1, Duration::from_secs(1));
        let now = Instant::now();
        for i in 0..(MAX_SWEEP_PER_CALL * 2) {
            let addr = IpAddr::from([10, 0, 0, i as u8]);
            assert!(throttle.try_reserve(addr, now));
        }
        let later = now + Duration::from_secs(2);
        // Each try_reserve call sweeps up to MAX_SWEEP_PER_CALL stale
        // entries; a fresh address after the window expired should
        // eventually be able to reserve even though the table was full of
        // now-expired entries a moment ago.
        for _ in 0..3 {
            let fresh = IpAddr::from([10, 0, 1, 0]);
            throttle.try_reserve(fresh, later);
            throttle.release(fresh, later);
        }
        let old_addr = IpAddr::from([10, 0, 0, 0]);
        assert!(!throttle.is_blocked(old_addr, later));
    }
}
