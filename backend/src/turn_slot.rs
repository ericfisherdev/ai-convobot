use std::sync::atomic::{AtomicBool, Ordering};

/// Serialises a whole chat turn end to end: from the moment a handler inserts
/// the user's message up to the moment the reply (and its attitude update)
/// are persisted.
///
/// `llm::GENERATION_LOCK` only serialises the model itself, taken on the
/// worker thread once generation actually starts. That leaves a window
/// between a handler inserting the user message and the worker thread
/// reading history back out: a second request's user message can land in
/// that window and get answered by the first request's reply (or worse,
/// deleted out from under a regenerate). `TurnSlot` closes that window by
/// being claimed by the handler *before* the user-turn insert and released
/// by the worker only after the reply is fully persisted.
///
/// This is deliberately not a queue: a caller that finds the slot already
/// claimed gets turned away (HTTP 409) rather than piling up behind a lock,
/// so a burst of sends doesn't leave requests waiting indefinitely.
pub struct TurnSlot(AtomicBool);

impl TurnSlot {
    pub const fn new() -> Self {
        Self(AtomicBool::new(false))
    }

    /// Attempts to claim the slot for a new turn.
    ///
    /// Returns `None` if a turn is already in flight. Returns
    /// `Some(TurnGuard)` on success; the guard releases the slot on drop,
    /// including when the holder's thread panics, so a crashing worker
    /// cannot wedge the server into permanently refusing new turns.
    pub fn try_claim(&'static self) -> Option<TurnGuard> {
        self.0
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| TurnGuard { slot: self })
    }
}

/// Holds a claim on a [`TurnSlot`]. Releases it on drop.
///
/// A unit struct over a `&'static TurnSlot` is `Send`, so unlike a
/// `MutexGuard` it can be moved into a `std::thread::spawn` closure and
/// dropped there once the worker thread finishes the turn.
pub struct TurnGuard {
    slot: &'static TurnSlot,
}

impl Drop for TurnGuard {
    fn drop(&mut self) {
        self.slot.0.store(false, Ordering::Release);
    }
}

/// The process-wide turn slot used by the prompting endpoints.
pub static ACTIVE_TURN: TurnSlot = TurnSlot::new();

#[cfg(test)]
mod tests {
    use super::TurnSlot;

    #[test]
    fn second_claim_fails_until_the_first_is_dropped() {
        static SLOT: TurnSlot = TurnSlot::new();

        let first = SLOT.try_claim().expect("slot should be free");
        assert!(SLOT.try_claim().is_none(), "slot is already claimed");

        drop(first);

        assert!(
            SLOT.try_claim().is_some(),
            "slot should be free again after the guard drops"
        );
    }

    #[test]
    fn guard_is_send_and_releases_when_the_worker_thread_finishes() {
        static SLOT: TurnSlot = TurnSlot::new();

        let guard = SLOT.try_claim().expect("slot should be free");
        let handle = std::thread::spawn(move || {
            let _guard = guard;
        });
        handle.join().expect("worker thread should not panic");

        assert!(
            SLOT.try_claim().is_some(),
            "slot should be released once the worker thread drops its guard"
        );
    }
}
