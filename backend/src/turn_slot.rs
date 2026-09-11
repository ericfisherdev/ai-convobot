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

/// A joiner's own auto-extraction (#186) serialises against itself only: at
/// most one extraction in flight per joiner. It must never contend with
/// [`ACTIVE_TURN`], which a reply claims for the duration of its own
/// `GenerateRequest` — extraction runs on `llm::ResidentExtractor` and
/// `compaction::commit::commit`, neither of which touches the state
/// `ACTIVE_TURN` protects, and both are already serialised against a
/// concurrent reply by `llm::GENERATION_LOCK`. Claiming `ACTIVE_TURN` for
/// extraction instead would fail every `GenerateRequest` that lands while an
/// extraction (a model-bound extract + merge) is still running, not just the
/// one frame that queued it.
pub static JOINER_EXTRACTION: TurnSlot = TurnSlot::new();

/// Serialises every test in the crate that claims [`ACTIVE_TURN`] directly,
/// against every other such test.
///
/// `ACTIVE_TURN` is a process-wide static by design (it mirrors production,
/// where one process is ever only one joiner), so `cargo test`'s default
/// parallelism can otherwise make two unrelated tests' real generations
/// contend for the same slot: `remote_generation.rs`'s own
/// `local_model_generation_claims_and_releases_the_shared_turn_slot` and
/// every real host-and-joiner test in `multiplayer::two_instance_tests`
/// must all acquire this lock before touching `ACTIVE_TURN`, not a lock
/// private to either file, or the two can still race each other.
///
/// A `tokio::sync::Mutex`, not `std::sync::Mutex`: the async
/// `#[actix_web::test]`s in `two_instance_tests.rs` hold the guard across
/// several `.await` points (a whole real round), which clippy's
/// `await_holding_lock` correctly refuses for a std lock. The plain,
/// synchronous `#[test]` in `remote_generation.rs` uses
/// [`tokio::sync::Mutex::blocking_lock`] instead, which is safe there
/// precisely because that test runs with no tokio runtime of its own.
#[cfg(test)]
pub(crate) static ACTIVE_TURN_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

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

    #[test]
    fn guard_is_released_when_the_worker_thread_panics() {
        static SLOT: TurnSlot = TurnSlot::new();

        let guard = SLOT.try_claim().expect("slot should be free");
        let handle = std::thread::spawn(move || {
            let _guard = guard;
            panic!("simulated worker panic");
        });

        handle
            .join()
            .expect_err("worker thread should have panicked");

        assert!(
            SLOT.try_claim().is_some(),
            "slot should be released even though the worker thread panicked"
        );
    }
}
