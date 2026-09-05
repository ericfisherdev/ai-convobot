//! A single-slot cache that keeps one loaded resource resident across calls,
//! reloading it only when the caller's key changes.
//!
//! Kept generic over `K`/`V` so it is unit-testable without a GGUF file;
//! `llm.rs` instantiates it with `ModelKey` and `LlamaModel`.

use std::sync::{Arc, Mutex};

use crate::database::{ConfigView, Device};

/// The config inputs that decide how a model is loaded.
///
/// Deliberately holds the **inputs** to GPU layer resolution, not the
/// resolved layer count. Keying on the resolved count would thrash: once a
/// model is resident on the GPU, `GpuAllocator::detect_gpu_memory` sees less
/// free VRAM, `calculate_optimal_layers_v2` returns fewer layers, the key
/// changes, the model reloads with fewer layers, free VRAM goes back up, and
/// so on. GPU detection therefore only runs when a (re)load actually
/// happens, not on every turn.
#[derive(Clone, PartialEq)]
pub struct ModelKey {
    pub model_path: String,
    pub device: Device,
    pub gpu_layers: usize,
    pub dynamic_gpu_allocation: bool,
    pub vram_limit_gb: usize,
    pub gpu_safety_margin: f32,
    pub min_free_vram_mb: u64,
}

impl ModelKey {
    pub fn from_config(config: &ConfigView) -> Self {
        Self {
            model_path: config.llm_model_path.clone(),
            device: config.device.clone(),
            gpu_layers: config.gpu_layers,
            dynamic_gpu_allocation: config.dynamic_gpu_allocation,
            vram_limit_gb: config.vram_limit_gb,
            gpu_safety_margin: config.gpu_safety_margin,
            min_free_vram_mb: config.min_free_vram_mb,
        }
    }
}

/// Holds at most one `(key, value)` pair, shared out via `Arc`.
///
/// `load` runs while the slot mutex is held. Generation is already
/// serialised elsewhere (see `GENERATION_LOCK` in `llm.rs`), so holding the
/// lock across a load costs nothing extra and stops `evict` from racing a
/// load in progress.
pub struct ResidentCache<K, V> {
    slot: Mutex<Option<(K, Arc<V>)>>,
}

impl<K, V> ResidentCache<K, V> {
    pub const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }
}

impl<K: Clone + PartialEq, V> ResidentCache<K, V> {
    /// Returns the resident value for `key`, loading it via `load` if the
    /// slot is empty or holds a different key.
    ///
    /// The old value (if any) is dropped before `load` runs, so two values
    /// are never held in memory at once. On a load error the slot is left
    /// empty so the next call retries.
    pub fn get_or_load<E>(
        &self,
        key: K,
        load: impl FnOnce(&K) -> Result<V, E>,
    ) -> Result<(Arc<V>, bool), E> {
        let mut slot = self
            .slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((existing_key, value)) = slot.as_ref() {
            if *existing_key == key {
                return Ok((Arc::clone(value), true));
            }
        }
        // Drop the old value (if any) before loading the new one.
        *slot = None;
        let value = Arc::new(load(&key)?);
        *slot = Some((key, Arc::clone(&value)));
        Ok((value, false))
    }

    /// Clears the slot, returning the evicted key if anything was resident.
    ///
    /// A generation in flight holds its own `Arc` clone, so the underlying
    /// model stays alive until that turn finishes; this only stops it from
    /// being handed out to new turns.
    pub fn evict(&self) -> Option<K> {
        let mut slot = self
            .slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slot.take().map(|(key, _)| key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    fn cache() -> ResidentCache<String, u32> {
        ResidentCache::new()
    }

    #[test]
    fn second_get_with_same_key_does_not_load_again() {
        let cache = cache();
        let counter = AtomicU32::new(0);
        let load = |_: &String| -> Result<u32, ()> {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(42)
        };

        let (first, reused_first) = cache.get_or_load("a".to_string(), load).unwrap();
        let (second, reused_second) = cache.get_or_load("a".to_string(), load).unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert!(!reused_first);
        assert!(reused_second);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn changed_key_loads_exactly_once() {
        let cache = cache();
        let counter = AtomicU32::new(0);
        let load = |_: &String| -> Result<u32, ()> {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(7)
        };

        cache.get_or_load("a".to_string(), load).unwrap();
        cache.get_or_load("a".to_string(), load).unwrap();
        cache.get_or_load("b".to_string(), load).unwrap();
        cache.get_or_load("b".to_string(), load).unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn old_value_is_dropped_before_new_load() {
        struct Guard(&'static AtomicBool);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        static DROPPED: AtomicBool = AtomicBool::new(false);
        let cache: ResidentCache<String, Guard> = ResidentCache::new();

        cache
            .get_or_load("a".to_string(), |_| Ok::<Guard, ()>(Guard(&DROPPED)))
            .unwrap();
        assert!(!DROPPED.load(Ordering::SeqCst));

        cache
            .get_or_load("b".to_string(), |_| {
                assert!(
                    DROPPED.load(Ordering::SeqCst),
                    "old value must be dropped first"
                );
                Ok::<Guard, ()>(Guard(&DROPPED))
            })
            .unwrap();
    }

    #[test]
    fn evict_clears_slot_and_next_get_reloads() {
        let cache = cache();
        let counter = AtomicU32::new(0);
        let load = |_: &String| -> Result<u32, ()> {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(1)
        };

        cache.get_or_load("a".to_string(), load).unwrap();
        assert_eq!(cache.evict(), Some("a".to_string()));
        assert_eq!(cache.evict(), None);

        cache.get_or_load("a".to_string(), load).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn failed_load_leaves_slot_empty_so_next_call_retries() {
        let cache: ResidentCache<String, u32> = ResidentCache::new();

        let result = cache.get_or_load("a".to_string(), |_| Err::<u32, &str>("boom"));
        assert!(result.is_err());
        assert_eq!(cache.evict(), None);

        let (value, reused) = cache
            .get_or_load("a".to_string(), |_| Ok::<u32, &str>(1))
            .unwrap();
        assert_eq!(*value, 1);
        assert!(!reused);
    }
}
