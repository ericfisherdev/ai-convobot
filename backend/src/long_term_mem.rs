use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};
use tantivy::collector::TopDocs;
use tantivy::error::TantivyError;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy};

type QueryCache = Mutex<HashMap<String, (Vec<String>, Instant)>>;

/// Directory tantivy stores the long-term-memory index in.
const INDEX_DIR: &str = "longterm_memory";

/// Heap budget handed to the writer's single indexing thread.
///
/// This is tantivy's undocumented per-thread minimum
/// (`MEMORY_BUDGET_NUM_BYTES_MIN`, not re-exported from `tantivy::indexer`),
/// not an up-front allocation: the underlying arena grows lazily in 1 MB
/// pages, and a one-document-per-commit workload never comes close to
/// exhausting it. `Index::writer_with_num_threads(1, ..)` is used instead of
/// `Index::writer(..)` so the process holds exactly one indexing thread for
/// its lifetime rather than the 3 threads `writer(50_000_000)` would pick.
const WRITER_HEAP_BYTES: usize = 15_000_000;

pub struct LongTermMem {
    index: Index,
    chat_field: Field,
    writer: Mutex<IndexWriter>,
    reader: IndexReader,
    query_cache: QueryCache,
    /// Bumped by every commit (`add_entry`/`erase_memory`), after the cache
    /// is cleared. `get_matches` snapshots this before searching and only
    /// caches its result if it is still current at insert time, so a search
    /// that was already in flight when a commit landed cannot resurrect
    /// pre-commit results into the cache after the commit's own clear.
    generation: AtomicU64,
}

/// Process-wide singleton, mirroring the `RESIDENT_MODEL`/`LLAMA_BACKEND`
/// pattern in `llm.rs`. tantivy allows exactly one `IndexWriter` per index
/// directory (enforced with an fs2 exclusive lock on `.tantivy-writer.lock`),
/// so every caller sharing one writer for the process lifetime is what makes
/// concurrent `add_entry` calls queue on `writer` instead of failing with
/// `TantivyError::LockFailure(LockBusy, ..)`.
static LONG_TERM_MEM: OnceLock<LongTermMem> = OnceLock::new();
static INIT_LOCK: Mutex<()> = Mutex::new(());

impl LongTermMem {
    /// Returns the process-wide long-term memory index, opening it on first
    /// use. A failed open is not cached, so a transient startup failure (e.g.
    /// the directory not yet being writable) can succeed on a later call.
    pub fn shared() -> tantivy::Result<&'static LongTermMem> {
        if let Some(ltm) = LONG_TERM_MEM.get() {
            return Ok(ltm);
        }
        // Serialise first initialisation so two threads racing in here
        // cannot both call `open_at` and have the loser fail with
        // `LockBusy` while the winner succeeds.
        let _guard = INIT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(ltm) = LONG_TERM_MEM.get() {
            return Ok(ltm);
        }
        let ltm = Self::open_at(Path::new(INDEX_DIR))?;
        Ok(LONG_TERM_MEM.get_or_init(|| ltm))
    }

    /// Opens (or creates) the index at `dir`, building the single long-lived
    /// writer and a `Manual`-reload reader. Split out from `shared()` so
    /// tests can point it at a `tempfile::TempDir` instead of the process's
    /// `longterm_memory` directory.
    fn open_at(dir: &Path) -> tantivy::Result<Self> {
        let mut schema_builder = SchemaBuilder::default();
        let chat_field = schema_builder.add_text_field("chat", TEXT | STORED);
        let schema = schema_builder.build();
        if !dir.exists() {
            fs::create_dir_all(dir)?;
        }
        let index = match Index::open_in_dir(dir) {
            Ok(index) => index,
            Err(_) => Index::create_in_dir(dir, schema)?,
        };

        let writer = index.writer_with_num_threads(1, WRITER_HEAP_BYTES)?;

        // `Manual` instead of the default `OnCommit`: this process is the
        // only writer, so we reload deterministically right after our own
        // commit (see `add_entry`/`erase_memory`) rather than relying on
        // `OnCommit`'s async `meta.json` file-watcher thread, which can
        // still miss a just-committed document on an immediate re-query.
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let query_cache = Mutex::new(HashMap::new());

        Ok(LongTermMem {
            index,
            chat_field,
            writer: Mutex::new(writer),
            reader,
            query_cache,
            generation: AtomicU64::new(0),
        })
    }

    pub fn add_entry(&self, text: &str) -> Result<(), TantivyError> {
        let mut writer = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        writer.add_document(tantivy::doc!(
            self.chat_field => text
        ))?;
        writer.commit()?;
        self.reader.reload()?;

        // Clear cache when new entries are added to ensure fresh results
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.clear();
        }
        self.generation.fetch_add(1, Ordering::SeqCst);

        Ok(())
    }

    /// Searches the index. The query cache is process-wide (backed by the
    /// same singleton every caller shares via `shared()`), which is what its
    /// 5-minute TTL was written for.
    pub fn get_matches(
        &self,
        query_string: &str,
        limit: usize,
    ) -> Result<Vec<String>, TantivyError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut sanitized_query = query_string.replace("\n", " ");
        sanitized_query = sanitized_query
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .to_lowercase();

        // Create cache key including limit for proper caching
        let cache_key = format!("{}:{}", sanitized_query, limit);

        // Check cache first
        if let Ok(cache) = self.query_cache.lock() {
            if let Some((results, timestamp)) = cache.get(&cache_key) {
                // Cache for 5 minutes
                if timestamp.elapsed() < Duration::from_secs(300) {
                    return Ok(results.clone());
                }
            }
        }

        // Snapshot the generation before searching: if a commit lands (and
        // clears the cache) while this search is still running, the
        // generation will have moved by the time we reach the insert below,
        // and `cache_insert_if_fresh` skips caching this now-stale result.
        let generation_before = self.generation.load(Ordering::SeqCst);

        // Use the shared reader instead of creating a new one
        let searcher = self.reader.searcher();
        let qp = QueryParser::for_index(&self.index, vec![self.chat_field]);
        let query = match qp.parse_query(&sanitized_query) {
            Ok(q) => q,
            Err(e) => return Err(TantivyError::from(e)),
        };

        let matches: Vec<(f32, tantivy::DocAddress)> =
            searcher.search(&query, &TopDocs::with_limit(limit))?;
        let mut result: Vec<String> = Vec::new();

        for (_, text_addr) in matches {
            let retrieved = searcher.doc(text_addr)?;
            let r = retrieved
                .get_first(self.chat_field)
                .and_then(|val| val.as_text())
                .unwrap_or("");
            result.push(r.to_string());
        }

        self.cache_insert_if_fresh(cache_key, generation_before, result.clone());

        Ok(result)
    }

    /// Inserts `result` into the query cache unless a commit landed after
    /// `generation_before` was captured, in which case the result was
    /// computed against pre-commit data and inserting it now would resurrect
    /// stale results for up to the cache's 5-minute TTL right after the
    /// commit's own clear.
    fn cache_insert_if_fresh(
        &self,
        cache_key: String,
        generation_before: u64,
        result: Vec<String>,
    ) {
        if let Ok(mut cache) = self.query_cache.lock() {
            if self.generation.load(Ordering::SeqCst) != generation_before {
                return;
            }
            // Limit cache size to prevent memory issues
            if cache.len() > 100 {
                cache.clear();
            }
            cache.insert(cache_key, (result, Instant::now()));
        }
    }

    pub fn erase_memory(&self) -> Result<(), TantivyError> {
        let mut writer = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        writer.delete_all_documents()?;
        writer.commit()?;
        self.reader.reload()?;

        // Clear cache when memory is erased
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.clear();
        }
        self.generation.fetch_add(1, Ordering::SeqCst);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn add_entry_is_searchable_immediately_on_same_instance() {
        let dir = tempfile::TempDir::new().unwrap();
        let ltm = LongTermMem::open_at(dir.path()).unwrap();

        ltm.add_entry("the cat sat").unwrap();

        let matches = ltm.get_matches("cat", 5).unwrap();
        assert_eq!(matches, vec!["the cat sat".to_string()]);
    }

    #[test]
    fn two_add_entries_back_to_back_do_not_hit_lock_busy() {
        let dir = tempfile::TempDir::new().unwrap();
        let ltm = LongTermMem::open_at(dir.path()).unwrap();

        ltm.add_entry("first entry").unwrap();
        ltm.add_entry("second entry").unwrap();

        let matches = ltm.get_matches("entry", 5).unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn concurrent_add_entries_all_succeed() {
        let dir = tempfile::TempDir::new().unwrap();
        let ltm = LongTermMem::open_at(dir.path()).unwrap();

        thread::scope(|scope| {
            for i in 0..4 {
                let ltm = &ltm;
                scope.spawn(move || {
                    ltm.add_entry(&format!("concurrent entry {i}")).unwrap();
                });
            }
        });

        let matches = ltm.get_matches("concurrent", 10).unwrap();
        assert_eq!(matches.len(), 4);
    }

    #[test]
    fn erase_memory_leaves_no_matches() {
        let dir = tempfile::TempDir::new().unwrap();
        let ltm = LongTermMem::open_at(dir.path()).unwrap();

        ltm.add_entry("something memorable").unwrap();
        ltm.erase_memory().unwrap();

        let matches = ltm.get_matches("memorable", 5).unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn add_entry_invalidates_query_cache() {
        let dir = tempfile::TempDir::new().unwrap();
        let ltm = LongTermMem::open_at(dir.path()).unwrap();

        assert!(ltm.get_matches("dog", 5).unwrap().is_empty());

        ltm.add_entry("the dog barked").unwrap();

        let matches = ltm.get_matches("dog", 5).unwrap();
        assert_eq!(matches, vec!["the dog barked".to_string()]);
    }

    /// Regression test for a race CodeRabbit flagged on PR #116:
    /// `get_matches` captures `generation_before`, searches, then inserts
    /// into the cache. If `add_entry` commits and clears the cache in
    /// between, the late insert must not resurrect the pre-commit result.
    /// The barrier forces the commit to finish before the simulated late
    /// insert runs, without any timing-dependent sleep.
    #[test]
    fn add_entry_racing_a_search_does_not_poison_the_cache_with_stale_results() {
        let dir = tempfile::TempDir::new().unwrap();
        let ltm = LongTermMem::open_at(dir.path()).unwrap();
        ltm.add_entry("original entry").unwrap();

        // What get_matches would have captured before starting its search.
        let generation_before = ltm.generation.load(Ordering::SeqCst);

        let barrier = Barrier::new(2);
        thread::scope(|scope| {
            scope.spawn(|| {
                ltm.add_entry("second entry added mid search").unwrap();
                barrier.wait();
            });
            scope.spawn(|| {
                barrier.wait();
                // Runs only after add_entry's commit and cache clear above
                // have completed, simulating the in-flight search's insert
                // landing right after them.
                ltm.cache_insert_if_fresh(
                    "original:5".to_string(),
                    generation_before,
                    vec!["stale result computed before the commit".to_string()],
                );
            });
        });

        assert!(ltm.query_cache.lock().unwrap().get("original:5").is_none());
    }

    /// Same race, on the `erase_memory` mutation path.
    #[test]
    fn erase_memory_racing_a_search_does_not_poison_the_cache_with_stale_results() {
        let dir = tempfile::TempDir::new().unwrap();
        let ltm = LongTermMem::open_at(dir.path()).unwrap();
        ltm.add_entry("original entry").unwrap();

        let generation_before = ltm.generation.load(Ordering::SeqCst);

        let barrier = Barrier::new(2);
        thread::scope(|scope| {
            scope.spawn(|| {
                ltm.erase_memory().unwrap();
                barrier.wait();
            });
            scope.spawn(|| {
                barrier.wait();
                ltm.cache_insert_if_fresh(
                    "original:5".to_string(),
                    generation_before,
                    vec!["stale result computed before the erase".to_string()],
                );
            });
        });

        assert!(ltm.query_cache.lock().unwrap().get("original:5").is_none());
    }

    #[test]
    fn cache_insert_if_fresh_inserts_when_generation_is_unchanged() {
        let dir = tempfile::TempDir::new().unwrap();
        let ltm = LongTermMem::open_at(dir.path()).unwrap();

        let generation_before = ltm.generation.load(Ordering::SeqCst);
        ltm.cache_insert_if_fresh(
            "fresh:5".to_string(),
            generation_before,
            vec!["fresh result".to_string()],
        );

        assert_eq!(
            ltm.query_cache.lock().unwrap().get("fresh:5").unwrap().0,
            vec!["fresh result".to_string()]
        );
    }
}
