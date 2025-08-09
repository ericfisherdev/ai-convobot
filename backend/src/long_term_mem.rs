use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tantivy::collector::TopDocs;
use tantivy::error::TantivyError;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, IndexReader};

pub struct LongTermMem {
    index: Index,
    chat_field: Field,
    reader: Arc<IndexReader>,
    query_cache: Arc<Mutex<HashMap<String, (Vec<String>, Instant)>>>,
}

impl LongTermMem {
    pub fn connect() -> tantivy::Result<Self> {
        let mut schema_builder = SchemaBuilder::default();
        let chat_field = schema_builder.add_text_field("chat", TEXT | STORED);
        let schema = schema_builder.build();
        if !Path::new("longterm_memory").exists() {
            fs::create_dir("longterm_memory")?;
        }
        let companion_vector = match Index::open_in_dir("longterm_memory") {
            Ok(index) => index,
            Err(_) => Index::create_in_dir("longterm_memory", schema)?,
        };

        // Create shared reader for better performance
        let reader = Arc::new(companion_vector.reader()?);
        let query_cache = Arc::new(Mutex::new(HashMap::new()));

        Ok(LongTermMem {
            index: companion_vector,
            chat_field,
            reader,
            query_cache,
        })
    }

    pub fn add_entry(&self, text: &str) -> Result<(), TantivyError> {
        let mut writer = self.index.writer(50_000_000)?;
        writer.add_document(tantivy::doc!(
            self.chat_field => text
        ))?;
        writer.commit()?;

        // Clear cache when new entries are added to ensure fresh results
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.clear();
        }

        Ok(())
    }

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

        // Cache the results
        if let Ok(mut cache) = self.query_cache.lock() {
            // Limit cache size to prevent memory issues
            if cache.len() > 100 {
                cache.clear();
            }
            cache.insert(cache_key, (result.clone(), Instant::now()));
        }

        Ok(result)
    }

    pub fn erase_memory(&self) -> Result<(), TantivyError> {
        let mut writer = self.index.writer(50_000_000)?;
        writer.delete_all_documents()?;
        writer.commit()?;

        // Clear cache when memory is erased
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.clear();
        }

        Ok(())
    }

    pub fn refresh_reader(&self) -> Result<(), TantivyError> {
        // Force refresh the reader to see latest changes
        self.reader.reload()?;
        Ok(())
    }

    pub fn get_cache_stats(&self) -> (usize, usize) {
        if let Ok(cache) = self.query_cache.lock() {
            let total_entries = cache.len();
            let expired_entries = cache
                .iter()
                .filter(|(_, (_, timestamp))| timestamp.elapsed() >= Duration::from_secs(300))
                .count();
            (total_entries, expired_entries)
        } else {
            (0, 0)
        }
    }
}
