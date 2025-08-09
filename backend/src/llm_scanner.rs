use chrono::Local;
use rusqlite::{params, Connection, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

/// LLM Scanner module for discovering GGUF model files across different platforms.
/// 
/// This module handles path normalization and ensures cross-platform compatibility
/// for Windows, Linux, and macOS filesystems. It uses `Path::display()` for string
/// conversion to properly handle different path separators and encodings.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub path: String,
    pub filename: String,
    pub size_bytes: u64,
    pub directory: String,
    pub last_modified: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryInfo {
    pub id: i32,
    pub path: String,
    pub created_at: String,
}

pub struct LlmScanner {
    database_path: String,
}

impl LlmScanner {
    pub fn new() -> Self {
        Self {
            database_path: "companion_database.db".to_string(),
        }
    }

    /// Get default directories relative to the executable
    fn get_default_directories() -> Vec<PathBuf> {
        let mut dirs = Vec::new();
        
        if let Ok(exe_path) = env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                // ./llms directory
                let llms_dir = exe_dir.join("llms");
                dirs.push(llms_dir);
                
                // ../llms directory (one level up)
                if let Some(parent_dir) = exe_dir.parent() {
                    let parent_llms_dir = parent_dir.join("llms");
                    dirs.push(parent_llms_dir);
                }
            }
        }
        
        dirs
    }

    /// Scan a directory for GGUF model files
    fn scan_directory(dir_path: &Path) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        
        if !dir_path.exists() {
            return models;
        }

        for entry in WalkDir::new(dir_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            
            // Check if it's a file with .gguf extension (case-insensitive)
            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension.to_ascii_lowercase() == "gguf" {
                        if let Ok(metadata) = fs::metadata(path) {
                            let size_bytes = metadata.len();
                            
                            let last_modified = metadata
                                .modified()
                                .unwrap_or(SystemTime::UNIX_EPOCH)
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();
                            
                            let last_modified_str = chrono::DateTime::from_timestamp(last_modified as i64, 0)
                                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                                .unwrap_or_else(|| "Unknown".to_string());
                            
                            models.push(ModelInfo {
                                path: path.display().to_string(),
                                filename: path.file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string(),
                                size_bytes,
                                directory: dir_path.display().to_string(),
                                last_modified: last_modified_str,
                            });
                        }
                    }
                }
            }
        }
        
        models
    }

    /// Get all configured directories from the database
    pub fn get_directories(&self) -> Result<Vec<DirectoryInfo>> {
        let conn = Connection::open(&self.database_path)?;
        let mut stmt = conn.prepare("SELECT id, path, created_at FROM llm_directories ORDER BY id")?;
        
        let directories = stmt.query_map([], |row| {
            Ok(DirectoryInfo {
                id: row.get(0)?,
                path: row.get(1)?,
                created_at: row.get(2)?,
            })
        })?;
        
        let mut result = Vec::new();
        for dir in directories {
            result.push(dir?);
        }
        
        Ok(result)
    }

    /// Add a new directory to scan for models
    pub fn add_directory(&self, path: &str) -> Result<()> {
        // Normalize the path for cross-platform compatibility
        let path_buf = PathBuf::from(path);
        
        // Try to canonicalize, but if it fails (e.g., directory doesn't exist yet),
        // just clean up the path as much as possible
        let normalized_path = if let Ok(canonical) = path_buf.canonicalize() {
            canonical
        } else {
            // Clean up the path manually for cross-platform compatibility
            let cleaned = path_buf
                .components()
                .collect::<PathBuf>();
            cleaned
        };
        
        // Convert to string using display() for better cross-platform compatibility
        let path_string = normalized_path.display().to_string();
        
        let conn = Connection::open(&self.database_path)?;
        let created_at = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        
        conn.execute(
            "INSERT OR IGNORE INTO llm_directories (path, created_at) VALUES (?1, ?2)",
            params![path_string, created_at],
        )?;
        
        Ok(())
    }

    /// Remove a directory from the scan list
    pub fn remove_directory(&self, id: i32) -> Result<()> {
        let conn = Connection::open(&self.database_path)?;
        conn.execute(
            "DELETE FROM llm_directories WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// Scan all configured directories and default directories for models
    pub fn scan_for_models(&self) -> Result<Vec<ModelInfo>> {
        let mut all_models = Vec::new();
        let mut seen_paths = HashSet::new();
        
        // Scan default directories
        for dir in Self::get_default_directories() {
            if dir.exists() {
                for model in Self::scan_directory(&dir) {
                    if seen_paths.insert(model.path.clone()) {
                        all_models.push(model);
                    }
                }
            }
        }
        
        // Scan custom directories from database
        let custom_dirs = self.get_directories()?;
        for dir_info in custom_dirs {
            let dir_path = Path::new(&dir_info.path);
            for model in Self::scan_directory(dir_path) {
                if seen_paths.insert(model.path.clone()) {
                    all_models.push(model);
                }
            }
        }
        
        // Sort by filename for consistent ordering
        all_models.sort_by(|a, b| a.filename.cmp(&b.filename));
        
        Ok(all_models)
    }

    /// Check if the old llm_model_path exists and migrate it to a directory
    pub fn migrate_existing_config(&self) -> Result<()> {
        let conn = Connection::open(&self.database_path)?;
        
        // Get the current llm_model_path from config
        let model_path: Option<String> = conn
            .query_row(
                "SELECT llm_model_path FROM config WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .ok();
        
        if let Some(path) = model_path {
            let path_buf = PathBuf::from(&path);
            
            // If it's a file path and the file exists, add its parent directory
            if path_buf.is_file() {
                if let Some(parent) = path_buf.parent() {
                    let parent_str = parent.display().to_string();
                    // Add the parent directory to the scan list
                    self.add_directory(&parent_str)?;
                }
            }
            // If it's already a directory, add it directly
            else if path_buf.is_dir() {
                self.add_directory(&path)?;
            }
        }
        
        Ok(())
    }
}