//! The single definition of every on-disk name the server reads or writes,
//! all resolved under one configurable data directory (#107).
//!
//! Before this module, `database::DATABASE_PATH` and
//! `long_term_mem::INDEX_DIR` were relative literals resolved against the
//! process's working directory, and the three avatar handlers in `main.rs`
//! each hardcoded `"assets/avatar.png"` separately. `main()` now resolves
//! `COMPANION_DATA_DIR` (via `settings::from_env`) to an absolute path once
//! at startup and installs it with [`init`]; every accessor here joins that
//! directory instead of duplicating the literal.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// SQLite database file name, moved from the old `database::DATABASE_PATH`.
const DB_FILE_NAME: &str = "companion_database.db";
/// Tantivy long-term-memory index directory name, moved from the old
/// `long_term_mem::INDEX_DIR`.
const LTM_DIR_NAME: &str = "longterm_memory";
/// Directory the companion avatar (and any future static upload) lives in.
const ASSETS_DIR_NAME: &str = "assets";
/// Companion avatar file name within [`assets_dir`].
const AVATAR_FILE_NAME: &str = "avatar.png";

/// Installed once by `main()` after `COMPANION_DATA_DIR` has been resolved
/// to an absolute path and created. Unit tests never call `init` (a
/// `OnceLock` cannot be reset and tests run in parallel on the same
/// process); they exercise the uninitialised default instead. The
/// initialised path is covered by the `backend/tests/env_config.rs`
/// integration tests, each of which spawns its own process.
static DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Installs the resolved data directory. Returns `Err` with the
/// already-installed directory if called more than once; `main()` only ever
/// calls this once, so a second call is a bug and is mapped to an
/// `io::Error` there rather than panicking here.
pub fn init(data_dir: PathBuf) -> Result<(), PathBuf> {
    DATA_DIR.set(data_dir)
}

/// The active data directory, or `.` if [`init`] has not been called yet
/// (unit tests, and any `open_at`-style helper that predates #107).
pub fn data_dir() -> &'static Path {
    DATA_DIR
        .get()
        .map(PathBuf::as_path)
        .unwrap_or(Path::new("."))
}

/// Path to the SQLite database file.
pub fn db_path() -> PathBuf {
    data_dir().join(DB_FILE_NAME)
}

/// Path to the tantivy long-term-memory index directory.
pub fn ltm_dir() -> PathBuf {
    data_dir().join(LTM_DIR_NAME)
}

/// Path to the directory holding the companion avatar.
pub fn assets_dir() -> PathBuf {
    data_dir().join(ASSETS_DIR_NAME)
}

/// Path to the companion avatar file.
pub fn avatar_path() -> PathBuf {
    assets_dir().join(AVATAR_FILE_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    // `init` is deliberately never called here: `DATA_DIR` is a
    // process-wide `OnceLock` and unit tests run in parallel, so any test
    // that installed a value would leak it into every other test in this
    // binary. The initialised path is covered by the integration tests in
    // `backend/tests/env_config.rs`, each of which spawns its own process.

    #[test]
    fn uninitialised_accessors_resolve_under_the_current_directory() {
        assert_eq!(data_dir(), Path::new("."));
    }

    #[test]
    fn db_path_ends_in_the_expected_file_name() {
        assert_eq!(db_path(), Path::new("./companion_database.db"));
    }

    #[test]
    fn ltm_dir_ends_in_the_expected_directory_name() {
        assert_eq!(ltm_dir(), Path::new("./longterm_memory"));
    }

    #[test]
    fn avatar_path_ends_in_the_expected_directory_and_file_name() {
        assert_eq!(assets_dir(), Path::new("./assets"));
        assert_eq!(avatar_path(), Path::new("./assets/avatar.png"));
    }
}
