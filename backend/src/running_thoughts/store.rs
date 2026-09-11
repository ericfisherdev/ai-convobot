//! The persistence seam for running thoughts (#215).
//!
//! [`RunningThoughtStore`] is the trait later issues (#216 generation, #217
//! routes, #219 checkpoint-overlap handling) are written against, so they
//! can be unit-tested with [`RecordingStore`] instead of the hardwired
//! `companion_database.db` (`Database::open()` has no path parameter),
//! matching `compaction::store::CompactionStore`/`RecordingStore`.
//!
//! [`SqliteRunningThoughtStore`] is the production impl: each trait method
//! opens the shared database, like every other `Database` associated fn,
//! then delegates to a `pub(crate) fn <name>_on(con: &Connection, ..)`
//! helper, so the unit tests below can run the helpers directly against
//! `Database::open_at(tempdir)`. Only [`delete_from_on`] is ever called
//! inside a transaction the trait method opens for it (the same
//! select-then-delete-in-one-transaction shape `insert_facts_on` uses);
//! every other helper is a single statement.
//!
//! #216 wires `insert`/`get`/`recent_for`/`latest_for` in (the chained
//! context a live round reads and writes); `list`/`in_range`/`update_text`/
//! `delete`/`delete_from` stay unreached from production until #217/#219/#220
//! wire them in turn, hence the per-item `#[allow(dead_code)]` below instead
//! of a blanket one. Every method is already exercised by the unit tests in
//! this module, which is what the #215 acceptance criteria ask for.

use rusqlite::{params, Connection, Error, OptionalExtension, Result, Row, TransactionBehavior};

use crate::database::{get_current_date, Database};
use crate::running_thoughts::types::{NewRunningThought, RunningThought};

/// Creates the `running_thoughts` table (and its index) if it does not
/// already exist. Shared by `Database::init` (called after `companion`
/// already exists) and the tests below, which build the schema on a
/// temp-file connection without going through `Database::init`'s hardwired
/// path.
///
/// `AUTOINCREMENT`, not a plain `INTEGER PRIMARY KEY`: a plain rowid can be
/// reused after a delete, so after #217's delete-then-regenerate a panel
/// still holding a stale id could `PATCH` the wrong row — gaps in the id
/// sequence matter here.
///
/// `ON DELETE CASCADE` on `companion_id`, matching `compactions`:
/// `Database::open_at` turns `PRAGMA foreign_keys` on for every connection,
/// so an unknown `companion_id` fails at insert instead of producing an
/// orphan.
///
/// Deliberately **no** foreign key to `messages`: `from_message_id`/
/// `through_message_id` are references by value, so a thought outlives the
/// messages it is about (#214's "edits do not retroactively invalidate"
/// rule) instead of cascading away with them.
pub(crate) fn create_tables(con: &Connection) -> Result<()> {
    con.execute(
        "CREATE TABLE IF NOT EXISTS running_thoughts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            companion_id INTEGER NOT NULL REFERENCES companion(id) ON DELETE CASCADE,
            speaker_id TEXT NOT NULL,
            from_message_id INTEGER NOT NULL,
            through_message_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            edited INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )",
        [],
    )?;
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_running_thoughts_author
            ON running_thoughts(companion_id, speaker_id, id)",
        [],
    )?;
    Ok(())
}

/// Column list shared by every query that reads a full `running_thoughts`
/// row, in the order [`thought_from_row`] expects.
const THOUGHT_COLUMNS: &str =
    "id, companion_id, speaker_id, from_message_id, through_message_id, text, edited, created_at";

fn thought_from_row(row: &Row) -> Result<RunningThought> {
    Ok(RunningThought {
        id: row.get(0)?,
        companion_id: row.get(1)?,
        speaker_id: row.get(2)?,
        from_message_id: row.get(3)?,
        through_message_id: row.get(4)?,
        text: row.get(5)?,
        edited: row.get(6)?,
        created_at: row.get(7)?,
    })
}

/// Inserts a new thought and returns its id. `SqliteFailure
/// (ConstraintViolation)` if `thought.companion_id` does not exist.
pub(crate) fn insert_on(con: &Connection, thought: &NewRunningThought) -> Result<i64> {
    con.execute(
        "INSERT INTO running_thoughts (companion_id, speaker_id, from_message_id, through_message_id, text, edited, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            thought.companion_id,
            thought.speaker_id,
            thought.from_message_id,
            thought.through_message_id,
            thought.text,
            thought.edited,
            get_current_date(),
        ],
    )?;
    Ok(con.last_insert_rowid())
}

/// `Ok(None)` for an unknown id, never `QueryReturnedNoRows`.
pub(crate) fn get_on(con: &Connection, id: i64) -> Result<Option<RunningThought>> {
    con.query_row(
        &format!("SELECT {THOUGHT_COLUMNS} FROM running_thoughts WHERE id = ?"),
        params![id],
        thought_from_row,
    )
    .optional()
}

/// Every thought for `companion_id`, every speaker, oldest first —
/// transcript order, the order #217/#218 render.
pub(crate) fn list_on(con: &Connection, companion_id: i32) -> Result<Vec<RunningThought>> {
    let mut stmt = con.prepare(&format!(
        "SELECT {THOUGHT_COLUMNS} FROM running_thoughts WHERE companion_id = ? ORDER BY id"
    ))?;
    let rows = stmt.query_map(params![companion_id], thought_from_row)?;
    rows.collect()
}

/// The last `limit` thoughts for `speaker_id` (and only that speaker —
/// per-speaker scoping is what gives #220 its isolation for free),
/// returned in chronological order. Fewer than `limit` rows if fewer exist.
/// What #216's chained context reads.
pub(crate) fn recent_for_on(
    con: &Connection,
    companion_id: i32,
    speaker_id: &str,
    limit: usize,
) -> Result<Vec<RunningThought>> {
    let mut stmt = con.prepare(&format!(
        "SELECT {THOUGHT_COLUMNS} FROM running_thoughts
         WHERE companion_id = ? AND speaker_id = ?
         ORDER BY id DESC LIMIT ?"
    ))?;
    let rows = stmt.query_map(
        params![companion_id, speaker_id, limit as i64],
        thought_from_row,
    )?;
    let mut thoughts = rows.collect::<Result<Vec<_>>>()?;
    thoughts.reverse();
    Ok(thoughts)
}

/// The highest-id thought for `speaker_id`, or `None` if that speaker has
/// none yet.
pub(crate) fn latest_for_on(
    con: &Connection,
    companion_id: i32,
    speaker_id: &str,
) -> Result<Option<RunningThought>> {
    con.query_row(
        &format!(
            "SELECT {THOUGHT_COLUMNS} FROM running_thoughts
             WHERE companion_id = ? AND speaker_id = ?
             ORDER BY id DESC LIMIT 1"
        ),
        params![companion_id, speaker_id],
        thought_from_row,
    )
    .optional()
}

/// Every thought (any speaker) for `companion_id` whose `[from_message_id,
/// through_message_id]` range *overlaps* `[from, through]`, ordered by id —
/// what #219 checks a checkpoint's own range against.
#[allow(dead_code)] // wired up by #219: reached once its overlap check calls this
pub(crate) fn in_range_on(
    con: &Connection,
    companion_id: i32,
    from: i32,
    through: i32,
) -> Result<Vec<RunningThought>> {
    let mut stmt = con.prepare(&format!(
        "SELECT {THOUGHT_COLUMNS} FROM running_thoughts
         WHERE companion_id = ? AND from_message_id <= ? AND through_message_id >= ?
         ORDER BY id"
    ))?;
    let rows = stmt.query_map(params![companion_id, through, from], thought_from_row)?;
    rows.collect()
}

/// Rewrites `text` and marks the row `edited`. `QueryReturnedNoRows` if `id`
/// does not exist (checked via `changes() == 0`, so a silent no-op is
/// impossible).
pub(crate) fn update_text_on(con: &Connection, id: i64, text: &str) -> Result<()> {
    let changed = con.execute(
        "UPDATE running_thoughts SET text = ?, edited = 1 WHERE id = ?",
        params![text, id],
    )?;
    if changed == 0 {
        return Err(Error::QueryReturnedNoRows);
    }
    Ok(())
}

/// `QueryReturnedNoRows` if `id` does not exist (so #217 can 404).
pub(crate) fn delete_on(con: &Connection, id: i64) -> Result<()> {
    let changed = con.execute("DELETE FROM running_thoughts WHERE id = ?", params![id])?;
    if changed == 0 {
        return Err(Error::QueryReturnedNoRows);
    }
    Ok(())
}

/// Deletes every thought (any speaker) for `companion_id` whose
/// `through_message_id >= message_id` and returns the deleted rows, in id
/// order (`Ok(vec![])` when none match).
///
/// Overlap, not "starts at": a thought about a round that merely *reaches*
/// `message_id` is about content from `message_id` onward and must be
/// regenerated too (same reasoning as `compaction::store::
/// retire_stale_within_on`). #217's regenerate uses the returned rows'
/// `from_message_id` to know each round's real starting point to
/// regenerate from, and to re-insert the originals if it fails partway.
///
/// Takes no transaction of its own — same split as every other helper here
/// — but unlike them this one runs a `SELECT` then a `DELETE`, so it is the
/// caller's job (`SqliteRunningThoughtStore::delete_from`) to wrap the call
/// in one, the same way `insert_facts_on` relies on its own caller for
/// atomicity.
pub(crate) fn delete_from_on(
    con: &Connection,
    companion_id: i32,
    message_id: i32,
) -> Result<Vec<RunningThought>> {
    let mut stmt = con.prepare(&format!(
        "SELECT {THOUGHT_COLUMNS} FROM running_thoughts
         WHERE companion_id = ? AND through_message_id >= ?
         ORDER BY id"
    ))?;
    let deleted = stmt
        .query_map(params![companion_id, message_id], thought_from_row)?
        .collect::<Result<Vec<_>>>()?;
    drop(stmt);

    con.execute(
        "DELETE FROM running_thoughts WHERE companion_id = ? AND through_message_id >= ?",
        params![companion_id, message_id],
    )?;
    Ok(deleted)
}

/// The persistence seam for running thoughts. See each `_on` helper above
/// for the SQL and error conventions this mirrors one-for-one.
pub trait RunningThoughtStore {
    /// `SqliteFailure(ConstraintViolation)` if `thought.companion_id` does
    /// not exist.
    fn insert(&self, thought: NewRunningThought) -> Result<i64>;

    /// `Ok(None)` for an unknown id, never `QueryReturnedNoRows`.
    fn get(&self, id: i64) -> Result<Option<RunningThought>>;

    /// Every thought for `companion_id`, every speaker, ordered by `id`.
    fn list(&self, companion_id: i32) -> Result<Vec<RunningThought>>;

    /// The last `limit` thoughts for `speaker_id` only, chronological
    /// order, fewer than `limit` if fewer exist.
    fn recent_for(
        &self,
        companion_id: i32,
        speaker_id: &str,
        limit: usize,
    ) -> Result<Vec<RunningThought>>;

    /// The highest-id thought for `speaker_id`, or `None`.
    fn latest_for(&self, companion_id: i32, speaker_id: &str) -> Result<Option<RunningThought>>;

    /// Every thought (any speaker) whose range overlaps `[from, through]`,
    /// ordered by `id`.
    #[allow(dead_code)] // wired up by #219: reached once its overlap check calls this
    fn in_range(&self, companion_id: i32, from: i32, through: i32) -> Result<Vec<RunningThought>>;

    /// Rewrites `text` and sets `edited = true`. `QueryReturnedNoRows` if
    /// `id` does not exist.
    fn update_text(&self, id: i64, text: &str) -> Result<()>;

    /// `QueryReturnedNoRows` if `id` does not exist.
    fn delete(&self, id: i64) -> Result<()>;

    /// Deletes every thought (any speaker) with `through_message_id >=
    /// message_id`, in one transaction, and returns the deleted rows in id
    /// order.
    fn delete_from(&self, companion_id: i32, message_id: i32) -> Result<Vec<RunningThought>>;
}

/// Production [`RunningThoughtStore`], opening `Database::open()` per call,
/// exactly like every other `Database` associated fn.
pub struct SqliteRunningThoughtStore;

impl RunningThoughtStore for SqliteRunningThoughtStore {
    fn insert(&self, thought: NewRunningThought) -> Result<i64> {
        let con = Database::open()?;
        insert_on(&con, &thought)
    }

    fn get(&self, id: i64) -> Result<Option<RunningThought>> {
        let con = Database::open()?;
        get_on(&con, id)
    }

    fn list(&self, companion_id: i32) -> Result<Vec<RunningThought>> {
        let con = Database::open()?;
        list_on(&con, companion_id)
    }

    fn recent_for(
        &self,
        companion_id: i32,
        speaker_id: &str,
        limit: usize,
    ) -> Result<Vec<RunningThought>> {
        let con = Database::open()?;
        recent_for_on(&con, companion_id, speaker_id, limit)
    }

    fn latest_for(&self, companion_id: i32, speaker_id: &str) -> Result<Option<RunningThought>> {
        let con = Database::open()?;
        latest_for_on(&con, companion_id, speaker_id)
    }

    fn in_range(&self, companion_id: i32, from: i32, through: i32) -> Result<Vec<RunningThought>> {
        let con = Database::open()?;
        in_range_on(&con, companion_id, from, through)
    }

    fn update_text(&self, id: i64, text: &str) -> Result<()> {
        let con = Database::open()?;
        update_text_on(&con, id, text)
    }

    fn delete(&self, id: i64) -> Result<()> {
        let con = Database::open()?;
        delete_on(&con, id)
    }

    fn delete_from(&self, companion_id: i32, message_id: i32) -> Result<Vec<RunningThought>> {
        let mut con = Database::open()?;
        // `Immediate`, not the default `Deferred`: `delete_from_on` reads
        // before it writes, so a deferred transaction starts as a read and
        // only upgrades to a write lock at the `DELETE`. Under WAL, a
        // concurrent writer that commits in between can make that upgrade
        // fail with `SQLITE_BUSY_SNAPSHOT` (the busy timeout does not retry
        // a write-upgrade rejection), which would abort the `DELETE`
        // without ever returning a mismatched row set. Taking the write
        // lock up front avoids the race entirely, matching every other
        // select-then-write transaction in `database.rs`.
        let tx = con.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let deleted = delete_from_on(&tx, companion_id, message_id)?;
        tx.commit()?;
        Ok(deleted)
    }
}

/// In-memory [`RunningThoughtStore`], mirroring
/// `compaction::store::RecordingStore`'s shape so other modules' tests use
/// it the same way. Reproduces the documented error variants
/// (`QueryReturnedNoRows` from `update_text`/`delete` on unknown ids) so a
/// test written against it behaves like the SQLite one.
#[cfg(test)]
pub(crate) struct RecordingStore {
    pub(crate) thoughts: std::sync::Mutex<Vec<RunningThought>>,
    next_id: std::sync::atomic::AtomicI64,
}

#[cfg(test)]
impl RecordingStore {
    pub(crate) fn new() -> Self {
        Self {
            thoughts: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicI64::new(0),
        }
    }
}

#[cfg(test)]
impl RunningThoughtStore for RecordingStore {
    fn insert(&self, thought: NewRunningThought) -> Result<i64> {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        self.thoughts.lock().unwrap().push(RunningThought {
            id,
            companion_id: thought.companion_id,
            speaker_id: thought.speaker_id,
            from_message_id: thought.from_message_id,
            through_message_id: thought.through_message_id,
            text: thought.text,
            edited: thought.edited,
            created_at: get_current_date(),
        });
        Ok(id)
    }

    fn get(&self, id: i64) -> Result<Option<RunningThought>> {
        Ok(self
            .thoughts
            .lock()
            .unwrap()
            .iter()
            .find(|t| t.id == id)
            .cloned())
    }

    fn list(&self, companion_id: i32) -> Result<Vec<RunningThought>> {
        let mut thoughts: Vec<RunningThought> = self
            .thoughts
            .lock()
            .unwrap()
            .iter()
            .filter(|t| t.companion_id == companion_id)
            .cloned()
            .collect();
        thoughts.sort_by_key(|t| t.id);
        Ok(thoughts)
    }

    fn recent_for(
        &self,
        companion_id: i32,
        speaker_id: &str,
        limit: usize,
    ) -> Result<Vec<RunningThought>> {
        let mut thoughts: Vec<RunningThought> = self
            .thoughts
            .lock()
            .unwrap()
            .iter()
            .filter(|t| t.companion_id == companion_id && t.speaker_id == speaker_id)
            .cloned()
            .collect();
        thoughts.sort_by_key(|t| t.id);
        if thoughts.len() > limit {
            thoughts = thoughts.split_off(thoughts.len() - limit);
        }
        Ok(thoughts)
    }

    fn latest_for(&self, companion_id: i32, speaker_id: &str) -> Result<Option<RunningThought>> {
        Ok(self
            .thoughts
            .lock()
            .unwrap()
            .iter()
            .filter(|t| t.companion_id == companion_id && t.speaker_id == speaker_id)
            .max_by_key(|t| t.id)
            .cloned())
    }

    fn in_range(&self, companion_id: i32, from: i32, through: i32) -> Result<Vec<RunningThought>> {
        let mut thoughts: Vec<RunningThought> = self
            .thoughts
            .lock()
            .unwrap()
            .iter()
            .filter(|t| {
                t.companion_id == companion_id
                    && t.from_message_id <= through
                    && t.through_message_id >= from
            })
            .cloned()
            .collect();
        thoughts.sort_by_key(|t| t.id);
        Ok(thoughts)
    }

    fn update_text(&self, id: i64, text: &str) -> Result<()> {
        let mut thoughts = self.thoughts.lock().unwrap();
        match thoughts.iter_mut().find(|t| t.id == id) {
            Some(t) => {
                t.text = text.to_string();
                t.edited = true;
                Ok(())
            }
            None => Err(Error::QueryReturnedNoRows),
        }
    }

    fn delete(&self, id: i64) -> Result<()> {
        let mut thoughts = self.thoughts.lock().unwrap();
        let before = thoughts.len();
        thoughts.retain(|t| t.id != id);
        if thoughts.len() == before {
            return Err(Error::QueryReturnedNoRows);
        }
        Ok(())
    }

    fn delete_from(&self, companion_id: i32, message_id: i32) -> Result<Vec<RunningThought>> {
        let mut thoughts = self.thoughts.lock().unwrap();
        let (removed, kept): (Vec<_>, Vec<_>) = thoughts
            .drain(..)
            .partition(|t| t.companion_id == companion_id && t.through_message_id >= message_id);
        *thoughts = kept;
        let mut removed = removed;
        removed.sort_by_key(|t| t.id);
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds the schema `create_tables` needs (`companion`, then
    /// `running_thoughts`) on a temp-file connection, and seeds one
    /// companion row. Mirrors `compaction::store::tests::fresh_db`.
    fn fresh_db() -> (tempfile::TempDir, Connection) {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        con.execute(
            "CREATE TABLE IF NOT EXISTS companion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                persona TEXT,
                example_dialogue TEXT,
                first_message TEXT,
                long_term_mem INTEGER,
                short_term_mem INTEGER,
                roleplay BOOLEAN,
                dialogue_tuning BOOLEAN,
                avatar_path TEXT,
                compacted_through INTEGER
            )",
            [],
        )
        .unwrap();
        create_tables(&con).unwrap();

        con.execute(
            "INSERT INTO companion (id, name, persona, example_dialogue, first_message, long_term_mem, short_term_mem, roleplay, dialogue_tuning, avatar_path) VALUES (1, 'Test', '', '', '', 0, 0, 0, 0, '')",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO companion (id, name, persona, example_dialogue, first_message, long_term_mem, short_term_mem, roleplay, dialogue_tuning, avatar_path) VALUES (2, 'Other', '', '', '', 0, 0, 0, 0, '')",
            [],
        )
        .unwrap();
        (dir, con)
    }

    fn a_thought(from: i32, through: i32) -> NewRunningThought {
        NewRunningThought {
            companion_id: 1,
            speaker_id: "user".to_string(),
            from_message_id: from,
            through_message_id: through,
            text: "a thought".to_string(),
            edited: false,
        }
    }

    #[test]
    fn create_tables_is_idempotent_and_leaves_other_tables_alone() {
        let (_dir, con) = fresh_db();

        let companion_columns_before: Vec<String> = con
            .prepare("PRAGMA table_info(companion)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<Result<_>>()
            .unwrap();

        // Calling it again on a connection that already has the table must
        // not error or change anything.
        create_tables(&con).unwrap();
        create_tables(&con).unwrap();

        let companion_columns_after: Vec<String> = con
            .prepare("PRAGMA table_info(companion)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<Result<_>>()
            .unwrap();
        assert_eq!(companion_columns_before, companion_columns_after);

        let running_thoughts_columns: Vec<String> = con
            .prepare("PRAGMA table_info(running_thoughts)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<Result<_>>()
            .unwrap();
        assert_eq!(
            running_thoughts_columns,
            vec![
                "id",
                "companion_id",
                "speaker_id",
                "from_message_id",
                "through_message_id",
                "text",
                "edited",
                "created_at"
            ]
        );
    }

    #[test]
    fn insert_then_get_round_trips_every_field_including_edited() {
        let (_dir, con) = fresh_db();
        let mut thought = a_thought(1, 3);
        thought.edited = true;
        thought.text = "the user seems excited about the trip".to_string();

        let id = insert_on(&con, &thought).unwrap();
        let got = get_on(&con, id).unwrap().unwrap();

        assert_eq!(got.id, id);
        assert_eq!(got.companion_id, 1);
        assert_eq!(got.speaker_id, "user");
        assert_eq!(got.from_message_id, 1);
        assert_eq!(got.through_message_id, 3);
        assert_eq!(got.text, "the user seems excited about the trip");
        assert!(got.edited);
        assert!(!got.created_at.is_empty());
    }

    #[test]
    fn get_returns_none_for_an_unknown_id() {
        let (_dir, con) = fresh_db();
        assert!(get_on(&con, 999).unwrap().is_none());
    }

    #[test]
    fn ids_do_not_get_reused_after_a_delete() {
        let (_dir, con) = fresh_db();
        let first = insert_on(&con, &a_thought(1, 1)).unwrap();
        delete_on(&con, first).unwrap();
        let second = insert_on(&con, &a_thought(1, 1)).unwrap();
        assert!(second > first);
    }

    #[test]
    fn insert_with_an_unknown_companion_id_is_a_constraint_violation() {
        let (_dir, con) = fresh_db();
        let mut thought = a_thought(1, 1);
        thought.companion_id = 999;
        let err = insert_on(&con, &thought).unwrap_err();
        assert!(matches!(
            err,
            Error::SqliteFailure(e, _) if e.code == rusqlite::ErrorCode::ConstraintViolation
        ));
    }

    #[test]
    fn list_orders_by_id_across_speakers() {
        let (_dir, con) = fresh_db();
        let mut a = a_thought(1, 2);
        a.speaker_id = "user".to_string();
        let mut b = a_thought(3, 4);
        b.speaker_id = "bot1".to_string();
        let id_a = insert_on(&con, &a).unwrap();
        let id_b = insert_on(&con, &b).unwrap();

        let listed = list_on(&con, 1).unwrap();
        assert_eq!(
            listed.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![id_a, id_b]
        );
    }

    #[test]
    fn list_only_returns_the_given_companion() {
        let (_dir, con) = fresh_db();
        insert_on(&con, &a_thought(1, 1)).unwrap();
        let mut other = a_thought(1, 1);
        other.companion_id = 2;
        insert_on(&con, &other).unwrap();

        assert_eq!(list_on(&con, 2).unwrap().len(), 1);
        assert_eq!(list_on(&con, 1).unwrap().len(), 1);
    }

    #[test]
    fn recent_for_returns_only_that_speakers_last_n_in_chronological_order() {
        let (_dir, con) = fresh_db();
        let mut user_ids = Vec::new();
        for i in 1..=5 {
            let mut t = a_thought(i, i);
            t.speaker_id = "user".to_string();
            user_ids.push(insert_on(&con, &t).unwrap());
        }
        let mut bot = a_thought(1, 1);
        bot.speaker_id = "bot1".to_string();
        insert_on(&con, &bot).unwrap();

        let recent = recent_for_on(&con, 1, "user", 3).unwrap();
        assert_eq!(
            recent.iter().map(|t| t.id).collect::<Vec<_>>(),
            user_ids[2..].to_vec()
        );
    }

    #[test]
    fn recent_for_returns_fewer_than_limit_when_fewer_exist() {
        let (_dir, con) = fresh_db();
        let id = insert_on(&con, &a_thought(1, 1)).unwrap();
        let recent = recent_for_on(&con, 1, "user", 10).unwrap();
        assert_eq!(recent.iter().map(|t| t.id).collect::<Vec<_>>(), vec![id]);
    }

    #[test]
    fn latest_for_is_none_for_a_speaker_with_no_rows() {
        let (_dir, con) = fresh_db();
        assert!(latest_for_on(&con, 1, "user").unwrap().is_none());
    }

    #[test]
    fn latest_for_returns_the_highest_id_for_that_speaker() {
        let (_dir, con) = fresh_db();
        insert_on(&con, &a_thought(1, 1)).unwrap();
        let last = insert_on(&con, &a_thought(2, 2)).unwrap();
        let mut bot = a_thought(1, 1);
        bot.speaker_id = "bot1".to_string();
        insert_on(&con, &bot).unwrap();

        let latest = latest_for_on(&con, 1, "user").unwrap().unwrap();
        assert_eq!(latest.id, last);
    }

    #[test]
    fn in_range_includes_overlapping_and_excludes_disjoint_ranges() {
        let (_dir, con) = fresh_db();
        let starts_before = insert_on(&con, &a_thought(5, 9)).unwrap();
        let ends_after = insert_on(&con, &a_thought(11, 15)).unwrap();
        let fully_inside = insert_on(&con, &a_thought(9, 10)).unwrap();
        let outside = insert_on(&con, &a_thought(20, 25)).unwrap();

        let in_range = in_range_on(&con, 1, 8, 12).unwrap();
        let ids: Vec<i64> = in_range.iter().map(|t| t.id).collect();
        assert!(ids.contains(&starts_before));
        assert!(ids.contains(&ends_after));
        assert!(ids.contains(&fully_inside));
        assert!(!ids.contains(&outside));
    }

    #[test]
    fn update_text_rewrites_and_flips_edited() {
        let (_dir, con) = fresh_db();
        let id = insert_on(&con, &a_thought(1, 1)).unwrap();

        update_text_on(&con, id, "the user's own words").unwrap();

        let got = get_on(&con, id).unwrap().unwrap();
        assert_eq!(got.text, "the user's own words");
        assert!(got.edited);
    }

    #[test]
    fn update_text_on_an_unknown_id_is_query_returned_no_rows() {
        let (_dir, con) = fresh_db();
        let err = update_text_on(&con, 999, "x").unwrap_err();
        assert!(matches!(err, Error::QueryReturnedNoRows));
    }

    #[test]
    fn delete_on_an_unknown_id_is_query_returned_no_rows() {
        let (_dir, con) = fresh_db();
        let err = delete_on(&con, 999).unwrap_err();
        assert!(matches!(err, Error::QueryReturnedNoRows));
    }

    #[test]
    fn delete_removes_the_row() {
        let (_dir, con) = fresh_db();
        let id = insert_on(&con, &a_thought(1, 1)).unwrap();
        delete_on(&con, id).unwrap();
        assert!(get_on(&con, id).unwrap().is_none());
    }

    #[test]
    fn delete_from_removes_rows_spanning_or_after_the_message_id_and_returns_them() {
        let (_dir, con) = fresh_db();
        let earlier = insert_on(&con, &a_thought(1, 4)).unwrap();
        let spanning = insert_on(&con, &a_thought(3, 8)).unwrap();
        let later = insert_on(&con, &a_thought(10, 12)).unwrap();
        let mut other_companion = a_thought(10, 12);
        other_companion.companion_id = 2;
        let other_id = insert_on(&con, &other_companion).unwrap();

        let deleted = delete_from_on(&con, 1, 5).unwrap();

        assert_eq!(
            deleted.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![spanning, later]
        );
        assert!(get_on(&con, earlier).unwrap().is_some());
        assert!(get_on(&con, spanning).unwrap().is_none());
        assert!(get_on(&con, later).unwrap().is_none());
        assert!(get_on(&con, other_id).unwrap().is_some());
    }

    #[test]
    fn delete_from_with_no_matching_rows_returns_an_empty_vec() {
        let (_dir, con) = fresh_db();
        insert_on(&con, &a_thought(1, 2)).unwrap();
        let deleted = delete_from_on(&con, 1, 100).unwrap();
        assert!(deleted.is_empty());
    }

    #[test]
    fn recording_store_matches_sqlite_ordering_and_error_variants() {
        let store = RecordingStore::new();

        let id1 = store.insert(a_thought(1, 2)).unwrap();
        let id2 = store.insert(a_thought(3, 4)).unwrap();
        assert_eq!(
            store
                .list(1)
                .unwrap()
                .iter()
                .map(|t| t.id)
                .collect::<Vec<_>>(),
            vec![id1, id2]
        );

        assert!(store.latest_for(1, "user").unwrap().unwrap().id == id2);
        assert!(store.latest_for(1, "nobody").unwrap().is_none());

        store.update_text(id1, "edited text").unwrap();
        assert_eq!(store.get(id1).unwrap().unwrap().text, "edited text");
        assert!(store.get(id1).unwrap().unwrap().edited);
        assert!(matches!(
            store.update_text(999, "x").unwrap_err(),
            Error::QueryReturnedNoRows
        ));

        let in_range = store.in_range(1, 2, 3).unwrap();
        assert_eq!(
            in_range.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![id1, id2]
        );

        assert!(matches!(
            store.delete(999).unwrap_err(),
            Error::QueryReturnedNoRows
        ));
        store.delete(id1).unwrap();
        assert!(store.get(id1).unwrap().is_none());

        let id3 = store.insert(a_thought(10, 12)).unwrap();
        let deleted = store.delete_from(1, 11).unwrap();
        assert_eq!(deleted.iter().map(|t| t.id).collect::<Vec<_>>(), vec![id3]);
        // id2 ([3, 4]) predates the cutoff and must survive; only id3
        // ([10, 12], through_message_id >= 11) was removed.
        assert_eq!(
            store
                .list(1)
                .unwrap()
                .iter()
                .map(|t| t.id)
                .collect::<Vec<_>>(),
            vec![id2]
        );
    }
}
