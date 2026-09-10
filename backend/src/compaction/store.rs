//! The persistence seam for conversation compaction (#171).
//!
//! [`CompactionStore`] is the trait the rest of the feature (later issues'
//! trigger/extract/validate/commit/render modules) is written against, so
//! it can be unit-tested with [`RecordingStore`] instead of the hardwired
//! `companion_database.db` (`Database::open()` has no path parameter),
//! matching how `chat_turn::TurnStore`/`RecordingStore` already do this for
//! the turn lifecycle.
//!
//! [`SqliteCompactionStore`] is the production impl: each trait method
//! opens the shared database, like every other `Database` associated fn,
//! then delegates to a `pub(crate) fn <name>_on(con: &Connection, ..)`
//! helper (the same split `Database::read_config`/`write_config` use), so
//! the unit tests below can run the helpers directly against
//! `Database::open_at(tempdir)`. The `_on` helpers never open a
//! transaction of their own except `insert_facts_on`, which is called
//! *inside* a transaction the trait method (or #175's `commit_checkpoint`)
//! already opened — `Transaction` derefs to `Connection`, so passing `&tx`
//! where these helpers expect `&Connection` composes cleanly.
//!
//! Nothing outside this module's own tests calls `CompactionStore` yet:
//! #172-#181 wire the trait, `SqliteCompactionStore`, and each `_on` helper
//! in progressively as trigger, extraction, rendering, commit, routes and
//! the review UI land. Every method is already exercised by the unit tests
//! below, which is what the #171 acceptance criteria ask for.
#![allow(dead_code)]

use rusqlite::{params, Connection, Error, OptionalExtension, Result, Row, ToSql};

use crate::compaction::types::{Checkpoint, CompactionStatus, Fact, FactDraft, NewDraft, Pin};
// Only named directly by this module's own tests (production code reaches
// these through `NewDraft`/`FactDraft` fields without naming the types).
#[cfg(test)]
use crate::compaction::types::{CompactionTrigger, FactCategory, FactSubject};
use crate::database::{get_current_date, Database};

/// Creates the three compaction tables (and their indexes) if they do not
/// already exist. Shared by `Database::init` (called after the `companion`
/// and `messages` tables it references both exist) and the tests below,
/// which build the schema on a temp-file connection without going through
/// `Database::init`'s hardwired path.
///
/// `ON DELETE CASCADE` on all three foreign keys is required, not
/// optional: `Database::open_at` turns `PRAGMA foreign_keys` on, so
/// without it `DELETE /api/message` would start failing the moment any
/// message is pinned, and #181's clear-chat and pin removal get the
/// cascade for free.
pub(crate) fn create_tables(con: &Connection) -> Result<()> {
    con.execute(
        "CREATE TABLE IF NOT EXISTS compactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            companion_id INTEGER NOT NULL REFERENCES companion(id) ON DELETE CASCADE,
            from_message_id INTEGER NOT NULL,
            through_message_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            trigger TEXT NOT NULL,
            raw_model_output TEXT,
            summary TEXT,
            rolling_summary TEXT,
            attitude_ratings TEXT,
            needs_merge INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            committed_at TEXT
        )",
        [],
    )?;
    con.execute(
        "CREATE TABLE IF NOT EXISTS compaction_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            compaction_id INTEGER NOT NULL REFERENCES compactions(id) ON DELETE CASCADE,
            category TEXT NOT NULL,
            subject TEXT,
            text TEXT NOT NULL,
            quote_speaker TEXT,
            sources TEXT NOT NULL,
            replaces TEXT NOT NULL DEFAULT '[]',
            relation_to TEXT,
            relation TEXT,
            canon INTEGER NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            superseded_by INTEGER REFERENCES compaction_facts(id),
            rejected_reason TEXT
        )",
        [],
    )?;
    con.execute(
        "CREATE TABLE IF NOT EXISTS pinned_messages (
            message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
            pinned_at TEXT NOT NULL
        )",
        [],
    )?;
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_compaction_facts_compaction ON compaction_facts(compaction_id, active)",
        [],
    )?;
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_compactions_companion_status ON compactions(companion_id, status)",
        [],
    )?;
    Ok(())
}

/// Column list shared by every query that reads a full `compactions` row,
/// in the order [`checkpoint_from_row`] expects.
const CHECKPOINT_COLUMNS: &str = "id, companion_id, from_message_id, through_message_id, status, trigger, raw_model_output, summary, rolling_summary, attitude_ratings, needs_merge, created_at, committed_at";

fn checkpoint_from_row(row: &Row) -> Result<Checkpoint> {
    Ok(Checkpoint {
        id: row.get(0)?,
        companion_id: row.get(1)?,
        from_message_id: row.get(2)?,
        through_message_id: row.get(3)?,
        status: row.get(4)?,
        trigger: row.get(5)?,
        raw_model_output: row.get(6)?,
        summary: row.get(7)?,
        rolling_summary: row.get(8)?,
        attitude_ratings: row.get(9)?,
        needs_merge: row.get(10)?,
        created_at: row.get(11)?,
        committed_at: row.get(12)?,
    })
}

/// Column list shared by every query that reads a full `compaction_facts`
/// row, table-qualified since `active_facts_on` reads it through a `JOIN`.
/// In the order [`fact_from_row`] expects.
const FACT_COLUMNS: &str = "compaction_facts.id, compaction_facts.compaction_id, compaction_facts.category, compaction_facts.subject, compaction_facts.text, compaction_facts.quote_speaker, compaction_facts.sources, compaction_facts.replaces, compaction_facts.relation_to, compaction_facts.relation, compaction_facts.canon, compaction_facts.active, compaction_facts.superseded_by, compaction_facts.rejected_reason";

/// Maps a malformed stored `sources`/`replaces` JSON array to
/// `FromSqlConversionFailure` rather than silently falling back to an
/// empty vec.
fn parse_id_json_column<T: serde::de::DeserializeOwned>(
    raw: &str,
    column_index: usize,
) -> Result<T> {
    serde_json::from_str(raw).map_err(|e| {
        Error::FromSqlConversionFailure(column_index, rusqlite::types::Type::Text, Box::new(e))
    })
}

fn fact_from_row(row: &Row) -> Result<Fact> {
    let sources_json: String = row.get(6)?;
    let replaces_json: String = row.get(7)?;
    Ok(Fact {
        id: row.get(0)?,
        compaction_id: row.get(1)?,
        category: row.get(2)?,
        subject: row.get(3)?,
        text: row.get(4)?,
        quote_speaker: row.get(5)?,
        sources: parse_id_json_column(&sources_json, 6)?,
        replaces: parse_id_json_column(&replaces_json, 7)?,
        relation_to: row.get(8)?,
        relation: row.get(9)?,
        canon: row.get(10)?,
        active: row.get(11)?,
        superseded_by: row.get(12)?,
        rejected_reason: row.get(13)?,
    })
}

/// Inserts a new `draft`-status checkpoint and returns its id.
///
/// `SqliteFailure(ConstraintViolation)` if `draft.companion_id` does not
/// exist.
pub(crate) fn insert_draft_on(con: &Connection, draft: &NewDraft) -> Result<i64> {
    con.execute(
        "INSERT INTO compactions (companion_id, from_message_id, through_message_id, status, trigger, raw_model_output, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            draft.companion_id,
            draft.from_message_id,
            draft.through_message_id,
            &CompactionStatus::Draft as &dyn ToSql,
            &draft.trigger as &dyn ToSql,
            draft.raw_model_output,
            get_current_date(),
        ],
    )?;
    Ok(con.last_insert_rowid())
}

/// `Ok(None)` for an unknown id, never `QueryReturnedNoRows`.
pub(crate) fn get_checkpoint_on(con: &Connection, id: i64) -> Result<Option<Checkpoint>> {
    con.query_row(
        &format!("SELECT {CHECKPOINT_COLUMNS} FROM compactions WHERE id = ?"),
        params![id],
        checkpoint_from_row,
    )
    .optional()
}

/// The single `status = 'draft'` row for `companion_id`, if any.
pub(crate) fn pending_draft_on(con: &Connection, companion_id: i32) -> Result<Option<Checkpoint>> {
    con.query_row(
        &format!(
            "SELECT {CHECKPOINT_COLUMNS} FROM compactions WHERE companion_id = ? AND status = ?"
        ),
        params![companion_id, &CompactionStatus::Draft as &dyn ToSql],
        checkpoint_from_row,
    )
    .optional()
}

/// Every checkpoint for `companion_id`, all statuses, ordered by `id`.
pub(crate) fn list_checkpoints_on(con: &Connection, companion_id: i32) -> Result<Vec<Checkpoint>> {
    let mut stmt = con.prepare(&format!(
        "SELECT {CHECKPOINT_COLUMNS} FROM compactions WHERE companion_id = ? ORDER BY id"
    ))?;
    let rows = stmt.query_map(params![companion_id], checkpoint_from_row)?;
    rows.collect()
}

/// The highest-id `status = 'committed'` row for `companion_id`, if any.
pub(crate) fn latest_committed_on(
    con: &Connection,
    companion_id: i32,
) -> Result<Option<Checkpoint>> {
    con.query_row(
        &format!(
            "SELECT {CHECKPOINT_COLUMNS} FROM compactions WHERE companion_id = ? AND status = ? ORDER BY id DESC LIMIT 1"
        ),
        params![companion_id, &CompactionStatus::Committed as &dyn ToSql],
        checkpoint_from_row,
    )
    .optional()
}

/// Sets `status`, also stamping `committed_at` when transitioning to
/// `Committed`. `QueryReturnedNoRows` when `id` does not exist (checked via
/// `changes() == 0`, so a silent no-op is impossible).
pub(crate) fn update_status_on(con: &Connection, id: i64, status: CompactionStatus) -> Result<()> {
    let changed = if status == CompactionStatus::Committed {
        con.execute(
            "UPDATE compactions SET status = ?, committed_at = ? WHERE id = ?",
            params![&status as &dyn ToSql, get_current_date(), id],
        )?
    } else {
        con.execute(
            "UPDATE compactions SET status = ? WHERE id = ?",
            params![&status as &dyn ToSql, id],
        )?
    };
    if changed == 0 {
        return Err(Error::QueryReturnedNoRows);
    }
    Ok(())
}

/// Fills in `raw_model_output`/`summary`/`attitude_ratings` on an existing
/// checkpoint row. `QueryReturnedNoRows` if `id` does not exist (checked via
/// `changes() == 0`, matching `update_status_on`).
pub(crate) fn set_extraction_result_on(
    con: &Connection,
    id: i64,
    raw_model_output: Option<String>,
    summary: Option<String>,
    attitude_ratings: Option<String>,
) -> Result<()> {
    let changed = con.execute(
        "UPDATE compactions SET raw_model_output = ?, summary = ?, attitude_ratings = ? WHERE id = ?",
        params![raw_model_output, summary, attitude_ratings, id],
    )?;
    if changed == 0 {
        return Err(Error::QueryReturnedNoRows);
    }
    Ok(())
}

/// Inserts `facts` for `compaction_id`, one row per draft, returning the
/// new ids in input order. `replaces`, `relation_to`, and `relation` are
/// written exactly as given; the rows named in `replaces` are never
/// touched here (that is `supersede`, called by #175's commit).
///
/// Takes no transaction of its own: `SqliteFailure(ConstraintViolation)` if
/// `compaction_id` does not exist surfaces mid-loop exactly like any other
/// statement, and it is the caller's job (the trait method below, or
/// #175's `commit_checkpoint`) to wrap the call in one so a failure partway
/// through never leaves a partial set of rows committed.
pub(crate) fn insert_facts_on(
    con: &Connection,
    compaction_id: i64,
    facts: &[FactDraft],
) -> Result<Vec<i64>> {
    let mut ids = Vec::with_capacity(facts.len());
    for draft in facts {
        let active = draft.rejected_reason.is_none();
        let sources_json =
            serde_json::to_string(&draft.sources).expect("Vec<i32> always serializes");
        let replaces_json =
            serde_json::to_string(&draft.replaces).expect("Vec<i64> always serializes");
        con.execute(
            "INSERT INTO compaction_facts (compaction_id, category, subject, text, quote_speaker, sources, replaces, relation_to, relation, canon, active, rejected_reason)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                compaction_id,
                &draft.category as &dyn ToSql,
                draft.subject.clone(),
                draft.text,
                draft.quote_speaker,
                sources_json,
                replaces_json,
                draft.relation_to.clone(),
                draft.relation,
                draft.canon,
                active,
                draft.rejected_reason,
            ],
        )?;
        ids.push(con.last_insert_rowid());
    }
    Ok(ids)
}

/// Every fact whose checkpoint is `committed`, ordered by insertion order.
/// The `status = 'committed'` predicate is load-bearing: `insert_facts_on`
/// derives `active = 1` for every un-rejected item at *draft* time, before
/// the user has reviewed anything, so without it a pending draft's rows
/// would render into the prompt and a discarded draft's rows would linger
/// forever. Scoping to committed checkpoints means `discard` only needs
/// `update_status(Discarded)` and `commit_checkpoint` needs no extra flag
/// flip: the rows become visible the moment the checkpoint's status flips.
pub(crate) fn active_facts_on(con: &Connection, companion_id: i32) -> Result<Vec<Fact>> {
    let mut stmt = con.prepare(&format!(
        "SELECT {FACT_COLUMNS} FROM compaction_facts
         JOIN compactions c ON c.id = compaction_facts.compaction_id
         WHERE c.companion_id = ? AND c.status = ? AND compaction_facts.active = 1 AND compaction_facts.rejected_reason IS NULL
         ORDER BY compaction_facts.id"
    ))?;
    let rows = stmt.query_map(
        params![companion_id, &CompactionStatus::Committed as &dyn ToSql],
        fact_from_row,
    )?;
    rows.collect()
}

/// Every fact for `compaction_id`, including rejected/inactive rows (the
/// review card in #180 needs them), ordered by `id`.
pub(crate) fn facts_for_on(con: &Connection, compaction_id: i64) -> Result<Vec<Fact>> {
    let mut stmt = con.prepare(&format!(
        "SELECT {FACT_COLUMNS} FROM compaction_facts WHERE compaction_id = ? ORDER BY compaction_facts.id"
    ))?;
    let rows = stmt.query_map(params![compaction_id], fact_from_row)?;
    rows.collect()
}

/// Marks `fact_id` inactive and superseded by `by`. `QueryReturnedNoRows`
/// if `fact_id` is unknown; `SqliteFailure(ConstraintViolation)` if `by` is
/// unknown (enforced by the `superseded_by` foreign key).
pub(crate) fn supersede_on(con: &Connection, fact_id: i64, by: i64) -> Result<()> {
    let changed = con.execute(
        "UPDATE compaction_facts SET active = 0, superseded_by = ? WHERE id = ?",
        params![by, fact_id],
    )?;
    if changed == 0 {
        return Err(Error::QueryReturnedNoRows);
    }
    Ok(())
}

/// `None` = never compacted (or reset by #181's clear-chat) *and* an
/// unknown `companion_id` — matching `RecordingStore::compacted_through`,
/// which has no way to distinguish the two either (a missing map entry
/// flattens to `None`). `.optional()` turns the `QueryReturnedNoRows` an
/// unmatched `WHERE id = ?` would otherwise raise into that same `Ok(None)`;
/// `.flatten()` then collapses `Option<Option<i32>>` (row found or not, and
/// separately, `compacted_through` NULL or not) into the one `Option<i32>`
/// this returns.
pub(crate) fn compacted_through_on(con: &Connection, companion_id: i32) -> Result<Option<i32>> {
    con.query_row(
        "SELECT compacted_through FROM companion WHERE id = ?",
        params![companion_id],
        |row| row.get(0),
    )
    .optional()
    .map(Option::flatten)
}

pub(crate) fn set_compacted_through_on(
    con: &Connection,
    companion_id: i32,
    through: Option<i32>,
) -> Result<()> {
    con.execute(
        "UPDATE companion SET compacted_through = ? WHERE id = ?",
        params![through, companion_id],
    )?;
    Ok(())
}

/// Idempotent: `INSERT OR IGNORE` on the message id primary key.
/// `SqliteFailure(ConstraintViolation)` for an unknown message (`OR IGNORE`
/// only suppresses the primary-key conflict, not the foreign key one).
pub(crate) fn pin_on(con: &Connection, message_id: i32) -> Result<()> {
    con.execute(
        "INSERT OR IGNORE INTO pinned_messages (message_id, pinned_at) VALUES (?, ?)",
        params![message_id, get_current_date()],
    )?;
    Ok(())
}

/// No error if `message_id` was not pinned.
pub(crate) fn unpin_on(con: &Connection, message_id: i32) -> Result<()> {
    con.execute(
        "DELETE FROM pinned_messages WHERE message_id = ?",
        params![message_id],
    )?;
    Ok(())
}

pub(crate) fn pins_on(con: &Connection) -> Result<Vec<Pin>> {
    let mut stmt =
        con.prepare("SELECT message_id, pinned_at FROM pinned_messages ORDER BY message_id")?;
    let rows = stmt.query_map([], |row| {
        Ok(Pin {
            message_id: row.get(0)?,
            pinned_at: row.get(1)?,
        })
    })?;
    rows.collect()
}

/// The persistence seam between the rest of the compaction feature and the
/// database, so it can be unit-tested against [`RecordingStore`] instead of
/// the hardwired `companion_database.db`.
///
/// #175 extends this with its own single-method transactional
/// `commit_checkpoint(CommitRecord)`, since it owns `CommitRecord` and the
/// merge semantics — this trait does not define that method.
pub trait CompactionStore {
    /// `SqliteFailure(ConstraintViolation)` if `draft.companion_id` does
    /// not exist.
    fn insert_draft(&self, draft: NewDraft) -> Result<i64>;

    /// `Ok(None)` for an unknown id, never `QueryReturnedNoRows`.
    fn get_checkpoint(&self, id: i64) -> Result<Option<Checkpoint>>;

    /// The single `status = 'draft'` row (#172 uses it for "only one draft
    /// pending").
    fn pending_draft(&self, companion_id: i32) -> Result<Option<Checkpoint>>;

    /// Ordered by `id`, all statuses.
    fn list_checkpoints(&self, companion_id: i32) -> Result<Vec<Checkpoint>>;

    /// Highest-id `status = 'committed'` row (#174 renders its
    /// `summary`/`rolling_summary` every turn).
    fn latest_committed(&self, companion_id: i32) -> Result<Option<Checkpoint>>;

    /// Sets `committed_at = now` when `status == Committed`;
    /// `QueryReturnedNoRows` when `id` does not exist.
    fn update_status(&self, id: i64, status: CompactionStatus) -> Result<()>;

    /// Fills in a draft's extraction result: the model's raw output,
    /// summary, and attitude ratings (raw JSON; #176 gives this a typed
    /// shape). #185's `fill_draft` calls this once on success (all three
    /// `Some`) and once per discard path (`raw_model_output` only, the
    /// other two `None`, immediately followed by `update_status(Discarded)`).
    /// `QueryReturnedNoRows` if `id` does not exist.
    fn set_extraction_result(
        &self,
        id: i64,
        raw_model_output: Option<String>,
        summary: Option<String>,
        attitude_ratings: Option<String>,
    ) -> Result<()>;

    /// One transaction, ids in input order; `SqliteFailure(ConstraintViolation)`
    /// if the checkpoint does not exist. Writes `replaces`, `relation_to`,
    /// and `relation` exactly as given; it does not touch the rows named
    /// in `replaces` (that is `supersede`, called by #175's commit).
    fn insert_facts(&self, compaction_id: i64, facts: &[FactDraft]) -> Result<Vec<i64>>;

    /// Only rows whose checkpoint is `committed`; see [`active_facts_on`]
    /// for why that predicate matters.
    fn active_facts(&self, companion_id: i32) -> Result<Vec<Fact>>;

    /// All rows including rejected/inactive (the review card in #180 needs
    /// them).
    fn facts_for(&self, compaction_id: i64) -> Result<Vec<Fact>>;

    /// Sets `active = 0, superseded_by = by`; `QueryReturnedNoRows` if
    /// `fact_id` is unknown; `SqliteFailure(ConstraintViolation)` if `by`
    /// is unknown.
    fn supersede(&self, fact_id: i64, by: i64) -> Result<()>;

    /// `None` = never compacted / reset (what #181's clear-chat needs; #172
    /// and #174 read it every turn).
    fn compacted_through(&self, companion_id: i32) -> Result<Option<i32>>;

    fn set_compacted_through(&self, companion_id: i32, through: Option<i32>) -> Result<()>;

    /// `INSERT OR IGNORE`, idempotent; `ConstraintViolation` for an unknown
    /// message.
    fn pin(&self, message_id: i32) -> Result<()>;

    /// No error if not pinned.
    fn unpin(&self, message_id: i32) -> Result<()>;

    /// Ordered by `message_id`.
    fn pins(&self) -> Result<Vec<Pin>>;
}

/// Production [`CompactionStore`], opening `Database::open()` per call,
/// exactly like every other `Database` associated fn.
pub struct SqliteCompactionStore;

impl CompactionStore for SqliteCompactionStore {
    fn insert_draft(&self, draft: NewDraft) -> Result<i64> {
        let con = Database::open()?;
        insert_draft_on(&con, &draft)
    }

    fn get_checkpoint(&self, id: i64) -> Result<Option<Checkpoint>> {
        let con = Database::open()?;
        get_checkpoint_on(&con, id)
    }

    fn pending_draft(&self, companion_id: i32) -> Result<Option<Checkpoint>> {
        let con = Database::open()?;
        pending_draft_on(&con, companion_id)
    }

    fn list_checkpoints(&self, companion_id: i32) -> Result<Vec<Checkpoint>> {
        let con = Database::open()?;
        list_checkpoints_on(&con, companion_id)
    }

    fn latest_committed(&self, companion_id: i32) -> Result<Option<Checkpoint>> {
        let con = Database::open()?;
        latest_committed_on(&con, companion_id)
    }

    fn update_status(&self, id: i64, status: CompactionStatus) -> Result<()> {
        let con = Database::open()?;
        update_status_on(&con, id, status)
    }

    fn set_extraction_result(
        &self,
        id: i64,
        raw_model_output: Option<String>,
        summary: Option<String>,
        attitude_ratings: Option<String>,
    ) -> Result<()> {
        let con = Database::open()?;
        set_extraction_result_on(&con, id, raw_model_output, summary, attitude_ratings)
    }

    fn insert_facts(&self, compaction_id: i64, facts: &[FactDraft]) -> Result<Vec<i64>> {
        let con = Database::open()?;
        let tx = con.unchecked_transaction()?;
        let ids = insert_facts_on(&tx, compaction_id, facts)?;
        tx.commit()?;
        Ok(ids)
    }

    fn active_facts(&self, companion_id: i32) -> Result<Vec<Fact>> {
        let con = Database::open()?;
        active_facts_on(&con, companion_id)
    }

    fn facts_for(&self, compaction_id: i64) -> Result<Vec<Fact>> {
        let con = Database::open()?;
        facts_for_on(&con, compaction_id)
    }

    fn supersede(&self, fact_id: i64, by: i64) -> Result<()> {
        let con = Database::open()?;
        supersede_on(&con, fact_id, by)
    }

    fn compacted_through(&self, companion_id: i32) -> Result<Option<i32>> {
        let con = Database::open()?;
        compacted_through_on(&con, companion_id)
    }

    fn set_compacted_through(&self, companion_id: i32, through: Option<i32>) -> Result<()> {
        let con = Database::open()?;
        set_compacted_through_on(&con, companion_id, through)
    }

    fn pin(&self, message_id: i32) -> Result<()> {
        let con = Database::open()?;
        pin_on(&con, message_id)
    }

    fn unpin(&self, message_id: i32) -> Result<()> {
        let con = Database::open()?;
        unpin_on(&con, message_id)
    }

    fn pins(&self) -> Result<Vec<Pin>> {
        let con = Database::open()?;
        pins_on(&con)
    }
}

/// In-memory [`CompactionStore`], mirroring `chat_turn::RecordingStore`'s
/// shape so other modules' tests use it the same way. Reproduces the
/// documented error variants (`QueryReturnedNoRows` from
/// `update_status`/`supersede` on unknown ids) so a test written against it
/// behaves like the SQLite one.
#[cfg(test)]
pub(crate) struct RecordingStore {
    pub(crate) checkpoints: std::sync::Mutex<Vec<Checkpoint>>,
    pub(crate) facts: std::sync::Mutex<Vec<Fact>>,
    pub(crate) pins: std::sync::Mutex<std::collections::BTreeSet<i32>>,
    pub(crate) compacted_through: std::sync::Mutex<std::collections::HashMap<i32, Option<i32>>>,
    next_checkpoint_id: std::sync::atomic::AtomicI64,
    next_fact_id: std::sync::atomic::AtomicI64,
}

#[cfg(test)]
impl RecordingStore {
    pub(crate) fn new() -> Self {
        Self {
            checkpoints: std::sync::Mutex::new(Vec::new()),
            facts: std::sync::Mutex::new(Vec::new()),
            pins: std::sync::Mutex::new(std::collections::BTreeSet::new()),
            compacted_through: std::sync::Mutex::new(std::collections::HashMap::new()),
            next_checkpoint_id: std::sync::atomic::AtomicI64::new(0),
            next_fact_id: std::sync::atomic::AtomicI64::new(0),
        }
    }
}

#[cfg(test)]
impl CompactionStore for RecordingStore {
    fn insert_draft(&self, draft: NewDraft) -> Result<i64> {
        let id = self
            .next_checkpoint_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        self.checkpoints.lock().unwrap().push(Checkpoint {
            id,
            companion_id: draft.companion_id,
            from_message_id: draft.from_message_id,
            through_message_id: draft.through_message_id,
            status: CompactionStatus::Draft,
            trigger: draft.trigger,
            raw_model_output: draft.raw_model_output,
            summary: None,
            rolling_summary: None,
            attitude_ratings: None,
            needs_merge: false,
            created_at: get_current_date(),
            committed_at: None,
        });
        Ok(id)
    }

    fn get_checkpoint(&self, id: i64) -> Result<Option<Checkpoint>> {
        Ok(self
            .checkpoints
            .lock()
            .unwrap()
            .iter()
            .find(|c| c.id == id)
            .cloned())
    }

    fn pending_draft(&self, companion_id: i32) -> Result<Option<Checkpoint>> {
        Ok(self
            .checkpoints
            .lock()
            .unwrap()
            .iter()
            .find(|c| c.companion_id == companion_id && c.status == CompactionStatus::Draft)
            .cloned())
    }

    fn list_checkpoints(&self, companion_id: i32) -> Result<Vec<Checkpoint>> {
        Ok(self
            .checkpoints
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.companion_id == companion_id)
            .cloned()
            .collect())
    }

    fn latest_committed(&self, companion_id: i32) -> Result<Option<Checkpoint>> {
        Ok(self
            .checkpoints
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.companion_id == companion_id && c.status == CompactionStatus::Committed)
            .max_by_key(|c| c.id)
            .cloned())
    }

    fn update_status(&self, id: i64, status: CompactionStatus) -> Result<()> {
        let mut checkpoints = self.checkpoints.lock().unwrap();
        let checkpoint = checkpoints
            .iter_mut()
            .find(|c| c.id == id)
            .ok_or(Error::QueryReturnedNoRows)?;
        checkpoint.status = status;
        if status == CompactionStatus::Committed {
            checkpoint.committed_at = Some(get_current_date());
        }
        Ok(())
    }

    fn set_extraction_result(
        &self,
        id: i64,
        raw_model_output: Option<String>,
        summary: Option<String>,
        attitude_ratings: Option<String>,
    ) -> Result<()> {
        let mut checkpoints = self.checkpoints.lock().unwrap();
        let checkpoint = checkpoints
            .iter_mut()
            .find(|c| c.id == id)
            .ok_or(Error::QueryReturnedNoRows)?;
        checkpoint.raw_model_output = raw_model_output;
        checkpoint.summary = summary;
        checkpoint.attitude_ratings = attitude_ratings;
        Ok(())
    }

    fn insert_facts(&self, compaction_id: i64, facts: &[FactDraft]) -> Result<Vec<i64>> {
        let mut ids = Vec::with_capacity(facts.len());
        let mut stored = self.facts.lock().unwrap();
        for draft in facts {
            let id = self
                .next_fact_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            let active = draft.rejected_reason.is_none();
            stored.push(Fact {
                id,
                compaction_id,
                category: draft.category,
                subject: draft.subject.clone(),
                text: draft.text.clone(),
                quote_speaker: draft.quote_speaker.clone(),
                sources: draft.sources.clone(),
                replaces: draft.replaces.clone(),
                relation_to: draft.relation_to.clone(),
                relation: draft.relation.clone(),
                canon: draft.canon,
                active,
                superseded_by: None,
                rejected_reason: draft.rejected_reason.clone(),
            });
            ids.push(id);
        }
        Ok(ids)
    }

    fn active_facts(&self, companion_id: i32) -> Result<Vec<Fact>> {
        let committed_ids: std::collections::HashSet<i64> = self
            .checkpoints
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.companion_id == companion_id && c.status == CompactionStatus::Committed)
            .map(|c| c.id)
            .collect();
        Ok(self
            .facts
            .lock()
            .unwrap()
            .iter()
            .filter(|f| {
                committed_ids.contains(&f.compaction_id) && f.active && f.rejected_reason.is_none()
            })
            .cloned()
            .collect())
    }

    fn facts_for(&self, compaction_id: i64) -> Result<Vec<Fact>> {
        Ok(self
            .facts
            .lock()
            .unwrap()
            .iter()
            .filter(|f| f.compaction_id == compaction_id)
            .cloned()
            .collect())
    }

    fn supersede(&self, fact_id: i64, by: i64) -> Result<()> {
        let mut facts = self.facts.lock().unwrap();
        let fact = facts
            .iter_mut()
            .find(|f| f.id == fact_id)
            .ok_or(Error::QueryReturnedNoRows)?;
        fact.active = false;
        fact.superseded_by = Some(by);
        Ok(())
    }

    fn compacted_through(&self, companion_id: i32) -> Result<Option<i32>> {
        Ok(self
            .compacted_through
            .lock()
            .unwrap()
            .get(&companion_id)
            .copied()
            .flatten())
    }

    fn set_compacted_through(&self, companion_id: i32, through: Option<i32>) -> Result<()> {
        self.compacted_through
            .lock()
            .unwrap()
            .insert(companion_id, through);
        Ok(())
    }

    fn pin(&self, message_id: i32) -> Result<()> {
        self.pins.lock().unwrap().insert(message_id);
        Ok(())
    }

    fn unpin(&self, message_id: i32) -> Result<()> {
        self.pins.lock().unwrap().remove(&message_id);
        Ok(())
    }

    fn pins(&self) -> Result<Vec<Pin>> {
        Ok(self
            .pins
            .lock()
            .unwrap()
            .iter()
            .map(|&message_id| Pin {
                message_id,
                pinned_at: get_current_date(),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds the schema `create_tables` needs (`companion`, `messages`,
    /// then the three compaction tables) on a temp-file connection, and
    /// seeds one companion row and three message rows. Mirrors how
    /// `database.rs`'s own config tests build a local `create_config_table`
    /// rather than calling into `Database::init`'s hardwired path.
    fn fresh_db() -> (tempfile::TempDir, Connection) {
        let dir = tempfile::TempDir::new().unwrap();
        let con = Database::open_at(dir.path().join("t.db")).unwrap();
        con.execute(crate::database::messages_ddl(), []).unwrap();
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
        for i in 1..=3 {
            con.execute(
                "INSERT INTO messages (id, ai, speaker_id, content, created_at) VALUES (?, 0, 'user', 'hi', 'now')",
                params![i],
            )
            .unwrap();
        }
        (dir, con)
    }

    fn a_draft() -> NewDraft {
        NewDraft {
            companion_id: 1,
            from_message_id: 1,
            through_message_id: 3,
            trigger: CompactionTrigger::Threshold,
            raw_model_output: None,
        }
    }

    #[test]
    fn draft_insert_is_found_by_pending_draft_until_committed() {
        let (_dir, con) = fresh_db();

        let id = insert_draft_on(&con, &a_draft()).unwrap();
        let pending = pending_draft_on(&con, 1).unwrap().unwrap();
        assert_eq!(pending.id, id);
        assert_eq!(pending.status, CompactionStatus::Draft);
        assert!(pending.committed_at.is_none());

        update_status_on(&con, id, CompactionStatus::Committed).unwrap();

        let checkpoint = get_checkpoint_on(&con, id).unwrap().unwrap();
        assert_eq!(checkpoint.status, CompactionStatus::Committed);
        assert!(checkpoint.committed_at.is_some());
        assert!(pending_draft_on(&con, 1).unwrap().is_none());
    }

    #[test]
    fn get_checkpoint_returns_none_for_an_unknown_id() {
        let (_dir, con) = fresh_db();
        assert!(get_checkpoint_on(&con, 999).unwrap().is_none());
    }

    #[test]
    fn update_status_on_an_unknown_id_is_query_returned_no_rows() {
        let (_dir, con) = fresh_db();
        let err = update_status_on(&con, 999, CompactionStatus::Committed).unwrap_err();
        assert!(matches!(err, Error::QueryReturnedNoRows));
    }

    #[test]
    fn set_extraction_result_on_fills_in_summary_and_attitude_and_errors_on_an_unknown_id() {
        let (_dir, con) = fresh_db();
        let compaction_id = insert_draft_on(&con, &a_draft()).unwrap();

        set_extraction_result_on(
            &con,
            compaction_id,
            Some("raw output".to_string()),
            Some("a summary".to_string()),
            Some("{\"trust\":50}".to_string()),
        )
        .unwrap();

        let checkpoint = get_checkpoint_on(&con, compaction_id).unwrap().unwrap();
        assert_eq!(checkpoint.raw_model_output.as_deref(), Some("raw output"));
        assert_eq!(checkpoint.summary.as_deref(), Some("a summary"));
        assert_eq!(
            checkpoint.attitude_ratings.as_deref(),
            Some("{\"trust\":50}")
        );

        let err = set_extraction_result_on(&con, 999, None, None, None).unwrap_err();
        assert!(matches!(err, Error::QueryReturnedNoRows));
    }

    #[test]
    fn insert_facts_stores_a_rejected_item_inactive() {
        let (_dir, con) = fresh_db();
        let compaction_id = insert_draft_on(&con, &a_draft()).unwrap();

        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "rejected item".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: false,
            rejected_reason: Some("not relevant".to_string()),
        };
        let ids = insert_facts_on(&con, compaction_id, std::slice::from_ref(&draft)).unwrap();

        let stored = facts_for_on(&con, compaction_id).unwrap();
        let fact = stored.iter().find(|f| f.id == ids[0]).unwrap();
        assert!(!fact.active);
        assert_eq!(fact.rejected_reason.as_deref(), Some("not relevant"));
    }

    #[test]
    fn insert_facts_then_facts_for_round_trips_replaces_and_relation_columns() {
        let (_dir, con) = fresh_db();
        let compaction_id = insert_draft_on(&con, &a_draft()).unwrap();

        let companion_state = FactDraft {
            category: FactCategory::CompanionState,
            subject: Some(FactSubject::Companion),
            text: "is happier now".to_string(),
            quote_speaker: None,
            sources: vec![2],
            replaces: vec![1],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let person = FactDraft {
            category: FactCategory::Person,
            subject: Some(FactSubject::Person("Ann".to_string())),
            text: "Ann is the user's sister".to_string(),
            quote_speaker: None,
            sources: vec![3],
            replaces: vec![],
            relation_to: Some(FactSubject::User),
            relation: Some("sister".to_string()),
            canon: true,
            rejected_reason: None,
        };

        let ids = insert_facts_on(
            &con,
            compaction_id,
            &[companion_state.clone(), person.clone()],
        )
        .unwrap();
        let stored = facts_for_on(&con, compaction_id).unwrap();

        let stored_companion_state = stored.iter().find(|f| f.id == ids[0]).unwrap();
        assert_eq!(stored_companion_state.replaces, vec![1]);
        assert_eq!(stored_companion_state.relation_to, None);
        assert_eq!(stored_companion_state.relation, None);

        let stored_person = stored.iter().find(|f| f.id == ids[1]).unwrap();
        assert_eq!(stored_person.replaces, Vec::<i64>::new());
        assert_eq!(stored_person.relation_to, Some(FactSubject::User));
        assert_eq!(stored_person.relation.as_deref(), Some("sister"));
        assert_eq!(
            stored_person.subject,
            Some(FactSubject::Person("Ann".to_string()))
        );
    }

    #[test]
    fn active_facts_excludes_inactive_rejected_and_superseded_rows() {
        let (_dir, con) = fresh_db();
        let compaction_id = insert_draft_on(&con, &a_draft()).unwrap();
        update_status_on(&con, compaction_id, CompactionStatus::Committed).unwrap();

        let live = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "live".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let rejected = FactDraft {
            rejected_reason: Some("no".to_string()),
            text: "rejected".to_string(),
            ..live.clone()
        };
        let to_supersede = FactDraft {
            text: "will be superseded".to_string(),
            ..live.clone()
        };

        let live_ids = insert_facts_on(&con, compaction_id, std::slice::from_ref(&live)).unwrap();
        insert_facts_on(&con, compaction_id, std::slice::from_ref(&rejected)).unwrap();
        let superseded_ids =
            insert_facts_on(&con, compaction_id, std::slice::from_ref(&to_supersede)).unwrap();
        supersede_on(&con, superseded_ids[0], live_ids[0]).unwrap();

        let active = active_facts_on(&con, 1).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, live_ids[0]);
    }

    #[test]
    fn active_facts_excludes_pending_draft_rows_sqlite() {
        let (_dir, con) = fresh_db();
        let compaction_id = insert_draft_on(&con, &a_draft()).unwrap();

        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "pending".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        insert_facts_on(&con, compaction_id, std::slice::from_ref(&draft)).unwrap();

        assert!(active_facts_on(&con, 1).unwrap().is_empty());

        update_status_on(&con, compaction_id, CompactionStatus::Committed).unwrap();
        assert_eq!(active_facts_on(&con, 1).unwrap().len(), 1);

        let second_compaction_id = insert_draft_on(&con, &a_draft()).unwrap();
        insert_facts_on(&con, second_compaction_id, std::slice::from_ref(&draft)).unwrap();
        update_status_on(&con, second_compaction_id, CompactionStatus::Discarded).unwrap();
        assert_eq!(active_facts_on(&con, 1).unwrap().len(), 1);
    }

    #[test]
    fn active_facts_excludes_pending_draft_rows_recording_store() {
        let store = RecordingStore::new();
        let draft = NewDraft {
            companion_id: 1,
            from_message_id: 1,
            through_message_id: 3,
            trigger: CompactionTrigger::Threshold,
            raw_model_output: None,
        };
        let compaction_id = store.insert_draft(draft.clone()).unwrap();

        let fact_draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "pending".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        store
            .insert_facts(compaction_id, std::slice::from_ref(&fact_draft))
            .unwrap();

        assert!(store.active_facts(1).unwrap().is_empty());

        store
            .update_status(compaction_id, CompactionStatus::Committed)
            .unwrap();
        assert_eq!(store.active_facts(1).unwrap().len(), 1);

        let second_compaction_id = store.insert_draft(draft).unwrap();
        store
            .insert_facts(second_compaction_id, std::slice::from_ref(&fact_draft))
            .unwrap();
        store
            .update_status(second_compaction_id, CompactionStatus::Discarded)
            .unwrap();
        assert_eq!(store.active_facts(1).unwrap().len(), 1);
    }

    #[test]
    fn supersede_on_an_unknown_id_is_query_returned_no_rows() {
        let (_dir, con) = fresh_db();
        let compaction_id = insert_draft_on(&con, &a_draft()).unwrap();
        let draft = FactDraft {
            category: FactCategory::Milestone,
            subject: None,
            text: "live".to_string(),
            quote_speaker: None,
            sources: vec![1],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            rejected_reason: None,
        };
        let ids = insert_facts_on(&con, compaction_id, std::slice::from_ref(&draft)).unwrap();

        let err = supersede_on(&con, 999, ids[0]).unwrap_err();
        assert!(matches!(err, Error::QueryReturnedNoRows));
    }

    #[test]
    fn pin_twice_is_idempotent_and_deleting_the_message_cascades_the_pin() {
        let (_dir, con) = fresh_db();
        pin_on(&con, 1).unwrap();
        pin_on(&con, 1).unwrap();
        assert_eq!(pins_on(&con).unwrap().len(), 1);

        con.execute("DELETE FROM messages WHERE id = ?", params![1])
            .unwrap();
        assert!(pins_on(&con).unwrap().is_empty());
    }

    #[test]
    fn unpin_an_unpinned_message_is_not_an_error() {
        let (_dir, con) = fresh_db();
        unpin_on(&con, 1).unwrap();
        assert!(pins_on(&con).unwrap().is_empty());
    }

    #[test]
    fn set_compacted_through_none_resets() {
        let (_dir, con) = fresh_db();
        assert_eq!(compacted_through_on(&con, 1).unwrap(), None);

        set_compacted_through_on(&con, 1, Some(2)).unwrap();
        assert_eq!(compacted_through_on(&con, 1).unwrap(), Some(2));

        set_compacted_through_on(&con, 1, None).unwrap();
        assert_eq!(compacted_through_on(&con, 1).unwrap(), None);
    }

    #[test]
    fn compacted_through_on_an_unknown_companion_is_ok_none_not_an_error() {
        let (_dir, con) = fresh_db();
        // Matches `RecordingStore::compacted_through`, which has no
        // separate "unknown companion" error path either.
        assert_eq!(compacted_through_on(&con, 999).unwrap(), None);
    }
}
