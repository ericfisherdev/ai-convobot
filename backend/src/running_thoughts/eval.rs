//! A replay eval for running-thought quality: regenerates every stored note
//! of a *copy* of a real database through the production path
//! (`generate_thought` on `ResidentCharacterModel`), chaining the
//! regenerated notes rather than the stored ones, and writes a JSON report
//! of stored-vs-regenerated text per note. This is how the 2026-09-12 prompt,
//! sampler, sentence-stop and own-name-guard changes were measured; run it
//! again before changing any of them.
//!
//! Env-gated `#[test]`, same shape and same reasons as `compaction::eval`
//! (binary crate, no `lib.rs`, so an in-crate test is the only place the
//! pipeline is callable directly):
//!
//! ```text
//! cd backend
//! AI_COMPANION_THOUGHT_DB=/path/to/COPY.db \
//! AI_COMPANION_THOUGHT_OUT=/path/to/report.json \
//!   cargo test --bin ai-companion running_thoughts::eval::thought_eval \
//!   -- --nocapture --test-threads=1
//! ```
//!
//! The DB is copied into a temp data dir before `paths::init`, so the file
//! named by the variable is never written. The name filter is not optional:
//! `paths::init` can only run once per process.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;

use serde::Serialize;

use crate::database::Database;
use crate::llm::{
    CharacterCompletion, CharacterModel, PromptSpeakers, ResidentCharacterModel, SqliteTranscript,
};
use crate::participants::{ParticipantId, ParticipantRegistry};
use crate::running_thoughts::generate::generate_thought;
use crate::running_thoughts::hook::thought_range;
use crate::running_thoughts::prompt::{ThoughtInputs, THOUGHT_CHAIN_LENGTH};
use crate::running_thoughts::store::{RunningThoughtStore, SqliteRunningThoughtStore};
use crate::running_thoughts::types::RunningThought;

const DB_ENV: &str = "AI_COMPANION_THOUGHT_DB";
const OUT_ENV: &str = "AI_COMPANION_THOUGHT_OUT";

/// Wraps the production model to count calls and keep raw completions, so
/// the report can show when the own-name guard retried and what it rejected.
struct RecordingModel {
    inner: ResidentCharacterModel,
    calls: Cell<usize>,
    raw: RefCell<Vec<String>>,
}

impl CharacterModel for RecordingModel {
    fn complete_in_character(
        &self,
        system: &str,
        user: &str,
        max_tokens: usize,
        keep_going: &mut dyn FnMut(&str) -> bool,
    ) -> std::io::Result<CharacterCompletion> {
        self.calls.set(self.calls.get() + 1);
        let completion = self
            .inner
            .complete_in_character(system, user, max_tokens, keep_going)?;
        self.raw.borrow_mut().push(completion.text.clone());
        Ok(completion)
    }
}

#[derive(Serialize)]
struct NoteResult {
    thought_id: i64,
    from_message_id: i32,
    through_message_id: i32,
    db_text: String,
    db_edited: bool,
    new_text: Option<String>,
    model_calls: usize,
    raw_attempts: Vec<String>,
}

#[derive(Serialize)]
struct Report {
    notes: Vec<NoteResult>,
}

fn write_report(path: &PathBuf, report: &Report) {
    let json = serde_json::to_string_pretty(report).expect("report should serialise");
    std::fs::write(path, json).expect("failed to write the report");
}

#[test]
fn thought_eval() {
    let Ok(db_source) = std::env::var(DB_ENV) else {
        println!("skipped: set {DB_ENV} to a COPY of a companion database to run the thought eval");
        return;
    };
    let out_path = std::env::var(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| panic!("set {OUT_ENV} to the report path"));

    let data_dir = tempfile::tempdir().expect("failed to create the harness data dir");
    std::fs::copy(&db_source, data_dir.path().join("companion_database.db"))
        .expect("failed to copy the database into the harness dir");
    crate::paths::init(data_dir.path().to_path_buf())
        .expect("paths::init should not already be set — run this test with a name filter");
    Database::init().expect("failed to initialise the harness database");

    let companion = Database::get_companion_data().expect("companion");
    let companion_id = Database::get_companion_id().expect("companion id");
    let user = Database::get_user_data().expect("user");
    let speakers = PromptSpeakers {
        registry: ParticipantRegistry::solo(&user.name, &companion.name, None),
        self_id: ParticipantId::CHAR,
    };

    let stored = SqliteRunningThoughtStore
        .list(companion_id)
        .expect("list thoughts");
    println!("\n# Thought eval — {} stored notes", stored.len());

    let model = RecordingModel {
        inner: ResidentCharacterModel,
        calls: Cell::new(0),
        raw: RefCell::new(Vec::new()),
    };
    let mut chain: Vec<RunningThought> = Vec::new();
    let mut report = Report { notes: Vec::new() };

    for (index, original) in stored.iter().enumerate() {
        let round = thought_range(
            &SqliteTranscript,
            original.from_message_id,
            original.through_message_id,
        )
        .expect("round messages");
        let previous_start = chain.len().saturating_sub(THOUGHT_CHAIN_LENGTH);
        let inputs = ThoughtInputs {
            companion_id,
            speaker_id: ParticipantId::CHAR,
            previous: chain[previous_start..].to_vec(),
            round,
            from_message_id: original.from_message_id,
            through_message_id: original.through_message_id,
        };

        println!(
            "\n-- note {}/{} (id {}, msgs {}..{}) --",
            index + 1,
            stored.len(),
            original.id,
            original.from_message_id,
            original.through_message_id
        );
        model.calls.set(0);
        model.raw.borrow_mut().clear();
        let generated = generate_thought(&inputs, &companion, &speakers, &model, &|new| {
            Ok(RunningThought {
                id: original.id,
                companion_id: new.companion_id,
                speaker_id: new.speaker_id,
                from_message_id: new.from_message_id,
                through_message_id: new.through_message_id,
                text: new.text,
                edited: new.edited,
                created_at: String::new(),
            })
        });
        let new_text = match generated {
            Ok(thought) => {
                chain.push(thought.clone());
                Some(thought.text)
            }
            Err(e) => {
                println!("!! {e}");
                None
            }
        };
        println!("DB : {}", original.text);
        println!("NEW: {}", new_text.as_deref().unwrap_or("<none>"));

        report.notes.push(NoteResult {
            thought_id: original.id,
            from_message_id: original.from_message_id,
            through_message_id: original.through_message_id,
            db_text: original.text.clone(),
            db_edited: original.edited,
            new_text,
            model_calls: model.calls.get(),
            raw_attempts: model.raw.borrow().clone(),
        });
        write_report(&out_path, &report);
    }

    assert!(!report.notes.is_empty(), "no stored notes to replay");
}
