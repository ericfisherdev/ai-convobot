//! Runs one thought-generation call and persists the result (#216).
//! `Database`-free: persona, names, previous notes and the round are all
//! injected, so #220's joiner can call [`generate_thought`] with its own
//! card, registry snapshot and local store.

use crate::database::CompanionView;
use crate::llm::{CharacterModel, PromptSpeakers};
use crate::running_thoughts::prompt::{
    build_thought_prompt, clean_thought, ThoughtInputs, THOUGHT_MAX_TOKENS,
};
use crate::running_thoughts::store::RunningThoughtStore;
use crate::running_thoughts::types::{NewRunningThought, RunningThought};

/// Why a thought was not produced/stored. A failure here must never fail
/// the reply that follows it — every caller logs this and moves on.
#[derive(Debug)]
pub enum ThoughtError {
    /// The character model failed to complete.
    Generate(std::io::Error),
    /// The model produced only whitespace once cleaned.
    Empty,
    /// The store rejected the insert (or the follow-up read of it).
    Store(rusqlite::Error),
    /// Reading this speaker's chained context or the round's own messages
    /// failed — the seam-based half of building a [`ThoughtInputs`]
    /// (`running_thoughts::hook::thought_inputs_for_range`).
    Inputs(std::io::Error),
}

impl std::fmt::Display for ThoughtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThoughtError::Generate(e) => write!(f, "thought generation failed: {e}"),
            ThoughtError::Empty => write!(f, "the model produced an empty thought"),
            ThoughtError::Store(e) => write!(f, "failed to persist the thought: {e}"),
            ThoughtError::Inputs(e) => write!(f, "failed to read thought inputs: {e}"),
        }
    }
}

impl std::error::Error for ThoughtError {}

/// Builds the prompt, runs the character model, cleans the note, persists
/// it via `insert`. The one code path for both a live round
/// (`chat_turn::PendingTurn::think`) and #217's regenerate route. Nothing is
/// written on `Generate`/`Empty`.
pub fn generate_thought(
    inputs: &ThoughtInputs,
    companion: &CompanionView,
    speakers: &PromptSpeakers,
    model: &dyn CharacterModel,
    insert: &dyn Fn(NewRunningThought) -> rusqlite::Result<RunningThought>,
) -> Result<RunningThought, ThoughtError> {
    let prompt = build_thought_prompt(inputs, &companion.persona, speakers);
    let raw = model
        .complete_in_character(&prompt.system, &prompt.user, THOUGHT_MAX_TOKENS)
        .map_err(ThoughtError::Generate)?;
    let text = clean_thought(&raw, speakers.self_name()).ok_or(ThoughtError::Empty)?;

    insert(NewRunningThought {
        companion_id: inputs.companion_id,
        speaker_id: inputs.speaker_id.to_string(),
        from_message_id: inputs.from_message_id,
        through_message_id: inputs.through_message_id,
        text,
        edited: false,
    })
    .map_err(ThoughtError::Store)
}

/// [`generate_thought`] over a [`RunningThoughtStore`]: `insert` then `get`
/// (the store's `insert` returns only the id). What #220's joiner thinker
/// (`multiplayer::remote_generation::think_into`) calls with
/// `SqliteRunningThoughtStore`. #217's regenerate loop instead builds the
/// same insert closure inline and drives generation through `main.rs`'s
/// `host_thought_writer(speakers)`, the same `(inputs, insert) -> ..`
/// closure `PendingTurn::think`'s two live-round callers already pass
/// around, so a regenerated thought is produced exactly the way a live
/// round's would be.
pub fn generate_thought_into(
    store: &dyn RunningThoughtStore,
    inputs: &ThoughtInputs,
    companion: &CompanionView,
    speakers: &PromptSpeakers,
    model: &dyn CharacterModel,
) -> Result<RunningThought, ThoughtError> {
    generate_thought(inputs, companion, speakers, model, &|thought| {
        let id = store.insert(thought)?;
        store.get(id)?.ok_or(rusqlite::Error::QueryReturnedNoRows)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Message;
    use crate::llm::FakeCharacterModel;
    use crate::participants::{ParticipantId, ParticipantRegistry};
    use crate::running_thoughts::store::RecordingStore;

    fn companion() -> CompanionView {
        CompanionView {
            name: "Bob".to_string(),
            persona: "a warm, curious persona".to_string(),
            example_dialogue: String::new(),
            first_message: String::new(),
            long_term_mem: 0,
            short_term_mem: 0,
            roleplay: false,
            dialogue_tuning: false,
            avatar_path: String::new(),
        }
    }

    fn speakers() -> PromptSpeakers {
        PromptSpeakers {
            registry: ParticipantRegistry::solo("Alice", "Bob", None),
            self_id: ParticipantId::CHAR,
        }
    }

    fn inputs() -> ThoughtInputs {
        ThoughtInputs {
            companion_id: 1,
            speaker_id: ParticipantId::CHAR,
            previous: vec![],
            round: vec![Message {
                id: 5,
                ai: false,
                speaker_id: "user".to_string(),
                content: "I got the job!".to_string(),
                created_at: String::new(),
            }],
            from_message_id: 5,
            through_message_id: 5,
        }
    }

    #[test]
    fn generate_thought_into_records_a_prompt_carrying_persona_notes_and_round() {
        let store = RecordingStore::new();
        let model = FakeCharacterModel::returning([Ok("I'm proud of them.".to_string())]);

        generate_thought_into(&store, &inputs(), &companion(), &speakers(), &model)
            .expect("generation should succeed");

        let prompts = model.prompts.lock().unwrap();
        assert_eq!(prompts.len(), 1);
        let (system, user) = &prompts[0];
        assert!(system.contains("a warm, curious persona"));
        assert!(user.contains("none yet"));
        assert!(user.contains("I got the job!"));
    }

    #[test]
    fn generate_thought_into_on_success_inserts_the_cleaned_text_with_the_inputs_range() {
        let store = RecordingStore::new();
        let model = FakeCharacterModel::returning([Ok("Bob: I'm proud of them.".to_string())]);

        let stored = generate_thought_into(&store, &inputs(), &companion(), &speakers(), &model)
            .expect("generation should succeed");

        assert_eq!(stored.text, "I'm proud of them.");
        assert_eq!(stored.companion_id, 1);
        assert_eq!(stored.speaker_id, ParticipantId::CHAR.to_string());
        assert_eq!(stored.from_message_id, 5);
        assert_eq!(stored.through_message_id, 5);
        assert!(!stored.edited);
        assert_eq!(store.thoughts.lock().unwrap().len(), 1);
    }

    #[test]
    fn generate_thought_into_on_a_model_error_inserts_nothing() {
        let store = RecordingStore::new();
        let model = FakeCharacterModel::returning([Err(std::io::Error::other("no model loaded"))]);

        let err = generate_thought_into(&store, &inputs(), &companion(), &speakers(), &model)
            .unwrap_err();

        assert!(matches!(err, ThoughtError::Generate(_)));
        assert!(store.thoughts.lock().unwrap().is_empty());
    }

    #[test]
    fn generate_thought_into_on_a_whitespace_only_output_inserts_nothing() {
        let store = RecordingStore::new();
        let model = FakeCharacterModel::returning([Ok("   \n\n  ".to_string())]);

        let err = generate_thought_into(&store, &inputs(), &companion(), &speakers(), &model)
            .unwrap_err();

        assert!(matches!(err, ThoughtError::Empty));
        assert!(store.thoughts.lock().unwrap().is_empty());
    }
}
