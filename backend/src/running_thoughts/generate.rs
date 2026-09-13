//! Runs one thought-generation call and persists the result (#216).
//! `Database`-free: persona, names, previous notes and the round are all
//! injected, so #220's joiner can call [`generate_thought`] with its own
//! card, registry snapshot and local store.

use crate::database::CompanionView;
use crate::llm::{CharacterModel, PromptSpeakers};
use crate::running_thoughts::prompt::{
    build_thought_prompt, clean_thought, count_sentences, truncate_to_sentences, ThoughtInputs,
    THOUGHT_MAX_TOKENS, THOUGHT_SENTENCE_LIMIT,
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
///
/// Generation stops at the `THOUGHT_SENTENCE_LIMIT`-th sentence boundary,
/// with `THOUGHT_MAX_TOKENS` as the backstop. A note that names the
/// companion in the third person (`"Jinx looks around the room…"`) is
/// regenerated once — a first-person note has no reason to contain its
/// author's name, and every such slip seen in the eval carried it; the
/// second attempt is kept whatever it says, so a name mentioned legitimately
/// costs one extra call and nothing more.
pub fn generate_thought(
    inputs: &ThoughtInputs,
    companion: &CompanionView,
    speakers: &PromptSpeakers,
    model: &dyn CharacterModel,
    insert: &dyn Fn(NewRunningThought) -> rusqlite::Result<RunningThought>,
) -> Result<RunningThought, ThoughtError> {
    let prompt = build_thought_prompt(
        inputs,
        &companion.persona,
        &companion.example_dialogue,
        speakers,
    );
    let self_name = speakers.self_name();
    let mut text = complete_one_note(model, &prompt.system, &prompt.user, self_name)?;
    if names_self(&text, self_name) {
        text = complete_one_note(model, &prompt.system, &prompt.user, self_name)?;
    }

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

/// One model call plus cleaning: stops at the sentence limit, trims the
/// token that crossed it, strips asides and a capped tail.
fn complete_one_note(
    model: &dyn CharacterModel,
    system: &str,
    user: &str,
    self_name: &str,
) -> Result<String, ThoughtError> {
    let completion = model
        .complete_in_character(system, user, THOUGHT_MAX_TOKENS, &mut |so_far| {
            count_sentences(so_far) < THOUGHT_SENTENCE_LIMIT
        })
        .map_err(ThoughtError::Generate)?;
    let within_limit = truncate_to_sentences(&completion.text, THOUGHT_SENTENCE_LIMIT);
    clean_thought(within_limit, self_name, completion.hit_token_cap).ok_or(ThoughtError::Empty)
}

/// True when a note mentions its own author by name, as a whole word — the
/// marker of a third-person slip in what should be first-person prose. A
/// possessive (`Bob's`) still counts; `Eve` inside `Even` does not, since a
/// false hit costs a whole second model call.
fn names_self(text: &str, self_name: &str) -> bool {
    if self_name.is_empty() {
        return false;
    }
    text.match_indices(self_name).any(|(start, _)| {
        let before = text[..start].chars().next_back();
        let after = text[start + self_name.len()..].chars().next();
        !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
    })
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
        assert!(user.contains("What just happened (Bob has not responded yet):"));
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
    fn generate_thought_into_regenerates_once_when_the_note_names_its_author() {
        let store = RecordingStore::new();
        let model = FakeCharacterModel::returning([
            Ok("Bob looks around the room, unsure.".to_string()),
            Ok("I'm not sure about this room.".to_string()),
        ]);

        let stored = generate_thought_into(&store, &inputs(), &companion(), &speakers(), &model)
            .expect("generation should succeed");

        assert_eq!(stored.text, "I'm not sure about this room.");
        assert_eq!(model.prompts.lock().unwrap().len(), 2);
        assert_eq!(store.thoughts.lock().unwrap().len(), 1);
    }

    #[test]
    fn generate_thought_into_keeps_the_second_attempt_even_if_it_still_names_the_author() {
        let store = RecordingStore::new();
        let model = FakeCharacterModel::returning([
            Ok("Bob is wary.".to_string()),
            Ok("Bob is still wary.".to_string()),
        ]);

        let stored = generate_thought_into(&store, &inputs(), &companion(), &speakers(), &model)
            .expect("generation should succeed");

        assert_eq!(stored.text, "Bob is still wary.");
        assert_eq!(model.prompts.lock().unwrap().len(), 2);
    }

    #[test]
    fn generate_thought_into_truncates_past_the_sentence_limit() {
        let store = RecordingStore::new();
        let model = FakeCharacterModel::returning([Ok(
            "One. Two. Three. Four that the fake did not stop at.".to_string(),
        )]);

        let stored = generate_thought_into(&store, &inputs(), &companion(), &speakers(), &model)
            .expect("generation should succeed");

        assert_eq!(stored.text, "One. Two. Three.");
    }

    #[test]
    fn generate_thought_into_strips_stage_directions() {
        let store = RecordingStore::new();
        let model = FakeCharacterModel::returning([Ok(
            "I'm proud of them. *I scribble in my notebook.* Really.".to_string(),
        )]);

        let stored = generate_thought_into(&store, &inputs(), &companion(), &speakers(), &model)
            .expect("generation should succeed");

        assert_eq!(stored.text, "I'm proud of them. Really.");
    }

    #[test]
    fn names_self_matches_only_whole_words() {
        assert!(!names_self("Even so, I'm fine.", "Eve"));
        assert!(names_self("Eve is wary.", "Eve"));
        assert!(names_self("Bob's hands shake.", "Bob"));
        assert!(!names_self("I told Bobby.", "Bob"));
        assert!(!names_self("anything", ""));
    }

    #[test]
    fn generate_thought_into_does_not_regenerate_on_a_word_that_merely_contains_the_name() {
        let store = RecordingStore::new();
        let model = FakeCharacterModel::returning([Ok("Bobbing along, I'm fine.".to_string())]);

        let stored = generate_thought_into(&store, &inputs(), &companion(), &speakers(), &model)
            .expect("generation should succeed");

        assert_eq!(stored.text, "Bobbing along, I'm fine.");
        assert_eq!(model.prompts.lock().unwrap().len(), 1);
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

    #[test]
    fn generate_thought_into_on_a_capped_completion_stores_the_text_trimmed_to_a_sentence() {
        let store = RecordingStore::new();
        let model = FakeCharacterModel::returning_capped([Ok(
            "I'm proud of them. I wonder what they'll do ne".to_string(),
        )]);

        let stored = generate_thought_into(&store, &inputs(), &companion(), &speakers(), &model)
            .expect("generation should succeed");

        assert_eq!(stored.text, "I'm proud of them.");
    }

    #[test]
    fn generate_thought_into_on_a_capped_whitespace_only_output_is_still_empty() {
        let store = RecordingStore::new();
        let model = FakeCharacterModel::returning_capped([Ok("   \n\n  ".to_string())]);

        let err = generate_thought_into(&store, &inputs(), &companion(), &speakers(), &model)
            .unwrap_err();

        assert!(matches!(err, ThoughtError::Empty));
        assert!(store.thoughts.lock().unwrap().is_empty());
    }

    #[test]
    fn generate_thought_into_on_a_capped_run_on_with_no_boundary_is_still_stored() {
        let store = RecordingStore::new();
        let model = FakeCharacterModel::returning_capped([Ok(
            "one long run-on that never ends and keeps going ab".to_string(),
        )]);

        let stored = generate_thought_into(&store, &inputs(), &companion(), &speakers(), &model)
            .expect("a capped run-on with no sentence boundary must still be stored");

        assert_eq!(
            stored.text,
            "one long run-on that never ends and keeps going ab"
        );
    }
}
