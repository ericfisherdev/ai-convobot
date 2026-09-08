use chrono::{DateTime, Local};
use serde::Serialize;
use std::io::Write;

use crate::attitude_formatter::AttitudeFormatter;
use crate::context_manager::ContextManager;
use crate::database::{
    contains_time_question, get_current_date, CompanionView, ConfigView, Database, Device, Message,
    PromptTemplate, UserView,
};
use crate::dialogue_tuning::DialogueTuning;
use crate::gpu_allocator::GpuAllocator;
use crate::inference_optimizer::INFERENCE_OPTIMIZER;
use crate::inference_performance::{ModelConfig, INFERENCE_TRACKER};
use crate::long_term_mem::LongTermMem;
use crate::model_cache::{ModelKey, ResidentCache};
use crate::model_metadata::{self, ModelFacts};
use crate::participants::{
    expand_placeholders, placeholder, Participant, ParticipantId, ParticipantRegistry,
};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::sync::{Mutex, OnceLock};

/// Maximum tokens submitted to llama.cpp in a single decode call.
const N_BATCH: u32 = 512;

/// Serialises generation. This does not protect two copies of the model from
/// coexisting in memory any more (the model is cached in `RESIDENT_MODEL`
/// and shared via `Arc`); it guarantees exactly one `LlamaContext` uses that
/// shared model at a time, and lets a config-driven key change free the old
/// model before loading the new one without racing a generation in flight.
static GENERATION_LOCK: Mutex<()> = Mutex::new(());

/// The currently loaded model, kept resident between turns. Reloaded only
/// when `ModelKey::from_config` changes (model path or GPU-related config).
static RESIDENT_MODEL: ResidentCache<ModelKey, LlamaModel> = ResidentCache::new();

/// llama.cpp keeps process-global state, so its backend must be initialised
/// exactly once. Later calls reuse the handle stored here.
static LLAMA_BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static LLAMA_BACKEND_INIT_LOCK: Mutex<()> = Mutex::new(());

/// Returns the process-wide llama.cpp backend, initialising it on first use.
///
/// # Errors
/// Returns `std::io::ErrorKind::Other` if llama.cpp fails to initialise.
fn llama_backend() -> Result<&'static LlamaBackend, std::io::Error> {
    if let Some(backend) = LLAMA_BACKEND.get() {
        return Ok(backend);
    }
    // Serialise initialisation so two concurrent requests cannot both call
    // llama_backend_init and have one fail with BackendAlreadyInitialized.
    let _guard = LLAMA_BACKEND_INIT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(backend) = LLAMA_BACKEND.get() {
        return Ok(backend);
    }
    let backend = LlamaBackend::init().map_err(|e| {
        std::io::Error::other(format!("Failed to initialize llama.cpp backend: {}", e))
    })?;
    Ok(LLAMA_BACKEND.get_or_init(|| backend))
}

/// Merges consecutive turns from the same speaker into one, because several
/// chat templates reject a history that does not strictly alternate. The
/// bool is "spoken by `self_id`" (see `AssembledPrompt::chat_history`), so
/// merging two different non-self speakers (e.g. two other bots, or a bot
/// and the user) into one produces a single `user`-role message carrying
/// both of their lines.
fn merge_consecutive_turns(history: &[(bool, String)]) -> Vec<(bool, String)> {
    let mut merged: Vec<(bool, String)> = Vec::with_capacity(history.len());
    for (is_self, content) in history {
        match merged.last_mut() {
            Some((last_is_self, last_content)) if last_is_self == is_self => {
                last_content.push('\n');
                last_content.push_str(content);
            }
            _ => merged.push((*is_self, content.clone())),
        }
    }
    merged
}

/// Renders the prompt using the chat template stored inside the GGUF file.
///
/// llama.cpp chat templates only know `system`/`user`/`assistant` roles, so
/// this can express at most two speakers: the turn's own participant is
/// `assistant`, everyone else is `user`. With more than one other
/// participant that collapses several speakers onto the `user` role; they
/// are told apart by the `Name: ` prefix `render_history` already put in
/// their content.
///
/// # Errors
/// Returns a message describing why the model's template could not be used,
/// so the caller can fall back to the plain-text transcript format.
fn apply_gguf_chat_template(
    model: &LlamaModel,
    system_content: &str,
    history: &[(bool, String)],
) -> Result<String, String> {
    let template = model
        .chat_template(None)
        .map_err(|e| format!("model does not carry a usable chat template: {}", e))?;

    let mut messages: Vec<LlamaChatMessage> = Vec::with_capacity(history.len() + 1);
    if !system_content.trim().is_empty() {
        messages.push(
            LlamaChatMessage::new("system".to_string(), system_content.to_string())
                .map_err(|e| format!("invalid system message: {}", e))?,
        );
    }

    let merged = merge_consecutive_turns(history);

    for (is_self, content) in &merged {
        let role = if *is_self { "assistant" } else { "user" };
        messages.push(
            LlamaChatMessage::new(role.to_string(), content.clone())
                .map_err(|e| format!("invalid chat message: {}", e))?,
        );
    }

    if messages.is_empty() {
        return Err("no messages to render".to_string());
    }

    model
        .apply_chat_template(&template, &messages, true)
        .map_err(|e| format!("failed to apply chat template: {}", e))
}

/// Joins display names into prose: `A` for one, `A and B` for two, `A, B and
/// C` for three or more.
fn join_names(names: &[&str]) -> String {
    match names {
        [] => String::new(),
        [only] => only.to_string(),
        [first, second] => format!("{} and {}", first, second),
        _ => {
            let (last, rest) = names.split_last().expect("names is non-empty here");
            format!("{} and {}", rest.join(", "), last)
        }
    }
}

/// Builds the system-portion components of the prompt for the given
/// template, inserting `attitude_context` (when non-empty) as its own
/// component before the template's instruct terminator, so it always ends up
/// inside the system block rather than after the conversation history.
///
/// Names come from `speakers`; personas, example dialogue and dialogue
/// tuning still come from the caller's own `user`/`companion` rows (a
/// joiner's own persona, not the host's). When a bot other than `speakers`'
/// own turn is present, an `Also present: ...` component names the rest, so
/// no other participant is left for the model to invent.
fn build_base_components(
    template: &PromptTemplate,
    user: &UserView,
    companion: &CompanionView,
    rp: &str,
    tuned_dialogue: &str,
    attitude_context: &str,
    speakers: &PromptSpeakers,
) -> Vec<String> {
    let participants = &speakers.registry;
    let user_name = speakers.user_name();
    let self_name = speakers.self_name();
    let all_names: Vec<&str> = participants
        .iter()
        .map(|p| p.display_name.as_str())
        .collect();
    let other_bot_names: Vec<&str> = speakers
        .others()
        .filter(|p| p.id != ParticipantId::USER)
        .map(|p| p.display_name.as_str())
        .collect();
    let also_present = if other_bot_names.is_empty() {
        String::new()
    } else {
        format!("Also present: {}.\n", join_names(&other_bot_names))
    };

    if *template == PromptTemplate::Default || *template == PromptTemplate::Auto {
        let mut components = vec![
            format!(
                "Text transcript of a conversation between {}. {}\n",
                join_names(&all_names),
                rp
            ),
            format!(
                "{}'s Persona: {}\n",
                user_name,
                expand_placeholders(&user.persona, participants)
            ),
        ];
        if !also_present.is_empty() {
            components.push(also_present);
        }
        if !attitude_context.is_empty() {
            components.push(attitude_context.to_string());
        }
        components.push(format!(
            "{}'s Persona: {}\n<START>\n",
            self_name,
            expand_placeholders(&companion.persona, participants)
        ));
        components.push(format!(
            "{}\n<START>\n",
            expand_placeholders(&companion.example_dialogue, participants)
        ));
        components.push(format!("{}\n<START>\n", tuned_dialogue));
        components
    } else if *template == PromptTemplate::Llama2 {
        let mut components = vec![format!(
            "<<SYS>>\nYou are {}, {}\n",
            self_name,
            expand_placeholders(&companion.persona, participants)
        )];
        if !also_present.is_empty() {
            components.push(also_present);
        }
        if !attitude_context.is_empty() {
            components.push(attitude_context.to_string());
        }
        components.push(format!(
            "you are talking with {}, {} is {}\n{}\n[INST]\n",
            user_name,
            user_name,
            expand_placeholders(&user.persona, participants),
            rp
        ));
        components.push(format!(
            "{}\n",
            expand_placeholders(&companion.example_dialogue, participants)
        ));
        components.push(format!("{}\n[/INST]\n", tuned_dialogue));
        components
    } else {
        let mut components = vec![
            format!(
                "<s>[INST]Text transcript of a conversation between {}. {}\n",
                join_names(&all_names),
                rp
            ),
            format!(
                "{}'s Persona: {}\n",
                user_name,
                expand_placeholders(&user.persona, participants)
            ),
        ];
        if !also_present.is_empty() {
            components.push(also_present);
        }
        if !attitude_context.is_empty() {
            components.push(attitude_context.to_string());
        }
        components.push(format!(
            "{}'s Persona: {}[/INST]\n<s>[INST]\n",
            self_name,
            expand_placeholders(&companion.persona, participants)
        ));
        components.push(format!(
            "{}[/INST]\n<s>[INST]\n",
            expand_placeholders(&companion.example_dialogue, participants)
        ));
        components.push(format!("{}[/INST]\n", tuned_dialogue));
        components
    }
}

/// Generates a reply and returns it once generation finishes.
///
/// Does not persist the reply: the caller (#131's `PendingTurn::reply`)
/// inserts it through `TurnStore::insert_reply`, so every reply — local or
/// remote — is persisted through the one store call.
///
/// # Errors
/// Propagates model load, tokenization and decode failures as
/// `std::io::ErrorKind::Other`.
pub fn prompt(
    prompt: &str,
    companion_id: i32,
    transcript: &dyn TranscriptSource,
    speakers: &PromptSpeakers,
) -> Result<String, std::io::Error> {
    generate(prompt, companion_id, &mut |_token| {}, transcript, speakers)
}

/// Generates a reply, invoking `on_token` with each token as it is produced.
///
/// The callback runs on the generating thread, so it must not block. Does
/// not persist the reply; see [`prompt`].
///
/// # Errors
/// Propagates model load, tokenization and decode failures as
/// `std::io::ErrorKind::Other`.
pub fn prompt_streaming(
    prompt: &str,
    companion_id: i32,
    on_token: &mut dyn FnMut(&str),
    transcript: &dyn TranscriptSource,
    speakers: &PromptSpeakers,
) -> Result<String, std::io::Error> {
    generate(prompt, companion_id, on_token, transcript, speakers)
}

/// Whether each turn should print the full attitude block it injected.
///
/// Enabled with `AI_COMPANION_ATTITUDE_DEBUG=1`. Off by default, so normal
/// console output is unchanged.
fn attitude_debug_enabled() -> bool {
    std::env::var("AI_COMPANION_ATTITUDE_DEBUG")
        .map(|value| value == "1")
        .unwrap_or(false)
}

/// Seed for multinomial sampling.
///
/// `AI_COMPANION_SAMPLER_SEED` pins it so two runs over the same prompt are
/// comparable (the rest of the sampler chain is deterministic given the same
/// logits); unset, each generation is seeded randomly as before.
fn sampler_seed() -> u32 {
    match std::env::var("AI_COMPANION_SAMPLER_SEED") {
        Ok(value) => match value.trim().parse::<u32>() {
            Ok(seed) => seed,
            Err(e) => {
                // Falling back silently would defeat the whole point of the
                // variable, so say so rather than quietly randomising.
                eprintln!(
                    "AI_COMPANION_SAMPLER_SEED is not a u32 ({:?}: {}); using a random seed",
                    value, e
                );
                rand::random::<u32>()
            }
        },
        Err(_) => rand::random::<u32>(),
    }
}

/// The conversation record a prompt is assembled from: the newest messages,
/// read before generation.
///
/// Read-only: `generate` used to append the reply it just produced through
/// this same seam, but #131 moved that persistence out to the caller
/// (`PendingTurn::reply`, via `TurnStore::insert_reply`), so every reply —
/// local or remote — goes through one store call instead of two different
/// ones depending on who generated it. Solo and host turns read through
/// [`SqliteTranscript`]; a joiner (#130) generates from a transcript it
/// received over the wire, hence [`InMemoryTranscript`].
pub trait TranscriptSource {
    /// The newest `limit` messages, oldest first.
    ///
    /// # Errors
    /// Returns `std::io::ErrorKind::Other` if the underlying read fails.
    fn recent_messages(&self, limit: usize) -> std::io::Result<Vec<Message>>;
}

/// The production [`TranscriptSource`], backed by `companion_database.db`.
pub struct SqliteTranscript;

impl TranscriptSource for SqliteTranscript {
    fn recent_messages(&self, limit: usize) -> std::io::Result<Vec<Message>> {
        Database::get_x_messages(limit, 0).map_err(|e| {
            eprintln!("Error while getting short term memory entries: {}", e);
            std::io::Error::other("Error while getting short term memory entries")
        })
    }
}

/// A [`TranscriptSource`] over a fixed, in-memory list of messages: used by
/// tests now, and by #130's joiner.
#[allow(dead_code)] // wired up by #130's joiner; exercised directly by this module's tests today
pub struct InMemoryTranscript(pub Vec<Message>);

impl TranscriptSource for InMemoryTranscript {
    fn recent_messages(&self, limit: usize) -> std::io::Result<Vec<Message>> {
        let start = self.0.len().saturating_sub(limit);
        Ok(self.0[start..].to_vec())
    }
}

/// An owned, `Clone + Send` snapshot of who is in the chat and which of them
/// the current turn is generating for. Owned so it can move into
/// `web::block` closures and the `stream-generation` thread without holding
/// the shared registry's lock across a generation.
#[derive(Clone)]
pub struct PromptSpeakers {
    pub registry: ParticipantRegistry,
    pub self_id: ParticipantId,
}

impl PromptSpeakers {
    /// The display name of the participant this turn is generating for.
    /// Falls back to the raw id if `self_id` is somehow not in the registry
    /// (never expected: every registry always carries `char`, and a joiner
    /// always knows its own id).
    pub fn self_name(&self) -> &str {
        self.registry
            .display_name(&self.self_id)
            .unwrap_or(self.self_id.as_str())
    }

    /// The display name of the human participant. Falls back to the raw id
    /// for the same reason as `self_name`.
    pub fn user_name(&self) -> &str {
        self.registry
            .display_name(&ParticipantId::USER)
            .unwrap_or(ParticipantId::USER.as_str())
    }

    /// Every participant except the one this turn is generating for, in join
    /// order.
    pub fn others(&self) -> impl Iterator<Item = &Participant> {
        let self_id = &self.self_id;
        self.registry.iter().filter(move |p| &p.id != self_id)
    }
}

/// Everything a prompt is made of, before the model is involved.
///
/// Returned by `assemble_prompt` so the exact text a turn would send — the
/// attitude block in particular — can be inspected without running inference.
#[derive(Serialize)]
pub struct AssembledPrompt {
    /// The system portion plus, for every template but `Auto`, the conversation
    /// history spliced into it.
    pub system_prompt: String,
    /// Role-tagged history, used only by the `Auto` template.
    pub chat_history: Vec<(bool, String)>,
    /// The attitude block that was folded into `system_prompt`.
    pub attitude_context: String,
    /// The history after `ContextManager` trimming.
    pub managed_messages: Vec<Message>,
}

/// The result of rendering a message history for one turn.
struct RenderedHistory {
    /// Text ready to append to the system prompt, for every template but
    /// `Auto` (which uses `chat_history` instead and ignores this).
    spliced: String,
    /// Role-tagged history, used only by the `Auto` template. The bool is
    /// "spoken by `speakers.self_id`", documented on
    /// `AssembledPrompt::chat_history`.
    chat_history: Vec<(bool, String)>,
}

/// Whether `Auto`-template `chat_history` entries need an inlined `"Name: "`
/// prefix on non-self turns: true when there is more than one non-self
/// participant, since the chat template's own role framing can no longer
/// say who spoke. The single source of truth for that decision — both
/// `render_history` (which decides whether to inline the prefix) and
/// `generate`'s `Auto` fallback (which decides whether content already
/// carries one) call this instead of each re-deriving it, so the two can
/// never fall out of agreement. Not carried on `AssembledPrompt` itself:
/// that struct is serialised verbatim by `GET /api/debug/prompt`, whose
/// solo-mode JSON shape must stay unchanged.
fn auto_prefixes_names(speakers: &PromptSpeakers) -> bool {
    speakers.others().count() > 1
}

/// Renders `managed` (already trimmed by `ContextManager`) into both the
/// plain-text splice used by every template but `Auto`, and the role-tagged
/// `chat_history` `Auto` renders through the model's own chat template.
///
/// Speaker names come from `speakers.registry`; a `speaker_id` no longer in
/// the registry (a bot that has since left) falls back to the raw id rather
/// than being attributed to `self`.
///
/// Under `Auto`, a non-self turn is prefixed with `"{name}: "` only when
/// there is more than one non-self participant — i.e. some bot other than
/// `self` besides the user — since with exactly two participants the chat
/// template's own role framing already identifies the other speaker and
/// solo output must stay byte-identical.
fn render_history(
    managed: &[Message],
    speakers: &PromptSpeakers,
    template: &PromptTemplate,
) -> RenderedHistory {
    let len = managed.len();
    let auto_show_names = auto_prefixes_names(speakers);
    let mut spliced = String::new();
    let mut chat_history: Vec<(bool, String)> = Vec::with_capacity(len);

    for (message_counter, message) in (1..).zip(managed.iter()) {
        let participant_id = ParticipantId::parse(&message.speaker_id).ok();
        let is_self = participant_id.as_ref() == Some(&speakers.self_id);
        let display_name = participant_id
            .as_ref()
            .and_then(|id| speakers.registry.display_name(id))
            .unwrap_or(message.speaker_id.as_str());
        let text = &message.content;
        let mut formatted_message = format!("{}: {}\n", display_name, text);
        let inject_time = message_counter == len && contains_time_question(&formatted_message);
        if inject_time {
            formatted_message = format!(
                "\n* it's currently {} *\n{}",
                get_current_date(),
                formatted_message
            );
        }
        match template {
            PromptTemplate::Auto => {
                // The chat template supplies the speaker framing for the
                // only other participant in solo chat, so the message
                // carries its own text rather than a "Name: " prefix; with
                // more than one other participant, the role alone can no
                // longer say who spoke, so the name is spelled out.
                let mut content = if auto_show_names && !is_self {
                    format!("{}: {}", display_name, text)
                } else {
                    text.clone()
                };
                if inject_time {
                    content = format!("* it's currently {} *\n{}", get_current_date(), content);
                }
                chat_history.push((is_self, content));
            }
            PromptTemplate::Llama2 | PromptTemplate::Mistral => {
                if !is_self {
                    spliced += &format!("[INST]{}", formatted_message);
                } else {
                    spliced += &format!("{}[/INST]\n", formatted_message);
                }
            }
            _ => {
                spliced += &formatted_message;
            }
        }
    }

    RenderedHistory {
        spliced,
        chat_history,
    }
}

/// Builds the prompt for one turn without loading a model.
///
/// This is the whole of `generate`'s string assembly: `generate` calls it and
/// consumes the result, and `GET /api/debug/prompt` calls it to show what would
/// be sent. `user_message` is what the turn's long-term memory recall is keyed
/// on, so an inspection with an empty message simply recalls nothing.
///
/// The caller passes the `ConfigView` it is using for the rest of the turn:
/// `PUT /api/config` does not take `GENERATION_LOCK`, so a second read here
/// could see a different template or token budget than the one the reply is
/// rendered with.
///
/// # Errors
/// Propagates user and companion load failures as `std::io::ErrorKind::Other`.
pub fn assemble_prompt(
    user_message: &str,
    companion_id: i32,
    long_term_memory: &LongTermMem,
    config: &ConfigView,
    transcript: &dyn TranscriptSource,
    speakers: &PromptSpeakers,
) -> Result<AssembledPrompt, std::io::Error> {
    let user: UserView = match Database::get_user_data() {
        Ok(user) => user,
        Err(e) => {
            eprintln!("Error while getting user data: {}", e);
            return Err(std::io::Error::other("Error while getting user data"));
        }
    };
    let companion: CompanionView = match Database::get_companion_data() {
        Ok(companion) => companion,
        Err(e) => {
            eprintln!("Error while getting companion data: {}", e);
            return Err(std::io::Error::other("Error while getting companion data"));
        }
    };
    // Every name below comes from `speakers`, not from the `user`/`companion`
    // rows above: a joiner's own `companion` row is its own card, but the
    // names in its prompt must reflect the host's shared registry so it
    // names the host's user (and any other bots) correctly.
    let participants = &speakers.registry;
    let mut base_prompt: String;
    let mut rp: &str = "";
    let mut tuned_dialogue: String = String::from("");
    if companion.roleplay {
        rp = "gestures and other non-verbal actions are written between asterisks (for example, *waves hello* or *moves closer*)";
    }
    if companion.dialogue_tuning {
        if let Ok(dialogue) = DialogueTuning::get_random_dialogue() {
            tuned_dialogue = format!(
                "{}: {}\n{}: {}",
                speakers.user_name(),
                dialogue.user_msg,
                speakers.self_name(),
                dialogue.ai_msg
            );
        };
    }
    // Initialize context manager for intelligent memory management. Built
    // before the attitude block because the memory block below is trimmed
    // against its attitude token budget.
    let context_manager = ContextManager::new(config.clone());

    // Load and integrate attitude context. This must happen before
    // base_components is built, so the attitude block lands inside the
    // system portion of the prompt rather than after the conversation
    // history (and the instruct terminator it ends with).
    let attitude_formatter = AttitudeFormatter::new();
    let attitudes = match Database::get_all_companion_attitudes(companion_id) {
        Ok(attitudes) => attitudes,
        Err(e) => {
            eprintln!("Warning: Could not load attitudes: {}", e);
            Vec::new()
        }
    };

    let third_parties = match Database::get_all_third_party_individuals() {
        Ok(parties) => parties,
        Err(e) => {
            eprintln!("Warning: Could not load third parties: {}", e);
            Vec::new()
        }
    };

    // Add attitude context to prompt if attitudes exist
    let attitude_context = if !attitudes.is_empty() {
        let context = attitude_formatter.format_attitude_context(
            &attitudes,
            &third_parties,
            speakers.user_name(),
        );
        if !context.is_empty() {
            format!("\n{}\n", context)
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    // Append the moments that shaped those feelings, so the companion can say
    // why it feels the way it does and not only how strongly. Appending to
    // `attitude_context` keeps the block inside the same system portion and
    // counted by the same `attitude_tokens` figure below.
    let mut attitude_context = attitude_context;
    // Filtered to user rows in the query: limiting across every target type
    // first would let third-party rows take all five slots and render an empty
    // block even with qualifying user memories just below the cutoff.
    match Database::get_priority_attitude_memories_for_target(companion_id, "user", 5) {
        Ok(mut memories) => {
            // Trim from the tail (lowest priority first) until the whole
            // attitude block fits its budget, so memories can never starve
            // message history.
            while !memories.is_empty() {
                let block = attitude_formatter.format_attitude_memories(&memories);
                if block.is_empty() {
                    break;
                }
                let candidate = format!("{}{}\n", attitude_context, block);
                if ContextManager::estimate_tokens(&candidate)
                    <= context_manager.attitude_token_budget
                {
                    attitude_context = candidate;
                    break;
                }
                memories.pop();
            }
        }
        Err(e) => eprintln!("Warning: Could not load attitude memories: {}", e),
    }

    if !attitude_context.is_empty() {
        println!(
            "✓ Attitude context integrated: {} characters",
            attitude_context.len()
        );
    }

    // Build base prompt components.
    // Auto renders through the model's own chat template, so its system content
    // must be plain prose; the Mistral branch below would embed [INST] markers.
    let base_components = build_base_components(
        &config.prompt_template,
        &user,
        &companion,
        rp,
        &tuned_dialogue,
        &attitude_context,
        speakers,
    );

    base_prompt = base_components.join("");

    if companion.long_term_mem > 0 {
        let long_term_memory_entries: Vec<String> =
            match long_term_memory.get_matches(user_message, companion.long_term_mem) {
                Ok(entries) => entries,
                Err(e) => {
                    eprintln!("Error while getting long term memory entries: {}", e);
                    return Err(std::io::Error::other(
                        "Error while getting long term memory entries",
                    ));
                }
            };
        for entry in long_term_memory_entries {
            let entry = expand_placeholders(&entry, participants);
            if config.prompt_template == PromptTemplate::Llama2 {
                base_prompt += &format!("[INST]{}[/INST]\n", entry);
            } else if config.prompt_template == PromptTemplate::Mistral {
                base_prompt += &format!("<s>[INST]{}[/INST]\n", entry);
            } else {
                base_prompt += &entry;
            }
        }
    }
    // `TranscriptSource` impls log the cause of a read failure themselves.
    let short_term_memory_entries: Vec<Message> =
        transcript.recent_messages(if companion.short_term_mem > 0 {
            companion.short_term_mem
        } else {
            50
        })?;

    // Apply context management to optimize memory usage
    let managed_messages = context_manager.manage_message_context(short_term_memory_entries);
    let rendered = render_history(&managed_messages, speakers, &config.prompt_template);
    base_prompt += &rendered.spliced;
    let chat_history = rendered.chat_history;

    if attitude_debug_enabled() && !attitude_context.is_empty() {
        // `managed_messages` is the trimmed history, so its length is the turn
        // index a console transcript can be lined up against.
        println!(
            "🧭 Attitude block (turn {}):\n{}",
            managed_messages.len(),
            attitude_context
        );
    }

    Ok(AssembledPrompt {
        system_prompt: base_prompt,
        chat_history,
        attitude_context,
        managed_messages,
    })
}

/// Resolves how many layers llama.cpp should offload to the GPU.
///
/// llama.cpp expresses GPU offloading as a single layer count, so this runs
/// before building the model parameters. Only called from `load_model`, i.e.
/// only when a (re)load actually happens: running GPU detection on every
/// turn would see less free VRAM once the model is resident and thrash the
/// layer count down, see `ModelKey`. `facts` is the real GGUF metadata read
/// by `load_model`, not the config inputs, so it must never be folded into
/// `ModelKey`.
///
/// `layer_count_known` is `false` when `facts.layer_count` is `load_model`'s
/// 32-layer fallback guess rather than a real header read. The static-layer
/// clamp below only fires when the count is known: clamping a user's
/// configured layer count down to a guess could silently under-offload a
/// model that actually has far more layers than the guess.
fn resolve_gpu_layers(config: &ConfigView, facts: &ModelFacts, layer_count_known: bool) -> u32 {
    if config.device == Device::GPU || config.device == Device::Metal {
        if config.dynamic_gpu_allocation {
            let allocator = GpuAllocator::new()
                .with_safety_margin(config.gpu_safety_margin)
                .with_min_free_vram(config.min_free_vram_mb);

            match allocator.detect_gpu_memory(&config.device) {
                Ok(gpu_info) => {
                    println!("🔍 GPU Detection: {}", gpu_info);

                    let vram_limit = GpuAllocator::vram_limit_from_config(config.vram_limit_gb);
                    let allocation = allocator.plan_for_model(&gpu_info, facts, vram_limit);

                    println!("🎯 Dynamic Allocation: {}", allocation);
                    allocation.gpu_layers as u32
                }
                Err(e) => {
                    eprintln!("⚠️ GPU detection failed, using configured layers: {}", e);
                    config.gpu_layers as u32
                }
            }
        } else if layer_count_known && config.gpu_layers as u32 > facts.layer_count {
            println!(
                "📌 Static Allocation: clamping configured {} GPU layers to the model's {} layers",
                config.gpu_layers, facts.layer_count
            );
            facts.layer_count
        } else {
            println!("📌 Static Allocation: {} GPU layers", config.gpu_layers);
            config.gpu_layers as u32
        }
    } else {
        println!("💻 CPU-only inference mode");
        0
    }
}

/// Loads the GGUF file named by `config.llm_model_path`. Only runs when
/// `RESIDENT_MODEL` needs a (re)load, i.e. on the first turn and whenever
/// `ModelKey::from_config` changes.
///
/// Reads the model's real architecture, layer count, and size from its GGUF
/// header before resolving GPU layers. That read is a pure function of
/// `config.llm_model_path`, which is already part of `ModelKey`, so it
/// cannot cause the resident model to reload on its own (see `ModelKey`'s
/// no-thrash doc comment in `model_cache.rs`).
fn load_model(backend: &LlamaBackend, config: &ConfigView) -> Result<LlamaModel, std::io::Error> {
    let model_path = std::path::Path::new(&config.llm_model_path);
    let (facts, layer_count_known) = match model_metadata::read_model_facts(model_path) {
        Ok(facts) => {
            println!(
                "📐 Model: {}, {} layers, {} MB",
                facts.architecture,
                facts.layer_count,
                facts.size_mb()
            );
            (facts, true)
        }
        Err(e) => {
            eprintln!(
                "⚠️ Failed to read model metadata ({}), falling back to a 32-layer estimate",
                e
            );
            let file_size_bytes = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
            let facts = ModelFacts {
                path: config.llm_model_path.clone(),
                architecture: "unknown".to_string(),
                layer_count: 32,
                file_size_bytes,
            };
            (facts, false)
        }
    };

    let gpu_layers = resolve_gpu_layers(config, &facts, layer_count_known);
    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(gpu_layers)
        .with_use_mmap(true); // Memory-mapped model loading reduces RAM usage

    print!("📚 Loading model... ");
    // Console output is best-effort: the stream and the persisted reply are
    // the product, so a broken stdout pipe (e.g. `... | head`) must not panic
    // this thread.
    let _ = std::io::stdout().flush();
    let load_start = std::time::Instant::now();
    let model = LlamaModel::load_from_file(backend, model_path, &model_params)
        .map_err(|e| std::io::Error::other(format!("Failed to load llm model: {}", e)))?;
    println!(
        "✓ Model loaded in {:.2}s ({} GPU layers)",
        load_start.elapsed().as_secs_f64(),
        gpu_layers
    );
    // Post-load truth check: the header read above should always agree
    // with what llama.cpp itself resolves. This should never fire. Skipped
    // when the header read failed, since `facts.layer_count` is then a
    // guess rather than a fact, and disagreeing with it is expected.
    if layer_count_known && model.n_layer() != facts.layer_count {
        eprintln!(
            "⚠️ Layer count mismatch: GGUF header reported {} but loaded model has {}",
            facts.layer_count,
            model.n_layer()
        );
    }
    Ok(model)
}

/// Frees the resident model, if any, returning whether one was resident and
/// the path it was loaded from. A generation in flight keeps its own `Arc`
/// clone, so the model stays alive until that turn finishes; this only stops
/// it from being handed out to new turns. The next turn reloads it.
pub fn unload_model() -> (bool, Option<String>) {
    match RESIDENT_MODEL.evict() {
        Some(key) => (true, Some(key.model_path)),
        None => (false, None),
    }
}

/// Fixed template tokens that end a reply no matter which participants are
/// in the chat.
const TOKEN_MARKERS: [&str; 4] = ["[/INST]", "<</SYS>>", "[s]", "<|user|>"];

/// Precomputes the markers `generate`'s halting and cleanup logic watch for,
/// generalising the old two-participant checks (`\n{user}:`, `{companion}:`,
/// `{user}:`, plus the four template tokens above) to however many
/// participants are in the chat.
struct ReplyTrimmer {
    /// `"\n{name}:"` for every participant except `self`: the model starting
    /// a new line as someone else means it has begun impersonating them.
    line_markers: Vec<String>,
    /// `"{name}:"` for every participant, `self` included: catches a speaker
    /// named mid-line rather than at the start of one.
    speaker_markers: Vec<String>,
    /// `"\n{name}: "` (trailing space) for every participant, used only to
    /// find where a spurious extra turn begins so it can be cut off.
    cut_markers: Vec<String>,
}

impl ReplyTrimmer {
    fn new(speakers: &PromptSpeakers) -> Self {
        let line_markers = speakers
            .others()
            .map(|p| format!("\n{}:", p.display_name))
            .collect();
        let speaker_markers = speakers
            .registry
            .iter()
            .map(|p| format!("{}:", p.display_name))
            .collect();
        let cut_markers = speakers
            .registry
            .iter()
            .map(|p| format!("\n{}: ", p.display_name))
            .collect();
        ReplyTrimmer {
            line_markers,
            speaker_markers,
            cut_markers,
        }
    }

    /// Whether `generated` shows the model starting to speak as, or name,
    /// another participant, or emitting one of the fixed template tokens.
    fn should_stop(&self, generated: &str) -> bool {
        self.speaker_markers
            .iter()
            .any(|marker| generated.contains(marker.as_str()))
            || self
                .line_markers
                .iter()
                .any(|marker| generated.contains(marker.as_str()))
            || TOKEN_MARKERS
                .iter()
                .any(|marker| generated.contains(marker))
    }

    /// Strips every marker above and template token out of `generated`, cuts
    /// off anything from the earliest spurious extra turn onward, and trims
    /// the leading whitespace generation tends to start with.
    fn clean(&self, generated: &str) -> String {
        let mut cleaned = generated.to_string();
        // Cut first: a `line_markers` removal below deletes the newline a
        // `cut_markers` entry needs to match, which would otherwise let a
        // spurious extra turn survive with only its name stripped whenever
        // it was followed by more text (see the #127 review discussion).
        if let Some(cut_at) = self
            .cut_markers
            .iter()
            .filter_map(|marker| cleaned.find(marker.as_str()))
            .min()
        {
            cleaned.truncate(cut_at);
        }
        for marker in &self.line_markers {
            cleaned = cleaned.replace(marker.as_str(), "");
        }
        cleaned = cleaned
            .replace("[INST]", "")
            .replace("[/INST]", "")
            .replace("<</SYS>>", "")
            .replace("<s>", "")
            .replace("</s>", "")
            .replace("<|user|>", "");
        cleaned.trim_start().to_string()
    }
}

fn generate(
    prompt: &str,
    companion_id: i32,
    on_token: &mut dyn FnMut(&str),
    transcript: &dyn TranscriptSource,
    speakers: &PromptSpeakers,
) -> Result<String, std::io::Error> {
    let _generation_guard = GENERATION_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let start_time = std::time::Instant::now();
    let long_term_memory = match LongTermMem::shared() {
        Ok(ltm) => ltm,
        Err(e) => {
            eprintln!("Error while connecting to tantivy: {}", e);
            return Err(std::io::Error::other("Error while connecting to tantivy"));
        }
    };
    let local: DateTime<Local> = Local::now();
    let formatted_date = local.format("* at %A %d.%m.%Y %H:%M *\n").to_string();
    let config: ConfigView = match Database::get_config() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Error while getting config: {}", e);
            return Err(std::io::Error::other("Error while getting config"));
        }
    };

    let backend = llama_backend()?;

    let (model, was_resident) = RESIDENT_MODEL
        .get_or_load(ModelKey::from_config(&config), |_| {
            load_model(backend, &config)
        })?;
    if was_resident {
        println!("♻️ Reusing resident model");
    }
    let model_ready = start_time.elapsed();

    // Calculate CPU cores for optimizations
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4); // Fallback to 4 cores if detection fails

    println!("🚀 Generating AI response with optimized session...");
    let assembled = assemble_prompt(
        prompt,
        companion_id,
        long_term_memory,
        &config,
        transcript,
        speakers,
    )?;
    let AssembledPrompt {
        system_prompt: base_prompt,
        chat_history,
        attitude_context,
        managed_messages,
    } = assembled;
    // Built from the config passed into `assemble_prompt`, so the budgets below
    // match the ones the assembly was trimmed against.
    let context_manager = ContextManager::new(config.clone());

    // Calculate token usage for memory management. base_prompt already
    // contains the attitude text (it was folded into base_components above),
    // so it is subtracted back out here to avoid double counting it.
    let attitude_tokens = ContextManager::estimate_tokens(&attitude_context);
    let system_tokens =
        ContextManager::estimate_tokens(&base_prompt).saturating_sub(attitude_tokens);
    let message_tokens = managed_messages
        .iter()
        .map(|msg| ContextManager::estimate_tokens(&msg.content))
        .sum::<usize>();

    // Get response token limit and print memory stats
    let response_token_limit =
        context_manager.get_response_token_limit(system_tokens + attitude_tokens + message_tokens);
    let memory_stats =
        context_manager.get_memory_stats(system_tokens, attitude_tokens, message_tokens);
    memory_stats.print_stats();

    // Initialize performance tracking
    let session_id = format!(
        "llm_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );

    let model_config = ModelConfig {
        model_path: config.llm_model_path.clone(),
        gpu_layers: config.gpu_layers as i32,
        device_type: config.device.to_string(),
    };

    let input_tokens = (system_tokens + attitude_tokens + message_tokens) as u32;

    // Start performance tracking
    if let Ok(mut tracker) = INFERENCE_TRACKER.lock() {
        tracker.start_session(session_id.clone(), model_config.clone(), input_tokens);
    }

    // The old `llm` crate hid sampling behind InferenceParameters::default().
    // llama.cpp requires an explicit sampler chain, so these values reproduce
    // a conventional chat preset.
    let seed = sampler_seed();
    println!("🎲 Sampler seed: {}", seed);
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(model.n_vocab(), 64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.8),
        LlamaSampler::dist(seed),
    ]);

    // Size the KV cache from the budget the ContextManager already computed.
    let context_size = context_manager.token_budget.total.max(512) as u32;
    let context_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(context_size))
        .with_n_batch(N_BATCH)
        .with_n_threads(cpu_cores as i32)
        .with_n_threads_batch(cpu_cores as i32);

    let mut llama_context = match model.new_context(backend, context_params) {
        Ok(context) => context,
        Err(e) => {
            return Err(std::io::Error::other(format!(
                "Failed to create llama context: {}",
                e
            )))
        }
    };

    let full_prompt = if config.prompt_template == PromptTemplate::Auto {
        match apply_gguf_chat_template(&model, &base_prompt, &chat_history) {
            Ok(rendered) => {
                println!("🧩 Using the chat template embedded in the GGUF file");
                rendered
            }
            Err(e) => {
                // Not fatal: fall back to the plain transcript the Default
                // template produces, which works with any model.
                eprintln!(
                    "⚠️ Auto template unavailable ({}), falling back to the transcript format",
                    e
                );
                // `render_history` already embedded a "Name: " prefix into
                // non-self content whenever there was more than one other
                // participant to tell apart; here that content is used as
                // is, and only the two bare cases (self, and the sole other
                // participant in solo chat) need a prefix added.
                let auto_show_names = auto_prefixes_names(speakers);
                let mut fallback = base_prompt.clone();
                for (is_self, content) in &chat_history {
                    if *is_self {
                        fallback += &format!("{}: {}\n", speakers.self_name(), content);
                    } else if auto_show_names {
                        fallback += &format!("{}\n", content);
                    } else {
                        fallback += &format!("{}: {}\n", speakers.user_name(), content);
                    }
                }
                format!("{}{}: ", fallback, speakers.self_name())
            }
        }
    } else {
        format!("{}{}: ", base_prompt, speakers.self_name())
    };
    let prompt_tokens = match model.str_to_token(&full_prompt, AddBos::Always) {
        Ok(tokens) => tokens,
        Err(e) => {
            return Err(std::io::Error::other(format!(
                "Failed to tokenize prompt: {}",
                e
            )))
        }
    };

    if prompt_tokens.len() >= context_size as usize {
        return Err(std::io::Error::other(format!(
            "Prompt is {} tokens but the context window is only {}",
            prompt_tokens.len(),
            context_size
        )));
    }

    // Feed the prompt in n_batch-sized chunks; only the final token needs logits.
    let last_prompt_index = prompt_tokens.len() - 1;
    let mut batch = LlamaBatch::new(N_BATCH as usize, 1);
    for (i, token) in prompt_tokens.iter().enumerate() {
        let is_last = i == last_prompt_index;
        if let Err(e) = batch.add(*token, i as i32, &[0], is_last) {
            return Err(std::io::Error::other(format!(
                "Failed to build prompt batch: {}",
                e
            )));
        }
        if batch.n_tokens() as u32 == N_BATCH || is_last {
            if let Err(e) = llama_context.decode(&mut batch) {
                return Err(std::io::Error::other(format!(
                    "Failed to evaluate prompt: {}",
                    e
                )));
            }
            batch.clear();
        }
    }

    let mut end_of_generation = String::new();
    let mut tokens_generated = 0u32;
    let mut first_token_recorded = false;
    let mut first_token_at: Option<std::time::Duration> = None;
    let trimmer = ReplyTrimmer::new(speakers);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_cur = prompt_tokens.len() as i32;

    while (tokens_generated as usize) < response_token_limit && (n_cur as u32) < context_size {
        // `sample` already accepts the token into the chain, so calling
        // `accept` here too would push it into the penalties ring buffer twice.
        let token = sampler.sample(&llama_context, -1);

        // Honour the model's own end-of-generation tokens, which the previous
        // string-only halting could not see.
        if model.is_eog_token(token) {
            break;
        }

        let piece = match model.token_to_piece(token, &mut decoder, false, None) {
            Ok(piece) => piece,
            Err(e) => {
                eprintln!("Failed to decode token: {}", e);
                break;
            }
        };

        // Track first token for time-to-first-token metric
        if !first_token_recorded {
            if let Ok(mut tracker) = INFERENCE_TRACKER.lock() {
                tracker.record_first_token(&session_id);
            }
            first_token_recorded = true;
            first_token_at = Some(start_time.elapsed());
        }

        tokens_generated += 1;
        end_of_generation.push_str(&piece);
        on_token(&piece);
        print!("{piece}");
        // Best-effort, same rationale as the flush in `load_model` above.
        let _ = std::io::stdout().flush();

        // Update token count for progress tracking
        if let Ok(mut tracker) = INFERENCE_TRACKER.lock() {
            tracker.update_token_count(&session_id, tokens_generated);
        }

        if trimmer.should_stop(&end_of_generation) {
            break;
        }

        batch.clear();
        if let Err(e) = batch.add(token, n_cur, &[0], true) {
            eprintln!("Failed to queue generated token: {}", e);
            break;
        }
        if let Err(e) = llama_context.decode(&mut batch) {
            eprintln!("Failed to decode generated token: {}", e);
            break;
        }
        n_cur += 1;
    }
    println!();

    let companion_text = trimmer.clean(&end_of_generation);
    match long_term_memory.add_entry(&format!(
        "{}{}: {}\n{}: {}\n",
        formatted_date,
        placeholder(&ParticipantId::USER),
        prompt,
        placeholder(&speakers.self_id),
        companion_text
    )) {
        Ok(_) => {}
        Err(e) => eprintln!("Error while adding message to long-term memory: {}", e),
    };

    // Complete the performance tracking session
    if let Ok(mut tracker) = INFERENCE_TRACKER.lock() {
        if let Err(e) = tracker.complete_session(&session_id) {
            eprintln!("Failed to complete performance tracking session: {}", e);
        }
    }

    // Record performance statistics
    let response_time = start_time.elapsed();
    INFERENCE_OPTIMIZER.record_response_time(response_time);

    // Enhanced performance telemetry
    let tokens_per_second = if tokens_generated > 0 {
        tokens_generated as f64 / response_time.as_secs_f64()
    } else {
        0.0
    };

    println!("⚡ Performance Metrics:");
    println!(
        "  • Model ready: {:.2}s ({})",
        model_ready.as_secs_f64(),
        if was_resident { "resident" } else { "loaded" }
    );
    if let Some(first_token_at) = first_token_at {
        println!(
            "  • Time to first token (from request): {:.2}s",
            first_token_at.as_secs_f64()
        );
    }
    println!("  • Total time: {:.2}s", response_time.as_secs_f64());
    println!("  • Tokens generated: {}", tokens_generated);
    println!("  • Tokens per second: {:.1}", tokens_per_second);
    println!("  • CPU cores used: {}", cpu_cores);
    println!("  • Context size: {} tokens", input_tokens);

    // Print performance stats periodically
    let stats = INFERENCE_OPTIMIZER.get_stats();
    if stats.total_requests.is_multiple_of(10) {
        println!(
            "📈 Performance: {} requests, avg response time: {:?}",
            stats.total_requests, stats.avg_response_time
        );
    }

    Ok(companion_text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::participants::ParticipantKind;

    const ATTITUDE_MARKER: &str = "MARKER: current relationship context";

    fn user() -> UserView {
        UserView {
            name: "TestUser".to_string(),
            persona: "a curious tester".to_string(),
        }
    }

    fn companion() -> CompanionView {
        CompanionView {
            name: "TestCompanion".to_string(),
            persona: "a helpful companion".to_string(),
            example_dialogue: "TestUser: Hi\nTestCompanion: Hello!".to_string(),
            first_message: "Hello!".to_string(),
            long_term_mem: 0,
            short_term_mem: 0,
            roleplay: false,
            dialogue_tuning: false,
            avatar_path: String::new(),
        }
    }

    /// Solo-chat speakers: `user` = TestUser, `char` = TestCompanion,
    /// generating for `char` — the two-participant case every solo-mode
    /// regression guard below is checked against.
    fn solo_speakers() -> PromptSpeakers {
        PromptSpeakers {
            registry: ParticipantRegistry::solo(&user().name, &companion().name, None),
            self_id: ParticipantId::CHAR,
        }
    }

    /// Three-speaker registry used by the multi-speaker tests below:
    /// `user` = Alice, `char` = Ada, `bot1` = Bob, in join order.
    fn three_speaker_registry() -> ParticipantRegistry {
        let mut registry = ParticipantRegistry::solo("Alice", "Ada", None);
        registry
            .insert(Participant {
                id: ParticipantId::parse("bot1").unwrap(),
                display_name: "Bob".to_string(),
                kind: ParticipantKind::HostBot,
                avatar: None,
            })
            .unwrap();
        registry
    }

    fn three_speaker_speakers(self_id: ParticipantId) -> PromptSpeakers {
        PromptSpeakers {
            registry: three_speaker_registry(),
            self_id,
        }
    }

    fn message(speaker_id: &str, content: &str) -> Message {
        Message {
            id: 0,
            ai: speaker_id != "user",
            speaker_id: speaker_id.to_string(),
            content: content.to_string(),
            created_at: String::new(),
        }
    }

    fn marker_index(joined: &str) -> usize {
        joined
            .find(ATTITUDE_MARKER)
            .expect("attitude marker missing from rendered prompt")
    }

    #[test]
    fn default_template_places_attitude_before_companion_start() {
        let components = build_base_components(
            &PromptTemplate::Default,
            &user(),
            &companion(),
            "",
            "",
            ATTITUDE_MARKER,
            &solo_speakers(),
        );
        let joined = components.join("");
        let start_index = joined
            .find("<START>")
            .expect("no <START> marker in rendered prompt");
        assert!(marker_index(&joined) < start_index);
    }

    #[test]
    fn auto_template_places_attitude_before_companion_start() {
        let components = build_base_components(
            &PromptTemplate::Auto,
            &user(),
            &companion(),
            "",
            "",
            ATTITUDE_MARKER,
            &solo_speakers(),
        );
        let joined = components.join("");
        let start_index = joined
            .find("<START>")
            .expect("no <START> marker in rendered prompt");
        assert!(marker_index(&joined) < start_index);
    }

    #[test]
    fn llama2_template_places_attitude_before_first_inst() {
        let components = build_base_components(
            &PromptTemplate::Llama2,
            &user(),
            &companion(),
            "",
            "",
            ATTITUDE_MARKER,
            &solo_speakers(),
        );
        let joined = components.join("");
        let inst_index = joined
            .find("[/INST]")
            .expect("no [/INST] marker in rendered prompt");
        assert!(marker_index(&joined) < inst_index);
    }

    #[test]
    fn mistral_template_places_attitude_before_first_inst() {
        let components = build_base_components(
            &PromptTemplate::Mistral,
            &user(),
            &companion(),
            "",
            "",
            ATTITUDE_MARKER,
            &solo_speakers(),
        );
        let joined = components.join("");
        let inst_index = joined
            .find("[/INST]")
            .expect("no [/INST] marker in rendered prompt");
        assert!(marker_index(&joined) < inst_index);
    }

    #[test]
    fn empty_attitude_context_adds_no_component() {
        let with_attitude = build_base_components(
            &PromptTemplate::Default,
            &user(),
            &companion(),
            "",
            "",
            ATTITUDE_MARKER,
            &solo_speakers(),
        );
        let without_attitude = build_base_components(
            &PromptTemplate::Default,
            &user(),
            &companion(),
            "",
            "",
            "",
            &solo_speakers(),
        );
        assert_eq!(with_attitude.len(), without_attitude.len() + 1);
        assert!(!without_attitude.join("").contains(ATTITUDE_MARKER));
    }

    #[test]
    fn three_speaker_header_lists_everyone_and_names_other_bots() {
        let speakers = three_speaker_speakers(ParticipantId::CHAR);
        let components = build_base_components(
            &PromptTemplate::Default,
            &user(),
            &companion(),
            "",
            "",
            "",
            &speakers,
        );
        let joined = components.join("");
        assert!(joined.contains("Alice, Ada and Bob"));
        assert!(joined.contains("Also present: Bob."));
    }

    #[test]
    fn solo_header_has_no_also_present_component() {
        let components = build_base_components(
            &PromptTemplate::Default,
            &user(),
            &companion(),
            "",
            "",
            "",
            &solo_speakers(),
        );
        let joined = components.join("");
        assert!(joined.contains("TestUser and TestCompanion"));
        assert!(!joined.contains("Also present"));
    }

    #[test]
    fn join_names_covers_one_two_and_three_or_more() {
        assert_eq!(join_names(&["Alice"]), "Alice");
        assert_eq!(join_names(&["Alice", "Bob"]), "Alice and Bob");
        assert_eq!(
            join_names(&["Alice", "Bob", "Carol"]),
            "Alice, Bob and Carol"
        );
    }

    #[test]
    fn render_history_default_template_lists_every_speaker_by_name() {
        let speakers = three_speaker_speakers(ParticipantId::parse("bot1").unwrap());
        let managed = vec![
            message("user", "hi"),
            message("char", "hello"),
            message("bot1", "hey"),
        ];
        let rendered = render_history(&managed, &speakers, &PromptTemplate::Default);
        assert_eq!(rendered.spliced, "Alice: hi\nAda: hello\nBob: hey\n");
    }

    #[test]
    fn render_history_auto_only_self_turns_are_bare_and_marked_assistant() {
        let speakers = three_speaker_speakers(ParticipantId::parse("bot1").unwrap());
        let managed = vec![
            message("user", "hi"),
            message("char", "hello"),
            message("bot1", "hey"),
        ];
        let rendered = render_history(&managed, &speakers, &PromptTemplate::Auto);
        assert_eq!(
            rendered.chat_history,
            vec![
                (false, "Alice: hi".to_string()),
                (false, "Ada: hello".to_string()),
                (true, "hey".to_string()),
            ]
        );
    }

    #[test]
    fn render_history_auto_stays_bare_in_solo_chat() {
        let managed = vec![message("user", "hi"), message("char", "hello")];
        let rendered = render_history(&managed, &solo_speakers(), &PromptTemplate::Auto);
        assert_eq!(
            rendered.chat_history,
            vec![(false, "hi".to_string()), (true, "hello".to_string())]
        );
    }

    #[test]
    fn merge_consecutive_turns_combines_adjacent_non_self_speakers() {
        let history = vec![
            (false, "Alice: hi".to_string()),
            (false, "Ada: hello".to_string()),
            (true, "hey".to_string()),
        ];
        let merged = merge_consecutive_turns(&history);
        assert_eq!(
            merged,
            vec![
                (false, "Alice: hi\nAda: hello".to_string()),
                (true, "hey".to_string()),
            ]
        );
    }

    #[test]
    fn reply_trimmer_should_stop_fires_on_any_participant_name() {
        let speakers = three_speaker_speakers(ParticipantId::parse("bot1").unwrap());
        let trimmer = ReplyTrimmer::new(&speakers);
        assert!(trimmer.should_stop("sure thing\nBob:"));
        assert!(trimmer.should_stop("sure thing\nAlice:"));
    }

    #[test]
    fn reply_trimmer_clean_cuts_at_the_earliest_extra_turn() {
        let speakers = three_speaker_speakers(ParticipantId::parse("bot1").unwrap());
        let trimmer = ReplyTrimmer::new(&speakers);
        assert_eq!(trimmer.clean("sure!\nBob: hi"), "sure!");
    }

    /// Regression guard for the #127 review finding: a non-self speaker's
    /// impersonated line followed by more text must be cut entirely, not
    /// merely have its name stripped by the `line_markers` removal.
    #[test]
    fn reply_trimmer_clean_cuts_a_non_self_extra_turn_followed_by_text() {
        let speakers = three_speaker_speakers(ParticipantId::CHAR);
        let trimmer = ReplyTrimmer::new(&speakers);
        assert_eq!(trimmer.clean("sure!\nAlice: hi"), "sure!");
    }

    #[test]
    fn reply_trimmer_reproduces_solo_behaviour() {
        let trimmer = ReplyTrimmer::new(&solo_speakers());
        assert_eq!(trimmer.clean("Hello there!\nTestUser:"), "Hello there!");
    }

    #[test]
    fn in_memory_transcript_returns_the_tail_oldest_first() {
        let transcript = InMemoryTranscript(vec![
            message("user", "a"),
            message("char", "b"),
            message("user", "c"),
        ]);
        let recent = transcript.recent_messages(2).unwrap();
        assert_eq!(
            recent
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "c"]
        );
    }
}
