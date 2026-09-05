use chrono::{DateTime, Local};
use std::io::Write;

use crate::attitude_formatter::AttitudeFormatter;
use crate::context_manager::ContextManager;
use crate::database::{
    contains_time_question, get_current_date, CompanionView, ConfigView, Database, Device, Message,
    NewMessage, PromptTemplate, UserView,
};
use crate::dialogue_tuning::DialogueTuning;
use crate::gpu_allocator::GpuAllocator;
use crate::inference_optimizer::INFERENCE_OPTIMIZER;
use crate::inference_performance::{ModelConfig, INFERENCE_TRACKER};
use crate::long_term_mem::LongTermMem;

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

/// Serialises generation. Each call loads its own copy of the model, so two
/// concurrent requests would hold two full copies in memory at once.
static GENERATION_LOCK: Mutex<()> = Mutex::new(());

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

/// Renders the prompt using the chat template stored inside the GGUF file.
///
/// Consecutive turns from the same speaker are merged, because several chat
/// templates reject a history that does not strictly alternate.
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

    let mut merged: Vec<(bool, String)> = Vec::with_capacity(history.len());
    for (is_ai, content) in history {
        match merged.last_mut() {
            Some((last_is_ai, last_content)) if last_is_ai == is_ai => {
                last_content.push('\n');
                last_content.push_str(content);
            }
            _ => merged.push((*is_ai, content.clone())),
        }
    }

    for (is_ai, content) in &merged {
        let role = if *is_ai { "assistant" } else { "user" };
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

/// Builds the system-portion components of the prompt for the given
/// template, inserting `attitude_context` (when non-empty) as its own
/// component before the template's instruct terminator, so it always ends up
/// inside the system block rather than after the conversation history.
fn build_base_components(
    template: &PromptTemplate,
    user: &UserView,
    companion: &CompanionView,
    rp: &str,
    tuned_dialogue: &str,
    attitude_context: &str,
) -> Vec<String> {
    if *template == PromptTemplate::Default || *template == PromptTemplate::Auto {
        let mut components = vec![
            format!(
                "Text transcript of a conversation between {} and {}. {}\n",
                user.name, companion.name, rp
            ),
            format!(
                "{}'s Persona: {}\n",
                user.name,
                user.persona
                    .replace("{{char}}", &companion.name)
                    .replace("{{user}}", &user.name)
            ),
        ];
        if !attitude_context.is_empty() {
            components.push(attitude_context.to_string());
        }
        components.push(format!(
            "{}'s Persona: {}\n<START>\n",
            companion.name,
            companion
                .persona
                .replace("{{char}}", &companion.name)
                .replace("{{user}}", &user.name)
        ));
        components.push(format!(
            "{}\n<START>\n",
            companion
                .example_dialogue
                .replace("{{char}}", &companion.name)
                .replace("{{user}}", &user.name)
        ));
        components.push(format!("{}\n<START>\n", tuned_dialogue));
        components
    } else if *template == PromptTemplate::Llama2 {
        let mut components = vec![format!(
            "<<SYS>>\nYou are {}, {}\n",
            companion.name,
            companion
                .persona
                .replace("{{char}}", &companion.name)
                .replace("{{user}}", &user.name)
        )];
        if !attitude_context.is_empty() {
            components.push(attitude_context.to_string());
        }
        components.push(format!(
            "you are talking with {}, {} is {}\n{}\n[INST]\n",
            user.name,
            user.name,
            user.persona
                .replace("{{char}}", &companion.name)
                .replace("{{user}}", &user.name),
            rp
        ));
        components.push(format!(
            "{}\n",
            companion
                .example_dialogue
                .replace("{{char}}", &companion.name)
                .replace("{{user}}", &user.name)
        ));
        components.push(format!("{}\n[/INST]\n", tuned_dialogue));
        components
    } else {
        let mut components = vec![
            format!(
                "<s>[INST]Text transcript of a conversation between {} and {}. {}\n",
                user.name, companion.name, rp
            ),
            format!(
                "{}'s Persona: {}\n",
                user.name,
                user.persona
                    .replace("{{char}}", &companion.name)
                    .replace("{{user}}", &user.name)
            ),
        ];
        if !attitude_context.is_empty() {
            components.push(attitude_context.to_string());
        }
        components.push(format!(
            "{}'s Persona: {}[/INST]\n<s>[INST]\n",
            companion.name,
            companion
                .persona
                .replace("{{char}}", &companion.name)
                .replace("{{user}}", &user.name)
        ));
        components.push(format!(
            "{}[/INST]\n<s>[INST]\n",
            companion
                .example_dialogue
                .replace("{{char}}", &companion.name)
                .replace("{{user}}", &user.name)
        ));
        components.push(format!("{}[/INST]\n", tuned_dialogue));
        components
    }
}

/// Generates a reply and returns it once generation finishes.
///
/// # Errors
/// Propagates model load, tokenization and decode failures as
/// `std::io::ErrorKind::Other`.
pub fn prompt(prompt: &str, companion_id: i32) -> Result<String, std::io::Error> {
    generate(prompt, companion_id, &mut |_token| {})
}

/// Generates a reply, invoking `on_token` with each token as it is produced.
///
/// The callback runs on the generating thread, so it must not block.
///
/// # Errors
/// Propagates model load, tokenization and decode failures as
/// `std::io::ErrorKind::Other`.
pub fn prompt_streaming(
    prompt: &str,
    companion_id: i32,
    on_token: &mut dyn FnMut(&str),
) -> Result<String, std::io::Error> {
    generate(prompt, companion_id, on_token)
}

fn generate(
    prompt: &str,
    companion_id: i32,
    on_token: &mut dyn FnMut(&str),
) -> Result<String, std::io::Error> {
    let _generation_guard = GENERATION_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let start_time = std::time::Instant::now();
    let long_term_memory = match LongTermMem::connect() {
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

    // llama.cpp expresses GPU offloading as a single layer count, so resolve
    // that number before building the model parameters.
    let gpu_layers: u32 = if config.device == Device::GPU || config.device == Device::Metal {
        if config.dynamic_gpu_allocation {
            let allocator = GpuAllocator::new()
                .with_safety_margin(config.gpu_safety_margin)
                .with_min_free_vram(config.min_free_vram_mb);

            match allocator.detect_gpu_memory(&config.device) {
                Ok(gpu_info) => {
                    println!("🔍 GPU Detection: {}", gpu_info);

                    let vram_limit = if config.vram_limit_gb > 0 {
                        Some(config.vram_limit_gb as f32)
                    } else {
                        None
                    };

                    // Estimate model size (this would ideally come from model metadata)
                    let estimated_model_size_mb = 4096;
                    let estimated_total_layers = 32;

                    // Use the new optimized allocation method
                    let allocation = allocator.calculate_optimal_layers_v2(
                        &gpu_info,
                        &config.llm_model_path,
                        estimated_model_size_mb,
                        estimated_total_layers,
                        vram_limit,
                    );

                    println!("🎯 Dynamic Allocation: {}", allocation);
                    allocation.gpu_layers as u32
                }
                Err(e) => {
                    eprintln!("⚠️ GPU detection failed, using configured layers: {}", e);
                    config.gpu_layers as u32
                }
            }
        } else {
            println!("📌 Static Allocation: {} GPU layers", config.gpu_layers);
            config.gpu_layers as u32
        }
    } else {
        println!("💻 CPU-only inference mode");
        0
    };

    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(gpu_layers)
        .with_use_mmap(true); // Memory-mapped model loading reduces RAM usage

    let backend = llama_backend()?;

    print!("📚 Loading model... ");
    std::io::stdout().flush().unwrap();
    let model = match LlamaModel::load_from_file(
        backend,
        std::path::Path::new(&config.llm_model_path),
        &model_params,
    ) {
        Ok(model) => model,
        Err(e) => {
            return Err(std::io::Error::other(format!(
                "Failed to load llm model: {}",
                e
            )))
        }
    };
    println!("✓ Model loaded");

    // Calculate CPU cores for optimizations
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4); // Fallback to 4 cores if detection fails

    println!("🚀 Generating AI response with optimized session...");
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
                user.name, dialogue.user_msg, companion.name, dialogue.ai_msg
            );
        };
    }
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
        let context =
            attitude_formatter.format_attitude_context(&attitudes, &third_parties, &user.name);
        if !context.is_empty() {
            format!("\n{}\n", context)
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    if !attitude_context.is_empty() {
        println!(
            "✓ Attitude context integrated: {} characters",
            attitude_context.len()
        );
    }

    // Build base prompt components for caching optimization
    // Auto renders through the model's own chat template, so its system content
    // must be plain prose; the Mistral branch below would embed [INST] markers.
    let base_components = build_base_components(
        &config.prompt_template,
        &user,
        &companion,
        rp,
        &tuned_dialogue,
        &attitude_context,
    );

    // Use cache optimization for base prompt construction
    let (optimized_base_prompt, cache_hit) =
        INFERENCE_OPTIMIZER.optimize_prompt_construction(&base_components, "", &[]);

    base_prompt = optimized_base_prompt;

    if cache_hit {
        println!("✓ Cache hit for base prompt construction");
    } else {
        println!("✗ Cache miss - caching base prompt for future use");
    }
    if companion.long_term_mem > 0 {
        let long_term_memory_entries: Vec<String> =
            match long_term_memory.get_matches(prompt, companion.long_term_mem) {
                Ok(entries) => entries,
                Err(e) => {
                    eprintln!("Error while getting long term memory entries: {}", e);
                    return Err(std::io::Error::other(
                        "Error while getting long term memory entries",
                    ));
                }
            };
        for entry in long_term_memory_entries {
            if config.prompt_template == PromptTemplate::Llama2 {
                base_prompt += &format!("[INST]{}[/INST]\n", entry)
                    .replace("{{char}}", &companion.name)
                    .replace("{{user}}", &user.name);
            } else if config.prompt_template == PromptTemplate::Mistral {
                base_prompt += &format!("<s>[INST]{}[/INST]\n", entry)
                    .replace("{{char}}", &companion.name)
                    .replace("{{user}}", &user.name);
            } else {
                base_prompt += &entry
                    .replace("{{char}}", &companion.name)
                    .replace("{{user}}", &user.name);
            }
        }
    }
    // Initialize context manager for intelligent memory management
    let context_manager = ContextManager::new(config.clone());

    let short_term_memory_entries: Vec<Message> = match Database::get_x_messages(
        if companion.short_term_mem > 0 {
            companion.short_term_mem
        } else {
            50
        },
        0,
    ) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("Error while getting short term memory entries: {}", e);
            return Err(std::io::Error::other(
                "Error while getting short term memory entries",
            ));
        }
    };

    // Apply context management to optimize memory usage
    let managed_messages = context_manager.manage_message_context(short_term_memory_entries);
    let short_term_mem_len = managed_messages.len();
    // Role-tagged history, used only by the Auto template. The string templates
    // below keep splicing turns straight into base_prompt.
    let mut chat_history: Vec<(bool, String)> = Vec::with_capacity(managed_messages.len());
    for (message_counter, message) in (1..).zip(managed_messages.iter()) {
        let prefix = if message.ai {
            &companion.name
        } else {
            &user.name
        };
        let text = &message.content;
        let mut formatted_message = format!("{}: {}\n", prefix, text);
        let inject_time =
            message_counter == short_term_mem_len && contains_time_question(&formatted_message);
        if inject_time {
            formatted_message = format!(
                "\n* it's currently {} *\n{}",
                get_current_date(),
                formatted_message
            );
        }
        if config.prompt_template == PromptTemplate::Auto {
            // The chat template supplies the speaker framing, so the message
            // carries its own text rather than a "Name: " prefix.
            let mut content = text.clone();
            if inject_time {
                content = format!("* it's currently {} *\n{}", get_current_date(), content);
            }
            chat_history.push((message.ai, content));
        } else if config.prompt_template == PromptTemplate::Llama2 {
            if !message.ai {
                base_prompt += &format!("[INST]{}", formatted_message);
            } else {
                base_prompt += &format!("{}[/INST]\n", formatted_message);
            }
        } else if config.prompt_template == PromptTemplate::Mistral {
            if !message.ai {
                base_prompt += &format!("<s>[INST]{}", formatted_message);
            } else {
                base_prompt += &format!("{}[/INST]\n", formatted_message);
            }
        } else {
            base_prompt += &formatted_message;
        }
    }

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
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(model.n_vocab(), 64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.8),
        LlamaSampler::dist(rand::random::<u32>()),
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
                let mut fallback = base_prompt.clone();
                for (is_ai, content) in &chat_history {
                    let speaker = if *is_ai { &companion.name } else { &user.name };
                    fallback += &format!("{}: {}\n", speaker, content);
                }
                format!("{}{}: ", fallback, companion.name)
            }
        }
    } else {
        format!("{}{}: ", base_prompt, companion.name)
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
    let eog = format!("\n{}:", user.name);
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
        }

        tokens_generated += 1;
        end_of_generation.push_str(&piece);
        on_token(&piece);
        print!("{piece}");
        std::io::stdout().flush().unwrap();

        // Update token count for progress tracking
        if let Ok(mut tracker) = INFERENCE_TRACKER.lock() {
            tracker.update_token_count(&session_id, tokens_generated);
        }

        if end_of_generation.contains(&eog)
            || end_of_generation.contains("[/INST]")
            || end_of_generation.contains("<</SYS>>")
            || end_of_generation.contains("[s]")
            || end_of_generation.contains(&format!("{}:", companion.name))
            || end_of_generation.contains(&format!("{}:", user.name))
            || end_of_generation.contains("<|user|>")
        {
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

    let x: String = end_of_generation
        .replace(&eog, "")
        .replace("[INST]", "")
        .replace("[/INST]", "")
        .replace("<</SYS>>", "")
        .replace("<s>", "")
        .replace("</s>", "")
        .replace("<|user|>", "");
    let companion_text = x
        .split(&format!("\n{}: ", companion.name))
        .next()
        .unwrap_or("");
    match Database::insert_message(NewMessage {
        ai: true,
        content: companion_text.to_string(),
    }) {
        Ok(_) => {}
        Err(e) => eprintln!(
            "Error while adding message to database/short-term memory: {}",
            e
        ),
    };
    match long_term_memory.add_entry(&format!(
        "{}{}: {}\n{}: {}\n",
        formatted_date, "{{user}}", prompt, "{{char}}", companion_text
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
    println!("  • Total time: {:.2}s", response_time.as_secs_f64());
    println!("  • Tokens generated: {}", tokens_generated);
    println!("  • Tokens per second: {:.1}", tokens_per_second);
    println!("  • CPU cores used: {}", cpu_cores);
    println!("  • Context size: {} tokens", input_tokens);

    // Print cache statistics periodically
    let stats = INFERENCE_OPTIMIZER.get_stats();
    if stats.total_requests.is_multiple_of(10) {
        let (cache_size, cache_hits, hit_rate) = INFERENCE_OPTIMIZER.get_cache_stats();
        println!(
            "📊 Cache Stats: {} entries, {} hits, {:.2}% hit rate",
            cache_size,
            cache_hits,
            hit_rate * 100.0
        );
        println!(
            "📈 Performance: {} requests, avg response time: {:?}",
            stats.total_requests, stats.avg_response_time
        );
    }

    Ok(companion_text.trim_start().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        );
        let without_attitude =
            build_base_components(&PromptTemplate::Default, &user(), &companion(), "", "", "");
        assert_eq!(with_attitude.len(), without_attitude.len() + 1);
        assert!(!without_attitude.join("").contains(ATTITUDE_MARKER));
    }
}
