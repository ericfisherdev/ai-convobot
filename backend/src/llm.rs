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
use crate::long_term_mem::LongTermMem;

pub fn prompt(prompt: &str) -> Result<String, std::io::Error> {
    let start_time = std::time::Instant::now();
    let long_term_memory = match LongTermMem::connect() {
        Ok(ltm) => ltm,
        Err(e) => {
            eprintln!("Error while connecting to tantivy: {}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Error while connecting to tantivy",
            ));
        }
    };
    let local: DateTime<Local> = Local::now();
    let formatted_date = local.format("* at %A %d.%m.%Y %H:%M *\n").to_string();
    let config: ConfigView = match Database::get_config() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Error while getting config: {}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Error while getting config",
            ));
        }
    };
    let user: UserView = match Database::get_user_data() {
        Ok(user) => user,
        Err(e) => {
            eprintln!("Error while getting user data: {}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Error while getting user data",
            ));
        }
    };
    let companion: CompanionView = match Database::get_companion_data() {
        Ok(companion) => companion,
        Err(e) => {
            eprintln!("Error while getting companion data: {}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Error while getting companion data",
            ));
        }
    };

    let llama_model_params = {
        let mut params = llm::ModelParameters::default();
        if config.device == Device::GPU || config.device == Device::Metal {
            params.use_gpu = true;

            // Use dynamic GPU allocation if enabled
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

                        let allocation = allocator.calculate_optimal_layers(
                            &gpu_info,
                            estimated_model_size_mb,
                            estimated_total_layers,
                            vram_limit,
                        );

                        println!("🎯 Dynamic Allocation: {}", allocation);
                        params.gpu_layers = Some(allocation.gpu_layers);
                    }
                    Err(e) => {
                        eprintln!("⚠️ GPU detection failed, using configured layers: {}", e);
                        params.gpu_layers = Some(config.gpu_layers);
                    }
                }
            } else {
                println!("📌 Static Allocation: {} GPU layers", config.gpu_layers);
                params.gpu_layers = Some(config.gpu_layers);
            }
        } else {
            params.use_gpu = false;
            params.gpu_layers = None;
            println!("💻 CPU-only inference mode");
        }
        params
    };

    let llama = llm::load(
        std::path::Path::new(&config.llm_model_path),
        llm::TokenizerSource::Embedded,
        llama_model_params,
        llm::load_progress_callback_stdout,
    );

    let llama = match llama {
        Ok(llama) => llama,
        Err(e) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to load llm model: {}", e.to_string()),
            ))
        }
    };

    let mut session = llama.start_session(Default::default());
    println!("Generating ai response...");
    let mut base_prompt: String;
    let mut rp: &str = "";
    let mut tuned_dialogue: String = String::from("");
    if companion.roleplay {
        rp = "gestures and other non-verbal actions are written between asterisks (for example, *waves hello* or *moves closer*)";
    }
    if companion.dialogue_tuning {
        match DialogueTuning::get_random_dialogue() {
            Ok(dialogue) => {
                tuned_dialogue = format!(
                    "{}: {}\n{}: {}",
                    &user.name, &dialogue.user_msg, &companion.name, &dialogue.ai_msg
                );
            }
            Err(_) => {}
        };
    }
    // Build base prompt components for caching optimization
    let base_components = if config.prompt_template == PromptTemplate::Default {
        vec![
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
            format!(
                "{}'s Persona: {}\n<START>\n",
                companion.name,
                companion
                    .persona
                    .replace("{{char}}", &companion.name)
                    .replace("{{user}}", &user.name)
            ),
            format!(
                "{}\n<START>\n",
                companion
                    .example_dialogue
                    .replace("{{char}}", &companion.name)
                    .replace("{{user}}", &user.name)
            ),
            format!("{}\n<START>\n", &tuned_dialogue),
        ]
    } else if config.prompt_template == PromptTemplate::Llama2 {
        vec![
            format!(
                "<<SYS>>\nYou are {}, {}\n",
                companion.name,
                companion
                    .persona
                    .replace("{{char}}", &companion.name)
                    .replace("{{user}}", &user.name)
            ),
            format!(
                "you are talking with {}, {} is {}\n{}\n[INST]\n",
                user.name,
                user.name,
                user.persona
                    .replace("{{char}}", &companion.name)
                    .replace("{{user}}", &user.name),
                rp
            ),
            format!(
                "{}\n",
                companion
                    .example_dialogue
                    .replace("{{char}}", &companion.name)
                    .replace("{{user}}", &user.name)
            ),
            format!("{}\n[/INST]\n", &tuned_dialogue),
        ]
    } else {
        vec![
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
            format!(
                "{}'s Persona: {}[/INST]\n<s>[INST]\n",
                companion.name,
                companion
                    .persona
                    .replace("{{char}}", &companion.name)
                    .replace("{{user}}", &user.name)
            ),
            format!(
                "{}[/INST]\n<s>[INST]\n",
                companion
                    .example_dialogue
                    .replace("{{char}}", &companion.name)
                    .replace("{{user}}", &user.name)
            ),
            format!("{}[/INST]\n", &tuned_dialogue),
        ]
    };

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
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
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
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Error while getting short term memory entries",
            ));
        }
    };

    // Apply context management to optimize memory usage
    let managed_messages = context_manager.manage_message_context(short_term_memory_entries);
    let mut message_counter = 1;
    let short_term_mem_len = managed_messages.len();
    for message in &managed_messages {
        let prefix = if message.ai {
            &companion.name
        } else {
            &user.name
        };
        let text = &message.content;
        let mut formatted_message = format!("{}: {}\n", prefix, text);
        if message_counter == short_term_mem_len && contains_time_question(&formatted_message) {
            formatted_message = format!(
                "\n* it's currently {} *\n{}",
                get_current_date(),
                formatted_message
            );
        }
        if config.prompt_template == PromptTemplate::Llama2 {
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
        message_counter += 1;
    }

    // Load and integrate attitude context
    let attitude_formatter = AttitudeFormatter::new();
    let attitudes = match Database::get_all_companion_attitudes(1) {
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

    // Insert attitude context before conversation history
    if !attitude_context.is_empty() {
        base_prompt += &attitude_context;
        println!(
            "✓ Attitude context integrated: {} characters",
            attitude_context.len()
        );
    }

    // Calculate token usage for memory management
    let system_tokens = ContextManager::estimate_tokens(&base_prompt);
    let attitude_tokens = ContextManager::estimate_tokens(&attitude_context);
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

    let mut end_of_generation = String::new();
    let eog = format!("\n{}:", user.name);
    let res = session.infer::<std::convert::Infallible>(
        llama.as_ref(),
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: llm::Prompt::Text(&format!("{}{}: ", &base_prompt, companion.name)),
            parameters: &llm::InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: Some(response_token_limit),
        },
        &mut Default::default(),
        |t| {
            match t {
                llm::InferenceResponse::SnapshotToken(_) => { /*print!("{token}");*/ }
                llm::InferenceResponse::PromptToken(_) => { /*print!("{token}");*/ }
                llm::InferenceResponse::InferredToken(token) => {
                    //  x = x.clone()+&token;
                    end_of_generation.push_str(&token);
                    print!("{token}");
                    if end_of_generation.contains(&eog)
                        || end_of_generation.contains("[/INST]")
                        || end_of_generation.contains("<</SYS>>")
                        || end_of_generation.contains("[s]")
                        || end_of_generation.contains(&format!("{}:", &companion.name))
                        || end_of_generation.contains(&format!("{}:", &user.name))
                        || end_of_generation.contains("<|user|>")
                    {
                        return Ok(llm::InferenceFeedback::Halt);
                    }
                }
                llm::InferenceResponse::EotToken => {}
            }
            std::io::stdout().flush().unwrap();
            Ok(llm::InferenceFeedback::Continue)
        },
    );
    let x: String = end_of_generation
        .replace(&eog, "")
        .replace("[INST]", "")
        .replace("[/INST]", "")
        .replace("<</SYS>>", "")
        .replace("<s>", "")
        .replace("</s>", "")
        .replace("<|user|>", "");
    match res {
        Ok(result) => println!("\n\nInference stats:\n{result}"),
        Err(err) => println!("\n{err}"),
    }
    let companion_text = x
        .split(&format!("\n{}: ", &companion.name))
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
        formatted_date, "{{user}}", &prompt, "{{char}}", &companion_text
    )) {
        Ok(_) => {}
        Err(e) => eprintln!("Error while adding message to long-term memory: {}", e),
    };

    // Record performance statistics
    let response_time = start_time.elapsed();
    INFERENCE_OPTIMIZER.record_response_time(response_time);

    // Print cache statistics periodically
    let stats = INFERENCE_OPTIMIZER.get_stats();
    if stats.total_requests % 10 == 0 {
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
