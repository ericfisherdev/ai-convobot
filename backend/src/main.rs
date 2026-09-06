use actix_web::{delete, get, post, put, web, App, HttpResponse, HttpServer};
use futures_util::StreamExt as _;
mod database;
use database::{
    CompanionAttitude, CompanionView, ConfigModify, Database, Message, NewMessage,
    ThirdPartyInteraction, UserView,
};
mod long_term_mem;
use long_term_mem::LongTermMem;
mod dialogue_tuning;
use dialogue_tuning::DialogueTuning;
mod character_card;
use character_card::CharacterCard;
use serde::Deserialize;
mod llm;
mod model_cache;
use crate::llm::{assemble_prompt, prompt, prompt_streaming};
use uuid::Uuid;
mod context_manager;
mod inference_optimizer;
use crate::inference_optimizer::{AttitudeStreamUpdate, StreamChunk, INFERENCE_OPTIMIZER};
mod session_manager;
mod token_budget;
use crate::session_manager::SessionManager;
mod attitude_engine;
use crate::attitude_engine::{LexiconScorer, ScorerConfig, TurnScorer};
mod attitude_formatter;
mod gpu_allocator;
use crate::gpu_allocator::{GpuAllocator, LayerAllocation};
mod system_memory;
// Removed unused system_memory imports
mod inference_performance;
use crate::inference_performance::{ModelConfig, ResponseEstimate, INFERENCE_TRACKER};
mod llm_scanner;
use crate::llm_scanner::LlmScanner;
mod turn_slot;
use crate::turn_slot::ACTIVE_TURN;
#[cfg(test)]
mod simple_tests;

use std::fs;
use std::fs::File;
use std::io::{Read, Write};

/// Runs synchronous work (rusqlite, tantivy) on actix's blocking pool so the
/// worker thread stays free to serve other requests while it runs.
///
/// `failure` is the user-facing prefix already used by the handlers ("Error
/// while getting config"); it is logged with the cause and returned as the
/// 500 body with the usual ", check logs for more information" tail.
async fn off_worker<T, E>(
    failure: &'static str,
    task: impl FnOnce() -> Result<T, E> + Send + 'static,
) -> Result<T, HttpResponse>
where
    T: Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    match web::block(task).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => {
            eprintln!("{}: {}", failure, e);
            Err(HttpResponse::InternalServerError()
                .body(format!("{}, check logs for more information", failure)))
        }
        Err(blocking) => {
            eprintln!("{}: blocking task failed: {}", failure, blocking);
            Err(HttpResponse::InternalServerError()
                .body(format!("{}, check logs for more information", failure)))
        }
    }
}

/// Reads `AI_COMPANION_WORKERS` to override actix's default worker count
/// (`available_parallelism`). Follows the same convention as
/// `AI_COMPANION_SAMPLER_SEED` / `AI_COMPANION_ATTITUDE_DEBUG` in `llm.rs`.
///
/// Returns `None` when unset (default worker count) or when the value fails
/// to parse as a `usize >= 1` (logged, default worker count kept — falling
/// back silently would defeat the point of the variable).
fn configured_workers() -> Option<usize> {
    match std::env::var("AI_COMPANION_WORKERS") {
        Ok(value) => match value.trim().parse::<usize>() {
            Ok(0) => {
                eprintln!("AI_COMPANION_WORKERS must be at least 1 (got 0); using the default worker count");
                None
            }
            Ok(workers) => Some(workers),
            Err(e) => {
                eprintln!(
                    "AI_COMPANION_WORKERS is not a usize ({:?}: {}); using the default worker count",
                    value, e
                );
                None
            }
        },
        Err(_) => None,
    }
}

#[cfg(test)]
mod off_worker_tests {
    use super::*;
    use actix_web::body::to_bytes;
    use actix_web::http::StatusCode;

    #[actix_web::test]
    async fn off_worker_returns_the_task_value() {
        let result = off_worker("x", || Ok::<_, rusqlite::Error>(7)).await;
        assert_eq!(result.ok(), Some(7));
    }

    #[actix_web::test]
    async fn off_worker_maps_a_task_error_to_a_500_with_the_existing_body() {
        let response = off_worker("x", || Err::<(), _>(rusqlite::Error::QueryReturnedNoRows))
            .await
            .expect_err("task returned an error");
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = to_bytes(response.into_body()).await.unwrap();
        assert_eq!(body, "x, check logs for more information");
    }

    #[actix_web::test]
    async fn off_worker_maps_a_panicking_task_to_a_500() {
        let response = off_worker("x", || -> Result<(), rusqlite::Error> { panic!("boom") })
            .await
            .expect_err("a panicking task should map to an error response");
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[actix_web::test]
    async fn off_worker_does_not_stall_other_requests_on_the_same_runtime() {
        let slow = off_worker("slow", || {
            std::thread::sleep(std::time::Duration::from_millis(300));
            Ok::<_, rusqlite::Error>(())
        });
        let fast = std::future::ready(());
        tokio::select! {
            _ = slow => panic!("the blocking task should not win the race against an already-ready future"),
            _ = fast => {}
        }
    }

    // Sets and clears the env var within one test rather than one assertion
    // per test fn, so parallel test threads never race each other over it.
    #[test]
    fn configured_workers_reads_the_env_var() {
        std::env::remove_var("AI_COMPANION_WORKERS");
        assert_eq!(configured_workers(), None);

        std::env::set_var("AI_COMPANION_WORKERS", "3");
        assert_eq!(configured_workers(), Some(3));

        std::env::set_var("AI_COMPANION_WORKERS", "0");
        assert_eq!(configured_workers(), None);

        std::env::set_var("AI_COMPANION_WORKERS", "not-a-number");
        assert_eq!(configured_workers(), None);

        std::env::remove_var("AI_COMPANION_WORKERS");
    }
}

#[get("/")]
async fn index() -> HttpResponse {
    HttpResponse::Ok().body(include_str!("../../dist/index.html"))
}

#[get("/assets/index-4rust.js")]
async fn js() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("application/javascript")
        .body(include_str!("../../dist/assets/index-4rust.js"))
}

#[get("/assets/index-4rust2.js")]
async fn js2() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("application/javascript")
        .body(include_str!("../../dist/assets/index-4rust2.js"))
}

#[get("/assets/index-4rust.css")]
async fn css() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/css")
        .body(include_str!("../../dist/assets/index-4rust.css"))
}

#[get("/ai_companion_logo.jpg")]
async fn project_logo() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("image/jpeg")
        .body(&include_bytes!("../../dist/ai_companion_logo.jpg")[..])
}

#[get("/assets/companion_avatar-4rust.jpg")]
async fn companion_avatar_img() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("image/jpeg")
        .body(&include_bytes!("../../dist/assets/companion_avatar-4rust.jpg")[..])
}

#[get("/manifest.json")]
async fn manifest() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("application/manifest+json")
        .body(include_str!("../../dist/manifest.json"))
}

#[get("/sw.js")]
async fn service_worker() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("application/javascript")
        .body(include_str!("../../dist/sw.js"))
}

#[get("/assets/avatar.png")]
async fn companion_avatar_custom() -> actix_web::Result<actix_web::HttpResponse> {
    match File::open("assets/avatar.png") {
        Ok(mut file) => {
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            Ok(actix_web::HttpResponse::Ok()
                .content_type("image/png")
                .body(buffer))
        }
        Err(_) => Err(actix_web::error::ErrorNotFound("File not found")),
    }
}

//              API

//              Message

#[derive(serde::Deserialize)]
struct MessageQuery {
    start_index: Option<usize>,
    limit: Option<usize>,
}

#[derive(serde::Serialize)]
struct MessagePage {
    messages: Vec<Message>,
    total_count: usize,
    has_more: bool,
}

#[get("/api/message")]
async fn message(query_params: web::Query<MessageQuery>) -> HttpResponse {
    let start_index: usize = query_params.start_index.unwrap_or(0);

    // 50 Messages is the max
    let limit: usize = query_params.limit.unwrap_or(15).min(50);

    // Get total message count for pagination metadata
    let total_count = match off_worker(
        "Error while getting message count",
        Database::get_total_message_count,
    )
    .await
    {
        Ok(count) => count,
        Err(response) => return response,
    };

    // Query to database, and return messages
    let messages: Vec<Message> =
        match off_worker("Error while getting messages from database", move || {
            Database::get_x_messages(limit, start_index)
        })
        .await
        {
            Ok(v) => v,
            Err(response) => return response,
        };

    let has_more = start_index + messages.len() < total_count;
    let message_page = MessagePage {
        messages,
        total_count,
        has_more,
    };

    let page_json = serde_json::to_string(&message_page)
        .unwrap_or(String::from("Error serializing message page as JSON"));
    HttpResponse::Ok().body(page_json)
}

#[post("/api/message")]
async fn message_post(received: web::Json<NewMessage>) -> HttpResponse {
    match Database::insert_message(received.into_inner()) {
        Ok(_) => HttpResponse::Ok().body("Message added!"),
        Err(e) => {
            println!("Failed to add message: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while adding message, check logs for more information")
        }
    }
}

#[delete("/api/message")]
async fn clear_messages() -> HttpResponse {
    match Database::erase_messages() {
        Ok(_) => HttpResponse::Ok().body("Chat log cleared!"),
        Err(e) => {
            println!("Failed to clear chat log: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while clearing chat log, check logs for more information")
        }
    }
}

#[get("/api/message/{id}")]
async fn message_id(id: web::Path<i32>) -> HttpResponse {
    let msg: Message = match Database::get_message(*id) {
        Ok(v) => v,
        Err(e) => {
            println!("Failed to get message at id {}: {}", id, e);
            return HttpResponse::InternalServerError().body(format!(
                "Error while getting message at id {}, check logs for more information",
                id
            ));
        }
    };
    let message_json =
        serde_json::to_string(&msg).unwrap_or(String::from("Error serializing message as JSON"));
    HttpResponse::Ok().body(message_json)
}

#[put("/api/message/{id}")]
async fn message_put(id: web::Path<i32>, received: web::Json<NewMessage>) -> HttpResponse {
    match Database::edit_message(*id, received.into_inner()) {
        Ok(_) => HttpResponse::Ok().body(format!("Message edited at id {}!", id)),
        Err(e) => {
            println!("Failed to edit message at id {}: {}", id, e);
            HttpResponse::InternalServerError().body(format!(
                "Error while editing message at id {}, check logs for more information",
                id
            ))
        }
    }
}

#[delete("/api/message/{id}")]
async fn message_delete(id: web::Path<i32>) -> HttpResponse {
    match Database::delete_message(*id) {
        Ok(_) => HttpResponse::Ok().body(format!("Message deleted at id {}!", id)),
        Err(e) => {
            println!("Failed to delete message at id {}: {}", id, e);
            HttpResponse::InternalServerError().body(format!(
                "Error while deleting message at id {}, check logs for more information",
                id
            ))
        }
    }
}

//              Companion

#[get("/api/companion")]
async fn companion() -> HttpResponse {
    let companion_data: CompanionView = match off_worker(
        "Error while getting companion data",
        Database::get_companion_data,
    )
    .await
    {
        Ok(v) => v,
        Err(response) => return response,
    };
    let companion_json: String = serde_json::to_string(&companion_data)
        .unwrap_or(String::from("Error serializing companion data as JSON"));
    HttpResponse::Ok().body(companion_json)
}

#[put("/api/companion")]
async fn companion_edit_data(received: web::Json<CompanionView>) -> HttpResponse {
    match Database::edit_companion(received.into_inner()) {
        Ok(_) => HttpResponse::Ok().body("Companion data edited!"),
        Err(e) => {
            println!("Failed to edit companion data: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while editing companion data, check logs for more information")
        }
    }
}

#[post("/api/companion/card")]
async fn companion_card(mut received: actix_web::web::Payload) -> HttpResponse {
    // curl -X POST -H "Content-Type: image/png" -T card.png http://localhost:3000/api/companion/card
    let mut data = web::BytesMut::new();
    while let Some(chunk) = received.next().await {
        let d = chunk.unwrap();
        data.extend_from_slice(&d);
    }
    let character_card: CharacterCard = match CharacterCard::load_character_card(&data) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error while loading character card from a file: {}", e);
            return HttpResponse::InternalServerError()
                .body("Error while importing character card, check logs for more information");
        }
    };
    let character_name = character_card.name.to_string();
    let mut avatar_file = match File::create("assets/avatar.png") {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "Error while creating 'avatar.png' file in a 'assets' folder: {}",
                e
            );
            return HttpResponse::InternalServerError()
                .body("Error while importing character card, check logs for more information");
        }
    };
    match avatar_file.write_all(&data) {
        Ok(_) => {}
        Err(e) => {
            eprintln!(
                "Error while writing bytes to 'avatar.png' file in a 'assets' folder: {}",
                e
            );
            return HttpResponse::InternalServerError()
                .body("Error while importing character card, check logs for more information");
        }
    };
    match Database::import_character_card(character_card, "assets/avatar.png") {
        Ok(_) => {}
        Err(e) => {
            eprintln!(
                "Error while changing companion avatar using character card: {}",
                e
            );
            return HttpResponse::InternalServerError()
                .body("Error while importing character card, check logs for more information");
        }
    };
    println!(
        "Character \"{}\" imported successfully! (from character card)",
        character_name
    );
    HttpResponse::Ok().body("Updated companion data via character card!")
}

#[post("/api/companion/characterJson")]
async fn companion_character_json(received: web::Json<CharacterCard>) -> HttpResponse {
    let character_name = received.name.to_string();
    match Database::import_character_json(received.into_inner()) {
        Ok(_) => {
            println!(
                "Character \"{}\" imported successfully! (from character JSON)",
                character_name
            );
            HttpResponse::Ok().body("Character json imported successfully!")
        }
        Err(e) => {
            println!("Failed to import character json: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while importing character json, check logs for more information")
        }
    }
}

#[get("/api/companion/characterJson")]
async fn get_companion_character_json() -> HttpResponse {
    match Database::get_companion_card_data() {
        Ok(v) => {
            let character_json: String = serde_json::to_string_pretty(&v as &CharacterCard)
                .unwrap_or(String::from("Error serializing companion data as JSON"));
            HttpResponse::Ok().body(character_json)
        }
        Err(e) => {
            println!("Failed to get companion card data: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while getting companion card data, check logs for more information")
        }
    }
}

#[post("/api/companion/avatar")]
async fn companion_avatar(mut received: actix_web::web::Payload) -> HttpResponse {
    // curl -X POST -H "Content-Type: image/png" -T avatar.png http://localhost:3000/api/companion/avatar
    let mut data = web::BytesMut::new();
    while let Some(chunk) = received.next().await {
        let d = chunk.unwrap();
        data.extend_from_slice(&d);
    }
    if fs::metadata("assets").is_err() {
        match fs::create_dir("assets") {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error while creating 'assets' directory: {}", e);
                return HttpResponse::InternalServerError()
                    .body("Error while importing character card, check logs for more information");
            }
        };
    }
    let mut avatar_file = match File::create("assets/avatar.png") {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "Error while creating 'avatar.png' file in a 'assets' folder: {}",
                e
            );
            return HttpResponse::InternalServerError()
                .body("Error while importing character card, check logs for more information");
        }
    };
    match avatar_file.write_all(&data) {
        Ok(_) => {}
        Err(e) => {
            eprintln!(
                "Error while writing bytes to 'avatar.png' file in a 'assets' folder: {}",
                e
            );
            return HttpResponse::InternalServerError()
                .body("Error while importing character card, check logs for more information");
        }
    };
    match Database::change_companion_avatar("assets/avatar.png") {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error while changing companion avatar: {}", e);
            return HttpResponse::InternalServerError()
                .body("Error while changing companion avatar, check logs for more information");
        }
    };
    HttpResponse::Ok().body("Companion avatar changed!")
}

//              User

#[get("/api/user")]
async fn user() -> HttpResponse {
    let user_data: UserView =
        match off_worker("Error while getting user data", Database::get_user_data).await {
            Ok(v) => v,
            Err(response) => return response,
        };
    let user_json: String = serde_json::to_string(&user_data)
        .unwrap_or(String::from("Error serializing user data as JSON"));
    HttpResponse::Ok().body(user_json)
}

#[put("/api/user")]
async fn user_put(received: web::Json<UserView>) -> HttpResponse {
    match Database::edit_user(received.into_inner()) {
        Ok(_) => HttpResponse::Ok().body("User data edited!"),
        Err(e) => {
            println!("Failed to edit user data: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while editing user data, check logs for more information")
        }
    }
}

//              Memory

#[derive(Deserialize)]
struct LongTermMemMessage {
    entry: String,
}

#[post("/api/memory/longTerm")]
async fn add_memory_long_term_message(received: web::Json<LongTermMemMessage>) -> HttpResponse {
    let entry = received.into_inner().entry;
    match off_worker("Error while adding long term memory entry", move || {
        LongTermMem::connect()?.add_entry(&entry)
    })
    .await
    {
        Ok(_) => HttpResponse::Ok().body("Long term memory entry added!"),
        Err(response) => response,
    }
}

#[delete("/api/memory/longTerm")]
async fn erase_long_term() -> HttpResponse {
    match off_worker("Error while clearing long term memory", || {
        LongTermMem::connect()?.erase_memory()
    })
    .await
    {
        Ok(_) => HttpResponse::Ok().body("Long term memory cleared!"),
        Err(response) => response,
    }
}

#[post("/api/memory/dialogueTuning")]
async fn add_tuning_message() -> HttpResponse {
    let messages = match Database::get_x_messages(2, 0) {
        Ok(v) => v,
        Err(e) => {
            println!("Failed to get last 2 messages from database: {}", e);
            return HttpResponse::InternalServerError().body("Error while getting last 2 messages from database, check logs for more information");
        }
    };
    match DialogueTuning::insert(&messages[0].content, &messages[1].content) {
        Ok(_) => HttpResponse::Ok().body("Saved previous dialogue as template dialogue"),
        Err(e) => {
            println!(
                "Failed to save previous dialogue as template dialogue: {}",
                e
            );
            HttpResponse::InternalServerError().body("Error while saving previous dialogue as template dialogue, check logs for more information")
        }
    }
}

#[delete("/api/memory/dialogueTuning")]
async fn erase_tuning_message() -> HttpResponse {
    match DialogueTuning::clear_dialogues() {
        Ok(_) => HttpResponse::Ok().body("Dialogue tuning memory cleared!"),
        Err(e) => {
            println!("Failed to clear dialogue tuning: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while clearing dialogue tuning, check logs for more information")
        }
    }
}

//              Prompting

#[derive(Deserialize)]
struct Prompt {
    prompt: String,
}

#[derive(Deserialize)]
struct StreamingRequest {
    prompt: String,
}

/// Pre-processing shared by `/api/prompt` and `/api/prompt/stream`: third-party
/// mention tracking, new-person detection and interaction detection.
///
/// Returns the prompt to generate from when an interaction with a recorded
/// outcome matched, so the caller can generate with that added context.
fn preprocess_user_message(user_message: &str, companion_id: i32) -> Option<String> {
    // Track third-party mentions and display console output
    match Database::track_third_party_mentions(user_message) {
        Ok(mention_output) => {
            if !mention_output.is_empty() {
                println!("{}", mention_output);
            }
        }
        Err(e) => eprintln!("Failed to track third-party mentions: {}", e),
    }

    // Automatically detect new persons in the message
    if let Err(e) = Database::detect_new_persons_in_message(user_message, companion_id) {
        eprintln!("Failed to detect persons in message: {}", e);
        // Continue processing even if person detection fails
    }

    // Detect and handle interaction requests
    if let Ok(Some(interaction)) = Database::detect_interaction_request(user_message, companion_id)
    {
        if let Some(outcome) = interaction.outcome.as_ref() {
            let third_party_name = Database::get_third_party_by_id(interaction.third_party_id)
                .ok()
                .flatten()
                .map(|p| p.name)
                .unwrap_or_else(|| "unknown".to_string());
            return Some(format!(
                "{}\n[Context: Interaction with {} - {}]",
                user_message, third_party_name, outcome
            ));
        }
    }

    None
}

/// Longest user-turn excerpt stored on an attitude memory.
const MEMORY_EXCERPT_CHARS: usize = 200;

/// Single-line excerpt of a user turn, for `attitude_memories.message_context`.
///
/// Truncates on a character boundary, so a multi-byte message can never split
/// mid-codepoint.
fn message_excerpt(user_message: &str) -> String {
    let single_line = user_message
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    match single_line.char_indices().nth(MEMORY_EXCERPT_CHARS) {
        Some((byte_index, _)) => format!("{}…", &single_line[..byte_index]),
        None => single_line,
    }
}

/// Derives attitude deltas from one conversation turn and persists them.
///
/// Called after generation, once both sides of the turn are known, from both
/// `/api/prompt` and `/api/prompt/stream`. An attitude failure must never fail
/// the chat reply, so every error path here is logged and returns `None`
/// rather than propagating.
///
/// On success, returns the (previous, current) attitude pair for the user
/// target so callers can report or persist the change (e.g. into the SSE
/// chunk, or as an attitude memory).
fn finish_turn(
    companion_id: i32,
    user_id: i32,
    user_message: &str,
    companion_reply: &str,
) -> Option<(CompanionAttitude, CompanionAttitude)> {
    // `get_attitude` propagates real SQL failures (e.g. a busy write lock) as
    // `Err` rather than collapsing them into `Ok(None)`, so `Ok(None)` here
    // reliably means the row is absent, never "the read failed".
    let current = match Database::get_attitude(companion_id, user_id, "user") {
        Ok(Some(attitude)) => attitude,
        Ok(None) => {
            // Fresh database: seed the row from the companion's persona before
            // scoring, otherwise the UPDATE below would silently touch zero rows.
            // `seed_missing_user_attitude` is insert-only (never falls back to
            // an UPDATE), so if this "row absent" read raced a concurrent
            // writer that has since inserted the real row, the seed silently
            // no-ops instead of wiping accumulated state.
            let persona = match Database::get_companion_data() {
                Ok(companion_data) => companion_data.persona,
                Err(e) => {
                    eprintln!("Failed to load companion persona for attitude seed: {}", e);
                    return None;
                }
            };
            if let Err(e) = Database::seed_missing_user_attitude(companion_id, user_id, &persona) {
                eprintln!("Failed to seed initial user attitude: {}", e);
                return None;
            }
            match Database::get_attitude(companion_id, user_id, "user") {
                Ok(Some(attitude)) => attitude,
                Ok(None) => {
                    eprintln!("Attitude row missing immediately after seeding");
                    return None;
                }
                Err(e) => {
                    eprintln!("Failed to reload seeded attitude: {}", e);
                    return None;
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to load attitude before scoring turn: {}", e);
            return None;
        }
    };

    let persona = match Database::get_companion_data() {
        Ok(companion_data) => companion_data.persona,
        Err(e) => {
            eprintln!(
                "Failed to load companion persona for attitude baseline: {}",
                e
            );
            return None;
        }
    };
    let baseline = Database::adjust_attitude_for_persona(
        &Database::default_user_attitude(companion_id, user_id),
        &persona,
    );

    let scorer = LexiconScorer::new(ScorerConfig::new(baseline));
    let deltas = scorer.evaluate_turn(user_message, companion_reply, &current);

    match Database::apply_attitude_deltas(companion_id, user_id, "user", &deltas) {
        Ok(Some((previous, updated))) => {
            let formatter = crate::attitude_formatter::AttitudeFormatter::new();
            let attitude_changes =
                formatter.format_attitude_changes_for_console(&previous, &updated);
            if !attitude_changes.is_empty() {
                println!("{}", attitude_changes);
            }
            // One memory per turn at most, carrying what the user said so the
            // companion remembers why its feelings moved.
            if let Err(e) = Database::detect_attitude_change(
                companion_id,
                user_id,
                "user",
                &previous,
                &updated,
                Some(&message_excerpt(user_message)),
            ) {
                eprintln!("Failed to record attitude memory: {}", e);
            }
            Some((previous, updated))
        }
        Ok(None) => {
            eprintln!("Attitude row missing when applying deltas after seeding");
            None
        }
        Err(e) => {
            eprintln!("Failed to apply attitude deltas: {}", e);
            None
        }
    }
}

/// Scores the turn and renders the result as the stream's attitude payload.
///
/// Returns `None` when the turn moved nothing, so the streaming worker only
/// spends an extra SSE event when there is something to report.
fn attitude_update_after_turn(
    companion_id: i32,
    user_id: i32,
    user_message: &str,
    companion_reply: &str,
) -> Option<AttitudeStreamUpdate> {
    let (previous, current) = finish_turn(companion_id, user_id, user_message, companion_reply)?;
    let formatter = crate::attitude_formatter::AttitudeFormatter::new();
    let deltas = formatter.diff_attitudes(&previous, &current);
    if deltas.is_empty() {
        return None;
    }
    Some(AttitudeStreamUpdate {
        summary: formatter.generate_natural_language_summary(&current),
        attitude: current,
        deltas,
    })
}

/// Why a non-streaming turn failed, carried out of the blocking closure so the
/// handler can build the response on the async side (`HttpResponse` is not
/// `Send`, so it cannot be built inside the closure itself).
enum TurnError {
    Database {
        step: &'static str,
        source: rusqlite::Error,
    },
    Generate(std::io::Error),
}

impl TurnError {
    fn into_response(self) -> HttpResponse {
        match self {
            TurnError::Database { step, source } => {
                eprintln!("{}: {}", step, source);
                HttpResponse::InternalServerError()
                    .body(format!("{}, check logs for more information", step))
            }
            TurnError::Generate(e) => {
                println!("Failed to generate prompt: {}", e);
                HttpResponse::InternalServerError()
                    .body("Error while generating prompt, check logs for more information")
            }
        }
    }
}

#[post("/api/prompt")]
async fn prompt_message(received: web::Json<Prompt>) -> HttpResponse {
    let prompt_message = received.into_inner().prompt;
    let start_time = std::time::Instant::now();

    let companion_id = match off_worker(
        "Error while getting companion data",
        Database::get_companion_id,
    )
    .await
    {
        Ok(id) => id,
        Err(response) => return response,
    };
    // Claimed before the user-turn insert below and moved into the blocking
    // closure below, which holds it for the whole turn: an overlapping call
    // cannot insert its own user message between this one and the reply it is
    // about to generate, and a client disconnect cannot cut generation short
    // since `spawn_blocking` tasks are not cancelled.
    let Some(turn_guard) = ACTIVE_TURN.try_claim() else {
        return HttpResponse::Conflict()
            .body("A reply is still being generated; wait for it to finish before sending another message");
    };
    let user_id = 1; // Default user ID

    let result = web::block(move || -> Result<String, TurnError> {
        let _turn_guard = turn_guard;

        let interaction_prompt = preprocess_user_message(&prompt_message, companion_id);

        // Estimate response time based on message complexity
        let estimate = estimate_response_time_enhanced(&prompt_message);
        println!(
            "⏱️ Response ETA: {}s (range: {}-{}s, confidence: {:.1}%)",
            estimate.expected_seconds,
            estimate.min_seconds,
            estimate.max_seconds,
            estimate.confidence * 100.0
        );
        if !estimate.factors.is_empty() {
            println!("   Factors: {}", estimate.factors.join(", "));
        }

        Database::insert_message(NewMessage {
            ai: false,
            content: prompt_message.clone(),
        })
        .map_err(|source| TurnError::Database {
            step: "Error while adding message to database",
            source,
        })?;

        // Generate with the interaction context when one matched, otherwise
        // from the raw user message; either way the turn is scored against
        // what the user actually said.
        let generation_prompt = interaction_prompt.as_deref().unwrap_or(&prompt_message);
        let reply = prompt(generation_prompt, companion_id).map_err(TurnError::Generate)?;
        finish_turn(companion_id, user_id, &prompt_message, &reply);

        // Display actual response time
        let elapsed = start_time.elapsed();
        println!("✓ Response completed in {:.1}s", elapsed.as_secs_f32());

        Ok(reply)
    })
    .await;

    match result {
        Ok(Ok(reply)) => HttpResponse::Ok().body(reply),
        Ok(Err(turn_error)) => turn_error.into_response(),
        Err(blocking) => {
            println!(
                "Failed to generate prompt: blocking task failed: {}",
                blocking
            );
            HttpResponse::InternalServerError()
                .body("Error while generating prompt, check logs for more information")
        }
    }
}

#[get("/api/prompt/regenerate")]
async fn regenerate_prompt() -> HttpResponse {
    // Resolved before the delete below: it is read-only, so a lookup failure
    // here must not leave the conversation with its last message destroyed
    // and no replacement generated.
    let companion_id = match off_worker(
        "Error while getting companion data",
        Database::get_companion_id,
    )
    .await
    {
        Ok(id) => id,
        Err(response) => return response,
    };
    // Claimed before the delete below and moved into the blocking closure,
    // which holds it for the whole turn: without it, a regenerate racing a
    // live stream could delete the user message a worker thread is about to
    // answer.
    let Some(turn_guard) = ACTIVE_TURN.try_claim() else {
        return HttpResponse::Conflict()
            .body("A reply is still being generated; wait for it to finish before sending another message");
    };

    let result = web::block(move || -> Result<String, TurnError> {
        let _turn_guard = turn_guard;

        Database::delete_latest_message().map_err(|source| TurnError::Database {
            step: "Error while deleting latest message",
            source,
        })?;
        let prompt_msg = Database::get_latest_message()
            .map_err(|source| TurnError::Database {
                step: "Error while getting latest message",
                source,
            })?
            .content;
        prompt(&prompt_msg, companion_id).map_err(TurnError::Generate)
    })
    .await;

    match result {
        Ok(Ok(reply)) => HttpResponse::Ok().body(reply),
        Ok(Err(turn_error)) => turn_error.into_response(),
        Err(blocking) => {
            println!(
                "Failed to generate prompt: blocking task failed: {}",
                blocking
            );
            HttpResponse::InternalServerError()
                .body("Error while generating prompt, check logs for more information")
        }
    }
}

//              Config

#[get("/api/config")]
async fn config() -> HttpResponse {
    let config = match off_worker("Error while getting config", Database::get_config).await {
        Ok(v) => v,
        Err(response) => return response,
    };
    let config_json =
        serde_json::to_string(&config).unwrap_or(String::from("Error serializing config as JSON"));
    HttpResponse::Ok().body(config_json)
}

// Note: no eager model reload here. `llm::generate` compares the next turn's
// `ModelKey` against the resident one and reloads only if it changed — do
// not "optimize" this into an eager reload, it would reload on every config
// save even when nothing model-relevant changed.
#[put("/api/config")]
async fn config_post(received: web::Json<ConfigModify>) -> HttpResponse {
    match Database::change_config(received.into_inner()) {
        Ok(_) => HttpResponse::Ok().body("Config updated!"),
        Err(e) => {
            println!("Failed to update config: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while updating config, check logs for more information")
        }
    }
}

//              LLM Model Management

#[get("/api/llm/models")]
async fn get_llm_models() -> HttpResponse {
    let scanner = LlmScanner::new();

    // Perform migration of existing config if needed
    if let Err(e) = scanner.migrate_existing_config() {
        println!("Warning: Failed to migrate existing config: {}", e);
    }

    match scanner.scan_for_models() {
        Ok(models) => {
            let models_json = serde_json::to_string(&models)
                .unwrap_or(String::from("Error serializing models as JSON"));
            HttpResponse::Ok().body(models_json)
        }
        Err(e) => {
            println!("Failed to scan for models: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while scanning for models, check logs for more information")
        }
    }
}

#[get("/api/llm/directories")]
async fn get_llm_directories() -> HttpResponse {
    let scanner = LlmScanner::new();
    match scanner.get_directories() {
        Ok(directories) => {
            let directories_json = serde_json::to_string(&directories)
                .unwrap_or(String::from("Error serializing directories as JSON"));
            HttpResponse::Ok().body(directories_json)
        }
        Err(e) => {
            println!("Failed to get directories: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while getting directories, check logs for more information")
        }
    }
}

#[derive(Deserialize)]
struct AddDirectoryRequest {
    path: String,
}

#[post("/api/llm/directories")]
async fn add_llm_directory(received: web::Json<AddDirectoryRequest>) -> HttpResponse {
    let scanner = LlmScanner::new();
    match scanner.add_directory(&received.path) {
        Ok(_) => HttpResponse::Ok().body("Directory added successfully"),
        Err(e) => {
            println!("Failed to add directory: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while adding directory, check logs for more information")
        }
    }
}

#[delete("/api/llm/directories/{id}")]
async fn remove_llm_directory(id: web::Path<i32>) -> HttpResponse {
    let scanner = LlmScanner::new();
    match scanner.remove_directory(*id) {
        Ok(_) => HttpResponse::Ok().body("Directory removed successfully"),
        Err(e) => {
            println!("Failed to remove directory: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while removing directory, check logs for more information")
        }
    }
}

/// Frees the model kept resident between turns (see `llm::unload_model`).
/// The next turn reloads it from disk.
#[post("/api/llm/unload")]
async fn unload_llm_model() -> HttpResponse {
    let (unloaded, model_path) = llm::unload_model();
    HttpResponse::Ok().json(serde_json::json!({ "unloaded": unloaded, "model_path": model_path }))
}

//              Attitude Tracking

/// Query for `GET /api/debug/prompt`.
#[derive(Deserialize)]
struct PromptInspectParams {
    companion_id: Option<i32>,
    /// Message the long-term memory recall is keyed on. Omitted, the block is
    /// assembled without any recalled entries.
    prompt: Option<String>,
}

/// Returns the prompt a turn would send, without loading a model.
///
/// The point is to make the attitude block inspectable: `attitude_context` in
/// the response is exactly the text `generate` folds into the system portion.
///
/// Because no model is loaded, `PromptTemplate::Auto` cannot be rendered
/// through the GGUF chat template here — for that template the response holds
/// the pre-template system text plus the role-tagged `chat_history`, not the
/// final rendered string. Every other template returns the finished prompt.
#[get("/api/debug/prompt")]
async fn inspect_prompt(query: web::Query<PromptInspectParams>) -> HttpResponse {
    let long_term_memory = match LongTermMem::connect() {
        Ok(ltm) => ltm,
        Err(e) => {
            println!("Failed to connect to long term memory: {}", e);
            return HttpResponse::InternalServerError().body(
                "Error while connecting to long term memory, check logs for more information",
            );
        }
    };

    // Named to avoid the `config` handler unit struct in this module.
    let config_view = match Database::get_config() {
        Ok(config_view) => config_view,
        Err(e) => {
            println!("Failed to get config: {}", e);
            return HttpResponse::InternalServerError()
                .body("Error while getting config, check logs for more information");
        }
    };

    let companion_id = match query.companion_id {
        Some(id) => id,
        None => match Database::get_companion_id() {
            Ok(id) => id,
            Err(e) => {
                println!("Failed to get companion id: {}", e);
                return HttpResponse::InternalServerError()
                    .body("Error while getting companion data, check logs for more information");
            }
        },
    };

    match assemble_prompt(
        query.prompt.as_deref().unwrap_or(""),
        companion_id,
        &long_term_memory,
        &config_view,
    ) {
        Ok(assembled) => HttpResponse::Ok().json(assembled),
        Err(e) => {
            println!("Failed to assemble prompt: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while assembling prompt, check logs for more information")
        }
    }
}

#[derive(Deserialize)]
struct AttitudeParams {
    companion_id: i32,
    target_id: i32,
    target_type: String,
}

#[get("/api/attitude")]
async fn get_attitude(query: web::Query<AttitudeParams>) -> HttpResponse {
    match Database::get_attitude(query.companion_id, query.target_id, &query.target_type) {
        Ok(Some(attitude)) => {
            let attitude_json = serde_json::to_string(&attitude)
                .unwrap_or(String::from("Error serializing attitude as JSON"));
            HttpResponse::Ok().body(attitude_json)
        }
        Ok(None) => HttpResponse::NotFound().body("Attitude not found"),
        Err(e) => {
            println!("Failed to get attitude: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while getting attitude, check logs for more information")
        }
    }
}

#[post("/api/attitude")]
async fn create_or_update_attitude(received: web::Json<CompanionAttitude>) -> HttpResponse {
    let attitude = received.into_inner();
    match Database::create_or_update_attitude(
        attitude.companion_id,
        attitude.target_id,
        &attitude.target_type,
        &attitude,
    ) {
        Ok(id) => HttpResponse::Ok().body(format!("Attitude created/updated with id: {}", id)),
        Err(e) => {
            println!("Failed to create/update attitude: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while creating/updating attitude, check logs for more information")
        }
    }
}

#[get("/api/attitude/companion/{companion_id}")]
async fn get_companion_attitudes(companion_id: web::Path<i32>) -> HttpResponse {
    let companion_id = *companion_id;
    match off_worker("Error while getting companion attitudes", move || {
        Database::get_all_companion_attitudes(companion_id)
    })
    .await
    {
        Ok(attitudes) => {
            let attitudes_json = serde_json::to_string(&attitudes)
                .unwrap_or(String::from("Error serializing attitudes as JSON"));
            HttpResponse::Ok().body(attitudes_json)
        }
        Err(response) => response,
    }
}

#[derive(serde::Serialize)]
struct AttitudeSummaryResponse {
    attitude: CompanionAttitude,
    summary: String,
}

#[get("/api/attitude/summary/{companion_id}/{user_id}")]
async fn get_attitude_summary(path: web::Path<(i32, i32)>) -> HttpResponse {
    let (companion_id, user_id) = path.into_inner();

    match Database::get_attitude(companion_id, user_id, "user") {
        Ok(Some(attitude)) => {
            let formatter = attitude_formatter::AttitudeFormatter::new();
            let summary = formatter.generate_natural_language_summary(&attitude);

            let response = AttitudeSummaryResponse { attitude, summary };

            match serde_json::to_string(&response) {
                Ok(json) => HttpResponse::Ok().body(json),
                Err(e) => {
                    println!("Failed to serialize attitude summary: {}", e);
                    HttpResponse::InternalServerError()
                        .body("Error while serializing attitude summary")
                }
            }
        }
        Ok(None) => HttpResponse::NotFound().body("Attitude not found"),
        Err(e) => {
            println!("Failed to get attitude for summary: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while getting attitude for summary, check logs for more information")
        }
    }
}

#[derive(Deserialize)]
struct AttitudeDimensionUpdate {
    companion_id: i32,
    target_id: i32,
    target_type: String,
    dimension: String,
    delta: f32,
}

#[put("/api/attitude/dimension")]
async fn update_attitude_dimension(received: web::Json<AttitudeDimensionUpdate>) -> HttpResponse {
    let update = received.into_inner();
    match Database::update_attitude_dimension(
        update.companion_id,
        update.target_id,
        &update.target_type,
        &update.dimension,
        update.delta,
    ) {
        Ok(_) => HttpResponse::Ok().body("Attitude dimension updated!"),
        Err(e) => {
            println!("Failed to update attitude dimension: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while updating attitude dimension, check logs for more information")
        }
    }
}

#[get("/api/attitude/memories/{companion_id}")]
async fn get_attitude_memories(companion_id: web::Path<i32>) -> HttpResponse {
    match Database::get_priority_attitude_memories(*companion_id, 20) {
        Ok(memories) => {
            let memories_json = serde_json::to_string(&memories)
                .unwrap_or(String::from("Error serializing attitude memories as JSON"));
            HttpResponse::Ok().body(memories_json)
        }
        Err(e) => {
            println!("Failed to get attitude memories: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while getting attitude memories, check logs for more information")
        }
    }
}

#[delete("/api/attitude/clear")]
async fn clear_attitudes() -> HttpResponse {
    let companion_id = 1;
    let user_id = 1;

    let companion_persona = match Database::get_companion_data() {
        Ok(companion_data) => companion_data.persona,
        Err(e) => {
            println!("Failed to get companion persona: {}", e);
            return HttpResponse::InternalServerError()
                .body("Error while getting companion data, check logs for more information");
        }
    };

    match Database::clear_companion_attitudes(companion_id) {
        Ok(_) => {
            match Database::create_initial_user_attitude(companion_id, user_id, &companion_persona)
            {
                Ok(_) => HttpResponse::Ok()
                    .body("Attitudes cleared and reset based on companion persona!"),
                Err(e) => {
                    println!("Failed to create initial attitude: {}", e);
                    HttpResponse::InternalServerError()
                        .body("Attitudes cleared but failed to create initial attitude, check logs for more information")
                }
            }
        }
        Err(e) => {
            println!("Failed to clear attitudes: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while clearing attitudes, check logs for more information")
        }
    }
}

#[post("/api/persons/detect")]
async fn detect_persons(received: web::Json<Prompt>) -> HttpResponse {
    let companion_id = 1; // Default companion ID - in a real system this would come from context

    match Database::detect_new_persons_in_message(&received.prompt, companion_id) {
        Ok(new_person_ids) => {
            let response = serde_json::json!({
                "detected_persons": new_person_ids,
                "message": format!("Detected {} new persons", new_person_ids.len())
            });
            HttpResponse::Ok().body(response.to_string())
        }
        Err(e) => {
            println!("Failed to detect persons: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while detecting persons, check logs for more information")
        }
    }
}

#[get("/api/persons")]
async fn get_all_persons() -> HttpResponse {
    match Database::get_all_third_party_individuals() {
        Ok(persons) => {
            let persons_json = serde_json::to_string(&persons)
                .unwrap_or(String::from("Error serializing persons as JSON"));
            HttpResponse::Ok().body(persons_json)
        }
        Err(e) => {
            println!("Failed to get all persons: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while getting persons, check logs for more information")
        }
    }
}

#[get("/api/persons/{name}")]
async fn get_person_by_name(name: web::Path<String>) -> HttpResponse {
    match Database::get_third_party_by_name(&name) {
        Ok(Some(person)) => {
            let person_json = serde_json::to_string(&person)
                .unwrap_or(String::from("Error serializing person as JSON"));
            HttpResponse::Ok().body(person_json)
        }
        Ok(None) => HttpResponse::NotFound().body("Person not found"),
        Err(e) => {
            println!("Failed to get person by name: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while getting person, check logs for more information")
        }
    }
}

#[post("/api/interactions/plan")]
async fn plan_interaction(received: web::Json<ThirdPartyInteraction>) -> HttpResponse {
    match Database::plan_third_party_interaction(&received.into_inner()) {
        Ok(interaction_id) => {
            let response = serde_json::json!({
                "success": true,
                "interaction_id": interaction_id,
                "message": "Interaction planned successfully"
            });
            HttpResponse::Ok().body(response.to_string())
        }
        Err(e) => {
            println!("Failed to plan interaction: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while planning interaction, check logs for more information")
        }
    }
}

#[get("/api/interactions/planned/{companion_id}")]
async fn get_planned_interactions(companion_id: web::Path<i32>) -> HttpResponse {
    match Database::get_planned_interactions(*companion_id, Some(10)) {
        Ok(interactions) => {
            let interactions_json = serde_json::to_string(&interactions)
                .unwrap_or(String::from("Error serializing interactions as JSON"));
            HttpResponse::Ok().body(interactions_json)
        }
        Err(e) => {
            println!("Failed to get planned interactions: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while getting planned interactions, check logs for more information")
        }
    }
}

#[post("/api/interactions/{interaction_id}/complete")]
async fn complete_interaction(interaction_id: web::Path<i32>) -> HttpResponse {
    match Database::generate_interaction_outcome(*interaction_id) {
        Ok(outcome) => {
            let response = serde_json::json!({
                "success": true,
                "outcome": outcome,
                "message": "Interaction completed successfully"
            });
            HttpResponse::Ok().body(response.to_string())
        }
        Err(e) => {
            println!("Failed to complete interaction: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while completing interaction, check logs for more information")
        }
    }
}

#[get("/api/interactions/history/{companion_id}/{third_party_id}")]
async fn get_interaction_history(params: web::Path<(i32, i32)>) -> HttpResponse {
    let (companion_id, third_party_id) = params.into_inner();
    match Database::get_interaction_history(companion_id, third_party_id) {
        Ok(history) => {
            let history_json = serde_json::to_string(&history)
                .unwrap_or(String::from("Error serializing history as JSON"));
            HttpResponse::Ok().body(history_json)
        }
        Err(e) => {
            println!("Failed to get interaction history: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while getting interaction history, check logs for more information")
        }
    }
}

#[derive(Deserialize)]
struct InteractionQuery {
    message: String,
    companion_id: i32,
}

#[post("/api/interactions/detect")]
async fn detect_interaction(received: web::Json<InteractionQuery>) -> HttpResponse {
    match Database::detect_interaction_request(&received.message, received.companion_id) {
        Ok(Some(interaction)) => {
            let interaction_json = serde_json::to_string(&interaction)
                .unwrap_or(String::from("Error serializing interaction as JSON"));
            HttpResponse::Ok().body(interaction_json)
        }
        Ok(None) => HttpResponse::Ok().body("{\"message\": \"No interaction detected\"}"),
        Err(e) => {
            println!("Failed to detect interaction: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while detecting interaction, check logs for more information")
        }
    }
}

#[post("/api/persons/cleanup-duplicates")]
async fn cleanup_duplicate_third_parties() -> HttpResponse {
    match Database::cleanup_duplicate_third_parties() {
        Ok(count) => {
            let response = serde_json::json!({
                "message": format!("Cleaned up {} duplicate third party entries", count),
                "cleaned_count": count
            });
            HttpResponse::Ok().body(response.to_string())
        }
        Err(e) => {
            println!("Failed to cleanup duplicate third parties: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while cleaning up duplicates, check logs for more information")
        }
    }
}

#[post("/api/persons/cleanup-invalid")]
async fn cleanup_invalid_third_parties() -> HttpResponse {
    match Database::cleanup_invalid_third_parties() {
        Ok(count) => {
            let response = serde_json::json!({
                "message": format!("Cleaned up {} invalid third party entries", count),
                "cleaned_count": count
            });
            HttpResponse::Ok().body(response.to_string())
        }
        Err(e) => {
            println!("Failed to cleanup invalid third parties: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while cleaning up invalid entries, check logs for more information")
        }
    }
}

#[derive(Deserialize)]
struct EstimateRequest {
    message: String,
}

#[post("/api/estimate-response-time")]
async fn estimate_response_time_endpoint(req: web::Json<EstimateRequest>) -> HttpResponse {
    let estimate = estimate_response_time_enhanced(&req.message);
    let response = serde_json::json!({
        "min_seconds": estimate.min_seconds,
        "expected_seconds": estimate.expected_seconds,
        "max_seconds": estimate.max_seconds,
        "confidence": estimate.confidence,
        "factors": estimate.factors
    });
    HttpResponse::Ok().json(response)
}

/// Streams a reply token by token as Server-Sent Events.
///
/// Each event carries a `StreamChunk` as JSON. The final event has
/// `is_complete: true` and no content.
#[post("/api/prompt/stream")]
async fn start_streaming_session(received: web::Json<StreamingRequest>) -> HttpResponse {
    let request = received.into_inner();
    let user_message = request.prompt.clone();
    // Generated server-side: a caller-supplied id could collide with a live
    // session and cross-wire the two streams.
    let session_id = format!("stream-{}", Uuid::new_v4());

    let companion_id = match off_worker(
        "Error while getting companion data",
        Database::get_companion_id,
    )
    .await
    {
        Ok(id) => id,
        Err(response) => return response,
    };
    // Claimed before the user-turn insert below: without it, a second
    // request's insert could land between this one and the worker thread
    // reading history, and the worker would answer both messages at once.
    // Moved into the spawned closure and dropped only after the reply (and
    // its attitude update) is persisted, so the slot covers the whole turn.
    let Some(turn_guard) = ACTIVE_TURN.try_claim() else {
        return HttpResponse::Conflict()
            .body("A reply is still being generated; wait for it to finish before sending another message");
    };

    let user_id = 1; // Default user ID

    // The generator reads recent messages back out of the database, so the
    // user's turn has to be persisted before generation starts. The turn
    // slot claimed above is what actually prevents another turn's insert
    // from landing in between; this insert alone is not enough. Runs off the
    // worker thread like the rest of the chat path.
    let pre_work_message = user_message.clone();
    let interaction_prompt = match off_worker("Error while adding message to database", move || {
        let interaction_prompt = preprocess_user_message(&pre_work_message, companion_id);
        Database::insert_message(NewMessage {
            ai: false,
            content: pre_work_message.clone(),
        })?;
        Ok::<_, rusqlite::Error>(interaction_prompt)
    })
    .await
    {
        Ok(v) => v,
        Err(response) => return response,
    };

    let rx = INFERENCE_OPTIMIZER.start_streaming_session(session_id.clone());

    // Generation is CPU-bound and blocking, so it runs on its own thread rather
    // than occupying an actix worker for the whole response.
    let worker_session = session_id.clone();
    // Cloned before the move into `generation_prompt` below: the interaction
    // context (if any) is what gets generated from, but the attitude engine
    // needs to score what the user actually said.
    let scored_message = user_message.clone();
    let generation_prompt = interaction_prompt.unwrap_or(user_message);
    std::thread::spawn(move || {
        // Held for the whole worker thread; dropped below only after the
        // reply and its attitude update are persisted.
        let _turn_guard = turn_guard;
        let mut token_count = 0usize;
        let result = prompt_streaming(&generation_prompt, companion_id, &mut |token| {
            token_count += 1;
            // A send failure means the client hung up; generation still runs to
            // completion so the reply is persisted.
            let _ = INFERENCE_OPTIMIZER.stream_chunk(
                &worker_session,
                StreamChunk {
                    request_id: worker_session.clone(),
                    content: token.to_string(),
                    is_complete: false,
                    token_count: Some(token_count),
                    error: None,
                    attitude: None,
                },
            );
        });

        let final_chunk = match result {
            // The streamed pieces include stop markers that are stripped before
            // the reply is persisted, so the final chunk carries the cleaned
            // text for the client to settle on.
            Ok(reply) => {
                // Persisted before the client sees `is_complete: true`, so the
                // row is already updated by the time the caller can react to it.
                if let Some(update) =
                    attitude_update_after_turn(companion_id, user_id, &scored_message, &reply)
                {
                    // Sent ahead of the final chunk so the client has the new
                    // attitude before it settles the reply bubble.
                    let _ = INFERENCE_OPTIMIZER.stream_chunk(
                        &worker_session,
                        StreamChunk {
                            request_id: worker_session.clone(),
                            content: String::new(),
                            is_complete: false,
                            token_count: Some(token_count),
                            error: None,
                            attitude: Some(update),
                        },
                    );
                }
                StreamChunk {
                    request_id: worker_session.clone(),
                    content: reply,
                    is_complete: true,
                    token_count: Some(token_count),
                    error: None,
                    attitude: None,
                }
            }
            Err(e) => {
                eprintln!("Failed to generate streamed prompt: {}", e);
                StreamChunk {
                    request_id: worker_session.clone(),
                    content: String::new(),
                    is_complete: true,
                    token_count: Some(token_count),
                    error: Some(e.to_string()),
                    attitude: None,
                }
            }
        };

        let _ = INFERENCE_OPTIMIZER.stream_chunk(&worker_session, final_chunk);
        INFERENCE_OPTIMIZER.end_streaming_session(&worker_session);
    });

    let event_stream = futures_util::stream::unfold(rx, |mut rx| async move {
        let chunk = rx.recv().await?;
        // JSON-encoding the chunk keeps newlines inside a token from being read
        // as SSE record separators.
        let payload = match serde_json::to_string(&chunk) {
            Ok(payload) => payload,
            Err(e) => {
                eprintln!("Failed to serialize stream chunk: {}", e);
                return None;
            }
        };
        let bytes = web::Bytes::from(format!("data: {}\n\n", payload));
        Some((Ok::<web::Bytes, actix_web::Error>(bytes), rx))
    });

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .append_header(("Cache-Control", "no-cache"))
        .append_header(("X-Accel-Buffering", "no"))
        .streaming(event_stream)
}

#[get("/api/inference/stats")]
async fn get_inference_stats() -> HttpResponse {
    let stats = INFERENCE_OPTIMIZER.get_stats();

    let response = serde_json::json!({
        "performance": {
            "total_requests": stats.total_requests,
            "avg_response_time_ms": stats.avg_response_time.as_millis(),
            "batch_processed": stats.batch_processed,
            "streaming_sessions": stats.streaming_sessions
        }
    });

    HttpResponse::Ok().json(response)
}

// Session Management Endpoints
#[derive(Deserialize)]
struct CreateSessionRequest {
    user_id: Option<i32>,
}

#[post("/api/session")]
async fn create_session(
    session_manager: web::Data<SessionManager>,
    req: web::Json<CreateSessionRequest>,
) -> HttpResponse {
    // Resolved server-side rather than trusted from the request: the client
    // cannot know the real companion id (CompanionView exposes no id), so a
    // client-supplied value would always be the hardcoded default.
    let companion_id = match Database::get_companion_id() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Failed to get companion id: {}", e);
            return HttpResponse::InternalServerError()
                .body("Error while getting companion data, check logs for more information");
        }
    };
    match session_manager.create_session(companion_id, req.user_id) {
        Ok(session) => {
            let response_json =
                serde_json::to_string(&session).unwrap_or_else(|_| "{}".to_string());
            HttpResponse::Ok().body(response_json)
        }
        Err(e) => {
            println!("Failed to create session: {}", e);
            HttpResponse::InternalServerError().body(format!("Error creating session: {}", e))
        }
    }
}

#[get("/api/session/{session_id}")]
async fn get_session(
    session_manager: web::Data<SessionManager>,
    session_id: web::Path<String>,
) -> HttpResponse {
    match session_manager.get_session(&session_id) {
        Ok(session) => {
            let response_json =
                serde_json::to_string(&session).unwrap_or_else(|_| "{}".to_string());
            HttpResponse::Ok().body(response_json)
        }
        Err(e) => HttpResponse::NotFound().body(format!("Session not found: {}", e)),
    }
}

#[derive(Deserialize)]
struct UpdateAttitudeRequest {
    session_id: String,
    attitude: CompanionAttitude,
}

#[put("/api/session/attitude")]
async fn update_session_attitude(
    session_manager: web::Data<SessionManager>,
    req: web::Json<UpdateAttitudeRequest>,
) -> HttpResponse {
    match session_manager.update_attitude(&req.session_id, req.attitude.clone()) {
        Ok(()) => HttpResponse::Ok().body("Attitude updated successfully"),
        Err(e) => {
            println!("Failed to update session attitude: {}", e);
            HttpResponse::InternalServerError().body(format!("Error updating attitude: {}", e))
        }
    }
}

#[post("/api/session/{session_id}/end")]
async fn end_session(
    session_manager: web::Data<SessionManager>,
    session_id: web::Path<String>,
) -> HttpResponse {
    match session_manager.end_session(&session_id) {
        Ok(()) => HttpResponse::Ok().body("Session ended successfully"),
        Err(e) if e.contains("not found") => {
            println!("Failed to end session: {}", e);
            HttpResponse::NotFound().body(format!("Error ending session: {}", e))
        }
        Err(e) => {
            println!("Failed to end session: {}", e);
            HttpResponse::InternalServerError().body(format!("Error ending session: {}", e))
        }
    }
}

#[get("/api/session/stats/summary")]
async fn get_session_stats(session_manager: web::Data<SessionManager>) -> HttpResponse {
    match session_manager.get_session_stats() {
        Ok(stats) => {
            let stats_json = serde_json::to_string(&stats).unwrap_or_else(|_| "{}".to_string());
            HttpResponse::Ok().body(stats_json)
        }
        Err(e) => {
            println!("Failed to get session stats: {}", e);
            HttpResponse::InternalServerError().body(format!("Error getting stats: {}", e))
        }
    }
}

#[get("/api/gpu/memory")]
async fn get_gpu_memory() -> HttpResponse {
    let config_data = match Database::get_config() {
        Ok(cfg) => cfg,
        Err(e) => {
            println!("Failed to get config: {}", e);
            return HttpResponse::InternalServerError().body("Failed to get configuration");
        }
    };

    let allocator = GpuAllocator::new()
        .with_safety_margin(config_data.gpu_safety_margin)
        .with_min_free_vram(config_data.min_free_vram_mb);

    match allocator.detect_gpu_memory(&config_data.device) {
        Ok(gpu_info) => match serde_json::to_string(&gpu_info) {
            Ok(json) => HttpResponse::Ok().body(json),
            Err(e) => {
                println!("Failed to serialize GPU memory info: {}", e);
                HttpResponse::InternalServerError().body("Failed to serialize GPU info")
            }
        },
        Err(e) => {
            println!("Failed to detect GPU memory: {}", e);
            HttpResponse::InternalServerError().body(format!("Failed to detect GPU memory: {}", e))
        }
    }
}

#[get("/api/gpu/allocation")]
async fn get_gpu_allocation() -> HttpResponse {
    let config_data = match Database::get_config() {
        Ok(cfg) => cfg,
        Err(e) => {
            println!("Failed to get config: {}", e);
            return HttpResponse::InternalServerError().body("Failed to get configuration");
        }
    };

    if !config_data.dynamic_gpu_allocation {
        let static_allocation = LayerAllocation {
            gpu_layers: config_data.gpu_layers,
            cpu_layers: 0, // We don't know total layers without loading the model
            total_layers: config_data.gpu_layers,
            estimated_vram_usage_mb: 0,
            allocation_strategy: crate::gpu_allocator::AllocationStrategy::MaxGpu,
        };
        match serde_json::to_string(&static_allocation) {
            Ok(json) => return HttpResponse::Ok().body(json),
            Err(e) => {
                println!("Failed to serialize allocation: {}", e);
                return HttpResponse::InternalServerError().body("Failed to serialize allocation");
            }
        }
    }

    let allocator = GpuAllocator::new()
        .with_safety_margin(config_data.gpu_safety_margin)
        .with_min_free_vram(config_data.min_free_vram_mb);

    match allocator.detect_gpu_memory(&config_data.device) {
        Ok(gpu_info) => {
            let vram_limit = if config_data.vram_limit_gb > 0 {
                Some(config_data.vram_limit_gb as f32)
            } else {
                None
            };

            // Estimate model size (this would ideally come from model metadata)
            let estimated_model_size_mb = 4096; // 4GB default estimate
            let estimated_total_layers = 32; // Default layer count

            let allocation = allocator.calculate_optimal_layers(
                &gpu_info,
                estimated_model_size_mb,
                estimated_total_layers,
                vram_limit,
            );

            match serde_json::to_string(&allocation) {
                Ok(json) => HttpResponse::Ok().body(json),
                Err(e) => {
                    println!("Failed to serialize allocation: {}", e);
                    HttpResponse::InternalServerError().body("Failed to serialize allocation")
                }
            }
        }
        Err(e) => {
            println!("Failed to detect GPU memory: {}", e);
            HttpResponse::InternalServerError().body(format!("Failed to detect GPU memory: {}", e))
        }
    }
}

//

/// Estimate response time based on message complexity
fn estimate_response_time_enhanced(msg: &str) -> ResponseEstimate {
    // Get current model configuration
    let db_config = match Database::get_config() {
        Ok(cfg) => cfg,
        Err(_) => {
            // Fallback to conservative estimate if config not available
            return ResponseEstimate {
                min_seconds: 15,
                expected_seconds: 30,
                max_seconds: 120,
                confidence: 0.3,
                factors: vec!["Configuration unavailable - using conservative estimate".to_string()],
            };
        }
    };

    let model_config = ModelConfig {
        model_path: db_config.llm_model_path,
        gpu_layers: db_config.gpu_layers as i32,
        device_type: db_config.device.to_string(),
    };

    // Use the performance tracker for accurate estimation
    let mut tracker = match INFERENCE_TRACKER.lock() {
        Ok(tracker) => tracker,
        Err(_) => {
            // Fallback if tracker is not available
            return ResponseEstimate {
                min_seconds: 10,
                expected_seconds: 25,
                max_seconds: 90,
                confidence: 0.4,
                factors: vec![
                    "Performance tracker unavailable - using fallback estimate".to_string()
                ],
            };
        }
    };

    tracker.estimate_response_time(msg, &model_config)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let port: u16 = 3000;
    let hostname: &str = "0.0.0.0";

    match Database::init() {
        Ok(_) => {}
        Err(e) => eprintln!("⚠️ Failed to connect to sqlite database: {}\n", e),
    }

    match LongTermMem::connect() {
        Ok(_) => {}
        Err(e) => eprintln!("⚠️ Failed to connect to tantivy: {}\n", e),
    }

    match DialogueTuning::create() {
        Ok(_) => {}
        Err(e) => eprintln!(
            "⚠️ Failed to create dialogue tuning table in sqlite database: {}\n",
            e
        ),
    }

    println!("AI Companion v1 successfully launched! 🚀\n");

    println!("Listening on:\n  -> http://{}:{}/", hostname, port);
    println!("  -> http://localhost:{}/\n", port);
    // Credit is retained per the MIT license; the upstream URL no longer
    // resolves, so it is not printed.
    println!("Originally by Hubert \"Hukasx0\" Kasperek\n");

    // Initialize session manager with 30 minute timeout
    let session_manager = web::Data::new(SessionManager::new(30));

    let mut server = HttpServer::new(move || {
        App::new()
            .app_data(session_manager.clone())
            .service(index)
            .service(js)
            .service(js2)
            .service(css)
            .service(project_logo)
            .service(companion_avatar_img)
            .service(companion_avatar_custom)
            .service(manifest)
            .service(service_worker)
            .service(message)
            .service(clear_messages)
            .service(message_id)
            .service(message_put)
            .service(message_delete)
            .service(message_post)
            .service(companion)
            .service(companion_edit_data)
            .service(companion_card)
            .service(companion_character_json)
            .service(get_companion_character_json)
            .service(companion_avatar)
            .service(user)
            .service(user_put)
            .service(add_memory_long_term_message)
            .service(erase_long_term)
            .service(add_tuning_message)
            .service(erase_tuning_message)
            .service(prompt_message)
            .service(regenerate_prompt)
            .service(config)
            .service(config_post)
            .service(get_llm_models)
            .service(get_llm_directories)
            .service(add_llm_directory)
            .service(remove_llm_directory)
            .service(unload_llm_model)
            .service(inspect_prompt)
            .service(get_attitude)
            .service(create_or_update_attitude)
            .service(get_companion_attitudes)
            .service(get_attitude_summary)
            .service(update_attitude_dimension)
            .service(get_attitude_memories)
            .service(clear_attitudes)
            .service(detect_persons)
            .service(get_all_persons)
            .service(get_person_by_name)
            .service(cleanup_duplicate_third_parties)
            .service(cleanup_invalid_third_parties)
            .service(estimate_response_time_endpoint)
            .service(plan_interaction)
            .service(get_planned_interactions)
            .service(complete_interaction)
            .service(get_interaction_history)
            .service(detect_interaction)
            .service(start_streaming_session)
            .service(get_inference_stats)
            .service(create_session)
            .service(get_session)
            .service(update_session_attitude)
            .service(end_session)
            .service(get_session_stats)
            .service(get_gpu_memory)
            .service(get_gpu_allocation)
    });
    if let Some(workers) = configured_workers() {
        server = server.workers(workers);
    }
    server.bind((hostname, port))?.run().await
}
