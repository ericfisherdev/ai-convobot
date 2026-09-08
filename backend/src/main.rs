use actix_web::{delete, get, post, put, web, App, HttpResponse, HttpServer};
use futures_util::StreamExt as _;
mod database;
use database::{
    CompanionAttitude, CompanionView, ConfigChangeError, ConfigModify, Database, Device, Message,
    MessageEdit, NewMessage, NewMessageRequest, PoppedReply, ThirdPartyInteraction, UserView,
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
mod model_metadata;
use crate::llm::{assemble_prompt, prompt, prompt_streaming, PromptSpeakers, SqliteTranscript};
use uuid::Uuid;
mod context_manager;
mod inference_optimizer;
use crate::inference_optimizer::{
    AttitudeStreamUpdate, StreamChunk, StreamSession, INFERENCE_OPTIMIZER,
};
mod session_manager;
mod token_budget;
use crate::session_manager::SessionManager;
mod attitude_engine;
mod attitude_formatter;
mod gpu_allocator;
use crate::gpu_allocator::{GpuAllocator, LayerAllocation};
use crate::model_metadata::ModelFacts;
mod system_memory;
// Removed unused system_memory imports
mod inference_performance;
use crate::inference_performance::{ModelConfig, ResponseEstimate, INFERENCE_TRACKER};
mod llm_scanner;
use crate::llm_scanner::LlmScanner;
mod turn_slot;
use crate::turn_slot::{TurnGuard, ACTIVE_TURN};
mod chat_turn;
use crate::chat_turn::{PendingTurn, PersistedReply, SqliteTurnStore, TurnStore};
mod multiplayer;
mod participants;
mod paths;
mod settings;
use crate::multiplayer::avatar as multiplayer_avatar;
use crate::multiplayer::config::MultiplayerMode;
use crate::multiplayer::host::{
    require_host_mode, HostConfigSource, HostSettings, SqliteHostConfig,
};
use crate::multiplayer::join_throttle::JoinThrottle;
use crate::multiplayer::joiner::{
    JoinerHandle, JoinerIdentity, JoinerShared, UnimplementedGeneration,
};
use crate::multiplayer::remote_bots::RemoteBots;
use crate::multiplayer::round::{
    plan_round, run_round, NoRemotes, NoopSink, RemoteGenerator, RoundPlan, RoundSink,
};
use crate::participants::{avatar_from, ParticipantId, ParticipantRegistry};
use std::sync::Arc;
#[cfg(test)]
pub(crate) mod simple_tests;

use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::RwLock;

/// Runs synchronous work (rusqlite, tantivy) on actix's blocking pool so the
/// worker thread stays free to serve other requests while it runs.
///
/// `failure` is the user-facing prefix already used by the handlers ("Error
/// while getting config"); it is logged with the cause and returned as the
/// 500 body with the usual ", check logs for more information" tail.
///
/// `Result<T, HttpResponse>` trips `clippy::result_large_err` on the
/// dependency versions this crate resolves to; every caller matches on
/// `Ok`/`Err(response)` and returns `response` as-is, so boxing it here
/// would just move an identical `Box::new`/deref pair into each of them.
#[allow(clippy::result_large_err)]
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

/// Formats a startup storage failure with the path it was trying to use, so
/// the process's stderr names the exact thing to fix instead of a bare
/// driver error.
fn storage_error(what: &str, path: &Path, e: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::other(format!(
        "cannot initialise {what} at {}: {e}",
        path.display()
    ))
}

/// Brings the on-disk state up to the current schema before the server
/// binds. Returns the first failure as an `io::Error` so `main` can
/// propagate it and exit non-zero instead of serving 500s from a
/// half-initialised process (a corrupt or unwritable `companion_database.db`,
/// or a `longterm_memory/` directory tantivy cannot open, previously logged
/// a warning and kept running).
///
/// Every path comes from `paths`, which `main()` has already pointed at
/// `COMPANION_DATA_DIR` (default: the working directory) via `paths::init`.
/// Also creates the `assets/` directory so the avatar handlers can write to
/// it without a fresh checkout hitting a missing-directory error on first
/// upload, and the `multiplayer/avatars/` directory so a joiner's avatar
/// upload never hits the same error on a host's first join.
fn init_storage() -> std::io::Result<()> {
    let db_path = paths::db_path();
    Database::init().map_err(|e| storage_error("sqlite database", &db_path, e))?;

    let index_path = paths::ltm_dir();
    LongTermMem::shared().map_err(|e| storage_error("tantivy index", &index_path, e))?;

    DialogueTuning::create().map_err(|e| storage_error("dialogue tuning table", &db_path, e))?;

    let assets_dir = paths::assets_dir();
    fs::create_dir_all(&assets_dir)
        .map_err(|e| storage_error("assets directory", &assets_dir, e))?;

    let participant_avatars_dir = paths::participant_avatars_dir();
    fs::create_dir_all(&participant_avatars_dir)
        .map_err(|e| storage_error("participant avatar directory", &participant_avatars_dir, e))?;

    Ok(())
}

/// Builds the shared participant registry from the persisted user/companion
/// rows. Called once at startup: a host without a usable `user`/`char` pair
/// is unusable, so a read failure here is fatal, the same as the rest of
/// `init_storage`.
fn seed_participants() -> std::io::Result<ParticipantRegistry> {
    let user_data = Database::get_user_data().map_err(|e| {
        std::io::Error::other(format!(
            "cannot seed participant registry: failed to load user data: {e}"
        ))
    })?;
    let companion_data = Database::get_companion_data().map_err(|e| {
        std::io::Error::other(format!(
            "cannot seed participant registry: failed to load companion data: {e}"
        ))
    })?;
    Ok(ParticipantRegistry::solo(
        &user_data.name,
        &companion_data.name,
        avatar_from(&companion_data.avatar_path),
    ))
}

/// Re-reads the two reserved participants (`user`, `char`) from the database
/// and applies them to the shared registry. Called by every handler that can
/// change a name or avatar, on the success path only.
///
/// A failure here is only logged, never turned into an HTTP error: `llm.rs`
/// re-reads `user`/`companion` from the database on every turn anyway, so a
/// stale shared registry only affects the participant list surfaced to the
/// frontend (#134), never the next generated reply.
fn refresh_reserved_participants(participants: &web::Data<RwLock<ParticipantRegistry>>) {
    let user_data = match Database::get_user_data() {
        Ok(user_data) => user_data,
        Err(e) => {
            eprintln!("Failed to refresh participant registry after edit: {}", e);
            return;
        }
    };
    let companion_data = match Database::get_companion_data() {
        Ok(companion_data) => companion_data,
        Err(e) => {
            eprintln!("Failed to refresh participant registry after edit: {}", e);
            return;
        }
    };
    let mut registry = participants
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Err(e) = registry.rename(&ParticipantId::USER, &user_data.name, None) {
        eprintln!("Failed to refresh user participant: {}", e);
    }
    if let Err(e) = registry.rename(
        &ParticipantId::CHAR,
        &companion_data.name,
        avatar_from(&companion_data.avatar_path),
    ) {
        eprintln!("Failed to refresh companion participant: {}", e);
    }
}

/// Snapshots the shared participant registry for one turn generated by the
/// host companion. Owned so it can move into `web::block` closures and the
/// `stream-generation` thread without holding the registry's lock across a
/// generation; a joiner (#130) builds its own `PromptSpeakers` from the
/// transcript it receives instead of calling this.
fn snapshot_speakers(registry: &web::Data<RwLock<ParticipantRegistry>>) -> PromptSpeakers {
    let registry = registry
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    PromptSpeakers {
        registry,
        self_id: ParticipantId::CHAR,
    }
}

/// The current chat's participant display names: passed to `TurnStore`/
/// person-detection so none of them is ever mistaken for a new third party.
fn participant_display_names(speakers: &PromptSpeakers) -> Vec<String> {
    speakers
        .registry
        .iter()
        .map(|p| p.display_name.clone())
        .collect()
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

/// The URL the frontend fetches the companion avatar from. This is stored in
/// the database by `Database::import_character_card`/`change_companion_avatar`
/// and served by `companion_avatar_custom` below; it is a URL path, not a
/// filesystem path, so it is never routed through `paths` the way the actual
/// on-disk file (`paths::avatar_path()`) is.
const AVATAR_URL_PATH: &str = "assets/avatar.png";

#[get("/assets/avatar.png")]
async fn companion_avatar_custom() -> actix_web::Result<actix_web::HttpResponse> {
    match File::open(paths::avatar_path()) {
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

/// Writes the companion avatar under `paths::assets_dir()`, creating the
/// directory if this is the first upload. Shared by `companion_card` and
/// `companion_avatar`, which previously duplicated this create-then-write
/// logic (and, in `companion_card`'s case, never created the directory at
/// all, so a fresh checkout's first character-card import failed outright).
fn write_companion_avatar(bytes: &[u8]) -> std::io::Result<()> {
    fs::create_dir_all(paths::assets_dir())?;
    fs::write(paths::avatar_path(), bytes)
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

/// Whether older messages remain past this page. Shared by the SQLite path
/// and the joiner-mirror path in [`message`] so the two can never drift
/// into reporting `has_more` differently for the same `(start_index,
/// page_len, total_count)` triple; [`RemoteTranscript::page`] applies the
/// same formula on its own data.
fn has_more_messages(start_index: usize, page_len: usize, total_count: usize) -> bool {
    start_index + page_len < total_count
}

#[get("/api/message")]
async fn message(
    query_params: web::Query<MessageQuery>,
    joiner: Option<web::Data<JoinerHandle>>,
) -> HttpResponse {
    let start_index: usize = query_params.start_index.unwrap_or(0);

    // 50 Messages is the max
    let limit: usize = query_params.limit.unwrap_or(15).min(50);

    // A joiner has no local `messages` table of its own to query: it
    // answers from the mirror `joiner::run` keeps in sync with the host
    // instead of ever touching SQLite here.
    let message_page = if let Some(joiner) = joiner {
        let (messages, total_count, has_more) = {
            let shared = joiner.read().unwrap_or_else(|p| p.into_inner());
            shared.transcript.page(start_index, limit)
        };
        MessagePage {
            messages,
            total_count,
            has_more,
        }
    } else {
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

        let has_more = has_more_messages(start_index, messages.len(), total_count);
        MessagePage {
            messages,
            total_count,
            has_more,
        }
    };

    let page_json = serde_json::to_string(&message_page)
        .unwrap_or(String::from("Error serializing message page as JSON"));
    HttpResponse::Ok().body(page_json)
}

#[post("/api/message")]
async fn message_post(
    received: web::Json<NewMessageRequest>,
    joiner: Option<web::Data<JoinerHandle>>,
) -> HttpResponse {
    if let Some(response) = reject_if_joiner(&joiner) {
        return response;
    }

    let new_message: NewMessage = match received.into_inner().try_into() {
        Ok(new_message) => new_message,
        Err(e) => return HttpResponse::BadRequest().body(e.to_string()),
    };
    match Database::insert_message(new_message) {
        Ok(_) => HttpResponse::Ok().body("Message added!"),
        Err(e) => {
            println!("Failed to add message: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while adding message, check logs for more information")
        }
    }
}

#[delete("/api/message")]
async fn clear_messages(joiner: Option<web::Data<JoinerHandle>>) -> HttpResponse {
    if let Some(response) = reject_if_joiner(&joiner) {
        return response;
    }

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
async fn message_put(
    id: web::Path<i32>,
    received: web::Json<MessageEdit>,
    joiner: Option<web::Data<JoinerHandle>>,
) -> HttpResponse {
    if let Some(response) = reject_if_joiner(&joiner) {
        return response;
    }

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
async fn message_delete(
    id: web::Path<i32>,
    joiner: Option<web::Data<JoinerHandle>>,
) -> HttpResponse {
    if let Some(response) = reject_if_joiner(&joiner) {
        return response;
    }

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
async fn companion_edit_data(
    received: web::Json<CompanionView>,
    participants: web::Data<RwLock<ParticipantRegistry>>,
) -> HttpResponse {
    match Database::edit_companion(received.into_inner()) {
        Ok(_) => {
            refresh_reserved_participants(&participants);
            HttpResponse::Ok().body("Companion data edited!")
        }
        Err(e) => {
            println!("Failed to edit companion data: {}", e);
            HttpResponse::InternalServerError()
                .body("Error while editing companion data, check logs for more information")
        }
    }
}

#[post("/api/companion/card")]
async fn companion_card(
    mut received: actix_web::web::Payload,
    participants: web::Data<RwLock<ParticipantRegistry>>,
) -> HttpResponse {
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
    if let Err(e) = write_companion_avatar(&data) {
        eprintln!(
            "Error while writing 'avatar.png' file in the 'assets' folder: {}",
            e
        );
        return HttpResponse::InternalServerError()
            .body("Error while importing character card, check logs for more information");
    }
    match Database::import_character_card(character_card, AVATAR_URL_PATH) {
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
    refresh_reserved_participants(&participants);
    println!(
        "Character \"{}\" imported successfully! (from character card)",
        character_name
    );
    HttpResponse::Ok().body("Updated companion data via character card!")
}

#[post("/api/companion/characterJson")]
async fn companion_character_json(
    received: web::Json<CharacterCard>,
    participants: web::Data<RwLock<ParticipantRegistry>>,
) -> HttpResponse {
    let character_name = received.name.to_string();
    match Database::import_character_json(received.into_inner()) {
        Ok(_) => {
            refresh_reserved_participants(&participants);
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
async fn companion_avatar(
    mut received: actix_web::web::Payload,
    participants: web::Data<RwLock<ParticipantRegistry>>,
) -> HttpResponse {
    // curl -X POST -H "Content-Type: image/png" -T avatar.png http://localhost:3000/api/companion/avatar
    let mut data = web::BytesMut::new();
    while let Some(chunk) = received.next().await {
        let d = chunk.unwrap();
        data.extend_from_slice(&d);
    }
    if let Err(e) = write_companion_avatar(&data) {
        eprintln!(
            "Error while writing 'avatar.png' file in the 'assets' folder: {}",
            e
        );
        return HttpResponse::InternalServerError()
            .body("Error while importing character card, check logs for more information");
    }
    match Database::change_companion_avatar(AVATAR_URL_PATH) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error while changing companion avatar: {}", e);
            return HttpResponse::InternalServerError()
                .body("Error while changing companion avatar, check logs for more information");
        }
    };
    refresh_reserved_participants(&participants);
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
async fn user_put(
    received: web::Json<UserView>,
    participants: web::Data<RwLock<ParticipantRegistry>>,
) -> HttpResponse {
    match Database::edit_user(received.into_inner()) {
        Ok(_) => {
            refresh_reserved_participants(&participants);
            HttpResponse::Ok().body("User data edited!")
        }
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
        LongTermMem::shared()?.add_entry(&entry)
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
        LongTermMem::shared()?.erase_memory()
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

/// Renders a completed turn's attitude change into the stream's attitude
/// chunk payload.
///
/// The turn itself is scored and persisted by `chat_turn::finish_turn`
/// (via `PendingTurn::complete`); this only shapes the result for the SSE
/// attitude chunk. Returns `None` when the turn moved nothing, so the
/// streaming worker only spends an extra SSE event when there is something
/// to report.
fn attitude_stream_update(
    previous: &CompanionAttitude,
    current: &CompanionAttitude,
) -> Option<AttitudeStreamUpdate> {
    let formatter = crate::attitude_formatter::AttitudeFormatter::new();
    let deltas = formatter.diff_attitudes(previous, current);
    if deltas.is_empty() {
        return None;
    }
    Some(AttitudeStreamUpdate {
        summary: formatter.generate_natural_language_summary(current),
        attitude: current.clone(),
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
    /// The newest message is not an AI reply with a preceding turn, so
    /// `regenerate_prompt` has nothing to pop.
    NothingToRegenerate,
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
            TurnError::NothingToRegenerate => HttpResponse::Conflict().body(
                "The newest message is not a companion reply, so there is nothing to regenerate",
            ),
        }
    }
}

#[cfg(test)]
mod turn_error_tests {
    use super::*;
    use actix_web::body::to_bytes;
    use actix_web::http::StatusCode;

    #[actix_web::test]
    async fn nothing_to_regenerate_maps_to_409_with_a_clear_body() {
        let response = TurnError::NothingToRegenerate.into_response();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = to_bytes(response.into_body()).await.unwrap();
        assert_eq!(
            body,
            "The newest message is not a companion reply, so there is nothing to regenerate"
        );
    }
}

#[cfg(test)]
mod stream_turn_tests {
    use super::*;
    use crate::chat_turn::RecordingStore;
    use crate::turn_slot::TurnSlot;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::mpsc::error::TryRecvError;

    fn solo_plan() -> RoundPlan {
        RoundPlan {
            speakers: vec![ParticipantId::CHAR],
        }
    }

    #[test]
    fn failed_generation_sends_one_user_turn_an_error_chunk_and_releases_the_slot() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = Arc::new(RecordingStore::new(None));
        let pending = PendingTurn::begin(&guard, store.as_ref(), 1, 1, "hello".to_string())
            .expect("insert should succeed");

        let session_id = format!("test-{}", Uuid::new_v4());
        let (stream, mut rx) = INFERENCE_OPTIMIZER.start_streaming_session(session_id);

        let thread_store = store.clone();
        let handle = std::thread::spawn(move || {
            stream_round(
                guard,
                pending,
                stream,
                thread_store.as_ref(),
                solo_plan(),
                |_prompt, _on_token| Err(std::io::Error::other("no model")),
                &NoRemotes,
                Duration::from_secs(30),
            );
        });
        handle.join().expect("worker thread should not panic");

        assert_eq!(store.inserted.lock().unwrap().len(), 1);
        assert!(store.finished.lock().unwrap().is_empty());

        let chunk = rx
            .try_recv()
            .expect("a failed generation should still send a terminal chunk");
        assert!(chunk.is_complete);
        assert!(chunk.error.is_some());
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Disconnected)));

        assert!(
            SLOT.try_claim().is_some(),
            "turn slot should be released once stream_round returns"
        );
    }

    #[test]
    fn successful_generation_forwards_tokens_then_the_cleaned_reply() {
        static SLOT: TurnSlot = TurnSlot::new();
        let guard = SLOT.try_claim().expect("slot should be free");
        let store = Arc::new(RecordingStore::new(None));
        let pending = PendingTurn::begin(&guard, store.as_ref(), 1, 1, "hello".to_string())
            .expect("insert should succeed");

        let session_id = format!("test-{}", Uuid::new_v4());
        let (stream, mut rx) = INFERENCE_OPTIMIZER.start_streaming_session(session_id);

        let thread_store = store.clone();
        let handle = std::thread::spawn(move || {
            stream_round(
                guard,
                pending,
                stream,
                thread_store.as_ref(),
                solo_plan(),
                |_prompt, on_token| {
                    on_token("Hel");
                    on_token("lo");
                    Ok("Hello".to_string())
                },
                &NoRemotes,
                Duration::from_secs(30),
            );
        });
        handle.join().expect("worker thread should not panic");

        let first = rx.try_recv().expect("first token chunk");
        assert_eq!(first.content, "Hel");
        assert_eq!(first.token_count, Some(1));
        assert!(!first.is_complete);

        let second = rx.try_recv().expect("second token chunk");
        assert_eq!(second.content, "lo");
        assert_eq!(second.token_count, Some(2));
        assert!(!second.is_complete);

        // No attitude chunk: `RecordingStore::finish_turn` always returns
        // `None`, so `stream_round` has nothing to report before the reply
        // settles.
        let final_chunk = rx.try_recv().expect("final chunk");
        assert!(final_chunk.is_complete);
        assert_eq!(final_chunk.content, "Hello");
        assert!(final_chunk.attitude.is_none());

        assert!(matches!(rx.try_recv(), Err(TryRecvError::Disconnected)));

        assert_eq!(
            *store.finished.lock().unwrap(),
            vec![("hello".to_string(), "Hello".to_string())]
        );
    }

    #[test]
    fn attitude_stream_update_is_none_when_nothing_moved() {
        let attitude = crate::simple_tests::tests::attitude_fixture();

        assert!(attitude_stream_update(&attitude, &attitude).is_none());
    }
}

/// Gates `/api/prompt`, `/api/prompt/regenerate` and `/api/prompt/stream`
/// on non-joiner mode: a joiner has no model of its own to answer with
/// (that is #153's job), so it must never claim [`ACTIVE_TURN`] or insert a
/// user turn `joiner::run`'s mirror does not own. `Some(response)` is a
/// ready-to-return `409` the three call sites return as-is; `None` means
/// "not a joiner, proceed".
fn reject_if_joiner(joiner: &Option<web::Data<JoinerHandle>>) -> Option<HttpResponse> {
    joiner.as_ref().map(|_| {
        HttpResponse::Conflict().body("this instance is a joiner; send messages from the host")
    })
}

#[post("/api/prompt")]
async fn prompt_message(
    received: web::Json<Prompt>,
    registry: web::Data<RwLock<ParticipantRegistry>>,
    joiner: Option<web::Data<JoinerHandle>>,
) -> HttpResponse {
    if let Some(response) = reject_if_joiner(&joiner) {
        return response;
    }

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
    // The round's remote timeout budget: read once here rather than inside
    // `run_round` (#131's round orchestrator, `multiplayer::round`, is
    // `Database`-free), so it stays testable against a fake `TurnStore`.
    let timeout = match off_worker("Error while getting config", Database::get_config).await {
        Ok(loaded_config) => {
            std::time::Duration::from_secs(loaded_config.remote_generation_timeout_secs)
        }
        Err(response) => return response,
    };
    // Claimed before the user-turn insert below and moved into the blocking
    // closure below, which holds it for the whole round: an overlapping call
    // cannot insert its own user message between this one and the replies it
    // is about to generate, and a client disconnect cannot cut generation
    // short since `spawn_blocking` tasks are not cancelled.
    let Some(turn_guard) = ACTIVE_TURN.try_claim() else {
        return HttpResponse::Conflict()
            .body("A reply is still being generated; wait for it to finish before sending another message");
    };
    let user_id = 1; // Default user ID

    let speakers = snapshot_speakers(&registry);
    let participant_names = participant_display_names(&speakers);
    let plan = plan_round(&prompt_message, &speakers.registry);

    let result = web::block(move || -> Result<Option<String>, TurnError> {
        let _turn_guard = turn_guard;
        let store = SqliteTurnStore::new(participant_names);

        let pending = PendingTurn::begin(
            &_turn_guard,
            &store,
            companion_id,
            user_id,
            prompt_message.clone(),
        )
        .map_err(|source| TurnError::Database {
            step: "Error while adding message to database",
            source,
        })?;

        // Estimate response time based on message complexity. Console-only,
        // so moving it after the insert (it used to run first) has no
        // observable effect on the response.
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

        let outcome = run_round(
            _turn_guard,
            pending,
            plan,
            &store,
            &mut |generation_prompt, _on_token| {
                prompt(
                    generation_prompt,
                    companion_id,
                    &SqliteTranscript,
                    &speakers,
                )
            },
            &NoRemotes,
            timeout,
            &mut NoopSink,
        )
        .map_err(TurnError::Generate)?;

        // Display actual response time
        let elapsed = start_time.elapsed();
        println!("✓ Response completed in {:.1}s", elapsed.as_secs_f32());

        Ok(outcome.host_reply.map(|reply| reply.text))
    })
    .await;

    match result {
        Ok(Ok(Some(reply))) => HttpResponse::Ok().body(reply),
        // Unreachable until #132 can filter the host bot out of a round's
        // plan; `plan_round` always puts `char` first today.
        Ok(Ok(None)) => HttpResponse::NoContent().finish(),
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
async fn regenerate_prompt(
    registry: web::Data<RwLock<ParticipantRegistry>>,
    joiner: Option<web::Data<JoinerHandle>>,
) -> HttpResponse {
    if let Some(response) = reject_if_joiner(&joiner) {
        return response;
    }

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

    let speakers = snapshot_speakers(&registry);
    let participant_names = participant_display_names(&speakers);

    let result = web::block(move || -> Result<String, TurnError> {
        let _turn_guard = turn_guard;
        let store = SqliteTurnStore::new(participant_names);

        let user_turn =
            match Database::pop_latest_ai_reply().map_err(|source| TurnError::Database {
                step: "Error while removing the latest reply",
                source,
            })? {
                PoppedReply::Removed { user_turn } => user_turn,
                PoppedReply::NothingToRegenerate => return Err(TurnError::NothingToRegenerate),
            };
        let reply = prompt(
            &user_turn.content,
            companion_id,
            &SqliteTranscript,
            &speakers,
        )
        .map_err(TurnError::Generate)?;
        // `llm::generate` no longer persists the reply itself (#131); this
        // handler is the one caller that relied on that implicit insert.
        store
            .insert_reply(&ParticipantId::CHAR, &reply)
            .map_err(|source| TurnError::Database {
                step: "Error while adding message to database",
                source,
            })?;
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
    // Read before the write so a role change can be reported: `main()`
    // builds the joiner's identity, and registers its routes' `app_data`,
    // once at startup (#130), so flipping `multiplayer_mode` here has no
    // effect until the process restarts. A read failure here is reported
    // as its own error rather than folded into `mode_changed`: silently
    // treating "could not read the previous mode" as "the mode changed"
    // would misreport a restart requirement on every save until the read
    // starts working again.
    // `config_view`, not `config`: a unit struct named `config` already
    // exists in this module for the `GET /api/config` handler.
    let previous_mode = match Database::get_config() {
        Ok(config_view) => config_view.multiplayer_mode.to_string(),
        Err(e) => {
            println!("Failed to read config before update: {}", e);
            return HttpResponse::InternalServerError()
                .body("Error while reading config, check logs for more information");
        }
    };
    let mode_changed = previous_mode != received.multiplayer_mode;

    match Database::change_config(received.into_inner()) {
        Ok(_) if mode_changed => HttpResponse::Ok()
            .body("Config updated! The multiplayer role change takes effect after a restart."),
        Ok(_) => HttpResponse::Ok().body("Config updated!"),
        Err(ConfigChangeError::Invalid(msg)) => HttpResponse::BadRequest().body(msg),
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
async fn inspect_prompt(
    query: web::Query<PromptInspectParams>,
    registry: web::Data<RwLock<ParticipantRegistry>>,
) -> HttpResponse {
    let long_term_memory = match LongTermMem::shared() {
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

    let speakers = snapshot_speakers(&registry);
    match assemble_prompt(
        query.prompt.as_deref().unwrap_or(""),
        companion_id,
        long_term_memory,
        &config_view,
        &SqliteTranscript,
        &speakers,
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
async fn detect_persons(
    received: web::Json<Prompt>,
    registry: web::Data<RwLock<ParticipantRegistry>>,
) -> HttpResponse {
    let companion_id = 1; // Default companion ID - in a real system this would come from context
    let participant_names = participant_display_names(&snapshot_speakers(&registry));

    match Database::detect_new_persons_in_message(
        &received.prompt,
        companion_id,
        &participant_names,
    ) {
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

/// The [`RoundSink`] that drives `/api/prompt/stream`'s SSE session.
///
/// Keeps today's chunk shape: `token` forwards only the host companion's
/// tokens (other speakers' tokens are dropped; their persisted replies reach
/// the UI through the `refreshMessages` call the frontend already makes
/// after the final chunk); `round_complete` sends the attitude chunk when
/// there is one, then the final `is_complete: true` chunk carrying the host
/// bot's reply text. #133 extends this with speaker-tagged chunks; nothing
/// else in the SSE path should need to move.
///
/// `stream` is `Option` so `round_complete` (which only gets `&mut self`)
/// can take it and call [`StreamSession::finish`], which needs to consume
/// it. If a round ends in `Err` before `round_complete` ever runs,
/// [`SseRoundSink::finish_with_error`] takes it instead.
struct SseRoundSink {
    stream: Option<StreamSession>,
    request_id: String,
    token_count: usize,
    host_reply_text: Option<String>,
}

impl SseRoundSink {
    fn new(stream: StreamSession) -> Self {
        let request_id = stream.id().to_string();
        SseRoundSink {
            stream: Some(stream),
            request_id,
            token_count: 0,
            host_reply_text: None,
        }
    }

    /// Ends the session with a terminal error chunk. Called by
    /// [`stream_round`] when `run_round` returns `Err` before
    /// `round_complete` ran, so `self.stream` is still held.
    fn finish_with_error(mut self, error_message: String) {
        if let Some(stream) = self.stream.take() {
            stream.finish(StreamChunk {
                request_id: self.request_id,
                content: String::new(),
                is_complete: true,
                token_count: Some(self.token_count),
                error: Some(error_message),
                attitude: None,
            });
        }
    }
}

impl RoundSink for SseRoundSink {
    fn reply_started(&mut self, _speaker: &ParticipantId) {}

    fn token(&mut self, speaker: &ParticipantId, text: &str) {
        if *speaker != ParticipantId::CHAR {
            return;
        }
        self.token_count += 1;
        if let Some(stream) = &self.stream {
            // A send failure means the client hung up; generation still runs
            // to completion so the reply is persisted.
            let _ = stream.send(StreamChunk {
                request_id: self.request_id.clone(),
                content: text.to_string(),
                is_complete: false,
                token_count: Some(self.token_count),
                error: None,
                attitude: None,
            });
        }
    }

    fn reply_complete(&mut self, reply: &PersistedReply) {
        if reply.speaker_id == ParticipantId::CHAR {
            self.host_reply_text = Some(reply.text.clone());
        }
    }

    fn speaker_skipped(&mut self, _speaker: &ParticipantId, _notice: &PersistedReply) {}

    fn round_complete(&mut self, attitude: Option<&(CompanionAttitude, CompanionAttitude)>) {
        let Some(stream) = self.stream.take() else {
            return;
        };
        // Persisted before the client sees `is_complete: true`, so the row
        // is already updated by the time the caller can react to it.
        if let Some((previous, current)) = attitude {
            if let Some(update) = attitude_stream_update(previous, current) {
                // Sent ahead of the final chunk so the client has the new
                // attitude before it settles the reply bubble.
                let _ = stream.send(StreamChunk {
                    request_id: self.request_id.clone(),
                    content: String::new(),
                    is_complete: false,
                    token_count: Some(self.token_count),
                    error: None,
                    attitude: Some(update),
                });
            }
        }
        stream.finish(StreamChunk {
            request_id: self.request_id.clone(),
            content: self.host_reply_text.clone().unwrap_or_default(),
            is_complete: true,
            token_count: Some(self.token_count),
            error: None,
            attitude: None,
        });
    }
}

/// Runs one streamed round on the calling thread and ends the SSE session on
/// every path: a thin wrapper over [`run_round`] that owns the
/// `StreamSession` through [`SseRoundSink`].
///
/// Holds `turn_guard` until after the terminal chunk has gone out (`run_round`
/// binds it first and drops it on return), meaning the turn slot reopens
/// only once the client has already seen the reply settle or fail. A failed
/// `host_generate` still ends the session (with an error chunk) and releases
/// the slot.
#[allow(clippy::too_many_arguments)] // see `run_round`'s identical allow
fn stream_round(
    turn_guard: TurnGuard,
    pending: PendingTurn,
    stream: StreamSession,
    store: &impl TurnStore,
    plan: RoundPlan,
    mut host_generate: impl FnMut(&str, &mut dyn FnMut(&str)) -> std::io::Result<String>,
    remotes: &dyn RemoteGenerator,
    timeout: std::time::Duration,
) {
    let mut sink = SseRoundSink::new(stream);
    let result = run_round(
        turn_guard,
        pending,
        plan,
        store,
        &mut host_generate,
        remotes,
        timeout,
        &mut sink,
    );
    if let Err(e) = result {
        eprintln!("Failed to generate streamed prompt: {}", e);
        sink.finish_with_error(e.to_string());
    }
}

/// Streams a reply token by token as Server-Sent Events.
///
/// Each event carries a `StreamChunk` as JSON. The final event has
/// `is_complete: true` and no content.
#[post("/api/prompt/stream")]
async fn start_streaming_session(
    received: web::Json<StreamingRequest>,
    registry: web::Data<RwLock<ParticipantRegistry>>,
    joiner: Option<web::Data<JoinerHandle>>,
) -> HttpResponse {
    if let Some(response) = reject_if_joiner(&joiner) {
        return response;
    }

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
    // See `prompt_message`'s identical fetch: the round's remote timeout
    // budget, read once here since `multiplayer::round` is `Database`-free.
    let timeout = match off_worker("Error while getting config", Database::get_config).await {
        Ok(loaded_config) => {
            std::time::Duration::from_secs(loaded_config.remote_generation_timeout_secs)
        }
        Err(response) => return response,
    };
    // Claimed before the user-turn insert below: without it, a second
    // request's insert could land between this one and the worker thread
    // reading history, and the worker would answer both messages at once.
    // Travels with the pending turn through the pre-work closure below, then
    // into the spawned closure, and is dropped only after the reply (and its
    // attitude update) is persisted, so the slot covers the whole round.
    let Some(turn_guard) = ACTIVE_TURN.try_claim() else {
        return HttpResponse::Conflict()
            .body("A reply is still being generated; wait for it to finish before sending another message");
    };

    let user_id = 1; // Default user ID

    let speakers = snapshot_speakers(&registry);
    let participant_names = participant_display_names(&speakers);
    let stream_participant_names = participant_names.clone();
    // With no registered joiners the plan is always `[char]`.
    let plan = plan_round(&user_message, &speakers.registry);

    // The generator reads recent messages back out of the database, so the
    // user's turn has to be persisted before generation starts. The turn
    // slot claimed above is what actually prevents another turn's insert
    // from landing in between; this insert alone is not enough. Runs off the
    // worker thread like the rest of the chat path. If `begin` fails the
    // guard drops right here inside the closure, releasing the slot exactly
    // as before generation ever starts.
    let (pending, turn_guard) =
        match off_worker("Error while adding message to database", move || {
            let store = SqliteTurnStore::new(participant_names);
            let pending =
                PendingTurn::begin(&turn_guard, &store, companion_id, user_id, user_message)?;
            Ok::<_, rusqlite::Error>((pending, turn_guard))
        })
        .await
        {
            Ok(v) => v,
            Err(response) => return response,
        };

    let (stream, rx) = INFERENCE_OPTIMIZER.start_streaming_session(session_id.clone());

    // Generation is CPU-bound and blocking, so it runs on its own thread rather
    // than occupying an actix worker for the whole response.
    let spawn_result = std::thread::Builder::new()
        .name("stream-generation".into())
        .spawn(move || {
            let store = SqliteTurnStore::new(stream_participant_names);
            stream_round(
                turn_guard,
                pending,
                stream,
                &store,
                plan,
                |generation_prompt, on_token| {
                    prompt_streaming(
                        generation_prompt,
                        companion_id,
                        on_token,
                        &SqliteTranscript,
                        &speakers,
                    )
                },
                &NoRemotes,
                timeout,
            );
        });
    // A failed spawn drops the closure immediately, which drops `stream` and
    // `turn_guard` right here: the session still ends with a terminal error
    // chunk (via `StreamSession`'s `Drop`) and the turn slot is still
    // released.
    if let Err(e) = spawn_result {
        eprintln!("Failed to spawn streaming generation thread: {}", e);
    }

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
    // Sweep expired sessions before reporting so an operator polling this
    // endpoint also bounds the map, the same way create_session does.
    if let Err(e) = session_manager.cleanup_expired_sessions() {
        println!("Failed to clean up expired sessions: {}", e);
    }
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

/// `GET /api/gpu/allocation` response body. Flattens `LayerAllocation` so
/// existing consumers keep seeing `gpu_layers`/`cpu_layers`/`total_layers`/
/// `estimated_vram_usage_mb`/`allocation_strategy` at the top level, with
/// the real model facts added under `model`.
#[derive(serde::Serialize)]
struct GpuAllocationReport {
    #[serde(flatten)]
    allocation: LayerAllocation,
    model: ModelFacts,
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

    // GGUF header reads are a small blocking file read; keep them off the
    // actix worker thread, same as `off_worker`, but handled directly here
    // so invalid metadata (a bad/foreign file) can be told apart from an
    // operational failure (unreadable file, panicking blocking task): the
    // former is a 422 the caller can act on, the latter is a 500.
    let model_path = config_data.llm_model_path.clone();
    let facts = match web::block(move || {
        model_metadata::read_model_facts(std::path::Path::new(&model_path))
    })
    .await
    {
        Ok(Ok(facts)) => facts,
        Ok(Err(
            e @ (model_metadata::ModelFactsError::NotGguf(_)
            | model_metadata::ModelFactsError::MissingKey(_)
            | model_metadata::ModelFactsError::UnexpectedType { .. }),
        )) => {
            println!("Invalid model metadata: {}", e);
            return HttpResponse::UnprocessableEntity()
                .body(format!("Invalid model metadata: {}", e));
        }
        Ok(Err(e)) => {
            println!("Failed to read model metadata: {}", e);
            return HttpResponse::InternalServerError()
                .body("Failed to read model metadata, check logs for more information");
        }
        Err(e) => {
            println!("Failed to read model metadata: blocking task failed: {}", e);
            return HttpResponse::InternalServerError()
                .body("Failed to read model metadata, check logs for more information");
        }
    };

    if !config_data.dynamic_gpu_allocation {
        let total_layers = facts.layer_count as usize;
        // A CPU-only device always loads zero GPU layers (see
        // `llm::resolve_gpu_layers`'s device check), regardless of the
        // configured `gpu_layers` value, so report that instead of
        // pretending a static GPU plan applies.
        let (gpu_layers, allocation_strategy) = if config_data.device == Device::CPU {
            (0, crate::gpu_allocator::AllocationStrategy::CpuFallback)
        } else {
            let gpu_layers = std::cmp::min(config_data.gpu_layers, total_layers);
            // gpu_layers is clamped above and can land below total_layers
            // (or at zero), so the reported strategy has to reflect that
            // instead of always claiming a full GPU offload.
            let allocation_strategy = if gpu_layers == 0 {
                crate::gpu_allocator::AllocationStrategy::CpuFallback
            } else if gpu_layers >= total_layers {
                crate::gpu_allocator::AllocationStrategy::MaxGpu
            } else {
                crate::gpu_allocator::AllocationStrategy::Balanced
            };
            (gpu_layers, allocation_strategy)
        };
        let cpu_layers = total_layers.saturating_sub(gpu_layers);
        let report = GpuAllocationReport {
            allocation: LayerAllocation {
                gpu_layers,
                cpu_layers,
                total_layers,
                estimated_vram_usage_mb: 0,
                allocation_strategy,
            },
            model: facts,
        };
        return match serde_json::to_string(&report) {
            Ok(json) => HttpResponse::Ok().body(json),
            Err(e) => {
                println!("Failed to serialize allocation: {}", e);
                HttpResponse::InternalServerError().body("Failed to serialize allocation")
            }
        };
    }

    let allocator = GpuAllocator::new()
        .with_safety_margin(config_data.gpu_safety_margin)
        .with_min_free_vram(config_data.min_free_vram_mb);

    match allocator.detect_gpu_memory(&config_data.device) {
        Ok(gpu_info) => {
            let vram_limit = GpuAllocator::vram_limit_from_config(config_data.vram_limit_gb);
            // Same `plan_for_model` entry point `load_model` uses, so this
            // report shows what generation will actually do.
            let allocation = allocator.plan_for_model(&gpu_info, &facts, vram_limit);
            let report = GpuAllocationReport {
                allocation,
                model: facts,
            };

            match serde_json::to_string(&report) {
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

//              Multiplayer

/// The current participant list, gated to `Host` mode like every other
/// `/api/multiplayer/*` route (`require_host_mode`, #129's
/// `host::host_password_or_404`).
#[get("/api/multiplayer/participants")]
async fn multiplayer_participants(
    host_config: web::Data<Arc<dyn HostConfigSource>>,
    participants: web::Data<RwLock<ParticipantRegistry>>,
    remote_bots: web::Data<RemoteBots>,
) -> HttpResponse {
    if let Err(response) = require_host_mode(host_config.get_ref()).await {
        return response;
    }
    let summaries: Vec<_> = {
        let registry = participants.read().unwrap_or_else(|p| p.into_inner());
        registry
            .iter()
            .map(|p| crate::multiplayer::host::participant_summary(p, &remote_bots))
            .collect()
    };
    HttpResponse::Ok().json(summaries)
}

/// Serves a transferred remote bot's avatar, the counterpart to
/// `companion_avatar_custom` for multiplayer participants. 404s for an
/// unknown id, a reserved (`user`/`char`) id, or a `RemoteBot` that has not
/// uploaded an avatar this run.
#[get("/api/multiplayer/participants/{id}/avatar")]
async fn multiplayer_participant_avatar(
    host_config: web::Data<Arc<dyn HostConfigSource>>,
    participants: web::Data<RwLock<ParticipantRegistry>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    if let Err(response) = require_host_mode(host_config.get_ref()).await {
        return Ok(response);
    }

    let id = match ParticipantId::parse(&path.into_inner()) {
        Ok(id) => id,
        Err(_) => {
            return Ok(HttpResponse::BadRequest()
                .json(serde_json::json!({ "error": "invalid participant id" })))
        }
    };

    let is_remote_bot = {
        let registry = participants.read().unwrap_or_else(|p| p.into_inner());
        matches!(
            registry.get(&id).map(|p| &p.kind),
            Some(crate::participants::ParticipantKind::RemoteBot)
        )
    };
    if !is_remote_bot {
        return Ok(
            HttpResponse::NotFound().json(serde_json::json!({ "error": "participant not found" }))
        );
    }

    let Some((format, file_path)) = multiplayer_avatar::find_stored_avatar(&id) else {
        return Ok(HttpResponse::NotFound()
            .json(serde_json::json!({ "error": "no avatar stored for this participant" })));
    };

    let id_for_log = id.clone();
    let bytes = match web::block(move || fs::read(&file_path)).await {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(e)) => {
            eprintln!(
                "multiplayer: failed to read avatar for {}: {}",
                id_for_log, e
            );
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "failed to read avatar, check logs for more information"
            })));
        }
        Err(e) => {
            eprintln!(
                "multiplayer: blocking task failed reading avatar for {}: {}",
                id_for_log, e
            );
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "failed to read avatar, check logs for more information"
            })));
        }
    };

    Ok(HttpResponse::Ok()
        .content_type(format.content_type())
        .body(bytes))
}

/// This instance's own multiplayer role and, in `joiner` mode, its
/// connection state. `#134` is the frontend consumer of this shape.
///
/// `solo`/`host` mode: `{ "mode": "...", "state": null }` — participants for
/// `host` mode live at `GET /api/multiplayer/participants` instead.
/// `joiner` mode: `JoinerState`'s own tagged JSON (`state`, plus `reason` or
/// `last_error` depending on which state it is) merged with `mode`,
/// `attempts`, `host_address`, `participant_id` and `participants`.
#[get("/api/multiplayer/status")]
async fn multiplayer_status(joiner: Option<web::Data<JoinerHandle>>) -> HttpResponse {
    let mode = match off_worker("Error while reading multiplayer config", || {
        Database::get_config().map(|c| c.multiplayer_mode)
    })
    .await
    {
        Ok(mode) => mode,
        Err(response) => return response,
    };

    let Some(joiner) = joiner else {
        return HttpResponse::Ok().json(serde_json::json!({ "mode": mode, "state": null }));
    };

    let shared = joiner.read().unwrap_or_else(|p| p.into_inner());
    let mut body = serde_json::to_value(&shared.state).unwrap_or_else(|_| serde_json::json!({}));
    if let serde_json::Value::Object(fields) = &mut body {
        fields.insert("mode".to_string(), serde_json::json!(mode));
        fields.insert("attempts".to_string(), serde_json::json!(shared.attempts));
        fields.insert(
            "host_address".to_string(),
            serde_json::json!(shared.host_address),
        );
        fields.insert(
            "participant_id".to_string(),
            serde_json::json!(shared.participant_id),
        );
        fields.insert(
            "participants".to_string(),
            serde_json::json!(shared.participants),
        );
    }
    HttpResponse::Ok().json(body)
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
    // `.to_string()` first (rather than `std::io::Error::other(e)` directly):
    // `main`'s `Result` return prints its `Err` with `Debug`, and boxing a
    // `SettingsError` there would render its derived struct-field `Debug`
    // instead of the message-shaped `Display` impl that names
    // `COMPANION_PORT`.
    let settings = settings::from_env().map_err(|e| std::io::Error::other(e.to_string()))?;

    // `join` with an already-absolute `settings.data_dir` returns that path
    // unchanged, so both a relative `COMPANION_DATA_DIR` (resolved against
    // the working directory) and an absolute one end up absolute here.
    let data_dir = std::env::current_dir()?.join(&settings.data_dir);
    fs::create_dir_all(&data_dir).map_err(|e| storage_error("data directory", &data_dir, e))?;
    paths::init(data_dir.clone())
        .map_err(|_| std::io::Error::other("paths::init called more than once"))?;

    init_storage()?;

    println!("AI Companion v1 successfully launched! 🚀\n");

    println!(
        "Listening on:\n  -> http://{}:{}/",
        settings.host, settings.port
    );
    println!("  -> http://localhost:{}/\n", settings.port);
    println!("Data directory: {}\n", data_dir.display());
    // Credit is retained per the MIT license; the upstream URL no longer
    // resolves, so it is not printed.
    println!("Originally by Hubert \"Hukasx0\" Kasperek\n");

    // Initialize session manager with 30 minute timeout
    let session_manager = web::Data::new(SessionManager::new(30));

    // Shared across every worker via `Data`'s `Arc`. Seeded once at startup;
    // a host without a usable `user`/`char` pair is unusable, so a failure
    // here fails startup the same way `init_storage` does.
    let participants = web::Data::new(RwLock::new(seed_participants()?));

    // Multiplayer host state (#129). Registered unconditionally, same as
    // `participants`: the mode is a runtime-toggleable `PUT /api/config`
    // field, not a startup-time constant, so every route gates itself per
    // request instead of the routes being registered conditionally.
    let host_config: web::Data<Arc<dyn HostConfigSource>> =
        web::Data::new(Arc::new(SqliteHostConfig) as Arc<dyn HostConfigSource>);
    let join_throttle = web::Data::new(JoinThrottle::new(5, std::time::Duration::from_secs(600)));
    let host_settings = web::Data::new(HostSettings::default());
    let remote_bots = web::Data::new(RemoteBots::new());

    // Multiplayer joiner state (#130). Built once here, before the server
    // starts accepting connections, from the config as it stood at
    // startup: `PUT /api/config` can flip the multiplayer fields at
    // runtime, but a joiner's identity (participant id, host address,
    // password) only takes effect on the next restart. `joiner` stays
    // `None` outside `Joiner` mode; every joiner-mode HTTP handler takes
    // `Option<web::Data<JoinerHandle>>` and treats `None` as "not a
    // joiner", the same seam `host_config`'s routes use for `Host` mode.
    let multiplayer_config = Database::get_config()
        .map_err(|e| std::io::Error::other(format!("cannot read multiplayer config: {e}")))?;
    let joiner: Option<web::Data<JoinerHandle>> =
        if multiplayer_config.multiplayer_mode == MultiplayerMode::Joiner {
            let companion_data = Database::get_companion_data().map_err(|e| {
                std::io::Error::other(format!(
                    "cannot read companion data for joiner identity: {e}"
                ))
            })?;
            let identity = JoinerIdentity::from_config(&multiplayer_config, &companion_data)
                .map_err(std::io::Error::other)?;
            let handle: JoinerHandle = Arc::new(RwLock::new(JoinerShared::new(&identity)));
            actix_web::rt::spawn(crate::multiplayer::joiner::run(
                handle.clone(),
                identity,
                Arc::new(UnimplementedGeneration),
            ));
            Some(web::Data::new(handle))
        } else {
            None
        };

    let mut server = HttpServer::new(move || {
        let mut app = App::new()
            .app_data(session_manager.clone())
            .app_data(participants.clone())
            .app_data(host_config.clone())
            .app_data(join_throttle.clone())
            .app_data(host_settings.clone())
            .app_data(remote_bots.clone());
        if let Some(joiner) = &joiner {
            app = app.app_data(joiner.clone());
        }
        app.service(index)
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
            .service(crate::multiplayer::host::multiplayer_ws)
            .service(multiplayer_participants)
            .service(multiplayer_participant_avatar)
            .service(multiplayer_status)
    });
    if let Some(workers) = configured_workers() {
        server = server.workers(workers);
    }
    server
        .bind((settings.host.as_str(), settings.port))?
        .run()
        .await
}
