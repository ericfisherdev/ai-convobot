use actix_web::{get, post, delete, put, App, web, HttpResponse, HttpServer};
use futures_util::StreamExt as _;
mod database;
use database::{Database, Message, NewMessage, CompanionView, UserView, ConfigModify, CompanionAttitude, ThirdPartyIndividual, ThirdPartyInteraction};
mod long_term_mem;
use long_term_mem::LongTermMem;
mod dialogue_tuning;
use dialogue_tuning::DialogueTuning;
mod character_card;
use character_card::CharacterCard;
use serde::Deserialize;
mod llm;
use crate::llm::prompt;
mod context_manager;
use crate::context_manager::{ContextManager, OptimizedContext};
mod inference_optimizer;
use crate::inference_optimizer::{INFERENCE_OPTIMIZER, StreamChunk};
mod token_budget;
use crate::token_budget::{TokenBudget, TokenUsageMonitor};
mod session_manager;
use crate::session_manager::{SessionManager, Session};
mod attitude_formatter;
use crate::attitude_formatter::AttitudeFormatter;
#[cfg(test)]
mod simple_tests;

use std::fs;
use std::fs::File;
use std::io::{Write, Read};

#[get("/")]
async fn index() -> HttpResponse {
    HttpResponse::Ok().body(include_str!("../../dist/index.html"))
}

#[get("/assets/index-4rust.js")]
async fn js() -> HttpResponse {
    HttpResponse::Ok().content_type("application/javascript").body(include_str!("../../dist/assets/index-4rust.js"))
}

#[get("/assets/index-4rust2.js")]
async fn js2() -> HttpResponse {
    HttpResponse::Ok().content_type("application/javascript").body(include_str!("../../dist/assets/index-4rust2.js"))
}

#[get("/assets/index-4rust.css")]
async fn css() -> HttpResponse {
    HttpResponse::Ok().content_type("text/css").body(include_str!("../../dist/assets/index-4rust.css"))
}

#[get("/ai_companion_logo.jpg")]
async fn project_logo() -> HttpResponse {
    HttpResponse::Ok().content_type("image/jpeg").body(&include_bytes!("../../dist/ai_companion_logo.jpg")[..])
}

#[get("/assets/companion_avatar-4rust.jpg")]
async fn companion_avatar_img() -> HttpResponse {
    HttpResponse::Ok().content_type("image/jpeg").body(&include_bytes!("../../dist/assets/companion_avatar-4rust.jpg")[..])
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
    let total_count = match Database::get_total_message_count() {
        Ok(count) => count,
        Err(e) => {
            println!("Failed to get total message count: {}", e);
            return HttpResponse::InternalServerError().body("Error while getting message count, check logs for more information");
        }
    };

    // Query to database, and return messages
    let messages: Vec<Message> = match Database::get_x_messages(limit, start_index) {
        Ok(v) => v,
        Err(e) => {
            println!("Failed to get messages from database: {}", e);
            return HttpResponse::InternalServerError().body("Error while getting messages from database, check logs for more information");
        },
    };
    
    let has_more = start_index + messages.len() < total_count;
    let message_page = MessagePage {
        messages,
        total_count,
        has_more,
    };
    
    let page_json = serde_json::to_string(&message_page).unwrap_or(String::from("Error serializing message page as JSON"));
    HttpResponse::Ok().body(page_json)
}

#[post("/api/message")]
async fn message_post(received: web::Json<NewMessage>) -> HttpResponse {
    match Database::insert_message(received.into_inner()) {
        Ok(_) => HttpResponse::Ok().body("Message added!"),
        Err(e) => {
            println!("Failed to add message: {}", e);
            HttpResponse::InternalServerError().body("Error while adding message, check logs for more information")
        }
    }
}

#[delete("/api/message")]
async fn clear_messages() -> HttpResponse {
    match Database::erase_messages() {
        Ok(_) => HttpResponse::Ok().body("Chat log cleared!"),
        Err(e) => {
            println!("Failed to clear chat log: {}", e);
            HttpResponse::InternalServerError().body("Error while clearing chat log, check logs for more information")
        }
    }
}

#[get("/api/message/{id}")]
async fn message_id(id: web::Path<i32>) -> HttpResponse {
    let msg: Message = match Database::get_message(*id) {
        Ok(v) => v,
        Err(e) => {
            println!("Failed to get message at id {}: {}", id, e);
            return HttpResponse::InternalServerError().body(format!("Error while getting message at id {}, check logs for more information", id));
        }
    };
    let message_json = serde_json::to_string(&msg).unwrap_or(String::from("Error serializing message as JSON"));
    HttpResponse::Ok().body(message_json)
}

#[put("/api/message/{id}")]
async fn message_put(id: web::Path<i32>, received: web::Json<NewMessage>) -> HttpResponse {
    match Database::edit_message(*id, received.into_inner()) {
        Ok(_) => HttpResponse::Ok().body(format!("Message edited at id {}!", id)),
        Err(e) => {
            println!("Failed to edit message at id {}: {}", id, e);
            HttpResponse::InternalServerError().body(format!("Error while editing message at id {}, check logs for more information", id))
        }
    }
}

#[delete("/api/message/{id}")]
async fn message_delete(id: web::Path<i32>) -> HttpResponse {
    match Database::delete_message(*id) {
        Ok(_) => HttpResponse::Ok().body(format!("Message deleted at id {}!", id)),
        Err(e) => {
            println!("Failed to delete message at id {}: {}", id, e);
            HttpResponse::InternalServerError().body(format!("Error while deleting message at id {}, check logs for more information", id))
        }
    }
}

//              Companion

#[get("/api/companion")]
async fn companion() -> HttpResponse {
    let companion_data: CompanionView = match Database::get_companion_data() {
        Ok(v) => v,
        Err(e) => {
            println!("Failed to get companion data: {}", e);
            return HttpResponse::InternalServerError().body("Error while getting companion data, check logs for more information");
        }
    };
    let companion_json: String = serde_json::to_string(&companion_data).unwrap_or(String::from("Error serializing companion data as JSON"));
    HttpResponse::Ok().body(companion_json)
}

#[put("/api/companion")]
async fn companion_edit_data(received: web::Json<CompanionView>) -> HttpResponse {
    match Database::edit_companion(received.into_inner()) {
        Ok(_) => HttpResponse::Ok().body("Companion data edited!"),
        Err(e) => {
            println!("Failed to edit companion data: {}", e);
            HttpResponse::InternalServerError().body("Error while editing companion data, check logs for more information")
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
            return HttpResponse::InternalServerError().body("Error while importing character card, check logs for more information");
        }
    };
    let character_name = character_card.name.to_string();
    let mut avatar_file = match File::create("assets/avatar.png") {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error while creating 'avatar.png' file in a 'assets' folder: {}", e);
            return HttpResponse::InternalServerError().body("Error while importing character card, check logs for more information");
        }
    };
    match avatar_file.write_all(&data) {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Error while writing bytes to 'avatar.png' file in a 'assets' folder: {}", e);
            return HttpResponse::InternalServerError().body("Error while importing character card, check logs for more information");
        }
    };
    match Database::import_character_card(character_card, "assets/avatar.png") {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Error while changing companion avatar using character card: {}", e);
            return HttpResponse::InternalServerError().body("Error while importing character card, check logs for more information");
        }
    };
    println!("Character \"{}\" imported successfully! (from character card)", character_name);
    HttpResponse::Ok().body("Updated companion data via character card!")

}

#[post("/api/companion/characterJson")]
async fn companion_character_json(received: web::Json<CharacterCard>) -> HttpResponse {
    let character_name = received.name.to_string();
    match Database::import_character_json(received.into_inner()) {
        Ok(_) => {
            println!("Character \"{}\" imported successfully! (from character JSON)", character_name);
            HttpResponse::Ok().body("Character json imported successfully!") 
        },
        Err(e) => {
            println!("Failed to import character json: {}", e);
            HttpResponse::InternalServerError().body("Error while importing character json, check logs for more information")
        }
    }
}

#[get("/api/companion/characterJson")]
async fn get_companion_character_json() -> HttpResponse {
    match Database::get_companion_card_data() {
        Ok(v) => { 
            let character_json: String = serde_json::to_string_pretty(&v as &CharacterCard).unwrap_or(String::from("Error serializing companion data as JSON"));
            return HttpResponse::Ok().body(character_json);
        },
        Err(e) => {
            println!("Failed to get companion card data: {}", e);
            return HttpResponse::InternalServerError().body("Error while getting companion card data, check logs for more information");
        },
    };
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
            Ok(_) => {},
            Err(e) => {
                eprintln!("Error while creating 'assets' directory: {}", e);
                return HttpResponse::InternalServerError().body("Error while importing character card, check logs for more information");
            }
        };
    }
    let mut avatar_file = match File::create("assets/avatar.png") {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error while creating 'avatar.png' file in a 'assets' folder: {}", e);
            return HttpResponse::InternalServerError().body("Error while importing character card, check logs for more information");
        }
    };
    match avatar_file.write_all(&data) {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Error while writing bytes to 'avatar.png' file in a 'assets' folder: {}", e);
            return HttpResponse::InternalServerError().body("Error while importing character card, check logs for more information");
        }
    };
    match Database::change_companion_avatar("assets/avatar.png") {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Error while changing companion avatar: {}", e);
            return HttpResponse::InternalServerError().body("Error while changing companion avatar, check logs for more information");
        }
    };
    HttpResponse::Ok().body("Companion avatar changed!")
}

//              User

#[get("/api/user")]
async fn user() -> HttpResponse {
    let user_data: UserView = match Database::get_user_data() {
        Ok(v) => v,
        Err(e) => {
            println!("Failed to get user data: {}", e);
            return HttpResponse::InternalServerError().finish();
        }
    };
    let user_json: String = serde_json::to_string(&user_data).unwrap_or(String::from("Error serializing user data as JSON"));
    HttpResponse::Ok().body(user_json)
}

#[put("/api/user")]
async fn user_put(received: web::Json<UserView>) -> HttpResponse {
    match Database::edit_user(received.into_inner()) {
        Ok(_) => HttpResponse::Ok().body("User data edited!"),
        Err(e) => {
            println!("Failed to edit user data: {}", e);
            HttpResponse::InternalServerError().body("Error while editing user data, check logs for more information")
        }
    }
}



//              Memory

#[derive(Deserialize)]
struct LongTermMemMessage {
    entry: String
}

#[post("/api/memory/longTerm")]
async fn add_memory_long_term_message(received: web::Json<LongTermMemMessage>) -> HttpResponse {
    let ltm = match LongTermMem::connect() {
        Ok(v) => v,
        Err(e) => {
            println!("Failed to connect to long term memory: {}", e);
            return HttpResponse::InternalServerError().body("Error while connecting to long term memory, check logs for more information");
        }
    };
    match ltm.add_entry(&received.into_inner().entry) {
        Ok(_) => HttpResponse::Ok().body("Long term memory entry added!"),
        Err(e) => {
            println!("Failed to add long term memory entry: {}", e);
            HttpResponse::InternalServerError().body("Error while adding long term memory entry, check logs for more information")
        }
    }
}

#[delete("/api/memory/longTerm")]
async fn erase_long_term() -> HttpResponse {
    let ltm = match LongTermMem::connect() {
        Ok(v) => v,
        Err(e) => {
            println!("Failed to connect to long term memory: {}", e);
            return HttpResponse::InternalServerError().body("Error while connecting to long term memory, check logs for more information");
        }
    };
    match ltm.erase_memory() {
        Ok(_) => HttpResponse::Ok().body("Long term memory cleared!"),
        Err(e) => {
            println!("Failed to clear long term memory: {}", e);
            HttpResponse::InternalServerError().body("Error while clearing long term memory, check logs for more information")
        }
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
            println!("Failed to save previous dialogue as template dialogue: {}", e);
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
            HttpResponse::InternalServerError().body("Error while clearing dialogue tuning, check logs for more information")
        }
    }
}


//              Prompting

#[derive(Deserialize)]
struct Prompt {
    prompt: String
}

#[derive(Deserialize)]
struct StreamingRequest {
    prompt: String,
    session_id: String,
}

#[post("/api/prompt")]
async fn prompt_message(received: web::Json<Prompt>) -> HttpResponse {
    let prompt_message = received.into_inner().prompt.clone();
    
    // Automatically detect new persons in the message
    let companion_id = 1; // Default companion ID
    if let Err(e) = Database::detect_new_persons_in_message(&prompt_message, companion_id) {
        eprintln!("Failed to detect persons in message: {}", e);
        // Continue processing even if person detection fails
    }
    
    // Detect and handle interaction requests
    if let Ok(Some(interaction)) = Database::detect_interaction_request(&prompt_message, companion_id) {
        // Store interaction context for LLM to use
        if interaction.outcome.is_some() {
            // If interaction has outcome, include it in the context
            let enhanced_prompt = format!("{}\n[Context: Interaction with {} - {}]", 
                prompt_message, 
                Database::get_third_party_by_id(interaction.third_party_id)
                    .ok()
                    .flatten()
                    .map(|p| p.name)
                    .unwrap_or_else(|| "unknown".to_string()),
                interaction.outcome.as_ref().unwrap_or(&"".to_string())
            );
            
            match Database::insert_message(NewMessage { ai: false, content: prompt_message.to_string() }) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Failed to add message to database: {}", e);
                    return HttpResponse::InternalServerError().body("Error while adding message to database, check logs for more information");
                }
            };
            
            // Generate response with interaction context
            match prompt(&enhanced_prompt) {
                Ok(v) => return HttpResponse::Ok().body(v),
                Err(e) => {
                    println!("Failed to generate prompt with interaction context: {}", e);
                }
            }
        }
    }
    
    match Database::insert_message(NewMessage { ai: false, content: prompt_message.to_string() }) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Failed to add message to database: {}", e);
            return HttpResponse::InternalServerError().body("Error while adding message to database, check logs for more information");
        }
    };
    match prompt(&prompt_message) {
        Ok(v) => HttpResponse::Ok().body(v),
        Err(e) => {
            println!("Failed to generate prompt: {}", e);
            HttpResponse::InternalServerError().body("Error while generating prompt, check logs for more information")
        }
    }
}

#[get("/api/prompt/regenerate")]
async fn regenerate_prompt() -> HttpResponse {
    match Database::delete_latest_message() {
        Ok(_) => {},
        Err(e) => {
            println!("Failed to delete latest message: {}", e);
            return HttpResponse::InternalServerError().body("Error while deleting latest message, check logs for more information");
        }
    }
    let prompt_msg: String = match Database::get_latest_message() {
        Ok(v) => v.content,
        Err(e) => {
            println!("Failed to get latest message: {}", e);
            return HttpResponse::InternalServerError().body("Error while getting latest message, check logs for more information");
        }
    };
    match prompt(&prompt_msg) {
        Ok(v) => HttpResponse::Ok().body(v),
        Err(e) => {
            println!("Failed to re-generate prompt: {}", e);
            HttpResponse::InternalServerError().body("Error while generating prompt, check logs for more information")
        }
    }
}

//              Config

#[get("/api/config")]
async fn config() -> HttpResponse {
    let config = match Database::get_config() {
        Ok(v) => v,
        Err(e) => {
            println!("Failed to get config: {}", e);
            return HttpResponse::InternalServerError().body("Error while getting config, check logs for more information");
        }
    };
    let config_json = serde_json::to_string(&config).unwrap_or(String::from("Error serializing config as JSON"));
    HttpResponse::Ok().body(config_json)
}

#[put("/api/config")]
async fn config_post(received: web::Json<ConfigModify>) -> HttpResponse {
    match Database::change_config(received.into_inner()) {
        Ok(_) => HttpResponse::Ok().body("Config updated!"),
        Err(e) => {
            println!("Failed to update config: {}", e);
            HttpResponse::InternalServerError().body("Error while updating config, check logs for more information")
        }
    }
}

//              Attitude Tracking

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
            let attitude_json = serde_json::to_string(&attitude).unwrap_or(String::from("Error serializing attitude as JSON"));
            HttpResponse::Ok().body(attitude_json)
        },
        Ok(None) => HttpResponse::NotFound().body("Attitude not found"),
        Err(e) => {
            println!("Failed to get attitude: {}", e);
            HttpResponse::InternalServerError().body("Error while getting attitude, check logs for more information")
        }
    }
}

#[post("/api/attitude")]
async fn create_or_update_attitude(received: web::Json<CompanionAttitude>) -> HttpResponse {
    let attitude = received.into_inner();
    match Database::create_or_update_attitude(attitude.companion_id, attitude.target_id, &attitude.target_type, &attitude) {
        Ok(id) => HttpResponse::Ok().body(format!("Attitude created/updated with id: {}", id)),
        Err(e) => {
            println!("Failed to create/update attitude: {}", e);
            HttpResponse::InternalServerError().body("Error while creating/updating attitude, check logs for more information")
        }
    }
}

#[get("/api/attitude/companion/{companion_id}")]
async fn get_companion_attitudes(companion_id: web::Path<i32>) -> HttpResponse {
    match Database::get_all_companion_attitudes(*companion_id) {
        Ok(attitudes) => {
            let attitudes_json = serde_json::to_string(&attitudes).unwrap_or(String::from("Error serializing attitudes as JSON"));
            HttpResponse::Ok().body(attitudes_json)
        },
        Err(e) => {
            println!("Failed to get companion attitudes: {}", e);
            HttpResponse::InternalServerError().body("Error while getting companion attitudes, check logs for more information")
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
    match Database::update_attitude_dimension(update.companion_id, update.target_id, &update.target_type, &update.dimension, update.delta) {
        Ok(_) => HttpResponse::Ok().body("Attitude dimension updated!"),
        Err(e) => {
            println!("Failed to update attitude dimension: {}", e);
            HttpResponse::InternalServerError().body("Error while updating attitude dimension, check logs for more information")
        }
    }
}

#[get("/api/attitude/memories/{companion_id}")]
async fn get_attitude_memories(companion_id: web::Path<i32>) -> HttpResponse {
    match Database::get_priority_attitude_memories(*companion_id, 20) {
        Ok(memories) => {
            let memories_json = serde_json::to_string(&memories).unwrap_or(String::from("Error serializing attitude memories as JSON"));
            HttpResponse::Ok().body(memories_json)
        },
        Err(e) => {
            println!("Failed to get attitude memories: {}", e);
            HttpResponse::InternalServerError().body("Error while getting attitude memories, check logs for more information")
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
        },
        Err(e) => {
            println!("Failed to detect persons: {}", e);
            HttpResponse::InternalServerError().body("Error while detecting persons, check logs for more information")
        }
    }
}

#[get("/api/persons")]
async fn get_all_persons() -> HttpResponse {
    match Database::get_all_third_party_individuals() {
        Ok(persons) => {
            let persons_json = serde_json::to_string(&persons).unwrap_or(String::from("Error serializing persons as JSON"));
            HttpResponse::Ok().body(persons_json)
        },
        Err(e) => {
            println!("Failed to get all persons: {}", e);
            HttpResponse::InternalServerError().body("Error while getting persons, check logs for more information")
        }
    }
}

#[get("/api/persons/{name}")]
async fn get_person_by_name(name: web::Path<String>) -> HttpResponse {
    match Database::get_third_party_by_name(&name) {
        Ok(Some(person)) => {
            let person_json = serde_json::to_string(&person).unwrap_or(String::from("Error serializing person as JSON"));
            HttpResponse::Ok().body(person_json)
        },
        Ok(None) => HttpResponse::NotFound().body("Person not found"),
        Err(e) => {
            println!("Failed to get person by name: {}", e);
            HttpResponse::InternalServerError().body("Error while getting person, check logs for more information")
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
        },
        Err(e) => {
            println!("Failed to plan interaction: {}", e);
            HttpResponse::InternalServerError().body("Error while planning interaction, check logs for more information")
        }
    }
}

#[get("/api/interactions/planned/{companion_id}")]
async fn get_planned_interactions(companion_id: web::Path<i32>) -> HttpResponse {
    match Database::get_planned_interactions(*companion_id, Some(10)) {
        Ok(interactions) => {
            let interactions_json = serde_json::to_string(&interactions).unwrap_or(String::from("Error serializing interactions as JSON"));
            HttpResponse::Ok().body(interactions_json)
        },
        Err(e) => {
            println!("Failed to get planned interactions: {}", e);
            HttpResponse::InternalServerError().body("Error while getting planned interactions, check logs for more information")
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
        },
        Err(e) => {
            println!("Failed to complete interaction: {}", e);
            HttpResponse::InternalServerError().body("Error while completing interaction, check logs for more information")
        }
    }
}

#[get("/api/interactions/history/{companion_id}/{third_party_id}")]
async fn get_interaction_history(params: web::Path<(i32, i32)>) -> HttpResponse {
    let (companion_id, third_party_id) = params.into_inner();
    match Database::get_interaction_history(companion_id, third_party_id) {
        Ok(history) => {
            let history_json = serde_json::to_string(&history).unwrap_or(String::from("Error serializing history as JSON"));
            HttpResponse::Ok().body(history_json)
        },
        Err(e) => {
            println!("Failed to get interaction history: {}", e);
            HttpResponse::InternalServerError().body("Error while getting interaction history, check logs for more information")
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
            let interaction_json = serde_json::to_string(&interaction).unwrap_or(String::from("Error serializing interaction as JSON"));
            HttpResponse::Ok().body(interaction_json)
        },
        Ok(None) => {
            HttpResponse::Ok().body("{\"message\": \"No interaction detected\"}")
        },
        Err(e) => {
            println!("Failed to detect interaction: {}", e);
            HttpResponse::InternalServerError().body("Error while detecting interaction, check logs for more information")
        }
    }
}

#[post("/api/prompt/stream")]
async fn start_streaming_session(received: web::Json<StreamingRequest>) -> HttpResponse {
    let request = received.into_inner();
    let session_id = request.session_id.clone();
    let session_id_clone = session_id.clone();
    
    // Start streaming session
    let mut _rx = INFERENCE_OPTIMIZER.start_streaming_session(session_id.clone());
    
    // In a real implementation, this would start async LLM inference
    // For now, we'll simulate streaming by sending chunks
    tokio::spawn(async move {
        // Simulate processing chunks
        for i in 1..=5 {
            let chunk = StreamChunk {
                request_id: session_id_clone.clone(),
                content: format!("Chunk {} of response... ", i),
                is_complete: i == 5,
                token_count: Some(i * 10),
            };
            
            if INFERENCE_OPTIMIZER.stream_chunk(&session_id_clone, chunk).is_err() {
                break;
            }
            
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }
        
        // End session
        INFERENCE_OPTIMIZER.end_streaming_session(&session_id_clone);
    });
    
    HttpResponse::Ok().json(serde_json::json!({
        "session_id": session_id,
        "status": "streaming_started"
    }))
}

#[get("/api/inference/stats")]
async fn get_inference_stats() -> HttpResponse {
    let stats = INFERENCE_OPTIMIZER.get_stats();
    let (cache_size, cache_hits, hit_rate) = INFERENCE_OPTIMIZER.get_cache_stats();
    
    let response = serde_json::json!({
        "performance": {
            "total_requests": stats.total_requests,
            "avg_response_time_ms": stats.avg_response_time.as_millis(),
            "batch_processed": stats.batch_processed,
            "streaming_sessions": stats.streaming_sessions
        },
        "cache": {
            "size": cache_size,
            "hits": cache_hits,
            "misses": stats.cache_misses,
            "hit_rate": hit_rate
        }
    });
    
    HttpResponse::Ok().json(response)
}

#[post("/api/inference/cache/cleanup")]
async fn cleanup_cache() -> HttpResponse {
    INFERENCE_OPTIMIZER.cleanup_cache();
    HttpResponse::Ok().json(serde_json::json!({
        "status": "cache_cleaned"
    }))
}

// Session Management Endpoints
#[derive(Deserialize)]
struct CreateSessionRequest {
    companion_id: i32,
    user_id: Option<i32>,
}

#[post("/api/session")]
async fn create_session(
    session_manager: web::Data<SessionManager>,
    req: web::Json<CreateSessionRequest>,
) -> HttpResponse {
    match session_manager.create_session(req.companion_id, req.user_id) {
        Ok(session) => {
            let response_json = serde_json::to_string(&session).unwrap_or_else(|_| "{}".to_string());
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
            let response_json = serde_json::to_string(&session).unwrap_or_else(|_| "{}".to_string());
            HttpResponse::Ok().body(response_json)
        }
        Err(e) => {
            HttpResponse::NotFound().body(format!("Session not found: {}", e))
        }
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
        Err(e) => {
            println!("Failed to end session: {}", e);
            HttpResponse::InternalServerError().body(format!("Error ending session: {}", e))
        }
    }
}

#[get("/api/session/stats/summary")]
async fn get_session_stats(
    session_manager: web::Data<SessionManager>,
) -> HttpResponse {
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

//

#[actix_web::main]
async fn main() -> std::io::Result<()> {

    let port: u16 = 3000;
    let hostname: &str = "0.0.0.0";

    match Database::new() {
        Ok(_) => { }
        Err(e) => eprintln!("⚠️ Failed to connect to sqlite database: {}\n", e),
    }

    match LongTermMem::connect() {
        Ok(_) => { }
        Err(e) => eprintln!("⚠️ Failed to connect to tantivy: {}\n", e),
    }

    match DialogueTuning::create() {
        Ok(_) => { }
        Err(e) => eprintln!("⚠️ Failed to create dialogue tuning table in sqlite database: {}\n", e),
    }

    println!("AI Companion v1 successfully launched! 🚀\n");

    println!("Listening on:\n  -> http://{}:{}/", hostname, port);
    println!("  -> http://localhost:{}/\n", port);
    println!("https://github.com/Hukasx0/ai-companion\n   By Hubert \"Hukasx0\" Kasperek\n");
    
    // Initialize session manager with 30 minute timeout
    let session_manager = web::Data::new(SessionManager::new(30));
    
    HttpServer::new(move || {
        App::new()
            .app_data(session_manager.clone())
            .service(index)
            .service(js)
            .service(js2)
            .service(css)
            .service(project_logo)
            .service(companion_avatar_img)
            .service(companion_avatar_custom)
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
            .service(get_attitude)
            .service(create_or_update_attitude)
            .service(get_companion_attitudes)
            .service(update_attitude_dimension)
            .service(get_attitude_memories)
            .service(detect_persons)
            .service(get_all_persons)
            .service(get_person_by_name)
            .service(plan_interaction)
            .service(get_planned_interactions)
            .service(complete_interaction)
            .service(get_interaction_history)
            .service(detect_interaction)
            .service(start_streaming_session)
            .service(get_inference_stats)
            .service(cleanup_cache)
            .service(create_session)
            .service(get_session)
            .service(update_session_attitude)
            .service(end_session)
            .service(get_session_stats)
    })
    .bind((hostname, port))?
    .run()
    .await
}
