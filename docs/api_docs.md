# AI Companion v1 API Documentation

## Introduction

The Companion API allows users to send and receive messages, manage companion settings, and retrieve various data related to the companion, user or backend.

## Base URL

The base URL for accessing the Companion API is `http://localhost:3000/api` or `http://<your_ip_address>:3000/api`

## Endpoints

### 1. Messages

#### 1.1 Get Messages

- **URL:** `/message`
- **Method:** `GET`
- **Description:** Retrieve a list of messages exchanged with the companion. In `joiner` multiplayer mode, answers from this instance's mirror of the host's transcript instead of its own database.
- **Parameters:**
  - `limit` (optional): The maximum number of messages to retrieve. Max is 50.
  - `offset` (optional): The offset for paginating through messages.
- **Response:**
  - Status: 200 OK
  - Body: Array of message objects.
- **Example Request:**
  ```http
  GET /message?limit=50&offset=0
  ```
- **Example Response:**
  ```json
  [
    {
      "id": 1,
      "ai": true,
      "speaker_id": "char",
      "content": "Hello there!",
      "created_at": "Saturday 20.04.2024 17:49"
    },
    {
      "id": 2,
      "ai": false,
      "speaker_id": "user",
      "content": "Hi, can you help me with something?",
      "created_at": "Saturday 20.04.2024 19:02"
    }
  ]
  ```
  `speaker_id` is the source of truth for who sent the message; `ai` is
  always derived from it (`speaker_id != "user"`) and kept for backward
  compatibility.

#### 1.2 Erase messages
- **URL:** `/message`
- **Method:** `DELETE`
- **Description:** Delete every message saved in short-term memory and chat log
- **Response:**
  - Status: 200 OK
  - Body: Chat log cleared!
  - Status: 409 Conflict — this instance is in `joiner` multiplayer mode; send messages from the host instead
- **Example Request:**
  ```http
  DELETE /message
  ```


#### 1.3 Add Message

- **URL:** `/message`
- **Method:** `POST`
- **Description:** Add a message to the database (without prompting the AI).
- **Request Body:**
  - `speaker_id` (string, preferred): Who sent the message (e.g. `"user"` or `"char"`).
  - `ai` (boolean, legacy): Indicates whether the message is from the AI (true) or user (false). Still accepted; `true` resolves to `speaker_id: "char"` and `false` to `speaker_id: "user"`.
  - `content` (string): The content of the message. Any `@Display Name` mention of a chat
    participant is stored as `@id` (e.g. `@bot1`), never in display-name form.
  - Either `speaker_id` or `ai` must be given. If both are given they must agree, or the request is rejected.
- **Response:**
  - Status: 200 OK
  - Body: Message added!
  - Status: 400 Bad Request
  - Body: error text, if neither `speaker_id` nor `ai` is given, or the two disagree.
  - Status: 409 Conflict — this instance is in `joiner` multiplayer mode; send messages from the host instead
- **Example Request:**
  ```http
  POST /message
  Content-Type: application/json

  {
    "speaker_id": "char",
    "content": "Message sent by AI"
  }
  ```
  The legacy form still works:
  ```http
  POST /message
  Content-Type: application/json

  {
    "ai": true,
    "content": "Message sent by AI"
  }
  ```

#### 1.4 Get message by ID

- **URL:** `/message/{id}`
- **Method:** `GET`
- **Description:** Retrieve a message by its ID
- **Path Parameters:**
  - `id` (integer): The ID of the message
- **Response:**
  - Status: 200 OK
  - Body: message object
- **Example Request:**
  ```http
  GET /message/2
  ```
- **Example Response:**
  ```json
    {
      "id": 2,
      "ai": false,
      "speaker_id": "user",
      "content": "Hi, can you help me with something?",
      "created_at": "Saturday 20.04.2024 19:02"
    }
  ```

#### 1.5 Edit Message

- **URL:** `/message/{id}`
- **Method:** `PUT`
- **Description:** Edit a message's text by its ID. The edit is content-only:
  it never changes which side (user or AI) the message is attributed to. Any
  `ai` field in the request body is ignored, so older clients that still send
  it keep working without flipping the message's role.
- **Path Parameters:**
  - `id` (integer): The ID of the message to edit
- **Request Body:**
  - `content` (string): The new content of the message. As with `POST /api/message`, any
    `@Display Name` mention is stored as `@id`.
- **Response:**
  - Status: 200 OK
  - Body: Message edited at id {id}
  - Status: 409 Conflict — this instance is in `joiner` multiplayer mode; send messages from the host instead
- **Example Request:**
  ```http
  PUT /message/{id}
  Content-Type: application/json

  {
    "content": "Edited message content"
  }
  ```

#### 1.6 Delete Message

- **URL:** `/message/{id}`
- **Method:** `DELETE`
- **Description:** Delete a message by its ID.
- **Path Parameters:**
  - `id` (integer): The ID of the message to delete.
- **Response:**
  - Status: 200 OK
  - Body: Message deleted at id {id}
  - Status: 409 Conflict — this instance is in `joiner` multiplayer mode; send messages from the host instead
- **Example Request:**
  ```http
  DELETE /message/1
  ```

### 2. Companion data

#### 2.1 Get Companion data

- **URL:** `/companion`
- **Method:** `GET`
- **Description:** Retrieve information about the companion.
- **Response:**
  - Status: 200 OK
  - Body: Companion object.
- **Example Request:**
  ```http
  GET /companion
  ```
- **Example Response:**
  ```json
  {
    "name": "Assistant",
    "persona": "Friendly assistant",
    "example_dialogue": "",
    "first_message": "Hello world!",
    "long_term_mem": 2,
    "short_term_mem": 5,
    "roleplay": true,
    "dialogue_tuning": false,
    "avatar_path": "/assets/companion_avatar-4rust.jpg"
  }
  ```

#### 2.2 Update Companion data

- **URL:** `/companion`
- **Method:** `PUT`
- **Description:** Update information about the companion.
- **Request Body:**
  - `name` (string): The name of the companion.
  - `persona` (string): The persona or description of the companion.
  - `example_dialogue` (string): Example dialogue for the companion.
  - `first_message` (string): First message sent by companion
  - `long_term_mem` (number): The number of entries that the AI ​​should recall from long-term memory during a conversation
  - `short_term_mem` (number): The number of entries that the AI ​​should recall from short-term memory during a conversation
  - `roleplay` (boolean): Should the AI ​​perform non-verbal actions between asterisks, e.g. *moves closer*, *waves hello*
  - `dialogue_tuning` (boolean): Should ai use message tuning
  - `avatar_path` (string): Path to the companion's avatar image.
- **Response:**
  - Status: 200 OK
  - Body: Companion data edited!
- **Example Request:**
  ```http
  PUT /companion
  Content-Type: application/json

  {
    "name": "Companion",
    "persona": "New companion",
    "example_dialogue": "",
    "first_message": "Hello friend!",
    "long_term_mem": 0,
    "short_term_mem": 2,
    "roleplay": false,
    "dialogue_tuning": true,
    "avatar_path": "/assets/companion_avatar-4rust.jpg"
  }
  ```

#### 2.3 Update Companion data via character card (.png) file

- **URL:** `/companion/card`
- **Method:** `POST`
- **Description:** Update information about the companion via character card file (character card files follow the standard character card format).
- **Response:**
  - Status: 200 OK
  - Body: Updated companion data via character card!
- **Example Request:**
  ```sh
  curl -X POST -H "Content-Type: image/png" -T card.png http://localhost:3000/api/companion/card
  ```

#### 2.4 Update Companion data via character JSON data

- **URL:** `/companion/characterJson`
- **Method:** `POST`
- **Description:** Update information about the companion via character json (character JSON follows the standard character card format).
- **Request Body:**
  - `name` (string): The name of the companion.
  - `description` (string): The persona or description of the companion.
  - `first_mes` (string): First message sent by companion
  - `mes_example` (string): Example dialogue for the companion.
- **Response:**
  - Status: 200 OK
  - Body: Character json imported successfully!
- **Example Request:**
  ```http
  PUT /companion/characterJson
  Content-Type: application/json

  {
    "name": "Companion",
    "description": "New companion",
    "first_mes": "Hello friend!",
    "mes_example": "",
  }
  ```

#### 2.5 Update Companion avatar

- **URL:** `/companion/avatar`
- **Method:** `POST`
- **Description:** Update companion avatar image
- **Response:**
  - Status: 200 OK
  - Body: Companion avatar changed!
- **Example Request:**
  ```sh
  curl -X POST -H "Content-Type: image/png" -T avatar.png http://localhost:3000/api/companion/avatar
  ```

### 3. User data

#### 3.1 Get User data

- **URL:** `/user`
- **Method:** `GET`
- **Description:** Retrieve information about the user.
- **Response:**
  - Status: 200 OK
  - Body: User object.
- **Example Request:**
  ```http
  GET /user
  ```
- **Example Response:**
  ```json
  {
    "name": "User",
    "persona": "User description"
  }
  ```

#### 3.2 Update User data

- **URL:** `/user`
- **Method:** `PUT`
- **Description:** Update information about the user.
- **Request Body:**
  - `name` (string): The name of the user.
  - `persona` (string): The persona or description of the user.
- **Response:**
  - Status: 200 OK
  - Body: User data edited!
- **Example Request:**
  ```http
  PUT /user
  Content-Type: application/json

  {
    "name": "John Doe",
    "persona": "Programmer that wants to use ai-companion as api for his project"
  }
  ```

### 4. Configuration

#### 4.1 Get Configuration

- **URL:** `/config`
- **Method:** `GET`
- **Description:** Retrieve configuration settings for the companion backend.
- **Response:**
  - Status: 200 OK
  - Body: Configuration settings object.
- **Example Request:**
  ```http
  GET /config
  ```
- **Example Response:**
  ```json
  {
    "device": "CPU",
    "llm_model_path": "/path/to/model.gguf",
    "gpu_layers": 20,
    "prompt_template": "Default",
    "multiplayer_mode": "solo",
    "multiplayer_password_set": false,
    "multiplayer_host_address": "",
    "multiplayer_participant_id": "",
    "mention_followup_depth": 1,
    "remote_generation_timeout_secs": 120
  }
  ```
  Note: `multiplayer_password` is never returned; `multiplayer_password_set` reports whether a host password is currently stored.

#### 4.2 Update Configuration

- **URL:** `/config`
- **Method:** `PUT`
- **Description:** Update configuration settings for the companion backend.
- **Request Body:**
  - `device` (string) ("CPU" || "GPU" || "Metal"): The device used for processing (CPU, GPU, Metal).
  - `llm_model_path` (string): Path to the language model.
  - `gpu_layers` (integer): Number of GPU layers.
  - `prompt_template` (string) ("Auto" || "Default" || "Llama2" || "Mistral"): Prompt template used to format the prompt. `Auto` renders the prompt with the chat template stored inside the GGUF file and falls back to `Default` when the model does not carry one. It is the default for new installs.
  - `multiplayer_mode` (string) ("solo" || "host" || "joiner"): The instance's multiplayer role. Defaults to `"solo"`.
  - `multiplayer_host_address` (string): The host's `host:port` to connect to. Required, and validated, only in `joiner` mode.
  - `multiplayer_participant_id` (string): This instance's participant id, `^[a-z][a-z0-9_]{0,15}$` (1-16 lowercase letters, digits or `_`, starting with a letter). Required, and validated, only in `joiner` mode.
  - `mention_followup_depth` (integer, 0-10): How many rounds of `@mention` follow-ups a reply can trigger.
  - `remote_generation_timeout_secs` (integer, 5-3600): How long to wait for a joiner bot's remote reply.
  - `multiplayer_password` (string, optional, write-only): The shared host/joiner password. Omitted or empty leaves the currently stored password unchanged; it is never echoed back by `GET /config`.
- **Response:**
  - Status: 200 OK
  - Body: Config updated!
  - Status: 400 Bad Request
  - Body: A message describing the invalid field, e.g. an unknown `multiplayer_mode`, an invalid `multiplayer_participant_id`, a `joiner` request missing its host address, or a `host` request with no password stored or supplied.
- **Example Request:**
  ```http
  PUT /config
  Content-Type: application/json

  {
    "device": "GPU",
    "llm_model_path": "/path/to/model.gguf",
    "gpu_layers": 30,
    "prompt_template": "Mistral",
    "multiplayer_mode": "solo",
    "multiplayer_host_address": "",
    "multiplayer_participant_id": "",
    "mention_followup_depth": 1,
    "remote_generation_timeout_secs": 120
  }
  ```

### 5. Memory

#### 5.1 Add entry to long-term memory

- **URL:** `/memory/longTerm`
- **Method:** `POST`
- **Description:** Add data to ai long-term memory
- **Request Body:**
  - `entry` (string): Information that you want to save in your companion's long-term memory, I recommend breaking large pieces of text into parts
- **Response:**
  - Status: 200 OK
  - Body: Long term memory entry added!
- **Example Request:**
  ```http
  PUT /memory/longTerm
  Content-Type: application/json

  {
    "entry": "AI Companion is a project that aims to provide a quick, simple, light and convenient way to create AI chatbots on your local computer"
  }
  ```

#### 5.2 Erase long-term memory

- **URL:** `/memory/longTerm`
- **Method:** `DELETE`
- **Description:** Clear long term memory.
- **Response:**
  - Status: 200 OK
  - Body: Long term memory cleared!
- **Example Request:**
  ```http
  DELETE /memory/longTerm
  ```

#### 5.3 Add last dialogue to dialogue tuning

- **URL:** `/memory/dialogueTuning`
- **Method:** `POST`
- **Description:** Adds the user's previous message and AI's response as dialogue tuning
- **Response:**
  - Status: 200 OK
  - Body: Saved previous dialogue as template dialogue
- **Example Request:**
  ```http
  POST /memory/dialogueTuing
  ```

#### 5.4 Erase dialogue tuning entries

  - **URL:** `/memory/dialogueTuning`
- **Method:** `DELETE`
- **Description:** Clear all dialogue tuning entries.
- **Response:**
  - Status: 200 OK
  - Body: Dialogue tuning memory cleared!
- **Example Request:**
  ```http
  DELETE /memory/dialogueTuning
  ```

### 6. Prompting

#### 6.1 Prompt the AI

- **URL:** `/prompt`
- **Method:** `POST`
- **Description:** Prompt the ai, (message and response are saved in short-term, long-term memory and chat log)
- **Request Body:**
  - `prompt` (string): Prompt to the AI
- **Response:**
  - Status: 200 OK
  - Body: generated text
  - Status: 409 Conflict — a turn (from `/prompt`, `/prompt/stream`, or `/prompt/regenerate`) is already in flight; wait for it to finish before sending another message
  - Status: 409 Conflict — this instance is in `joiner` multiplayer mode; send messages from the host instead
- **Example Request:**
  ```http
  POST /prompt
  Content-Type: application/json

  {
    "prompt": "what time is it currently?"
  }
  ```

#### 6.2 Regenerate the last answer

- **URL:** `/prompt/regenerate`
- **Method:** `GET`
- **Description:** Regenerate answer to your AI prompt (answer is saved in short-term, long-term memory and chat log)
- **Response:**
  - Status: 200 OK
  - Body: generated text
  - Status: 409 Conflict — a turn is already in flight; wait for it to finish before regenerating
  - Status: 409 Conflict — this instance is in `joiner` multiplayer mode; send messages from the host instead
- **Example Request:**
  ```http
  GET /prompt/regenerate
  ```

#### 6.3 Stream a prompt response

- **URL:** `/prompt/stream`
- **Method:** `POST`
- **Description:** Prompt the AI and receive the reply token by token as Server-Sent Events. The user message and the finished reply are saved in short-term memory, long-term memory and the chat log, exactly as with `/prompt`.
- **Request Body:**
  - `prompt` (string): Prompt to the AI
  - `session_id` (string): Caller-supplied identifier, echoed back on every chunk as `request_id`
- **Response:**
  - Status: 200 OK
  - Content-Type: `text/event-stream`
  - Body: a sequence of `data:` events, each carrying one JSON chunk, one per speaker action in the round. `event` says which of five kinds a chunk is; `speaker_id` is the speaker the chunk is about (empty on the attitude chunk and on `round_complete`/`error`, which are round-wide rather than per-speaker):
    - `reply_started`: a speaker is about to generate. Empty `content`, `is_complete: false`.
    - `token`: one token from `speaker_id`, appended to that speaker's `content` so far, `is_complete: false`. A `token`-event chunk with empty `content` and an `attitude` object instead is the attitude chunk: sent once after generation, only when the turn moved at least one attitude dimension. `attitude.attitude` is the companion's full post-turn `CompanionAttitude` toward the user, `attitude.summary` is its natural language rendering (with `{{companion}}` and `{{user}}` placeholders), and `attitude.deltas` lists `{ dimension, delta }` for the dimensions that moved.
    - `reply_complete`: `speaker_id`'s finished, persisted reply. `content` is the sanitized text, `message_id` is the new row's id, `is_complete: false`. A speaker skipped because it did not respond arrives as a bare `reply_complete` for `speaker_id: "system"` (no preceding `reply_started`) carrying the persisted notice.
    - `round_complete`: the round is over. Empty `content`, `is_complete: true`.
    - `error`: the round failed. `error` carries the failure message, empty `content`, `is_complete: true`.

    A full round is `reply_started`, zero or more `token`s, then `reply_complete`, repeated once per speaker (host companion first, then each joined bot in join order), followed by the optional attitude chunk and then `round_complete`. `is_complete` is `true` only on `round_complete` and `error`, so a client that only tracks that field still terminates correctly; a client that only appends non-final `content` renders `reply_complete`'s sanitised text after the raw tokens, so a client must switch on `event` to render each speaker's reply correctly. `message_id`, `error` and `attitude` are omitted when absent.
  - Status: 409 Conflict — a turn is already in flight; wait for it to finish before sending another message
  - Status: 409 Conflict — this instance is in `joiner` multiplayer mode; send messages from the host instead
- **Example Request:**
  ```http
  POST /prompt/stream
  Content-Type: application/json

  {
    "prompt": "what time is it currently?",
    "session_id": "abc123"
  }
  ```
- **Example Response:**
  ```
  data: {"request_id":"abc123","event":"reply_started","content":"","is_complete":false,"speaker_id":"char"}

  data: {"request_id":"abc123","event":"token","content":"It","is_complete":false,"token_count":1,"speaker_id":"char"}

  data: {"request_id":"abc123","event":"token","content":"'s","is_complete":false,"token_count":2,"speaker_id":"char"}

  data: {"request_id":"abc123","event":"reply_complete","content":"It's 10:04.","is_complete":false,"token_count":2,"speaker_id":"char","message_id":42}

  data: {"request_id":"abc123","event":"token","content":"","is_complete":false,"token_count":2,"speaker_id":"","attitude":{"attitude":{"companion_id":1,"target_id":1,"target_type":"user","trust":7.0,"...":0.0},"summary":"{{companion}} trusts {{user}}","deltas":[{"dimension":"trust","delta":3.0}]}}

  data: {"request_id":"abc123","event":"round_complete","content":"","is_complete":true,"token_count":2,"speaker_id":""}
  ```
- **Notes:**
  - A turn is claimed for the whole request, from the moment the user message is persisted until the reply (and its attitude update) is persisted. A second `/prompt`, `/prompt/stream`, or `/prompt/regenerate` call that arrives while a turn is in flight gets 409 immediately rather than queuing, so a burst of sends cannot interleave one turn's user message into another's.
  - If the client disconnects mid-stream, generation still runs to completion so every speaker's reply is persisted.

### Inspect the assembled prompt

- **URL:** `/debug/prompt`
- **Method:** `GET`
- **Description:** Returns the prompt a turn would send, without loading a model. Intended for verifying what the attitude block actually injects: `attitude_context` in the response is exactly the text `generate` folds into the system portion of the prompt.
- **Query Parameters:**
  - `companion_id` (integer, optional): defaults to the single companion the backend resolves server-side
  - `prompt` (string, optional): message the long-term memory recall is keyed on. Omitted, the prompt is assembled with no recalled entries.
- **Response:**
  - Status: 200 OK
  - Body: `{ "system_prompt": string, "chat_history": [[bool, string]], "attitude_context": string, "managed_messages": [Message] }`
- **Notes:**
  - No model is loaded, so for `PromptTemplate::Auto` the response holds the pre-template system text plus the role-tagged `chat_history` rather than the final rendered string. Every other template returns the finished prompt in `system_prompt`.
  - `docs/attitude_verification.md` records a comparison run made with this route and `backend/scripts/attitude_comparison.sh`.

### 7. Multiplayer

#### 7.1 Get this instance's multiplayer status

- **URL:** `/multiplayer/status`
- **Method:** `GET`
- **Description:** This instance's own multiplayer role (`multiplayer_mode` from `/config`) and, in `joiner` mode, its connection state to the host it is configured to join.
- **Response:**
  - Status: 200 OK
  - Body, `solo`/`host` mode: `{ "mode": "solo" | "host", "state": null }`. A `host` mode instance's connected participants are at `GET /multiplayer/participants` instead.
  - Body, `joiner` mode: `{ "mode": "joiner", "state": "disconnected" | "connecting" | "connected" | "rejected", "reason": string, "last_error": string | null, "attempts": number, "host_address": string, "participant_id": string, "participants": [ParticipantSummary] }`, where `reason` is present only when `state` is `"rejected"`; `last_error` is present (`null` or a message) only when `state` is `"disconnected"`, and stays `null` until a previous connection attempt has actually failed.
- **Example Request:**
  ```http
  GET /multiplayer/status
  ```
- **Example Response:**
  ```json
  {
    "mode": "joiner",
    "state": "connected",
    "attempts": 1,
    "host_address": "192.168.0.20:3000",
    "participant_id": "bot1",
    "participants": [
      { "id": "user", "display_name": "Alice", "kind": "human", "avatar_url": null, "connected": true }
    ]
  }
  ```
- **Notes:**
  - A `"rejected"` state (wrong password, or a duplicate/reserved participant id) is terminal: this instance stops retrying and stays in that state until restarted with a corrected config.
  - While in `joiner` mode, `GET /message` answers from this instance's mirror of the host's transcript, and `/prompt`, `/prompt/stream` and `/prompt/regenerate` all answer `409 Conflict`; see their entries above.

## Route index

Endpoint sections above cover the core messaging, companion, user, configuration, memory and prompting routes. The table below lists every route the backend registers, including those not yet written up in full. It is generated from the handler attributes in `backend/src/main.rs`.

| Method | Path |
| --- | --- |
| `GET` | `/api/attitude` |
| `POST` | `/api/attitude` |
| `DELETE` | `/api/attitude/clear` |
| `GET` | `/api/attitude/companion/{companion_id}` |
| `PUT` | `/api/attitude/dimension` |
| `GET` | `/api/attitude/memories/{companion_id}` |
| `GET` | `/api/attitude/summary/{companion_id}/{user_id}` |
| `GET` | `/api/companion` |
| `PUT` | `/api/companion` |
| `POST` | `/api/companion/avatar` |
| `POST` | `/api/companion/card` |
| `GET` | `/api/companion/characterJson` |
| `POST` | `/api/companion/characterJson` |
| `GET` | `/api/config` |
| `PUT` | `/api/config` |
| `POST` | `/api/estimate-response-time` |
| `GET` | `/api/gpu/allocation` |
| `GET` | `/api/gpu/memory` |
| `GET` | `/api/inference/stats` |
| `POST` | `/api/interactions/detect` |
| `GET` | `/api/interactions/history/{companion_id}/{third_party_id}` |
| `POST` | `/api/interactions/{interaction_id}/complete` |
| `POST` | `/api/interactions/plan` |
| `GET` | `/api/interactions/planned/{companion_id}` |
| `GET` | `/api/llm/directories` |
| `POST` | `/api/llm/directories` |
| `DELETE` | `/api/llm/directories/{id}` |
| `GET` | `/api/llm/models` |
| `POST` | `/api/llm/unload` |
| `DELETE` | `/api/memory/dialogueTuning` |
| `POST` | `/api/memory/dialogueTuning` |
| `DELETE` | `/api/memory/longTerm` |
| `POST` | `/api/memory/longTerm` |
| `DELETE` | `/api/message` |
| `GET` | `/api/message` |
| `POST` | `/api/message` |
| `DELETE` | `/api/message/{id}` |
| `GET` | `/api/message/{id}` |
| `PUT` | `/api/message/{id}` |
| `GET` | `/api/multiplayer/status` |
| `GET` | `/api/persons` |
| `POST` | `/api/persons/cleanup-duplicates` |
| `POST` | `/api/persons/cleanup-invalid` |
| `POST` | `/api/persons/detect` |
| `GET` | `/api/persons/{name}` |
| `POST` | `/api/prompt` |
| `GET` | `/api/prompt/regenerate` |
| `GET` | `/api/debug/prompt` |
| `POST` | `/api/prompt/stream` |
| `POST` | `/api/session` |
| `PUT` | `/api/session/attitude` |
| `GET` | `/api/session/{session_id}` |
| `POST` | `/api/session/{session_id}/end` |
| `GET` | `/api/session/stats/summary` |
| `GET` | `/api/user` |
| `PUT` | `/api/user` |

---

AI Companion v1

2024 Hubert Kasperek
