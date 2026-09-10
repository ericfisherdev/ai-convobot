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
      "created_at": "Saturday 20.04.2024 17:49",
      "pinned": false
    },
    {
      "id": 2,
      "ai": false,
      "speaker_id": "user",
      "content": "Hi, can you help me with something?",
      "created_at": "Saturday 20.04.2024 19:02",
      "pinned": true
    }
  ]
  ```
  `speaker_id` is the source of truth for who sent the message; `ai` is
  always derived from it (`speaker_id != "user"`) and kept for backward
  compatibility. `pinned` (#179) is whether the message is exempt from
  compaction (`POST`/`DELETE /message/{id}/pin`, section 1.7/1.8); a
  `joiner` mode instance always reports `false`, since pins live on the
  host.

#### 1.2 Erase messages
- **URL:** `/message`
- **Method:** `DELETE`
- **Description:** Delete every message saved in short-term memory and chat log. Unlike editing, deleting, or regenerating a single message, this is **not** mirrored to a connected joiner in `host` mode — a joiner's own transcript keeps showing the pre-clear history until it reconnects (known gap). Also resets conversation compaction: every checkpoint, extracted fact, and pinned message is deleted, and the compacted-through cutoff is cleared, so the next chat starts compaction from scratch.
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
      "created_at": "Saturday 20.04.2024 19:02",
      "pinned": false
    }
  ```

#### 1.5 Edit Message

- **URL:** `/message/{id}`
- **Method:** `PUT`
- **Description:** Edit a message's text by its ID. The edit is content-only:
  it never changes which side (user or AI) the message is attributed to. Any
  `ai` field in the request body is ignored, so older clients that still send
  it keep working without flipping the message's role. If the message falls
  inside a committed compaction checkpoint's range, that checkpoint is marked
  stale: its summary and facts still render (better than nothing), but the
  chat should offer a re-compaction (`POST /api/compaction/draft` with
  `{"from_stale": true}`).
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
- **Description:** Delete a message by its ID. If it was pinned, the pin is
  removed along with it. If the message fell inside a committed compaction
  checkpoint's range, that checkpoint is marked stale — see 1.5.
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

#### 1.7 Pin a message

- **URL:** `/message/{id}/pin`
- **Method:** `POST`
- **Description:** Pins a message so it stays in the prompt verbatim regardless of what a compaction checkpoint compacts over it (#179). Idempotent: pinning an already-pinned message is not an error.
- **Path Parameters:**
  - `id` (integer): The ID of the message to pin.
- **Response:**
  - Status: 200 OK
  - Body: Message pinned at id {id}!
  - Status: 404 Not Found — `id` does not name a message.
  - Status: 409 Conflict — this instance is in `joiner` multiplayer mode; pins live on the host.
- **Example Request:**
  ```http
  POST /message/1/pin
  ```

#### 1.8 Unpin a message

- **URL:** `/message/{id}/pin`
- **Method:** `DELETE`
- **Description:** Unpins a message. Idempotent: unpinning a message that was never pinned is not an error.
- **Path Parameters:**
  - `id` (integer): The ID of the message to unpin.
- **Response:**
  - Status: 200 OK
  - Body: Message unpinned at id {id}!
  - Status: 404 Not Found — `id` does not name a message.
  - Status: 409 Conflict — this instance is in `joiner` multiplayer mode; pins live on the host.
- **Example Request:**
  ```http
  DELETE /message/1/pin
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
    "remote_generation_timeout_secs": 120,
    "compact_threshold_tokens": null,
    "compact_min_messages": 8,
    "compaction_model_path": null,
    "heuristic_person_detection": true
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
  - `compact_threshold_tokens` (integer, optional, >= 256): Token budget a companion's recent-message window must exceed before conversation compaction drafts a checkpoint. `null` (the default) derives the threshold at runtime instead of using a fixed number.
  - `compact_min_messages` (integer, >= 2, default 8): Fewest uncompacted messages compaction will ever fire on, regardless of token count.
  - `compaction_model_path` (string, optional): Model used for compaction's summarisation/extraction passes. `null` (the default) uses `llm_model_path`.
  - `heuristic_person_detection` (boolean, default true): Whether the pre-compaction heuristic person-detection/third-party-mention-tracking/interaction-detection pass runs on every user turn. #177 adds `compaction::persons::PersonsObserver` (creates `third_party_individuals` rows with `source: "compaction"` from canon-validated `Person` facts at commit time) and this gate, but the default stays `true` for now since nothing in production reaches compaction's commit path until #179's route lands; flipping the default to `false` is deferred to follow-up issue #201 once that path is live. `POST /api/persons/detect` and `POST /api/interactions/detect` remain available as manual triggers regardless of this flag.
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
    "remote_generation_timeout_secs": 120,
    "compact_threshold_tokens": null,
    "compact_min_messages": 8,
    "compaction_model_path": null,
    "heuristic_person_detection": true
  }
  ```

### 5. Memory

#### 5.1 Add entry to long-term memory

- **URL:** `/memory/longTerm`
- **Method:** `POST`
- **Description:** Add data to ai long-term memory. Manual entries are plain text with no date prefix (the compaction feature dropped the old `* at <date> *` stamp for every entry, manual or automatic) and are not touched by extraction — they are dropped when `POST /memory/longTerm/rebuild` re-indexes, since rebuild only re-indexes facts.
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

#### 5.3 Rebuild long-term memory from facts

- **URL:** `/memory/longTerm/rebuild`
- **Method:** `POST`
- **Description:** Repair path for the tantivy long-term memory index: replaces its contents with the companion's currently active compaction facts (any manual entries added via 5.1 are dropped). The index is kept current automatically as checkpoints commit; this endpoint is only needed if it drifts, or after a schema mismatch recreated it empty on startup. Claims the same turn slot a chat reply or a compaction commit does, so a rebuild while either is in progress is rejected rather than racing it.
- **Response:**
  - Status: 200 OK
  - Body: `Long term memory rebuilt from {n} facts`
  - Status: 409 Conflict (a reply is being generated or a compaction draft is being committed)
- **Example Request:**
  ```http
  POST /memory/longTerm/rebuild
  ```

#### 5.4 Add last dialogue to dialogue tuning

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

#### 5.5 Erase dialogue tuning entries

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
- **Description:** Prompt the ai, (message and response are saved in short-term, long-term memory and chat log). In `host` multiplayer mode this runs the whole round — the host companion, then each connected joiner in join order, over the same socket connection `/api/multiplayer/ws` accepted — but the response body is only the host companion's reply; every joiner's reply (and any skip notice for one that did not respond) lands in the same round, visible via `GET /message` or `/prompt/stream`. A second `/prompt`, `/prompt/stream`, or `/prompt/regenerate` sent while a round is still in flight gets 409, whether or not the round has any joiners.
- **Request Body:**
  - `prompt` (string): Prompt to the AI
- **Response:**
  - Status: 200 OK
  - Body: `{ "reply": string, "compaction_draft_id": integer | null }` (#179; was a bare `text/plain` reply before this issue). `compaction_draft_id` is set when this round's end-of-turn compaction hook queued a new checkpoint draft, so a caller can start polling `GET /compaction/{id}` without waiting for a page refresh; `null` on every other round.
  - Status: 204 No Content — a mention-filtered round (multiplayer) that excluded the host companion; there is no reply to return.
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
- **Example Response:**
  ```json
  { "reply": "It's 10:04.", "compaction_draft_id": null }
  ```

#### 6.2 Regenerate the last answer

- **URL:** `/prompt/regenerate`
- **Method:** `GET`
- **Description:** Removes the newest bot reply (the host companion's own, or the trailing reply of a connected joiner from a round, #131) and regenerates it from the same owner — a joiner's bot regenerates on the joiner's own instance, over the same round protocol a live round uses, not on the host. The reply is anchored on the newest preceding user turn, which is not always its immediate predecessor (a mention follow-up can put another bot's reply in between). The answer is saved in short-term, long-term memory and the chat log exactly as a normal reply is; regenerating does not re-score attitude. In `host` mode, a connected joiner's transcript mirror is updated to match (the old reply removed, then the new one added) before this responds.
- **Response:**
  - Status: 200 OK
  - Body: generated text
  - Status: 409 Conflict — a turn is already in flight; wait for it to finish before regenerating
  - Status: 409 Conflict — the newest message is not a bot reply with a preceding user turn (it is a user message, a system notice, or the conversation has no earlier user turn to regenerate from); body: `The newest message is not a companion reply, so there is nothing to regenerate`
  - Status: 409 Conflict — the newest reply's owner is a remote bot that is not currently connected; nothing is changed; body: `{speaker_id} is not connected, so its reply cannot be regenerated`
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
    - `token`: one token from `speaker_id`, appended to that speaker's `content` so far, `is_complete: false`. A `token`-event chunk with empty `content` and an `attitude` object instead is the attitude chunk: sent once after generation, only when the turn moved at least one attitude dimension. `attitude.attitude` is the companion's full post-turn `CompanionAttitude` toward the user, `attitude.summary` is its natural language rendering (with `{{companion}}` and `{{user}}` placeholders), and `attitude.deltas` lists `{ dimension, delta }` for the dimensions that moved. A `token`-event chunk with empty `content` and `compaction_draft_id` set instead (#179) is the compaction-draft-ready chunk: sent once, only when this round's end-of-turn compaction hook queued a new checkpoint draft, so the client can start polling `GET /compaction/{id}` without waiting for a page refresh.
    - `reply_complete`: `speaker_id`'s finished, persisted reply. `content` is the sanitized text, `message_id` is the new row's id, `is_complete: false`. A speaker skipped because it did not respond arrives as a bare `reply_complete` for `speaker_id: "system"` (no preceding `reply_started`) carrying the persisted notice.
    - `round_complete`: the round is over. Empty `content`, `is_complete: true`.
    - `error`: the round failed. `error` carries the failure message, empty `content`, `is_complete: true`.

    A full round is `reply_started`, zero or more `token`s, then `reply_complete`, repeated once per speaker (host companion first, then each joined bot in join order), followed by the optional attitude chunk, then the optional compaction-draft-ready chunk, and then `round_complete`. `is_complete` is `true` only on `round_complete` and `error`, so a client that only tracks that field still terminates correctly; a client that only appends non-final `content` renders `reply_complete`'s sanitised text after the raw tokens, so a client must switch on `event` to render each speaker's reply correctly. `message_id`, `error`, `attitude` and `compaction_draft_id` are omitted when absent.
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
  - Body: `{ "system_prompt": string, "chat_history": [[bool, string]], "attitude_context": string, "managed_messages": [Message], "compaction": CompactionBlocks, "compacted_through": integer | null }`
  - `compaction` (conversation compaction): `{ "user_overlay": string, "companion_overlay": string, "rules": string, "story_so_far": string, "recent_detail": string, "pins": string, "over_budget_by": integer | null }` — the rendered compaction blocks folded into `system_prompt`, each `""` when that section has nothing to say. `over_budget_by` is only set when the user overlay, companion overlay, and rules block together exceed the compaction token slice (those three are never trimmed).
  - `compacted_through`: the checkpoint `managed_messages` starts after, `null` for a companion that has never been compacted.
- **Notes:**
  - No model is loaded, so for `PromptTemplate::Auto` the response holds the pre-template system text plus the role-tagged `chat_history` rather than the final rendered string. Every other template returns the finished prompt in `system_prompt`.
  - `docs/attitude_verification.md` records a comparison run made with this route and `backend/scripts/attitude_comparison.sh`.

### Unload resident models

- **URL:** `/llm/unload`
- **Method:** `POST`
- **Description:** Frees the resident model slot(s) `backend/src/llm.rs` keeps loaded between turns (the chat model) and, since #183, between compaction extraction jobs (the extraction model configured via `compaction_model_path`). The next turn or extraction call reloads whatever it needs from disk.
- **Query Parameters:**
  - `slot` (string, optional): `chat` frees only the chat model, `extractor` frees only the extraction model, `all` (the default when omitted) frees both.
- **Response:**
  - Status: 200 OK
  - Body: `{ "unloaded": bool, "model_path": string | null, "extractor_model_path": string | null }`. `unloaded` is `true` if either requested slot was occupied. `model_path` is the chat model's path if that slot was freed, `null` if it was not requested or was already empty; `extractor_model_path` is the same for the extraction slot.
- **Example Request:**
  ```http
  POST /llm/unload?slot=extractor
  ```
- **Example Response:**
  ```json
  { "unloaded": true, "model_path": null, "extractor_model_path": "/models/qwen3-4b-instruct-q4_k_m.gguf" }
  ```

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
      { "id": "user", "display_name": "Alice", "kind": "Human", "avatar_url": null, "connected": true }
    ]
  }
  ```
- **Notes:**
  - A `"rejected"` state (wrong password, or a duplicate/reserved participant id) is terminal: this instance stops retrying and stays in that state until restarted with a corrected config.
  - While in `joiner` mode, `GET /message` answers from this instance's mirror of the host's transcript, and `/prompt`, `/prompt/stream` and `/prompt/regenerate` all answer `409 Conflict`; see their entries above.

#### 7.2 List connected participants

- **URL:** `/multiplayer/participants`
- **Method:** `GET`
- **Description:** Every participant in the chat, in join order (`user`, then `char`, then every remote bot in the order it joined). Only meaningful on a `host` mode instance — a joiner's own roster is `participants` inside `GET /multiplayer/status` instead.
- **Response:**
  - Status: 200 OK
  - Body: array of `{ id, display_name, kind, avatar_url, connected }`. `kind` is one of `"Human"`, `"HostBot"`, `"RemoteBot"` (the Rust enum's own variant names — not lower- or snake-cased). `avatar_url` is `null` for `user` and `char` unless the companion has its own avatar; `connected` is always `true` for `Human`/`HostBot` and reflects the live socket connection for a `RemoteBot`.
  - Status: 404 Not Found — this instance is not currently in `host` mode.
- **Example Request:**
  ```http
  GET /multiplayer/participants
  ```
- **Example Response:**
  ```json
  [
    { "id": "user", "display_name": "Alice", "kind": "Human", "avatar_url": null, "connected": true },
    { "id": "char", "display_name": "Assistant", "kind": "HostBot", "avatar_url": "/assets/companion_avatar-4rust.jpg", "connected": true },
    { "id": "bot1", "display_name": "Ada", "kind": "RemoteBot", "avatar_url": "/api/multiplayer/participants/bot1/avatar", "connected": true }
  ]
  ```

#### 7.3 Get a remote bot's avatar

- **URL:** `/multiplayer/participants/{id}/avatar`
- **Method:** `GET`
- **Description:** The avatar a remote bot uploaded when it joined, if any. Only the format detected from the image's own magic bytes at upload time is served, regardless of what the joiner claimed its MIME type was.
- **Path Parameters:**
  - `id` (string): the participant id.
- **Response:**
  - Status: 200 OK
  - Content-Type: `image/png` or `image/jpeg`
  - Body: the raw image bytes.
  - Status: 404 Not Found — this instance is not currently in `host` mode, `id` is not a known participant, or that participant joined with no avatar.
- **Example Request:**
  ```http
  GET /multiplayer/participants/bot1/avatar
  ```

#### 7.4 Join the host's chat over WebSocket

- **URL:** `/multiplayer/ws`
- **Method:** `GET` (WebSocket upgrade)
- **Description:** The socket a joiner connects to. Registered on every instance, but only serves an upgrade in `host` mode; a `solo` or `joiner` instance answers `404` here just like 7.2 and 7.3.
- **Response:**
  - `101 Switching Protocols` on success, followed by the handshake below.
  - Status: 404 Not Found — this instance is not currently in `host` mode.
  - Status: 429 Too Many Requests — this address has had 5 failed join attempts within the last 10 minutes.

**Handshake.** Immediately after the upgrade, the host sends a `Challenge`. The joiner has 10 seconds to answer with a `Join`; anything else (a timeout, a malformed frame, or an unsupported `protocol_version`) gets `Rejected` and the socket is closed. On success the host answers `Joined`; every other currently-connected participant is sent a `ParticipantJoined` broadcast.

| Direction | Frame | Fields |
| --- | --- | --- |
| host → joiner | `Challenge` | `protocol_version` (`1`), `nonce` (base64, 32 random bytes) |
| joiner → host | `Join` | `protocol_version`, `id`, `display_name`, `avatar` (`{ mime, data_base64 }` or `null`, max 2 MB decoded, detected as PNG or JPEG from magic bytes), `proof` (base64 `HMAC-SHA256(key = password, msg = nonce_bytes ++ id_bytes)`) |
| host → joiner | `Joined` | `self_id`, `participants` (`[ParticipantSummary]`, same shape as 7.2's array), `transcript` (the last 50 messages, same shape `GET /message` returns) |
| host → joiner | `Rejected` | `reason`: one of the tag-only strings `"unsupported_protocol"`, `"bad_proof"`, `"no_host_password"`, `"duplicate_id"`, `"reserved_id"`, `"join_timeout"`, or `{ "invalid_avatar": "<detail>" }` |

Once admitted, both sides exchange:

| Direction | Frame | Fields |
| --- | --- | --- |
| host → joiner | `ParticipantJoined` | a `ParticipantSummary` (7.2's shape), flattened alongside `"type"` |
| host → joiner | `ParticipantLeft` | `id` |
| host → joiner | `Message` | one persisted message (`GET /message`'s row shape), flattened alongside `"type"` |
| host → joiner | `MessageEdited` | `message` (the edited row, in full) |
| host → joiner | `MessageRemoved` | `id` |
| host → joiner | `GenerateRequest` | `round_id`, `transcript` (the newest 50 messages, this round's context) |
| joiner → host | `Token` | `round_id`, `text` — one token of the reply this `GenerateRequest` asked for |
| joiner → host | `ReplyComplete` | `round_id`, `text` — the finished, persisted reply |
| joiner → host | `ReplyFailed` | `round_id`, `reason` — sent instead of a reply when the joiner cannot answer (a local turn already in progress, a failed generation thread, or the model erroring) |

Every frame is a JSON object tagged `"type"` (snake_case, e.g. `"reply_complete"`). Heartbeats are native WebSocket ping/pong control frames, not a JSON type of their own; the host pings every 15 seconds and drops the connection after 2 missed pongs.

### 8. Compaction

Conversation compaction (#171-#186): a checkpoint rolls a run of messages into a summary plus extracted facts, so old turns can drop out of the prompt without the companion losing what happened. Every route below is gated the same way the prompting routes are: `409 Conflict` in `joiner` multiplayer mode, since a joiner has no compaction store of its own.

#### 8.1 Trigger a draft manually

- **URL:** `/compaction/draft`
- **Method:** `POST`
- **Description:** Queues a new checkpoint draft over the uncompacted tail, the same way the automatic threshold/scene-break trigger would, and starts extraction on it in the background.
- **Response:**
  - Status: 202 Accepted
  - Body: `{ "draft_id": integer }`
  - Status: 409 Conflict — a turn is already in flight; wait for it to finish before triggering compaction
  - Status: 409 Conflict — a draft is already pending; body names its id, e.g. `a draft is already pending (id 3)`
  - Status: 409 Conflict — the uncompacted tail is shorter than `compact_min_messages`; body: `chat has {n} uncompacted messages; compaction needs at least {compact_min_messages}`
- **Example Request:**
  ```http
  POST /compaction/draft
  ```
- **Example Response:**
  ```json
  { "draft_id": 3 }
  ```

#### 8.2 List checkpoints and the pending draft

- **URL:** `/compaction`
- **Method:** `GET`
- **Response:**
  - Status: 200 OK
  - Body: `{ "checkpoints": [CheckpointSummary], "pending_draft": PendingDraftSummary | null }`, where `CheckpointSummary` is `{ id, from_message_id, through_message_id, status, trigger, committed_at, needs_merge }` (`status` is one of `"draft"`, `"committed"`, `"discarded"`, `"stale"`; `trigger` is one of `"threshold"`, `"scene_break"`, `"manual"`) and `PendingDraftSummary` is `{ id, from_message_id, through_message_id, created_at, phase }` (`phase` is `"extracting"` while the model is still running, `"review"` once it is ready).
- **Example Request:**
  ```http
  GET /compaction
  ```

#### 8.3 Get a checkpoint's detail

- **URL:** `/compaction/{id}`
- **Method:** `GET`
- **Path Parameters:**
  - `id` (integer): the checkpoint's id.
- **Response:**
  - Status: 200 OK
  - Body: a `CheckpointSummary` (flattened) plus `{ phase, summary_text, rolling_summary, facts, attitude }`. `facts` is every extracted fact, active and rejected alike: `{ id, category, subject, text, quote_speaker, sources, replaces, relation_to, relation, canon, active, superseded_by, rejected_reason }`, with `subject`/`relation_to` flattened to a bare string (`"user"`, `"companion"`, or a person's name). `attitude` is `{ current, rated, blended }`: `current` is the companion's live `CompanionAttitude` toward the user, `rated` is this draft's own narrative rating (`null` while still extracting), `blended` is always `null` until a later issue fills it in.
  - Status: 404 Not Found — `id` does not name a checkpoint.
- **Example Request:**
  ```http
  GET /compaction/3
  ```

#### 8.4 Commit a reviewed draft

- **URL:** `/compaction/{id}/commit`
- **Method:** `POST`
- **Description:** Applies the reviewed edits onto the draft's stored facts, re-validates whatever was touched, and promotes the result: the checkpoint flips to `committed`, its facts become active, and the companion's `compacted_through` advances. Claims the turn slot for the duration, since folding the rolling summary may run the model.
- **Path Parameters:**
  - `id` (integer): the draft's id.
- **Request Body:**
  - `items` (array): `{ id, accepted, text, quote }` per edited or struck fact — `id` is the fact's id from section 8.3's `facts`, `text`/`quote` are optional edits (`quote` for a `rule`/`key_quote` item's verbatim text, `text` for everything else), `accepted: false` strikes the item instead. A fact with no entry here keeps its stored verdict.
  - `summary` (string, optional): overrides the extracted summary; omitted, the stored summary is kept.
- **Response:**
  - Status: 200 OK
  - Body: the committed checkpoint's `CheckpointSummary`.
  - Status: 404 Not Found — `id` does not name a draft.
  - Status: 409 Conflict — a turn is already in flight, or the checkpoint is not a pending draft (already committed/discarded, or still extracting).
  - Status: 422 Unprocessable Entity — an `accepted: true` item still fails validation after the edit; body: an array of `{ item_id, reason }`. Also returned (body `{ needed, budget }`) when the accepted overlay/rule items alone would exceed the compaction token slice.
- **Example Request:**
  ```http
  POST /compaction/3/commit
  Content-Type: application/json

  { "items": [{ "id": 12, "accepted": false }], "summary": null }
  ```

#### 8.5 Discard a draft

- **URL:** `/compaction/{id}/discard`
- **Method:** `POST`
- **Description:** Discards a pending draft, leaving its extracted fact rows exactly as extraction stored them (never rendered, since the checkpoint is no longer `committed`).
- **Path Parameters:**
  - `id` (integer): the draft's id.
- **Response:**
  - Status: 200 OK
  - Status: 404 Not Found — `id` does not name a draft.
  - Status: 409 Conflict — the checkpoint is not a pending draft.
- **Example Request:**
  ```http
  POST /compaction/3/discard
  ```

See section 1.7/1.8 for the pin/unpin routes, which live under `/message/{id}/pin` rather than `/compaction` since they act on a message, not a checkpoint.

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
| `POST` | `/api/compaction/draft` |
| `GET` | `/api/companion` |
| `PUT` | `/api/companion` |
| `POST` | `/api/companion/avatar` |
| `POST` | `/api/companion/card` |
| `GET` | `/api/companion/characterJson` |
| `POST` | `/api/companion/characterJson` |
| `GET` | `/api/compaction` |
| `POST` | `/api/compaction/draft` |
| `GET` | `/api/compaction/{id}` |
| `POST` | `/api/compaction/{id}/commit` |
| `POST` | `/api/compaction/{id}/discard` |
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
| `POST` | `/api/memory/longTerm/rebuild` |
| `DELETE` | `/api/message` |
| `GET` | `/api/message` |
| `POST` | `/api/message` |
| `DELETE` | `/api/message/{id}` |
| `GET` | `/api/message/{id}` |
| `PUT` | `/api/message/{id}` |
| `POST` | `/api/message/{id}/pin` |
| `DELETE` | `/api/message/{id}/pin` |
| `GET` | `/api/multiplayer/participants` |
| `GET` | `/api/multiplayer/participants/{id}/avatar` |
| `GET` | `/api/multiplayer/status` |
| `GET` | `/api/multiplayer/ws` |
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
