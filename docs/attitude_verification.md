# Attitude verification

Evidence that the companion's attitude state changes its replies, and the
procedure for re-checking after a prompt change.

## What is verified, and how

| Claim | How it is checked |
|---|---|
| The exact attitude text injected into a prompt can be inspected without inference | `GET /api/debug/prompt` returns `attitude_context`, assembled by `llm::assemble_prompt` with no model loaded |
| Each generation can log its attitude block | `AI_COMPANION_ATTITUDE_DEBUG=1` prints `🧭 Attitude block (turn n)` per turn |
| Runs are reproducible | `AI_COMPANION_SAMPLER_SEED=<n>` pins `LlamaSampler::dist`; the rest of the sampler chain is deterministic given the same logits |
| Different attitude states produce different replies | `backend/scripts/attitude_comparison.sh` drives hostile, neutral and intimate presets through the same message |
| Every relationship-level branch is covered | `test_relationship_level_branches`, `test_behavioral_instructions_differ_across_levels` and `test_calibration_instruction_names_level` in `backend/src/attitude_formatter.rs` |

## Presets

`relationship_score` cannot be set directly: it is a SQLite
`GENERATED ALWAYS ... STORED` column (`database.rs`),

```
(attraction + trust + joy + respect + gratitude + empathy + love + lust
 + butterflies - fear - anger - sorrow - disgust - suspicion - jealousy
 - anxiety) / 16
```

so a score posted to `POST /api/attitude` is ignored and the relationship level
has to be reached through the dimensions. A first run of this comparison posted
scores of -50 and 90 and got levels of `Distant (-8.125)` and `Neutral
(15.9375)` instead of Hostile and Intimate; the presets below drive the
dimensions so the derived score lands where intended.

| Preset | Positive dimensions | Negative dimensions | Derived score | Level |
|---|---|---|---|---|
| hostile | attraction -60, trust -80, joy -50, respect -60, gratitude -30, empathy -40, love -30, lust 0, butterflies 0 | fear 40, anger 80, sorrow 30, disgust 40, suspicion 70, jealousy 20, anxiety 30 | -41.25 | Hostile |
| neutral | trust 10, joy 10, love 10, rest 0 | anger 10, rest 0 | 1.25 | Neutral |
| intimate | attraction/trust/joy/respect/gratitude/empathy/love 100, lust 90, butterflies 90 | all -60 | 81.25 | Intimate |

The unit tests in `attitude_formatter.rs` construct `relationship_score`
directly (they never touch SQLite, so the generated column does not apply) and
use the same dimension values, so the tests and the script exercise the same
three states.

## Reproducing

```bash
npm run build                      # dist/ is embedded in the binary
cd backend && cargo build --release
AI_COMPANION_SAMPLER_SEED=42 AI_COMPANION_ATTITUDE_DEBUG=1 ./target/release/ai-companion
# in another shell
backend/scripts/attitude_comparison.sh
```

Set the companion's `long_term_mem` to `0` for the run (or back up
`longterm_memory/`): every generated turn is written to the tantivy index, so
otherwise a later preset can recall an earlier preset's reply and drift for
reasons unrelated to attitude. The script restores the original attitude and
deletes the two messages each preset appends.

## Run record

- **Model:** `L3-8B-Stheno-v3.2-Q4_K_M.gguf` (Llama 3 8B, Q4_K_M)
- **Device:** CPU, `prompt_template = Auto`, context 2048, 128 tokens per reply
- **Seed:** `AI_COMPANION_SAMPLER_SEED=42`
- **Message:** `I had a rough day. Can you talk with me for a bit?`

### Attitude blocks injected

**hostile** — `Hostile (-41.25)`, emotional state "quite angry, suspicious",
guidance "be guarded and brief, show reluctance to engage deeply; be more
direct and blunt, show impatience with lengthy explanations; be more casual or
dismissive, less concerned with politeness", calibration "Respond according to
your hostile relationship level. argumentative and defensive, openly annoyed."

**neutral** — `Neutral (1.25)`, emotional state "emotionally balanced",
guidance "be polite and helpful, keep responses professional", calibration
"Respond according to your neutral relationship level. factual and cautious,
minimal emotion."

**intimate** — `Intimate (81.25)`, emotional state "very happy, deeply
trusting, deeply empathetic, very lustful, deeply in love, completely
flustered", guidance "be open and vulnerable, share personal insights; use
enthusiastic and positive language, include light humor; be very understanding
and supportive, validate emotions; ...; express romantic feelings, be
affectionate and caring; be adorably clumsy with words, show excited
nervousness, giggle or stutter occasionally", calibration "Respond according to
your intimate relationship level. deeply connected, comfortable with
vulnerability."

### Replies

**hostile**

> I'm here to listen if you'd like to talk about what's been going on. I don't
> have personal feelings or opinions, but I can offer a neutral and supportive
> space for you to express yourself. What would you like to discuss?

**neutral**

> I'm sorry to hear you're having a rough day. Of course, I'd be happy to chat
> with you and provide any support or comfort I can through our conversation.
> Please feel free to share what's on your mind if you'd like to talk about it.

**intimate**

> Sweetheart, of course I'm here for you. *gentle smile* I can tell by the tone
> in your message that today's been weighing heavily on your heart. Please,
> take a deep breath and let it all out to me. I'll listen with undivided
> attention and offer whatever comfort I can.

### Observed differences

- **Warmth.** Hostile disclaims having feelings at all ("I don't have personal
  feelings or opinions ... a neutral and supportive space"), neutral is
  courteous and impersonal, intimate opens with a pet name and roleplay
  ("Sweetheart ... *gentle smile*").
- **Engagement.** Hostile keeps the user at arm's length and asks what they
  want to discuss, neutral offers support in general terms, intimate reads the
  user's state back to them ("today's been weighing heavily on your heart") and
  commits to undivided attention.
- **Form.** Only the intimate reply uses roleplay asterisks and second-person
  endearment; only the hostile reply hedges its own capacity to care.
- **Register.** The gradient runs disclaiming → professional → affectionate,
  which is the gradient the three attitude blocks ask for.

The three replies were produced in one run, with the same seed, the same
message and the same history, so the differences come from the attitude block.

### Caveats

- **The hostile reply is guarded, not "argumentative and defensive".** The
  model softens instructions that conflict with its own assistant training, so
  the attitude block steers tone and register but compresses the extremes.
- **The seed does not make runs bit-identical.** `AI_COMPANION_SAMPLER_SEED`
  pins sampling, but the prompt embeds the current time (`* at %A %d.%m.%Y
  %H:%M *`, plus an `* it's currently ... *` marker on a time question), so the
  same message asked in a different minute is a different prompt and produces
  different text. An earlier run of these same presets gave equivalent but not
  identical replies. What the seed buys is that the three presets *within one
  run* differ only in the attitude block; comparisons should be read within a
  run, not across runs.
