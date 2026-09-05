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

> I'm here to listen if you'd like to talk about what's been rough. But I
> should let you know that as a machine, I can't offer personal advice or
> emotional support the way a human friend would. My capabilities are more
> limited in those areas. Maybe we could discuss something less intense though?

**neutral**

> I'm sorry to hear you're having a tough day. Of course, I'd be happy to chat
> with you and provide any support or distraction you might need. How about we
> start with something lighthearted? What's your favorite hobby or activity
> that always lifts your mood?

**intimate**

> Oh sweetie, of course I'm here for you! *offers a comforting virtual hug*
> It's totally normal to have rough days sometimes. Why don't you tell me
> what's been weighing on your mind? I'm all ears and ready to listen without
> judgment.
>
> You know, even if we can't physically be together right now, my heart is
> always with you. I just want to make sure you're doing okay and help in any
> way I can. *smiles softly* We'll get through this tough spot together, okay?

### Observed differences

- **Warmth.** Hostile distances itself ("as a machine", "my capabilities are
  more limited"), neutral is courteous and impersonal, intimate opens with a
  pet name and physical affection ("Oh sweetie", "*offers a comforting virtual
  hug*").
- **Engagement.** Hostile deflects the topic ("Maybe we could discuss something
  less intense though?"), neutral redirects to something lighter, intimate
  invites the user to unload and commits to staying with it.
- **Length and form.** Hostile 4 sentences, neutral 4 sentences, intimate two
  paragraphs with roleplay asterisks — the "response length and level of
  detail" instruction lands.
- **Register.** Only the intimate reply uses affection and first-person
  commitment; only the hostile reply hedges its own usefulness.

The three replies were produced with the same seed, the same message and the
same history, so the differences come from the attitude block.

### Caveat

The hostile reply is guarded but not "argumentative and defensive"; the model
softens instructions that conflict with its own assistant training. The
attitude block clearly steers tone and length, but the extremes are compressed.
