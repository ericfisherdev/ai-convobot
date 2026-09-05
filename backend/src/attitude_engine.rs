use std::collections::{HashMap, HashSet};

use crate::database::CompanionAttitude;

/// One of the 20 emotional dimensions tracked per `companion_attitudes` row.
///
/// `column()` is the only place a dimension turns into a SQL identifier, so a
/// caller can never hand a raw, injectable column name to the database layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttitudeDimension {
    Attraction,
    Trust,
    Fear,
    Anger,
    Joy,
    Sorrow,
    Disgust,
    Surprise,
    Curiosity,
    Respect,
    Suspicion,
    Gratitude,
    Jealousy,
    Empathy,
    Lust,
    Love,
    Anxiety,
    Butterflies,
    Submissiveness,
    Dominance,
}

impl AttitudeDimension {
    /// Every dimension, used to iterate the whole attitude for decay.
    pub const ALL: [AttitudeDimension; 20] = [
        AttitudeDimension::Attraction,
        AttitudeDimension::Trust,
        AttitudeDimension::Fear,
        AttitudeDimension::Anger,
        AttitudeDimension::Joy,
        AttitudeDimension::Sorrow,
        AttitudeDimension::Disgust,
        AttitudeDimension::Surprise,
        AttitudeDimension::Curiosity,
        AttitudeDimension::Respect,
        AttitudeDimension::Suspicion,
        AttitudeDimension::Gratitude,
        AttitudeDimension::Jealousy,
        AttitudeDimension::Empathy,
        AttitudeDimension::Lust,
        AttitudeDimension::Love,
        AttitudeDimension::Anxiety,
        AttitudeDimension::Butterflies,
        AttitudeDimension::Submissiveness,
        AttitudeDimension::Dominance,
    ];

    /// SQL column name backing this dimension in `companion_attitudes`.
    pub fn column(self) -> &'static str {
        match self {
            AttitudeDimension::Attraction => "attraction",
            AttitudeDimension::Trust => "trust",
            AttitudeDimension::Fear => "fear",
            AttitudeDimension::Anger => "anger",
            AttitudeDimension::Joy => "joy",
            AttitudeDimension::Sorrow => "sorrow",
            AttitudeDimension::Disgust => "disgust",
            AttitudeDimension::Surprise => "surprise",
            AttitudeDimension::Curiosity => "curiosity",
            AttitudeDimension::Respect => "respect",
            AttitudeDimension::Suspicion => "suspicion",
            AttitudeDimension::Gratitude => "gratitude",
            AttitudeDimension::Jealousy => "jealousy",
            AttitudeDimension::Empathy => "empathy",
            AttitudeDimension::Lust => "lust",
            AttitudeDimension::Love => "love",
            AttitudeDimension::Anxiety => "anxiety",
            AttitudeDimension::Butterflies => "butterflies",
            AttitudeDimension::Submissiveness => "submissiveness",
            AttitudeDimension::Dominance => "dominance",
        }
    }

    /// Current value of this dimension on the given attitude row.
    pub fn value_of(self, attitude: &CompanionAttitude) -> f32 {
        match self {
            AttitudeDimension::Attraction => attitude.attraction,
            AttitudeDimension::Trust => attitude.trust,
            AttitudeDimension::Fear => attitude.fear,
            AttitudeDimension::Anger => attitude.anger,
            AttitudeDimension::Joy => attitude.joy,
            AttitudeDimension::Sorrow => attitude.sorrow,
            AttitudeDimension::Disgust => attitude.disgust,
            AttitudeDimension::Surprise => attitude.surprise,
            AttitudeDimension::Curiosity => attitude.curiosity,
            AttitudeDimension::Respect => attitude.respect,
            AttitudeDimension::Suspicion => attitude.suspicion,
            AttitudeDimension::Gratitude => attitude.gratitude,
            AttitudeDimension::Jealousy => attitude.jealousy,
            AttitudeDimension::Empathy => attitude.empathy,
            AttitudeDimension::Lust => attitude.lust,
            AttitudeDimension::Love => attitude.love,
            AttitudeDimension::Anxiety => attitude.anxiety,
            AttitudeDimension::Butterflies => attitude.butterflies,
            AttitudeDimension::Submissiveness => attitude.submissiveness,
            AttitudeDimension::Dominance => attitude.dominance,
        }
    }

    /// Value of this dimension on a persisted `database::AttitudeDelta`.
    ///
    /// Lets a caller walk a stored delta dimension by dimension without
    /// repeating the 20-arm match.
    pub fn value_of_delta(self, delta: &crate::database::AttitudeDelta) -> f32 {
        match self {
            AttitudeDimension::Attraction => delta.attraction,
            AttitudeDimension::Trust => delta.trust,
            AttitudeDimension::Fear => delta.fear,
            AttitudeDimension::Anger => delta.anger,
            AttitudeDimension::Joy => delta.joy,
            AttitudeDimension::Sorrow => delta.sorrow,
            AttitudeDimension::Disgust => delta.disgust,
            AttitudeDimension::Surprise => delta.surprise,
            AttitudeDimension::Curiosity => delta.curiosity,
            AttitudeDimension::Respect => delta.respect,
            AttitudeDimension::Suspicion => delta.suspicion,
            AttitudeDimension::Gratitude => delta.gratitude,
            AttitudeDimension::Jealousy => delta.jealousy,
            AttitudeDimension::Empathy => delta.empathy,
            AttitudeDimension::Lust => delta.lust,
            AttitudeDimension::Love => delta.love,
            AttitudeDimension::Anxiety => delta.anxiety,
            AttitudeDimension::Butterflies => delta.butterflies,
            AttitudeDimension::Submissiveness => delta.submissiveness,
            AttitudeDimension::Dominance => delta.dominance,
        }
    }
}

/// A signed change to apply to one dimension of an attitude row.
///
/// This is the engine's own delta type, distinct from `database::AttitudeDelta`
/// (the 14-dimension wire format persisted into `attitude_memories.attitude_delta_json`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DimensionDelta {
    pub dimension: AttitudeDimension,
    pub delta: f32,
}

/// Turns a conversation turn into per-dimension deltas.
///
/// The seam that lets the lexicon-based `LexiconScorer` below be swapped for an
/// LLM-backed scorer later without touching any caller.
pub trait TurnScorer {
    fn evaluate_turn(
        &self,
        user_message: &str,
        companion_reply: &str,
        current: &CompanionAttitude,
    ) -> Vec<DimensionDelta>;
}

/// Tuning knobs shared by every `TurnScorer` implementation.
pub struct ScorerConfig {
    /// Largest magnitude a single turn may move any one dimension by.
    pub max_delta_per_turn: f32,
    /// Magnitude a dimension drifts toward `baseline` per turn when no cue touched it.
    pub decay_step: f32,
    /// Persona-adjusted resting attitude that untouched dimensions decay toward.
    pub baseline: CompanionAttitude,
}

impl ScorerConfig {
    pub fn new(baseline: CompanionAttitude) -> Self {
        Self {
            max_delta_per_turn: 5.0,
            decay_step: 0.5,
            baseline,
        }
    }
}

type Weights = &'static [(AttitudeDimension, f32)];

const PRAISE_CUES: &[(&str, Weights)] = &[
    (
        "wonderful",
        &[
            (AttitudeDimension::Joy, 3.0),
            (AttitudeDimension::Respect, 2.0),
        ],
    ),
    (
        "amazing",
        &[
            (AttitudeDimension::Joy, 3.0),
            (AttitudeDimension::Respect, 2.0),
        ],
    ),
    (
        "awesome",
        &[
            (AttitudeDimension::Joy, 3.0),
            (AttitudeDimension::Respect, 2.0),
        ],
    ),
    (
        "great",
        &[
            (AttitudeDimension::Joy, 2.0),
            (AttitudeDimension::Respect, 1.0),
        ],
    ),
    (
        "kind",
        &[
            (AttitudeDimension::Joy, 1.0),
            (AttitudeDimension::Respect, 2.0),
            (AttitudeDimension::Gratitude, 1.0),
        ],
    ),
    (
        "sweet",
        &[
            (AttitudeDimension::Joy, 2.0),
            (AttitudeDimension::Gratitude, 1.0),
        ],
    ),
    (
        "brilliant",
        &[
            (AttitudeDimension::Joy, 2.0),
            (AttitudeDimension::Respect, 3.0),
        ],
    ),
    ("impressive", &[(AttitudeDimension::Respect, 3.0)]),
    (
        "perfect",
        &[
            (AttitudeDimension::Joy, 2.0),
            (AttitudeDimension::Respect, 2.0),
        ],
    ),
];

const HOSTILITY_CUES: &[(&str, Weights)] = &[
    (
        "useless",
        &[
            (AttitudeDimension::Anger, 3.0),
            (AttitudeDimension::Trust, -2.0),
            (AttitudeDimension::Respect, -2.0),
            (AttitudeDimension::Suspicion, 1.0),
        ],
    ),
    (
        "stupid",
        &[
            (AttitudeDimension::Anger, 3.0),
            (AttitudeDimension::Respect, -3.0),
        ],
    ),
    (
        "idiot",
        &[
            (AttitudeDimension::Anger, 3.0),
            (AttitudeDimension::Respect, -3.0),
        ],
    ),
    (
        "hate",
        &[
            (AttitudeDimension::Anger, 3.0),
            (AttitudeDimension::Trust, -2.0),
        ],
    ),
    (
        "worthless",
        &[
            (AttitudeDimension::Anger, 2.0),
            (AttitudeDimension::Trust, -2.0),
            (AttitudeDimension::Respect, -2.0),
        ],
    ),
    (
        "pathetic",
        &[
            (AttitudeDimension::Anger, 2.0),
            (AttitudeDimension::Respect, -2.0),
        ],
    ),
    (
        "dumb",
        &[
            (AttitudeDimension::Anger, 2.0),
            (AttitudeDimension::Respect, -2.0),
        ],
    ),
    (
        "annoying",
        &[
            (AttitudeDimension::Anger, 2.0),
            (AttitudeDimension::Trust, -1.0),
        ],
    ),
    (
        "awful",
        &[
            (AttitudeDimension::Anger, 2.0),
            (AttitudeDimension::Trust, -1.0),
        ],
    ),
];

const AFFECTION_CUES: &[(&str, Weights)] = &[
    (
        "adore",
        &[
            (AttitudeDimension::Attraction, 3.0),
            (AttitudeDimension::Love, 2.0),
            (AttitudeDimension::Butterflies, 1.0),
        ],
    ),
    (
        "gorgeous",
        &[
            (AttitudeDimension::Attraction, 3.0),
            (AttitudeDimension::Butterflies, 2.0),
        ],
    ),
    (
        "beautiful",
        &[
            (AttitudeDimension::Attraction, 2.0),
            (AttitudeDimension::Butterflies, 1.0),
        ],
    ),
    ("attractive", &[(AttitudeDimension::Attraction, 2.0)]),
    (
        "cute",
        &[
            (AttitudeDimension::Attraction, 1.0),
            (AttitudeDimension::Butterflies, 1.0),
        ],
    ),
];

const GRATITUDE_CUES: &[(&str, Weights)] = &[
    (
        "thank you",
        &[
            (AttitudeDimension::Gratitude, 3.0),
            (AttitudeDimension::Trust, 1.0),
        ],
    ),
    (
        "thanks",
        &[
            (AttitudeDimension::Gratitude, 2.0),
            (AttitudeDimension::Trust, 1.0),
        ],
    ),
    ("grateful", &[(AttitudeDimension::Gratitude, 3.0)]),
    (
        "appreciate",
        &[
            (AttitudeDimension::Gratitude, 2.0),
            (AttitudeDimension::Trust, 1.0),
        ],
    ),
];

const CUE_TABLES: &[&[(&str, Weights)]] =
    &[PRAISE_CUES, HOSTILITY_CUES, AFFECTION_CUES, GRATITUDE_CUES];

/// Lowercases and splits on non-alphanumeric runs so punctuation never fuses
/// into a word and matching stays whole-word without a `regex` dependency.
fn words_of(message: &str) -> Vec<String> {
    message
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

/// True if `phrase`'s words appear as a contiguous run inside `words`.
fn contains_phrase(words: &[String], phrase: &str) -> bool {
    let phrase_words: Vec<&str> = phrase.split_whitespace().collect();
    if phrase_words.is_empty() || phrase_words.len() > words.len() {
        return false;
    }
    words.windows(phrase_words.len()).any(|window| {
        window
            .iter()
            .map(String::as_str)
            .eq(phrase_words.iter().copied())
    })
}

/// Scores a turn by matching a fixed lexicon of cue words/phrases against the
/// user's message, then decays every untouched dimension toward `config.baseline`.
///
/// A future LLM-backed `TurnScorer` can replace this without any caller change.
pub struct LexiconScorer {
    config: ScorerConfig,
}

impl LexiconScorer {
    pub fn new(config: ScorerConfig) -> Self {
        Self { config }
    }

    /// Sums cue weights per dimension for every cue that hits `user_message`.
    fn cue_deltas(&self, user_message: &str) -> HashMap<AttitudeDimension, f32> {
        let words = words_of(user_message);
        let word_set: HashSet<&str> = words.iter().map(String::as_str).collect();
        let mut sums: HashMap<AttitudeDimension, f32> = HashMap::new();

        for table in CUE_TABLES {
            for (cue, weights) in *table {
                let matched = if cue.contains(' ') {
                    contains_phrase(&words, cue)
                } else {
                    word_set.contains(cue)
                };
                if matched {
                    for (dimension, weight) in *weights {
                        *sums.entry(*dimension).or_insert(0.0) += weight;
                    }
                }
            }
        }

        // Curiosity about the companion: a question aimed at "you"/"your".
        if user_message.contains('?') && (word_set.contains("you") || word_set.contains("your")) {
            *sums.entry(AttitudeDimension::Curiosity).or_insert(0.0) += 2.0;
            *sums.entry(AttitudeDimension::Joy).or_insert(0.0) += 1.0;
        }

        sums
    }
}

impl TurnScorer for LexiconScorer {
    fn evaluate_turn(
        &self,
        user_message: &str,
        _companion_reply: &str,
        current: &CompanionAttitude,
    ) -> Vec<DimensionDelta> {
        let cue_sums = self.cue_deltas(user_message);
        let mut deltas = Vec::new();

        for dimension in AttitudeDimension::ALL {
            let current_value = dimension.value_of(current);

            let raw_delta = match cue_sums.get(&dimension) {
                Some(&sum) => sum.clamp(
                    -self.config.max_delta_per_turn,
                    self.config.max_delta_per_turn,
                ),
                None => {
                    let baseline_value = dimension.value_of(&self.config.baseline);
                    let diff = baseline_value - current_value;
                    if diff > 0.0 {
                        diff.min(self.config.decay_step)
                    } else if diff < 0.0 {
                        diff.max(-self.config.decay_step)
                    } else {
                        0.0
                    }
                }
            };

            // Never let a delta push the dimension outside -100..=100.
            let clamped_delta = raw_delta.clamp(-100.0 - current_value, 100.0 - current_value);

            if clamped_delta != 0.0 {
                deltas.push(DimensionDelta {
                    dimension,
                    delta: clamped_delta,
                });
            }
        }

        deltas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a `CompanionAttitude` with every dimension at `value` so tests
    /// only need to override the fields they care about via struct update.
    fn attitude_with(value: f32) -> CompanionAttitude {
        CompanionAttitude {
            id: None,
            companion_id: 1,
            target_id: 1,
            target_type: "user".to_string(),
            attraction: value,
            trust: value,
            fear: value,
            anger: value,
            joy: value,
            sorrow: value,
            disgust: value,
            surprise: value,
            curiosity: value,
            respect: value,
            suspicion: value,
            gratitude: value,
            jealousy: value,
            empathy: value,
            lust: value,
            love: value,
            anxiety: value,
            butterflies: value,
            submissiveness: value,
            dominance: value,
            relationship_score: Some(0.0),
            last_updated: "now".to_string(),
            created_at: "now".to_string(),
        }
    }

    fn scorer_with_baseline(baseline: CompanionAttitude) -> LexiconScorer {
        LexiconScorer::new(ScorerConfig::new(baseline))
    }

    fn delta_for(deltas: &[DimensionDelta], dimension: AttitudeDimension) -> Option<f32> {
        deltas
            .iter()
            .find(|d| d.dimension == dimension)
            .map(|d| d.delta)
    }

    #[test]
    fn positive_message_raises_joy_and_respect() {
        let current = attitude_with(40.0);
        let scorer = scorer_with_baseline(current.clone());
        let deltas = scorer.evaluate_turn("You're wonderful, thank you so much!", "", &current);

        assert!(delta_for(&deltas, AttitudeDimension::Joy).unwrap_or(0.0) > 0.0);
        assert!(delta_for(&deltas, AttitudeDimension::Respect).unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn hostile_message_raises_anger_and_lowers_trust() {
        let current = attitude_with(40.0);
        let scorer = scorer_with_baseline(current.clone());
        let deltas = scorer.evaluate_turn("You're useless and I don't trust you", "", &current);

        assert!(delta_for(&deltas, AttitudeDimension::Anger).unwrap_or(0.0) > 0.0);
        assert!(delta_for(&deltas, AttitudeDimension::Trust).unwrap_or(0.0) < 0.0);
    }

    #[test]
    fn neutral_message_produces_only_decay_deltas() {
        let baseline = attitude_with(40.0);
        let mut current = baseline.clone();
        current.joy = 55.0;
        current.anger = 25.0;
        let scorer = scorer_with_baseline(baseline);

        let deltas = scorer.evaluate_turn("The weather is fine today.", "", &current);

        assert_eq!(deltas.len(), 2);
        assert!(delta_for(&deltas, AttitudeDimension::Joy).unwrap() < 0.0);
        assert!(delta_for(&deltas, AttitudeDimension::Anger).unwrap() > 0.0);
    }

    #[test]
    fn dimension_above_baseline_decays_downward_and_below_decays_upward() {
        let baseline = attitude_with(40.0);
        let mut current = baseline.clone();
        current.trust = 60.0;
        current.fear = 10.0;
        let scorer = scorer_with_baseline(baseline);

        let deltas = scorer.evaluate_turn("neutral text", "", &current);

        let trust_delta = delta_for(&deltas, AttitudeDimension::Trust).unwrap();
        let fear_delta = delta_for(&deltas, AttitudeDimension::Fear).unwrap();
        assert_eq!(trust_delta, -0.5);
        assert_eq!(fear_delta, 0.5);
    }

    #[test]
    fn many_cue_hits_still_cap_at_max_delta_per_turn() {
        let current = attitude_with(40.0);
        let scorer = scorer_with_baseline(current.clone());

        let deltas = scorer.evaluate_turn(
            "wonderful amazing awesome great brilliant perfect",
            "",
            &current,
        );

        let joy_delta = delta_for(&deltas, AttitudeDimension::Joy).unwrap();
        assert_eq!(joy_delta, 5.0);
    }

    #[test]
    fn value_at_max_receives_no_positive_delta() {
        let mut current = attitude_with(40.0);
        current.joy = 100.0;
        let scorer = scorer_with_baseline(current.clone());

        let deltas = scorer.evaluate_turn("wonderful, amazing!", "", &current);

        assert_eq!(delta_for(&deltas, AttitudeDimension::Joy), None);
    }
}
