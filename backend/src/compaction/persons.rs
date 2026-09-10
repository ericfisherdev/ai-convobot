//! Turns extracted `Person` facts into `third_party_individuals` rows at
//! commit time (#177), replacing the pre-compaction heuristic detector as
//! the trusted source of third-party people. Runs as a
//! [`crate::compaction::commit::CommitObserver`], so a failure here can
//! never fail the commit that produced it.
//!
//! [`NON_NAME_WORDS`] is the single stop-word list `Database::is_likely_person_name`
//! (the heuristic detector) and [`is_junk_person_name`] (this module's own
//! filter, applied to a canon-validated `Person` fact's name) both check
//! against, so the two lists can never drift apart.
//!
//! [`importance_from_mentions`], [`is_junk_person_name`] and [`plan_upserts`]
//! are pure: no `Database`, no I/O. [`PersonSink`] is the persistence seam
//! [`PersonsObserver`] runs the plan through: [`SqlitePersonSink`] forwards
//! to `Database::upsert_compaction_person`, the same associated function the
//! `#[cfg(test)]` `open_at`-backed tests in `database.rs` exercise directly.

use crate::compaction::commit::CommitObserver;
use crate::compaction::types::{Checkpoint, Fact, FactCategory, FactSubject};
use crate::database::{Database, PersonUpsert};
use std::collections::HashMap;

/// Common English words and the "invalid" words the pre-#177 heuristic
/// detector filtered out of `third_party_individuals`, moved here from
/// `Database::is_likely_person_name` so it and [`is_junk_person_name`] share
/// one list instead of two that could drift apart.
pub(crate) const NON_NAME_WORDS: &[&str] = &[
    // Original words
    "the",
    "and",
    "or",
    "but",
    "if",
    "when",
    "where",
    "what",
    "who",
    "how",
    "why",
    "this",
    "that",
    "these",
    "those",
    "here",
    "there",
    "now",
    "then",
    "today",
    "tomorrow",
    "yesterday",
    "said",
    "told",
    "asked",
    "mentioned",
    "think",
    "know",
    // Body parts
    "hand",
    "hands",
    "shoulder",
    "shoulders",
    "head",
    "heads",
    "arm",
    "arms",
    "leg",
    "legs",
    "foot",
    "feet",
    "eye",
    "eyes",
    "ear",
    "ears",
    "nose",
    "mouth",
    "face",
    "hair",
    "neck",
    "back",
    "chest",
    "stomach",
    "knee",
    "knees",
    "elbow",
    "elbows",
    "finger",
    "fingers",
    "thumb",
    "thumbs",
    "toe",
    "toes",
    "ankle",
    "ankles",
    "wrist",
    "wrists",
    "hip",
    "hips",
    "body",
    "skin",
    "bone",
    "bones",
    "muscle",
    "muscles",
    // Common objects
    "class",
    "classes",
    "book",
    "books",
    "table",
    "tables",
    "chair",
    "chairs",
    "door",
    "doors",
    "window",
    "windows",
    "desk",
    "desks",
    "computer",
    "computers",
    "phone",
    "phones",
    "car",
    "cars",
    "house",
    "houses",
    "room",
    "rooms",
    "wall",
    "walls",
    "floor",
    "floors",
    "ceiling",
    "ceilings",
    "roof",
    "roofs",
    "street",
    "streets",
    "road",
    "roads",
    "building",
    "buildings",
    "office",
    "offices",
    // Abstract concepts and common words
    "should",
    "could",
    "would",
    "must",
    "might",
    "may",
    "can",
    "will",
    "shall",
    "thing",
    "things",
    "stuff",
    "matter",
    "matters",
    "way",
    "ways",
    "time",
    "times",
    "place",
    "places",
    "work",
    "works",
    "play",
    "plays",
    "run",
    "runs",
    "walk",
    "walks",
    "talk",
    "talks",
    "look",
    "looks",
    "feel",
    "feels",
    "want",
    "wants",
    "need",
    "needs",
    "use",
    "uses",
    "make",
    "makes",
    "take",
    "takes",
    "give",
    "gives",
    "get",
    "gets",
    "keep",
    "keeps",
    "let",
    "lets",
    "help",
    "helps",
    "show",
    "shows",
    "try",
    "tries",
    // Nature and environment
    "tree",
    "trees",
    "plant",
    "plants",
    "flower",
    "flowers",
    "grass",
    "ground",
    "sky",
    "sun",
    "moon",
    "star",
    "stars",
    "cloud",
    "clouds",
    "rain",
    "snow",
    "wind",
    "air",
    "water",
    "fire",
    "earth",
    "stone",
    "stones",
    "rock",
    "rocks",
    // Common activities/states
    "sleep",
    "wake",
    "eat",
    "drink",
    "sit",
    "stand",
    "lie",
    "move",
    "stop",
    "start",
    "end",
    "begin",
    "open",
    "close",
    "break",
    "fix",
    "clean",
    "wash",
    "dry",
    "cut",
    // Pronouns and determiners
    "it",
    "its",
    "them",
    "their",
    "theirs",
    "some",
    "any",
    "all",
    "each",
    "every",
    "few",
    "many",
    "much",
    "more",
    "most",
    "less",
    "least",
    "other",
    "another",
    "such",
    "own",
    "same",
    "different",
    "various",
    "several",
    "both",
    "either",
    "neither",
];

/// Pronouns [`is_junk_person_name`] rejects that [`NON_NAME_WORDS`] does not
/// already cover (`its`/`it`/`them`/`their`/`theirs` are already in that
/// list) — the capitalised-pronoun rows (`Her`, `You`, `His`, ...) the old
/// heuristic detector used to leave behind in `third_party_individuals`.
const PRONOUNS: &[&str] = &[
    "he", "she", "they", "him", "her", "his", "hers", "you", "your", "yours", "me", "my", "mine",
    "we", "us", "our", "ours", "i",
];

/// Shortest name length [`is_junk_person_name`] accepts. Below this, a
/// canon-validated `Person` fact's name is more likely an initial or a
/// fragment than an actual name.
const MIN_NAME_CHARS: usize = 3;

/// True when `name` is unfit to become a `third_party_individuals` row: a
/// pronoun, a common English stop word ([`NON_NAME_WORDS`]), or shorter than
/// [`MIN_NAME_CHARS`] characters. Case-insensitive; counts characters, not
/// bytes, so a short multi-byte name is measured correctly.
pub(crate) fn is_junk_person_name(name: &str) -> bool {
    let trimmed = name.trim();
    let lower = trimmed.to_lowercase();

    if trimmed.chars().count() < MIN_NAME_CHARS {
        return true;
    }
    PRONOUNS.contains(&lower.as_str()) || NON_NAME_WORDS.contains(&lower.as_str())
}

/// Importance score a compaction-sourced person is seeded/updated with,
/// derived from how many times they have been mentioned across committed
/// checkpoints. Monotonically increasing in `mentions` and always inside
/// the `third_party_individuals.importance_score` `CHECK(0..=1)` bound, so a
/// caller never needs to clamp again before storing it.
pub(crate) fn importance_from_mentions(mentions: i32) -> f32 {
    let mentions = mentions.max(1) as f32;
    (0.4 + 0.1 * mentions.ln()).clamp(0.0, 1.0)
}

/// Builds one [`PersonUpsert`] per distinct person named across `facts`'
/// active `Person` items, merging facts that (after whitespace
/// normalisation and case-insensitive comparison) name the same person into
/// a single upsert, so one checkpoint with several mentions of the same
/// person writes one row update, not several. Drops facts naming the user
/// or the companion (mirrors the user-name skip in
/// `Database::detect_new_persons_in_message`, extended here to the
/// companion too) and facts whose name is [`is_junk_person_name`]. Pure: no
/// `Database`, no I/O.
pub(crate) fn plan_upserts(
    facts: &[Fact],
    user_name: &str,
    companion_name: &str,
) -> Vec<PersonUpsert> {
    let user_name_lower = user_name.trim().to_lowercase();
    let companion_name_lower = companion_name.trim().to_lowercase();

    // Keyed by the normalised (lowercased) name so "Alice" and "alice"
    // within the same batch merge into one upsert; the stored `name` keeps
    // the first-seen casing.
    let mut by_name: HashMap<String, PersonUpsert> = HashMap::new();
    let mut order: Vec<String> = Vec::new();

    for fact in facts {
        if fact.category != FactCategory::Person || !fact.active {
            continue;
        }
        let Some(FactSubject::Person(raw_name)) = &fact.subject else {
            continue;
        };
        let name = raw_name.split_whitespace().collect::<Vec<_>>().join(" ");
        if name.is_empty() {
            continue;
        }
        let name_lower = name.to_lowercase();
        if name_lower == user_name_lower || name_lower == companion_name_lower {
            continue;
        }
        if is_junk_person_name(&name) {
            continue;
        }

        let (relationship_to_user, relationship_to_companion) = match &fact.relation_to {
            Some(FactSubject::User) => (fact.relation.clone(), None),
            Some(FactSubject::Companion) => (None, fact.relation.clone()),
            _ => (None, None),
        };

        let entry = by_name.entry(name_lower.clone()).or_insert_with(|| {
            order.push(name_lower.clone());
            PersonUpsert {
                name: name.clone(),
                relationship_to_user: None,
                relationship_to_companion: None,
                mentions: 0,
            }
        });
        entry.mentions += fact.sources.len().max(1) as i32;
        if entry.relationship_to_user.is_none() {
            entry.relationship_to_user = relationship_to_user;
        }
        if entry.relationship_to_companion.is_none() {
            entry.relationship_to_companion = relationship_to_companion;
        }
    }

    order
        .into_iter()
        .filter_map(|key| by_name.remove(&key))
        .collect()
}

/// The persistence seam [`PersonsObserver`] runs each planned upsert
/// through, mirroring `chat_turn::TurnStore`: lets the observer's
/// commit-time logic be unit-tested against an in-memory fake instead of
/// `companion_database.db`.
pub trait PersonSink {
    fn upsert(&self, p: &PersonUpsert) -> rusqlite::Result<i32>;
}

/// The production [`PersonSink`]: forwards to `Database::upsert_compaction_person`.
pub struct SqlitePersonSink;

impl PersonSink for SqlitePersonSink {
    fn upsert(&self, p: &PersonUpsert) -> rusqlite::Result<i32> {
        Database::upsert_compaction_person(p)
    }
}

/// Creates or updates `third_party_individuals` rows from a checkpoint's
/// newly active `Person` facts. Registered into
/// [`crate::compaction::CommitDeps`] by
/// [`crate::compaction::production_commit_deps`].
///
/// `user_name`/`companion_name` are read once, at construction
/// ([`PersonsObserver::from_database`]) rather than inside `on_committed`,
/// mirroring `AttitudeRecalibrator::from_config`'s "read what the observer
/// needs up front" shape — it also means `on_committed` itself never touches
/// `Database` directly, only through `sink`, so it can be unit-tested
/// against [`RecordingSink`] without a database.
///
/// A per-person upsert failure is logged and skipped rather than
/// propagated: the checkpoint's facts are already durable by the time this
/// runs, and one bad row must never be mistaken for a failed commit, the
/// same rule `chat_turn::finish_turn` documents for the attitude-scoring
/// path.
pub struct PersonsObserver<S: PersonSink> {
    sink: S,
    user_name: String,
    companion_name: String,
}

impl<S: PersonSink> PersonsObserver<S> {
    pub fn new(sink: S, user_name: String, companion_name: String) -> Self {
        Self {
            sink,
            user_name,
            companion_name,
        }
    }
}

impl PersonsObserver<SqlitePersonSink> {
    /// The production constructor: loads the user and companion names once.
    /// Returns `None` (logged) rather than propagating when either read
    /// fails, matching `production_commit_deps`'s "log and skip the push"
    /// convention for a constructor that can fail.
    pub fn from_database() -> Option<PersonsObserver<SqlitePersonSink>> {
        let user_name = match Database::get_user_data() {
            Ok(user) => user.name,
            Err(e) => {
                eprintln!(
                    "compaction: failed to load user name for persons observer, skipping this session: {e}"
                );
                return None;
            }
        };
        let companion_name = match Database::get_companion_data() {
            Ok(companion) => companion.name,
            Err(e) => {
                eprintln!(
                    "compaction: failed to load companion name for persons observer, skipping this session: {e}"
                );
                return None;
            }
        };
        Some(PersonsObserver::new(
            SqlitePersonSink,
            user_name,
            companion_name,
        ))
    }
}

impl<S: PersonSink> CommitObserver for PersonsObserver<S> {
    fn on_committed(
        &self,
        _checkpoint: &Checkpoint,
        facts: &[Fact],
        _superseded: &[i64],
    ) -> Result<(), String> {
        for plan in plan_upserts(facts, &self.user_name, &self.companion_name) {
            if let Err(e) = self.sink.upsert(&plan) {
                eprintln!(
                    "compaction persons: failed to upsert person {:?}: {e}",
                    plan.name
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn person_fact(id: i64, name: &str, sources: Vec<i32>) -> Fact {
        Fact {
            id,
            compaction_id: 1,
            category: FactCategory::Person,
            subject: Some(FactSubject::Person(name.to_string())),
            text: format!("{name}: friend"),
            quote_speaker: None,
            sources,
            replaces: Vec::new(),
            relation_to: Some(FactSubject::User),
            relation: Some("friend".to_string()),
            canon: true,
            active: true,
            superseded_by: None,
            rejected_reason: None,
        }
    }

    #[test]
    fn importance_from_mentions_is_monotonic_and_stays_in_bounds() {
        let one = importance_from_mentions(1);
        let five = importance_from_mentions(5);
        let fifty = importance_from_mentions(50);

        assert!((0.0..=1.0).contains(&one));
        assert!((0.0..=1.0).contains(&five));
        assert!((0.0..=1.0).contains(&fifty));
        assert!(five > one);
        assert!(fifty > five);

        // Non-positive input must never panic (`ln` of 0 or negative) and
        // must still clamp into range.
        let zero = importance_from_mentions(0);
        assert!((0.0..=1.0).contains(&zero));
    }

    #[test]
    fn is_junk_person_name_rejects_pronouns_stop_words_and_short_names() {
        assert!(is_junk_person_name("Her"));
        assert!(is_junk_person_name("you"));
        assert!(is_junk_person_name("His"));
        assert!(is_junk_person_name("the"));
        assert!(is_junk_person_name("Table"));
        assert!(is_junk_person_name("ab"));
        assert!(is_junk_person_name(""));

        assert!(!is_junk_person_name("Alice"));
        assert!(!is_junk_person_name("Bob"));
    }

    #[test]
    fn plan_upserts_merges_duplicate_names_and_sums_mentions() {
        let facts = vec![
            person_fact(1, "Alice", vec![10]),
            person_fact(2, "alice", vec![11, 12]),
        ];

        let plans = plan_upserts(&facts, "User", "Companion");

        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].name, "Alice");
        assert_eq!(plans[0].mentions, 3);
        assert_eq!(plans[0].relationship_to_user.as_deref(), Some("friend"));
    }

    #[test]
    fn plan_upserts_skips_the_user_and_companion_by_name() {
        let facts = vec![
            person_fact(1, "User", vec![1]),
            person_fact(2, "Companion", vec![1]),
            person_fact(3, "Alice", vec![1]),
        ];

        let plans = plan_upserts(&facts, "User", "Companion");

        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].name, "Alice");
    }

    #[test]
    fn plan_upserts_skips_junk_names_inactive_and_non_person_facts() {
        let mut junk = person_fact(1, "Her", vec![1]);
        let mut inactive = person_fact(2, "Bob", vec![1]);
        inactive.active = false;
        let mut non_person = person_fact(3, "Carl", vec![1]);
        non_person.category = FactCategory::Milestone;
        junk.subject = Some(FactSubject::Person("Her".to_string()));

        let plans = plan_upserts(&[junk, inactive, non_person], "User", "Companion");

        assert!(plans.is_empty());
    }

    #[test]
    fn plan_upserts_maps_relation_to_companion_separately_from_user() {
        let mut fact = person_fact(1, "Dana", vec![1]);
        fact.relation_to = Some(FactSubject::Companion);
        fact.relation = Some("colleague".to_string());

        let plans = plan_upserts(&[fact], "User", "Companion");

        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].relationship_to_user, None);
        assert_eq!(
            plans[0].relationship_to_companion.as_deref(),
            Some("colleague")
        );
    }

    struct RecordingSink {
        upserted: std::sync::Mutex<Vec<String>>,
        fail_names: Vec<String>,
    }

    impl RecordingSink {
        fn new(fail_names: Vec<&str>) -> Self {
            Self {
                upserted: std::sync::Mutex::new(Vec::new()),
                fail_names: fail_names.into_iter().map(str::to_string).collect(),
            }
        }
    }

    impl PersonSink for RecordingSink {
        fn upsert(&self, p: &PersonUpsert) -> rusqlite::Result<i32> {
            if self.fail_names.contains(&p.name) {
                return Err(rusqlite::Error::QueryReturnedNoRows);
            }
            self.upserted.lock().unwrap().push(p.name.clone());
            Ok(1)
        }
    }

    fn test_checkpoint() -> Checkpoint {
        Checkpoint {
            id: 1,
            companion_id: 1,
            from_message_id: 1,
            through_message_id: 2,
            status: crate::compaction::types::CompactionStatus::Committed,
            trigger: crate::compaction::types::CompactionTrigger::Manual,
            raw_model_output: None,
            summary: None,
            rolling_summary: None,
            attitude_ratings: None,
            needs_merge: false,
            created_at: "now".to_string(),
            committed_at: Some("now".to_string()),
        }
    }

    #[test]
    fn on_committed_upserts_one_row_per_distinct_person() {
        let facts = vec![
            person_fact(1, "Alice", vec![1]),
            person_fact(2, "Bob", vec![1]),
        ];
        let sink = RecordingSink::new(vec![]);
        let observer = PersonsObserver::new(sink, "User".to_string(), "Companion".to_string());

        observer
            .on_committed(&test_checkpoint(), &facts, &[])
            .unwrap();

        let mut upserted = observer.sink.upserted.lock().unwrap().clone();
        upserted.sort();
        assert_eq!(upserted, vec!["Alice".to_string(), "Bob".to_string()]);
    }

    #[test]
    fn on_committed_is_not_stopped_by_a_sink_error_on_one_person() {
        let facts = vec![
            person_fact(1, "Alice", vec![1]),
            person_fact(2, "Bob", vec![1]),
        ];
        let sink = RecordingSink::new(vec!["Alice"]);
        let observer = PersonsObserver::new(sink, "User".to_string(), "Companion".to_string());

        let result = observer.on_committed(&test_checkpoint(), &facts, &[]);

        assert!(result.is_ok());
        assert_eq!(
            *observer.sink.upserted.lock().unwrap(),
            vec!["Bob".to_string()]
        );
    }
}
