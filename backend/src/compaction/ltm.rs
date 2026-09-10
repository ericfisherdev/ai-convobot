//! Indexes committed facts into the tantivy long-term memory index (#178),
//! so keyword recall (`LongTermMem::get_matches`) returns facts — which fit
//! the compaction slice — instead of the raw turn pairs `llm.rs::generate`
//! used to write on every reply.
//!
//! [`fact_entry`] is the pure text a fact renders as inside the index: no
//! date, no clock time, ever (the whole point of dropping the `* at <date>
//! *` prefix). [`LtmObserver`] is the [`crate::compaction::commit::
//! CommitObserver`] impl that calls it: `add_fact` for every newly active
//! fact a commit produced, `remove_fact` for every id it superseded. It
//! takes its [`crate::long_term_mem::LongTermMem`] by reference rather than
//! reaching for [`crate::long_term_mem::LongTermMem::shared`] itself, so
//! tests can point it at a temporary index instead of the process-wide
//! singleton; [`crate::compaction::production_commit_deps`] resolves
//! `shared()` once and hands the `'static` reference in.

use crate::compaction::commit::CommitObserver;
use crate::compaction::types::{Checkpoint, Fact, FactCategory, FactSubject};
use crate::long_term_mem::LongTermMem;
use crate::participants::{placeholder, ParticipantId};

/// The text `LtmObserver` indexes for `fact`, and the text `render::render`
/// emits when the same string comes back through recall (`llm.rs::
/// assemble_prompt` runs it through `expand_placeholders` before splicing
/// it in, same as the placeholders in a persona). Never a date or a clock
/// time — the whole reason this module replaces the old `* at <date> *`
/// turn-pair entries.
///
/// `Rule`/`KeyQuote` facts are quotes: the indexed text is the quote itself,
/// prefixed with the speaker only when `quote_speaker` names one (`"user"`
/// maps to `{{user}}`, anything else set to `{{char}}`, matching
/// `context.rs`'s `quote_speaker_of` fallback). Every other category is
/// prefixed from `subject` instead: `{{char}}`/`{{user}}` for the two fixed
/// subjects, the person's own name for `FactSubject::Person`, and no prefix
/// at all when there is no subject to key off (`Milestone`, `Backstory`,
/// `OpenThread` facts typically have none).
pub fn fact_entry(fact: &Fact) -> String {
    if matches!(fact.category, FactCategory::Rule | FactCategory::KeyQuote) {
        return match fact.quote_speaker.as_deref() {
            Some("user") => format!("{}: {}", placeholder(&ParticipantId::USER), fact.text),
            Some(_) => format!("{}: {}", placeholder(&ParticipantId::CHAR), fact.text),
            None => fact.text.clone(),
        };
    }

    match &fact.subject {
        Some(FactSubject::Companion) => {
            format!("{}: {}", placeholder(&ParticipantId::CHAR), fact.text)
        }
        Some(FactSubject::User) => format!("{}: {}", placeholder(&ParticipantId::USER), fact.text),
        Some(FactSubject::Person(name)) => format!("{name}: {}", fact.text),
        None => fact.text.clone(),
    }
}

/// Runs after a checkpoint commits: indexes every newly active fact and
/// removes every superseded one. Holds its index by reference rather than
/// the process-wide singleton so tests never touch it — see the module doc.
pub struct LtmObserver<'a> {
    ltm: &'a LongTermMem,
}

impl<'a> LtmObserver<'a> {
    pub fn new(ltm: &'a LongTermMem) -> Self {
        Self { ltm }
    }
}

impl CommitObserver for LtmObserver<'_> {
    fn on_committed(
        &self,
        _checkpoint: &Checkpoint,
        facts: &[Fact],
        superseded: &[i64],
    ) -> Result<(), String> {
        for fact in facts {
            if let Err(e) = self.ltm.add_fact(fact.id, &fact_entry(fact)) {
                eprintln!(
                    "compaction ltm observer: failed to index fact {}: {e}",
                    fact.id
                );
            }
        }
        for &fact_id in superseded {
            if let Err(e) = self.ltm.remove_fact(fact_id) {
                eprintln!(
                    "compaction ltm observer: failed to remove superseded fact {fact_id}: {e}"
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::types::{CompactionStatus, CompactionTrigger};

    fn fact(category: FactCategory, subject: Option<FactSubject>, text: &str) -> Fact {
        Fact {
            id: 1,
            compaction_id: 1,
            category,
            subject,
            text: text.to_string(),
            quote_speaker: None,
            sources: vec![],
            replaces: vec![],
            relation_to: None,
            relation: None,
            canon: true,
            active: true,
            superseded_by: None,
            rejected_reason: None,
        }
    }

    fn a_checkpoint() -> Checkpoint {
        Checkpoint {
            id: 1,
            companion_id: 1,
            from_message_id: 1,
            through_message_id: 10,
            trigger: CompactionTrigger::Threshold,
            status: CompactionStatus::Committed,
            raw_model_output: None,
            summary: None,
            rolling_summary: None,
            attitude_ratings: None,
            needs_merge: false,
            created_at: "now".to_string(),
            committed_at: Some("now".to_string()),
            extraction_error: None,
        }
    }

    #[test]
    fn fact_entry_prefixes_companion_and_user_subjects_with_placeholders() {
        let companion = fact(
            FactCategory::CompanionState,
            Some(FactSubject::Companion),
            "is nervous",
        );
        let user = fact(
            FactCategory::UserState,
            Some(FactSubject::User),
            "loves cats",
        );
        assert_eq!(fact_entry(&companion), "{{char}}: is nervous");
        assert_eq!(fact_entry(&user), "{{user}}: loves cats");
    }

    #[test]
    fn fact_entry_prefixes_person_facts_with_the_persons_name() {
        let person = fact(
            FactCategory::Person,
            Some(FactSubject::Person("Wren".to_string())),
            "works at the bakery",
        );
        assert_eq!(fact_entry(&person), "Wren: works at the bakery");
    }

    #[test]
    fn fact_entry_has_no_prefix_when_there_is_no_subject() {
        let milestone = fact(FactCategory::Milestone, None, "met at the park");
        assert_eq!(fact_entry(&milestone), "met at the park");
    }

    #[test]
    fn fact_entry_uses_quote_speaker_for_rules_and_key_quotes() {
        let mut user_rule = fact(FactCategory::Rule, None, "always knock first");
        user_rule.quote_speaker = Some("user".to_string());
        let mut companion_quote = fact(FactCategory::KeyQuote, None, "I promise");
        companion_quote.quote_speaker = Some("companion".to_string());
        let mut unattributed = fact(FactCategory::KeyQuote, None, "no one knows who said this");
        unattributed.quote_speaker = None;

        assert_eq!(fact_entry(&user_rule), "{{user}}: always knock first");
        assert_eq!(fact_entry(&companion_quote), "{{char}}: I promise");
        assert_eq!(fact_entry(&unattributed), "no one knows who said this");
    }

    #[test]
    fn fact_entry_never_contains_a_date_or_clock_time() {
        let date_pattern = regex::Regex::new(r"\* at \w+ \d\d\.\d\d\.\d{4}").unwrap();
        let facts = [
            fact(
                FactCategory::CompanionState,
                Some(FactSubject::Companion),
                "is happy",
            ),
            fact(FactCategory::Milestone, None, "went to dinner"),
        ];
        for f in &facts {
            let entry = fact_entry(f);
            assert!(
                !date_pattern.is_match(&entry),
                "fact_entry produced a date-like string: {entry}"
            );
        }
    }

    #[test]
    fn observer_indexes_newly_active_facts_and_removes_superseded_ones() {
        let dir = tempfile::TempDir::new().unwrap();
        let ltm = LongTermMem::open_at(dir.path()).unwrap();
        ltm.add_fact(99, "a fact that will be superseded").unwrap();

        let observer = LtmObserver::new(&ltm);
        let checkpoint = a_checkpoint();
        let new_fact = fact(
            FactCategory::UserState,
            Some(FactSubject::User),
            "loves hiking",
        );

        observer
            .on_committed(&checkpoint, &[new_fact], &[99])
            .unwrap();

        let matches = ltm.get_matches("hiking", 5).unwrap();
        assert_eq!(matches, vec!["{{user}}: loves hiking".to_string()]);
        assert!(ltm.get_matches("superseded", 5).unwrap().is_empty());
    }
}
