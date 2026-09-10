//! [`SpeakerInfo`] for multiplayer (#182): canon is decided by the live
//! participant registry, not a fixed `user`/`char` pair. Free of any
//! host-only state — a `ParticipantRegistry` is plain owned data — so
//! #186's joiner-side extraction can build one from its own registry
//! snapshot too.

use crate::compaction::SpeakerInfo;
use crate::database::USER_SPEAKER_ID;
use crate::participants::{ParticipantId, ParticipantKind, ParticipantRegistry};

/// [`SpeakerInfo`] backed by a [`ParticipantRegistry`] snapshot: only
/// [`ParticipantKind::Human`] counts as canon (the epic's canon rule — the
/// host bot and every remote bot are advisory). A `speaker_id` the registry
/// does not recognise (a bot that has since disconnected, or `system`)
/// falls back to `id == USER_SPEAKER_ID` rather than erroring, so a stale
/// id cited by an already-selected checkpoint range stays advisory instead
/// of panicking or wrongly promoting to canon.
pub struct RegistrySpeakers(pub ParticipantRegistry);

impl SpeakerInfo for RegistrySpeakers {
    fn display_name(&self, speaker_id: &str) -> String {
        match ParticipantId::parse(speaker_id) {
            Ok(id) => self
                .0
                .display_name(&id)
                .map(str::to_string)
                .unwrap_or_else(|| speaker_id.to_string()),
            Err(_) => speaker_id.to_string(),
        }
    }

    fn is_canon(&self, speaker_id: &str) -> bool {
        ParticipantId::parse(speaker_id)
            .ok()
            .and_then(|id| self.0.get(&id))
            .map(|p| p.kind == ParticipantKind::Human)
            .unwrap_or(speaker_id == USER_SPEAKER_ID)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::extract::fill_draft;
    use crate::compaction::fixtures::synthetic_range;
    use crate::compaction::store::{CompactionStore, RecordingStore};
    use crate::compaction::types::{CompactionTrigger, FactCategory, NewDraft};
    use crate::compaction::CitedMessage;
    use crate::llm::FakeExtractor;
    use crate::participants::Participant;

    /// `user`/`char` plus a connected `bot1` (`RemoteBot`), the shape most
    /// multiplayer canon-predicate tests need.
    fn registry_with_bot1() -> ParticipantRegistry {
        let mut registry = ParticipantRegistry::solo("Alice", "Bob", None);
        registry
            .insert(Participant {
                id: ParticipantId::parse("bot1").unwrap(),
                display_name: "Ada".to_string(),
                kind: ParticipantKind::RemoteBot,
                avatar: None,
            })
            .unwrap();
        registry
    }

    #[test]
    fn user_is_canon() {
        let speakers = RegistrySpeakers(registry_with_bot1());
        assert!(speakers.is_canon("user"));
    }

    #[test]
    fn the_host_companion_and_a_remote_bot_are_both_advisory() {
        let speakers = RegistrySpeakers(registry_with_bot1());
        assert!(!speakers.is_canon("char"));
        assert!(!speakers.is_canon("bot1"));
    }

    #[test]
    fn an_id_absent_from_the_registry_is_advisory() {
        let speakers = RegistrySpeakers(registry_with_bot1());
        // Never inserted (or since disconnected): not `user`, so it falls
        // back to advisory rather than panicking on a missing lookup.
        assert!(!speakers.is_canon("bot2"));
    }

    #[test]
    fn system_notices_are_never_canon() {
        let speakers = RegistrySpeakers(registry_with_bot1());
        assert!(!speakers.is_canon("system"));
    }

    #[test]
    fn display_name_falls_back_to_the_raw_id_when_the_registry_does_not_know_it() {
        let speakers = RegistrySpeakers(registry_with_bot1());
        assert_eq!(speakers.display_name("bot1"), "Ada");
        assert_eq!(speakers.display_name("bot2"), "bot2");
    }

    /// `vex` as a connected `RemoteBot`, over #173's shared fixture (its own
    /// doc comment calls out ids 58/60 as "a third speaker ... for the
    /// multiplayer-shaped canon predicate test").
    fn registry_with_vex() -> ParticipantRegistry {
        let mut registry = ParticipantRegistry::solo("Alice", "Bob", None);
        registry
            .insert(Participant {
                id: ParticipantId::parse("vex").unwrap(),
                display_name: "Vex".to_string(),
                kind: ParticipantKind::RemoteBot,
                avatar: None,
            })
            .unwrap();
        registry
    }

    #[test]
    fn a_remote_bots_world_fact_and_named_person_are_rejected_but_the_identical_user_line_is_accepted(
    ) {
        let store = RecordingStore::new();
        let draft_id = store
            .insert_draft(NewDraft {
                companion_id: 1,
                from_message_id: 58,
                through_message_id: 59,
                trigger: CompactionTrigger::Threshold,
                raw_model_output: None,
            })
            .unwrap();
        let draft = store.get_checkpoint(draft_id).unwrap().unwrap();

        // A small, purpose-built range rather than the full 20-message
        // fixture: message 58 is vex's own line (the fixture's real text),
        // message 59 is a user line asserting the identical fact in the
        // model's own words for this test.
        let full_range = synthetic_range();
        let vex_line = full_range
            .iter()
            .find(|m| m.speaker_id == "vex")
            .cloned()
            .expect("the shared fixture carries a vex-sourced line");
        let range = vec![
            vex_line,
            CitedMessage {
                id: 59,
                speaker_id: "user".to_string(),
                content: "we found an old brass key under the porch step".to_string(),
            },
        ];

        let output = format!(
            r#"{{
                "companion_state": [], "user_state": [
                    {{"text": "found an old brass key under the porch step", "sources": [{vex_id}], "replaces": []}},
                    {{"text": "found an old brass key under the porch step", "sources": [59], "replaces": []}}
                ],
                "milestones": [], "backstory": [], "open_threads": [], "rules": [],
                "people": [
                    {{"name": "Finn", "relation_to": "user", "relation": "a dockhand vex mentioned", "sources": [{vex_id}]}}
                ],
                "key_quotes": [], "summary": "settling into the new place",
                "attitude": {{"trust":0,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0}}
            }}"#,
            vex_id = range[0].id
        );
        let extractor = FakeExtractor::returning(vec![Ok(output)]);
        let speakers = RegistrySpeakers(registry_with_vex());

        fill_draft(&store, &extractor, &draft, &range, &speakers, usize::MAX)
            .expect("fill_draft should succeed");

        let facts = store.facts_for(draft_id).unwrap();

        let user_state: Vec<_> = facts
            .iter()
            .filter(|f| f.category == FactCategory::UserState)
            .collect();
        assert_eq!(
            user_state.len(),
            2,
            "both drafted items are kept, rejected or not"
        );
        let vex_sourced = user_state
            .iter()
            .find(|f| f.sources == vec![range[0].id])
            .expect("the vex-sourced item should still be present, just rejected");
        assert!(
            !vex_sourced.active,
            "a bot-sourced world fact must never become an active user_state fact"
        );
        let user_sourced = user_state
            .iter()
            .find(|f| f.sources == vec![59])
            .expect("the user-sourced item should be present");
        assert!(
            user_sourced.active,
            "the identical fact, sourced from the human turn, should be accepted"
        );

        let people: Vec<_> = facts
            .iter()
            .filter(|f| f.category == FactCategory::Person)
            .collect();
        assert_eq!(people.len(), 1);
        assert!(
            !people[0].active,
            "a person introduced only by a bot turn must never become an active fact"
        );
    }
}
