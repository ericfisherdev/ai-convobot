//! The pure half of the extraction pass (#173): `serde` structs mirroring
//! the model's JSON output schema, [`parse_extraction`], and
//! [`to_fact_drafts`], which maps a parsed [`ExtractionOutput`] to
//! [`FactDraft`]s. The prompt, GBNF grammar that constrains the model to
//! this exact shape, chunking, and the `Extractor`-backed runner are #185's
//! half, built on these same types.
#![allow(dead_code)]

use serde::Deserialize;

use crate::compaction::types::{FactCategory, FactDraft, FactSubject};

/// The model's extraction output, one JSON object per compacted range.
/// `#[serde(deny_unknown_fields)]` so a model that drifts from the schema
/// fails parse instead of being silently accepted.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionOutput {
    pub companion_state: Vec<TextItem>,
    pub user_state: Vec<TextItem>,
    pub milestones: Vec<TextItem>,
    pub backstory: Vec<BackstoryItem>,
    pub open_threads: Vec<TextItem>,
    pub rules: Vec<QuoteItem>,
    pub people: Vec<PersonItem>,
    pub key_quotes: Vec<QuoteItem>,
    pub summary: String,
    pub attitude: AttitudeRatings,
}

/// One `companion_state`/`user_state`/`milestones`/`open_threads` entry.
/// `replaces` is only meaningful on `companion_state`/`user_state`: the ids
/// of prior overlay facts this item updates. `#[serde(default)]` since a
/// `milestones`/`open_threads` item never carries it.
#[derive(Debug, Clone, Deserialize)]
pub struct TextItem {
    pub text: String,
    pub sources: Vec<i32>,
    #[serde(default)]
    pub replaces: Vec<i64>,
}

/// One `backstory` entry: `about` picks the subject between the user and
/// the companion.
#[derive(Debug, Clone, Deserialize)]
pub struct BackstoryItem {
    pub about: Party,
    pub text: String,
    pub sources: Vec<i32>,
}

/// One `rules`/`key_quotes` entry: a verbatim quote plus who said it.
#[derive(Debug, Clone, Deserialize)]
pub struct QuoteItem {
    pub quote: String,
    pub speaker: Party,
    pub sources: Vec<i32>,
}

/// One `people` entry: a person other than the user or companion,
/// introduced in the compacted range.
#[derive(Debug, Clone, Deserialize)]
pub struct PersonItem {
    pub name: String,
    pub relation_to: Party,
    pub relation: String,
    pub sources: Vec<i32>,
}

/// Who a `backstory`/`rules`/`key_quotes`/`people` item is about or spoken
/// by. Serializes/deserializes lowercase (`"user"`/`"companion"`) to match
/// the grammar's constrained output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Party {
    User,
    Companion,
}

/// The eight `attitude_engine::AttitudeDimension` ratings the model emits
/// alongside the extracted facts, on a 0-100 scale (the grammar in #185
/// already restricts each to three digits; [`parse_extraction`] clamps
/// anyway so a malformed value can never escape this module).
#[derive(Debug, Clone, Deserialize)]
pub struct AttitudeRatings {
    pub trust: u8,
    pub love: u8,
    pub fear: u8,
    pub anger: u8,
    pub joy: u8,
    pub sorrow: u8,
    pub suspicion: u8,
    pub gratitude: u8,
}

/// Largest value an [`AttitudeRatings`] field may hold after
/// [`parse_extraction`] clamps it.
const MAX_ATTITUDE_RATING: u8 = 100;

impl AttitudeRatings {
    fn clamp_to_valid_range(&mut self) {
        self.trust = self.trust.min(MAX_ATTITUDE_RATING);
        self.love = self.love.min(MAX_ATTITUDE_RATING);
        self.fear = self.fear.min(MAX_ATTITUDE_RATING);
        self.anger = self.anger.min(MAX_ATTITUDE_RATING);
        self.joy = self.joy.min(MAX_ATTITUDE_RATING);
        self.sorrow = self.sorrow.min(MAX_ATTITUDE_RATING);
        self.suspicion = self.suspicion.min(MAX_ATTITUDE_RATING);
        self.gratitude = self.gratitude.min(MAX_ATTITUDE_RATING);
    }
}

/// Parses one model completion into an [`ExtractionOutput`], clamping every
/// [`AttitudeRatings`] field to `0..=100` afterward.
pub fn parse_extraction(raw: &str) -> Result<ExtractionOutput, serde_json::Error> {
    let mut output: ExtractionOutput = serde_json::from_str(raw)?;
    output.attitude.clamp_to_valid_range();
    Ok(output)
}

/// Maps a `Party` to the [`FactSubject`] it stands for. `Party` can only
/// ever be `User`/`Companion`, so this never produces `FactSubject::Person`.
fn subject_of(party: Party) -> FactSubject {
    match party {
        Party::User => FactSubject::User,
        Party::Companion => FactSubject::Companion,
    }
}

/// Maps one [`ExtractionOutput`] to the [`FactDraft`]s `validate::validate`
/// will run the canon rule and the rest over. Preserves each array's order
/// and item count; `validate` is what may reject an item, never this
/// function. `output.summary` and `output.attitude` go on the checkpoint
/// row, not into facts.
pub fn to_fact_drafts(output: &ExtractionOutput) -> Vec<FactDraft> {
    let mut drafts = Vec::new();

    for item in &output.companion_state {
        drafts.push(text_item_draft(
            item,
            FactCategory::CompanionState,
            Some(FactSubject::Companion),
        ));
    }
    for item in &output.user_state {
        drafts.push(text_item_draft(
            item,
            FactCategory::UserState,
            Some(FactSubject::User),
        ));
    }
    for item in &output.milestones {
        drafts.push(text_item_draft(item, FactCategory::Milestone, None));
    }
    for item in &output.backstory {
        drafts.push(FactDraft {
            category: FactCategory::Backstory,
            subject: Some(subject_of(item.about)),
            text: item.text.clone(),
            quote_speaker: None,
            sources: item.sources.clone(),
            replaces: Vec::new(),
            relation_to: None,
            relation: None,
            canon: false,
            rejected_reason: None,
        });
    }
    for item in &output.open_threads {
        drafts.push(text_item_draft(item, FactCategory::OpenThread, None));
    }
    for item in &output.rules {
        drafts.push(quote_item_draft(item, FactCategory::Rule));
    }
    for item in &output.people {
        drafts.push(FactDraft {
            category: FactCategory::Person,
            subject: Some(FactSubject::Person(item.name.clone())),
            text: item.relation.clone(),
            quote_speaker: None,
            sources: item.sources.clone(),
            replaces: Vec::new(),
            relation_to: Some(subject_of(item.relation_to)),
            relation: Some(item.relation.clone()),
            canon: false,
            rejected_reason: None,
        });
    }
    for item in &output.key_quotes {
        drafts.push(quote_item_draft(item, FactCategory::KeyQuote));
    }

    drafts
}

/// Shared by every `TextItem`-shaped array (`companion_state`, `user_state`,
/// `milestones`, `open_threads`): only `subject` and `replaces` differ, and
/// `replaces` is only ever non-empty for the two state categories, since
/// only `TextItem` carries it.
fn text_item_draft(
    item: &TextItem,
    category: FactCategory,
    subject: Option<FactSubject>,
) -> FactDraft {
    FactDraft {
        category,
        subject,
        text: item.text.clone(),
        quote_speaker: None,
        sources: item.sources.clone(),
        replaces: item.replaces.clone(),
        relation_to: None,
        relation: None,
        canon: false,
        rejected_reason: None,
    }
}

/// Shared by `rules` and `key_quotes`: both are a verbatim `QuoteItem`, only
/// the category differs.
fn quote_item_draft(item: &QuoteItem, category: FactCategory) -> FactDraft {
    FactDraft {
        category,
        subject: None,
        text: item.quote.clone(),
        quote_speaker: Some(item.speaker_token().to_string()),
        sources: item.sources.clone(),
        replaces: Vec::new(),
        relation_to: None,
        relation: None,
        canon: false,
        rejected_reason: None,
    }
}

impl QuoteItem {
    /// `quote_speaker` is stored as the same lowercase token every other
    /// speaker-shaped column uses (`"user"`/`"companion"`), not `Party`'s
    /// `Debug` form.
    fn speaker_token(&self) -> &'static str {
        match self.speaker {
            Party::User => "user",
            Party::Companion => "companion",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::fixtures::bad_draft;

    #[test]
    fn bad_draft_fixture_parses() {
        let output = bad_draft();
        assert!(!output.rules.is_empty());
    }

    #[test]
    fn an_extra_top_level_key_fails_to_parse() {
        let raw = r#"{
            "companion_state": [], "user_state": [], "milestones": [],
            "backstory": [], "open_threads": [], "rules": [], "people": [],
            "key_quotes": [], "summary": "s",
            "attitude": {"trust":0,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0},
            "extra_field": "not in the schema"
        }"#;
        assert!(parse_extraction(raw).is_err());
    }

    #[test]
    fn a_rating_of_250_clamps_to_100() {
        let raw = r#"{
            "companion_state": [], "user_state": [], "milestones": [],
            "backstory": [], "open_threads": [], "rules": [], "people": [],
            "key_quotes": [], "summary": "s",
            "attitude": {"trust":250,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0}
        }"#;
        let output = parse_extraction(raw).unwrap();
        assert_eq!(output.attitude.trust, 100);
    }

    #[test]
    fn to_fact_drafts_preserves_order_and_item_count() {
        let output = bad_draft();
        let total_items = output.companion_state.len()
            + output.user_state.len()
            + output.milestones.len()
            + output.backstory.len()
            + output.open_threads.len()
            + output.rules.len()
            + output.people.len()
            + output.key_quotes.len();
        let drafts = to_fact_drafts(&output);
        assert_eq!(drafts.len(), total_items);
    }

    #[test]
    fn a_people_item_maps_to_a_person_draft_with_relation_to_relation_and_no_quote_speaker() {
        let raw = r#"{
            "companion_state": [], "user_state": [], "milestones": [],
            "backstory": [], "open_threads": [],
            "rules": [], "key_quotes": [],
            "people": [{"name":"Ann","relation_to":"user","relation":"sister","sources":[46]}],
            "summary": "s",
            "attitude": {"trust":0,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0}
        }"#;
        let output = parse_extraction(raw).unwrap();
        let drafts = to_fact_drafts(&output);
        assert_eq!(drafts.len(), 1);
        let person = &drafts[0];
        assert_eq!(person.category, FactCategory::Person);
        assert_eq!(person.relation_to, Some(FactSubject::User));
        assert_eq!(person.relation.as_deref(), Some("sister"));
        assert_eq!(person.quote_speaker, None);
    }

    #[test]
    fn a_companion_state_item_with_replaces_maps_through_while_a_milestone_leaves_it_empty_and_has_no_relation_to(
    ) {
        let raw = r#"{
            "companion_state": [{"text":"is happier now","sources":[46],"replaces":[3,4]}],
            "user_state": [], "milestones": [{"text":"left home","sources":[47]}],
            "backstory": [], "open_threads": [], "rules": [], "people": [],
            "key_quotes": [], "summary": "s",
            "attitude": {"trust":0,"love":0,"fear":0,"anger":0,"joy":0,"sorrow":0,"suspicion":0,"gratitude":0}
        }"#;
        let output = parse_extraction(raw).unwrap();
        let drafts = to_fact_drafts(&output);
        assert_eq!(drafts[0].replaces, vec![3, 4]);
        assert!(drafts[1].replaces.is_empty());
        assert_eq!(drafts[1].relation_to, None);
    }
}
