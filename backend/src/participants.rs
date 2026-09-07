//! Single source of truth for "who is in the chat".
//!
//! Before this module, the prompt code hard-coded exactly two identities:
//! `{{char}}` and `{{user}}` were string-replaced independently in roughly
//! twenty places in `llm.rs`. Multi-instance chat (joiner bots, `bot1`,
//! `bot2`, ...) needs one place that knows the participant list, how each id
//! maps to a display name and avatar, how `{{id}}` placeholders expand, and
//! how `@id` / `@Display Name` mentions are recognised in free text.
//!
//! Everything here is constructed from plain values: no `Database` calls, no
//! HTTP, no globals. That makes it unit-testable on its own (like
//! `chat_turn.rs`) and, since a `ParticipantRegistry` holds only owned
//! `String`s, `Send + Sync` for free — `main.rs` relies on that to snapshot
//! it out of an `RwLock` (`registry.read().clone()`) and move it into a
//! blocking task or the generation thread.

use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fmt;

use crate::database::{CHAR_SPEAKER_ID, USER_SPEAKER_ID};

/// A participant identifier: `user`, `char`, or a joiner id like `bot1`.
///
/// Grammar: `^[a-z][a-z0-9_]{0,15}$` (1 to 16 lowercase ASCII characters,
/// starting with a letter). Chosen to be safe to embed directly in a
/// `{{id}}` placeholder or an `@id` mention without further escaping.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub struct ParticipantId(Cow<'static, str>);

impl ParticipantId {
    /// The reserved id for the human participant in every chat.
    pub const USER: ParticipantId = ParticipantId(Cow::Borrowed(USER_SPEAKER_ID));
    /// The reserved id for the (single, solo-mode) companion.
    pub const CHAR: ParticipantId = ParticipantId(Cow::Borrowed(CHAR_SPEAKER_ID));

    /// Validates `value` against the id grammar.
    ///
    /// # Errors
    /// Returns [`ParticipantError::InvalidId`] when `value` is empty, longer
    /// than 16 characters, does not start with a lowercase ASCII letter, or
    /// contains anything outside `[a-z0-9_]` after the first character.
    pub fn parse(value: &str) -> Result<Self, ParticipantError> {
        let len = value.len();
        let mut chars = value.chars();
        let first_ok = matches!(chars.next(), Some(c) if c.is_ascii_lowercase());
        let rest_ok = chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        if !first_ok || !rest_ok || !(1..=16).contains(&len) {
            return Err(ParticipantError::InvalidId {
                value: value.to_string(),
            });
        }
        Ok(ParticipantId(Cow::Owned(value.to_string())))
    }

    /// Whether this id is one of the two ids every registry always carries
    /// ([`ParticipantId::USER`], [`ParticipantId::CHAR`]).
    pub fn is_reserved(&self) -> bool {
        *self == ParticipantId::USER || *self == ParticipantId::CHAR
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ParticipantId {
    type Error = ParticipantError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        ParticipantId::parse(&value)
    }
}

impl fmt::Display for ParticipantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for ParticipantId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// The single `{{id}}` placeholder form for `id`, e.g. `{{bot1}}`. The
/// counterpart to what [`expand_placeholders`] recognises, and used by
/// `llm.rs` to write `{{user}}`/`{{char}}` into the long-term-memory entry
/// instead of repeating the literal string in two places.
pub fn placeholder(id: &ParticipantId) -> String {
    format!("{{{{{}}}}}", id)
}

/// Everything that can go wrong with a [`ParticipantId`] or a
/// [`ParticipantRegistry`] operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParticipantError {
    /// `value` does not satisfy the id grammar. Returned by
    /// [`ParticipantId::parse`].
    InvalidId { value: String },
    /// The id is [`ParticipantId::USER`] or [`ParticipantId::CHAR`], which
    /// every registry already carries. Returned by
    /// [`ParticipantRegistry::insert`] and [`ParticipantRegistry::remove`].
    Reserved(ParticipantId),
    /// The id is already present. Returned by [`ParticipantRegistry::insert`].
    Duplicate(ParticipantId),
    /// The id is not present. Returned by [`ParticipantRegistry::remove`] and
    /// [`ParticipantRegistry::rename`].
    Unknown(ParticipantId),
}

impl fmt::Display for ParticipantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParticipantError::InvalidId { value } => {
                write!(f, "{:?} is not a valid participant id (expected 1-16 lowercase letters, digits or underscores, starting with a letter)", value)
            }
            ParticipantError::Reserved(id) => {
                write!(f, "`{}` is a reserved participant id", id)
            }
            ParticipantError::Duplicate(id) => {
                write!(f, "`{}` is already a participant", id)
            }
            ParticipantError::Unknown(id) => {
                write!(f, "`{}` is not a participant", id)
            }
        }
    }
}

impl std::error::Error for ParticipantError {}

/// What kind of participant a [`Participant`] is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum ParticipantKind {
    /// The human at the keyboard.
    Human,
    /// The companion running in this process.
    HostBot,
    /// A companion joining from another instance. Not produced anywhere
    /// yet; #129 is what starts constructing these.
    #[allow(dead_code)]
    RemoteBot,
}

/// A path or URL the frontend can put directly in an `<img src>`. Today
/// always the companion's `avatar_path`; a transferred file under #129 later.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AvatarRef(String);

impl AvatarRef {
    pub fn new(value: impl Into<String>) -> Self {
        AvatarRef(value.into())
    }

    #[allow(dead_code)] // wired up once #134 surfaces avatars to the frontend
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AvatarRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Maps `CompanionView::avatar_path`-shaped strings to an optional
/// [`AvatarRef`]: an empty path (the default before any avatar is set) means
/// "no avatar" rather than an avatar at the empty path.
pub fn avatar_from(avatar_path: &str) -> Option<AvatarRef> {
    if avatar_path.is_empty() {
        None
    } else {
        Some(AvatarRef::new(avatar_path))
    }
}

/// One member of the chat: a stable id, a display name, what kind of
/// participant it is, and an optional avatar.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Participant {
    pub id: ParticipantId,
    pub display_name: String,
    pub kind: ParticipantKind,
    pub avatar: Option<AvatarRef>,
}

/// The chat's participant list, in join order (join order matters for
/// #131's round order).
///
/// Always constructed via [`ParticipantRegistry::solo`], so the two reserved
/// ids ([`ParticipantId::USER`], [`ParticipantId::CHAR`]) can never be
/// absent.
#[derive(Clone, Debug, Serialize)]
pub struct ParticipantRegistry {
    participants: Vec<Participant>,
}

impl ParticipantRegistry {
    /// Builds a solo-chat registry: `user` (human) then `char` (host bot),
    /// always in that order.
    pub fn solo(
        user_name: &str,
        companion_name: &str,
        companion_avatar: Option<AvatarRef>,
    ) -> Self {
        ParticipantRegistry {
            participants: vec![
                Participant {
                    id: ParticipantId::USER,
                    display_name: user_name.to_string(),
                    kind: ParticipantKind::Human,
                    avatar: None,
                },
                Participant {
                    id: ParticipantId::CHAR,
                    display_name: companion_name.to_string(),
                    kind: ParticipantKind::HostBot,
                    avatar: companion_avatar,
                },
            ],
        }
    }

    /// Adds a new participant.
    ///
    /// # Errors
    /// [`ParticipantError::Reserved`] if `p.id` is [`ParticipantId::USER`] or
    /// [`ParticipantId::CHAR`]; [`ParticipantError::Duplicate`] if `p.id` is
    /// already present.
    #[allow(dead_code)] // wired up by #129, which starts inserting RemoteBot participants
    pub fn insert(&mut self, p: Participant) -> Result<(), ParticipantError> {
        if p.id.is_reserved() {
            return Err(ParticipantError::Reserved(p.id));
        }
        if self.get(&p.id).is_some() {
            return Err(ParticipantError::Duplicate(p.id));
        }
        self.participants.push(p);
        Ok(())
    }

    /// Removes a participant.
    ///
    /// # Errors
    /// [`ParticipantError::Reserved`] for `user`/`char`;
    /// [`ParticipantError::Unknown`] if `id` is not present.
    #[allow(dead_code)] // wired up by #129
    pub fn remove(&mut self, id: &ParticipantId) -> Result<Participant, ParticipantError> {
        if id.is_reserved() {
            return Err(ParticipantError::Reserved(id.clone()));
        }
        let pos = self
            .participants
            .iter()
            .position(|p| &p.id == id)
            .ok_or_else(|| ParticipantError::Unknown(id.clone()))?;
        Ok(self.participants.remove(pos))
    }

    /// Updates a participant's display name and avatar. This is how the
    /// shared registry follows `PUT /api/user` and the companion-editing
    /// handlers.
    ///
    /// # Errors
    /// [`ParticipantError::Unknown`] if `id` is not present.
    pub fn rename(
        &mut self,
        id: &ParticipantId,
        display_name: &str,
        avatar: Option<AvatarRef>,
    ) -> Result<(), ParticipantError> {
        let p = self
            .participants
            .iter_mut()
            .find(|p| &p.id == id)
            .ok_or_else(|| ParticipantError::Unknown(id.clone()))?;
        p.display_name = display_name.to_string();
        p.avatar = avatar;
        Ok(())
    }

    pub fn get(&self, id: &ParticipantId) -> Option<&Participant> {
        self.participants.iter().find(|p| &p.id == id)
    }

    /// Convenience over `get(id).map(|p| &p.display_name)`, used by
    /// [`expand_placeholders`] and the mention functions below.
    pub fn display_name(&self, id: &ParticipantId) -> Option<&str> {
        self.get(id).map(|p| p.display_name.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = &Participant> {
        self.participants.iter()
    }

    /// Every participant that is not the human, i.e. every bot.
    #[allow(dead_code)] // wired up by #127/#129
    pub fn iter_bots(&self) -> impl Iterator<Item = &Participant> {
        self.participants
            .iter()
            .filter(|p| p.kind != ParticipantKind::Human)
    }

    #[allow(dead_code)] // wired up by #127/#129/#134
    pub fn len(&self) -> usize {
        self.participants.len()
    }

    #[allow(dead_code)] // required by clippy::len_without_is_empty; a registry from `solo` is never empty
    pub fn is_empty(&self) -> bool {
        self.participants.is_empty()
    }
}

/// Expands every `{{id}}` placeholder in `text` whose `id` is a known
/// participant into that participant's display name. `{{id}}` tokens that
/// are not a known participant id (a stray `{{companion}}` from the
/// attitude formatter, or a joiner id not yet in the registry) are copied
/// through untouched.
///
/// A single left-to-right scan over the original text, so a display name
/// that itself contains `{{user}}` is never re-expanded.
pub fn expand_placeholders(text: &str, registry: &ParticipantRegistry) -> String {
    let mut result = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("{{") {
        result.push_str(&rest[..start]);
        let after_open = &rest[start + 2..];
        let Some(end) = after_open.find("}}") else {
            // Unterminated `{{`: nothing left to close it, so copy the rest
            // of the text through untouched and stop scanning.
            result.push_str(&rest[start..]);
            rest = "";
            break;
        };
        let token = &after_open[..end];
        let replacement = ParticipantId::parse(token)
            .ok()
            .and_then(|id| registry.display_name(&id).map(str::to_string));
        match replacement {
            Some(name) => result.push_str(&name),
            None => result.push_str(&rest[start..start + 2 + end + 2]),
        }
        rest = &after_open[end + 2..];
    }
    result.push_str(rest);
    result
}

/// A recognised `@mention` in a piece of text: the byte range it occupies in
/// the original string, and the participant it refers to.
#[allow(dead_code)] // wired up by #127/#132; used internally by find/normalise/render_mentions today
struct MentionSpan {
    start: usize,
    end: usize,
    id: ParticipantId,
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Whether `candidate` matches the start of `after`, comparing case
/// insensitively and walking `char_indices` so a multi-byte character is
/// never split. Returns the byte length of the match in `after`.
fn matches_prefix_ci(after: &str, candidate: &str) -> Option<usize> {
    let mut after_chars = after.char_indices();
    let mut matched_end = 0;
    for cc in candidate.chars() {
        let (byte_pos, ac) = after_chars.next()?;
        if !ac.to_lowercase().eq(cc.to_lowercase()) {
            return None;
        }
        matched_end = byte_pos + ac.len_utf8();
    }
    Some(matched_end)
}

/// A candidate matches only when the character right after it is absent or
/// not a word character, so `@bot10` does not match a registered `bot1`.
fn boundary_ok(after: &str, matched_len: usize) -> bool {
    match after[matched_len..].chars().next() {
        None => true,
        Some(c) => !is_word_char(c),
    }
}

/// Scans `text` for `@mention`s of participants in `registry`.
///
/// An `@` counts only when at the start of `text` or preceded by a character
/// that is not alphanumeric/`_` (so `mail@example.com` is not a mention).
/// After the `@`, every display name is tried first (longest first, so
/// `@Ada Lovelace` beats `@Ada` when both are registered), case
/// insensitively; then every raw id, case sensitively (ids are lowercase by
/// construction). The scan resumes immediately after each match, so spans
/// can never overlap.
#[allow(dead_code)] // wired up by #127/#132; exercised directly by this module's tests today
fn scan_mentions(text: &str, registry: &ParticipantRegistry) -> Vec<MentionSpan> {
    let mut display_names: Vec<(&str, &ParticipantId)> = registry
        .iter()
        .map(|p| (p.display_name.as_str(), &p.id))
        .collect();
    display_names.sort_by_key(|(name, _)| std::cmp::Reverse(name.chars().count()));

    let mut ids: Vec<&ParticipantId> = registry.iter().map(|p| &p.id).collect();
    ids.sort_by_key(|id| std::cmp::Reverse(id.as_str().len()));

    let mut spans = Vec::new();
    let mut i = 0usize;
    while i < text.len() {
        let Some(rel) = text[i..].find('@') else {
            break;
        };
        let at_pos = i + rel;
        let after_at = at_pos + 1; // '@' is one ASCII byte
        let preceded_ok = match text[..at_pos].chars().next_back() {
            None => true,
            Some(c) => !is_word_char(c),
        };
        let mut advanced = false;
        if preceded_ok {
            let after = &text[after_at..];
            let hit = display_names
                .iter()
                .find_map(|(name, id)| {
                    matches_prefix_ci(after, name)
                        .filter(|&len| boundary_ok(after, len))
                        .map(|len| (len, (*id).clone()))
                })
                .or_else(|| {
                    ids.iter().find_map(|id| {
                        after.strip_prefix(id.as_str()).and_then(|_| {
                            let len = id.as_str().len();
                            boundary_ok(after, len).then_some((len, (*id).clone()))
                        })
                    })
                });
            if let Some((len, id)) = hit {
                let end = after_at + len;
                spans.push(MentionSpan {
                    start: at_pos,
                    end,
                    id,
                });
                i = end;
                advanced = true;
            }
        }
        if !advanced {
            i = after_at;
        }
    }
    spans
}

/// Every participant mentioned in `text`, in order of first appearance,
/// deduplicated.
#[allow(dead_code)] // wired up by #127/#132
pub fn find_mentions(text: &str, registry: &ParticipantRegistry) -> Vec<ParticipantId> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    for span in scan_mentions(text, registry) {
        if seen.insert(span.id.clone()) {
            result.push(span.id);
        }
    }
    result
}

fn rewrite_mentions(
    text: &str,
    registry: &ParticipantRegistry,
    render: impl Fn(&ParticipantId, &ParticipantRegistry) -> String,
) -> String {
    let spans = scan_mentions(text, registry);
    if spans.is_empty() {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut cursor = 0;
    for span in &spans {
        result.push_str(&text[cursor..span.start]);
        result.push_str(&render(&span.id, registry));
        cursor = span.end;
    }
    result.push_str(&text[cursor..]);
    result
}

/// Rewrites every mention in `text` to its storage form, `@id`. Used by
/// #125's `speaker_id` messages and #132's routing.
#[allow(dead_code)] // wired up by #132
pub fn normalise_mentions(text: &str, registry: &ParticipantRegistry) -> String {
    rewrite_mentions(text, registry, |id, _registry| format!("@{}", id))
}

/// Rewrites every mention in `text` to its prompt form, `@Display Name`.
#[allow(dead_code)] // wired up by #127
pub fn render_mentions(text: &str, registry: &ParticipantRegistry) -> String {
    rewrite_mentions(text, registry, |id, registry| {
        format!("@{}", registry.display_name(id).unwrap_or(id.as_str()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_with_bot1_ada() -> ParticipantRegistry {
        let mut registry = ParticipantRegistry::solo("TestUser", "TestCompanion", None);
        registry
            .insert(Participant {
                id: ParticipantId::parse("bot1").unwrap(),
                display_name: "Ada".to_string(),
                kind: ParticipantKind::HostBot,
                avatar: None,
            })
            .unwrap();
        registry
    }

    // --- ParticipantId grammar ---

    #[test]
    fn id_grammar_accepts_valid_ids() {
        assert!(ParticipantId::parse("bot1").is_ok());
        assert!(ParticipantId::parse("a").is_ok());
        assert!(ParticipantId::parse("a123456789012345").is_ok()); // 16 chars
    }

    #[test]
    fn id_grammar_rejects_invalid_ids() {
        assert!(ParticipantId::parse("").is_err());
        assert!(ParticipantId::parse("Bot1").is_err());
        assert!(ParticipantId::parse("1bot").is_err());
        assert!(ParticipantId::parse("a1234567890123456").is_err()); // 17 chars
        assert!(ParticipantId::parse("bot-1").is_err());
        assert!(ParticipantId::parse("bötx").is_err());
    }

    // --- ParticipantRegistry ---

    #[test]
    fn solo_yields_user_then_char() {
        let registry = ParticipantRegistry::solo("Alice", "Bob", None);
        let ids: Vec<&ParticipantId> = registry.iter().map(|p| &p.id).collect();
        assert_eq!(ids, vec![&ParticipantId::USER, &ParticipantId::CHAR]);
    }

    #[test]
    fn insert_rejects_reserved_ids() {
        let mut registry = ParticipantRegistry::solo("Alice", "Bob", None);
        let err = registry
            .insert(Participant {
                id: ParticipantId::USER,
                display_name: "Someone".to_string(),
                kind: ParticipantKind::Human,
                avatar: None,
            })
            .unwrap_err();
        assert_eq!(err, ParticipantError::Reserved(ParticipantId::USER));
    }

    #[test]
    fn insert_rejects_duplicates() {
        let mut registry = registry_with_bot1_ada();
        let err = registry
            .insert(Participant {
                id: ParticipantId::parse("bot1").unwrap(),
                display_name: "Someone else".to_string(),
                kind: ParticipantKind::HostBot,
                avatar: None,
            })
            .unwrap_err();
        assert_eq!(
            err,
            ParticipantError::Duplicate(ParticipantId::parse("bot1").unwrap())
        );
    }

    #[test]
    fn remove_rejects_reserved_ids() {
        let mut registry = ParticipantRegistry::solo("Alice", "Bob", None);
        let err = registry.remove(&ParticipantId::USER).unwrap_err();
        assert_eq!(err, ParticipantError::Reserved(ParticipantId::USER));
    }

    #[test]
    fn iter_bots_skips_the_human() {
        let registry = registry_with_bot1_ada();
        let bot_ids: Vec<&ParticipantId> = registry.iter_bots().map(|p| &p.id).collect();
        assert_eq!(
            bot_ids,
            vec![&ParticipantId::CHAR, &ParticipantId::parse("bot1").unwrap()]
        );
    }

    #[test]
    fn insertion_order_preserved_after_remove_and_reinsert() {
        let mut registry = registry_with_bot1_ada();
        registry
            .remove(&ParticipantId::parse("bot1").unwrap())
            .unwrap();
        registry
            .insert(Participant {
                id: ParticipantId::parse("bot2").unwrap(),
                display_name: "Grace".to_string(),
                kind: ParticipantKind::HostBot,
                avatar: None,
            })
            .unwrap();
        registry
            .insert(Participant {
                id: ParticipantId::parse("bot1").unwrap(),
                display_name: "Ada".to_string(),
                kind: ParticipantKind::HostBot,
                avatar: None,
            })
            .unwrap();
        let ids: Vec<String> = registry.iter().map(|p| p.id.to_string()).collect();
        assert_eq!(ids, vec!["user", "char", "bot2", "bot1"]);
    }

    // --- expand_placeholders ---

    #[test]
    fn expand_placeholders_expands_reserved_and_joiner_ids() {
        let registry = registry_with_bot1_ada();
        assert_eq!(
            expand_placeholders("{{user}} and {{char}} and {{bot1}}", &registry),
            "TestUser and TestCompanion and Ada"
        );
    }

    #[test]
    fn expand_placeholders_leaves_unknown_placeholders_untouched() {
        let registry = registry_with_bot1_ada();
        assert_eq!(
            expand_placeholders("{{bot9}} and {{companion}}", &registry),
            "{{bot9}} and {{companion}}"
        );
    }

    #[test]
    fn expand_placeholders_does_not_reexpand_a_substituted_display_name() {
        let mut registry = ParticipantRegistry::solo("{{user}}", "TestCompanion", None);
        // A display name that is itself a placeholder string stays literal:
        // the scan advances past the original `{{user}}` bytes, not the
        // freshly substituted ones.
        registry
            .rename(&ParticipantId::USER, "{{user}}", None)
            .unwrap();
        assert_eq!(expand_placeholders("{{user}}", &registry), "{{user}}");
    }

    #[test]
    fn expand_placeholders_matches_the_old_replace_chain_on_realistic_persona_text() {
        let registry = ParticipantRegistry::solo("Alice", "Bob", None);
        let persona = "{{char}} is a friendly companion who talks to {{user}} every day.";
        let old_style = persona
            .replace("{{char}}", "Bob")
            .replace("{{user}}", "Alice");
        assert_eq!(expand_placeholders(persona, &registry), old_style);
    }

    // --- mentions ---

    #[test]
    fn find_mentions_finds_id_and_display_name_mentions() {
        let mut registry = ParticipantRegistry::solo("TestUser", "TestCompanion", None);
        registry
            .insert(Participant {
                id: ParticipantId::parse("bot1").unwrap(),
                display_name: "Bot One".to_string(),
                kind: ParticipantKind::HostBot,
                avatar: None,
            })
            .unwrap();
        registry
            .insert(Participant {
                id: ParticipantId::parse("bot2").unwrap(),
                display_name: "Ada".to_string(),
                kind: ParticipantKind::HostBot,
                avatar: None,
            })
            .unwrap();
        let mentions = find_mentions("hey @bot1 and @Ada", &registry);
        assert_eq!(
            mentions,
            vec![
                ParticipantId::parse("bot1").unwrap(),
                ParticipantId::parse("bot2").unwrap()
            ]
        );
    }

    #[test]
    fn find_mentions_is_case_insensitive_on_display_names() {
        let registry = registry_with_bot1_ada();
        assert_eq!(
            find_mentions("hi @ada", &registry),
            vec![ParticipantId::parse("bot1").unwrap()]
        );
    }

    #[test]
    fn find_mentions_prefers_the_longer_display_name() {
        let mut registry = ParticipantRegistry::solo("TestUser", "TestCompanion", None);
        registry
            .insert(Participant {
                id: ParticipantId::parse("bot1").unwrap(),
                display_name: "Ada".to_string(),
                kind: ParticipantKind::HostBot,
                avatar: None,
            })
            .unwrap();
        registry
            .insert(Participant {
                id: ParticipantId::parse("bot2").unwrap(),
                display_name: "Ada Lovelace".to_string(),
                kind: ParticipantKind::HostBot,
                avatar: None,
            })
            .unwrap();
        assert_eq!(
            find_mentions("@Ada Lovelace is here", &registry),
            vec![ParticipantId::parse("bot2").unwrap()]
        );
    }

    #[test]
    fn find_mentions_ignores_email_addresses() {
        let registry = registry_with_bot1_ada();
        assert!(find_mentions("mail@example.com", &registry).is_empty());
    }

    #[test]
    fn find_mentions_allows_trailing_punctuation() {
        let registry = registry_with_bot1_ada();
        assert_eq!(
            find_mentions("@bot1!", &registry),
            vec![ParticipantId::parse("bot1").unwrap()]
        );
    }

    #[test]
    fn find_mentions_does_not_match_a_longer_id_as_a_prefix() {
        let registry = registry_with_bot1_ada();
        assert!(find_mentions("@bot10", &registry).is_empty());
    }

    #[test]
    fn find_mentions_handles_multibyte_display_names() {
        let mut registry = ParticipantRegistry::solo("TestUser", "TestCompanion", None);
        registry
            .insert(Participant {
                id: ParticipantId::parse("bot1").unwrap(),
                display_name: "Zoë".to_string(),
                kind: ParticipantKind::HostBot,
                avatar: None,
            })
            .unwrap();
        let text = "hi @Zoë!";
        let normalised = normalise_mentions(text, &registry);
        assert_eq!(normalised, "hi @bot1!");
        let rendered = render_mentions(&normalised, &registry);
        assert_eq!(rendered, "hi @Zoë!");
    }

    #[test]
    fn normalise_render_normalise_round_trips() {
        let registry = registry_with_bot1_ada();
        let text = "hey @Ada, how are you @bot1?";
        let normalised_once = normalise_mentions(text, &registry);
        let rendered = render_mentions(&normalised_once, &registry);
        let normalised_twice = normalise_mentions(&rendered, &registry);
        assert_eq!(normalised_once, normalised_twice);
    }
}
