//! Pure @mention routing decisions for the round orchestrator (#131).
//!
//! Two questions live here, both answerable from plain data with no
//! `Database` or HTTP dependency (like `attitude_engine.rs`/`turn_slot.rs`):
//! who speaks first in a round ([`plan_round`]), and, once a speaker's reply
//! is in hand, who its `@mention`s add to the round ([`schedule_follow_ups`]).
//! [`RoutingPolicy::max_followup_depth`] bounds how many mention-hops a
//! chain can run before it stops scheduling anyone, which is what keeps a
//! `@bot1`/`@bot2` back-and-forth from running forever.

use std::collections::{HashSet, VecDeque};

use crate::participants::{find_mentions, ParticipantId, ParticipantKind, ParticipantRegistry};

/// What the round orchestrator needs from `ConfigView::mention_followup_depth`
/// (#128). Depth `0` means mentions in bot replies never schedule anyone —
/// only the plan built directly from the user's message (or joiners, in the
/// no-mention default) ever speaks.
#[derive(Clone, Copy, Debug)]
pub struct RoutingPolicy {
    pub max_followup_depth: usize,
}

/// One speaker queued to talk this round, and how many mention-hops deep it
/// was scheduled. Depth `0` is a speaker scheduled by the user's message (or
/// the default plan); a follow-up triggered by a depth-`d` reply is `d + 1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledSpeaker {
    pub id: ParticipantId,
    pub depth: usize,
}

/// Who speaks, and in what order, for one round. Starts as a fixed list from
/// [`plan_round`] but can grow while the round runs, as bot-to-bot mentions
/// schedule follow-ups through [`schedule_follow_ups`].
pub struct RoundPlan {
    /// The speaking order, front to back.
    pending: VecDeque<ScheduledSpeaker>,
    /// Every `(id, depth)` pair ever queued, so a speaker is never scheduled
    /// twice at the same depth even after its entry has been popped.
    scheduled: HashSet<(ParticipantId, usize)>,
}

impl RoundPlan {
    fn empty() -> Self {
        RoundPlan {
            pending: VecDeque::new(),
            scheduled: HashSet::new(),
        }
    }

    /// Builds a plan from a fixed speaker order, all at depth `0`. What
    /// [`plan_round`] builds both branches from, and what `round.rs`'s own
    /// tests use to construct a plan without going through mention parsing.
    pub(crate) fn from_speakers(ids: impl IntoIterator<Item = ParticipantId>) -> Self {
        let mut plan = RoundPlan::empty();
        for id in ids {
            plan.schedule(id, 0);
        }
        plan
    }

    /// Queues `id` at `depth` at the back of the plan, unless that exact
    /// `(id, depth)` pair has already been queued once (even if it has since
    /// been popped). Returns whether it was newly queued.
    fn schedule(&mut self, id: ParticipantId, depth: usize) -> bool {
        if self.scheduled.insert((id.clone(), depth)) {
            self.pending.push_back(ScheduledSpeaker { id, depth });
            true
        } else {
            false
        }
    }

    /// Pops the next speaker to run, front first.
    pub fn next_speaker(&mut self) -> Option<ScheduledSpeaker> {
        self.pending.pop_front()
    }

    /// Whether `id` still has an entry waiting at any depth.
    pub fn is_pending(&self, id: &ParticipantId) -> bool {
        self.pending.iter().any(|speaker| &speaker.id == id)
    }

    /// The ids still waiting, front to back. For this module's and
    /// `round.rs`'s tests; no production caller needs the order without
    /// also popping it, so there is none today.
    #[allow(dead_code)]
    pub fn pending_ids(&self) -> Vec<ParticipantId> {
        self.pending
            .iter()
            .map(|speaker| speaker.id.clone())
            .collect()
    }
}

/// Whether `id` is a bot (`HostBot` or `RemoteBot`) `registry` knows about.
/// The only other kind, `Human`, is always `user`, so this is equivalent to
/// (and reads more clearly than) an `id != ParticipantId::USER` check.
fn is_bot(id: &ParticipantId, registry: &ParticipantRegistry) -> bool {
    matches!(
        registry.get(id).map(|p| &p.kind),
        Some(ParticipantKind::HostBot) | Some(ParticipantKind::RemoteBot)
    )
}

/// The round's speaker order with no mentions to route by: the host
/// companion, then every connected joiner, in join order. A joiner is only
/// ever present in `registry` while its socket is connected (`host.rs`
/// inserts it on join and removes it on disconnect), so `registry.iter_bots()`
/// already is the connected set.
fn default_plan(registry: &ParticipantRegistry) -> RoundPlan {
    RoundPlan::from_speakers(registry.iter_bots().map(|p| p.id.clone()))
}

/// Builds a round's starting speaker order from the user's message.
///
/// When `user_message` `@mentions` one or more bots, the plan is exactly
/// those bots, in registry turn order (`char` first, then `iter_bots()` join
/// order), not the order they appear in the text — so `"@bot2 and @bot1"`
/// still runs `bot1` before `bot2` if that is their join order. A mention of
/// `user` alone, or of nobody registered, falls back to [`default_plan`].
pub fn plan_round(
    user_message: &str,
    registry: &ParticipantRegistry,
    _policy: &RoutingPolicy,
) -> RoundPlan {
    let mentioned: HashSet<ParticipantId> = find_mentions(user_message, registry)
        .into_iter()
        .filter(|id| is_bot(id, registry))
        .collect();

    if mentioned.is_empty() {
        return default_plan(registry);
    }

    RoundPlan::from_speakers(
        registry
            .iter_bots()
            .filter(|p| mentioned.contains(&p.id))
            .map(|p| p.id.clone()),
    )
}

/// Reads `reply`'s `@mention`s and appends whichever ones should get a
/// follow-up turn, in registry turn order. Returns what it appended, for
/// logging and tests.
///
/// Nothing is appended once `speaker.depth` reaches `policy.max_followup_depth`
/// — the termination guarantee that keeps a mention chain from running past
/// that many hops past the user's message. Otherwise a mention is dropped
/// when it names `user`, the speaker itself, a bot already pending in the
/// plan (it will see this reply when its already-scheduled turn runs, so a
/// second turn would be a duplicate), or a `(id, depth + 1)` the plan has
/// already scheduled (so two speakers mentioning the same bot only queue it
/// once).
pub fn schedule_follow_ups(
    plan: &mut RoundPlan,
    reply: &str,
    speaker: &ScheduledSpeaker,
    registry: &ParticipantRegistry,
    policy: &RoutingPolicy,
) -> Vec<ParticipantId> {
    if speaker.depth >= policy.max_followup_depth {
        return Vec::new();
    }
    let next_depth = speaker.depth + 1;

    let mentioned: Vec<ParticipantId> = find_mentions(reply, registry)
        .into_iter()
        .filter(|id| is_bot(id, registry))
        .filter(|id| *id != speaker.id)
        .filter(|id| !plan.is_pending(id))
        .collect();

    let mut appended = Vec::new();
    for p in registry.iter_bots() {
        if mentioned.contains(&p.id) && plan.schedule(p.id.clone(), next_depth) {
            appended.push(p.id.clone());
        }
    }
    appended
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::participants::{AvatarRef, Participant, ParticipantKind};

    fn bot(id: &str) -> ParticipantId {
        ParticipantId::parse(id).unwrap()
    }

    fn insert_bot(registry: &mut ParticipantRegistry, id: &str, display_name: &str) {
        registry
            .insert(Participant {
                id: bot(id),
                display_name: display_name.to_string(),
                kind: ParticipantKind::HostBot,
                avatar: None,
            })
            .unwrap();
    }

    /// `user` ("Eric"), `char` ("Ada"), `bot1` ("Bob"), `bot2` ("Cleo"), in
    /// that join order.
    fn fixture() -> ParticipantRegistry {
        let mut registry = ParticipantRegistry::solo("Eric", "Ada", None::<AvatarRef>);
        insert_bot(&mut registry, "bot1", "Bob");
        insert_bot(&mut registry, "bot2", "Cleo");
        registry
    }

    fn policy(max_followup_depth: usize) -> RoutingPolicy {
        RoutingPolicy { max_followup_depth }
    }

    fn scheduled(id: &str, depth: usize) -> ScheduledSpeaker {
        ScheduledSpeaker { id: bot(id), depth }
    }

    // --- plan_round ---

    #[test]
    fn user_mention_of_one_bot_routes_only_to_it() {
        let plan = plan_round("@bot1 hi", &fixture(), &policy(1));
        assert_eq!(plan.pending_ids(), vec![bot("bot1")]);
    }

    #[test]
    fn user_mention_by_display_name_routes_the_same() {
        let plan = plan_round("@Bob hi", &fixture(), &policy(1));
        assert_eq!(plan.pending_ids(), vec![bot("bot1")]);
    }

    #[test]
    fn user_mention_of_two_bots_keeps_turn_order() {
        let plan = plan_round("@bot2 and @bot1", &fixture(), &policy(1));
        assert_eq!(plan.pending_ids(), vec![bot("bot1"), bot("bot2")]);
    }

    #[test]
    fn user_mention_of_char_only_routes_to_char() {
        let plan = plan_round("@Ada?", &fixture(), &policy(1));
        assert_eq!(plan.pending_ids(), vec![ParticipantId::CHAR]);
    }

    #[test]
    fn unknown_mention_falls_back_to_the_default_plan() {
        let plan = plan_round("@bot9 hi", &fixture(), &policy(1));
        assert_eq!(
            plan.pending_ids(),
            vec![ParticipantId::CHAR, bot("bot1"), bot("bot2")]
        );
    }

    #[test]
    fn mention_of_user_alone_falls_back_to_the_default_plan() {
        let plan = plan_round("@Eric hi", &fixture(), &policy(1));
        assert_eq!(
            plan.pending_ids(),
            vec![ParticipantId::CHAR, bot("bot1"), bot("bot2")]
        );
    }

    // --- schedule_follow_ups ---

    #[test]
    fn bot_mentioning_another_bot_at_depth_0_adds_one_follow_up() {
        let registry = fixture();
        let mut plan = plan_round("@char hi", &registry, &policy(1));
        let char_speaker = plan.next_speaker().unwrap();
        assert_eq!(
            char_speaker,
            ScheduledSpeaker {
                id: ParticipantId::CHAR,
                depth: 0
            }
        );

        let appended = schedule_follow_ups(
            &mut plan,
            "@bot1 do you agree?",
            &ScheduledSpeaker {
                id: ParticipantId::CHAR,
                depth: 0,
            },
            &registry,
            &policy(1),
        );
        assert_eq!(appended, vec![bot("bot1")]);
        assert_eq!(plan.pending_ids(), vec![bot("bot1")]);
    }

    #[test]
    fn mention_of_a_bot_still_pending_adds_nothing() {
        let registry = fixture();
        let mut plan = default_plan(&registry);
        let popped = plan.next_speaker().unwrap();
        assert_eq!(popped.id, ParticipantId::CHAR);
        assert!(plan.is_pending(&bot("bot1")));

        let appended = schedule_follow_ups(
            &mut plan,
            "@bot1 what do you think?",
            &popped,
            &registry,
            &policy(1),
        );
        assert!(appended.is_empty());
        assert_eq!(plan.pending_ids(), vec![bot("bot1"), bot("bot2")]);
    }

    #[test]
    fn follow_up_mentioning_back_does_not_add_a_second_at_depth_1() {
        let registry = fixture();
        let mut plan = RoundPlan::from_speakers([ParticipantId::CHAR]);
        let char_speaker = plan.next_speaker().unwrap();
        // Depth 1 (the fixture's default policy): `char@0` mentioning
        // `bot1` is still within budget, so `bot1@1` gets scheduled.
        schedule_follow_ups(
            &mut plan,
            "@bot1 do you agree?",
            &char_speaker,
            &registry,
            &policy(1),
        );
        let bot1_speaker = plan.next_speaker().unwrap();
        assert_eq!(bot1_speaker, scheduled("bot1", 1));

        // `bot1@1` is already at the policy's depth cap, so its own
        // `@Ada` mention of the speaker that scheduled it schedules nothing,
        // regardless of what it mentions.
        let appended =
            schedule_follow_ups(&mut plan, "@Ada sure", &bot1_speaker, &registry, &policy(1));
        assert!(appended.is_empty());
        assert!(plan.pending_ids().is_empty());
    }

    #[test]
    fn two_speakers_mentioning_the_same_bot_schedule_it_once() {
        let registry = fixture();
        let mut plan = RoundPlan::from_speakers([ParticipantId::CHAR, bot("bot2")]);
        let char_speaker = ScheduledSpeaker {
            id: ParticipantId::CHAR,
            depth: 0,
        };
        let bot2_speaker = ScheduledSpeaker {
            id: bot("bot2"),
            depth: 0,
        };

        let first =
            schedule_follow_ups(&mut plan, "@bot1 hi", &char_speaker, &registry, &policy(1));
        assert_eq!(first, vec![bot("bot1")]);

        let second = schedule_follow_ups(
            &mut plan,
            "@bot1 also hi",
            &bot2_speaker,
            &registry,
            &policy(1),
        );
        assert!(second.is_empty());

        assert_eq!(
            plan.pending_ids()
                .iter()
                .filter(|id| **id == bot("bot1"))
                .count(),
            1
        );
    }

    #[test]
    fn a_bot_mentioning_itself_adds_nothing() {
        let registry = fixture();
        let mut plan = RoundPlan::from_speakers([bot("bot1")]);
        let bot1_speaker = ScheduledSpeaker {
            id: bot("bot1"),
            depth: 0,
        };
        let appended = schedule_follow_ups(
            &mut plan,
            "as @bot1 I think so",
            &bot1_speaker,
            &registry,
            &policy(1),
        );
        assert!(appended.is_empty());
    }

    #[test]
    fn depth_zero_policy_never_schedules_follow_ups() {
        let registry = fixture();
        let mut plan = RoundPlan::from_speakers([ParticipantId::CHAR]);
        let char_speaker = ScheduledSpeaker {
            id: ParticipantId::CHAR,
            depth: 0,
        };
        let appended =
            schedule_follow_ups(&mut plan, "@bot1 hi", &char_speaker, &registry, &policy(0));
        assert!(appended.is_empty());
    }

    #[test]
    fn depth_two_policy_allows_exactly_two_hops() {
        let registry = fixture();
        let p = policy(2);
        let mut plan = RoundPlan::from_speakers([ParticipantId::CHAR]);

        let char_speaker = plan.next_speaker().unwrap();
        let first = schedule_follow_ups(&mut plan, "@bot1 hi", &char_speaker, &registry, &p);
        assert_eq!(first, vec![bot("bot1")]);

        let bot1_speaker = plan.next_speaker().unwrap();
        assert_eq!(bot1_speaker.depth, 1);
        let second = schedule_follow_ups(&mut plan, "@char sure", &bot1_speaker, &registry, &p);
        assert_eq!(second, vec![ParticipantId::CHAR]);

        let char_speaker_2 = plan.next_speaker().unwrap();
        assert_eq!(char_speaker_2.depth, 2);
        let third = schedule_follow_ups(&mut plan, "@bot1 again", &char_speaker_2, &registry, &p);
        assert!(third.is_empty());

        assert!(plan.next_speaker().is_none());
    }
}
