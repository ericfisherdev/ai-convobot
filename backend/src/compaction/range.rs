//! Pure, I/O-free decision of *which* messages a queued compaction draft
//! (#172) should span, once [`crate::compaction::trigger::should_compact`]
//! has decided one should be queued at all.

use crate::compaction::types::CompactionTrigger;
use crate::compaction::MessageRef;

/// A checkpoint's message span: `[from_id, through_id]`, both inclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionRange {
    pub from_id: i32,
    pub through_id: i32,
}

/// How many of `eligible_len` eligible messages a checkpoint's range may
/// cover before the last `short_term_mem` are excluded: those stay in the
/// live prompt window regardless of trigger or of how the range's start was
/// chosen. Shared by [`select_range`] and [`select_recompaction_range`] so
/// the "never touch the short-term tail" rule has exactly one
/// implementation.
fn kept_boundary(eligible_len: usize, short_term_mem: usize) -> usize {
    eligible_len.saturating_sub(short_term_mem)
}

/// Picks the range a new draft should cover, or `None` when there is not
/// enough uncompacted material yet.
///
/// `messages` is the full uncompacted tail as the caller has it; this
/// function is what actually enforces "never overlaps a committed
/// checkpoint" by filtering to `id > compacted_through` itself, rather than
/// trusting the caller to have done so already. The last `short_term_mem`
/// messages are never included, regardless of trigger: those stay in the
/// live prompt window. For [`CompactionTrigger::SceneBreak`], the range is
/// additionally cut before the human turn that just finished the round, so
/// `through_id` never includes it.
pub fn select_range(
    compacted_through: Option<i32>,
    messages: &[MessageRef],
    short_term_mem: usize,
    min_messages: usize,
    trigger: CompactionTrigger,
) -> Option<CompactionRange> {
    let eligible: Vec<&MessageRef> = messages
        .iter()
        .filter(|m| compacted_through.is_none_or(|through| m.id > through))
        .collect();
    debug_assert!(
        eligible.windows(2).all(|pair| pair[0].id < pair[1].id),
        "messages must be sorted ascending by id with no duplicates"
    );

    let mut end = kept_boundary(eligible.len(), short_term_mem);

    if trigger == CompactionTrigger::SceneBreak {
        if let Some(break_idx) = eligible.iter().rposition(|m| m.is_human) {
            end = end.min(break_idx);
        }
    }

    // `end == 0` on top of `end < min_messages`: every real `CompactionConfig`
    // has `min_messages >= 1`, which already implies this, but a
    // `min_messages: 0` caller must not fall through to `eligible[end - 1]`
    // with nothing eligible.
    if end == 0 || end < min_messages {
        return None;
    }

    Some(CompactionRange {
        from_id: eligible[0].id,
        through_id: eligible[end - 1].id,
    })
}

/// Picks the range a re-compaction from a stale checkpoint should cover
/// (#181): starts at `oldest_stale_from` (the earliest `Stale` checkpoint's
/// own start, so the new draft re-covers every stale range at once) and
/// keeps the same short-term tail [`select_range`] always excludes. No
/// trigger/`min_messages` gate — a re-compaction is always worth running
/// once a stale checkpoint exists, however small the range.
///
/// `None` when no message with `id >= oldest_stale_from` survives (every
/// message in the stale range was deleted) or when the short-term-tail cut
/// leaves nothing before it either; the caller reports this as a 409 rather
/// than queuing a doomed draft.
pub fn select_recompaction_range(
    oldest_stale_from: i32,
    messages: &[MessageRef],
    short_term_mem: usize,
) -> Option<CompactionRange> {
    let eligible: Vec<&MessageRef> = messages
        .iter()
        .filter(|m| m.id >= oldest_stale_from)
        .collect();
    debug_assert!(
        eligible.windows(2).all(|pair| pair[0].id < pair[1].id),
        "messages must be sorted ascending by id with no duplicates"
    );

    let end = kept_boundary(eligible.len(), short_term_mem);
    if end == 0 {
        return None;
    }

    Some(CompactionRange {
        from_id: oldest_stale_from,
        through_id: eligible[end - 1].id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(id: i32, is_human: bool) -> MessageRef {
        MessageRef {
            id,
            is_human,
            tokens: 1,
        }
    }

    /// 10 alternating messages, ids 1..=10, human/AI/human/AI/...
    fn ten_alternating() -> Vec<MessageRef> {
        (1..=10).map(|id| msg(id, id % 2 == 1)).collect()
    }

    #[test]
    fn fresh_chat_picks_from_the_first_id() {
        let messages = ten_alternating();
        let range = select_range(None, &messages, 2, 2, CompactionTrigger::Threshold).unwrap();
        assert_eq!(range.from_id, 1);
        assert_eq!(range.through_id, 8);
    }

    #[test]
    fn a_second_compaction_starts_after_compacted_through() {
        let messages = ten_alternating();
        let range = select_range(Some(4), &messages, 2, 2, CompactionTrigger::Threshold).unwrap();
        assert_eq!(range.from_id, 5);
        assert_eq!(range.through_id, 8);
    }

    #[test]
    fn messages_at_or_below_compacted_through_are_dropped_even_when_passed_in() {
        let messages = ten_alternating();
        let range = select_range(Some(4), &messages, 0, 2, CompactionTrigger::Threshold).unwrap();
        assert!(messages.iter().all(|m| m.id > 4 || range.from_id > m.id));
        assert_eq!(range.from_id, 5);
    }

    #[test]
    fn the_last_short_term_mem_ids_are_never_in_the_range_for_every_trigger() {
        for trigger in [
            CompactionTrigger::Threshold,
            CompactionTrigger::SceneBreak,
            CompactionTrigger::Manual,
        ] {
            let messages = ten_alternating();
            let range = select_range(None, &messages, 3, 2, trigger);
            if let Some(range) = range {
                assert!(
                    range.through_id <= 7,
                    "trigger {:?} must exclude the last 3 messages",
                    trigger
                );
            }
        }
    }

    #[test]
    fn scene_break_cuts_before_the_last_human_turn() {
        // ids 1..=10 alternating human/ai starting human; id 9 is the last
        // human turn.
        let messages = ten_alternating();
        let range = select_range(None, &messages, 0, 2, CompactionTrigger::SceneBreak).unwrap();
        assert_eq!(range.through_id, 8);
    }

    #[test]
    fn scene_break_also_respects_short_term_mem_when_that_is_stricter() {
        let messages = ten_alternating();
        // short_term_mem=5 excludes the last 5 ids (6..=10), which is a
        // stricter cut than the scene break (which would only exclude id 9,
        // 10).
        let range = select_range(None, &messages, 5, 2, CompactionTrigger::SceneBreak).unwrap();
        assert_eq!(range.through_id, 5);
    }

    #[test]
    fn a_range_under_min_messages_is_none() {
        let messages = ten_alternating();
        assert!(select_range(None, &messages, 2, 9, CompactionTrigger::Threshold).is_none());
    }

    #[test]
    fn short_term_mem_covering_the_whole_tail_is_none() {
        let messages = ten_alternating();
        assert!(select_range(None, &messages, 10, 2, CompactionTrigger::Threshold).is_none());
    }

    /// The one invariant every trigger and every `select_range`/
    /// `select_recompaction_range` caller relies on: the last `short_term_mem`
    /// messages are never part of a returned range, for any message count
    /// and any `compacted_through`. Looped rather than asserted once per
    /// fixed case, since `pop_latest_bot_reply_on` only ever removes the
    /// newest row (always inside this kept tail), so this is the whole
    /// guarantee that matters for #181's edit/delete stale-marking to stay
    /// consistent with what got compacted.
    #[test]
    fn select_range_never_returns_the_last_short_term_mem_messages() {
        for message_count in [0usize, 1, 5, 10, 20] {
            let messages: Vec<MessageRef> = (1..=message_count as i32)
                .map(|id| msg(id, id % 2 == 1))
                .collect();
            for compacted_through in [None, Some(0), Some(message_count as i32 - 1)] {
                for short_term_mem in 1..=10usize {
                    for trigger in [
                        CompactionTrigger::Threshold,
                        CompactionTrigger::SceneBreak,
                        CompactionTrigger::Manual,
                    ] {
                        if let Some(range) =
                            select_range(compacted_through, &messages, short_term_mem, 0, trigger)
                        {
                            if message_count >= short_term_mem {
                                let first_kept_id = messages[message_count - short_term_mem].id;
                                assert!(
                                    range.through_id < first_kept_id,
                                    "trigger {:?}, short_term_mem {}, compacted_through {:?}: \
                                     through_id {} must be before the kept tail starting at {}",
                                    trigger,
                                    short_term_mem,
                                    compacted_through,
                                    range.through_id,
                                    first_kept_id
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn recompaction_range_starts_at_the_oldest_stale_from_and_keeps_the_short_term_tail() {
        let messages = ten_alternating();
        let range = select_recompaction_range(3, &messages, 2).unwrap();
        assert_eq!(range.from_id, 3);
        assert_eq!(range.through_id, 8);
    }

    #[test]
    fn recompaction_range_is_none_when_every_message_at_or_after_the_stale_start_is_gone() {
        // Everything from id 3 onward (the stale checkpoint's own start) was
        // deleted; only earlier messages remain.
        let messages: Vec<MessageRef> = (1..=2).map(|id| msg(id, id % 2 == 1)).collect();
        assert!(select_recompaction_range(3, &messages, 2).is_none());
    }

    #[test]
    fn recompaction_range_is_none_when_the_short_term_tail_covers_everything_left() {
        let messages = ten_alternating();
        // Only ids 9, 10 survive at or after the stale start (8); with
        // short_term_mem=2 both are held back for the live prompt window.
        let range = select_recompaction_range(9, &messages, 2);
        assert!(range.is_none());
    }
}
