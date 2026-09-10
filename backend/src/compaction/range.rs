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

    let mut end = eligible.len().saturating_sub(short_term_mem);

    if trigger == CompactionTrigger::SceneBreak {
        if let Some(break_idx) = eligible.iter().rposition(|m| m.is_human) {
            end = end.min(break_idx);
        }
    }

    if end < min_messages {
        return None;
    }

    Some(CompactionRange {
        from_id: eligible[0].id,
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
}
