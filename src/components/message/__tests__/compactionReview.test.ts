import { describe, it, expect } from 'vitest';
import {
  applyServerRejection,
  editItemQuote,
  editItemText,
  editSummary,
  groupByCategory,
  initialReviewState,
  toggleAccepted,
  toReviewPayload,
} from '../compactionReview';
import { CheckpointDetail, CompactionFact } from '../../interfaces/Compaction';

const aFact = (overrides: Partial<CompactionFact> = {}): CompactionFact => ({
  id: 1,
  category: 'milestone',
  subject: 'user',
  text: 'moved in together',
  quote_speaker: null,
  sources: [3],
  replaces: [],
  relation_to: null,
  relation: null,
  canon: true,
  active: true,
  superseded_by: null,
  rejected_reason: null,
  ...overrides,
});

const aDraft = (
  facts: CompactionFact[],
  summary = 'a summary',
  contradictions: CheckpointDetail['contradictions'] = []
): CheckpointDetail => ({
  id: 1,
  from_message_id: 1,
  through_message_id: 10,
  status: 'draft',
  trigger: 'threshold',
  committed_at: null,
  needs_merge: false,
  extraction_error: null,
  phase: 'review',
  summary_text: summary,
  rolling_summary: null,
  facts,
  attitude: {
    current: {} as CheckpointDetail['attitude']['current'],
    rated: null,
    blended: null,
  },
  contradictions,
});

describe('compactionReview', () => {
  describe('initialReviewState', () => {
    it('starts a validator-rejected item unaccepted', () => {
      const rejected = aFact({ id: 2, rejected_reason: 'duplicate of an active fact' });
      const kept = aFact({ id: 3, rejected_reason: null });
      const state = initialReviewState(aDraft([rejected, kept]));

      expect(state.items.find((i) => i.fact.id === 2)?.accepted).toBe(false);
      expect(state.items.find((i) => i.fact.id === 3)?.accepted).toBe(true);
    });

    it('seeds the summary from the draft and quote items from their stored text', () => {
      const quoteFact = aFact({ id: 4, category: 'rule', text: 'never go to the lake alone' });
      const state = initialReviewState(aDraft([quoteFact], 'stored summary'));

      expect(state.summary).toBe('stored summary');
      expect(state.items[0].quote).toBe('never go to the lake alone');
    });

    it('seeds summaryReason from a summary-level contradiction, naming the thought and its quote (#219)', () => {
      const state = initialReviewState(
        aDraft([], 'they tended to each other\'s wounds', [
          {
            fact_id: null,
            thought_id: 12,
            thought_text: 'He stitched my wounds; he was not hurt',
            quote: "tended to each other's wounds",
          },
        ])
      );

      expect(state.summaryReason).toContain('12');
      expect(state.summaryReason).toContain('He stitched my wounds; he was not hurt');
      expect(state.summaryReason).toContain("tended to each other's wounds");
    });

    it('seeds an item\'s contradiction detail when its fact_id is flagged (#219)', () => {
      const flagged = aFact({ id: 30, text: 'has traveled far from the coast before' });
      const untouched = aFact({ id: 31 });
      const state = initialReviewState(
        aDraft([flagged, untouched], 'a summary', [
          {
            fact_id: 30,
            thought_id: 9,
            thought_text: 'Vi has never left the coast',
            quote: 'has traveled far from the coast before',
          },
        ])
      );

      const flaggedItem = state.items.find((i) => i.fact.id === 30);
      const untouchedItem = state.items.find((i) => i.fact.id === 31);
      expect(flaggedItem?.contradiction).toContain('Vi has never left the coast');
      expect(untouchedItem?.contradiction).toBeNull();
      expect(state.summaryReason).toBeNull();
    });

    it('leaves summaryReason null with no contradictions at all', () => {
      const state = initialReviewState(aDraft([aFact()]));
      expect(state.summaryReason).toBeNull();
    });
  });

  describe('toggleAccepted / editItemText / toReviewPayload', () => {
    it('strikes an item so its payload carries accepted: false', () => {
      const fact = aFact({ id: 5 });
      let state = initialReviewState(aDraft([fact]));
      state = toggleAccepted(state, 5);

      const payload = toReviewPayload(state);
      expect(payload.items).toEqual([{ id: 5, accepted: false }]);
    });

    it('includes edited text in the payload and omits untouched items entirely', () => {
      const edited = aFact({ id: 6, text: 'original text' });
      const untouched = aFact({ id: 7, text: 'unchanged text' });
      let state = initialReviewState(aDraft([edited, untouched]));
      state = editItemText(state, 6, 'a rewritten fact');

      const payload = toReviewPayload(state);
      const editedReview = payload.items.find((i) => i.id === 6);
      const untouchedReview = payload.items.find((i) => i.id === 7);

      // An untouched item must be *omitted*, not sent as a bare
      // `{ id, accepted }` -- `apply_review` only leaves a fact exactly as
      // extracted for ids absent from `request.items`; a present entry with
      // `accepted: false` overwrites the validator's own `rejected_reason`.
      expect(editedReview).toEqual({ id: 6, accepted: true, text: 'a rewritten fact' });
      expect(untouchedReview).toBeUndefined();
    });

    it('omits a validator-rejected item the user never touched, but sends one they explicitly re-accepted', () => {
      const struckUntouched = aFact({ id: 20, rejected_reason: 'duplicate of an active fact' });
      const struckReaccepted = aFact({ id: 21, rejected_reason: 'duplicate of an active fact' });
      let state = initialReviewState(aDraft([struckUntouched, struckReaccepted]));
      // Both start `accepted: false` (validator-rejected); only #21 is
      // explicitly toggled back on by the user.
      state = toggleAccepted(state, 21);

      const payload = toReviewPayload(state);

      expect(payload.items.find((i) => i.id === 20)).toBeUndefined();
      expect(payload.items.find((i) => i.id === 21)).toEqual({ id: 21, accepted: true });
    });

    it('edits a rule/key_quote item through its quote field, not text', () => {
      const quoteFact = aFact({ id: 8, category: 'key_quote', text: 'I will always be there' });
      let state = initialReviewState(aDraft([quoteFact]));
      state = editItemQuote(state, 8, 'I will always be there for you');

      const payload = toReviewPayload(state);
      expect(payload.items).toEqual([
        { id: 8, accepted: true, quote: 'I will always be there for you' },
      ]);
    });
  });

  describe('applyServerRejection', () => {
    it('marks the rejected item and leaves other items\' edits untouched', () => {
      const rejectedQuote = aFact({ id: 9, category: 'rule', text: 'never set foot near the lake' });
      const otherEdited = aFact({ id: 10, text: 'original text' });
      let state = initialReviewState(aDraft([rejectedQuote, otherEdited]));
      state = editItemQuote(state, 9, 'never set foot near the lake alone, ever');
      state = editItemText(state, 10, 'a completely different fact');

      state = applyServerRejection(state, [
        { item_id: 9, reason: 'quote is not verbatim in any cited message' },
      ]);

      const rejectedItem = state.items.find((i) => i.fact.id === 9);
      const otherItem = state.items.find((i) => i.fact.id === 10);

      expect(rejectedItem?.serverReason).toBe('quote is not verbatim in any cited message');
      expect(rejectedItem?.accepted).toBe(false);
      expect(rejectedItem?.quote).toBe('never set foot near the lake alone, ever');

      expect(otherItem?.serverReason).toBeNull();
      expect(otherItem?.text).toBe('a completely different fact');
      expect(otherItem?.accepted).toBe(true);
    });

    it('clears a stale serverReason once a later attempt no longer rejects the item', () => {
      const fact = aFact({ id: 11 });
      let state = initialReviewState(aDraft([fact]));
      state = applyServerRejection(state, [{ item_id: 11, reason: 'too long' }]);
      expect(state.items[0].serverReason).toBe('too long');

      state = applyServerRejection(state, []);
      expect(state.items[0].serverReason).toBeNull();
    });

    it('routes an item_id: null entry to summaryReason and strikes no item (#219)', () => {
      const fact = aFact({ id: 40 });
      let state = initialReviewState(aDraft([fact]));

      state = applyServerRejection(state, [
        { item_id: null, reason: 'contradicts running thought 12' },
      ]);

      expect(state.summaryReason).toBe('contradicts running thought 12');
      expect(state.items[0].serverReason).toBeNull();
      expect(state.items[0].accepted).toBe(true);
    });

    it('clears a stale summaryReason once a later attempt no longer rejects the summary', () => {
      let state = initialReviewState(aDraft([]));
      state = applyServerRejection(state, [
        { item_id: null, reason: 'contradicts running thought 3' },
      ]);
      expect(state.summaryReason).toBe('contradicts running thought 3');

      state = applyServerRejection(state, []);
      expect(state.summaryReason).toBeNull();
    });
  });

  describe('editSummary', () => {
    it('clears summaryReason when the user edits the summary (#219)', () => {
      const state = initialReviewState(
        aDraft([], 'stale summary', [
          { fact_id: null, thought_id: 1, thought_text: 'a note', quote: 'stale summary' },
        ])
      );
      expect(state.summaryReason).not.toBeNull();

      const edited = editSummary(state, 'a corrected summary');

      expect(edited.summaryReason).toBeNull();
      expect(edited.summary).toBe('a corrected summary');
    });
  });

  describe('groupByCategory', () => {
    it('orders sections companion_state, user_state, milestone, backstory, open_thread, rule, key_quote, person', () => {
      const facts = [
        aFact({ id: 1, category: 'person' }),
        aFact({ id: 2, category: 'rule' }),
        aFact({ id: 3, category: 'companion_state' }),
        aFact({ id: 4, category: 'milestone' }),
      ];
      const state = initialReviewState(aDraft(facts));

      const grouped = groupByCategory(state.items);
      expect(Array.from(grouped.keys())).toEqual(['companion_state', 'milestone', 'rule', 'person']);
    });
  });
});
