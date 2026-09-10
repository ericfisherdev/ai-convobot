// Pure, React-free state for the draft review card (#180). Mirrors
// `lib/roundStream.ts`'s pattern: all of the review card's edit/toggle logic
// lives here so it is unit-testable without a DOM, and `CompactionDraftCard`
// only wires these functions to props/handlers.
import {
  CompactionDraftReview,
  CompactionFact,
  CheckpointDetail,
  FactCategory,
  ItemReview,
  RejectedItem,
} from '../interfaces/Compaction';

// `rule`/`key_quote` items store their verbatim quote in `Fact.text` (the
// backend's `compaction::review::apply_review` maps `ItemReview.quote` onto
// `FactDraft.text` for exactly these two categories); every other category
// edits `Fact.text` directly via `ItemReview.text`. `ReviewItem` mirrors that
// split so the card can label the field "quote" only where it means one.
const isQuoteCategory = (category: FactCategory): boolean =>
  category === 'rule' || category === 'key_quote';

export interface ReviewItem {
  fact: CompactionFact;
  accepted: boolean;
  text: string;
  quote?: string;
  // Set by `applyServerRejection` when a commit attempt's `422` named this
  // item; distinct from `fact.rejected_reason`, which is the validator's
  // verdict at extraction time. `null` until a commit is attempted.
  serverReason: string | null;
}

export interface ReviewState {
  items: ReviewItem[];
  summary: string;
}

// One entry per category, in the fixed order the card renders sections.
const CATEGORY_ORDER: FactCategory[] = [
  'companion_state',
  'user_state',
  'milestone',
  'backstory',
  'open_thread',
  'rule',
  'key_quote',
  'person',
];

// `accepted` starts false for anything the validator already rejected
// (`rejected_reason !== null`) so a struck item stays struck through until
// the user explicitly accepts it.
export function initialReviewState(draft: CheckpointDetail): ReviewState {
  return {
    summary: draft.summary_text ?? '',
    items: draft.facts.map((fact) => ({
      fact,
      accepted: fact.rejected_reason === null,
      text: fact.text,
      quote: isQuoteCategory(fact.category) ? fact.text : undefined,
      serverReason: null,
    })),
  };
}

function updateItem(
  state: ReviewState,
  factId: number,
  update: (item: ReviewItem) => ReviewItem
): ReviewState {
  return {
    ...state,
    items: state.items.map((item) => (item.fact.id === factId ? update(item) : item)),
  };
}

export function toggleAccepted(state: ReviewState, factId: number): ReviewState {
  return updateItem(state, factId, (item) => ({ ...item, accepted: !item.accepted }));
}

export function editItemText(state: ReviewState, factId: number, text: string): ReviewState {
  return updateItem(state, factId, (item) => ({ ...item, text }));
}

export function editItemQuote(state: ReviewState, factId: number, quote: string): ReviewState {
  return updateItem(state, factId, (item) => ({ ...item, quote }));
}

export function editSummary(state: ReviewState, summary: string): ReviewState {
  return { ...state, summary };
}

// A commit attempt's `422` body -- `compaction::review::ReviewError::Rejected`
// serialises as a JSON array of `RejectedItem`s, one per item that failed
// re-validation (more than one edited item can fail in the same attempt).
// Every item's `serverReason` is recomputed from `rejections`, not just the
// named ones: an item this attempt no longer rejects (the user fixed it, or
// it was never touched) must clear whatever reason an earlier attempt left
// behind. A freshly rejected item is also struck (`accepted: false`), same
// as a validator rejection at extraction time -- the backend only
// re-validates `accepted: true` items, so a server rejection always lands
// on one that was checked; forcing it back to unchecked makes "accept
// anyway" an explicit choice instead of silently resubmitting the same
// failing edit. No other item, and no other field on the rejected item
// (its edited `text`/`quote`), is touched.
export function applyServerRejection(
  state: ReviewState,
  rejections: RejectedItem[]
): ReviewState {
  const reasonByItemId = new Map(rejections.map((r) => [r.item_id, r.reason]));
  return {
    ...state,
    items: state.items.map((item) => {
      const serverReason = reasonByItemId.get(item.fact.id) ?? null;
      return {
        ...item,
        serverReason,
        accepted: serverReason !== null ? false : item.accepted,
      };
    }),
  };
}

// One entry per item, `text`/`quote` included only when they differ from
// the fact's stored value -- an untouched item is sent as a bare
// `{ id, accepted }` so the backend's `apply_review` leaves it exactly as
// extracted.
export function toReviewPayload(state: ReviewState): CompactionDraftReview {
  const items: ItemReview[] = state.items.map((item) => {
    const review: ItemReview = { id: item.fact.id, accepted: item.accepted };
    if (isQuoteCategory(item.fact.category)) {
      if (item.quote !== undefined && item.quote !== item.fact.text) {
        review.quote = item.quote;
      }
    } else if (item.text !== item.fact.text) {
      review.text = item.text;
    }
    return review;
  });
  return { items, summary: state.summary };
}

export function groupByCategory(items: ReviewItem[]): Map<FactCategory, ReviewItem[]> {
  const grouped = new Map<FactCategory, ReviewItem[]>();
  for (const category of CATEGORY_ORDER) {
    const inCategory = items.filter((item) => item.fact.category === category);
    if (inCategory.length > 0) {
      grouped.set(category, inCategory);
    }
  }
  return grouped;
}
