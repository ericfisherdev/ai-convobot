// Pure, React-free state for the draft review card (#180). Mirrors
// `lib/roundStream.ts`'s pattern: all of the review card's edit/toggle logic
// lives here so it is unit-testable without a DOM, and `CompactionDraftCard`
// only wires these functions to props/handlers.
import {
  CompactionDraftReview,
  CompactionFact,
  CheckpointDetail,
  ContradictionView,
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

// The "both sides together" rendering (#219 AC) for one contradiction: the
// thought it conflicts with, in full, plus the exact words the judge quoted.
// Shared by the summary and every flagged item, so the two read identically.
function formatContradiction(contradiction: ContradictionView): string {
  return `Contradicts the companion's note #${contradiction.thought_id}: "${contradiction.thought_text}" — "${contradiction.quote}"`;
}

export interface ReviewItem {
  fact: CompactionFact;
  accepted: boolean;
  text: string;
  quote?: string;
  // Set by `applyServerRejection` when a commit attempt's `422` named this
  // item; distinct from `fact.rejected_reason`, which is the validator's
  // verdict at extraction time. `null` until a commit is attempted.
  serverReason: string | null;
  // Set by `initialReviewState` (#219) when `CheckpointDetail.contradictions`
  // flags this item's own fact id: the full thought text and quote, so the
  // card can show both sides together. `fact.rejected_reason` already
  // carries the terse verdict string on its own; this is the richer detail
  // only the initial `GET` carries -- a `422` retry's `RejectedItem` has no
  // thought text to show, so this is never touched after the initial load.
  contradiction: string | null;
}

export interface ReviewState {
  items: ReviewItem[];
  summary: string;
  // Set by `initialReviewState` from the summary-level entry in
  // `CheckpointDetail.contradictions` (`fact_id: null`), or by
  // `applyServerRejection` from a commit attempt's `422` naming the summary
  // (`item_id: null`) -- either way, the reason text to show under the
  // Summary textarea. `null` when the summary is not currently flagged.
  summaryReason: string | null;
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
  const summaryContradiction = draft.contradictions.find((c) => c.fact_id === null);
  return {
    summary: draft.summary_text ?? '',
    summaryReason: summaryContradiction ? formatContradiction(summaryContradiction) : null,
    items: draft.facts.map((fact) => {
      const flagged = draft.contradictions.find((c) => c.fact_id === fact.id);
      return {
        fact,
        accepted: fact.rejected_reason === null,
        text: fact.text,
        quote: isQuoteCategory(fact.category) ? fact.text : undefined,
        serverReason: null,
        contradiction: flagged ? formatContradiction(flagged) : null,
      };
    }),
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

// Clears `summaryReason` (#219): the user is fixing the summary, and the
// server is what re-judges it at the next commit attempt.
export function editSummary(state: ReviewState, summary: string): ReviewState {
  return { ...state, summary, summaryReason: null };
}

// A commit attempt's `422` body -- `compaction::review::ReviewError::Rejected`
// serialises as a JSON array of `RejectedItem`s, one per item (or, per #219,
// the summary) that failed re-validation. Every item's `serverReason` (and
// `summaryReason`) is recomputed from `rejections`, not just the named ones:
// something this attempt no longer rejects (the user fixed it, or it was
// never touched) must clear whatever reason an earlier attempt left behind.
// A freshly rejected item is also struck (`accepted: false`), same as a
// validator rejection at extraction time -- the backend only re-validates
// `accepted: true` items, so a server rejection always lands on one that was
// checked; forcing it back to unchecked makes "accept anyway" an explicit
// choice instead of silently resubmitting the same failing edit. `item_id:
// null` (#219: the summary was flagged, at extraction time or by the
// commit-time re-check) routes to `summaryReason` instead of the item map --
// the summary has no `accepted` toggle to strike. No other item, and no
// other field on the rejected item (its edited `text`/`quote`/`summary`), is
// touched.
export function applyServerRejection(
  state: ReviewState,
  rejections: RejectedItem[]
): ReviewState {
  const summaryRejection = rejections.find((r) => r.item_id === null);
  const reasonByItemId = new Map(
    rejections
      .filter((r): r is RejectedItem & { item_id: number } => r.item_id !== null)
      .map((r) => [r.item_id, r.reason])
  );
  return {
    ...state,
    summaryReason: summaryRejection ? summaryRejection.reason : null,
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

// One entry per item the user actually changed (acceptance, text or
// quote); an untouched item is omitted so `apply_review` keeps it exactly
// as extracted -- sending it would overwrite a validator's
// `rejected_reason` with "struck at review" or re-validate a passing item.
export function toReviewPayload(state: ReviewState): CompactionDraftReview {
  const items: ItemReview[] = [];
  for (const item of state.items) {
    const review: ItemReview = { id: item.fact.id, accepted: item.accepted };
    const extractedAccepted = item.fact.rejected_reason === null;
    let edited = false;
    if (isQuoteCategory(item.fact.category)) {
      if (item.quote !== undefined && item.quote !== item.fact.text) {
        review.quote = item.quote;
        edited = true;
      }
    } else if (item.text !== item.fact.text) {
      review.text = item.text;
      edited = true;
    }
    if (edited || item.accepted !== extractedAccepted) {
      items.push(review);
    }
  }
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
