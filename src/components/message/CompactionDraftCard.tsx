import { useEffect, useState } from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Checkbox } from '../ui/checkbox';
import { Textarea } from '../ui/textarea';
import { cn } from '../../lib/utils';
import {
  applyServerRejection,
  editItemQuote,
  editItemText,
  editSummary,
  groupByCategory,
  initialReviewState,
  ReviewItem,
  ReviewState,
  toggleAccepted,
  toReviewPayload,
} from './compactionReview';
import { AttitudeRatings, CheckpointDetail, CompactionDraftReview, FactCategory, RejectedItem } from '../interfaces/Compaction';

const CATEGORY_LABELS: Record<FactCategory, string> = {
  companion_state: 'Companion state',
  user_state: 'User state',
  milestone: 'Milestone',
  backstory: 'Backstory',
  open_thread: 'Open thread',
  rule: 'Rule',
  key_quote: 'Key quote',
  person: 'Person',
};

const isQuoteCategory = (category: FactCategory): boolean =>
  category === 'rule' || category === 'key_quote';

// The eight dimensions `AttitudeRatings` (a draft's own extraction pass)
// covers, in the attitude row's display order. A subset of the full
// `AttitudeData` the companion tracks -- see `compaction::extract::AttitudeRatings`.
const RATED_DIMENSIONS: Array<{ key: keyof AttitudeRatings; label: string }> = [
  { key: 'trust', label: 'Trust' },
  { key: 'love', label: 'Love' },
  { key: 'fear', label: 'Fear' },
  { key: 'anger', label: 'Anger' },
  { key: 'joy', label: 'Joy' },
  { key: 'sorrow', label: 'Sorrow' },
  { key: 'suspicion', label: 'Suspicion' },
  { key: 'gratitude', label: 'Gratitude' },
];

// A two-tier jump (design doc edge case: a first compaction can blend a
// long way from the live attitude in one step) gets an arrow marker so it
// reads as a deliberate move, not noise.
const ATTITUDE_JUMP_THRESHOLD = 15;

interface CompactionDraftCardProps {
  draft: CheckpointDetail;
  // How many messages have arrived since `draft.through_message_id` -- 0
  // once the chat has not moved on from the draft's own range.
  messagesSinceDraft: number;
  busy: boolean;
  onCommit: (review: CompactionDraftReview) => void;
  onDiscard: () => void;
  onJumpToMessage: (id: number) => void;
  // The most recent commit attempt's `422` body, normalised to an array --
  // `compaction::review::ReviewError::Rejected` serialises as one
  // `RejectedItem` per re-validated item that still failed. Empty when no
  // attempt has failed (yet).
  serverRejections: RejectedItem[];
  // Read-only view used by `CompactionNotice`'s "Show notes" dialog: no
  // checkboxes, no editing, no Commit/Discard footer.
  readOnly?: boolean;
  // Controlled review state: `PendingDraftMarker` owns this so it survives
  // the mobile drawer (`vaul`'s `DrawerContent` unmounts its children on
  // close, which would otherwise discard every edit/strike/rejection the
  // moment the drawer closes). Falls back to an internal `useState` when
  // omitted -- `CompactionNotice`'s read-only dialog has no need to own it.
  state?: ReviewState;
  onStateChange?: (update: (prev: ReviewState) => ReviewState) => void;
}

function ItemRow({
  item,
  readOnly,
  onToggle,
  onEditText,
  onEditQuote,
  onJumpToMessage,
}: {
  item: ReviewItem;
  readOnly: boolean;
  onToggle: () => void;
  onEditText: (text: string) => void;
  onEditQuote: (quote: string) => void;
  onJumpToMessage: (id: number) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draftText, setDraftText] = useState(item.text);
  const [draftQuote, setDraftQuote] = useState(item.quote ?? '');
  const quoteCategory = isQuoteCategory(item.fact.category);
  const struck = !item.accepted;
  const reason = item.serverReason ?? item.fact.rejected_reason;

  const handleSave = () => {
    onEditText(draftText);
    if (quoteCategory) {
      onEditQuote(draftQuote);
    }
    setEditing(false);
  };

  const handleCancel = () => {
    setDraftText(item.text);
    setDraftQuote(item.quote ?? '');
    setEditing(false);
  };

  return (
    <div className="flex flex-col gap-1 py-2 border-b last:border-b-0">
      <div className="flex items-start gap-2">
        <Checkbox
          checked={item.accepted}
          onCheckedChange={onToggle}
          disabled={readOnly}
          aria-label={`Accept ${CATEGORY_LABELS[item.fact.category]} item`}
        />
        <div className="flex-1 min-w-0">
          {editing ? (
            <Textarea
              value={draftText}
              onChange={(e) => setDraftText(e.target.value)}
              className="text-xs"
            />
          ) : (
            <p className={cn('text-xs', struck && 'line-through text-muted-foreground')}>
              {item.text}
            </p>
          )}
          {quoteCategory && (
            editing ? (
              <Textarea
                value={draftQuote}
                onChange={(e) => setDraftQuote(e.target.value)}
                placeholder="Verbatim quote"
                className="text-xs mt-1"
              />
            ) : (
              <p className={cn('text-xs italic mt-1', struck && 'line-through text-muted-foreground')}>
                &ldquo;{item.quote}&rdquo;{item.fact.quote_speaker ? ` — ${item.fact.quote_speaker}` : ''}
              </p>
            )
          )}
          {reason && (
            <p className="text-[10px] text-destructive mt-1">{reason}</p>
          )}
          <div className="flex items-center gap-1 mt-1 flex-wrap">
            {item.fact.sources.map((sourceId) => (
              <button
                key={sourceId}
                type="button"
                className="text-[10px] underline text-muted-foreground hover:text-foreground"
                onClick={() => onJumpToMessage(sourceId)}
              >
                #{sourceId}
              </button>
            ))}
          </div>
        </div>
        {!readOnly && (
          editing ? (
            <div className="flex flex-col gap-1">
              <button
                onClick={handleSave}
                aria-label="Save item"
                className="text-[10px] px-2 py-1 bg-primary text-primary-foreground rounded hover:bg-primary/90 transition-colors"
              >
                Save
              </button>
              <button
                onClick={handleCancel}
                className="text-[10px] px-2 py-1 bg-secondary text-secondary-foreground rounded hover:bg-secondary/90 transition-colors"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setEditing(true)}
              aria-label="Edit item"
              className="text-[10px] px-1 py-1 hover:bg-secondary rounded transition-colors"
            >
              ✎
            </button>
          )
        )}
      </div>
    </div>
  );
}

export function CompactionDraftCard({
  draft,
  messagesSinceDraft,
  busy,
  onCommit,
  onDiscard,
  onJumpToMessage,
  serverRejections,
  readOnly = false,
  state: controlledState,
  onStateChange,
}: CompactionDraftCardProps) {
  const [internalState, setInternalState] = useState<ReviewState>(() => initialReviewState(draft));
  const state = controlledState ?? internalState;

  // Routes every update through whichever store owns the state: the
  // controlled `onStateChange` when the caller passed one, the internal
  // `useState` otherwise. Forwards `updater` itself rather than
  // pre-computing `updater(state)` against this render's closed-over
  // `state` -- two `applyUpdate` calls in the same handler (e.g.
  // `ItemRow.handleSave` editing both text and quote) would otherwise both
  // read the same stale `state` and the second call's result would clobber
  // the first's, the same stale-closure bug `setState(prev => ...)` exists
  // to avoid.
  const applyUpdate = (updater: (prev: ReviewState) => ReviewState) => {
    if (onStateChange) {
      onStateChange(updater);
    } else {
      setInternalState(updater);
    }
  };

  useEffect(() => {
    // A controlled owner (`PendingDraftMarker`) applies a server rejection
    // itself, once, at the moment the `422` arrives (see its `handleCommit`).
    // Re-applying here on every mount would re-strike an item the user has
    // since re-accepted: the mobile drawer unmounts this card on close and
    // remounts it on reopen, but `serverRejections` is unchanged (it is
    // still the same rejection from the earlier failed attempt), so this
    // effect would run again against the user's already-corrected state.
    // The uncontrolled path (internal `useState`, used by `CompactionNotice`'s
    // read-only dialog) has no such owner and keeps applying it here.
    if (onStateChange) return;
    applyUpdate((prev) => applyServerRejection(prev, serverRejections));
    // Only re-run when `serverRejections` (or the controlled/uncontrolled
    // mode itself) actually changes -- `applyUpdate` is omitted from the
    // dep list since it is redefined every render but forwards the updater
    // unchanged, so including it would only cause needless re-runs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serverRejections, onStateChange]);

  const grouped = groupByCategory(state.items);
  const { current, rated, blended } = draft.attitude;

  return (
    <Card className="text-xs bg-muted/30 max-w-2xl w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-xs font-medium text-muted-foreground">
          Continuity notes drafted for messages {draft.from_message_id}-{draft.through_message_id}
        </CardTitle>
        {messagesSinceDraft > 0 && (
          <p className="text-[10px] text-muted-foreground">
            {messagesSinceDraft} messages since this draft
          </p>
        )}
        {draft.status === 'stale' && (
          <p className="text-[10px] text-muted-foreground">
            based on edited history, re-compact?
          </p>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        {Array.from(grouped.entries()).map(([category, items]) => (
          <div key={category}>
            <h4 className="text-[10px] font-semibold uppercase text-muted-foreground mb-1">
              {CATEGORY_LABELS[category]}
            </h4>
            {items.map((item) => (
              <ItemRow
                key={item.fact.id}
                item={item}
                readOnly={readOnly}
                onToggle={() => applyUpdate((prev) => toggleAccepted(prev, item.fact.id))}
                onEditText={(text) => applyUpdate((prev) => editItemText(prev, item.fact.id, text))}
                onEditQuote={(quote) => applyUpdate((prev) => editItemQuote(prev, item.fact.id, quote))}
                onJumpToMessage={onJumpToMessage}
              />
            ))}
          </div>
        ))}

        <div>
          <h4 className="text-[10px] font-semibold uppercase text-muted-foreground mb-1">
            Summary
          </h4>
          <Textarea
            value={state.summary}
            onChange={(e) => applyUpdate((prev) => editSummary(prev, e.target.value))}
            disabled={readOnly}
            className="text-xs"
          />
        </div>

        <div>
          <h4 className="text-[10px] font-semibold uppercase text-muted-foreground mb-1">
            Attitude preview
          </h4>
          <table className="w-full text-[10px]">
            <thead>
              <tr className="text-muted-foreground">
                <th className="text-left font-normal">Dimension</th>
                <th className="text-right font-normal">Current</th>
                <th className="text-right font-normal">Rated</th>
                <th className="text-right font-normal">Blended</th>
              </tr>
            </thead>
            <tbody>
              {RATED_DIMENSIONS.map(({ key, label }) => {
                const currentValue = current[key];
                const ratedValue = rated ? rated[key] : null;
                const blendedValue = blended ? blended[key] : null;
                const jumped =
                  blendedValue !== null && Math.abs(blendedValue - currentValue) >= ATTITUDE_JUMP_THRESHOLD;
                return (
                  <tr key={key}>
                    <td className="text-left">{label}</td>
                    <td className="text-right">{currentValue}</td>
                    <td className="text-right">{ratedValue ?? '—'}</td>
                    <td className="text-right">
                      {jumped && <span aria-hidden="true">→ </span>}
                      {blendedValue ?? '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
      {!readOnly && (
        <CardFooter className="gap-2">
          <Button size="sm" disabled={busy} onClick={() => onCommit(toReviewPayload(state))}>
            Commit
          </Button>
          <Button size="sm" variant="outline" disabled={busy} onClick={onDiscard}>
            Discard
          </Button>
        </CardFooter>
      )}
    </Card>
  );
}
