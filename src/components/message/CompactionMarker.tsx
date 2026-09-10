import { useEffect, useState } from 'react';
import { toast } from 'sonner';
import { useCompaction } from '../context/compactionContext';
import { useMessages } from '../context/messageContext';
import { useMobile } from '../../hooks/useMobile';
import { CheckpointDetail, CompactionDraftReview, DraftPhase, RejectedItem } from '../interfaces/Compaction';
import { CompactionDraftCard } from './CompactionDraftCard';
import { CompactionNotice } from './CompactionNotice';
import { CompactionFailedNotice } from './CompactionFailedNotice';
import { applyServerRejection, initialReviewState, ReviewState } from './compactionReview';
import { scrollToMessage } from '../../lib/messageAnchors';
import { Button } from '../ui/button';
import { Drawer, DrawerContent, DrawerTrigger } from '../ui/drawer';

// A `422` commit rejection's body: `ReviewError::UnknownItem` serialises as
// a bare `{ item_id, reason }`, `ReviewError::Rejected` as an array of the
// same shape (see `compaction/review.rs::CompactionCommitError::into_response`).
// The card only cares about the list either way.
function toRejectedItems(body: unknown): RejectedItem[] {
  if (Array.isArray(body)) {
    return body as RejectedItem[];
  }
  if (body && typeof body === 'object' && 'item_id' in body) {
    return [body as RejectedItem];
  }
  return [];
}

// A pending draft's own marker: a lightweight "still drafting" notice while
// `phase === 'extracting'`, the full review card once `phase === 'review'`.
function PendingDraftMarker({
  draftId,
  phase,
  messageId,
  compact,
}: {
  draftId: number;
  phase: DraftPhase;
  messageId: number;
  compact: boolean;
}) {
  const { fetchDetail, commit, discard, refresh } = useCompaction();
  const { messages, refreshMessages } = useMessages();
  const [detail, setDetail] = useState<CheckpointDetail | null>(null);
  // Owned here, not inside `CompactionDraftCard`, so the mobile drawer
  // (`vaul`'s `DrawerContent` unmounts its children when it closes) does not
  // throw away every edit/strike the user made while reviewing.
  const [review, setReview] = useState<ReviewState | null>(null);
  const [busy, setBusy] = useState(false);
  const [rejections, setRejections] = useState<RejectedItem[]>([]);
  const [drawerOpen, setDrawerOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    if (phase === 'review') {
      fetchDetail(draftId).then((result) => {
        if (cancelled) return;
        setDetail(result);
        setReview(result ? initialReviewState(result) : null);
      });
    }
    return () => {
      cancelled = true;
    };
  }, [draftId, phase, fetchDetail]);

  const messagesSinceDraft = messages.filter((m) => m.id > messageId).length;

  const handleCommit = async (review: CompactionDraftReview) => {
    setBusy(true);
    const outcome = await commit(draftId, review);
    setBusy(false);
    if (outcome.ok) {
      setRejections([]);
      toast.success('Continuity notes committed');
      refresh();
      refreshMessages();
      return;
    }
    if (outcome.status === 422) {
      const rejected = toRejectedItems(outcome.body);
      setRejections(rejected);
      // Applied once, here, at the moment the rejection arrives -- not by
      // the card's own mount effect (removed below), which would otherwise
      // re-strike an item the user re-accepted every time the mobile
      // drawer closes and reopens (it persists this same `rejections`
      // across the remount and would re-run against it).
      setReview((prev) => (prev ? applyServerRejection(prev, rejected) : prev));
    }
    // Any other failure is already toasted by `useCompaction().commit`.
  };

  const handleDiscard = async () => {
    setBusy(true);
    await discard(draftId);
    setBusy(false);
  };

  if (phase === 'extracting' || !detail || !review) {
    return (
      <div className="message-container flex justify-center animate-in fade-in-0 duration-300">
        <div className="chat-bubble text-xs text-muted-foreground bg-muted/50 rounded-full px-3 py-1">
          Continuity notes are being drafted…
        </div>
      </div>
    );
  }

  const card = (
    <CompactionDraftCard
      draft={detail}
      messagesSinceDraft={messagesSinceDraft}
      busy={busy}
      onCommit={handleCommit}
      onDiscard={handleDiscard}
      onJumpToMessage={scrollToMessage}
      serverRejections={rejections}
      state={review}
      onStateChange={(update) => setReview((prev) => (prev ? update(prev) : prev))}
    />
  );

  if (!compact) {
    return <div className="flex justify-center">{card}</div>;
  }

  return (
    <div className="message-container flex justify-center animate-in fade-in-0 duration-300">
      <div className="chat-bubble text-xs text-muted-foreground bg-muted/50 rounded-full px-3 py-1 flex items-center gap-2">
        <span>Continuity notes ready for review</span>
        <Drawer open={drawerOpen} onOpenChange={setDrawerOpen}>
          <DrawerTrigger asChild>
            <Button size="sm" variant="outline">
              Review
            </Button>
          </DrawerTrigger>
          <DrawerContent className="h-[96%]">
            <div className="overflow-y-auto p-4 flex justify-center">{card}</div>
          </DrawerContent>
        </Drawer>
      </div>
    </div>
  );
}

interface CompactionMarkerProps {
  messageId: number;
  // Forced compact layout for `VirtualMessageList`, whose fixed
  // `ITEM_HEIGHT` cannot fit the full card inline. `MessageScroll` leaves
  // this unset and instead relies on `isMobile` below.
  compact?: boolean;
}

// Decides what (if anything) renders after message `messageId`: the pending
// draft's card if this is its last covered message, the committed/stale
// notice if a checkpoint ends here, otherwise nothing. Kept separate from
// `Message.tsx` so that component's own speaker dispatch stays untouched.
export function CompactionMarker({ messageId, compact: compactProp = false }: CompactionMarkerProps) {
  const { pendingDraft, checkpoints } = useCompaction();
  const { isMobile } = useMobile();
  const compact = compactProp || isMobile;

  if (pendingDraft?.through_message_id === messageId) {
    return (
      <PendingDraftMarker
        draftId={pendingDraft.id}
        phase={pendingDraft.phase}
        messageId={messageId}
        compact={compact}
      />
    );
  }

  // The listing carries every status; only a committed (or stale) checkpoint
  // leaves a notice behind. A discarded row ends at the same message its
  // draft did and must not read as coverage. A failed row (#208) is neither
  // -- it never produced anything to render -- so it gets its own notice
  // instead of `CompactionNotice`'s "Show notes" dialog, which assumes
  // there are facts to look at.
  const checkpoint = checkpoints.find(
    (c) =>
      c.through_message_id === messageId &&
      (c.status === 'committed' || c.status === 'stale' || c.status === 'failed')
  );
  if (checkpoint) {
    return checkpoint.status === 'failed' ? (
      <CompactionFailedNotice checkpoint={checkpoint} />
    ) : (
      <CompactionNotice checkpoint={checkpoint} />
    );
  }

  return null;
}
