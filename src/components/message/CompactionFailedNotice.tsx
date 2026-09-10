import { useState } from 'react';
import { useCompaction } from '../context/compactionContext';
import { CheckpointSummary } from '../interfaces/Compaction';

// The collapsed line left behind by a checkpoint whose extraction pipeline
// itself failed (#208) -- a model/store I/O error (load failure, OOM, a
// truncated completion) -- rather than one that produced a draft to review
// or a `Discarded` row `fill_draft` rejected on content grounds. Mirrors
// `CompactionNotice`'s layout/styling, but there is no "Show notes" dialog:
// a failed draft has no facts to look at.
//
// `Retry` re-triggers a fresh manual draft (`triggerDraft()`): the failed
// row is already terminal (`status !== 'draft'`), so `should_compact`/the
// manual-trigger route never see it as a pending draft blocking a new one
// -- see `compaction::trigger::should_compact`, `compaction::store::
// pending_draft_on`. `Dismiss` only hides this notice locally; there is
// nothing to discard server-side (`POST /api/compaction/{id}/discard`
// would 409 -- `commit::discard` only accepts a `Draft`-status row -- and a
// failed draft already does not block anything, so there is nothing that
// action would need to undo).
export function CompactionFailedNotice({ checkpoint }: { checkpoint: CheckpointSummary }) {
  const { triggerDraft } = useCompaction();
  const [retrying, setRetrying] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) {
    return null;
  }

  const handleRetry = async () => {
    setRetrying(true);
    await triggerDraft();
    setRetrying(false);
  };

  return (
    <div className="message-container flex justify-center animate-in fade-in-0 duration-300">
      <div className="flex flex-col items-center gap-1 text-center max-w-[85%]">
        <div className="chat-bubble text-xs text-destructive bg-destructive/10 rounded-full px-3 py-1 flex items-center gap-2 flex-wrap justify-center">
          <span>
            Continuity notes failed to draft
            {checkpoint.extraction_error ? `: ${checkpoint.extraction_error}` : ''}
          </span>
          <button
            type="button"
            className="underline hover:text-foreground disabled:opacity-50"
            onClick={handleRetry}
            disabled={retrying}
          >
            Retry
          </button>
          <button
            type="button"
            className="underline hover:text-foreground"
            onClick={() => setDismissed(true)}
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}
