import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../ui/dialog';
import { useCompaction } from '../context/compactionContext';
import { CheckpointDetail, CheckpointSummary } from '../interfaces/Compaction';
import { CompactionDraftCard } from './CompactionDraftCard';
import { scrollToMessage } from '../../lib/messageAnchors';

// The collapsed line left behind by a committed (or stale) checkpoint, at
// the position of the last message it covers. Mirrors `SystemNotice`'s
// styling in `Message.tsx` -- centred, muted, non-interactive by default.
export function CompactionNotice({ checkpoint }: { checkpoint: CheckpointSummary }) {
  const { fetchDetail, triggerDraft } = useCompaction();
  const [open, setOpen] = useState(false);
  const [detail, setDetail] = useState<CheckpointDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [recompacting, setRecompacting] = useState(false);

  // Fetched on open, not on mount: most committed checkpoints are never
  // expanded, so there is no reason to load every one's facts up front.
  const handleOpenChange = async (next: boolean) => {
    setOpen(next);
    if (next && detail === null) {
      setLoading(true);
      const result = await fetchDetail(checkpoint.id);
      setDetail(result);
      setLoading(false);
    }
  };

  const handleRecompact = async () => {
    setRecompacting(true);
    await triggerDraft({ fromStale: true });
    setRecompacting(false);
  };

  return (
    <div className="message-container flex justify-center animate-in fade-in-0 duration-300">
      <div className="flex flex-col items-center gap-1 text-center max-w-[85%]">
        <div className="chat-bubble text-xs text-muted-foreground bg-muted/50 rounded-full px-3 py-1 flex items-center gap-2">
          <span>Continuity notes cover messages up to here</span>
          <Dialog open={open} onOpenChange={handleOpenChange}>
            <DialogTrigger asChild>
              <button type="button" className="underline hover:text-foreground">
                Show notes
              </button>
            </DialogTrigger>
            <DialogContent className="max-h-[80vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Continuity notes</DialogTitle>
              </DialogHeader>
              {loading || !detail ? (
                <p className="text-sm text-muted-foreground">Loading…</p>
              ) : (
                <CompactionDraftCard
                  draft={detail}
                  messagesSinceDraft={0}
                  busy={false}
                  onCommit={() => {}}
                  onDiscard={() => {}}
                  onJumpToMessage={scrollToMessage}
                  serverRejections={[]}
                  readOnly
                />
              )}
            </DialogContent>
          </Dialog>
        </div>
        {checkpoint.status === 'stale' && (
          <button
            type="button"
            className="text-[10px] underline text-muted-foreground hover:text-foreground disabled:opacity-50"
            onClick={handleRecompact}
            disabled={recompacting}
          >
            based on edited history, re-compact?
          </button>
        )}
      </div>
    </div>
  );
}
