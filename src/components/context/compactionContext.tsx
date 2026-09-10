import React, { createContext, useState, useContext, useCallback, useEffect, useRef, ReactNode } from 'react';
import { toast } from 'sonner';
import { useMessages } from './messageContext';
import {
  CheckpointSummary,
  PendingDraftSummary,
  CheckpointDetail,
  CompactionDraftReview,
} from '../interfaces/Compaction';

// The result of `commit`: on `422` the raw body is handed back to the
// caller instead of toasted, so a review card (a later issue) can render
// the rejection/over-budget detail inline next to the item it names.
export type CommitOutcome =
  | { ok: true; checkpoint: CheckpointSummary }
  | { ok: false; status: number; body: unknown };

interface CompactionContextType {
  checkpoints: CheckpointSummary[];
  pendingDraft: PendingDraftSummary | null;
  pinnedIds: Set<number>;
  refresh: () => Promise<void>;
  fetchDetail: (id: number) => Promise<CheckpointDetail | null>;
  triggerDraft: () => Promise<void>;
  commit: (id: number, review: CompactionDraftReview) => Promise<CommitOutcome>;
  discard: (id: number) => Promise<void>;
  pin: (id: number) => Promise<void>;
  unpin: (id: number) => Promise<void>;
  draftReady: (id: number) => void;
}

const CompactionContext = createContext<CompactionContextType | undefined>(undefined);

// How often `draftReady`'s poll re-checks a queued draft's phase.
const DRAFT_POLL_INTERVAL_MS = 5_000;

interface CompactionProviderProps {
  children: ReactNode;
}

export const CompactionProvider: React.FC<CompactionProviderProps> = ({ children }) => {
  const { messages, refreshMessages } = useMessages();

  const [checkpoints, setCheckpoints] = useState<CheckpointSummary[]>([]);
  const [pendingDraft, setPendingDraft] = useState<PendingDraftSummary | null>(null);

  const pinnedIds = new Set(messages.filter((m) => m.pinned).map((m) => m.id));

  // A poll tick (`draftReady`) and a caller-triggered `refresh()` can be in
  // flight at once; only the most recently issued request's result is
  // applied, matching `participantsContext.tsx`'s `latestRequestId` guard.
  const latestRequestId = useRef(0);

  const refresh = useCallback(async (): Promise<void> => {
    const requestId = ++latestRequestId.current;
    const isStale = () => requestId !== latestRequestId.current;
    try {
      const response = await fetch('/api/compaction');
      if (!response.ok) {
        throw new Error(`GET /api/compaction returned ${response.status}`);
      }
      const data: { checkpoints: CheckpointSummary[]; pending_draft: PendingDraftSummary | null } =
        await response.json();
      if (isStale()) return;
      setCheckpoints(data.checkpoints);
      setPendingDraft(data.pending_draft);
    } catch (error) {
      if (isStale()) return;
      console.error(error);
      toast.error(`Error while fetching compaction checkpoints: ${error}`);
    }
  }, []);

  const fetchDetail = useCallback(async (id: number): Promise<CheckpointDetail | null> => {
    try {
      const response = await fetch(`/api/compaction/${id}`);
      if (!response.ok) {
        throw new Error(`GET /api/compaction/${id} returned ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error(error);
      toast.error(`Error while fetching compaction checkpoint ${id}: ${error}`);
      return null;
    }
  }, []);

  const draftPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopDraftPoll = useCallback(() => {
    if (draftPollRef.current !== null) {
      clearInterval(draftPollRef.current);
      draftPollRef.current = null;
    }
  }, []);

  // Polls `GET /api/compaction/{id}` every `DRAFT_POLL_INTERVAL_MS` until the
  // draft reaches `review` phase or disappears (a lookup failure -- e.g. it
  // was discarded from under it), refreshing the listing either way so a
  // review card (a later issue) picks up the change. Only one poll runs at
  // a time: a second call replaces whatever was already running, matching
  // the backend's "one pending draft" rule.
  const draftReady = useCallback(
    (id: number) => {
      stopDraftPoll();
      refresh();
      const poll = async () => {
        const detail = await fetchDetail(id);
        if (!detail || detail.phase === 'review') {
          stopDraftPoll();
          refresh();
        }
      };
      draftPollRef.current = setInterval(poll, DRAFT_POLL_INTERVAL_MS);
    },
    [fetchDetail, refresh, stopDraftPoll]
  );

  const triggerDraft = useCallback(async (): Promise<void> => {
    try {
      const response = await fetch('/api/compaction/draft', { method: 'POST' });
      if (response.status === 202) {
        const data: { draft_id: number } = await response.json();
        draftReady(data.draft_id);
        return;
      }
      const reason = await response.text();
      throw new Error(reason || `POST /api/compaction/draft returned ${response.status}`);
    } catch (error) {
      console.error(error);
      toast.error(`Error while triggering compaction: ${error}`);
    }
  }, [draftReady]);

  const commit = useCallback(
    async (id: number, review: CompactionDraftReview): Promise<CommitOutcome> => {
      try {
        const response = await fetch(`/api/compaction/${id}/commit`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(review),
        });
        if (response.ok) {
          const checkpoint: CheckpointSummary = await response.json();
          refresh();
          return { ok: true, checkpoint };
        }
        if (response.status === 422) {
          const body = await response.json();
          return { ok: false, status: 422, body };
        }
        const text = await response.text();
        toast.error(`Error while committing compaction draft: ${text || response.status}`);
        return { ok: false, status: response.status, body: text };
      } catch (error) {
        console.error(error);
        toast.error(`Error while committing compaction draft: ${error}`);
        return { ok: false, status: 0, body: String(error) };
      }
    },
    [refresh]
  );

  const discard = useCallback(
    async (id: number): Promise<void> => {
      try {
        const response = await fetch(`/api/compaction/${id}/discard`, { method: 'POST' });
        if (!response.ok) {
          throw new Error(`POST /api/compaction/${id}/discard returned ${response.status}`);
        }
        refresh();
      } catch (error) {
        console.error(error);
        toast.error(`Error while discarding compaction draft: ${error}`);
      }
    },
    [refresh]
  );

  const pin = useCallback(
    async (id: number): Promise<void> => {
      try {
        const response = await fetch(`/api/message/${id}/pin`, { method: 'POST' });
        if (!response.ok) {
          throw new Error(`POST /api/message/${id}/pin returned ${response.status}`);
        }
        refreshMessages();
      } catch (error) {
        console.error(error);
        toast.error(`Error while pinning message ${id}: ${error}`);
      }
    },
    [refreshMessages]
  );

  const unpin = useCallback(
    async (id: number): Promise<void> => {
      try {
        const response = await fetch(`/api/message/${id}/pin`, { method: 'DELETE' });
        if (!response.ok) {
          throw new Error(`DELETE /api/message/${id}/pin returned ${response.status}`);
        }
        refreshMessages();
      } catch (error) {
        console.error(error);
        toast.error(`Error while unpinning message ${id}: ${error}`);
      }
    },
    [refreshMessages]
  );

  useEffect(() => {
    refresh();
    return () => stopDraftPoll();
  }, [refresh, stopDraftPoll]);

  return (
    <CompactionContext.Provider
      value={{
        checkpoints,
        pendingDraft,
        pinnedIds,
        refresh,
        fetchDetail,
        triggerDraft,
        commit,
        discard,
        pin,
        unpin,
        draftReady,
      }}
    >
      {children}
    </CompactionContext.Provider>
  );
};

export const useCompaction = (): CompactionContextType => {
  const context = useContext(CompactionContext);
  if (!context) {
    throw new Error('useCompaction must be used within a CompactionProvider');
  }
  return context;
};
