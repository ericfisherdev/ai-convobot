import React, { createContext, useState, useContext, useCallback, useEffect, useRef, ReactNode } from 'react';
import { toast } from 'sonner';
import { useConfigData } from './configContext';
import { MultiplayerMode } from '../interfaces/Config';
import { RunningThought, RunningThoughtEdit } from '../interfaces/RunningThought';
import { initialRoundStreamState, readStreamChunks, reduceStreamChunk } from '../../lib/roundStream';

// Progress while `regenerateFrom` is rewriting: `fromMessageId` is the range
// boundary the caller asked for, `rewritten` the count of `thought` chunks
// the stream has delivered so far -- `RunningThoughtsPanel`'s progress line.
export interface RegeneratingProgress {
  fromMessageId: number;
  rewritten: number;
}

interface RunningThoughtsContextType {
  thoughts: RunningThought[];
  // The speaker id a live round's `thought_started` chunk named, until its
  // `thought` chunk (or a drop) clears it. `null` when nothing is pending.
  pendingSpeakerId: string | null;
  regenerating: RegeneratingProgress | null;
  refresh: () => Promise<void>;
  editThought: (id: number, text: string) => Promise<boolean>;
  deleteThought: (id: number) => Promise<boolean>;
  beginPending: (speakerId: string) => void;
  receiveThought: (thought: RunningThought) => void;
  dropPending: () => void;
  regenerateFrom: (fromMessageId: number) => Promise<void>;
}

const RunningThoughtsContext = createContext<RunningThoughtsContextType | undefined>(undefined);

// How often a joiner instance re-polls `GET /api/thoughts`: a joiner never
// sees the host's round stream, matching `participantsContext.tsx`'s cadence.
const POLL_INTERVAL_MS = 10_000;

interface RunningThoughtsProviderProps {
  children: ReactNode;
}

export const RunningThoughtsProvider: React.FC<RunningThoughtsProviderProps> = ({ children }) => {
  const configContext = useConfigData();
  const isJoiner = configContext?.config?.multiplayer_mode === MultiplayerMode.Joiner;

  const [thoughts, setThoughts] = useState<RunningThought[]>([]);
  const [pendingSpeakerId, setPendingSpeakerId] = useState<string | null>(null);
  const [regenerating, setRegenerating] = useState<RegeneratingProgress | null>(null);

  // A poll tick and a caller-triggered `refresh()` can be in flight at once;
  // only the most recently issued request's result is applied, matching
  // `compactionContext.tsx`/`participantsContext.tsx`'s identical guard.
  const latestRequestId = useRef(0);

  const refresh = useCallback(async (): Promise<void> => {
    const requestId = ++latestRequestId.current;
    const isStale = () => requestId !== latestRequestId.current;
    try {
      const response = await fetch('/api/thoughts');
      // A joiner has no thoughts of its own to answer with yet; treat a
      // `409` there as "no thoughts here" rather than toasting the user
      // every poll tick.
      if (response.status === 409 && isJoiner) {
        if (isStale()) return;
        setThoughts([]);
        return;
      }
      if (!response.ok) {
        throw new Error(`GET /api/thoughts returned ${response.status}`);
      }
      const data: { thoughts: RunningThought[] } = await response.json();
      if (isStale()) return;
      setThoughts(data.thoughts);
    } catch (error) {
      if (isStale()) return;
      console.error(error);
      toast.error(`Error while fetching running thoughts: ${error}`);
    }
  }, [isJoiner]);

  const editThought = useCallback(async (id: number, text: string): Promise<boolean> => {
    try {
      const response = await fetch(`/api/thoughts/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text } satisfies RunningThoughtEdit),
      });
      if (response.status === 422) {
        const body: { reason: string } = await response.json();
        toast.error(body.reason);
        return false;
      }
      // `thought_edit` claims `ACTIVE_TURN` and 409s with a plain-text
      // reason while a reply is streaming -- toast that reason rather than
      // falling into the generic `!response.ok` branch below, which would
      // surface the bare status code instead.
      if (response.status === 409) {
        toast.error(await response.text());
        return false;
      }
      if (!response.ok) {
        throw new Error(`PATCH /api/thoughts/${id} returned ${response.status}`);
      }
      const updated: RunningThought = await response.json();
      setThoughts((prev) => prev.map((t) => (t.id === id ? updated : t)));
      return true;
    } catch (error) {
      console.error(error);
      toast.error(`Error while editing running thought ${id}: ${error}`);
      return false;
    }
  }, []);

  const deleteThought = useCallback(async (id: number): Promise<boolean> => {
    try {
      const response = await fetch(`/api/thoughts/${id}`, { method: 'DELETE' });
      // Same as `editThought` above: `thought_delete` also claims
      // `ACTIVE_TURN` and 409s with a plain-text reason.
      if (response.status === 409) {
        toast.error(await response.text());
        return false;
      }
      if (!response.ok) {
        throw new Error(`DELETE /api/thoughts/${id} returned ${response.status}`);
      }
      setThoughts((prev) => prev.filter((t) => t.id !== id));
      return true;
    } catch (error) {
      console.error(error);
      toast.error(`Error while deleting running thought ${id}: ${error}`);
      return false;
    }
  }, []);

  const beginPending = useCallback((speakerId: string) => {
    setPendingSpeakerId(speakerId);
  }, []);

  const receiveThought = useCallback((thought: RunningThought) => {
    setThoughts((prev) => {
      const index = prev.findIndex((t) => t.id === thought.id);
      if (index === -1) return [...prev, thought];
      return prev.map((t) => (t.id === thought.id ? thought : t));
    });
    setPendingSpeakerId(null);
  }, []);

  const dropPending = useCallback(() => {
    setPendingSpeakerId(null);
  }, []);

  // Rewrites every thought from `fromMessageId` forward: the backend has
  // already deleted those rows by the time this resolves, so the local
  // state drops them up front rather than waiting for each replacement to
  // stream in. Not fire-and-forget -- the promise only resolves once the
  // stream ends, so a caller (the panel's confirm button) can await it.
  const regenerateFrom = useCallback(
    async (fromMessageId: number): Promise<void> => {
      try {
        const response = await fetch('/api/thoughts/regenerate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ from_message_id: fromMessageId }),
        });

        if (response.status === 409) {
          toast.error('Still replying; wait for it to finish before regenerating thoughts');
          return;
        }
        if (!response.ok || !response.body) {
          throw new Error(`POST /api/thoughts/regenerate returned ${response.status}`);
        }

        // Mirrors `SqliteRunningThoughtStore::delete_from`'s own predicate
        // (`through_message_id >= from_message_id`), not `from_message_id`:
        // a row whose range spans `fromMessageId` (from < fromMessageId <=
        // through) is deleted server-side too, and keeping it here would
        // disagree with what the backend just did.
        setThoughts((prev) => prev.filter((t) => t.through_message_id < fromMessageId));
        setRegenerating({ fromMessageId, rewritten: 0 });

        let streamState = initialRoundStreamState();
        let roundComplete = false;
        let streamErrorMessage: string | null = null;
        const nextTempId = () => 0; // the regenerate stream carries no reply bubbles.

        for await (const chunk of readStreamChunks(response.body)) {
          const { state: nextState, effects } = reduceStreamChunk(streamState, chunk, nextTempId);
          streamState = nextState;
          for (const effect of effects) {
            switch (effect.type) {
              case 'thought_started':
                beginPending(effect.speakerId);
                break;
              case 'thought':
                receiveThought(effect.thought);
                setRegenerating((prev) =>
                  prev ? { ...prev, rewritten: prev.rewritten + 1 } : { fromMessageId, rewritten: 1 }
                );
                break;
              case 'thought_dropped':
                dropPending();
                break;
              case 'round_complete':
                roundComplete = true;
                break;
              case 'error':
                streamErrorMessage = effect.message;
                break;
              default:
                break;
            }
          }
        }

        if (streamErrorMessage) {
          throw new Error(streamErrorMessage);
        }
        if (!roundComplete) {
          // The stream closed without a terminal chunk -- the same rule
          // `ChatWindow.promptMessage` applies to `/api/prompt/stream`.
          throw new Error('regeneration ended without completing');
        }
        // `regenerate_from` (backend) re-inserts every non-owned speaker's
        // row in the deleted range unchanged, under a new id, without
        // emitting a sink event for any of them -- the stream only carries
        // this instance's own rewritten rows. Resync from the source of
        // truth so those reappear instead of staying dropped until an
        // unrelated reload.
        await refresh();
      } catch (error) {
        console.error(error);
        toast.error(`Error while regenerating running thoughts: ${error}`);
        refresh();
      } finally {
        setRegenerating(null);
      }
    },
    [beginPending, receiveThought, dropPending, refresh]
  );

  // Waits for `ConfigProvider`'s own fetch to resolve before the first
  // `refresh()`: `isJoiner` (and so `refresh`'s 409 handling) is not known
  // yet on the very first render, and firing early would risk toasting a
  // 409 this instance is actually a joiner for. `configContext?.config` is a
  // new object on every `ConfigProvider` fetch (including a later config
  // save), so this effect -- and so `refresh()` -- reruns then too; that is
  // a harmless extra GET, not something this effect tries to avoid.
  useEffect(() => {
    if (!configContext?.config) return;
    refresh();
  }, [configContext?.config, refresh]);

  // Only a joiner polls: the stream is the live path in host/solo mode, and
  // `refresh()` above already covers the initial load and reloads.
  useEffect(() => {
    if (!isJoiner) return;
    const interval = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [isJoiner, refresh]);

  return (
    <RunningThoughtsContext.Provider
      value={{
        thoughts,
        pendingSpeakerId,
        regenerating,
        refresh,
        editThought,
        deleteThought,
        beginPending,
        receiveThought,
        dropPending,
        regenerateFrom,
      }}
    >
      {children}
    </RunningThoughtsContext.Provider>
  );
};

export const useRunningThoughts = (): RunningThoughtsContextType => {
  const context = useContext(RunningThoughtsContext);
  if (!context) {
    throw new Error('useRunningThoughts must be used within a RunningThoughtsProvider');
  }
  return context;
};
