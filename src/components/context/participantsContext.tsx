import React, { createContext, useState, useContext, useEffect, useCallback, ReactNode } from 'react';
import { toast } from "sonner";
import { useConfigData } from './configContext';
import { useUserData } from './userContext';
import { useCompanionData } from './companionContext';
import { MultiplayerMode } from '../interfaces/Config';
import { Participant, MultiplayerStatus } from '../interfaces/Participant';

interface ParticipantsProviderProps {
  children: ReactNode;
}

interface ParticipantsContextType {
  participants: Participant[];
  status: MultiplayerStatus;
  getParticipant: (id: string) => Participant | undefined;
  refreshParticipants: () => Promise<void>;
  loaded: boolean;
}

const ParticipantsContext = createContext<ParticipantsContextType | undefined>(undefined);

// How often to re-poll `/api/multiplayer/participants` (host) or
// `/api/multiplayer/status` (joiner). #133's SSE stream carries no
// participant events, so this poll is the primary way a strip picks up a
// joiner connecting or dropping; `ChatWindow` supplements it with an
// immediate `refreshParticipants()` after a round completes or errors.
const POLL_INTERVAL_MS = 10_000;

const SOLO_STATUS: MultiplayerStatus = { mode: 'solo', state: null };

// A backend registry always carries `user` and `char` first (see
// `ParticipantRegistry::solo`), and the frontend already has that data
// locally via `userContext`/`companionContext` -- no need to wait on a
// network round trip to show them. `mergeRemote` drops the `user`/`char`
// rows a `host`/`joiner` response also carries, so the local pair is never
// duplicated.
function mergeRemote(remote: Participant[]): Participant[] {
  return remote.filter((p) => p.id !== 'user' && p.id !== 'char');
}

export const ParticipantsProvider: React.FC<ParticipantsProviderProps> = ({ children }) => {
  const configContext = useConfigData();
  const userDataContext = useUserData();
  const companionDataContext = useCompanionData();

  const mode = configContext?.config?.multiplayer_mode;
  const userName = userDataContext?.userData?.name || 'User';
  const companionName = companionDataContext?.companionData?.name || 'Assistant';
  // `companionContext.tsx` already appends a cache-busting timestamp to
  // `avatar_path`, so it is used as-is.
  const companionAvatarUrl = companionDataContext?.companionData?.avatar_path || null;

  const [remoteParticipants, setRemoteParticipants] = useState<Participant[]>([]);
  const [status, setStatus] = useState<MultiplayerStatus>(SOLO_STATUS);
  const [loaded, setLoaded] = useState(false);

  const localParticipants: Participant[] = [
    { id: 'user', display_name: userName, kind: 'Human', avatar_url: null, connected: true },
    { id: 'char', display_name: companionName, kind: 'HostBot', avatar_url: companionAvatarUrl, connected: true },
  ];

  const fetchParticipants = useCallback(async (): Promise<void> => {
    if (mode === MultiplayerMode.Host) {
      try {
        const response = await fetch('/api/multiplayer/participants');
        if (!response.ok) {
          throw new Error('');
        }
        const data: Participant[] = await response.json();
        setRemoteParticipants(mergeRemote(data));
        setStatus({ mode: 'host', state: null });
      } catch (error) {
        console.error(error);
        toast.error(`Error while fetching multiplayer participants: ${error}`);
      } finally {
        setLoaded(true);
      }
      return;
    }

    if (mode === MultiplayerMode.Joiner) {
      try {
        const response = await fetch('/api/multiplayer/status');
        if (!response.ok) {
          throw new Error('');
        }
        const data: MultiplayerStatus = await response.json();
        setRemoteParticipants(mergeRemote(data.participants ?? []));
        setStatus(data);
      } catch (error) {
        console.error(error);
        toast.error(`Error while fetching multiplayer status: ${error}`);
      } finally {
        setLoaded(true);
      }
      return;
    }

    // Solo mode: no `/api/multiplayer/*` requests at all.
    setRemoteParticipants([]);
    setStatus(SOLO_STATUS);
    setLoaded(true);
  }, [mode]);

  useEffect(() => {
    fetchParticipants();

    if (mode !== MultiplayerMode.Host && mode !== MultiplayerMode.Joiner) {
      return;
    }

    const interval = setInterval(fetchParticipants, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [mode, fetchParticipants]);

  const participants = [...localParticipants, ...remoteParticipants];

  const getParticipant = (id: string): Participant | undefined =>
    participants.find((p) => p.id === id);

  return (
    <ParticipantsContext.Provider
      value={{ participants, status, getParticipant, refreshParticipants: fetchParticipants, loaded }}
    >
      {children}
    </ParticipantsContext.Provider>
  );
};

export const useParticipants = (): ParticipantsContextType => {
  const context = useContext(ParticipantsContext);
  if (!context) {
    throw new Error('useParticipants must be used within a ParticipantsProvider');
  }
  return context;
};
