import React, { createContext, useContext, useState, useCallback, useEffect, useRef, ReactNode } from 'react';
import { AttitudeData, AttitudeDimensionUpdate, AttitudeStreamUpdate, ATTITUDE_DIMENSIONS } from '../interfaces/AttitudeData';

interface AttitudeContextType {
    attitudes: AttitudeData[];
    loading: boolean;
    error: string | null;

    // Methods
    fetchAttitudes: (companionId: number) => Promise<void>;
    getAttitude: (companionId: number, targetId: number, targetType: string) => Promise<AttitudeData | null>;
    createOrUpdateAttitude: (attitude: Partial<AttitudeData>) => Promise<boolean>;
    updateAttitudeDimension: (update: AttitudeDimensionUpdate) => Promise<boolean>;

    // Companion's attitude toward the chatting user, owned here so the
    // summary bar keeps rendering the last known values while a refetch or a
    // stream update is in flight.
    userAttitude: AttitudeData | null;
    userAttitudeSummary: string;
    // Dimensions moved by the most recent turn, keyed by dimension name.
    lastTurnDeltas: Record<string, number>;
    // False until the first `refreshUserAttitude` settles, so a consumer can
    // tell "nothing yet" apart from "no attitude row".
    userAttitudeLoaded: boolean;
    refreshUserAttitude: (companionId: number, userId: number) => Promise<void>;
    applyAttitudeStreamUpdate: (update: AttitudeStreamUpdate) => void;

    // Current selection
    selectedAttitude: AttitudeData | null;
    setSelectedAttitude: (attitude: AttitudeData | null) => void;
}

interface AttitudeSummaryResponse {
    attitude: AttitudeData;
    summary: string;
}

// Signed change per dimension between two snapshots, mirroring the backend's
// `AttitudeFormatter::diff_attitudes` so both sides hide the same noise.
const DELTA_THRESHOLD = 1;

const diffAttitudes = (previous: AttitudeData | null, current: AttitudeData): Record<string, number> => {
    if (!previous) return {};

    const deltas: Record<string, number> = {};
    for (const dimension of ATTITUDE_DIMENSIONS) {
        const delta = (current[dimension.key] as number) - (previous[dimension.key] as number);
        if (Math.abs(delta) >= DELTA_THRESHOLD) {
            deltas[dimension.key] = delta;
        }
    }
    return deltas;
};

const AttitudeContext = createContext<AttitudeContextType | undefined>(undefined);

interface AttitudeProviderProps {
    children: ReactNode;
}

export const AttitudeProvider: React.FC<AttitudeProviderProps> = ({ children }) => {
    const [attitudes, setAttitudes] = useState<AttitudeData[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedAttitude, setSelectedAttitude] = useState<AttitudeData | null>(null);
    const [userAttitude, setUserAttitude] = useState<AttitudeData | null>(null);
    // Mirrors `userAttitude` so a refresh can diff against the current value
    // without a state updater having to reach for it (updaters stay pure).
    const userAttitudeRef = useRef<AttitudeData | null>(null);
    const [userAttitudeSummary, setUserAttitudeSummary] = useState<string>('');
    const [lastTurnDeltas, setLastTurnDeltas] = useState<Record<string, number>>({});
    const [userAttitudeLoaded, setUserAttitudeLoaded] = useState(false);
    // Ids of the last refresh, so the `attitude-update` fallback can refetch
    // without the event carrying them.
    const userTargetRef = useRef<{ companionId: number; userId: number } | null>(null);
    // Bumped by every refresh and every stream update. A refresh only applies
    // its response while it is still the newest writer, so a slow request can
    // never overwrite fresher state (and never mint inverted deltas by diffing
    // a stale snapshot against it).
    const userAttitudeWriteRef = useRef(0);
    const selectedAttitudeRef = useRef(selectedAttitude);
    useEffect(() => {
        selectedAttitudeRef.current = selectedAttitude;
    }, [selectedAttitude]);
    useEffect(() => {
        userAttitudeRef.current = userAttitude;
    }, [userAttitude]);

    const fetchAttitudes = useCallback(async (companionId: number): Promise<void> => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`/api/attitude/companion/${companionId}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setAttitudes(data);
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to fetch attitudes';
            setError(errorMessage);
            console.error('Error fetching attitudes:', err);
        } finally {
            setLoading(false);
        }
    }, []);

    const getAttitude = useCallback(async (companionId: number, targetId: number, targetType: string): Promise<AttitudeData | null> => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`/api/attitude?companion_id=${companionId}&target_id=${targetId}&target_type=${targetType}`);
            if (response.status === 404) {
                return null; // Attitude doesn't exist
            }
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data;
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to get attitude';
            setError(errorMessage);
            console.error('Error getting attitude:', err);
            return null;
        } finally {
            setLoading(false);
        }
    }, []);

    const createOrUpdateAttitude = useCallback(async (attitude: Partial<AttitudeData>): Promise<boolean> => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/attitude', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ...attitude,
                    last_updated: new Date().toISOString(),
                    created_at: attitude.created_at || new Date().toISOString()
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Refresh attitudes if we have a companion_id
            if (attitude.companion_id) {
                await fetchAttitudes(attitude.companion_id);
            }

            return true;
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to create/update attitude';
            setError(errorMessage);
            console.error('Error creating/updating attitude:', err);
            return false;
        } finally {
            setLoading(false);
        }
    }, [fetchAttitudes]);

    const updateAttitudeDimension = useCallback(async (update: AttitudeDimensionUpdate): Promise<boolean> => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/attitude/dimension', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(update),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Refresh attitudes
            await fetchAttitudes(update.companion_id);

            // Update selected attitude if it matches
            const selected = selectedAttitudeRef.current;
            if (selected &&
                selected.companion_id === update.companion_id &&
                selected.target_id === update.target_id &&
                selected.target_type === update.target_type) {
                setSelectedAttitude(await getAttitude(update.companion_id, update.target_id, update.target_type));
            }

            return true;
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to update attitude dimension';
            setError(errorMessage);
            console.error('Error updating attitude dimension:', err);
            return false;
        } finally {
            setLoading(false);
        }
    }, [fetchAttitudes, getAttitude]);

    // Replaces the user attitude in place: values already on screen stay
    // rendered for the whole request, so the bar never blanks mid-refetch.
    const refreshUserAttitude = useCallback(async (companionId: number, userId: number): Promise<void> => {
        userTargetRef.current = { companionId, userId };
        const write = ++userAttitudeWriteRef.current;

        const applyRefreshed = (attitude: AttitudeData, summary: string | null) => {
            if (write !== userAttitudeWriteRef.current) return;
            setLastTurnDeltas(diffAttitudes(userAttitudeRef.current, attitude));
            userAttitudeRef.current = attitude;
            setUserAttitude(attitude);
            if (summary !== null) {
                setUserAttitudeSummary(summary);
            }
        };

        try {
            const response = await fetch(`/api/attitude/summary/${companionId}/${userId}`);
            if (response.ok) {
                const data: AttitudeSummaryResponse = await response.json();
                applyRefreshed(data.attitude, data.summary);
                return;
            }

            // Older backends have no summary endpoint; the raw attitude is
            // enough for the bar, which can summarize it locally.
            const attitude = await getAttitude(companionId, userId, 'user');
            if (attitude) {
                applyRefreshed(attitude, null);
            }
        } catch (err) {
            console.error('Error refreshing user attitude:', err);
        } finally {
            setUserAttitudeLoaded(true);
        }
    }, [getAttitude]);

    // The stream already carries the post-turn attitude, so this path costs no
    // request and reports the backend's own deltas rather than a local diff.
    const applyAttitudeStreamUpdate = useCallback((update: AttitudeStreamUpdate): void => {
        // Supersedes any refresh already in flight: the stream carries the
        // post-turn state, which is newer than anything a request in progress
        // can return.
        userAttitudeWriteRef.current += 1;
        const deltas: Record<string, number> = {};
        for (const { dimension, delta } of update.deltas) {
            deltas[dimension] = delta;
        }
        userAttitudeRef.current = update.attitude;
        setUserAttitude(update.attitude);
        setUserAttitudeSummary(update.summary);
        setLastTurnDeltas(deltas);
        setUserAttitudeLoaded(true);
    }, []);

    // Fallback for turns that deliver no attitude chunk (e.g. impersonation,
    // which never runs the attitude engine).
    useEffect(() => {
        const handleAttitudeUpdate = () => {
            const target = userTargetRef.current;
            if (target) {
                refreshUserAttitude(target.companionId, target.userId);
            }
        };

        window.addEventListener('attitude-update', handleAttitudeUpdate);
        return () => {
            window.removeEventListener('attitude-update', handleAttitudeUpdate);
        };
    }, [refreshUserAttitude]);

    const contextValue: AttitudeContextType = {
        attitudes,
        loading,
        error,
        fetchAttitudes,
        getAttitude,
        createOrUpdateAttitude,
        updateAttitudeDimension,
        userAttitude,
        userAttitudeSummary,
        lastTurnDeltas,
        userAttitudeLoaded,
        refreshUserAttitude,
        applyAttitudeStreamUpdate,
        selectedAttitude,
        setSelectedAttitude,
    };

    return (
        <AttitudeContext.Provider value={contextValue}>
            {children}
        </AttitudeContext.Provider>
    );
};

export const useAttitude = (): AttitudeContextType => {
    const context = useContext(AttitudeContext);
    if (context === undefined) {
        throw new Error('useAttitude must be used within an AttitudeProvider');
    }
    return context;
};
