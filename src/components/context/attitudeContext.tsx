import React, { createContext, useContext, useState, useCallback, useEffect, useRef, ReactNode } from 'react';
import { AttitudeData, AttitudeDimensionUpdate } from '../interfaces/AttitudeData';

interface AttitudeContextType {
    attitudes: AttitudeData[];
    loading: boolean;
    error: string | null;

    // Methods
    fetchAttitudes: (companionId: number) => Promise<void>;
    getAttitude: (companionId: number, targetId: number, targetType: string) => Promise<AttitudeData | null>;
    createOrUpdateAttitude: (attitude: Partial<AttitudeData>) => Promise<boolean>;
    updateAttitudeDimension: (update: AttitudeDimensionUpdate) => Promise<boolean>;

    // Current selection
    selectedAttitude: AttitudeData | null;
    setSelectedAttitude: (attitude: AttitudeData | null) => void;
}

const AttitudeContext = createContext<AttitudeContextType | undefined>(undefined);

interface AttitudeProviderProps {
    children: ReactNode;
}

export const AttitudeProvider: React.FC<AttitudeProviderProps> = ({ children }) => {
    const [attitudes, setAttitudes] = useState<AttitudeData[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedAttitude, setSelectedAttitude] = useState<AttitudeData | null>(null);
    const selectedAttitudeRef = useRef(selectedAttitude);
    useEffect(() => {
        selectedAttitudeRef.current = selectedAttitude;
    }, [selectedAttitude]);

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

    const contextValue: AttitudeContextType = {
        attitudes,
        loading,
        error,
        fetchAttitudes,
        getAttitude,
        createOrUpdateAttitude,
        updateAttitudeDimension,
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
