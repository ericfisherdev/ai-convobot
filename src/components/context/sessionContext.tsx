import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useCompanionData } from './companionContext';
import { useAttitude } from './attitudeContext';

interface Session {
    id: string;
    companion_id: number;
    user_id: number | null;
    created_at: string;
    last_activity: string;
    attitude_state: any[];
    is_active: boolean;
}

interface SessionContextType {
    session: Session | null;
    sessionId: string | null;
    loading: boolean;
    error: string | null;
    
    // Methods
    initializeSession: () => Promise<void>;
    endSession: () => Promise<void>;
    updateSessionActivity: () => Promise<void>;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

interface SessionProviderProps {
    children: ReactNode;
}

export const SessionProvider: React.FC<SessionProviderProps> = ({ children }) => {
    const [session, setSession] = useState<Session | null>(null);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    const companionContext = useCompanionData();
    const { fetchAttitudes } = useAttitude();
    
    const companionData = companionContext?.companionData;
    
    // Initialize session on component mount
    useEffect(() => {
        const storedSessionId = localStorage.getItem('ai_companion_session_id');
        if (storedSessionId) {
            setSessionId(storedSessionId);
            loadSession(storedSessionId);
        } else if (companionData) {
            initializeSession();
        }
    }, [companionData]);
    
    // Auto-save session activity every 5 minutes
    useEffect(() => {
        if (!sessionId) return;
        
        const interval = setInterval(() => {
            updateSessionActivity();
        }, 5 * 60 * 1000); // 5 minutes
        
        return () => clearInterval(interval);
    }, [sessionId]);
    
    // Clean up on unmount
    useEffect(() => {
        return () => {
            if (sessionId) {
                // Persist session state when component unmounts
                fetch(`/api/session/${sessionId}/end`, { method: 'POST' })
                    .catch(err => console.error('Error ending session:', err));
            }
        };
    }, [sessionId]);
    
    const loadSession = async (id: string): Promise<void> => {
        setLoading(true);
        setError(null);
        
        try {
            const response = await fetch(`/api/session/${id}`);
            if (!response.ok) {
                // Session expired or not found, create new one
                await initializeSession();
                return;
            }
            
            const sessionData = await response.json();
            setSession(sessionData);
            
            // Load attitudes from session
            // Using companion_id 1 as there's only one companion in the system
            if (companionData && sessionData.attitude_state.length > 0) {
                await fetchAttitudes(1);
            }
            
            console.log('✅ Session loaded:', id);
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to load session';
            setError(errorMessage);
            console.error('Error loading session:', err);
            
            // Try to create new session on error
            await initializeSession();
        } finally {
            setLoading(false);
        }
    };
    
    const initializeSession = async (): Promise<void> => {
        if (!companionData) {
            console.warn('Cannot initialize session without companion data');
            return;
        }
        
        setLoading(true);
        setError(null);
        
        try {
            const response = await fetch('/api/session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    companion_id: 1, // Using default companion_id
                    user_id: 1, // Using default user_id
                }),
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const sessionData = await response.json();
            setSession(sessionData);
            setSessionId(sessionData.id);
            
            // Store session ID in localStorage for persistence
            localStorage.setItem('ai_companion_session_id', sessionData.id);
            
            // Load attitudes from session
            if (sessionData.attitude_state.length > 0) {
                await fetchAttitudes(1); // Using default companion_id
            }
            
            console.log('🚀 New session initialized:', sessionData.id);
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to initialize session';
            setError(errorMessage);
            console.error('Error initializing session:', err);
        } finally {
            setLoading(false);
        }
    };
    
    const endSession = async (): Promise<void> => {
        if (!sessionId) return;
        
        setLoading(true);
        setError(null);
        
        try {
            const response = await fetch(`/api/session/${sessionId}/end`, {
                method: 'POST',
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // Clear session data
            setSession(null);
            setSessionId(null);
            localStorage.removeItem('ai_companion_session_id');
            
            console.log('🔚 Session ended:', sessionId);
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to end session';
            setError(errorMessage);
            console.error('Error ending session:', err);
        } finally {
            setLoading(false);
        }
    };
    
    const updateSessionActivity = async (): Promise<void> => {
        if (!sessionId) return;
        
        try {
            // The backend automatically updates activity when we get the session
            const response = await fetch(`/api/session/${sessionId}`);
            if (response.ok) {
                const sessionData = await response.json();
                setSession(sessionData);
            }
        } catch (err) {
            console.error('Error updating session activity:', err);
        }
    };
    
    return (
        <SessionContext.Provider
            value={{
                session,
                sessionId,
                loading,
                error,
                initializeSession,
                endSession,
                updateSessionActivity,
            }}
        >
            {children}
        </SessionContext.Provider>
    );
};

export const useSession = (): SessionContextType => {
    const context = useContext(SessionContext);
    if (!context) {
        throw new Error('useSession must be used within a SessionProvider');
    }
    return context;
};