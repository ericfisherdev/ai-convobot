import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { Progress } from "../ui/progress";
import { AttitudeData, ATTITUDE_DIMENSIONS } from '../interfaces/AttitudeData';
import { useAttitude } from '../context/attitudeContext';
import { useCompanionData } from '../context/companionContext';
import { useUserData } from '../context/userContext';

interface AttitudeSummaryBarProps {
    companionId: number;
    userId: number;
}

interface AttitudeSummaryResponse {
    attitude: AttitudeData;
    summary: string;
}

export const AttitudeSummaryBar: React.FC<AttitudeSummaryBarProps> = ({ companionId, userId }) => {
    const { getAttitude } = useAttitude();
    const companionDataContext = useCompanionData();
    const companionData = companionDataContext?.companionData;
    const userDataContext = useUserData();
    const userData = userDataContext?.userData;
    const [attitude, setAttitude] = useState<AttitudeData | null>(null);
    const [summary, setSummary] = useState<string>('');
    const [loading, setLoading] = useState(true);

    // Fetch attitude summary from backend
    const fetchAttitudeSummary = useCallback(async () => {
        try {
            const response = await fetch(`/api/attitude/summary/${companionId}/${userId}`);
            if (response.ok) {
                const data: AttitudeSummaryResponse = await response.json();
                setAttitude(data.attitude);
                
                // Replace placeholders with actual names
                const formattedSummary = data.summary
                    .replace(/\{\{companion\}\}/g, companionData?.name || 'Companion')
                    .replace(/\{\{user\}\}/g, userData?.name || 'User');
                setSummary(formattedSummary);
            } else {
                // Fallback to regular attitude fetch if summary endpoint not available
                const attitudeData = await getAttitude(companionId, userId, 'user');
                if (attitudeData) {
                    setAttitude(attitudeData);
                    generateLocalSummary(attitudeData);
                }
            }
        } catch (error) {
            console.error('Error fetching attitude summary:', error);
            // Fallback to regular attitude fetch
            const attitudeData = await getAttitude(companionId, userId, 'user');
            if (attitudeData) {
                setAttitude(attitudeData);
                generateLocalSummary(attitudeData);
            }
        } finally {
            setLoading(false);
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [companionId, userId, companionData?.name, userData?.name, getAttitude]); // generateLocalSummary is stable

    // Generate summary locally if backend endpoint not available
    const generateLocalSummary = useCallback((attitudeData: AttitudeData) => {
        const companionName = companionData?.name || 'Companion';
        const userName = userData?.name || 'User';
        
        // Find dominant emotions
        const emotions = [
            { key: 'love', value: attitudeData.love },
            { key: 'attraction', value: attitudeData.attraction },
            { key: 'lust', value: attitudeData.lust },
            { key: 'trust', value: attitudeData.trust },
            { key: 'anger', value: attitudeData.anger },
            { key: 'suspicion', value: attitudeData.suspicion },
            { key: 'curiosity', value: attitudeData.curiosity },
            { key: 'butterflies', value: attitudeData.butterflies }
        ].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

        const dominant = emotions[0];
        const secondary = emotions[1];

        let summaryText = '';

        // Generate contextual summary based on dominant emotions
        if (dominant.value > 70) {
            if (dominant.key === 'love' && secondary.key === 'trust') {
                summaryText = `${companionName} is deeply in love with ${userName}`;
            } else if (dominant.key === 'attraction' && attitudeData.lust > 50) {
                summaryText = `${companionName} really wants to be intimate with ${userName}`;
            } else if (dominant.key === 'anger') {
                summaryText = `${companionName} is upset with ${userName}`;
            } else if (dominant.key === 'curiosity' && attitudeData.butterflies > 50) {
                summaryText = `${companionName} is nervously excited about ${userName}`;
            } else if (dominant.key === 'trust') {
                summaryText = `${companionName} deeply trusts ${userName}`;
            } else {
                summaryText = `${companionName} feels strongly about ${userName}`;
            }
        } else if (dominant.value > 40) {
            if (dominant.key === 'love') {
                summaryText = `${companionName} cares about ${userName}`;
            } else if (dominant.key === 'attraction') {
                summaryText = `${companionName} is attracted to ${userName}`;
            } else if (dominant.key === 'curiosity') {
                summaryText = `${companionName} is curious about ${userName}`;
            } else {
                summaryText = `${companionName} has mixed feelings about ${userName}`;
            }
        } else if (dominant.value < -40) {
            if (dominant.key === 'anger' || dominant.key === 'suspicion') {
                summaryText = `${companionName} is upset and distrustful of ${userName}`;
            } else {
                summaryText = `${companionName} has negative feelings toward ${userName}`;
            }
        } else {
            summaryText = `${companionName} feels neutral toward ${userName}`;
        }

        setSummary(summaryText);
    }, [companionData?.name, userData?.name]);

    // Filter for significant attitudes (> 30 or < -30)
    const significantAttitudes = useMemo(() => {
        if (!attitude) return [];
        
        return ATTITUDE_DIMENSIONS.filter(dimension => {
            const value = attitude[dimension.key as keyof AttitudeData] as number;
            return Math.abs(value) > 30;
        }).map(dimension => ({
            ...dimension,
            value: attitude[dimension.key as keyof AttitudeData] as number
        }));
    }, [attitude]);

    // Responsive grid columns
    const getGridColumns = () => {
        const attitudeCount = significantAttitudes.length;
        if (attitudeCount === 0) return '';
        if (attitudeCount === 1) return 'grid-cols-1';
        if (attitudeCount === 2) return 'grid-cols-2';
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3';
    };

    const formatValue = (value: number) => {
        return value > 0 ? `+${value.toFixed(0)}` : value.toFixed(0);
    };

    const getProgressValue = (value: number) => {
        return ((value + 100) / 200) * 100; // Convert -100 to +100 range to 0-100%
    };

    useEffect(() => {
        if (companionId && userId) {
            fetchAttitudeSummary();
        }
    }, [companionId, userId, fetchAttitudeSummary]);

    // Update on message send/receive
    useEffect(() => {
        const handleAttitudeUpdate = () => {
            fetchAttitudeSummary();
        };

        window.addEventListener('attitude-update', handleAttitudeUpdate);
        return () => {
            window.removeEventListener('attitude-update', handleAttitudeUpdate);
        };
    }, [fetchAttitudeSummary]);

    if (loading || !attitude || significantAttitudes.length === 0) {
        return null;
    }

    return (
        <div className="attitude-summary-container px-4 py-3 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            {/* Natural language summary */}
            <p className="text-center mb-3 text-sm text-muted-foreground italic">
                {summary}
            </p>
            
            {/* Attitude bars grid */}
            <div className={`grid ${getGridColumns()} gap-3 max-w-4xl mx-auto`}>
                {significantAttitudes.map(attitude => (
                    <div key={attitude.key} className="space-y-1">
                        <div className="flex justify-between items-center">
                            <span className="text-xs font-medium">{attitude.label}</span>
                            <span 
                                className="text-xs font-mono"
                                style={{ color: attitude.color }}
                            >
                                {formatValue(attitude.value)}
                            </span>
                        </div>
                        <Progress 
                            value={getProgressValue(attitude.value)} 
                            className="h-1.5"
                            style={{ 
                                '--progress-background': attitude.color + '20',
                                '--progress-foreground': attitude.color 
                            } as React.CSSProperties}
                        />
                    </div>
                ))}
            </div>
        </div>
    );
};