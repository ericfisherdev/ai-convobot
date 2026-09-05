import React, { useEffect, useMemo, useCallback } from 'react';
import { Progress } from "../ui/progress";
import {
    AttitudeData,
    ATTITUDE_DIMENSIONS,
    ATTITUDE_DISPLAY_THRESHOLD,
    CORE_ATTITUDE_DIMENSIONS,
} from '../interfaces/AttitudeData';
import { useAttitude } from '../context/attitudeContext';
import { useCompanionData } from '../context/companionContext';
import { useUserData } from '../context/userContext';

interface AttitudeSummaryBarProps {
    companionId: number;
    userId: number;
    // Magnitude above which a non-core, unchanged dimension is still shown.
    significanceThreshold?: number;
}

export const AttitudeSummaryBar: React.FC<AttitudeSummaryBarProps> = ({
    companionId,
    userId,
    significanceThreshold = ATTITUDE_DISPLAY_THRESHOLD,
}) => {
    const {
        userAttitude: attitude,
        userAttitudeSummary,
        lastTurnDeltas,
        userAttitudeLoaded,
        refreshUserAttitude,
    } = useAttitude();
    const companionDataContext = useCompanionData();
    const companionData = companionDataContext?.companionData;
    const userDataContext = useUserData();
    const userData = userDataContext?.userData;

    const companionName = companionData?.name || 'Companion';
    const userName = userData?.name || 'User';

    useEffect(() => {
        if (companionId && userId) {
            refreshUserAttitude(companionId, userId);
        }
    }, [companionId, userId, refreshUserAttitude]);

    // Used when the backend returned no summary (older builds have no summary
    // endpoint, and the raw attitude fallback carries no prose).
    const generateLocalSummary = useCallback((attitudeData: AttitudeData): string => {
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

        // Generate contextual summary based on dominant emotions
        if (dominant.value > 70) {
            if (dominant.key === 'love' && secondary.key === 'trust') {
                return `${companionName} is deeply in love with ${userName}`;
            } else if (dominant.key === 'attraction' && attitudeData.lust > 50) {
                return `${companionName} really wants to be intimate with ${userName}`;
            } else if (dominant.key === 'anger') {
                return `${companionName} is upset with ${userName}`;
            } else if (dominant.key === 'curiosity' && attitudeData.butterflies > 50) {
                return `${companionName} is nervously excited about ${userName}`;
            } else if (dominant.key === 'trust') {
                return `${companionName} deeply trusts ${userName}`;
            }
            return `${companionName} feels strongly about ${userName}`;
        }

        if (dominant.value > 40) {
            if (dominant.key === 'love') {
                return `${companionName} cares about ${userName}`;
            } else if (dominant.key === 'attraction') {
                return `${companionName} is attracted to ${userName}`;
            } else if (dominant.key === 'curiosity') {
                return `${companionName} is curious about ${userName}`;
            }
            return `${companionName} has mixed feelings about ${userName}`;
        }

        if (dominant.value < -40) {
            if (dominant.key === 'anger' || dominant.key === 'suspicion') {
                return `${companionName} is upset and distrustful of ${userName}`;
            }
            return `${companionName} has negative feelings toward ${userName}`;
        }

        return `${companionName} feels neutral toward ${userName}`;
    }, [companionName, userName]);

    const summary = useMemo(() => {
        if (userAttitudeSummary) {
            return userAttitudeSummary
                .replace(/\{\{companion\}\}/g, companionName)
                .replace(/\{\{user\}\}/g, userName);
        }
        return attitude ? generateLocalSummary(attitude) : '';
    }, [userAttitudeSummary, attitude, companionName, userName, generateLocalSummary]);

    // Core dimensions are always shown so the bar never renders empty; anything
    // strong enough, or moved by the last turn, joins them.
    const visibleAttitudes = useMemo(() => {
        if (!attitude) return [];

        return ATTITUDE_DIMENSIONS.filter(dimension => {
            const value = attitude[dimension.key] as number;
            return (CORE_ATTITUDE_DIMENSIONS as readonly string[]).includes(dimension.key)
                || Math.abs(value) > significanceThreshold
                || dimension.key in lastTurnDeltas;
        }).map(dimension => ({
            ...dimension,
            value: attitude[dimension.key] as number,
            delta: lastTurnDeltas[dimension.key]
        }));
    }, [attitude, lastTurnDeltas, significanceThreshold]);

    // Responsive grid columns
    const getGridColumns = () => {
        const attitudeCount = visibleAttitudes.length;
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

    // Only the very first load hides the bar; a later refetch keeps the last
    // known values on screen instead of flashing empty.
    if (!userAttitudeLoaded || !attitude) {
        return null;
    }

    return (
        <div
            className="attitude-summary-container px-4 py-3 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
            data-testid="attitude-summary-bar"
        >
            {/* Natural language summary */}
            <p className="text-center mb-3 text-sm text-muted-foreground italic">
                {summary}
            </p>

            {/* Attitude bars grid */}
            <div className={`grid ${getGridColumns()} gap-3 max-w-4xl mx-auto`}>
                {visibleAttitudes.map(dimension => (
                    <div key={dimension.key} className="space-y-1">
                        <div className="flex justify-between items-center">
                            <span className="text-xs font-medium">{dimension.label}</span>
                            <div className="flex items-center gap-1">
                                {dimension.delta !== undefined && (
                                    <span
                                        className={`text-xs font-mono ${dimension.delta > 0 ? 'text-emerald-500' : 'text-red-500'}`}
                                        data-testid={`attitude-delta-${dimension.key}`}
                                        aria-label={`${dimension.label} changed by ${formatValue(dimension.delta)}`}
                                    >
                                        {formatValue(dimension.delta)}
                                    </span>
                                )}
                                <span
                                    className="text-xs font-mono"
                                    style={{ color: dimension.color }}
                                >
                                    {formatValue(dimension.value)}
                                </span>
                            </div>
                        </div>
                        <Progress
                            value={getProgressValue(dimension.value)}
                            className="h-1.5"
                            style={{
                                '--progress-background': dimension.color + '20',
                                '--progress-foreground': dimension.color
                            } as React.CSSProperties}
                        />
                    </div>
                ))}
            </div>
        </div>
    );
};
