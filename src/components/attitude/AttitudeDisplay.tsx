import React from 'react';
import { Card } from "../ui/card";
import { Progress } from "../ui/progress";
import { Badge } from "../ui/badge";
import { AttitudeData, ATTITUDE_DIMENSIONS, RELATIONSHIP_STATUSES } from '../interfaces/AttitudeData';

interface AttitudeDisplayProps {
    attitude: AttitudeData;
    targetName?: string;
    showDetails?: boolean;
}

export const AttitudeDisplay: React.FC<AttitudeDisplayProps> = ({ 
    attitude, 
    targetName = "Unknown",
    showDetails = true
}) => {
    const getRelationshipStatus = (score: number) => {
        for (const [status, config] of Object.entries(RELATIONSHIP_STATUSES)) {
            if (score >= config.range[0] && score <= config.range[1]) {
                return { status, ...config };
            }
        }
        return { status: 'neutral', ...RELATIONSHIP_STATUSES.neutral };
    };

    const relationshipStatus = getRelationshipStatus(attitude.relationship_score || 0);

    const formatValue = (value: number) => {
        return value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1);
    };

    const getProgressValue = (value: number) => {
        return ((value + 100) / 200) * 100; // Convert -100 to +100 range to 0-100%
    };

    return (
        <Card className="p-6 space-y-4">
            <div className="flex items-center justify-between">
                <div>
                    <h3 className="text-lg font-semibold">{targetName}</h3>
                    <p className="text-sm text-muted-foreground">
                        {attitude.target_type === 'user' ? 'User' : 'Third Party'}
                    </p>
                </div>
                <div className="text-right">
                    <Badge 
                        variant="outline" 
                        style={{ 
                            borderColor: relationshipStatus.color,
                            color: relationshipStatus.color 
                        }}
                    >
                        {relationshipStatus.label}
                    </Badge>
                    <p className="text-sm text-muted-foreground mt-1">
                        Score: {formatValue(attitude.relationship_score || 0)}
                    </p>
                </div>
            </div>

            {showDetails && (
                <div className="space-y-3">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {ATTITUDE_DIMENSIONS.map(dimension => {
                            const value = attitude[dimension.key as keyof AttitudeData] as number;
                            return (
                                <div key={dimension.key} className="space-y-1">
                                    <div className="flex justify-between items-center">
                                        <span className="text-sm font-medium">{dimension.label}</span>
                                        <span 
                                            className="text-sm font-mono"
                                            style={{ color: dimension.color }}
                                        >
                                            {formatValue(value)}
                                        </span>
                                    </div>
                                    <Progress 
                                        value={getProgressValue(value)} 
                                        className="h-2"
                                        style={{ 
                                            '--progress-background': dimension.color + '20',
                                            '--progress-foreground': dimension.color 
                                        } as React.CSSProperties}
                                    />
                                </div>
                            );
                        })}
                    </div>
                    
                    <div className="pt-2 border-t text-xs text-muted-foreground">
                        <p>Last updated: {new Date(attitude.last_updated).toLocaleString()}</p>
                        <p>Created: {new Date(attitude.created_at).toLocaleString()}</p>
                    </div>
                </div>
            )}
        </Card>
    );
};