export interface AttitudeData {
    id?: number;
    companion_id: number;
    target_id: number;
    target_type: string; // 'user' | 'third_party'
    attraction: number;
    trust: number;
    fear: number;
    anger: number;
    joy: number;
    sorrow: number;
    disgust: number;
    surprise: number;
    curiosity: number;
    respect: number;
    suspicion: number;
    gratitude: number;
    jealousy: number;
    empathy: number;
    lust: number;
    love: number;
    anxiety: number;
    butterflies: number;
    submissiveness: number;
    dominance: number;
    relationship_score?: number;
    last_updated: string;
    created_at: string;
}

export interface AttitudeDimensionUpdate {
    companion_id: number;
    target_id: number;
    target_type: string;
    dimension: string;
    delta: number;
}

export const ATTITUDE_DIMENSIONS = [
    { key: 'attraction', label: 'Attraction', color: '#FF69B4', category: 'positive' },
    { key: 'trust', label: 'Trust', color: '#4CAF50', category: 'positive' },
    { key: 'fear', label: 'Fear', color: '#9C27B0', category: 'negative' },
    { key: 'anger', label: 'Anger', color: '#F44336', category: 'negative' },
    { key: 'joy', label: 'Joy', color: '#FFEB3B', category: 'positive' },
    { key: 'sorrow', label: 'Sorrow', color: '#607D8B', category: 'negative' },
    { key: 'disgust', label: 'Disgust', color: '#795548', category: 'negative' },
    { key: 'surprise', label: 'Surprise', color: '#FF9800', category: 'neutral' },
    { key: 'curiosity', label: 'Curiosity', color: '#2196F3', category: 'positive' },
    { key: 'respect', label: 'Respect', color: '#8BC34A', category: 'positive' },
    { key: 'suspicion', label: 'Suspicion', color: '#FF5722', category: 'negative' },
    { key: 'gratitude', label: 'Gratitude', color: '#00BCD4', category: 'positive' },
    { key: 'jealousy', label: 'Jealousy', color: '#E91E63', category: 'negative' },
    { key: 'empathy', label: 'Empathy', color: '#9E9E9E', category: 'positive' },
    { key: 'lust', label: 'Lust', color: '#FF1493', category: 'positive' },
    { key: 'love', label: 'Love', color: '#DC143C', category: 'positive' },
    { key: 'anxiety', label: 'Anxiety', color: '#DDA0DD', category: 'negative' },
    { key: 'butterflies', label: 'Butterflies', color: '#FFB6C1', category: 'positive' },
    { key: 'submissiveness', label: 'Submissiveness', color: '#90EE90', category: 'neutral' },
    { key: 'dominance', label: 'Dominance', color: '#FF4500', category: 'neutral' }
] as const;

export const RELATIONSHIP_STATUSES = {
    'hostile': { label: 'Hostile', color: '#F44336', range: [-100, -61] },
    'unfriendly': { label: 'Unfriendly', color: '#FF5722', range: [-60, -21] },
    'neutral': { label: 'Neutral', color: '#9E9E9E', range: [-20, 20] },
    'friendly': { label: 'Friendly', color: '#8BC34A', range: [21, 60] },
    'close': { label: 'Close', color: '#4CAF50', range: [61, 80] },
    'intimate': { label: 'Intimate', color: '#2196F3', range: [81, 100] }
} as const;