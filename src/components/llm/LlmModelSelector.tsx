import React, { useState, useEffect } from 'react';
import { ModelInfo } from '../interfaces/Config';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Label } from '../ui/label';

interface LlmModelSelectorProps {
    selectedModel: string | undefined;
    onModelSelect: (modelPath: string) => void;
    refreshTrigger?: number;
}

export const LlmModelSelector: React.FC<LlmModelSelectorProps> = ({
    selectedModel,
    onModelSelect,
    refreshTrigger
}) => {
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetchModels();
    }, [refreshTrigger]);

    const fetchModels = async () => {
        setLoading(true);
        try {
            const response = await fetch('/api/llm/models');
            if (response.ok) {
                const data = await response.json();
                setModels(data);
            } else {
                console.error('Failed to fetch models');
            }
        } catch (error) {
            console.error('Error fetching models:', error);
        } finally {
            setLoading(false);
        }
    };

    const formatFileSize = (bytes: number): string => {
        const gb = bytes / (1024 * 1024 * 1024);
        if (gb >= 1) {
            return `${gb.toFixed(2)} GB`;
        }
        const mb = bytes / (1024 * 1024);
        return `${mb.toFixed(2)} MB`;
    };

    const groupModelsByDirectory = () => {
        const grouped: { [key: string]: ModelInfo[] } = {};
        models.forEach(model => {
            if (!grouped[model.directory]) {
                grouped[model.directory] = [];
            }
            grouped[model.directory].push(model);
        });
        return grouped;
    };

    const groupedModels = groupModelsByDirectory();

    return (
        <div className="space-y-2">
            <Label htmlFor="model-select">Select LLM Model</Label>
            <Select
                value={selectedModel || ''}
                onValueChange={onModelSelect}
                disabled={loading}
            >
                <SelectTrigger id="model-select">
                    <SelectValue placeholder={loading ? "Loading models..." : "Select a model"} />
                </SelectTrigger>
                <SelectContent>
                    {models.length === 0 && !loading ? (
                        <SelectItem value="" disabled>
                            No models found
                        </SelectItem>
                    ) : (
                        Object.entries(groupedModels).map(([directory, dirModels]) => (
                            <div key={directory}>
                                <div className="px-2 py-1 text-sm font-semibold text-muted-foreground">
                                    {directory}
                                </div>
                                {dirModels.map(model => (
                                    <SelectItem key={model.path} value={model.path}>
                                        <div className="flex items-center justify-between w-full">
                                            <span>{model.filename}</span>
                                            <span className="ml-2 text-xs text-muted-foreground">
                                                {formatFileSize(model.size_bytes)}
                                            </span>
                                        </div>
                                    </SelectItem>
                                ))}
                            </div>
                        ))
                    )}
                </SelectContent>
            </Select>
            {selectedModel && (
                <div className="text-xs text-muted-foreground">
                    Current: {selectedModel}
                </div>
            )}
        </div>
    );
};