import React, { useState, useEffect } from 'react';
import { ModelInfo } from '../interfaces/Config';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Label } from '../ui/label';

// The `Select`'s internal value for "no model chosen" when `allowNone` is
// set — Radix `Select.Item` rejects an empty-string value, so this stands in
// for it and `onModelSelect` receives `''` instead (the same "blank means
// unset" convention `MemorySettings` uses for `compact_threshold_tokens`).
const NONE_VALUE = '__none__';

interface LlmModelSelectorProps {
    selectedModel: string | undefined;
    onModelSelect: (modelPath: string) => void;
    refreshTrigger?: number;
    /** Defaults to `"model-select"`; set to disambiguate multiple pickers on one page. */
    id?: string;
    /** Defaults to `"Select LLM Model"`. */
    label?: string;
    /** When set, only models matching this predicate are listed. */
    filter?: (model: ModelInfo) => boolean;
    /** Adds a leading "Same as chat model" option; selecting it calls `onModelSelect('')`. */
    allowNone?: boolean;
}

export const LlmModelSelector: React.FC<LlmModelSelectorProps> = ({
    selectedModel,
    onModelSelect,
    refreshTrigger,
    id = 'model-select',
    label = 'Select LLM Model',
    filter,
    allowNone = false,
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

    const filteredModels = filter ? models.filter(filter) : models;
    // A `filter` must never hide the currently selected model: that would
    // leave the trigger showing a blank placeholder while the underlying
    // config value is still set to a real (just filtered-out) path.
    const selectedModelInfo = selectedModel ? models.find(model => model.path === selectedModel) : undefined;
    const visibleModels =
        selectedModelInfo && !filteredModels.includes(selectedModelInfo)
            ? [selectedModelInfo, ...filteredModels]
            : filteredModels;

    const groupModelsByDirectory = () => {
        const grouped: { [key: string]: ModelInfo[] } = {};
        visibleModels.forEach(model => {
            if (!grouped[model.directory]) {
                grouped[model.directory] = [];
            }
            grouped[model.directory].push(model);
        });
        return grouped;
    };

    const groupedModels = groupModelsByDirectory();

    const handleValueChange = (value: string) => {
        onModelSelect(value === NONE_VALUE ? '' : value);
    };

    return (
        <div className="space-y-2">
            <Label htmlFor={id}>{label}</Label>
            <Select
                value={selectedModel || (allowNone ? NONE_VALUE : '')}
                onValueChange={handleValueChange}
                disabled={loading}
            >
                <SelectTrigger id={id}>
                    <SelectValue placeholder={loading ? "Loading models..." : "Select a model"} />
                </SelectTrigger>
                <SelectContent>
                    {allowNone && (
                        <SelectItem value={NONE_VALUE}>Same as chat model</SelectItem>
                    )}
                    {visibleModels.length === 0 && !loading ? (
                        <div className="px-2 py-1.5 text-sm text-muted-foreground">
                            No models found
                        </div>
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
