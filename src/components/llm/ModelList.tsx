import React, { useState, useEffect } from 'react';
import { ModelInfo } from '../interfaces/Config';
import { Card, CardContent } from '../ui/card';
import { FileText, Calendar, HardDrive } from 'lucide-react';

interface ModelListProps {
    selectedModel?: string;
    onModelSelect?: (modelPath: string) => void;
    refreshTrigger?: number;
}

export const ModelList: React.FC<ModelListProps> = ({
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

    if (loading) {
        return (
            <div className="text-center py-4">
                <div className="text-sm text-muted-foreground">Loading models...</div>
            </div>
        );
    }

    if (models.length === 0) {
        return (
            <Card>
                <CardContent className="py-6">
                    <div className="text-center text-muted-foreground">
                        <FileText className="h-12 w-12 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">No GGUF models found</p>
                        <p className="text-xs mt-2">Add directories above to scan for models</p>
                    </div>
                </CardContent>
            </Card>
        );
    }

    return (
        <div className="space-y-4">
            <div className="text-sm font-medium">
                Available Models ({models.length} found)
            </div>
            {Object.entries(groupedModels).map(([directory, dirModels]) => (
                <div key={directory} className="space-y-2">
                    <div className="text-xs text-muted-foreground font-medium">
                        {directory}
                    </div>
                    <div className="grid gap-2">
                        {dirModels.map(model => (
                            <Card
                                key={model.path}
                                className={`cursor-pointer transition-colors ${
                                    selectedModel === model.path 
                                        ? 'border-primary bg-primary/5' 
                                        : 'hover:bg-secondary/50'
                                }`}
                                onClick={() => onModelSelect && onModelSelect(model.path)}
                            >
                                <CardContent className="p-3">
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2">
                                                <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                                                <span className="font-medium text-sm truncate">
                                                    {model.filename}
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                                                <div className="flex items-center gap-1">
                                                    <HardDrive className="h-3 w-3" />
                                                    <span>{formatFileSize(model.size_bytes)}</span>
                                                </div>
                                                <div className="flex items-center gap-1">
                                                    <Calendar className="h-3 w-3" />
                                                    <span>{model.last_modified}</span>
                                                </div>
                                            </div>
                                        </div>
                                        {selectedModel === model.path && (
                                            <div className="ml-2 px-2 py-1 bg-primary text-primary-foreground text-xs rounded">
                                                Selected
                                            </div>
                                        )}
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </div>
            ))}
        </div>
    );
};