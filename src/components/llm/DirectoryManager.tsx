import React, { useState, useEffect } from 'react';
import { DirectoryInfo } from '../interfaces/Config';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { X, FolderPlus } from 'lucide-react';

interface DirectoryManagerProps {
    onDirectoriesChange?: () => void;
}

export const DirectoryManager: React.FC<DirectoryManagerProps> = ({ onDirectoriesChange }) => {
    const [directories, setDirectories] = useState<DirectoryInfo[]>([]);
    const [newPath, setNewPath] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchDirectories();
    }, []);

    const fetchDirectories = async () => {
        try {
            const response = await fetch('/api/llm/directories');
            if (response.ok) {
                const data = await response.json();
                setDirectories(data);
            } else {
                console.error('Failed to fetch directories');
            }
        } catch (error) {
            console.error('Error fetching directories:', error);
        }
    };

    const handleAddDirectory = async () => {
        if (!newPath.trim()) {
            setError('Please enter a valid directory path');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/llm/directories', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ path: newPath.trim() }),
            });

            if (response.ok) {
                setNewPath('');
                await fetchDirectories();
                if (onDirectoriesChange) {
                    onDirectoriesChange();
                }
            } else {
                setError('Failed to add directory');
            }
        } catch (error) {
            console.error('Error adding directory:', error);
            setError('Failed to add directory');
        } finally {
            setLoading(false);
        }
    };

    const handleRemoveDirectory = async (id: number) => {
        setLoading(true);
        try {
            const response = await fetch(`/api/llm/directories/${id}`, {
                method: 'DELETE',
            });

            if (response.ok) {
                await fetchDirectories();
                if (onDirectoriesChange) {
                    onDirectoriesChange();
                }
            } else {
                console.error('Failed to remove directory');
            }
        } catch (error) {
            console.error('Error removing directory:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-4">
            <div className="flex items-center space-x-2">
                <Input
                    type="text"
                    placeholder="Enter directory path to scan for models"
                    value={newPath}
                    onChange={(e) => setNewPath(e.target.value)}
                    onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                            handleAddDirectory();
                        }
                    }}
                    disabled={loading}
                    className="flex-1"
                />
                <Button
                    onClick={handleAddDirectory}
                    disabled={loading || !newPath.trim()}
                    size="sm"
                >
                    <FolderPlus className="h-4 w-4 mr-2" />
                    Add Directory
                </Button>
            </div>

            {error && (
                <div className="text-red-500 text-sm">{error}</div>
            )}

            {directories.length > 0 && (
                <div className="space-y-2">
                    <label className="text-sm font-medium">Custom Directories:</label>
                    {directories.map((dir) => (
                        <div
                            key={dir.id}
                            className="flex items-center justify-between p-2 bg-secondary rounded-md"
                        >
                            <span className="text-sm truncate flex-1">{dir.path}</span>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleRemoveDirectory(dir.id)}
                                disabled={loading}
                            >
                                <X className="h-4 w-4" />
                            </Button>
                        </div>
                    ))}
                </div>
            )}

            <div className="text-xs text-muted-foreground">
                Default directories: ./llms and ../llms (relative to executable)
            </div>
        </div>
    );
};