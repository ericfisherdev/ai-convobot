import React, { useState, useEffect, useCallback } from 'react';
import { Card } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { AttitudeDisplay } from './AttitudeDisplay';
import { AttitudeData, ATTITUDE_DIMENSIONS } from '../interfaces/AttitudeData';

interface AttitudeManagerProps {
    companionId: number;
}

export const AttitudeManager: React.FC<AttitudeManagerProps> = ({ companionId }) => {
    const [attitudes, setAttitudes] = useState<AttitudeData[]>([]);
    const [selectedAttitude, setSelectedAttitude] = useState<AttitudeData | null>(null);
    const [newAttitude, setNewAttitude] = useState<{
        target_id: string;
        target_type: 'user' | 'third_party';
        targetName: string;
    }>({
        target_id: '',
        target_type: 'user',
        targetName: ''
    });

    const fetchAttitudes = useCallback(async () => {
        try {
            const response = await fetch(`/api/attitude/companion/${companionId}`);
            if (response.ok) {
                const data = await response.json();
                setAttitudes(data);
            }
        } catch (error) {
            console.error('Error fetching attitudes:', error);
        }
    }, [companionId]);

    useEffect(() => {
        fetchAttitudes();
    }, [fetchAttitudes]);

    const createNewAttitude = async () => {
        if (!newAttitude.target_id || !newAttitude.targetName) return;

        const attitudeData: Partial<AttitudeData> = {
            companion_id: companionId,
            target_id: parseInt(newAttitude.target_id),
            target_type: newAttitude.target_type,
            attraction: 0,
            trust: 0,
            fear: 0,
            anger: 0,
            joy: 0,
            sorrow: 0,
            disgust: 0,
            surprise: 0,
            curiosity: 0,
            respect: 0,
            suspicion: 0,
            gratitude: 0,
            jealousy: 0,
            empathy: 0,
            last_updated: new Date().toISOString(),
            created_at: new Date().toISOString()
        };

        try {
            const response = await fetch('/api/attitude', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(attitudeData),
            });

            if (response.ok) {
                fetchAttitudes();
                setNewAttitude({ target_id: '', target_type: 'user', targetName: '' });
            }
        } catch (error) {
            console.error('Error creating attitude:', error);
        }
    };

    const updateAttitudeDimension = async (dimension: string, delta: number) => {
        if (!selectedAttitude) return;

        try {
            const response = await fetch('/api/attitude/dimension', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    companion_id: selectedAttitude.companion_id,
                    target_id: selectedAttitude.target_id,
                    target_type: selectedAttitude.target_type,
                    dimension,
                    delta
                }),
            });

            if (response.ok) {
                fetchAttitudes();
                // Update selected attitude
                const updatedAttitudes = await fetch(`/api/attitude/companion/${companionId}`)
                    .then(res => res.json());
                const updated = updatedAttitudes.find((a: AttitudeData) => 
                    a.companion_id === selectedAttitude.companion_id && 
                    a.target_id === selectedAttitude.target_id &&
                    a.target_type === selectedAttitude.target_type
                );
                setSelectedAttitude(updated || null);
            }
        } catch (error) {
            console.error('Error updating attitude dimension:', error);
        }
    };

    return (
        <div className="space-y-6" data-testid="attitude-manager">
            <Card className="p-6">
                <h2 className="text-xl font-semibold mb-4">Attitude Tracking System</h2>
                
                <Tabs defaultValue="overview" className="w-full">
                    <TabsList className="grid w-full grid-cols-3">
                        <TabsTrigger value="overview">Overview</TabsTrigger>
                        <TabsTrigger value="details">Details</TabsTrigger>
                        <TabsTrigger value="manage">Manage</TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="overview" className="space-y-4">
                        <div className="grid gap-4">
                            {attitudes.length === 0 ? (
                                <p className="text-muted-foreground">No attitudes tracked yet.</p>
                            ) : (
                                attitudes.map(attitude => (
                                    <AttitudeDisplay
                                        key={`${attitude.target_id}-${attitude.target_type}`}
                                        attitude={attitude}
                                        targetName={`${attitude.target_type === 'user' ? 'User' : 'Third Party'} ${attitude.target_id}`}
                                        showDetails={false}
                                    />
                                ))
                            )}
                        </div>
                    </TabsContent>
                    
                    <TabsContent value="details" className="space-y-4">
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <Label>Select Attitude to View</Label>
                                <Select 
                                    value={selectedAttitude ? `${selectedAttitude.target_id}-${selectedAttitude.target_type}` : ''}
                                    onValueChange={(value) => {
                                        const [targetId, targetType] = value.split('-');
                                        const attitude = attitudes.find(a => 
                                            a.target_id === parseInt(targetId) && a.target_type === targetType
                                        );
                                        setSelectedAttitude(attitude || null);
                                    }}
                                >
                                    <SelectTrigger>
                                        <SelectValue placeholder="Select an attitude to view details" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {attitudes.map(attitude => (
                                            <SelectItem 
                                                key={`${attitude.target_id}-${attitude.target_type}`}
                                                value={`${attitude.target_id}-${attitude.target_type}`}
                                            >
                                                {attitude.target_type === 'user' ? 'User' : 'Third Party'} {attitude.target_id}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>
                        </div>
                        
                        {selectedAttitude && (
                            <div className="space-y-4">
                                <AttitudeDisplay
                                    attitude={selectedAttitude}
                                    targetName={`${selectedAttitude.target_type === 'user' ? 'User' : 'Third Party'} ${selectedAttitude.target_id}`}
                                    showDetails={true}
                                />
                                
                                <Card className="p-4">
                                    <h4 className="font-semibold mb-3">Quick Adjustments</h4>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                        {ATTITUDE_DIMENSIONS.map(dimension => (
                                            <div key={dimension.key} className="space-y-1">
                                                <Label className="text-xs">{dimension.label}</Label>
                                                <div className="flex gap-1">
                                                    <Button 
                                                        size="sm" 
                                                        variant="outline"
                                                        onClick={() => updateAttitudeDimension(dimension.key, -5)}
                                                        className="px-2 text-xs"
                                                    >
                                                        -5
                                                    </Button>
                                                    <Button 
                                                        size="sm" 
                                                        variant="outline"
                                                        onClick={() => updateAttitudeDimension(dimension.key, 5)}
                                                        className="px-2 text-xs"
                                                    >
                                                        +5
                                                    </Button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </Card>
                            </div>
                        )}
                    </TabsContent>
                    
                    <TabsContent value="manage" className="space-y-4">
                        <Card className="p-4">
                            <h4 className="font-semibold mb-3">Create New Attitude</h4>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="targetId">Target ID</Label>
                                    <Input
                                        id="targetId"
                                        type="number"
                                        placeholder="Target ID"
                                        value={newAttitude.target_id}
                                        onChange={(e) => setNewAttitude(prev => ({
                                            ...prev,
                                            target_id: e.target.value
                                        }))}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="targetType">Target Type</Label>
                                    <Select 
                                        value={newAttitude.target_type}
                                        onValueChange={(value: 'user' | 'third_party') => setNewAttitude(prev => ({
                                            ...prev,
                                            target_type: value
                                        }))}
                                    >
                                        <SelectTrigger>
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="user">User</SelectItem>
                                            <SelectItem value="third_party">Third Party</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="targetName">Display Name</Label>
                                    <Input
                                        id="targetName"
                                        placeholder="Display name"
                                        value={newAttitude.targetName}
                                        onChange={(e) => setNewAttitude(prev => ({
                                            ...prev,
                                            targetName: e.target.value
                                        }))}
                                    />
                                </div>
                            </div>
                            <Button 
                                onClick={createNewAttitude}
                                className="mt-4"
                                disabled={!newAttitude.target_id || !newAttitude.targetName}
                            >
                                Create Attitude
                            </Button>
                        </Card>
                    </TabsContent>
                </Tabs>
            </Card>
        </div>
    );
};