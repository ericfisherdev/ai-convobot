import { useState } from "react"
import { toast } from "sonner"

import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

import { ConfigInterface, ModelInfo } from "../interfaces/Config"
import { LlmModelSelector } from "../llm/LlmModelSelector"
import { useCompaction } from "../context/compactionContext"
import { CheckpointSummary } from "../interfaces/Compaction"
import { formatMessageDate } from "../../lib/utils"
import { FieldLabel } from "./FieldLabel"

interface MemorySettingsProps {
  config: ConfigInterface;
  onChange: (next: ConfigInterface) => void;
}

// Extraction models are small instruct-tuned checkpoints, not the (often
// larger, base or roleplay-tuned) chat model. Matches filenames like
// "Qwen3-4B-Instruct...", "llama-3.2-3b-it...", or "...-chat...".
const INSTRUCT_MODEL_FILTER = /instruct|-it[-_.]|chat/i;

// The six blocks rendered into the system prompt by `compaction::render`,
// mirroring the backend's `RenderedBlocks` (nested under `compaction` in
// `GET /api/debug/prompt`'s `AssembledPrompt` body).
interface RenderedPromptBlocks {
  user_overlay: string;
  companion_overlay: string;
  rules: string;
  story_so_far: string;
  recent_detail: string;
  pins: string;
}

const BLOCK_LABELS: Record<keyof RenderedPromptBlocks, string> = {
  user_overlay: "User overlay",
  companion_overlay: "Companion overlay",
  rules: "Rules",
  story_so_far: "Story so far",
  recent_detail: "Recent detail",
  pins: "Pins",
};

function checkpointBadges(checkpoint: CheckpointSummary) {
  return (
    <div className="flex gap-1 mt-1">
      {checkpoint.status === 'stale' && (
        <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-700 dark:text-yellow-400">
          Stale
        </span>
      )}
      {checkpoint.needs_merge && (
        <span className="text-xs px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-700 dark:text-blue-400">
          Needs merge
        </span>
      )}
    </div>
  );
}

export function MemorySettings({ config, onChange }: MemorySettingsProps) {
  const { checkpoints } = useCompaction();
  const [showAllModels, setShowAllModels] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [notesCheckpointId, setNotesCheckpointId] = useState<number | null>(null);
  const [renderedBlocks, setRenderedBlocks] = useState<RenderedPromptBlocks | null>(null);
  const [loadingNotes, setLoadingNotes] = useState(false);

  const handleRebuild = async () => {
    setRebuilding(true);
    try {
      const response = await fetch('/api/memory/longTerm/rebuild', { method: 'POST' });
      const text = await response.text();
      if (response.ok) {
        // The body carries the re-indexed fact count (e.g. "Long term
        // memory rebuilt from 12 facts") — show it verbatim rather than a
        // generic message.
        toast.success(text || 'Long-term index rebuilt successfully');
      } else {
        toast.error(text || 'Failed to rebuild long-term index');
      }
    } catch (error) {
      toast.error(`Error while rebuilding long-term index: ${error}`);
    } finally {
      setRebuilding(false);
    }
  };

  const handleViewNotes = async (checkpointId: number) => {
    setNotesCheckpointId(checkpointId);
    setRenderedBlocks(null);
    setLoadingNotes(true);
    try {
      const response = await fetch('/api/debug/prompt');
      if (!response.ok) {
        throw new Error(`GET /api/debug/prompt returned ${response.status}`);
      }
      const data: { compaction: RenderedPromptBlocks } = await response.json();
      setRenderedBlocks(data.compaction);
    } catch (error) {
      toast.error(`Error while fetching rendered notes: ${error}`);
    } finally {
      setLoadingNotes(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <FieldLabel htmlFor="compactThresholdTokens" tooltip="2 x the recent-message slice when blank">
          Compaction threshold (tokens)
        </FieldLabel>
        <Input
          id="compactThresholdTokens"
          type="number"
          min={256}
          placeholder="Automatic"
          value={config.compact_threshold_tokens ?? ''}
          onChange={(e) => {
            const raw = e.target.value;
            onChange({ ...config, compact_threshold_tokens: raw === '' ? null : parseInt(raw) });
          }}
        />
      </div>

      <div className="space-y-1">
        <Label htmlFor="compactMinMessages">Minimum messages before compaction</Label>
        <Input
          id="compactMinMessages"
          type="number"
          min={2}
          value={config.compact_min_messages}
          onChange={(e) => onChange({ ...config, compact_min_messages: parseInt(e.target.value) })}
        />
      </div>

      <div className="space-y-1">
        <FieldLabel
          htmlFor="compactionAttitudeWeight"
          tooltip="At each compaction commit, how far the companion's feelings move toward the story's rating: 0 keeps the running values, 1 adopts the rating"
        >
          Narrative attitude weight
        </FieldLabel>
        <Input
          id="compactionAttitudeWeight"
          type="number"
          step={0.05}
          min={0}
          max={1}
          value={config.compaction_attitude_weight}
          onChange={(e) => onChange({ ...config, compaction_attitude_weight: parseFloat(e.target.value) })}
        />
      </div>

      <div className="flex items-center justify-between">
        <FieldLabel
          htmlFor="heuristicPersonDetection"
          tooltip="Detect people the conversation mentions with lexical heuristics instead of an extra model pass"
        >
          Heuristic person detection
        </FieldLabel>
        <Switch
          id="heuristicPersonDetection"
          checked={config.heuristic_person_detection}
          onCheckedChange={(checked) => onChange({ ...config, heuristic_person_detection: checked })}
        />
      </div>

      <div className="space-y-2 border-t pt-4">
        <LlmModelSelector
          id="compaction-model-select"
          label="Extraction model"
          selectedModel={config.compaction_model_path ?? undefined}
          onModelSelect={(modelPath) => onChange({ ...config, compaction_model_path: modelPath || null })}
          allowNone
          filter={showAllModels ? undefined : (model: ModelInfo) => INSTRUCT_MODEL_FILTER.test(model.filename)}
        />
        <div className="flex items-center gap-2">
          <Switch id="showAllModels" checked={showAllModels} onCheckedChange={setShowAllModels} />
          <Label htmlFor="showAllModels" className="text-sm text-muted-foreground">Show all models</Label>
        </div>
        <p className="text-sm text-muted-foreground">
          3B-8B instruct models at Q4_K_M work best (for example Qwen3-4B or Llama 3.2 3B). The chat model is
          used when blank; switching models per compaction costs a reload.
        </p>
      </div>

      <div className="border-t pt-4">
        <Button variant="outline" onClick={handleRebuild} disabled={rebuilding}>
          {rebuilding ? 'Rebuilding...' : 'Rebuild long-term index'}
        </Button>
      </div>

      <div className="space-y-2 border-t pt-4">
        <Label className="text-base font-semibold">Checkpoint history</Label>
        {checkpoints.length === 0 ? (
          <p className="text-sm text-muted-foreground">No checkpoints yet.</p>
        ) : (
          <ul className="space-y-2">
            {checkpoints.map((checkpoint) => (
              <li key={checkpoint.id} className="flex items-center justify-between gap-2 p-2 border rounded-md text-sm">
                <div className="flex flex-col">
                  <span>Messages {checkpoint.from_message_id}-{checkpoint.through_message_id}</span>
                  <span className="text-xs text-muted-foreground">
                    {checkpoint.committed_at ? formatMessageDate(checkpoint.committed_at) : 'Not committed'}
                  </span>
                  {checkpointBadges(checkpoint)}
                </div>
                <Dialog onOpenChange={(open) => { if (open) handleViewNotes(checkpoint.id); }}>
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm">View rendered notes</Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
                    <DialogHeader>
                      <DialogTitle>Rendered notes</DialogTitle>
                      <DialogDescription>
                        The compaction blocks currently folded into this companion's system prompt.
                      </DialogDescription>
                    </DialogHeader>
                    {loadingNotes && notesCheckpointId === checkpoint.id ? (
                      <p className="text-sm text-muted-foreground">Loading...</p>
                    ) : renderedBlocks && notesCheckpointId === checkpoint.id ? (
                      <div className="space-y-3">
                        {(Object.keys(BLOCK_LABELS) as (keyof RenderedPromptBlocks)[]).map((key) => (
                          <div key={key}>
                            <Label className="text-xs uppercase text-muted-foreground">{BLOCK_LABELS[key]}</Label>
                            <pre className="whitespace-pre-wrap text-xs bg-muted p-2 rounded-md">
                              {renderedBlocks[key] || '(empty)'}
                            </pre>
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </DialogContent>
                </Dialog>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
