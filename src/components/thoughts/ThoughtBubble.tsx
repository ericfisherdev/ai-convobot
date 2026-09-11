import { useState } from "react";
import { Pencil, RotateCw, Trash2 } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "../ui/avatar";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Textarea } from "../ui/textarea";
import { TooltipProvider, Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip";
import { useParticipants } from "../context/participantsContext";
import { useUserData } from "../context/userContext";
import { useCompanionData } from "../context/companionContext";
import { useRunningThoughts } from "../context/runningThoughtsContext";
import { resolveSpeaker } from "../message/speakerResolution";
import { MessageMarkdown } from "../message/MessageMarkdown";
import { RunningThought } from "../interfaces/RunningThought";
import { cn, formatMessageDate } from "../../lib/utils";

interface ThoughtBubbleProps {
  thought: RunningThought;
}

// One running thought, `AiMessage`'s shell trimmed to what a thought needs:
// no reactions, no pin, and its own "Regenerate from here" instead of the
// message list's regenerate-last-reply control.
export function ThoughtBubble({ thought }: ThoughtBubbleProps) {
  const { participants } = useParticipants();
  const userDataContext = useUserData();
  const companionDataContext = useCompanionData();
  const { thoughts, editThought, deleteThought, regenerateFrom, regenerating } = useRunningThoughts();

  const [editing, setEditing] = useState(false);
  const [editedText, setEditedText] = useState(thought.text);
  const [confirmingRegenerate, setConfirmingRegenerate] = useState(false);

  const resolved = resolveSpeaker(thought.speaker_id, participants, {
    userName: userDataContext?.userData?.name || "User",
    companionName: companionDataContext?.companionData?.name || "Assistant",
    companionAvatarUrl: companionDataContext?.companionData?.avatar_path || "",
  });

  const handleEdit = () => {
    setEditedText(thought.text);
    setEditing(true);
  };

  const handleSave = async () => {
    const trimmed = editedText.trim();
    if (!trimmed) return;
    const ok = await editThought(thought.id, trimmed);
    if (ok) setEditing(false);
  };

  const handleCancel = () => {
    setEditedText(thought.text);
    setEditing(false);
  };

  const handleDelete = () => {
    deleteThought(thought.id);
  };

  // Later thoughts (regardless of this one) that "Regenerate from here"
  // would also rewrite -- `thoughts` is ascending by id, so everything
  // after this one's index qualifies.
  const laterCount = Math.max(0, thoughts.length - 1 - thoughts.findIndex((t) => t.id === thought.id));

  const handleConfirmRegenerate = () => {
    setConfirmingRegenerate(false);
    regenerateFrom(thought.from_message_id);
  };

  const actionsHidden = regenerating !== null;

  return (
    <div className="message-container group animate-in slide-in-from-left-5 duration-300">
      <div className="message-header flex items-center justify-between w-full mb-2">
        <div className="message-info flex items-center gap-2">
          <Avatar className="w-6 h-6">
            <AvatarImage src={resolved.avatarUrl ?? ''} alt={`${resolved.displayName} avatar`} />
            <AvatarFallback className="text-xs">AI</AvatarFallback>
          </Avatar>
          <span className="font-medium text-sm">{resolved.displayName}</span>
          <span className="text-xs opacity-50">{formatMessageDate(thought.created_at)}</span>
          {thought.edited && <Badge variant="outline">Your wording</Badge>}
        </div>

        {!actionsHidden && (
          <div className="message-actions flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            {editing ? (
              <>
                <button
                  onClick={handleSave}
                  aria-label="Save thought"
                  disabled={!editedText.trim()}
                  className="text-xs px-2 py-1 bg-primary text-primary-foreground rounded hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Save
                </button>
                <button
                  onClick={handleCancel}
                  className="text-xs px-2 py-1 bg-secondary text-secondary-foreground rounded hover:bg-secondary/90 transition-colors"
                >
                  Cancel
                </button>
              </>
            ) : confirmingRegenerate ? null : (
              <>
                <TooltipProvider delayDuration={250}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        onClick={handleEdit}
                        aria-label="Edit thought"
                        className="hover:bg-secondary rounded p-1 transition-colors"
                      >
                        <Pencil className="w-4 h-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      <p>Edit thought</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <TooltipProvider delayDuration={250}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        onClick={() => setConfirmingRegenerate(true)}
                        aria-label="Regenerate from here"
                        className="hover:bg-secondary rounded p-1 transition-colors"
                      >
                        <RotateCw className="w-4 h-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      <p>Regenerate from here</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <TooltipProvider delayDuration={250}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        onClick={handleDelete}
                        aria-label="Delete thought"
                        className="hover:bg-secondary rounded p-1 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      <p>Delete thought</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </>
            )}
          </div>
        )}
      </div>

      {!actionsHidden && confirmingRegenerate && (
        <div className="flex items-center gap-2 mb-2 text-xs text-muted-foreground">
          <span>{`Rewrite this and ${laterCount} later ${laterCount === 1 ? 'thought' : 'thoughts'}?`}</span>
          <Button size="sm" variant="secondary" aria-label="Confirm regenerate" onClick={handleConfirmRegenerate}>
            Confirm
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setConfirmingRegenerate(false)}>
            Cancel
          </Button>
        </div>
      )}

      <div className="message-content flex justify-start">
        <div className={cn("chat-bubble bg-secondary text-secondary-foreground max-w-[85%]")}>
          {editing ? (
            <Textarea value={editedText} onChange={(e) => setEditedText(e.target.value)} />
          ) : (
            <MessageMarkdown content={thought.text} />
          )}
        </div>
      </div>
    </div>
  );
}
