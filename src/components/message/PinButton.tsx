import { Pin, PinOff } from "lucide-react";
import { TooltipProvider, Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip";
import { useCompaction } from "../context/compactionContext";

interface PinButtonProps {
  messageId: number;
  pinned: boolean;
}

// Marks a message as exempt from compaction (#179's `pin`/`unpin`, which
// already refreshes messages and toasts on failure) so its exact wording
// survives into the compaction prompt verbatim.
export function PinButton({ messageId, pinned }: PinButtonProps) {
  const { pin, unpin } = useCompaction();

  return (
    <TooltipProvider delayDuration={250}>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            onClick={() => (pinned ? unpin(messageId) : pin(messageId))}
            aria-label={pinned ? "Unpin message" : "Pin message"}
            className="hover:bg-secondary rounded p-1 transition-colors"
          >
            {pinned ? <PinOff className="w-4 h-4" /> : <Pin className="w-4 h-4" />}
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          <p>{pinned ? "Unpin message" : "Pin message"}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
