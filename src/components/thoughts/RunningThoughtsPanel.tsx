import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronUp, NotebookPen } from "lucide-react";
import { ScrollArea } from "../ui/scroll-area";
import { useMobile } from "../../hooks/useMobile";
import { useConfigData } from "../context/configContext";
import { useRunningThoughts } from "../context/runningThoughtsContext";
import { ThoughtBubble } from "./ThoughtBubble";
import { PendingThoughtBubble } from "./PendingThoughtBubble";
import { cn } from "../../lib/utils";

interface RunningThoughtsPanelProps {
  className?: string;
}

// The chat-style panel the user curates running thoughts in (#214/#218):
// beside the chat on wide viewports, below the attitude bar on narrow ones
// (`ChatWindow.tsx` controls that placement; this component only renders
// its own contents).
export function RunningThoughtsPanel({ className }: RunningThoughtsPanelProps) {
  const { isMobile } = useMobile();
  const configContext = useConfigData();
  const { thoughts, pendingSpeakerId, regenerating } = useRunningThoughts();

  const [collapsed, setCollapsed] = useState(isMobile);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      const scrollContainer = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [thoughts, pendingSpeakerId]);

  // A chat with the feature off looks exactly as before (#214): render
  // nothing once the config has loaded and confirms it is disabled. Before
  // the config loads, `running_thoughts_enabled` is simply absent from
  // `null`, so the panel stays hidden until it is known to be on.
  if (!configContext?.config?.running_thoughts_enabled) {
    return null;
  }

  return (
    <div
      className={cn("flex flex-col min-h-0", className)}
      data-testid="running-thoughts-panel"
    >
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <NotebookPen className="w-4 h-4" />
          <span className="font-medium text-sm">Running thoughts</span>
        </div>
        <button
          onClick={() => setCollapsed((c) => !c)}
          aria-expanded={!collapsed}
          aria-label={collapsed ? "Expand running thoughts" : "Collapse running thoughts"}
          className="hover:bg-secondary rounded p-1 transition-colors"
        >
          {collapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
        </button>
      </div>

      {regenerating && (
        <div role="status" className="px-4 py-2 text-xs text-muted-foreground border-b">
          Rewriting thoughts… {regenerating.rewritten} done
        </div>
      )}

      {!collapsed && (
        <ScrollArea ref={scrollRef} className="flex-1 min-h-0">
          <div className="flex flex-col gap-4 p-4">
            {thoughts.length === 0 && !pendingSpeakerId && (
              <p className="text-sm text-muted-foreground">No thoughts yet</p>
            )}
            {thoughts.map((thought) => (
              <ThoughtBubble key={thought.id} thought={thought} />
            ))}
            {pendingSpeakerId && <PendingThoughtBubble speakerId={pendingSpeakerId} />}
          </div>
        </ScrollArea>
      )}
    </div>
  );
}
