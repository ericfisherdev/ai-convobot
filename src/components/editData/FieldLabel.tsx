import { Info } from "lucide-react"

import { Label } from "@/components/ui/label"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

interface FieldLabelProps {
  htmlFor: string;
  children: React.ReactNode;
  tooltip: string;
}

// Shared by `MultiplayerSettings` and `MemorySettings`: a form label with an
// info icon whose tooltip carries the field's explanation.
export function FieldLabel({ htmlFor, children, tooltip }: FieldLabelProps) {
  return (
    <Label htmlFor={htmlFor} className="flex flex-row gap-2">
      <div className="flex items-center gap-2">
        {children}
        <TooltipProvider delayDuration={0}>
          <Tooltip>
            <TooltipTrigger className="cursor-default"> <Info /></TooltipTrigger>
            <TooltipContent>
              <p>{tooltip}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </Label>
  );
}
