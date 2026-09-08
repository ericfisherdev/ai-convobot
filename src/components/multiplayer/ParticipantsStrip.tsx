import { Users } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "../ui/avatar";
import { Badge } from "../ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";
import { useParticipants } from "../context/participantsContext";
import { useMobile } from "../../hooks/useMobile";
import { cn } from "../../lib/utils";
import { Participant } from "../interfaces/Participant";
import companionAvatarDefault from "../../assets/companion_avatar.jpg";

// `ParticipantsProvider` always gives the local `user` entry `avatar_url:
// null` (there is no user-avatar feature), so it must not fall back to the
// companion's picture -- only a bot without its own avatar should. Falling
// through to `undefined` lets Radix's `AvatarFallback` show initials.
function avatarSrc(participant: Participant): string | undefined {
  if (participant.kind === 'Human') {
    return participant.avatar_url ?? undefined;
  }
  return participant.avatar_url || companionAvatarDefault;
}

// One row of every connected participant, rendered by `ChatWindow.tsx` under
// the header whenever `status.mode !== 'solo'`. Each chip is an avatar (with
// a green/grey connection dot) plus name; on mobile the name collapses,
// leaving only the avatar and a tooltip.
export function ParticipantsStrip() {
  const { participants } = useParticipants();
  const { isMobile } = useMobile();

  return (
    <TooltipProvider delayDuration={150}>
      <div
        className="flex items-center gap-2 px-4 py-2 border-b overflow-x-auto"
        data-testid="participants-strip"
      >
        <Users className="w-4 h-4 text-muted-foreground shrink-0" />
        {participants.map((participant) => (
          <Tooltip key={participant.id}>
            {/* `asChild` clones its child and forwards a ref, which
                `Badge` (a plain function component) cannot accept; give it
                a native `<span>` to clone instead. */}
            <TooltipTrigger asChild>
              <span className="inline-flex shrink-0">
                <Badge
                  variant="secondary"
                  className="flex items-center gap-1.5 pl-1"
                  data-testid={`participant-chip-${participant.id}`}
                >
                  <span className="relative inline-flex">
                    <Avatar className="w-5 h-5">
                      <AvatarImage
                        src={avatarSrc(participant)}
                        alt={`${participant.display_name} avatar`}
                      />
                      <AvatarFallback className="text-[9px]">
                        {participant.display_name.slice(0, 2).toUpperCase()}
                      </AvatarFallback>
                    </Avatar>
                    <span
                      className={cn(
                        "absolute -bottom-0.5 -right-0.5 w-2 h-2 rounded-full border border-background",
                        participant.connected ? "bg-green-500" : "bg-muted-foreground"
                      )}
                      data-testid={`participant-status-${participant.id}`}
                      data-connected={participant.connected}
                    />
                  </span>
                  {!isMobile && <span>{participant.display_name}</span>}
                </Badge>
              </span>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <p>{participant.display_name}</p>
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
    </TooltipProvider>
  );
}
