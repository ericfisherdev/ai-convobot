import { Avatar, AvatarFallback, AvatarImage } from "../ui/avatar";
import { useParticipants } from "../context/participantsContext";
import { useUserData } from "../context/userContext";
import { useCompanionData } from "../context/companionContext";
import { resolveSpeaker } from "../message/speakerResolution";

interface PendingThoughtBubbleProps {
  speakerId: string;
}

// The one thought still being written: no edit/delete/regenerate controls
// at all -- "not editable mid-write" is structural here, not a disabled
// flag on `ThoughtBubble`.
export function PendingThoughtBubble({ speakerId }: PendingThoughtBubbleProps) {
  const { participants } = useParticipants();
  const userDataContext = useUserData();
  const companionDataContext = useCompanionData();

  const resolved = resolveSpeaker(speakerId, participants, {
    userName: userDataContext?.userData?.name || "User",
    companionName: companionDataContext?.companionData?.name || "Assistant",
    companionAvatarUrl: companionDataContext?.companionData?.avatar_path || "",
  });

  return (
    <div className="message-container animate-in slide-in-from-left-5 duration-300">
      <div className="message-header flex items-center gap-2 mb-2">
        <Avatar className="w-6 h-6">
          <AvatarImage src={resolved.avatarUrl ?? ''} alt={`${resolved.displayName} avatar`} />
          <AvatarFallback className="text-xs">AI</AvatarFallback>
        </Avatar>
        <span className="font-medium text-sm">{resolved.displayName}</span>
        <span className="text-xs text-muted-foreground italic animate-pulse">is thinking...</span>
      </div>
      <div className="message-content flex justify-start">
        <div className="chat-bubble bg-secondary text-secondary-foreground">
          <div className="flex gap-1 py-1">
            <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
            <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce animation-delay-100" />
            <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce animation-delay-200" />
          </div>
        </div>
      </div>
    </div>
  );
}
