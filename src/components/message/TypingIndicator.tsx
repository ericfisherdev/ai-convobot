import { Avatar, AvatarFallback, AvatarImage } from "../ui/avatar";
import { useCompanionData } from "../context/companionContext";
import companionAvatar from "../../assets/companion_avatar.jpg";
import { CompanionData } from "../interfaces/CompanionData";

interface TypingIndicatorProps {
  isVisible: boolean;
}

export function TypingIndicator({ isVisible }: TypingIndicatorProps) {
  const companionDataContext = useCompanionData();
  const companionData: CompanionData = companionDataContext?.companionData ?? {} as CompanionData;

  if (!isVisible) return null;

  return (
    <div className='chat chat-start group animate-in slide-in-from-left-5 duration-300'>
      <div className="chat-image avatar">
        <div className="w-10 rounded-full relative">
          <Avatar>
            <AvatarImage src={companionData.avatar_path || companionAvatar} alt="Companion Avatar" />
            <AvatarFallback>AI</AvatarFallback>
          </Avatar>
          {/* Online indicator */}
          <div className="absolute -bottom-0 -right-0 w-3 h-3 bg-green-500 border-2 border-background rounded-full animate-pulse" />
        </div>
      </div>
      <div className="chat-header">
        {companionData.name || "Assistant"}
        <span className="text-xs ml-2 text-muted-foreground italic animate-pulse">
          is typing...
        </span>
      </div>
      <div className="chat-bubble bg-secondary text-secondary-foreground transition-all duration-200">
        <div className="flex gap-1 py-1">
          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce animation-delay-100" />
          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce animation-delay-200" />
        </div>
      </div>
    </div>
  );
}