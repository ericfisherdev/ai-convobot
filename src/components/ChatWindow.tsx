import { Avatar, AvatarFallback, AvatarImage } from "./ui/avatar";
import { ModeToggle } from "./mode-toggle";
import { EditDataPopup } from "./editData/EditDataPopup";
import { MessageScroll } from "./message/MessageScroll";
import { MobileChatInput } from "./mobile/MobileChatInput";
import { Textarea } from "./ui/textarea";
import { Menu, SendHorizontal } from "lucide-react";
import { Button } from "./ui/button";
import { useMobile } from "../hooks/useMobile";

import companionAvatar from "../assets/companion_avatar.jpg";

import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
  } from "@/components/ui/dropdown-menu"
import { useCompanionData } from "./context/companionContext";
import { CompanionData } from "./interfaces/CompanionData";
import { useMessages } from "./context/messageContext";
import { useState } from "react";
import { toast } from "sonner";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui/tooltip";
import { cn } from "../lib/utils";
import { AttitudeSummaryBar } from "./attitude/AttitudeSummaryBar";

const ChatWindow = () => {
  const companionDataContext = useCompanionData();
  const companionData: CompanionData = companionDataContext?.companionData ?? {} as CompanionData;
  const { isMobile, isStandalone } = useMobile();

  const { refreshMessages, pushMessage } = useMessages();

  const [userMessage, setUserMessage] = useState('');
  const [companionMessage, setCompanionMessage] = useState('');
  const [isImpersonating, setIsImpersonating] = useState(false);
  const [prevUserMessage, setPrevUserMessage] = useState('');

  const handleMessageChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (isImpersonating) {
      setCompanionMessage(event.target.value);
    } else {
      setUserMessage(event.target.value);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      isImpersonating ? sendMessageAsAi() : promptMessage();
    }
  };

  const promptMessage = async () => {
    try {
      const sendPromise = fetch('/api/prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: userMessage }),
      });
  
      const clearPromise = new Promise<void>(resolve => {
        setUserMessage('');
        resolve();
      });
  
      const pushSentMessagePromise = new Promise<void>(resolve => {
        pushMessage({
          id: -1,
          ai: false,
          content: userMessage,
          created_at: "now",
        });
        pushMessage({
          id: -2,
          ai: true,
          content: `${companionData.name} is typing...`,
          created_at: "",
        })
        resolve();
      });
  
      await Promise.all([sendPromise, clearPromise, pushSentMessagePromise]);
      refreshMessages();
      
      // Trigger attitude update
      window.dispatchEvent(new CustomEvent('attitude-update'));
  
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error(`Error while sending a message: ${error}`);
    }
  };

  const sendMessageAsAi = async () => {
    try {
      const sendPromise = await fetch('/api/message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ai: true, content: companionMessage }),
      });
      
      if (sendPromise.ok) {
        await refreshMessages();
        setUserMessage('');
        setCompanionMessage('');
        setIsImpersonating(false);
        
        // Trigger attitude update
        window.dispatchEvent(new CustomEvent('attitude-update'));
      }

    } catch (error) {
      console.error('Error sending message:', error);
      toast.error(`Error while sending a message: ${error}`);
    }
  };

  const toggleImpersonateMode = () => {
    setIsImpersonating(!isImpersonating);
    if (!isImpersonating) {
      setPrevUserMessage(userMessage);
      setUserMessage('');
    } else {
      setUserMessage(prevUserMessage);
    }
  };

    return (
        <div className={cn(
          "h-full flex flex-col",
          isStandalone && "mobile-safe-area"
        )}>
          {/* Header - responsive layout */}
          <div className={cn(
            "flex items-center justify-between p-4 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60",
            isMobile ? "px-4 py-3" : "px-6 py-4"
          )}>
            <div className='flex items-center gap-3'>
              <Avatar className={isMobile ? "w-8 h-8" : "w-10 h-10"}>
                <AvatarImage src={companionData.avatar_path || companionAvatar} alt="Companion Avatar" />
                <AvatarFallback>AI</AvatarFallback>
              </Avatar>
              {!isMobile && (
                <div className="flex flex-col">
                  <h1 className="font-semibold text-lg">{companionData.name || "AI Companion"}</h1>
                  <p className="text-sm text-muted-foreground">Online</p>
                </div>
              )}
              {isMobile && (
                <h1 className="font-semibold">{companionData.name || "AI Companion"}</h1>
              )}
            </div>
            
            <div className="flex items-center gap-2">
              <EditDataPopup />
              <ModeToggle />
            </div>
          </div>
          
          {/* Messages - takes remaining space */}
          <div className="flex-1 overflow-hidden">
            <MessageScroll />
          </div>
          
          {/* Attitude Summary Bar */}
          <AttitudeSummaryBar companionId={1} userId={1} />
          
          {/* Input - mobile optimized */}
          {isMobile ? (
            <MobileChatInput
              value={isImpersonating ? companionMessage : userMessage}
              onChange={(value) => isImpersonating ? setCompanionMessage(value) : setUserMessage(value)}
              onSend={() => isImpersonating ? sendMessageAsAi() : promptMessage()}
              onToggleImpersonate={toggleImpersonateMode}
              isImpersonating={isImpersonating}
              placeholder={isImpersonating ? `🥸 Type your message as ${companionData?.name}` : "Type your message"}
              companionName={companionData?.name}
            />
          ) : (
            /* Desktop input */
            <div className="p-4 border-t bg-background">
              <div className="flex items-center gap-2 max-w-4xl mx-auto">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm">
                      <Menu className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent side="top">
                    <DropdownMenuItem onClick={toggleImpersonateMode}>
                      {isImpersonating ? 'Stop impersonating' : 'Impersonate'}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
                
                <Textarea 
                  value={isImpersonating ? companionMessage : userMessage} 
                  onChange={handleMessageChange} 
                  cols={1} 
                  placeholder={isImpersonating ? `🥸 Type your message as ${companionData?.name}` : "Type your message"} 
                  onKeyDown={handleKeyDown}
                  className="min-h-[44px] max-h-[120px] resize-none"
                />
                
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button 
                        size="sm" 
                        onClick={() => {isImpersonating ? sendMessageAsAi() : promptMessage()}}
                        disabled={!(isImpersonating ? companionMessage : userMessage).trim()}
                      >
                        <SendHorizontal className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>{isImpersonating ? `Send message as ${companionData?.name}` : "Send message"}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            </div>
          )}
        </div>
    )
}

export default ChatWindow;