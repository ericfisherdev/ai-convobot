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

  const { refreshMessages, pushMessage, updateMessage } = useMessages();

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
    const sentMessage = userMessage;
    // Negative ids mark optimistic messages that refreshMessages later replaces
    // with the persisted rows. Each send needs its own id, because
    // updateMessage matches by id and a shared one would let two in-flight
    // streams write into each other's bubble.
    const streamingMessageId = -Date.now();
    try {
      setUserMessage('');
      pushMessage({
        id: -1,
        ai: false,
        content: sentMessage,
        created_at: new Date().toISOString(),
      });
      pushMessage({
        id: streamingMessageId,
        ai: true,
        content: `${companionData.name} is typing...`,
        created_at: new Date().toISOString(),
      });

      const response = await fetch('/api/prompt/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: sentMessage }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Streaming request failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let streamed = '';
      let streamError: string | null = null;

      // Server-Sent Events arrive as "data: {json}\n\n" records, and a single
      // read can contain a partial record, so hold the remainder in a buffer.
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const records = buffer.split('\n\n');
        buffer = records.pop() ?? '';

        for (const record of records) {
          const line = record.split('\n').find(part => part.startsWith('data: '));
          if (!line) continue;
          let chunk;
          try {
            chunk = JSON.parse(line.slice('data: '.length));
          } catch (parseError) {
            // A single malformed record should not abandon the whole stream.
            console.error('Failed to parse stream chunk:', parseError);
            continue;
          }

          if (chunk.is_complete) {
            if (chunk.error) {
              streamError = chunk.error;
            } else if (chunk.content) {
              // The final chunk carries the sanitized reply, which drops the
              // stop markers the raw token stream still contains.
              streamed = chunk.content;
              updateMessage(streamingMessageId, streamed);
            }
            continue;
          }

          streamed += chunk.content;
          updateMessage(streamingMessageId, streamed);
        }
      }

      if (streamError) {
        throw new Error(streamError);
      }

      refreshMessages();

      // Trigger attitude update
      window.dispatchEvent(new CustomEvent('attitude-update'));

    } catch (error) {
      console.error('Error sending message:', error);
      refreshMessages();
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
        <main className={cn(
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
          
          {/* Messages - takes remaining space and allows scrolling */}
          <div className="flex-1 min-h-0">
            <MessageScroll />
          </div>
          
          {/* Input - mobile optimized (moved above attitude summary) */}
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
                        aria-label={isImpersonating ? `Send message as ${companionData.name || "AI Companion"}` : "Send message"}
                      >
                        <SendHorizontal className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>{isImpersonating ? `Send message as ${companionData.name || "AI Companion"}` : "Send message"}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            </div>
          )}
          
          {/* Attitude Summary Bar (moved to bottom as status indicator) */}
          <AttitudeSummaryBar companionId={1} userId={1} />
        </main>
    )
}

export default ChatWindow;