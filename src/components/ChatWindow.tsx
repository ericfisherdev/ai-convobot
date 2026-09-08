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
import { useRef, useState } from "react";
import { toast } from "sonner";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui/tooltip";
import { cn } from "../lib/utils";
import { AttitudeSummaryBar } from "./attitude/AttitudeSummaryBar";
import { useAttitude } from "./context/attitudeContext";
import { useSession } from "./context/sessionContext";
import { useParticipants } from "./context/participantsContext";
import { ConnectionStatus } from "./multiplayer/ConnectionStatus";
import { ParticipantsStrip } from "./multiplayer/ParticipantsStrip";
import { JoinerBanner } from "./multiplayer/JoinerBanner";
import {
    initialRoundStreamState,
    parseStreamChunk,
    reduceStreamChunk,
    splitSseRecords,
    StreamEffect,
} from "../lib/roundStream";

const ChatWindow = () => {
  const companionDataContext = useCompanionData();
  const companionData: CompanionData = companionDataContext?.companionData ?? {} as CompanionData;
  const { isMobile, isStandalone } = useMobile();

  const { refreshMessages, pushMessage, updateMessage, settleMessage } = useMessages();
  const { applyAttitudeStreamUpdate } = useAttitude();
  const { session } = useSession();
  const { status, refreshParticipants } = useParticipants();

  const [userMessage, setUserMessage] = useState('');
  const [companionMessage, setCompanionMessage] = useState('');
  const [isImpersonating, setIsImpersonating] = useState(false);
  const [prevUserMessage, setPrevUserMessage] = useState('');
  // Blocks sending while a reply streams: the backend only guards one turn at
  // a time, so a second send before this one settles would corrupt turn order.
  const [isSending, setIsSending] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

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
      if (isSending || !(isImpersonating ? companionMessage : userMessage).trim()) {
        return;
      }
      isImpersonating ? sendMessageAsAi() : promptMessage();
    }
  };

  const promptMessage = async () => {
    if (isSending || !userMessage.trim()) {
      return;
    }
    const sentMessage = userMessage;
    setIsSending(true);
    try {
      setUserMessage('');
      pushMessage({
        id: -1,
        ai: false,
        speaker_id: 'user',
        content: sentMessage,
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
        if (response.status === 409) {
          throw new Error(`${companionData.name} is still replying`);
        }
        throw new Error(`Streaming request failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let streamState = initialRoundStreamState();
      let streamErrorMessage: string | null = null;

      // Negative ids mark optimistic messages that refreshMessages later
      // replaces with the persisted rows. Decremented per bubble, so three
      // speakers in one round get three distinct ids.
      const tempIdBase = -Date.now();
      let tempIdOffset = 0;
      const nextTempId = () => tempIdBase - tempIdOffset++;

      const applyStreamEffects = (effects: StreamEffect[]) => {
        for (const effect of effects) {
          switch (effect.type) {
            case 'open_bubble': {
              const displayName =
                effect.speakerId === 'char' ? companionData.name : effect.speakerId;
              pushMessage({
                id: effect.tempId,
                ai: effect.speakerId !== 'user',
                speaker_id: effect.speakerId,
                content: `${displayName} is typing...`,
                created_at: new Date().toISOString(),
              });
              break;
            }
            case 'set_content':
              updateMessage(effect.tempId, effect.content);
              break;
            case 'settle_bubble':
              settleMessage(effect.tempId, {
                id: effect.messageId ?? effect.tempId,
                content: effect.content,
                speaker_id: effect.speakerId,
              });
              break;
            case 'apply_attitude':
              applyAttitudeStreamUpdate(effect.update);
              break;
            case 'round_complete':
              // A bot may have dropped mid-round; pick that up immediately
              // rather than waiting for the next poll tick.
              refreshParticipants();
              break;
            case 'error':
              streamErrorMessage = effect.message;
              break;
          }
        }
      };

      // Server-Sent Events arrive as "data: {json}\n\n" records, and a single
      // read can contain a partial record, so hold the remainder in a buffer.
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const { records, rest } = splitSseRecords(buffer);
        buffer = rest;

        for (const record of records) {
          const chunk = parseStreamChunk(record);
          if (!chunk) continue;

          const { state: nextState, effects } = reduceStreamChunk(streamState, chunk, nextTempId);
          streamState = nextState;
          applyStreamEffects(effects);
        }
      }

      if (streamErrorMessage) {
        throw new Error(streamErrorMessage);
      }
      if (!streamState.roundComplete) {
        // The stream closed without a terminal chunk -- the client must
        // still re-enable input rather than stay disabled silently.
        throw new Error('round ended without completing');
      }

      refreshMessages();

      // The stream already delivered the new attitude; only fall back to a
      // refetch when it did not.
      if (!streamState.attitudeStreamed) {
        window.dispatchEvent(new CustomEvent('attitude-update'));
      }

    } catch (error) {
      console.error('Error sending message:', error);
      refreshMessages();
      refreshParticipants();
      toast.error(`Error while sending a message: ${error}`);
    } finally {
      setIsSending(false);
      // The re-render that clears `disabled` on the textarea hasn't
      // committed yet at this point in the synchronous finally block, and
      // focus() on a still-disabled control is a no-op — defer past the
      // commit.
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  };

  const sendMessageAsAi = async () => {
    if (isSending) {
      return;
    }
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
                  <ConnectionStatus status={status} />
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

          {/* Participant list - host/joiner mode only */}
          {status.mode !== 'solo' && <ParticipantsStrip />}

          {/* Messages - takes remaining space and allows scrolling */}
          <div className="flex-1 min-h-0">
            <MessageScroll />
          </div>

          {/* Input - mobile optimized (moved above attitude summary) */}
          {status.mode === 'joiner' ? (
            <JoinerBanner status={status} />
          ) : isMobile ? (
            <MobileChatInput
              value={isImpersonating ? companionMessage : userMessage}
              onChange={(value) => isImpersonating ? setCompanionMessage(value) : setUserMessage(value)}
              onSend={() => isImpersonating ? sendMessageAsAi() : promptMessage()}
              onToggleImpersonate={toggleImpersonateMode}
              isImpersonating={isImpersonating}
              placeholder={isImpersonating ? `🥸 Type your message as ${companionData?.name}` : "Type your message"}
              companionName={companionData?.name}
              disabled={isSending}
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
                  ref={inputRef}
                  value={isImpersonating ? companionMessage : userMessage}
                  onChange={handleMessageChange}
                  cols={1}
                  placeholder={isImpersonating ? `🥸 Type your message as ${companionData?.name}` : "Type your message"}
                  onKeyDown={handleKeyDown}
                  disabled={isSending}
                  className="min-h-[44px] max-h-[120px] resize-none"
                />

                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        size="sm"
                        onClick={() => {isImpersonating ? sendMessageAsAi() : promptMessage()}}
                        disabled={isSending || !(isImpersonating ? companionMessage : userMessage).trim()}
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
          {session && session.user_id !== null && (
            <AttitudeSummaryBar companionId={session.companion_id} userId={session.user_id} />
          )}
        </main>
    )
}

export default ChatWindow;
