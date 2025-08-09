import { useState, useEffect, useRef } from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { VirtualMessageList } from "./VirtualMessageList";
import { Message } from "./Message";
import { TypingIndicator } from "./TypingIndicator";
import { useMessages } from "../context/messageContext";
import { useMobile } from "../../hooks/useMobile";
import { cn } from "../../lib/utils";

export function MessageScroll() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const { messages, loadMoreMessages, isLoadingMore, hasMoreMessages } = useMessages();
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const [useVirtualScrolling, setUseVirtualScrolling] = useState<boolean>(false);
  const { isMobile } = useMobile();

  useEffect(() => {
    if (scrollRef.current) {
      const scrollContainer = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [messages]);
  
  // Handle mobile scroll behavior
  useEffect(() => {
    if (isMobile && scrollRef.current) {
      const scrollContainer = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]') as HTMLElement;
      if (scrollContainer) {
        // Better touch scrolling on mobile
        (scrollContainer.style as CSSStyleDeclaration & { webkitOverflowScrolling?: string }).webkitOverflowScrolling = 'touch';
        scrollContainer.style.overscrollBehavior = 'contain';
      }
    }
  }, [isMobile]);

  // Enable virtual scrolling for conversations with more than 100 messages
  useEffect(() => {
    setUseVirtualScrolling(messages.length > 100);
  }, [messages.length]);

  useEffect(() => {
    // Simulate typing indicator when processing messages
    if (messages.length > 0 && messages[messages.length - 1].ai === false) {
      setIsTyping(true);
      const timer = setTimeout(() => setIsTyping(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [messages]);

  const handleLoadMore = async () => {
    setIsTyping(true);
    await loadMoreMessages();
    setTimeout(() => setIsTyping(false), 1000);
  };
  
  // Expose typing state for parent components
  useEffect(() => {
    (window as Window & { setMessageTyping?: (typing: boolean) => void }).setMessageTyping = setIsTyping;
    return () => {
      delete (window as Window & { setMessageTyping?: (typing: boolean) => void }).setMessageTyping;
    };
  }, []);

  // Use virtual scrolling for large conversations, regular scrolling for smaller ones
  if (useVirtualScrolling) {
    return (
      <div className={cn(
        "w-full relative h-full",
        !isMobile && "rounded-md border backdrop-blur-sm"
      )}>
        <VirtualMessageList
          messages={messages}
          onLoadMore={loadMoreMessages}
          isLoadingMore={isLoadingMore}
          hasMoreMessages={hasMoreMessages}
          className="h-full"
        />
        {/* Typing indicator overlay */}
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
          <TypingIndicator isVisible={isTyping} />
        </div>
      </div>
    );
  }

  // Regular scrolling for smaller conversations
  return (
    <ScrollArea
      ref={scrollRef}
      className={cn(
        "w-full h-full touch-scroll smooth-scroll scroll-area",
        !isMobile && "rounded-md border backdrop-blur-sm"
      )}
    >
      <div className={cn(
        "h-full",
        isMobile ? "p-2 pb-4" : "p-4"
      )}>
        {hasMoreMessages && (
          <div className="mb-4 text-center">
            {isLoadingMore ? (
              <div className="animate-spin h-5 w-5 border-2 border-primary border-t-transparent rounded-full mx-auto" />
            ) : (
              <button
                className={cn(
                  "text-sm font-medium text-primary hover:text-primary/80 transition-colors duration-200 rounded-full hover:bg-primary/10",
                  isMobile ? "px-3 py-2 touch-target" : "px-4 py-2"
                )}
                onClick={handleLoadMore}
              >
                Load previous messages
              </button>
            )}
          </div>
        )}
        <div className={cn(
          "flex flex-col",
          isMobile ? "gap-3" : "gap-4"
        )}>
          {messages.map((message, index) => (
            <div 
              key={message.id || index} 
              className="animate-in fade-in-0 duration-300" 
              style={{ animationDelay: `${Math.min(index * 50, 500)}ms` }}
            >
              <Message 
                received={message.ai} 
                id={message.id} 
                regenerate={index === messages.length - 1 && index !== 0} 
                content={message.content} 
                created_at={message.created_at} 
              />
            </div>
          ))}
          <TypingIndicator isVisible={isTyping} />
          {/* Extra bottom padding to prevent message clipping */}
          <div className={cn(
            "flex-shrink-0",
            isMobile ? "h-8" : "h-6"
          )} />
        </div>
      </div>
    </ScrollArea>
  );
}
