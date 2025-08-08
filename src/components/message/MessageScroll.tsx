import { useState, useEffect, useRef } from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Message } from "./Message";
import { TypingIndicator } from "./TypingIndicator";
import { useMessages } from "../context/messageContext";
import { useMobile } from "../../hooks/useMobile";
import { cn } from "../../lib/utils";

export function MessageScroll() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const { messages, loadMoreMessages } = useMessages();
  const [hasMoreMessages, setHasMoreMessages] = useState<boolean>(true);
  const [isTyping, setIsTyping] = useState<boolean>(false);
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
        (scrollContainer.style as any).webkitOverflowScrolling = 'touch';
        scrollContainer.style.overscrollBehavior = 'contain';
      }
    }
  }, [isMobile]);

  useEffect(() => {
    setHasMoreMessages(messages.length >= 50);
    // Simulate typing indicator when processing messages
    if (messages.length > 0 && messages[messages.length - 1].ai === false) {
      setIsTyping(true);
      const timer = setTimeout(() => setIsTyping(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [messages]);

  const handleLoadMore = () => {
    setIsTyping(true);
    loadMoreMessages();
    setTimeout(() => setIsTyping(false), 1000);
  };
  
  // Expose typing state for parent components
  useEffect(() => {
    (window as any).setMessageTyping = setIsTyping;
    return () => {
      delete (window as any).setMessageTyping;
    };
  }, []);

  return (
    <ScrollArea
      ref={scrollRef}
      className={cn(
        "w-full touch-scroll smooth-scroll",
        isMobile ? "h-full" : "h-[70vh] md:h-[82vh] rounded-md border backdrop-blur-sm"
      )}
    >
      <div className={cn(
        "h-full",
        isMobile ? "p-2 pb-4" : "p-4"
      )}>
        {hasMoreMessages && (
          <div className="mb-4 text-center">
            <button
              className={cn(
                "text-sm font-medium text-primary hover:text-primary/80 transition-colors duration-200 rounded-full hover:bg-primary/10",
                isMobile ? "px-3 py-2 touch-target" : "px-4 py-2"
              )}
              onClick={handleLoadMore}
            >
              Load previous messages
            </button>
          </div>
        )}
        <div className={cn(
          "flex flex-col",
          isMobile ? "gap-3" : "gap-4"
        )}>
          {messages.map((message, index) => (
            <div 
              key={index} 
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
          {/* Bottom padding for mobile to prevent input overlap */}
          {isMobile && <div className="h-4" />}
        </div>
      </div>
    </ScrollArea>
  );
}
