import { useState, useEffect, useRef } from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Message } from "./Message";
import { TypingIndicator } from "./TypingIndicator";
import { useMessages } from "../context/messageContext";

export function MessageScroll() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const { messages, loadMoreMessages } = useMessages();
  const [hasMoreMessages, setHasMoreMessages] = useState<boolean>(true);
  const [isTyping, setIsTyping] = useState<boolean>(false);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight - scrollRef.current.clientHeight;
    }
  }, [messages]);

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
      className="h-[70vh] md:h-[82vh] w-full rounded-md border backdrop-blur-sm"
    >
      <div className="p-4 h-full">
        {hasMoreMessages && (
          <div className="mb-4 text-center">
            <button
              className="text-sm font-medium text-primary hover:text-primary/80 transition-colors duration-200 px-4 py-2 rounded-full hover:bg-primary/10"
              onClick={handleLoadMore}
            >
              Load previous messages
            </button>
          </div>
        )}
        <div className="flex flex-col gap-4">
          {messages.map((message, index) => (
            <div key={index} className="animate-in fade-in-0 duration-300" style={{ animationDelay: `${index * 50}ms` }}>
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
        </div>
      </div>
    </ScrollArea>
  );
}
