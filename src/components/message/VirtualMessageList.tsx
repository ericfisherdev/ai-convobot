import React, { useState, useEffect, useRef, useCallback } from 'react';
import { MessageInterface } from '../interfaces/Message';
import { Message } from './Message';
import { cn } from '../../lib/utils';
import { useMobile } from '../../hooks/useMobile';

interface VirtualMessageListProps {
  messages: MessageInterface[];
  onLoadMore: () => Promise<boolean>;
  isLoadingMore: boolean;
  hasMoreMessages: boolean;
  className?: string;
}

const ITEM_HEIGHT = 120; // Approximate height of a message
const OVERSCAN = 5; // Number of items to render outside of visible area

export function VirtualMessageList({ 
  messages, 
  onLoadMore, 
  isLoadingMore, 
  hasMoreMessages,
  className 
}: VirtualMessageListProps) {
  const { isMobile } = useMobile();
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerHeight, setContainerHeight] = useState(0);
  const [scrollTop, setScrollTop] = useState(0);
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true);

  // Calculate visible range
  const startIndex = Math.max(0, Math.floor(scrollTop / ITEM_HEIGHT) - OVERSCAN);
  const endIndex = Math.min(
    messages.length - 1,
    Math.floor((scrollTop + containerHeight) / ITEM_HEIGHT) + OVERSCAN
  );

  const visibleMessages = messages.slice(startIndex, endIndex + 1);
  const totalHeight = messages.length * ITEM_HEIGHT;
  const offsetY = startIndex * ITEM_HEIGHT;

  // Update container height when component mounts/resizes
  useEffect(() => {
    const updateHeight = () => {
      if (containerRef.current) {
        setContainerHeight(containerRef.current.clientHeight);
      }
    };

    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  // Auto-scroll to bottom for new messages
  useEffect(() => {
    if (isScrolledToBottom && containerRef.current) {
      const container = containerRef.current;
      container.scrollTop = container.scrollHeight - container.clientHeight;
    }
  }, [messages.length, isScrolledToBottom]);

  // Handle scroll events
  const handleScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    const container = event.currentTarget;
    const newScrollTop = container.scrollTop;
    const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
    
    setScrollTop(newScrollTop);
    setIsScrolledToBottom(isNearBottom);

    // Load more messages when scrolling near the top
    if (newScrollTop < 200 && hasMoreMessages && !isLoadingMore) {
      const currentScrollHeight = container.scrollHeight;
      onLoadMore().then((loaded) => {
        if (loaded) {
          // Maintain scroll position after loading more messages
          requestAnimationFrame(() => {
            const newScrollHeight = container.scrollHeight;
            const heightDifference = newScrollHeight - currentScrollHeight;
            container.scrollTop = newScrollTop + heightDifference;
          });
        }
      });
    }
  }, [hasMoreMessages, isLoadingMore, onLoadMore]);

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative overflow-auto",
        isMobile ? "touch-scroll" : "",
        className
      )}
      onScroll={handleScroll}
      style={{ height: '100%' }}
    >
      {/* Loading indicator for more messages */}
      {hasMoreMessages && (
        <div className="flex justify-center py-4">
          {isLoadingMore ? (
            <div className="animate-spin h-5 w-5 border-2 border-primary border-t-transparent rounded-full" />
          ) : (
            <button
              onClick={() => onLoadMore()}
              className={cn(
                "text-sm font-medium text-primary hover:text-primary/80 transition-colors duration-200 rounded-full hover:bg-primary/10",
                isMobile ? "px-3 py-2 touch-target" : "px-4 py-2"
              )}
            >
              Load previous messages
            </button>
          )}
        </div>
      )}
      
      {/* Virtual scrolling container */}
      <div style={{ height: totalHeight, position: 'relative' }}>
        <div
          style={{
            position: 'absolute',
            top: offsetY,
            left: 0,
            right: 0,
          }}
        >
          <div className={cn(
            "flex flex-col",
            isMobile ? "gap-3 px-2" : "gap-4 px-4"
          )}>
            {visibleMessages.map((message, index) => {
              const messageIndex = startIndex + index;
              return (
                <div
                  key={message.id || messageIndex}
                  className="animate-in fade-in-0 duration-300"
                  style={{
                    minHeight: ITEM_HEIGHT,
                    animationDelay: `${Math.min(index * 50, 500)}ms`
                  }}
                >
                  <Message
                    received={message.ai}
                    id={message.id}
                    regenerate={messageIndex === messages.length - 1 && messageIndex !== 0}
                    content={message.content}
                    created_at={message.created_at}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Bottom padding for mobile */}
      {isMobile && <div className="h-4" />}
      
      {/* Scroll to bottom button */}
      {!isScrolledToBottom && (
        <button
          onClick={() => {
            if (containerRef.current) {
              containerRef.current.scrollTop = containerRef.current.scrollHeight;
              setIsScrolledToBottom(true);
            }
          }}
          className="fixed bottom-20 right-4 z-10 bg-primary text-primary-foreground rounded-full p-2 shadow-lg hover:bg-primary/90 transition-colors"
          title="Scroll to bottom"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 14l-7 7m0 0l-7-7m7 7V3"
            />
          </svg>
        </button>
      )}
    </div>
  );
}