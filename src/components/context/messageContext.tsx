import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';
import { MessageInterface } from '../interfaces/Message';
import { toast } from "sonner";

interface MessagesProviderProps {
  children: ReactNode;
}

interface MessagesContextType {
  messages: MessageInterface[];
  refreshMessages: () => void;
  pushMessage: (message: MessageInterface) => void;
  updateMessage: (id: number, content: string) => void;
  settleMessage: (tempId: number, patch: { id: number; content: string; speaker_id: string }) => void;
  loadMoreMessages: () => Promise<boolean>;
  resetStart : () => void;
  isLoadingMore: boolean;
  hasMoreMessages: boolean;
}

const MessagesContext = createContext<MessagesContextType | undefined>(undefined);

export const useMessages = () => {
  const context = useContext(MessagesContext);
  if (!context) {
    throw new Error('useMessages must be used within a MessagesProvider');
  }
  return context;
};

export const MessagesProvider: React.FC<MessagesProviderProps> = ({ children }) => {
  const [messages, setMessages] = useState<MessageInterface[]>([]);
  const [refreshData, setRefreshData] = useState<boolean>(false);
  const [isLoadingMore, setIsLoadingMore] = useState<boolean>(false);
  const [hasMoreMessages, setHasMoreMessages] = useState<boolean>(true);

  useEffect(() => {
    fetchMessages().then((result) => {
      setMessages(result.messages);
      setHasMoreMessages(result.hasMore);
    });
  }, [refreshData]);

  const fetchMessages = async (startIndex: number = 0, limit: number = 50) => {
    try {
      const response = await fetch(`/api/message?limit=${limit}&start_index=${startIndex}`);
      if (!response.ok) {
        throw new Error('Failed to fetch messages');
      }
      const data = await response.json();

      // Handle both old format (array) and new format (object with pagination)
      if (Array.isArray(data)) {
        return { messages: data, hasMore: data.length === limit };
      } else {
        return { messages: data.messages || [], hasMore: data.has_more || false };
      }
    } catch (error) {
      console.error(error);
      toast.error(`Error while fetching messages: ${error}`);
      return { messages: [], hasMore: false };
    }
  };

  const refreshMessages = () => {
    setRefreshData(!refreshData);
  };

  const pushMessage = (message: MessageInterface) => {
    setMessages(prevMessages => [...prevMessages, message]);
  };

  // Replaces the body of an already-rendered message, used to grow the
  // placeholder reply as streamed tokens arrive.
  const updateMessage = (id: number, content: string) => {
    setMessages(prevMessages =>
      prevMessages.map(message =>
        message.id === id ? { ...message, content } : message
      )
    );
  };

  // Swaps a placeholder's negative temp id for its persisted row id, final
  // content and speaker in one update, so a round's settled bubble never
  // briefly shows the wrong id or speaker to a caller keying off it (e.g. a
  // later effect for the same bubble). `speaker_id` matters for a skipped
  // speaker's notice, which reuses that speaker's `reply_started` bubble but
  // settles as `system`.
  const settleMessage = (tempId: number, patch: { id: number; content: string; speaker_id: string }) => {
    setMessages(prevMessages =>
      prevMessages.map(message =>
        message.id === tempId
          ? { ...message, id: patch.id, content: patch.content, speaker_id: patch.speaker_id }
          : message
      )
    );
  };

  const loadMoreMessages = async (): Promise<boolean> => {
    if (isLoadingMore || !hasMoreMessages) {
      return false;
    }

    setIsLoadingMore(true);
    try {
      const newStartIndex = messages.length;
      const result = await fetchMessages(newStartIndex, 50);

      if (result.messages.length === 0) {
        setHasMoreMessages(false);
        return false;
      }

      // Prepend older messages to maintain chronological order
      setMessages(prevMessages => [...result.messages, ...prevMessages]);

      // Update hasMoreMessages based on API response
      setHasMoreMessages(result.hasMore);

      return true;
    } catch (error) {
      console.error('Error loading more messages:', error);
      return false;
    } finally {
      setIsLoadingMore(false);
    }
  };

  const resetStart = () => {
    setMessages([]);
    setHasMoreMessages(true);
  }

  return (
    <MessagesContext.Provider value={{
      messages,
      refreshMessages,
      pushMessage,
      updateMessage,
      settleMessage,
      loadMoreMessages,
      resetStart,
      isLoadingMore,
      hasMoreMessages
    }}>
      {children}
    </MessagesContext.Provider>
  );
};
