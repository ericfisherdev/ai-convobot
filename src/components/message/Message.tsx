import { Avatar, AvatarFallback, AvatarImage } from "../ui/avatar";
import { Pencil, RotateCw, ThumbsUp, Trash2, Smile, Check, CheckCheck } from "lucide-react";
import { useUserData } from "../context/userContext";
import { useCompanionData } from "../context/companionContext";

import companionAvatar from "../../assets/companion_avatar.jpg";
import { CompanionData } from "../interfaces/CompanionData";
import { UserData } from "../interfaces/UserData";
import { useEffect, useState, lazy } from "react";
import { cn, formatMessageDate } from "../../lib/utils";
import { useMessages } from "../context/messageContext";
import { Textarea } from "../ui/textarea";
import { TooltipProvider, Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip";
import { toast } from "sonner";

const Markdown = lazy(() => import('react-markdown'));

interface MessageScrollProps {
  received: boolean;
}

interface MessageScrollProps extends MessageProps {
  regenerate: boolean;
  content: string;
  created_at: string;
}


interface MessageProps {
  id: number;
  regenerate: boolean;
  content: string;
  created_at: string;
}

const UserMessage = ({ id, content, created_at }: MessageProps) => {
  const userDataContext = useUserData();
  const userData: UserData = userDataContext?.userData ?? {} as UserData;

  const { refreshMessages } = useMessages();

  const [editing, setEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(content);
  const [originalContent, setOriginalContent] = useState(content);
  const [isLiked, setIsLiked] = useState(false);
  const [showReactions, setShowReactions] = useState(false);
  const [isDelivered] = useState(true);
  const [isRead] = useState(Math.random() > 0.3); // Simulate read status

  const handleEdit = () => {
    setOriginalContent(content);
    setEditing(true);
  };

  const handleSave = async () => {
    try {
      const response = await fetch(`/api/message/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ai: false, content: editedContent }),
      });

      if (response.ok) {
        setEditing(false);
        refreshMessages();
        toast.success('Message updated successfully');
      } else {
        toast.error('Failed to update message');
        console.error('Failed to update message');
      }
    } catch (error) {
      toast.error(`Error updating message: ${error}`);
      console.error('Error updating message:', error);
    }
  };

  const handleCancel = () => {
    setEditedContent(originalContent);
    setEditing(false);
  };

  const handleDelete = async () => {
    try {
      const response = await fetch(`/api/message/${id}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        refreshMessages();
        toast.success('Message deleted successfully');
      } else {
        toast.error('Failed to delete message');
        console.error('Failed to delete message');
      }
    } catch (error) {
      toast.error(`Error deleting message: ${error}`);
      console.error('Error deleting message:', error);
    }
  };

  return (
    <div className='message-container group animate-in slide-in-from-right-5 duration-300'>
      {/* Unified Message Header */}
      <div className="message-header flex items-center justify-between w-full mb-2">
        <div className="message-info flex items-center gap-2">
          <span className="font-medium text-sm">{userData.name || "User"}</span>
          <span className="text-xs opacity-50">{formatMessageDate(created_at)}</span>
        </div>
        <div className="message-actions flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          {editing ? (
            <>
              <button onClick={handleSave} className="text-xs px-2 py-1 bg-primary text-primary-foreground rounded hover:bg-primary/90 transition-colors">Save</button>
              <button onClick={handleCancel} className="text-xs px-2 py-1 bg-secondary text-secondary-foreground rounded hover:bg-secondary/90 transition-colors">Cancel</button>
            </>
          ) : (
            <>
              <TooltipProvider delayDuration={250}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button 
                      onClick={() => setShowReactions(!showReactions)}
                      className="hover:bg-secondary rounded p-1 transition-colors"
                    >
                      <Smile className="w-4 h-4" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p>Add reaction</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider delayDuration={250}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button 
                      onClick={handleEdit}
                      className="hover:bg-secondary rounded p-1 transition-colors"
                    >
                      <Pencil className="w-4 h-4" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p>Edit message</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider delayDuration={250}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button 
                      onClick={handleDelete}
                      className="hover:bg-secondary rounded p-1 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p>Delete message</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </>
          )}
        </div>
      </div>

      {/* Message Content */}
      <div className="message-content flex justify-end">
        <div className="relative">
          <div className={cn(
            "chat-bubble bg-primary text-primary-foreground transition-all duration-200 hover:shadow-md",
            "relative overflow-visible max-w-[85%]"
          )}>
            {editing ? (
              <Textarea value={editedContent} onChange={(e) => setEditedContent(e.target.value)} />
            ) : (
              <Markdown>{content}</Markdown>
            )}
          </div>
          
          {/* Message status indicators */}
          <div className="flex items-center justify-end mt-1 gap-1 opacity-70">
            {isDelivered && (
              <div className="flex items-center">
                {isRead ? (
                  <CheckCheck className="w-3 h-3 text-blue-500" />
                ) : (
                  <Check className="w-3 h-3" />
                )}
              </div>
            )}
          </div>
          
          {/* Reaction emoji display */}
          {(isLiked || showReactions) && (
            <div className="absolute -bottom-2 -right-2 flex gap-1 animate-in zoom-in-50 duration-200">
              {isLiked && (
                <div className="bg-red-500 text-white rounded-full p-1 text-xs shadow-lg">
                  ❤️
                </div>
              )}
            </div>
          )}
          
          {/* Quick reaction hover menu */}
          <div className={cn(
            "absolute -top-12 right-0 bg-background border rounded-lg shadow-lg p-2 gap-1 transition-all duration-200",
            showReactions ? "flex animate-in fade-in-0 scale-in-95" : "hidden"
          )}>
            <button 
              onClick={() => { setIsLiked(!isLiked); setShowReactions(false); }}
              className="hover:bg-secondary rounded p-1 text-lg transition-colors"
            >
              ❤️
            </button>
            <button 
              onClick={() => setShowReactions(false)}
              className="hover:bg-secondary rounded p-1 text-lg transition-colors"
            >
              👍
            </button>
            <button 
              onClick={() => setShowReactions(false)}
              className="hover:bg-secondary rounded p-1 text-lg transition-colors"
            >
              😂
            </button>
            <button 
              onClick={() => setShowReactions(false)}
              className="hover:bg-secondary rounded p-1 text-lg transition-colors"
            >
              😮
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};


const AiMessage = ({ id, content, created_at, regenerate }: MessageProps) => {
  const companionDataContext = useCompanionData();
  const companionData: CompanionData = companionDataContext?.companionData ?? {} as CompanionData;

  const { refreshMessages } = useMessages();

  const [displayedContent, setDisplayedContent] = useState(content);
  const [editing, setEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(content);
  const [originalContent, setOriginalContent] = useState(content);
  const [isTyping, setIsTyping] = useState(false);
  const [reactions, setReactions] = useState<string[]>([]);
  const [showReactions, setShowReactions] = useState(false);

  useEffect(() => {
    if (content !== displayedContent && content) {
      setIsTyping(true);
      // Simulate typing effect
      const timer = setTimeout(() => {
        setDisplayedContent(content);
        setIsTyping(false);
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [content, displayedContent]);

  const handleEdit = () => {
    setOriginalContent(content);
    setEditing(true);
  };

  const handleSave = async () => {
    try {
      const response = await fetch(`/api/message/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ai: true, content: editedContent }),
      });

      if (response.ok) {
        setEditing(false);
        refreshMessages();
        toast.success('Message updated successfully');
      } else {
        toast.error('Failed to update message');
        console.error('Failed to update message');
      }
    } catch (error) {
      toast.error(`Error updating message: ${error}`);
      console.error('Error updating message:', error);
    }
  };

  const handleCancel = () => {
    setEditedContent(originalContent);
    setEditing(false);
  };

  const handleDelete = async () => {
    try {
      const response = await fetch(`/api/message/${id}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        refreshMessages();
        toast.success('Message deleted successfully');
      } else {
        toast.error('Failed to delete message');
        console.error('Failed to delete message');
      }
    } catch (error) {
      toast.error(`Error deleting message: ${error}`);
      console.error('Error deleting message:', error);
    }
  };

  const handleTuning = async () => {
    try {
      const response = await fetch('/api/memory/dialogueTuning', {
        method: 'POST',
      });

      if (response.ok) {
        refreshMessages();
        toast.success('Successfully added this response as dialogue tuning');
      } else {
        toast.error('Failed to add tuning message');
        console.error('Failed to add tuning message');
      }
    } catch (error) {
      toast.error(`Error adding tuning message: ${error}`);
      console.error('Error adding tuning message:', error);
    }
  };

  const handleRegenerate = async () => {
    const oc = displayedContent;
    try {
      setDisplayedContent("Regenerating a message...");
      const response = await fetch('/api/prompt/regenerate', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        refreshMessages();
        setDisplayedContent(await response.text());
      } else {
        toast.error('Failed to regenerate prompt');
        console.error('Failed to regenerate prompt');
        setDisplayedContent(oc);
      }
    } catch (error) {
      toast.error(`Error regenerating prompt: ${error}`);
      console.error('Error regenerating prompt:', error);
      setDisplayedContent(oc);
    }
  };


  return (
    <div className='message-container group animate-in slide-in-from-left-5 duration-300'>
      {/* Unified Message Header */}
      <div className="message-header flex items-center justify-between w-full mb-2">
        <div className="message-info flex items-center gap-2">
          <Avatar className="w-6 h-6">
            <AvatarImage src={companionData.avatar_path || companionAvatar} alt="Companion Avatar" />
            <AvatarFallback className="text-xs">AI</AvatarFallback>
          </Avatar>
          <span className="font-medium text-sm">{companionData.name || "Assistant"}</span>
          <span className="text-xs opacity-50">{formatMessageDate(created_at)}</span>
          {isTyping && (
            <span className="text-xs text-muted-foreground italic animate-pulse">
              is typing...
            </span>
          )}
        </div>
        <div className="message-actions flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          {editing ? (
            <>
              <button onClick={handleSave} className="text-xs px-2 py-1 bg-primary text-primary-foreground rounded hover:bg-primary/90 transition-colors">Save</button>
              <button onClick={handleCancel} className="text-xs px-2 py-1 bg-secondary text-secondary-foreground rounded hover:bg-secondary/90 transition-colors">Cancel</button>
            </>
          ) : (
            <>
              <TooltipProvider delayDuration={250}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button 
                      onClick={() => setShowReactions(!showReactions)}
                      className="hover:bg-secondary rounded p-1 transition-colors"
                    >
                      <Smile className="w-4 h-4" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p>Add reaction</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider delayDuration={250}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button 
                      onClick={handleEdit}
                      className="hover:bg-secondary rounded p-1 transition-colors"
                    >
                      <Pencil className="w-4 h-4" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p>Edit message</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              {regenerate && 
                <TooltipProvider delayDuration={250}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button 
                        onClick={handleTuning}
                        className="hover:bg-secondary rounded p-1 transition-colors"
                      >
                        <ThumbsUp className="w-4 h-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      <p>Good response</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              }
              <TooltipProvider delayDuration={250}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button 
                      onClick={handleDelete}
                      className="hover:bg-secondary rounded p-1 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p>Delete message</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </>
          )}
        </div>
      </div>

      {/* Message Content */}
      <div className="message-content flex justify-start">
        <div className="relative">
          {regenerate ? (
            <div className="flex flex-row gap-2 items-center">
              <div className={cn(
                "chat-bubble bg-secondary text-secondary-foreground transition-all duration-200 hover:shadow-md",
                "relative overflow-visible max-w-[85%]",
                isTyping && "animate-pulse"
              )}>
                {editing ? (
                  <Textarea value={editedContent} onChange={(e) => setEditedContent(e.target.value)} />
                ) : isTyping ? (
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce animation-delay-100" />
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce animation-delay-200" />
                  </div>
                ) : (
                  <Markdown>{displayedContent}</Markdown>
                )}
              </div>
              {!editing && !isTyping && (
                <TooltipProvider delayDuration={350}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button 
                        onClick={handleRegenerate}
                        className="hover:bg-primary/10 rounded p-1 transition-colors"
                      >
                        <RotateCw className="w-4 h-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="right">
                      <p>Regenerate message</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>
          ) : (
            <div className={cn(
              "chat-bubble bg-secondary text-secondary-foreground transition-all duration-200 hover:shadow-md",
              "relative overflow-visible max-w-[85%]",
              isTyping && "animate-pulse"
            )}>
              {editing ? (
                <Textarea value={editedContent} onChange={(e) => setEditedContent(e.target.value)} />
              ) : isTyping ? (
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{animationDelay: '0.1s'}} />
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{animationDelay: '0.2s'}} />
                </div>
              ) : (
                <Markdown>{displayedContent}</Markdown>
              )}
            </div>
          )}
          
          {/* Reaction display */}
          {reactions.length > 0 && (
            <div className="absolute -bottom-2 -left-2 flex gap-1 animate-in zoom-in-50 duration-200">
              {reactions.map((reaction, index) => (
                <div key={index} className="bg-primary text-primary-foreground rounded-full p-1 text-xs shadow-lg">
                  {reaction}
                </div>
              ))}
            </div>
          )}
          
          {/* Quick reaction hover menu */}
          <div className={cn(
            "absolute -top-12 left-0 bg-background border rounded-lg shadow-lg p-2 gap-1 transition-all duration-200",
            showReactions ? "flex animate-in fade-in-0 scale-in-95" : "hidden"
          )}>
            <button 
              onClick={() => { setReactions([...reactions, '❤️']); setShowReactions(false); }}
              className="hover:bg-secondary rounded p-1 text-lg transition-colors"
            >
              ❤️
            </button>
            <button 
              onClick={() => { setReactions([...reactions, '👍']); setShowReactions(false); }}
              className="hover:bg-secondary rounded p-1 text-lg transition-colors"
            >
              👍
            </button>
            <button 
              onClick={() => { setReactions([...reactions, '😂']); setShowReactions(false); }}
              className="hover:bg-secondary rounded p-1 text-lg transition-colors"
            >
              😂
            </button>
            <button 
              onClick={() => { setReactions([...reactions, '😮']); setShowReactions(false); }}
              className="hover:bg-secondary rounded p-1 text-lg transition-colors"
            >
              😮
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export function Message({ received, regenerate, id, content, created_at }: MessageScrollProps) {
  return (
    <>
      {received ? <AiMessage key={id} content={content} id={id} created_at={created_at} regenerate={regenerate} />: <UserMessage key={id} content={content} id={id} created_at={created_at} regenerate={false} /> }
    </>
  );
}
