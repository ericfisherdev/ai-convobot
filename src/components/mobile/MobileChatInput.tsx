import { useState, useEffect, useRef } from "react";
import { Button } from "../ui/button";
import { Textarea } from "../ui/textarea";
import { Menu, SendHorizontal, Keyboard, Mic } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";
import { cn } from "../../lib/utils";

interface MobileChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onToggleImpersonate: () => void;
  isImpersonating: boolean;
  placeholder: string;
  disabled?: boolean;
  companionName?: string;
}

export function MobileChatInput({
  value,
  onChange,
  onSend,
  onToggleImpersonate,
  isImpersonating,
  placeholder,
  disabled = false,
  companionName = "Assistant"
}: MobileChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isFocused, setIsFocused] = useState(false);
  const [keyboardHeight, setKeyboardHeight] = useState(0);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      const maxHeight = 120; // Max 5 lines approximately
      textareaRef.current.style.height = Math.min(scrollHeight, maxHeight) + 'px';
    }
  }, [value]);

  // Handle virtual keyboard on mobile
  useEffect(() => {
    const handleResize = () => {
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.clientHeight;
      const keyboardHeight = documentHeight - windowHeight;
      setKeyboardHeight(Math.max(0, keyboardHeight));
    };

    const handleFocus = () => {
      setIsFocused(true);
      // Scroll input into view on mobile
      setTimeout(() => {
        textareaRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 300);
    };

    const handleBlur = () => {
      setIsFocused(false);
      setKeyboardHeight(0);
    };

    const textarea = textareaRef.current;
    if (textarea) {
      textarea.addEventListener('focus', handleFocus);
      textarea.addEventListener('blur', handleBlur);
    }

    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      if (textarea) {
        textarea.removeEventListener('focus', handleFocus);
        textarea.removeEventListener('blur', handleBlur);
      }
    };
  }, []);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (value.trim() && !disabled) {
        onSend();
      }
    }
  };

  const handleTextareaChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(event.target.value);
  };

  // Voice input placeholder for future implementation
  const handleVoiceInput = () => {
    // TODO: Implement voice input functionality
    console.log('Voice input requested');
  };

  return (
    <div 
      className={cn(
        "sticky bottom-0 bg-background border-t z-50 transition-all duration-200",
        "mobile-safe-area",
        isFocused && "shadow-lg"
      )}
      style={{ 
        paddingBottom: keyboardHeight > 0 ? '8px' : undefined,
        transform: keyboardHeight > 0 ? `translateY(-${Math.min(keyboardHeight, 100)}px)` : 'none'
      }}
    >
      <div className="flex items-end gap-2 p-3 max-w-4xl mx-auto">
        {/* Menu Button */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button 
              variant="outline" 
              size="sm" 
              className="touch-target flex-shrink-0"
              disabled={disabled}
            >
              <Menu className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent side="top" align="start">
            <DropdownMenuItem onClick={onToggleImpersonate}>
              {isImpersonating ? 'Stop impersonating' : 'Impersonate'}
            </DropdownMenuItem>
            <DropdownMenuItem onClick={handleVoiceInput}>
              <Mic className="h-4 w-4 mr-2" />
              Voice Input
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Text Input */}
        <div className="flex-1 relative">
          <Textarea
            ref={textareaRef}
            value={value}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            className={cn(
              "min-h-[44px] max-h-[120px] resize-none pr-12 touch-target",
              "text-base", // Prevent zoom on iOS
              "border-2 rounded-2xl",
              isFocused && "border-primary",
              isImpersonating && "bg-primary/5 border-primary"
            )}
            rows={1}
          />
          
          {/* Character count for long messages */}
          {value.length > 200 && (
            <div className="absolute top-2 right-2 text-xs text-muted-foreground bg-background px-1 rounded">
              {value.length}/1000
            </div>
          )}
        </div>

        {/* Send Button */}
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="sm"
                onClick={onSend}
                disabled={disabled || !value.trim()}
                className={cn(
                  "touch-target flex-shrink-0 rounded-full transition-all duration-200",
                  value.trim() && !disabled 
                    ? "bg-primary hover:bg-primary/90 scale-110" 
                    : "scale-100"
                )}
              >
                <SendHorizontal className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="top">
              <p>
                {isImpersonating 
                  ? `Send message as ${companionName}` 
                  : "Send message"
                }
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Virtual keyboard indicator */}
      {isFocused && (
        <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1">
          <div className="bg-primary/20 rounded-full p-1">
            <Keyboard className="h-3 w-3 text-primary" />
          </div>
        </div>
      )}
    </div>
  );
}