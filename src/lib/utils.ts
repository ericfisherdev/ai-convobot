import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatMessageDate(dateString: string): string {
  const date = new Date(dateString);
  
  // Validate date and provide fallback for invalid dates
  if (isNaN(date.getTime())) {
    return "Just now";
  }
  
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const messageDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  
  const diffInDays = Math.floor((today.getTime() - messageDate.getTime()) / (1000 * 60 * 60 * 24));
  
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    }).toLowerCase();
  };
  
  const formatDate = (date: Date) => {
    const day = date.getDate();
    const suffix = day === 1 || day === 21 || day === 31 ? 'st' 
                 : day === 2 || day === 22 ? 'nd'
                 : day === 3 || day === 23 ? 'rd' 
                 : 'th';
    
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    }).replace(/\d+/, `${day}${suffix}`);
  };
  
  if (diffInDays === 0) {
    return `Today @ ${formatTime(date)}`;
  } else if (diffInDays === 1) {
    return `Yesterday @ ${formatTime(date)}`;
  } else if (diffInDays < 7) {
    const dayName = date.toLocaleDateString('en-US', { weekday: 'long' });
    return `${dayName} @ ${formatTime(date)}`;
  } else if (date.getFullYear() === now.getFullYear()) {
    return `${formatDate(date)} @ ${formatTime(date)}`;
  } else {
    return `${formatDate(date)}, ${date.getFullYear()} @ ${formatTime(date)}`;
  }
}
