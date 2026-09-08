import { Wifi, WifiOff, Loader2 } from "lucide-react";
import { cn } from "../../lib/utils";
import { MultiplayerStatus, MultiplayerConnectionState } from "../interfaces/Participant";

interface ConnectionStatusProps {
  status: MultiplayerStatus;
  className?: string;
}

const STATE_LABELS: Record<MultiplayerConnectionState, string> = {
  disconnected: 'Disconnected',
  connecting: 'Connecting',
  connected: 'Connected',
  rejected: 'Rejected',
};

// Small presentational mapping of `MultiplayerStatus` to text/colour, used
// both in `ChatWindow`'s header (host/joiner) and `JoinerBanner`, and in the
// Multiplayer tab of `EditData.tsx`. Takes `status` as a prop rather than
// reading `useParticipants()` itself so it stays usable wherever a caller
// already has a status value (and testable without a provider).
export function ConnectionStatus({ status, className }: ConnectionStatusProps) {
  if (status.mode === 'solo') {
    return <span className={cn("text-sm text-muted-foreground", className)}>Online</span>;
  }

  if (status.mode === 'host') {
    return (
      <span className={cn("flex items-center gap-1 text-sm text-muted-foreground", className)}>
        <Wifi className="w-3.5 h-3.5 text-green-500" />
        Hosting
      </span>
    );
  }

  // Joiner mode: `status.state` is always populated by the backend, but
  // fall back to `disconnected` defensively.
  const state = status.state ?? 'disconnected';
  const Icon = state === 'connected' ? Wifi : state === 'connecting' ? Loader2 : WifiOff;
  const colorClass =
    state === 'connected'
      ? 'text-green-500'
      : state === 'connecting'
        ? 'text-yellow-500 animate-spin'
        : 'text-destructive';

  let label = STATE_LABELS[state];
  if (state === 'connecting' && status.attempts) {
    label = `${label} (attempt ${status.attempts})`;
  }
  if (state === 'rejected' && status.reason) {
    label = `${label}: ${status.reason}`;
  }
  if (state === 'disconnected' && status.last_error) {
    label = `${label}: ${status.last_error}`;
  }

  return (
    <span
      className={cn("flex items-center gap-1 text-sm", className)}
      data-testid="connection-status"
      data-state={state}
    >
      <Icon className={cn("w-3.5 h-3.5", colorClass)} />
      {label}
    </span>
  );
}
