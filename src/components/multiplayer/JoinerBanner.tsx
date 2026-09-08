import { ConnectionStatus } from "./ConnectionStatus";
import { MultiplayerStatus } from "../interfaces/Participant";

interface JoinerBannerProps {
  status: MultiplayerStatus;
}

// Replaces the desktop `Textarea`/send block and `MobileChatInput` in
// `ChatWindow.tsx` while `status.mode === 'joiner'`: a joiner mirrors the
// host's chat and cannot send its own messages (the backend's 409 guard on
// `/api/prompt*` is the server-side half of this).
export function JoinerBanner({ status }: JoinerBannerProps) {
  return (
    <div className="p-4 border-t bg-background" data-testid="joiner-banner">
      <div className="flex flex-col items-center gap-1 max-w-4xl mx-auto text-center">
        <p className="text-sm font-medium">Mirroring host chat</p>
        <ConnectionStatus status={status} />
      </div>
    </div>
  );
}
