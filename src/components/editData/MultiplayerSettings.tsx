import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

import { ConfigInterface, MultiplayerMode } from "../interfaces/Config"
import { ConnectionStatus } from "../multiplayer/ConnectionStatus"
import { useParticipants } from "../context/participantsContext"
import { FieldLabel } from "./FieldLabel"

interface MultiplayerSettingsProps {
  config: ConfigInterface;
  onChange: (next: ConfigInterface) => void;
}

const MODE_DESCRIPTIONS: Record<MultiplayerMode, string> = {
  [MultiplayerMode.Solo]: "This companion runs on its own; no other instance can connect to it.",
  [MultiplayerMode.Host]: "Other instances can join this chat as joiner bots, authenticating with the shared password below.",
  [MultiplayerMode.Joiner]: "This instance connects out to a host instance and joins its chat as a bot.",
};

export function MultiplayerSettings({ config, onChange }: MultiplayerSettingsProps) {
  const mode = config.multiplayer_mode ?? MultiplayerMode.Solo;
  // The live, persisted connection state -- not the unsaved form edits
  // above, which only take effect after a save and restart.
  const { status } = useParticipants();

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <Label htmlFor="multiplayerMode">Multiplayer mode</Label>
        <Select
          value={mode}
          onValueChange={(value) => onChange({ ...config, multiplayer_mode: value as MultiplayerMode })}
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select a multiplayer mode" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={MultiplayerMode.Solo}>Solo</SelectItem>
            <SelectItem value={MultiplayerMode.Host}>Host</SelectItem>
            <SelectItem value={MultiplayerMode.Joiner}>Joiner</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-sm text-muted-foreground">{MODE_DESCRIPTIONS[mode]}</p>
      </div>

      <div data-testid="multiplayer-connection-status">
        {status.mode !== 'solo' && <ConnectionStatus status={status} />}
      </div>

      {mode === MultiplayerMode.Host && (
        <div className="space-y-4">
          <div className="space-y-1">
            <Label htmlFor="multiplayerPassword">Password</Label>
            <Input
              id="multiplayerPassword"
              type="password"
              value={config.multiplayer_password ?? ""}
              placeholder={config.multiplayer_password_set ? "Password is set. Leave blank to keep it" : "Required for host mode"}
              onChange={(e) => onChange({ ...config, multiplayer_password: e.target.value })}
            />
          </div>
          <div className="space-y-1">
            <FieldLabel htmlFor="mentionFollowupDepth" tooltip="How many rounds of @mention follow-ups a companion's reply can trigger before generation stops (used by #132).">
              Mention follow-up depth
            </FieldLabel>
            <Input
              id="mentionFollowupDepth"
              type="number"
              min={0}
              max={10}
              value={config.mention_followup_depth}
              onChange={(e) => onChange({ ...config, mention_followup_depth: parseInt(e.target.value) })}
            />
          </div>
          <div className="space-y-1">
            <FieldLabel htmlFor="remoteGenerationTimeoutSecs" tooltip="How long to wait for a joiner bot's remote reply before giving up (used by #131).">
              Remote generation timeout (seconds)
            </FieldLabel>
            <Input
              id="remoteGenerationTimeoutSecs"
              type="number"
              min={5}
              max={3600}
              value={config.remote_generation_timeout_secs}
              onChange={(e) => onChange({ ...config, remote_generation_timeout_secs: parseInt(e.target.value) })}
            />
          </div>
        </div>
      )}

      {mode === MultiplayerMode.Joiner && (
        <div className="space-y-4">
          <div className="space-y-1">
            <Label htmlFor="multiplayerHostAddress">Host address</Label>
            <Input
              id="multiplayerHostAddress"
              type="text"
              placeholder="192.168.0.20:3000"
              value={config.multiplayer_host_address}
              onChange={(e) => onChange({ ...config, multiplayer_host_address: e.target.value })}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="multiplayerParticipantId">Participant ID</Label>
            <Input
              id="multiplayerParticipantId"
              type="text"
              placeholder="bot1"
              value={config.multiplayer_participant_id}
              onChange={(e) => onChange({ ...config, multiplayer_participant_id: e.target.value })}
            />
            <p className="text-sm text-muted-foreground">
              1-16 characters: a-z, 0-9 and _, starting with a letter.
            </p>
          </div>
          <div className="space-y-1">
            <Label htmlFor="multiplayerJoinerPassword">Password</Label>
            <Input
              id="multiplayerJoinerPassword"
              type="password"
              value={config.multiplayer_password ?? ""}
              placeholder={config.multiplayer_password_set ? "Password is set. Leave blank to keep it" : "The host's shared password"}
              onChange={(e) => onChange({ ...config, multiplayer_password: e.target.value })}
            />
          </div>
          <p className="text-sm text-muted-foreground">
            Joiner mode connects at startup. Restart the app after saving.
          </p>
        </div>
      )}
    </div>
  );
}
