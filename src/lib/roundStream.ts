// Pure, React-free parsing and state-reduction for `/api/prompt/stream`'s
// speaker-tagged SSE chunks. `ChatWindow.tsx::promptMessage` is the only
// caller; kept here (not inline) so the reducer is unit-testable without a
// DOM or a running fetch.
import { StreamChunk, StreamEvent } from '../components/interfaces/Message';
import { AttitudeStreamUpdate } from '../components/interfaces/AttitudeData';

/// Splits a raw SSE byte buffer into complete "data: ...\n\n" records plus
/// whatever partial record is still waiting on the next `read()`.
export function splitSseRecords(buffer: string): { records: string[]; rest: string } {
  const parts = buffer.split('\n\n');
  const rest = parts.pop() ?? '';
  return { records: parts, rest };
}

// Parses one SSE record into a `StreamChunk`, defaulting the fields an
// older backend never sent so a mixed-version deploy still renders: no
// `event` is treated as `token`, or as `round_complete` when `is_complete`
// is set; no `speaker_id` falls back to `char`, the only speaker an older
// backend ever streamed. Returns `null` on a malformed record, so one bad
// record does not abandon the whole stream.
export function parseStreamChunk(record: string): StreamChunk | null {
  const line = record.split('\n').find(part => part.startsWith('data: '));
  if (!line) return null;
  try {
    const parsed = JSON.parse(line.slice('data: '.length)) as Partial<StreamChunk>;
    const event: StreamEvent = parsed.event ?? (parsed.is_complete ? 'round_complete' : 'token');
    return {
      request_id: parsed.request_id ?? '',
      event,
      content: parsed.content ?? '',
      is_complete: parsed.is_complete ?? false,
      token_count: parsed.token_count,
      speaker_id: parsed.speaker_id || 'char',
      message_id: parsed.message_id,
      error: parsed.error,
      attitude: parsed.attitude,
    };
  } catch (parseError) {
    console.error('Failed to parse stream chunk:', parseError);
    return null;
  }
}

// One speaker's reply bubble as the reducer sees it: `messageId === null`
// means it is still open (streaming), so `reduceStreamChunk` can tell an
// in-progress bubble apart from a settled one without extra state.
export interface StreamBubble {
  tempId: number;
  speakerId: string;
  content: string;
  messageId: number | null;
}

export interface RoundStreamState {
  bubbles: StreamBubble[];
  error: string | null;
  attitudeStreamed: boolean;
  roundComplete: boolean;
}

export function initialRoundStreamState(): RoundStreamState {
  return { bubbles: [], error: null, attitudeStreamed: false, roundComplete: false };
}

export type StreamEffect =
  | { type: 'open_bubble'; tempId: number; speakerId: string }
  | { type: 'set_content'; tempId: number; content: string }
  | { type: 'settle_bubble'; tempId: number; messageId: number | null; content: string }
  | { type: 'apply_attitude'; update: AttitudeStreamUpdate }
  | { type: 'round_complete' }
  | { type: 'error'; message: string };

// The last bubble not yet settled by a `reply_complete` -- the one a
// `token` appends to, or a `reply_complete` for the same speaker finalises.
// A round runs one speaker at a time (`reply_started`, tokens,
// `reply_complete`, next speaker), so at most one bubble is ever open.
function openBubble(bubbles: StreamBubble[]): StreamBubble | undefined {
  const last = bubbles[bubbles.length - 1];
  return last && last.messageId === null ? last : undefined;
}

export function reduceStreamChunk(
  state: RoundStreamState,
  chunk: StreamChunk,
  nextTempId: () => number,
): { state: RoundStreamState; effects: StreamEffect[] } {
  // A chunk with `attitude` set carries only the attitude update, never
  // content, regardless of its `event`.
  if (chunk.attitude) {
    return {
      state: { ...state, attitudeStreamed: true },
      effects: [{ type: 'apply_attitude', update: chunk.attitude }],
    };
  }

  switch (chunk.event) {
    case 'reply_started': {
      const tempId = nextTempId();
      const bubble: StreamBubble = {
        tempId,
        speakerId: chunk.speaker_id,
        content: '',
        messageId: null,
      };
      return {
        state: { ...state, bubbles: [...state.bubbles, bubble] },
        effects: [{ type: 'open_bubble', tempId, speakerId: chunk.speaker_id }],
      };
    }

    case 'token': {
      let bubbles = state.bubbles;
      const effects: StreamEffect[] = [];
      let current = openBubble(bubbles);
      if (!current) {
        const tempId = nextTempId();
        current = { tempId, speakerId: chunk.speaker_id, content: '', messageId: null };
        bubbles = [...bubbles, current];
        effects.push({ type: 'open_bubble', tempId, speakerId: chunk.speaker_id });
      }
      const tempId = current.tempId;
      const content = current.content + chunk.content;
      bubbles = bubbles.map(b => (b.tempId === tempId ? { ...b, content } : b));
      effects.push({ type: 'set_content', tempId, content });
      return { state: { ...state, bubbles }, effects };
    }

    case 'reply_complete': {
      let bubbles = state.bubbles;
      const effects: StreamEffect[] = [];
      let current = openBubble(bubbles);
      // A skipped speaker still gets its own `reply_started` (the round
      // does not know it will be skipped until the attempt fails), so the
      // notice settles that already-open bubble rather than orphaning it;
      // only a bare `reply_complete` with nothing open yet -- a `system`
      // notice with no `reply_started` at all, or a legacy single-reply
      // stream -- opens a fresh one.
      if (!current) {
        const tempId = nextTempId();
        current = { tempId, speakerId: chunk.speaker_id, content: '', messageId: null };
        bubbles = [...bubbles, current];
        effects.push({ type: 'open_bubble', tempId, speakerId: chunk.speaker_id });
      }
      const tempId = current.tempId;
      const messageId = chunk.message_id ?? null;
      // `speakerId` is relabelled to the chunk's own speaker (relevant only
      // for a skipped speaker's notice, reusing its `reply_started` bubble):
      // the persisted row really is `system`'s, which is what a later
      // `refreshMessages()` will show, so the reducer's own bookkeeping
      // should not disagree with it in the meantime.
      bubbles = bubbles.map(b =>
        b.tempId === tempId
          ? { ...b, speakerId: chunk.speaker_id, content: chunk.content, messageId }
          : b
      );
      effects.push({ type: 'settle_bubble', tempId, messageId, content: chunk.content });
      return { state: { ...state, bubbles }, effects };
    }

    case 'round_complete':
      return {
        state: { ...state, roundComplete: true },
        effects: [{ type: 'round_complete' }],
      };

    case 'error': {
      const message = chunk.error ?? 'Unknown streaming error';
      return {
        state: { ...state, error: message },
        effects: [{ type: 'error', message }],
      };
    }
  }
}
