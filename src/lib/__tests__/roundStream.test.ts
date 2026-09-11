import { describe, it, expect } from 'vitest'
import {
  initialRoundStreamState,
  parseStreamChunk,
  readStreamChunks,
  reduceStreamChunk,
  splitSseRecords,
  RoundStreamState,
  StreamEffect,
} from '../roundStream'
import { StreamChunk } from '../../components/interfaces/Message'

// Encodes chunks the way `/api/prompt/stream` does, then feeds them through
// `parseStreamChunk`/`reduceStreamChunk` exactly as `ChatWindow.tsx` does,
// starting temp ids at -1 and decrementing so assertions read naturally.
function runChunks(rawChunks: unknown[]): { state: RoundStreamState; effects: StreamEffect[] } {
  const sse = rawChunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join('')
  const { records, rest } = splitSseRecords(sse)
  expect(rest).toBe('')

  let state = initialRoundStreamState()
  const effects: StreamEffect[] = []
  let nextId = 0
  const nextTempId = () => --nextId

  for (const record of records) {
    const chunk = parseStreamChunk(record)
    if (!chunk) continue
    const result = reduceStreamChunk(state, chunk, nextTempId)
    state = result.state
    effects.push(...result.effects)
  }

  return { state, effects }
}

describe('roundStream', () => {
  it('produces one bubble per speaker, in order, tagged with speakerId and messageId', () => {
    const { state } = runChunks([
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'token', content: 'Hi', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'reply_complete', content: 'Hi there', is_complete: false, speaker_id: 'char', message_id: 10 },
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'bot1' },
      { request_id: 'r1', event: 'token', content: 'Yo', is_complete: false, speaker_id: 'bot1' },
      { request_id: 'r1', event: 'reply_complete', content: 'Yo!', is_complete: false, speaker_id: 'bot1', message_id: 11 },
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'bot2' },
      { request_id: 'r1', event: 'reply_complete', content: 'sup', is_complete: false, speaker_id: 'bot2', message_id: 12 },
      { request_id: 'r1', event: 'round_complete', content: '', is_complete: true, speaker_id: '' },
    ])

    expect(state.bubbles).toHaveLength(3)
    expect(state.bubbles.map(b => ({ speakerId: b.speakerId, content: b.content, messageId: b.messageId }))).toEqual([
      { speakerId: 'char', content: 'Hi there', messageId: 10 },
      { speakerId: 'bot1', content: 'Yo!', messageId: 11 },
      { speakerId: 'bot2', content: 'sup', messageId: 12 },
    ])
    expect(state.roundComplete).toBe(true)
  })

  it('renders a bare system reply_complete between two speakers as its own bubble', () => {
    const { state } = runChunks([
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'reply_complete', content: 'hi', is_complete: false, speaker_id: 'char', message_id: 1 },
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'bot1' },
      // bot1 never responded: no reply_started for `system`, just a bare
      // reply_complete carrying the persisted notice.
      { request_id: 'r1', event: 'reply_complete', content: 'bot1 did not respond', is_complete: false, speaker_id: 'system', message_id: 2 },
      { request_id: 'r1', event: 'round_complete', content: '', is_complete: true, speaker_id: '' },
    ])

    expect(state.bubbles.map(b => ({ speakerId: b.speakerId, content: b.content, messageId: b.messageId }))).toEqual([
      { speakerId: 'char', content: 'hi', messageId: 1 },
      { speakerId: 'system', content: 'bot1 did not respond', messageId: 2 },
    ])
  })

  it('leaves the partial bubble in place and records the error on a mid-round failure', () => {
    const { state } = runChunks([
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'reply_complete', content: 'hi', is_complete: false, speaker_id: 'char', message_id: 1 },
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'bot1' },
      { request_id: 'r1', event: 'token', content: 'par', is_complete: false, speaker_id: 'bot1' },
      { request_id: 'r1', event: 'error', content: '', is_complete: true, speaker_id: '', error: 'generation failed' },
    ])

    expect(state.error).toBe('generation failed')
    expect(state.roundComplete).toBe(false)
    expect(state.bubbles.map(b => ({ speakerId: b.speakerId, content: b.content, messageId: b.messageId }))).toEqual([
      { speakerId: 'char', content: 'hi', messageId: 1 },
      { speakerId: 'bot1', content: 'par', messageId: null },
    ])
  })

  it('applies the attitude chunk without touching bubble content', () => {
    const { state, effects } = runChunks([
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'reply_complete', content: 'hi', is_complete: false, speaker_id: 'char', message_id: 1 },
      {
        request_id: 'r1',
        event: 'token',
        content: '',
        is_complete: false,
        speaker_id: '',
        attitude: { attitude: { trust: 7 }, summary: 'warmer', deltas: [{ dimension: 'trust', delta: 3 }] },
      },
      { request_id: 'r1', event: 'round_complete', content: '', is_complete: true, speaker_id: '' },
    ])

    expect(state.attitudeStreamed).toBe(true)
    expect(state.bubbles).toEqual([{ tempId: -1, speakerId: 'char', content: 'hi', messageId: 1 }])
    expect(effects.filter(e => e.type === 'apply_attitude')).toHaveLength(1)
  })

  it('applies the compaction-draft-ready chunk once, before round_complete, without touching bubbles', () => {
    const { state, effects } = runChunks([
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'reply_complete', content: 'hi', is_complete: false, speaker_id: 'char', message_id: 1 },
      {
        request_id: 'r1',
        event: 'token',
        content: '',
        is_complete: false,
        speaker_id: '',
        compaction_draft_id: 3,
      },
      { request_id: 'r1', event: 'round_complete', content: '', is_complete: true, speaker_id: '' },
    ])

    expect(state.draftQueuedId).toBe(3)
    expect(state.bubbles).toEqual([{ tempId: -1, speakerId: 'char', content: 'hi', messageId: 1 }])
    const compactionEffects = effects.filter(e => e.type === 'compaction_draft')
    expect(compactionEffects).toEqual([{ type: 'compaction_draft', draftId: 3 }])
    expect(effects.indexOf(compactionEffects[0])).toBeLessThan(
      effects.findIndex(e => e.type === 'round_complete')
    )
  })

  it('yields no compaction_draft effect when the round queued no draft', () => {
    const { state, effects } = runChunks([
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'reply_complete', content: 'hi', is_complete: false, speaker_id: 'char', message_id: 1 },
      { request_id: 'r1', event: 'round_complete', content: '', is_complete: true, speaker_id: '' },
    ])

    expect(state.draftQueuedId).toBeNull()
    expect(effects.some(e => e.type === 'compaction_draft')).toBe(false)
  })

  it('still produces one bubble for a legacy stream with no event field', () => {
    const { state } = runChunks([
      { request_id: 'r1', content: 'Hel', is_complete: false },
      { request_id: 'r1', content: 'lo', is_complete: false },
      { request_id: 'r1', content: 'Hello', is_complete: true },
    ])

    expect(state.bubbles).toHaveLength(1)
    expect(state.bubbles[0].content).toBe('Hello')
    expect(state.roundComplete).toBe(true)
  })

  it('keeps an explicit empty speaker_id on a round-wide chunk instead of defaulting it to char', () => {
    const chunk = parseStreamChunk(
      `data: ${JSON.stringify({ request_id: 'r1', event: 'round_complete', content: '', is_complete: true, speaker_id: '' })}`
    )

    expect(chunk?.speaker_id).toBe('')
  })

  const aThought = {
    id: 1,
    companion_id: 1,
    speaker_id: 'char',
    from_message_id: 1,
    through_message_id: 2,
    text: 'the user seems excited',
    edited: false,
    created_at: '2026-01-01T00:00:00Z',
  }

  it('thought_started sets the pending speaker and opens no bubble', () => {
    const { state, effects } = runChunks([
      { request_id: 'r1', event: 'thought_started', content: '', is_complete: false, speaker_id: 'char' },
    ])

    expect(state.pendingThoughtSpeaker).toBe('char')
    expect(state.bubbles).toEqual([])
    expect(effects).toEqual([{ type: 'thought_started', speakerId: 'char' }])
  })

  it('a thought chunk appends to state.thoughts, clears pending, and opens no bubble', () => {
    const { state, effects } = runChunks([
      { request_id: 'r1', event: 'thought_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'token', content: '', is_complete: false, speaker_id: 'char', thought: aThought },
    ])

    expect(state.thoughts).toEqual([aThought])
    expect(state.pendingThoughtSpeaker).toBeNull()
    expect(state.bubbles).toEqual([])
    const thoughtEffects = effects.filter(e => e.type === 'thought')
    expect(thoughtEffects).toEqual([{ type: 'thought', thought: aThought }])
  })

  it('reply_started after a thought_started with no thought clears pending, emits thought_dropped, and still opens its bubble', () => {
    const { state, effects } = runChunks([
      { request_id: 'r1', event: 'thought_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
    ])

    expect(state.pendingThoughtSpeaker).toBeNull()
    expect(state.bubbles).toHaveLength(1)
    expect(effects).toEqual([
      { type: 'thought_started', speakerId: 'char' },
      { type: 'thought_dropped', speakerId: 'char' },
      { type: 'open_bubble', tempId: -1, speakerId: 'char' },
    ])
  })

  it('a thought chunk emits no thought_dropped effect', () => {
    const { effects } = runChunks([
      { request_id: 'r1', event: 'thought_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'token', content: '', is_complete: false, speaker_id: 'char', thought: aThought },
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
    ])

    expect(effects.some(e => e.type === 'thought_dropped')).toBe(false)
  })

  it('an unrecognised event returns the state unchanged and no effects instead of throwing', () => {
    const state = initialRoundStreamState()
    const chunk = {
      request_id: 'r1',
      // A newer backend's event this build does not know about yet.
      event: 'thought_regenerated' as unknown as StreamChunk['event'],
      content: '',
      is_complete: false,
      speaker_id: 'char',
    }

    const result = reduceStreamChunk(state, chunk, () => -1)

    expect(result).toEqual({ state, effects: [] })
  })
})

describe('readStreamChunks', () => {
  // Encodes each record the way `/api/prompt/stream` does, then splits the
  // whole byte sequence at `splitAt` -- mid-record when it falls inside one
  // -- across two separate `read()` results, the same partial-record
  // scenario `splitSseRecords`'s `rest` handles.
  function streamOf(chunks: unknown[], splitAt: number): ReadableStream<Uint8Array> {
    const encoder = new TextEncoder()
    const bytes = encoder.encode(chunks.map(c => `data: ${JSON.stringify(c)}\n\n`).join(''))
    const first = bytes.slice(0, splitAt)
    const second = bytes.slice(splitAt)
    return new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(first)
        controller.enqueue(second)
        controller.close()
      },
    })
  }

  it('reassembles a record split across two reads', async () => {
    const chunks = [
      { request_id: 'r1', event: 'reply_started', content: '', is_complete: false, speaker_id: 'char' },
      { request_id: 'r1', event: 'reply_complete', content: 'hi', is_complete: false, speaker_id: 'char', message_id: 1 },
    ]
    const encoder = new TextEncoder()
    const wholeLength = encoder.encode(`data: ${JSON.stringify(chunks[0])}\n\n`).length

    // Split partway through the first record's bytes, not on a boundary.
    const body = streamOf(chunks, wholeLength - 5)

    const received: StreamChunk[] = []
    for await (const chunk of readStreamChunks(body)) {
      received.push(chunk)
    }

    expect(received).toHaveLength(2)
    expect(received[0]).toMatchObject({ event: 'reply_started', speaker_id: 'char' })
    expect(received[1]).toMatchObject({ event: 'reply_complete', content: 'hi', message_id: 1 })
  })
})
