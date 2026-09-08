import { describe, it, expect } from 'vitest'
import {
  initialRoundStreamState,
  parseStreamChunk,
  reduceStreamChunk,
  splitSseRecords,
  RoundStreamState,
  StreamEffect,
} from '../roundStream'

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
})
