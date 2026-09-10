import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { CompactionProvider } from '../context/compactionContext';
import { MessagesProvider } from '../context/messageContext';
import { CompactionMarker } from '../message/CompactionMarker';
import { CheckpointSummary } from '../interfaces/Compaction';

vi.mock('sonner', () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}));

const jsonResponse = (body: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: () => Promise.resolve(body),
  text: () => Promise.resolve(typeof body === 'string' ? body : JSON.stringify(body)),
});

const emptyMessagePage = { messages: [], total_count: 0, has_more: false };

const aCheckpoint = (overrides: Partial<CheckpointSummary> = {}): CheckpointSummary => ({
  id: 1,
  from_message_id: 1,
  through_message_id: 10,
  status: 'committed',
  trigger: 'threshold',
  committed_at: '2024-01-01',
  needs_merge: false,
  ...overrides,
});

const renderMarker = (checkpoints: CheckpointSummary[], messageIds: number[] = [10]) => {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.startsWith('/api/message')) return Promise.resolve(jsonResponse(emptyMessagePage));
    if (url.startsWith('/api/compaction')) {
      return Promise.resolve(jsonResponse({ checkpoints, pending_draft: null }));
    }
    return Promise.resolve(jsonResponse({}));
  });
  vi.stubGlobal('fetch', fetchMock);

  render(
    <MessagesProvider>
      <CompactionProvider>
        {messageIds.map((id) => (
          <CompactionMarker key={id} messageId={id} />
        ))}
      </CompactionProvider>
    </MessagesProvider>
  );

  return fetchMock;
};

describe('CompactionMarker', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('does not render a notice for a discarded checkpoint at this message', async () => {
    // `waitFor`ing only the fetch call is not enough: it resolves as soon
    // as the provider's mount effect issues the request, before `await
    // response.json()` and the resulting `setCheckpoints` have had a
    // chance to render anything -- the negative assertion would then pass
    // whether or not the checkpoint lookup is filtered by status at all.
    // A committed sibling checkpoint in the same listing gives a positive
    // signal that the listing has actually rendered; only once that notice
    // is on screen does the discarded row's absence mean anything.
    renderMarker(
      [
        aCheckpoint({ id: 1, status: 'discarded', through_message_id: 10 }),
        aCheckpoint({ id: 2, status: 'committed', through_message_id: 20 }),
      ],
      [10, 20]
    );

    await screen.findByText('Continuity notes cover messages up to here');
    expect(screen.getAllByText('Continuity notes cover messages up to here')).toHaveLength(1);
  });

  it('renders the notice for a committed checkpoint at this message', async () => {
    renderMarker([aCheckpoint({ status: 'committed' })]);

    expect(await screen.findByText('Continuity notes cover messages up to here')).toBeInTheDocument();
  });

  it('renders the notice for a stale checkpoint at this message', async () => {
    renderMarker([aCheckpoint({ status: 'stale' })]);

    expect(await screen.findByText('Continuity notes cover messages up to here')).toBeInTheDocument();
  });
});
