import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
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

const renderMarker = (checkpoints: CheckpointSummary[]) => {
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
        <CompactionMarker messageId={10} />
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
    const fetchMock = renderMarker([aCheckpoint({ status: 'discarded' })]);

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(expect.stringMatching(/^\/api\/compaction($|\?)/))
    );
    expect(screen.queryByText('Continuity notes cover messages up to here')).not.toBeInTheDocument();
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
