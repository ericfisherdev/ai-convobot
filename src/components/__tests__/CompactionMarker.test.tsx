import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
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
  extraction_error: null,
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

  it('renders a failed notice with the extraction error, not the committed-checkpoint notice', async () => {
    renderMarker([
      aCheckpoint({ status: 'failed', extraction_error: 'model load failed' }),
    ]);

    expect(
      await screen.findByText('Continuity notes failed to draft: model load failed')
    ).toBeInTheDocument();
    // Falsifies "the marker just always renders `CompactionNotice`": that
    // component's copy, and its "Show notes" dialog trigger, must be absent
    // for a failed checkpoint.
    expect(screen.queryByText('Continuity notes cover messages up to here')).not.toBeInTheDocument();
    expect(screen.queryByText('Show notes')).not.toBeInTheDocument();
  });

  it('dismissing a failed notice hides it', async () => {
    renderMarker([aCheckpoint({ status: 'failed', extraction_error: 'boom' })]);

    const notice = await screen.findByText('Continuity notes failed to draft: boom');
    fireEvent.click(screen.getByText('Dismiss'));

    await waitFor(() => expect(notice).not.toBeInTheDocument());
  });

  it('retrying a failed notice re-triggers a compaction draft', async () => {
    const fetchMock = renderMarker([aCheckpoint({ status: 'failed', extraction_error: 'boom' })]);
    await screen.findByText('Continuity notes failed to draft: boom');

    fireEvent.click(screen.getByText('Retry'));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/compaction/draft',
        expect.objectContaining({ method: 'POST' })
      )
    );
  });

  it('a committed checkpoint wins over an older failed row at the same message (#208 review)', async () => {
    // Reproduces the exact shadowing bug: `Retry` re-triggers the same
    // uncompacted-tail range when no new messages arrived, so a successful
    // retry can commit at the identical `through_message_id` as the failed
    // attempt it replaces. `checkpoints` is ordered by ascending `id`, so a
    // plain `find` would permanently return the lower-id failed row.
    renderMarker([
      aCheckpoint({ id: 5, status: 'failed', extraction_error: 'boom', through_message_id: 10 }),
      aCheckpoint({ id: 6, status: 'committed', through_message_id: 10 }),
    ]);

    expect(await screen.findByText('Continuity notes cover messages up to here')).toBeInTheDocument();
    expect(screen.queryByText('Continuity notes failed to draft: boom')).not.toBeInTheDocument();
  });

  it('a stale checkpoint wins over a higher-id failed re-compaction at the same message', async () => {
    // A failed *re-compaction* attempt over an existing stale checkpoint's
    // range can land a higher-id `failed` row at the same boundary as that
    // still-valid `stale` one. "Newest id wins" alone would hide the
    // stale checkpoint's real (if outdated) content behind that failure;
    // a committed/stale match must be preferred regardless of id order.
    renderMarker([
      aCheckpoint({ id: 5, status: 'stale', through_message_id: 10 }),
      aCheckpoint({ id: 9, status: 'failed', extraction_error: 'boom', through_message_id: 10 }),
    ]);

    expect(await screen.findByText('Continuity notes cover messages up to here')).toBeInTheDocument();
    expect(screen.queryByText('Continuity notes failed to draft: boom')).not.toBeInTheDocument();
  });

  it('among only-failed matches at the same message, shows the most recent attempts reason', async () => {
    renderMarker([
      aCheckpoint({ id: 5, status: 'failed', extraction_error: 'first failure', through_message_id: 10 }),
      aCheckpoint({ id: 6, status: 'failed', extraction_error: 'second failure', through_message_id: 10 }),
    ]);

    expect(
      await screen.findByText('Continuity notes failed to draft: second failure')
    ).toBeInTheDocument();
    expect(screen.queryByText('Continuity notes failed to draft: first failure')).not.toBeInTheDocument();
  });
});
