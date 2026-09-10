import { useState } from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CompactionDraftCard } from '../message/CompactionDraftCard';
import { initialReviewState, ReviewState } from '../message/compactionReview';
import { CheckpointDetail, CompactionFact, RejectedItem } from '../interfaces/Compaction';

// A stable reference for `ControlledHarness`'s default `serverRejections`.
// A fresh `[]` literal as a default parameter is re-created on every render
// of the harness; since `CompactionDraftCard`'s `useEffect` depends on
// `serverRejections` by reference, that would retrigger the effect (which
// itself updates state through the controlled setter) on every render,
// looping forever.
const NO_REJECTIONS: RejectedItem[] = [];

const aFact = (overrides: Partial<CompactionFact> = {}): CompactionFact => ({
  id: 1,
  category: 'milestone',
  subject: 'user',
  text: 'moved in together',
  quote_speaker: null,
  sources: [3],
  replaces: [],
  relation_to: null,
  relation: null,
  canon: true,
  active: true,
  superseded_by: null,
  rejected_reason: null,
  ...overrides,
});

const currentAttitude = {
  id: 1,
  companion_id: 1,
  target_id: 1,
  target_type: 'user',
  attraction: 0,
  trust: 10,
  fear: 0,
  anger: 0,
  joy: 0,
  sorrow: 0,
  disgust: 0,
  surprise: 0,
  curiosity: 0,
  respect: 0,
  suspicion: 0,
  gratitude: 0,
  jealousy: 0,
  empathy: 0,
  lust: 0,
  love: 0,
  anxiety: 0,
  butterflies: 0,
  submissiveness: 0,
  dominance: 0,
  last_updated: '2024-01-01',
  created_at: '2024-01-01',
};

const aDraft = (facts: CompactionFact[]): CheckpointDetail => ({
  id: 1,
  from_message_id: 1,
  through_message_id: 10,
  status: 'draft',
  trigger: 'threshold',
  committed_at: null,
  needs_merge: false,
  phase: 'review',
  summary_text: 'a summary',
  rolling_summary: null,
  facts,
  attitude: {
    current: currentAttitude,
    rated: null,
    blended: null,
  },
});

const noop = () => {};

describe('CompactionDraftCard', () => {
  it('renders one section per category present in the draft', () => {
    const draft = aDraft([
      aFact({ id: 1, category: 'companion_state' }),
      aFact({ id: 2, category: 'rule' }),
      aFact({ id: 3, category: 'person' }),
    ]);

    render(
      <CompactionDraftCard
        draft={draft}
        messagesSinceDraft={0}
        busy={false}
        onCommit={noop}
        onDiscard={noop}
        onJumpToMessage={noop}
        serverRejections={[]}
      />
    );

    expect(screen.getByText('Companion state')).toBeInTheDocument();
    expect(screen.getByText('Rule')).toBeInTheDocument();
    expect(screen.getByText('Person')).toBeInTheDocument();
    expect(screen.queryByText('Backstory')).not.toBeInTheDocument();
  });

  it('unchecking an item then committing calls onCommit with that item accepted: false', async () => {
    const user = userEvent.setup();
    const draft = aDraft([aFact({ id: 1, category: 'milestone' })]);
    const onCommit = vi.fn();

    render(
      <CompactionDraftCard
        draft={draft}
        messagesSinceDraft={0}
        busy={false}
        onCommit={onCommit}
        onDiscard={noop}
        onJumpToMessage={noop}
        serverRejections={[]}
      />
    );

    await user.click(screen.getByRole('checkbox', { name: 'Accept Milestone item' }));
    await user.click(screen.getByRole('button', { name: 'Commit' }));

    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit.mock.calls[onCommit.mock.calls.length - 1][0].items).toEqual([
      { id: 1, accepted: false },
    ]);
  });

  it('shows a server rejection reason inline for the item it names while another edit keeps its text', () => {
    const rejectedFact = aFact({ id: 1, category: 'rule', text: 'never set foot near the lake' });
    const otherFact = aFact({ id: 2, category: 'milestone', text: 'moved in together' });
    const draft = aDraft([rejectedFact, otherFact]);

    render(
      <CompactionDraftCard
        draft={draft}
        messagesSinceDraft={0}
        busy={false}
        onCommit={noop}
        onDiscard={noop}
        onJumpToMessage={noop}
        serverRejections={[{ item_id: 1, reason: 'quote is not verbatim in any cited message' }]}
      />
    );

    expect(screen.getByText('quote is not verbatim in any cited message')).toBeInTheDocument();
    expect(screen.getByText('moved in together')).toBeInTheDocument();
  });

  it('calls onJumpToMessage with the source id when a #id chip is clicked', async () => {
    const user = userEvent.setup();
    const draft = aDraft([aFact({ id: 1, sources: [42] })]);
    const onJumpToMessage = vi.fn();

    render(
      <CompactionDraftCard
        draft={draft}
        messagesSinceDraft={0}
        busy={false}
        onCommit={noop}
        onDiscard={noop}
        onJumpToMessage={onJumpToMessage}
        serverRejections={[]}
      />
    );

    await user.click(screen.getByText('#42'));
    expect(onJumpToMessage).toHaveBeenCalledWith(42);
  });

  // A real controlled wrapper, mirroring how `PendingDraftMarker` wires
  // `state`/`onStateChange`: `onStateChange` receives an updater function
  // and threads it through `setState`, so two `applyUpdate` calls in the
  // same handler compose instead of one clobbering the other.
  function ControlledHarness({
    draft,
    onCommit = noop,
    serverRejections = NO_REJECTIONS,
  }: {
    draft: CheckpointDetail;
    onCommit?: (review: unknown) => void;
    serverRejections?: RejectedItem[];
  }) {
    const [state, setState] = useState<ReviewState>(() => initialReviewState(draft));
    return (
      <CompactionDraftCard
        draft={draft}
        messagesSinceDraft={0}
        busy={false}
        onCommit={onCommit}
        onDiscard={noop}
        onJumpToMessage={noop}
        serverRejections={serverRejections}
        state={state}
        onStateChange={(update) => setState((prev) => update(prev))}
      />
    );
  }

  it('routes edits through a controlled state/onStateChange pair instead of its own useState', async () => {
    // `PendingDraftMarker` lifts `ReviewState` out of this card so it
    // survives the mobile drawer unmounting; this proves edits flow through
    // the controlled pair rather than an internal state the parent cannot see.
    const user = userEvent.setup();
    const draft = aDraft([aFact({ id: 1, category: 'milestone' })]);
    const onCommit = vi.fn();

    render(<ControlledHarness draft={draft} onCommit={onCommit} />);

    await user.click(screen.getByRole('checkbox', { name: 'Accept Milestone item' }));
    await user.click(screen.getByRole('button', { name: 'Commit' }));

    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit.mock.calls[onCommit.mock.calls.length - 1][0].items).toEqual([
      { id: 1, accepted: false },
    ]);
  });

  it('saving both text and quote for a rule item in one handler keeps both edits (no stale-closure clobber)', async () => {
    // `ItemRow.handleSave` calls `onEditText` then `onEditQuote`
    // synchronously; against a plain `onStateChange(next)` (rather than a
    // functional updater), the second call would overwrite the first
    // because both read the same pre-save `state` closure.
    const user = userEvent.setup();
    const draft = aDraft([
      aFact({ id: 1, category: 'rule', text: 'never go to the lake alone' }),
    ]);

    render(<ControlledHarness draft={draft} />);

    await user.click(screen.getByRole('button', { name: 'Edit item' }));
    const textbox = screen.getAllByRole('textbox')[0];
    const quoteBox = screen.getAllByRole('textbox')[1];
    await user.clear(textbox);
    await user.type(textbox, 'a rewritten rule');
    await user.clear(quoteBox);
    await user.type(quoteBox, 'never set foot near the lake alone');
    await user.click(screen.getByRole('button', { name: 'Save item' }));

    // `toReviewPayload` only ever sends `quote` for a rule item, never
    // `text`, so the payload alone cannot tell the two update calls apart
    // -- check the card's own re-render, which shows both fields.
    expect(screen.getByText('a rewritten rule')).toBeInTheDocument();
    expect(screen.getByText(/never set foot near the lake alone/)).toBeInTheDocument();
  });

  it('renders no Commit/Discard controls in readOnly mode', () => {
    const draft = aDraft([aFact({ id: 1 })]);

    render(
      <CompactionDraftCard
        draft={draft}
        messagesSinceDraft={0}
        busy={false}
        onCommit={noop}
        onDiscard={noop}
        onJumpToMessage={noop}
        serverRejections={[]}
        readOnly
      />
    );

    expect(screen.queryByRole('button', { name: 'Commit' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Discard' })).not.toBeInTheDocument();
  });
});
