// Remark plugin that turns a stored `@id` mention (the normalised form
// `backend/src/participants.rs::normalise_mentions` writes to the database)
// into a `<span class="mention">@Display Name</span>` node for rendering.
// Only `text` nodes are visited, so a mention inside inline code or a fenced
// code block is left untouched. Stored content itself is never rewritten:
// this only changes what `MessageMarkdown` renders, never what is saved or
// sent to the backend.
import type { Root, Text, PhrasingContent } from 'mdast';
import { visit, SKIP, type VisitorResult } from 'unist-util-visit';

import { Participant } from '../components/interfaces/Participant';

// Mirrors `backend/src/participants.rs::ParticipantId`'s grammar: 1-16
// lowercase letters, digits or `_`, starting with a letter. The trailing
// negative lookahead is the frontend's counterpart to `boundary_ok`, so
// `@bot1x` is never mistaken for a match on a registered `bot1`.
const MENTION_SOURCE = '@([a-z][a-z0-9_]{0,15})(?![a-z0-9_])';

interface MentionNode {
  type: 'mention';
  data: {
    hName: 'span';
    hProperties: { className: string[]; 'data-participant-id': string };
  };
  children: Text[];
}

export function remarkMentions(participants: Participant[]) {
  const byId = new Map(participants.map((p) => [p.id, p]));

  return (tree: Root) => {
    visit(tree, 'text', (node: Text, index, parent): VisitorResult => {
      if (index === undefined || index === null || !parent) {
        return;
      }

      // A fresh `RegExp` per node/visit avoids relying on resetting a
      // shared instance's `lastIndex` correctly between nodes.
      const pattern = new RegExp(MENTION_SOURCE, 'g');
      const replacement: PhrasingContent[] = [];
      let lastIndex = 0;
      let matched = false;
      let match: RegExpExecArray | null;

      while ((match = pattern.exec(node.value)) !== null) {
        const participant = byId.get(match[1]);
        if (!participant) {
          continue;
        }
        matched = true;
        if (match.index > lastIndex) {
          replacement.push({ type: 'text', value: node.value.slice(lastIndex, match.index) });
        }
        const mention: MentionNode = {
          type: 'mention',
          data: {
            hName: 'span',
            hProperties: { className: ['mention'], 'data-participant-id': participant.id },
          },
          children: [{ type: 'text', value: `@${participant.display_name}` }],
        };
        replacement.push(mention as unknown as PhrasingContent);
        lastIndex = match.index + match[0].length;
      }

      if (!matched) {
        return;
      }

      if (lastIndex < node.value.length) {
        replacement.push({ type: 'text', value: node.value.slice(lastIndex) });
      }

      parent.children.splice(index, 1, ...replacement);
      // Resume scanning right after the inserted nodes so they are never
      // revisited.
      return [SKIP, index + replacement.length];
    });
  };
}
