import { lazy, Suspense } from "react";
import { useParticipants } from "../context/participantsContext";
import { remarkMentions } from "../../lib/remarkMentions";

const Markdown = lazy(() => import('react-markdown'));

interface MessageMarkdownProps {
  content: string;
}

// Wraps the lazy `react-markdown` import shared by every message shell and
// renders `@id` mentions (see `remarkMentions`) against the live participant
// list. Stored content is never rewritten; this only affects what renders.
// `main.tsx` mounts `App` with no `Suspense` ancestor, so the first render
// of any message could otherwise suspend indefinitely while the chunk
// loads; `fallback={null}` keeps that invisible rather than blank/erroring.
export function MessageMarkdown({ content }: MessageMarkdownProps) {
  const { participants } = useParticipants();

  return (
    <Suspense fallback={null}>
      <Markdown remarkPlugins={[[remarkMentions, participants]]}>
        {content}
      </Markdown>
    </Suspense>
  );
}
