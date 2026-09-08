import { lazy } from "react";
import { useParticipants } from "../context/participantsContext";
import { remarkMentions } from "../../lib/remarkMentions";

const Markdown = lazy(() => import('react-markdown'));

interface MessageMarkdownProps {
  content: string;
}

// Wraps the lazy `react-markdown` import shared by every message shell and
// renders `@id` mentions (see `remarkMentions`) against the live participant
// list. Stored content is never rewritten; this only affects what renders.
export function MessageMarkdown({ content }: MessageMarkdownProps) {
  const { participants } = useParticipants();

  return (
    <Markdown remarkPlugins={[[remarkMentions, participants]]}>
      {content}
    </Markdown>
  );
}
