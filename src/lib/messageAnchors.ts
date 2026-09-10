// Scroll targets for individual messages (#180): `MessageScroll.tsx` and
// `VirtualMessageList.tsx` put `id={messageAnchorId(message.id)}` on each
// message's wrapper `div`, so a `draft_ready`/`compaction_draft` stream
// effect can scroll straight to the message a new draft covers.
export const messageAnchorId = (id: number): string => `message-${id}`;

// A no-op when the target message has no node in the DOM -- true for any
// message `VirtualMessageList` has windowed out. Scrolling to a message
// outside the virtual window is a documented limitation, not a bug: the
// next `scrollToMessage` call (or the user's own scroll) picks it up once
// it renders.
export function scrollToMessage(id: number, behavior: ScrollBehavior = 'smooth'): void {
  document.getElementById(messageAnchorId(id))?.scrollIntoView({ block: 'center', behavior });
}
