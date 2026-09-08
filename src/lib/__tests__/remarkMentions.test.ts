import { describe, it, expect } from 'vitest'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import type { Root, Paragraph, Text, InlineCode } from 'mdast'
import { remarkMentions } from '../remarkMentions'
import { Participant } from '../../components/interfaces/Participant'

const participants: Participant[] = [
  { id: 'bot1', display_name: 'Ada', kind: 'RemoteBot', avatar_url: null, connected: true },
  { id: 'user', display_name: 'Alice', kind: 'Human', avatar_url: null, connected: true },
  { id: 'char', display_name: 'Bob', kind: 'HostBot', avatar_url: null, connected: true },
]

function makeProcessor() {
  return unified().use(remarkParse).use(remarkMentions, participants)
}

async function processTree(markdown: string): Promise<Root> {
  const processor = makeProcessor()
  const tree = processor.parse(markdown)
  return (await processor.run(tree)) as Root
}

function firstParagraph(tree: Root): Paragraph {
  return tree.children[0] as Paragraph
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyNode = any

describe('remarkMentions', () => {
  it('turns @bot1 into a mention span carrying the display name', async () => {
    const tree = await processTree('hi @bot1')
    const paragraph = firstParagraph(tree)
    const mention = paragraph.children.find((node: AnyNode) => node.type === 'mention') as AnyNode

    expect(mention).toBeDefined()
    expect(mention.data.hName).toBe('span')
    expect(mention.data.hProperties.className).toEqual(['mention'])
    expect(mention.data.hProperties['data-participant-id']).toBe('bot1')
    expect((mention.children[0] as Text).value).toBe('@Ada')
  })

  it('resolves @user and @char to the human and companion names', async () => {
    const tree = await processTree('@user and @char')
    const paragraph = firstParagraph(tree)
    const mentions = paragraph.children.filter((node: AnyNode) => node.type === 'mention') as AnyNode[]

    expect(mentions.map((m) => (m.children[0] as Text).value)).toEqual(['@Alice', '@Bob'])
  })

  it('leaves an unknown mention as plain text', async () => {
    const tree = await processTree('hi @bot9, are you there?')
    const paragraph = firstParagraph(tree)

    expect(paragraph.children.some((node: AnyNode) => node.type === 'mention')).toBe(false)
    const text = paragraph.children.find((node: AnyNode) => node.type === 'text') as Text
    expect(text.value).toContain('@bot9')
  })

  it('does not touch a mention inside inline code', async () => {
    const tree = await processTree('`@bot1`')
    const paragraph = firstParagraph(tree)

    expect(paragraph.children.some((node: AnyNode) => node.type === 'mention')).toBe(false)
    const code = paragraph.children.find((node: AnyNode) => node.type === 'inlineCode') as InlineCode
    expect(code.value).toBe('@bot1')
  })

  it('does not match a longer id as a mention', async () => {
    const tree = await processTree('hi @bot1x')
    const paragraph = firstParagraph(tree)

    expect(paragraph.children.some((node: AnyNode) => node.type === 'mention')).toBe(false)
  })

  it('does not mutate the original text node', async () => {
    const processor = makeProcessor()
    const tree = processor.parse('hi @bot1') as Root
    const paragraph = firstParagraph(tree)
    const originalTextNode = paragraph.children[0] as Text

    await processor.run(tree)

    expect(originalTextNode.value).toBe('hi @bot1')
  })
})
