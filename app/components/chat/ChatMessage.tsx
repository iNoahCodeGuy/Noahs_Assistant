'use client'

import { Bot, User } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '../hooks/useChat'
import { InlineForm } from './InlineForm'

// Markers matching the backend constants
const CRUSH_FORM_MARKER = 'Message for Noah:'
const RECRUITER_FORM_MARKER = 'fill this out so we can best assist you'

const CRUSH_FIELDS = [
  { key: 'name', label: 'Name' },
  { key: 'contact', label: 'Number or social' },
  { key: 'message', label: 'Message for Noah', multiline: true },
]

const RECRUITER_FIELDS = [
  { key: 'name', label: 'Name', required: true },
  { key: 'phone', label: 'Phone' },
  { key: 'email', label: 'Email', required: true },
  { key: 'company', label: 'Company' },
  { key: 'referral', label: 'How did you find this website?' },
  { key: 'additional', label: 'Additional information', multiline: true },
]

/**
 * Detect if a message contains a form marker and return the form type + intro text.
 */
function detectForm(content: string): { type: 'crush' | 'recruiter'; intro: string } | null {
  if (content.includes(CRUSH_FORM_MARKER)) {
    // Extract text before the form fields
    const idx = content.indexOf('Name:')
    const intro = idx > 0 ? content.slice(0, idx).trim() : content.split('\n')[0]
    return { type: 'crush', intro }
  }
  if (content.includes(RECRUITER_FORM_MARKER)) {
    const idx = content.indexOf('Name:')
    const intro = idx > 0 ? content.slice(0, idx).trim() : content.split('\n')[0]
    return { type: 'recruiter', intro }
  }
  return null
}

interface ChatMessageProps {
  message: Message
  onSendMessage?: (content: string) => void
}

/**
 * Individual message bubble component
 * Handles user/assistant styling, source display, and inline form rendering
 * Uses react-markdown for full markdown rendering (headers, images, lists, tables, bold, links)
 */
export function ChatMessage({ message, onSendMessage }: ChatMessageProps) {
  const isUser = message.role === 'user'
  const form = !isUser ? detectForm(message.content) : null

  return (
    <div className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-chat-primary to-chat-secondary flex items-center justify-center flex-shrink-0">
          <Bot className="w-5 h-5 text-white" />
        </div>
      )}

      <div className={`max-w-2xl rounded-2xl px-6 py-4 ${
        isUser
          ? 'bg-gradient-to-r from-chat-primary to-chat-secondary text-white'
          : 'bg-chat-surface border border-chat-border'
      }`}>
        {isUser ? (
          <p className="leading-relaxed whitespace-pre-wrap">{message.content}</p>
        ) : form && onSendMessage ? (
          // Render intro text + inline form
          <div className="leading-relaxed">
            <div className="chat-markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                {form.intro}
              </ReactMarkdown>
            </div>
            <InlineForm
              fields={form.type === 'crush' ? CRUSH_FIELDS : RECRUITER_FIELDS}
              onSubmit={onSendMessage}
              note={form.type === 'crush' ? 'Want to stay anonymous? Just leave name and contact info blank.' : undefined}
            />
          </div>
        ) : (
          <div className="chat-markdown leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {message.sources && message.sources.length > 0 && (
          <div className="mt-4 pt-4 border-t border-chat-border">
            <p className="text-xs text-gray-400 mb-2">Sources:</p>
            <div className="space-y-1">
              {message.sources.map((source, idx) => (
                <div key={idx} className="text-xs text-gray-500">
                  {source.doc_id} - {source.section.slice(0, 50)}...
                  <span className="text-chat-primary ml-2">
                    ({(source.similarity * 100).toFixed(0)}% match)
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0">
          <User size={20} className="text-gray-300" />
        </div>
      )}
    </div>
  )
}

// Extracted markdown component overrides
const markdownComponents = {
  h2: ({ children }: { children?: React.ReactNode }) => (
    <h2 className="text-lg font-semibold text-white mt-4 mb-2 first:mt-0">{children}</h2>
  ),
  h3: ({ children }: { children?: React.ReactNode }) => (
    <h3 className="text-base font-semibold text-gray-200 mt-3 mb-1.5">{children}</h3>
  ),
  p: ({ children }: { children?: React.ReactNode }) => (
    <p className="mb-3 last:mb-0 whitespace-pre-wrap">{children}</p>
  ),
  strong: ({ children }: { children?: React.ReactNode }) => (
    <strong className="font-semibold text-white">{children}</strong>
  ),
  em: ({ children }: { children?: React.ReactNode }) => (
    <em className="italic text-gray-300">{children}</em>
  ),
  a: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-chat-primary hover:text-chat-secondary underline transition-colors">
      {children}
    </a>
  ),
  ul: ({ children }: { children?: React.ReactNode }) => (
    <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>
  ),
  ol: ({ children }: { children?: React.ReactNode }) => (
    <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>
  ),
  li: ({ children }: { children?: React.ReactNode }) => (
    <li className="text-gray-300">{children}</li>
  ),
  img: ({ src, alt }: { src?: string; alt?: string }) => (
    <figure className="my-4">
      <img
        src={src}
        alt={alt || ''}
        className="rounded-lg w-full max-w-lg mx-auto border border-chat-border"
        loading="lazy"
      />
      {alt && (
        <figcaption className="text-xs text-gray-500 text-center mt-2 italic">{alt}</figcaption>
      )}
    </figure>
  ),
  table: ({ children }: { children?: React.ReactNode }) => (
    <div className="overflow-x-auto my-3">
      <table className="w-full text-sm border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }: { children?: React.ReactNode }) => (
    <thead className="border-b border-chat-border">{children}</thead>
  ),
  th: ({ children }: { children?: React.ReactNode }) => (
    <th className="text-left px-3 py-2 font-semibold text-gray-300">{children}</th>
  ),
  td: ({ children }: { children?: React.ReactNode }) => (
    <td className="px-3 py-2 text-gray-400 border-t border-chat-border/50">{children}</td>
  ),
  code: ({ children, className }: { children?: React.ReactNode; className?: string }) => {
    const isBlock = className?.includes('language-')
    if (isBlock) {
      return (
        <pre className="bg-black/40 rounded-lg p-4 my-3 overflow-x-auto text-sm">
          <code className="text-gray-300">{children}</code>
        </pre>
      )
    }
    return <code className="bg-black/30 px-1.5 py-0.5 rounded text-sm text-gray-300">{children}</code>
  },
  pre: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  blockquote: ({ children }: { children?: React.ReactNode }) => (
    <blockquote className="border-l-2 border-chat-primary/50 pl-4 my-3 text-gray-400 italic">{children}</blockquote>
  ),
  hr: () => <hr className="border-chat-border my-4" />,
}
