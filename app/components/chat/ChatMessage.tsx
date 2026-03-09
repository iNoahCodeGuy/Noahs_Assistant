'use client'

import { Bot, User } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '../hooks/useChat'

interface ChatMessageProps {
  message: Message
}

/**
 * Individual message bubble component
 * Handles user/assistant styling and source display
 * Uses react-markdown for full markdown rendering (headers, images, lists, tables, bold, links)
 */
export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

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
        ) : (
          <div className="chat-markdown leading-relaxed">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h2: ({ children }) => (
                  <h2 className="text-lg font-semibold text-white mt-4 mb-2 first:mt-0">{children}</h2>
                ),
                h3: ({ children }) => (
                  <h3 className="text-base font-semibold text-gray-200 mt-3 mb-1.5">{children}</h3>
                ),
                p: ({ children }) => (
                  <p className="mb-3 last:mb-0 whitespace-pre-wrap">{children}</p>
                ),
                strong: ({ children }) => (
                  <strong className="font-semibold text-white">{children}</strong>
                ),
                em: ({ children }) => (
                  <em className="italic text-gray-300">{children}</em>
                ),
                a: ({ href, children }) => (
                  <a href={href} target="_blank" rel="noopener noreferrer" className="text-chat-primary hover:text-chat-secondary underline transition-colors">
                    {children}
                  </a>
                ),
                ul: ({ children }) => (
                  <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>
                ),
                li: ({ children }) => (
                  <li className="text-gray-300">{children}</li>
                ),
                img: ({ src, alt }) => (
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
                table: ({ children }) => (
                  <div className="overflow-x-auto my-3">
                    <table className="w-full text-sm border-collapse">{children}</table>
                  </div>
                ),
                thead: ({ children }) => (
                  <thead className="border-b border-chat-border">{children}</thead>
                ),
                th: ({ children }) => (
                  <th className="text-left px-3 py-2 font-semibold text-gray-300">{children}</th>
                ),
                td: ({ children }) => (
                  <td className="px-3 py-2 text-gray-400 border-t border-chat-border/50">{children}</td>
                ),
                code: ({ children, className }) => {
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
                pre: ({ children }) => <>{children}</>,
                blockquote: ({ children }) => (
                  <blockquote className="border-l-2 border-chat-primary/50 pl-4 my-3 text-gray-400 italic">{children}</blockquote>
                ),
                hr: () => <hr className="border-chat-border my-4" />,
              }}
            >
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
