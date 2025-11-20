'use client'

import { Bot, User } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import type { Message } from '../hooks/useChat'

interface ChatMessageProps {
  message: Message
}

/**
 * Individual message bubble component
 * Handles user/assistant styling and source display
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
          <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
        ) : (
          <div className="max-w-none leading-relaxed text-gray-100">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkBreaks]}
              components={{
                code({ node, inline, className, children, ...props }: any) {
                  const match = /language-(\w+)/.exec(className || '')
                  const codeString = String(children).replace(/\n$/, '')

                  return !inline && match ? (
                    <div className="my-4">
                      <SyntaxHighlighter
                        style={oneDark}
                        language={match[1]}
                        PreTag="div"
                        className="rounded-lg"
                        {...props}
                      >
                        {codeString}
                      </SyntaxHighlighter>
                    </div>
                  ) : (
                    <code className="bg-gray-800 px-1.5 py-0.5 rounded text-sm" {...props}>
                      {children}
                    </code>
                  )
                },
                p({ children }) {
                  return <p className="mb-4 last:mb-0">{children}</p>
                },
                ul({ children }) {
                  return <ul className="list-disc list-inside mb-4 space-y-1">{children}</ul>
                },
                ol({ children }) {
                  return <ol className="list-decimal list-inside mb-4 space-y-1">{children}</ol>
                },
                li({ children }) {
                  return <li className="ml-4">{children}</li>
                },
                h1({ children }) {
                  return <h1 className="text-2xl font-bold mb-4 mt-6">{children}</h1>
                },
                h2({ children }) {
                  return <h2 className="text-xl font-bold mb-3 mt-5">{children}</h2>
                },
                h3({ children }) {
                  return <h3 className="text-lg font-semibold mb-2 mt-4">{children}</h3>
                },
                blockquote({ children }) {
                  return <blockquote className="border-l-4 border-gray-600 pl-4 italic my-4">{children}</blockquote>
                },
                a({ href, children }) {
                  return <a href={href} className="text-blue-400 hover:text-blue-300 underline" target="_blank" rel="noopener noreferrer">{children}</a>
                },
                details({ children }) {
                  return <details className="my-4">{children}</details>
                },
                summary({ children }) {
                  return <summary className="cursor-pointer font-semibold mb-2">{children}</summary>
                },
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
                  📚 {source.doc_id} - {source.section.slice(0, 50)}...
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
