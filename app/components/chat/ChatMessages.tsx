'use client'

import { useEffect, useRef } from 'react'
import { Sparkles, Bot } from 'lucide-react'
import { ChatMessage } from './ChatMessage'
import type { Message } from '../hooks/useChat'

interface ChatMessagesProps {
  messages: Message[]
  loading: boolean
}

/**
 * Messages container with auto-scroll and loading indicator
 */
export function ChatMessages({ messages, loading }: ChatMessagesProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="text-center py-20">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-chat-primary to-chat-secondary mb-6">
              <Sparkles className="w-10 h-10 text-white" />
            </div>
            <h2 className="text-2xl font-bold mb-2 gradient-text">
              Welcome! How can I help you today?
            </h2>
            <p className="text-gray-400">
              Ask me about Noah's background, experience, or technical skills
            </p>
          </div>
        )}
        
        {messages.map((message, index) => (
          <ChatMessage key={index} message={message} />
        ))}
        
        {loading && (
          <div className="flex gap-4">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-chat-primary to-chat-secondary flex items-center justify-center flex-shrink-0">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="bg-chat-surface border border-chat-border rounded-2xl px-6 py-4">
              <div className="flex gap-2">
                <div className="w-2 h-2 rounded-full bg-chat-primary animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 rounded-full bg-chat-primary animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 rounded-full bg-chat-primary animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
    </div>
  )
}
