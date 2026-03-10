'use client'

import { useEffect, useRef } from 'react'
import { Bot } from 'lucide-react'
import { ChatMessage } from './ChatMessage'
import type { Message } from '../hooks/useChat'

interface ChatMessagesProps {
  messages: Message[]
  loading: boolean
  onSendMessage?: (content: string) => void
}

/**
 * Messages container with auto-scroll and loading indicator
 */
export function ChatMessages({ messages, loading, onSendMessage }: ChatMessagesProps) {
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
            <h2 className="text-2xl font-bold mb-2 gradient-text">
              Portfolia
            </h2>
            <p className="text-gray-400 max-w-lg mx-auto mb-8">
              Noah&apos;s AI-powered portfolio assistant. I know about his projects, career, technical stack, and there&apos;s an MMA fighter story. Pick a lane or ask whatever you want.
            </p>
            {onSendMessage && (
              <div className="grid grid-cols-2 gap-3 max-w-lg mx-auto">
                {[
                  'Learn more about Noah',
                  'See what Noah has built',
                  'Just looking around',
                  'Confess a crush',
                ].map((option) => (
                  <button
                    key={option}
                    onClick={() => onSendMessage(option)}
                    className="text-left px-4 py-3 rounded-lg bg-chat-surface border border-chat-border text-gray-300 hover:border-chat-primary hover:text-white transition-colors text-sm"
                  >
                    {option}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        
        {messages.map((message, index) => (
          <ChatMessage key={index} message={message} onSendMessage={onSendMessage} />
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
