'use client'

import { Bot, User } from 'lucide-react'
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
        <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
        
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
