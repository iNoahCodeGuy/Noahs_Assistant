'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, ChevronDown, Sparkles } from 'lucide-react'

type Message = {
  role: 'user' | 'assistant'
  content: string
  sources?: Array<{
    doc_id: string
    section: string
    similarity: number
  }>
}

type Role = 
  | 'Hiring Manager (nontechnical)'
  | 'Hiring Manager (technical)'
  | 'Software Developer'
  | 'Just looking around'
  | "Looking to confess I've had a crush on Noah for years"

const ROLES: Role[] = [
  'Hiring Manager (nontechnical)',
  'Hiring Manager (technical)',
  'Software Developer',
  'Just looking around',
  "Looking to confess I've had a crush on Noah for years"
]

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedRole, setSelectedRole] = useState<Role>('Hiring Manager (nontechnical)')
  const [sessionId] = useState(() => crypto.randomUUID())
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage: Message = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: input,
          role: selectedRole,
          session_id: sessionId,
          chat_history: messages
        })
      })

      if (!response.ok) throw new Error('Failed to get response')

      const data = await response.json()
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources
      }
      
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage: Message = {
        role: 'assistant',
        content: "I apologize, but I'm having trouble connecting right now. Please try again in a moment."
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-screen gradient-bg">
      {/* Header */}
      <header className="border-b border-chat-border bg-chat-surface/80 backdrop-blur-lg">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-chat-primary to-chat-secondary flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">Noah's AI Assistant</h1>
              <p className="text-xs text-gray-400">Interactive Resume & Career Assistant</p>
            </div>
          </div>
          
          {/* Role Selector */}
          <div className="relative">
            <select
              value={selectedRole}
              onChange={(e) => setSelectedRole(e.target.value as Role)}
              className="appearance-none bg-chat-surface border border-chat-border rounded-lg px-4 py-2 pr-10 text-sm focus:outline-none focus:border-chat-primary transition-colors cursor-pointer"
            >
              {ROLES.map(role => (
                <option key={role} value={role}>{role}</option>
              ))}
            </select>
            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
          </div>
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-8 space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-chat-primary to-chat-secondary flex items-center justify-center">
                <Bot className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-2xl font-bold mb-2 gradient-text">
                Welcome! How can I help you today?
              </h2>
              <p className="text-gray-400">
                Ask me about Noah's background, experience, or technical skills
              </p>
            </div>
          )}

          {messages.map((message, idx) => (
            <div key={idx} className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {message.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-chat-primary to-chat-secondary flex items-center justify-center flex-shrink-0">
                  <Bot className="w-5 h-5 text-white" />
                </div>
              )}
              
              <div className={`max-w-2xl rounded-2xl px-6 py-4 ${
                message.role === 'user' 
                  ? 'bg-gradient-to-r from-chat-primary to-chat-secondary text-white'
                  : 'bg-chat-surface border border-chat-border'
              }`}>
                <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-chat-border">
                    <p className="text-xs text-gray-400 mb-2">Sources:</p>
                    <div className="space-y-1">
                      {message.sources.map((source, i) => (
                        <div key={i} className="text-xs text-gray-500">
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

              {message.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0">
                  <User className="w-5 h-5 text-gray-300" />
                </div>
              )}
            </div>
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

      {/* Input */}
      <div className="border-t border-chat-border bg-chat-surface/80 backdrop-blur-lg">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <form onSubmit={sendMessage} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me anything about Noah..."
              className="flex-1 bg-chat-bg border border-chat-border rounded-xl px-6 py-3 focus:outline-none focus:border-chat-primary transition-colors"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className="bg-gradient-to-r from-chat-primary to-chat-secondary rounded-xl px-6 py-3 font-medium hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Send className="w-5 h-5" />
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
