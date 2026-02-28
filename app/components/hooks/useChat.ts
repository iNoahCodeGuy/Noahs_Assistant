'use client'

import { useState } from 'react'

export type Role = 
  | 'Hiring Manager (nontechnical)'
  | 'Hiring Manager (technical)'
  | 'Software Developer'
  | 'Just looking around'
  | "Looking to confess I've had a crush on Noah for years"

export interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Array<{
    doc_id: string
    section: string
    similarity: number
  }>
}

/**
 * Custom hook for managing chat state and API interactions
 * Separates business logic from UI components
 */
export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedRole, setSelectedRole] = useState<Role>('Hiring Manager (nontechnical)')
  const [sessionId] = useState(() => crypto.randomUUID())

  const sendMessage = async (content?: string) => {
    const messageContent = content || input
    if (!messageContent.trim() || loading) return

    // Add user message
    const userMessage: Message = { role: 'user', content: messageContent }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: messageContent,
          role: selectedRole,
          session_id: sessionId,
          chat_history: messages.map(m => ({
            role: m.role,
            content: m.content
          }))
        })
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources
      }
      
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      }])
    } finally {
      setLoading(false)
    }
  }

  return {
    messages,
    input,
    setInput,
    loading,
    selectedRole,
    setSelectedRole,
    sendMessage,
    sessionId
  }
}
