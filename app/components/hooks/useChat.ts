'use client'

import { useState, useMemo } from 'react'

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

  // True when the last assistant message contains a form that hasn't been submitted yet
  const formActive = useMemo(() => {
    if (messages.length === 0) return false
    const last = messages[messages.length - 1]
    if (last.role !== 'assistant') return false
    return last.content.includes('Message for Noah:') || last.content.includes('fill this out so we can best assist you')
  }, [messages])

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
        // Try to extract error message from response body
        const errorData = await response.json().catch(() => null)
        const errorAnswer = errorData?.answer
        throw new Error(errorAnswer || 'Failed to get response')
      }

      const data = await response.json()

      // Handle backend error responses (success: false)
      if (data.success === false) {
        throw new Error(data.answer || data.error || 'Backend returned an error')
      }

      const answer = data.answer || data.response
      if (!answer) {
        throw new Error('No answer received from backend')
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: answer,
        sources: data.sources
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMsg = error instanceof Error ? error.message : 'Something went wrong.'
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: errorMsg.length < 200 ? errorMsg : 'Something went wrong. Try again in a moment.'
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
    formActive,
    selectedRole,
    setSelectedRole,
    sendMessage,
    sessionId
  }
}
