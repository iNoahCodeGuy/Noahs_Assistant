'use client'

import { useState, useEffect } from 'react'

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
 *
 * Design: Portfolia messages first
 * - On mount, fetch initial greeting with empty role
 * - Backend detects no role and returns conversational greeting
 * - User's first response triggers role inference
 */
export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedRole, setSelectedRole] = useState<Role | ''>('')  // Start with empty role
  const [sessionId] = useState(() => crypto.randomUUID())
  const [sessionMemory, setSessionMemory] = useState<Record<string, any>>({})  // Persist session state
  const [greetingSent, setGreetingSent] = useState(false)

  // Fetch initial greeting on mount (Portfolia messages first)
  useEffect(() => {
    if (!greetingSent && messages.length === 0) {
      fetchInitialGreeting()
    }
  }, [greetingSent, messages.length])

  const fetchInitialGreeting = async () => {
    setGreetingSent(true)
    setLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: '',  // Empty query triggers initial greeting
          role: '',   // Empty role triggers greeting flow
          session_id: sessionId,
          chat_history: [],
          session_memory: sessionMemory  // Send session state
        })
      })

      if (!response.ok) {
        throw new Error('Failed to get initial greeting')
      }

      const data = await response.json()

      // Update session memory from backend
      if (data.session_memory) {
        setSessionMemory(data.session_memory)
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer
      }

      setMessages([assistantMessage])
    } catch (error) {
      console.error('Error fetching initial greeting:', error)
      // Fallback greeting if API fails
      setMessages([{
        role: 'assistant',
        content: `👋 Hey! I'm Portfolia — Noah's AI Assistant, and I'm genuinely excited you're here!\n\nI'm a full-stack generative AI application built to help people understand how production AI systems actually work. Think of me as both a working demo and a teaching tool — every conversation shows you real RAG architecture, vector search, LLM orchestration, and enterprise-grade patterns in action.\n\nI can walk you through the engineering side (architecture, code, data pipelines), the business value (ROI, team efficiency, enterprise adoption), career insights about Noah and full-stack AI development — or we can just have a conversation and see where it goes!\n\nWhat brings you here today?`
      }])
    } finally {
      setLoading(false)
    }
  }

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
          session_memory: sessionMemory,  // Send session state
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

      // Update session memory from backend
      if (data.session_memory) {
        setSessionMemory(data.session_memory)
      }

      // Update role if backend inferred it
      if (data.role && data.role !== selectedRole) {
        setSelectedRole(data.role as Role)
      }

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
