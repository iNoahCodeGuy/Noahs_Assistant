'use client'

import { useChat } from './components/hooks/useChat'
import { ChatHeader } from './components/chat/ChatHeader'
import { ChatMessages } from './components/chat/ChatMessages'
import { ChatInput } from './components/chat/ChatInput'

/**
 * Main chat page - refactored version
 * 
 * BEFORE: 224 lines with all logic mixed together
 * AFTER: 28 lines that clearly shows the structure
 * 
 * Benefits:
 * - Easy to understand at a glance
 * - Each component is testable independently
 * - Can update header/messages/input without touching other parts
 * - Business logic (useChat) separated from UI
 */
export default function Home() {
  const {
    messages,
    input,
    setInput,
    loading,
    selectedRole,
    setSelectedRole,
    sendMessage
  } = useChat()

  return (
    <div className="flex flex-col h-screen gradient-bg">
      <ChatHeader 
        role={selectedRole} 
        onRoleChange={setSelectedRole} 
      />
      
      <ChatMessages 
        messages={messages} 
        loading={loading} 
      />
      
      <ChatInput
        value={input}
        onChange={setInput}
        onSubmit={() => sendMessage()}
        disabled={loading}
      />
    </div>
  )
}
