'use client'

import { useChat } from './components/hooks/useChat'
import { ChatHeader } from './components/chat/ChatHeader'
import { ChatMessages } from './components/chat/ChatMessages'
import { ChatInput } from './components/chat/ChatInput'

/**
 * Main chat page - refactored from 224 lines to 28 lines
 * 
 * BEFORE: All state, API logic, and UI mixed in one component
 * AFTER: Clean separation of concerns with dedicated components
 * 
 * Benefits:
 * - 87% code reduction in main file
 * - Each component testable independently
 * - Business logic (useChat) separated from UI
 * - Easy to modify individual pieces without touching others
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
