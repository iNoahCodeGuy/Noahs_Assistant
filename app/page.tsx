'use client'

import { useChat } from './components/hooks/useChat'
import { ChatHeader } from './components/chat/ChatHeader'
import { ChatMessages } from './components/chat/ChatMessages'
import { ChatInput } from './components/chat/ChatInput'

/** Chat page: state and API logic live in useChat; UI in the chat components. */
export default function Home() {
  const {
    messages,
    input,
    setInput,
    loading,
    formActive,
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
        onSendMessage={(content) => sendMessage(content)}
      />
      
      <ChatInput
        value={input}
        onChange={setInput}
        onSubmit={() => sendMessage()}
        disabled={loading || formActive}
        placeholder={formActive ? 'Please fill out the form above first' : undefined}
      />
    </div>
  )
}
