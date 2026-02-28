# Page.tsx Refactoring Guide

## Current Issues

The `app/page.tsx` file is **224 lines** in a single component, making it:
- Hard to navigate and understand
- Difficult to test individual pieces
- Challenging to maintain and update
- Prone to bugs when making changes

## Proposed Component Structure

```
app/
├── page.tsx (50 lines) ← Main orchestrator
└── components/
    ├── chat/
    │   ├── ChatHeader.tsx (30 lines) ← Header with role selector
    │   ├── ChatMessages.tsx (40 lines) ← Message list container
    │   ├── ChatMessage.tsx (60 lines) ← Individual message bubble
    │   ├── ChatInput.tsx (40 lines) ← Input form
    │   └── EmptyState.tsx (20 lines) ← Welcome screen
    ├── ui/
    │   ├── RoleSelector.tsx (30 lines) ← Dropdown component
    │   └── LoadingDots.tsx (15 lines) ← Typing indicator
    └── hooks/
        └── useChat.ts (50 lines) ← Chat logic & API calls
```

## Benefits of Refactoring

### Before (Current):
```tsx
// One 224-line component with everything mixed together
export default function Home() {
  // State management
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  
  // API logic
  const sendMessage = async () => { /* 30 lines */ }
  
  // Rendering
  return (
    <div>
      {/* 150 lines of JSX */}
    </div>
  )
}
```

### After (Proposed):
```tsx
// Clean 50-line orchestrator
export default function Home() {
  const { messages, sendMessage, loading, input, setInput, selectedRole, setSelectedRole } = useChat()
  
  return (
    <div className="flex flex-col h-screen gradient-bg">
      <ChatHeader role={selectedRole} onRoleChange={setSelectedRole} />
      <ChatMessages messages={messages} loading={loading} />
      <ChatInput 
        value={input}
        onChange={setInput}
        onSubmit={sendMessage}
        disabled={loading}
      />
    </div>
  )
}
```

## Step-by-Step Refactoring Plan

### Step 1: Extract Custom Hook (useChat.ts)
Move all state management and API logic:
```tsx
// app/components/hooks/useChat.ts
export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedRole, setSelectedRole] = useState<Role>('Hiring Manager (nontechnical)')
  const [sessionId] = useState(() => crypto.randomUUID())

  const sendMessage = async (content: string) => {
    // All API logic here (currently 30 lines in page.tsx)
  }

  return { messages, sendMessage, loading, input, setInput, selectedRole, setSelectedRole }
}
```

**Benefits:**
- ✅ Testable in isolation
- ✅ Reusable in other components
- ✅ Separates logic from presentation

### Step 2: Extract ChatHeader Component
```tsx
// app/components/chat/ChatHeader.tsx
interface ChatHeaderProps {
  role: Role
  onRoleChange: (role: Role) => void
}

export function ChatHeader({ role, onRoleChange }: ChatHeaderProps) {
  return (
    <header className="border-b border-chat-border bg-chat-surface/80 backdrop-blur-lg">
      <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* Logo and title */}
        </div>
        <RoleSelector value={role} onChange={onRoleChange} />
      </div>
    </header>
  )
}
```

**Benefits:**
- ✅ Clear responsibility (just the header)
- ✅ Easy to update logo/title without touching other code
- ✅ Props are explicit and typed

### Step 3: Extract ChatMessage Component
```tsx
// app/components/chat/ChatMessage.tsx
interface ChatMessageProps {
  message: Message
  isUser: boolean
}

export function ChatMessage({ message, isUser }: ChatMessageProps) {
  return (
    <div className={cn("flex gap-4", isUser ? "justify-end" : "justify-start")}>
      {!isUser && <BotAvatar />}
      
      <MessageBubble isUser={isUser}>
        <MessageContent content={message.content} />
        {message.sources && <MessageSources sources={message.sources} />}
      </MessageBubble>
      
      {isUser && <UserAvatar />}
    </div>
  )
}
```

**Benefits:**
- ✅ Single responsibility (render one message)
- ✅ Easy to add features (reactions, timestamps, etc.)
- ✅ Can be unit tested with mock data

### Step 4: Extract ChatInput Component
```tsx
// app/components/chat/ChatInput.tsx
interface ChatInputProps {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  disabled: boolean
}

export function ChatInput({ value, onChange, onSubmit, disabled }: ChatInputProps) {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!value.trim() || disabled) return
    onSubmit()
  }

  return (
    <div className="border-t border-chat-border bg-chat-surface/80 backdrop-blur-lg">
      <div className="max-w-4xl mx-auto px-4 py-4">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="Ask me anything about Noah..."
            className="flex-1 bg-chat-bg border border-chat-border rounded-xl px-6 py-3"
            disabled={disabled}
          />
          <SendButton disabled={!value.trim() || disabled} />
        </form>
      </div>
    </div>
  )
}
```

**Benefits:**
- ✅ Easy to add features (voice input, attachments, emoji picker)
- ✅ Clear props interface
- ✅ Isolated testing

## Quick Wins (Low-Effort Improvements)

### 1. Extract Inline Styles to Constants
```tsx
// Before: Hard to read
className="flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}"

// After: Clear and reusable
const MESSAGE_ALIGNMENT = {
  user: 'justify-end',
  assistant: 'justify-start'
} as const

className={`flex gap-4 ${MESSAGE_ALIGNMENT[message.role]}`}
```

### 2. Extract Magic Strings
```tsx
// Before: Repeated throughout
'Hiring Manager (nontechnical)'

// After: Single source of truth
const DEFAULT_ROLE = 'Hiring Manager (nontechnical)' as const
```

### 3. Use Utility Functions
```tsx
// Before: Logic in JSX
{message.sources && message.sources.length > 0 && (...)}

// After: Clean helper
function hasS sources(message: Message) {
  return message.sources && message.sources.length > 0
}

{hasSources(message) && <MessageSources sources={message.sources} />}
```

## Recommended Order of Refactoring

1. **Extract `useChat` hook** (30 min) ← Start here, biggest impact
2. **Extract `ChatInput`** (15 min) ← Easy, self-contained
3. **Extract `ChatHeader`** (15 min) ← Simple, clear boundaries
4. **Extract `ChatMessage`** (30 min) ← More complex, but worth it
5. **Extract smaller UI components** (15 min each) ← Polish

**Total time**: ~2 hours
**Result**: Code goes from 224 lines → 50 lines main + 6 focused components

## When to Refactor?

### ✅ Good Time to Refactor:
- Before adding new features (makes it easier)
- When you notice bugs in complex areas
- During low-priority work weeks
- When onboarding new developers

### ⚠️ Wait to Refactor If:
- You're about to demo to users
- Active bugs need urgent fixing
- Major deadline approaching
- Feature is working and rarely touched

## Testing Strategy

After refactoring:
```tsx
// Test individual components
describe('ChatMessage', () => {
  it('renders user message on right', () => {
    render(<ChatMessage message={mockUser Message} isUser />)
    expect(screen.getByText(mockUserMessage.content)).toBeInTheDocument()
  })
})

// Test custom hook
describe('useChat', () => {
  it('sends message and updates state', async () => {
    const { result } = renderHook(() => useChat())
    await act(() => result.current.sendMessage('test'))
    expect(result.current.messages).toHaveLength(2) // user + assistant
  })
})
```

## Further Reading

- [React Component Patterns](https://kentcdodds.com/blog/compound-components)
- [Custom Hooks Best Practices](https://react.dev/learn/reusing-logic-with-custom-hooks)
- [Separation of Concerns in React](https://www.robinwieruch.de/react-component-composition/)
