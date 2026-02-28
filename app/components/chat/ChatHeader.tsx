'use client'

import { Sparkles, ChevronDown } from 'lucide-react'
import type { Role } from '../hooks/useChat'

const ROLES: Role[] = [
  'Hiring Manager (nontechnical)',
  'Hiring Manager (technical)',
  'Software Developer',
  'Just looking around',
  "Looking to confess I've had a crush on Noah for years"
]

interface ChatHeaderProps {
  role: Role
  onRoleChange: (role: Role) => void
}

/**
 * Chat header with logo and role selector
 */
export function ChatHeader({ role, onRoleChange }: ChatHeaderProps) {
  return (
    <header className="border-b border-chat-border bg-chat-surface/80 backdrop-blur-lg">
      <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-chat-primary to-chat-secondary flex items-center justify-center">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold gradient-text">Portfolia</h1>
            <p className="text-xs text-gray-400">Noah's AI Assistant - Interactive Resume & Career Assistant</p>
          </div>
        </div>
        
        <div className="relative">
          <select
            value={role}
            onChange={(e) => onRoleChange(e.target.value as Role)}
            className="appearance-none bg-chat-surface border border-chat-border rounded-lg px-4 py-2 pr-10 text-sm focus:outline-none focus:border-chat-primary transition-colors cursor-pointer"
          >
            {ROLES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
          <ChevronDown 
            size={16} 
            className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-gray-400" 
          />
        </div>
      </div>
    </header>
  )
}
