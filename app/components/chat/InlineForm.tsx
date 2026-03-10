'use client'

import { useState } from 'react'

interface FormField {
  label: string
  key: string
  required?: boolean
  multiline?: boolean
}

interface InlineFormProps {
  fields: FormField[]
  onSubmit: (formatted: string) => void
  note?: string
}

/**
 * Inline form rendered inside a chat message bubble.
 * On submit, formats field values as "Label: value" text and sends as a chat message.
 */
export function InlineForm({ fields, onSubmit, note }: InlineFormProps) {
  const [values, setValues] = useState<Record<string, string>>({})
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    // Format as "Label: value" lines so the backend parser can handle it
    const lines = fields
      .map(f => `${f.label}: ${(values[f.key] || '').trim()}`)
      .join('\n')

    setSubmitted(true)
    onSubmit(lines)
  }

  if (submitted) {
    return (
      <p className="text-gray-400 italic text-sm mt-2">Submitted.</p>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="mt-4 space-y-3 max-w-md">
      {fields.map(f => (
        <div key={f.key}>
          {f.multiline ? (
            <textarea
              placeholder={f.label + (f.required ? ' *' : '')}
              value={values[f.key] || ''}
              onChange={e => setValues(prev => ({ ...prev, [f.key]: e.target.value }))}
              rows={3}
              className="w-full bg-chat-bg border border-chat-border rounded-lg px-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-chat-primary transition-colors resize-none"
            />
          ) : (
            <input
              type="text"
              placeholder={f.label + (f.required ? ' *' : '')}
              value={values[f.key] || ''}
              onChange={e => setValues(prev => ({ ...prev, [f.key]: e.target.value }))}
              className="w-full bg-chat-bg border border-chat-border rounded-lg px-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-chat-primary transition-colors"
            />
          )}
        </div>
      ))}
      {note && (
        <p className="text-xs text-gray-500">{note}</p>
      )}
      <button
        type="submit"
        className="bg-gradient-to-r from-chat-primary to-chat-secondary rounded-lg px-5 py-2 text-sm font-medium hover:opacity-90 transition-opacity"
      >
        Submit
      </button>
    </form>
  )
}
