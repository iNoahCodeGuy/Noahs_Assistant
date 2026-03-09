'use client'

import { useEffect, useState } from 'react'
import {
  MessageSquare, Users, Target, TrendingUp,
  BarChart3, Layers, Clock, ArrowLeft, RefreshCw,
  ChevronDown, ChevronUp
} from 'lucide-react'
import Link from 'next/link'

interface Analytics {
  totalConversations: number
  avgTurns: number
  captureRate: number
  avgCaptureTurn: number | null
  visitorTypes: Record<string, number>
  topicCounts: Record<string, number>
  projectCounts: Record<string, number>
  depthDistribution: Record<string, number>
  recent: Array<{
    session_id: string
    role: string
    visitor_type: string
    turn_count: number
    data_captured: boolean
    capture_turn: number | null
    topics: string[]
    projects: string[]
    max_depth: number
    started_at: string
    last_active_at: string
  }>
}

interface Transcript {
  turn_number: number
  role: string
  content: string
  message_intent: string | null
  created_at: string
}

function StatCard({ icon: Icon, label, value, sub }: {
  icon: typeof MessageSquare
  label: string
  value: string | number
  sub?: string
}) {
  return (
    <div className="bg-chat-surface border border-chat-border rounded-2xl p-6 flex flex-col gap-2">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-chat-primary/20 to-chat-secondary/20 flex items-center justify-center">
          <Icon className="w-5 h-5 text-chat-primary" />
        </div>
        <span className="text-sm text-gray-400">{label}</span>
      </div>
      <div className="text-3xl font-bold text-white mt-1">{value}</div>
      {sub && <span className="text-xs text-gray-500">{sub}</span>}
    </div>
  )
}

function BarList({ data, color = 'from-chat-primary to-chat-secondary' }: {
  data: Record<string, number>
  color?: string
}) {
  const sorted = Object.entries(data).sort((a, b) => b[1] - a[1])
  const max = sorted.length > 0 ? sorted[0][1] : 1

  if (sorted.length === 0) {
    return <p className="text-gray-500 text-sm">No data yet</p>
  }

  return (
    <div className="space-y-3">
      {sorted.map(([key, count]) => (
        <div key={key}>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-300 truncate max-w-[70%]">{key}</span>
            <span className="text-gray-400 font-mono">{count}</span>
          </div>
          <div className="h-2 bg-chat-bg rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-500`}
              style={{ width: `${(count / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

function DepthGauge({ distribution }: { distribution: Record<string, number> }) {
  const total = Object.values(distribution).reduce((s, v) => s + v, 0) || 1
  const labels: Record<string, string> = { '1': 'Quick', '2': 'Medium', '3': 'Deep Dive' }
  const colors: Record<string, string> = {
    '1': 'bg-violet-400',
    '2': 'bg-chat-primary',
    '3': 'bg-chat-secondary'
  }

  return (
    <div className="space-y-3">
      {['1', '2', '3'].map(level => {
        const count = distribution[level] || 0
        const pct = ((count / total) * 100).toFixed(0)
        return (
          <div key={level} className="flex items-center gap-3">
            <span className="text-sm text-gray-400 w-20">{labels[level]}</span>
            <div className="flex-1 h-3 bg-chat-bg rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${colors[level]} transition-all duration-500`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="text-sm text-gray-500 font-mono w-12 text-right">{pct}%</span>
          </div>
        )
      })}
    </div>
  )
}

function TranscriptViewer({ sessionId, onClose }: { sessionId: string, onClose: () => void }) {
  const [messages, setMessages] = useState<Transcript[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`/api/analytics/transcript?session_id=${sessionId}`)
      .then(r => r.json())
      .then(data => setMessages(data.messages || []))
      .catch(() => setMessages([]))
      .finally(() => setLoading(false))
  }, [sessionId])

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-chat-surface border border-chat-border rounded-2xl max-w-2xl w-full max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-chat-border">
          <h3 className="font-semibold text-white">Conversation Transcript</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
            Close
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {loading ? (
            <p className="text-gray-500 text-center py-8">Loading...</p>
          ) : messages.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No messages recorded for this session</p>
          ) : (
            messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] rounded-xl px-4 py-3 text-sm ${
                  m.role === 'user'
                    ? 'bg-gradient-to-r from-chat-primary to-chat-secondary text-white'
                    : 'bg-chat-bg border border-chat-border text-gray-300'
                }`}>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs opacity-60">
                      Turn {m.turn_number} {m.message_intent ? `· ${m.message_intent}` : ''}
                    </span>
                  </div>
                  <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [data, setData] = useState<Analytics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [viewingTranscript, setViewingTranscript] = useState<string | null>(null)
  const [expandedRow, setExpandedRow] = useState<string | null>(null)

  const fetchData = () => {
    setLoading(true)
    fetch('/api/analytics')
      .then(r => r.json())
      .then(d => {
        if (d.error) throw new Error(d.error)
        setData(d)
        setError('')
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchData() }, [])

  if (loading) {
    return (
      <div className="min-h-screen gradient-bg flex items-center justify-center">
        <div className="flex gap-2">
          <div className="w-3 h-3 rounded-full bg-chat-primary animate-bounce" style={{ animationDelay: '0ms' }} />
          <div className="w-3 h-3 rounded-full bg-chat-primary animate-bounce" style={{ animationDelay: '150ms' }} />
          <div className="w-3 h-3 rounded-full bg-chat-primary animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen gradient-bg flex items-center justify-center">
        <div className="bg-chat-surface border border-chat-border rounded-2xl p-8 text-center max-w-md">
          <p className="text-red-400 mb-4">{error}</p>
          <button onClick={fetchData} className="px-4 py-2 rounded-xl bg-gradient-to-r from-chat-primary to-chat-secondary text-white text-sm">
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!data) return null

  return (
    <div className="min-h-screen gradient-bg">
      {viewingTranscript && (
        <TranscriptViewer sessionId={viewingTranscript} onClose={() => setViewingTranscript(null)} />
      )}

      {/* Header */}
      <header className="border-b border-chat-border bg-chat-surface/80 backdrop-blur-lg sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="text-gray-400 hover:text-white transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-xl font-bold gradient-text">Portfolia Analytics</h1>
              <p className="text-xs text-gray-500">Conversation intelligence dashboard</p>
            </div>
          </div>
          <button
            onClick={fetchData}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-chat-bg border border-chat-border text-gray-400 hover:text-white hover:border-chat-primary transition-colors text-sm"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Stat Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard icon={MessageSquare} label="Total Conversations" value={data.totalConversations} />
          <StatCard icon={TrendingUp} label="Avg Turns / Conversation" value={data.avgTurns} />
          <StatCard
            icon={Target}
            label="Data Capture Rate"
            value={`${data.captureRate}%`}
            sub={data.avgCaptureTurn !== null ? `Avg capture at turn ${data.avgCaptureTurn}` : 'No captures yet'}
          />
          <StatCard
            icon={Clock}
            label="Avg Capture Turn"
            value={data.avgCaptureTurn ?? '—'}
            sub="Turn where info is collected"
          />
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Visitor Types */}
          <div className="bg-chat-surface border border-chat-border rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-5">
              <Users className="w-4 h-4 text-chat-primary" />
              <h2 className="font-semibold text-white text-sm">Visitor Types</h2>
            </div>
            <BarList data={data.visitorTypes} />
          </div>

          {/* Topics */}
          <div className="bg-chat-surface border border-chat-border rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-5">
              <BarChart3 className="w-4 h-4 text-chat-primary" />
              <h2 className="font-semibold text-white text-sm">Topics Discussed</h2>
            </div>
            <BarList data={data.topicCounts} color="from-violet-500 to-purple-500" />
          </div>

          {/* Projects */}
          <div className="bg-chat-surface border border-chat-border rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-5">
              <Layers className="w-4 h-4 text-chat-primary" />
              <h2 className="font-semibold text-white text-sm">Projects Asked About</h2>
            </div>
            <BarList data={data.projectCounts} color="from-pink-500 to-rose-500" />
          </div>
        </div>

        {/* Depth + Recent Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Depth Distribution */}
          <div className="bg-chat-surface border border-chat-border rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-5">
              <TrendingUp className="w-4 h-4 text-chat-primary" />
              <h2 className="font-semibold text-white text-sm">Conversation Depth</h2>
            </div>
            <DepthGauge distribution={data.depthDistribution} />
          </div>

          {/* Recent Conversations */}
          <div className="lg:col-span-2 bg-chat-surface border border-chat-border rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-5">
              <MessageSquare className="w-4 h-4 text-chat-primary" />
              <h2 className="font-semibold text-white text-sm">Recent Conversations</h2>
            </div>
            <div className="space-y-2 max-h-[400px] overflow-y-auto">
              {data.recent.length === 0 ? (
                <p className="text-gray-500 text-sm text-center py-8">No conversations yet</p>
              ) : (
                data.recent.map(conv => (
                  <div key={conv.session_id}>
                    <button
                      onClick={() => setExpandedRow(expandedRow === conv.session_id ? null : conv.session_id)}
                      className="w-full text-left bg-chat-bg border border-chat-border/50 rounded-xl p-4 hover:border-chat-primary/50 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3 min-w-0">
                          <div className={`w-2 h-2 rounded-full flex-shrink-0 ${conv.data_captured ? 'bg-green-400' : 'bg-gray-600'}`} />
                          <span className="text-sm text-gray-300 truncate">
                            {conv.visitor_type || 'unknown'}
                          </span>
                          <span className="text-xs text-gray-500">
                            {conv.turn_count} turns
                          </span>
                          {conv.data_captured && (
                            <span className="text-xs px-2 py-0.5 rounded-full bg-green-400/10 text-green-400 flex-shrink-0">
                              captured @ turn {conv.capture_turn}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-3 flex-shrink-0">
                          <span className="text-xs text-gray-500">
                            {new Date(conv.started_at).toLocaleDateString()}
                          </span>
                          {expandedRow === conv.session_id
                            ? <ChevronUp className="w-4 h-4 text-gray-500" />
                            : <ChevronDown className="w-4 h-4 text-gray-500" />
                          }
                        </div>
                      </div>
                    </button>

                    {expandedRow === conv.session_id && (
                      <div className="mt-1 bg-chat-bg/50 border border-chat-border/30 rounded-xl p-4 space-y-3 text-sm">
                        <div className="flex flex-wrap gap-4">
                          <div>
                            <span className="text-gray-500">Role:</span>{' '}
                            <span className="text-gray-300">{conv.role || '—'}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Depth:</span>{' '}
                            <span className="text-gray-300">
                              {conv.max_depth === 3 ? 'Deep Dive' : conv.max_depth === 2 ? 'Medium' : 'Quick'}
                            </span>
                          </div>
                        </div>
                        {conv.topics && conv.topics.length > 0 && (
                          <div>
                            <span className="text-gray-500 text-xs">Topics:</span>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {conv.topics.map(t => (
                                <span key={t} className="text-xs px-2 py-0.5 rounded-full bg-chat-primary/10 text-chat-primary">
                                  {t}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        {conv.projects && conv.projects.length > 0 && (
                          <div>
                            <span className="text-gray-500 text-xs">Projects:</span>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {conv.projects.map(p => (
                                <span key={p} className="text-xs px-2 py-0.5 rounded-full bg-chat-secondary/10 text-chat-secondary">
                                  {p}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        <button
                          onClick={() => setViewingTranscript(conv.session_id)}
                          className="text-xs text-chat-primary hover:text-chat-secondary transition-colors"
                        >
                          View full transcript →
                        </button>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
