import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

function getSupabase() {
  const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL || ''
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || ''
  return createClient(url, key)
}

export async function GET() {
  try {
    const supabase = getSupabase()

    // Fetch all sessions
    const { data: sessions, error: sessErr } = await supabase
      .from('conversation_sessions')
      .select('*')
      .order('started_at', { ascending: false })

    if (sessErr) throw sessErr

    const all = sessions || []
    const totalConversations = all.length
    const totalTurns = all.reduce((s, r) => s + (r.turn_count || 0), 0)
    const avgTurns = totalConversations > 0 ? +(totalTurns / totalConversations).toFixed(1) : 0
    const captured = all.filter(r => r.data_captured)
    const captureRate = totalConversations > 0 ? +((captured.length / totalConversations) * 100).toFixed(1) : 0
    const avgCaptureTurn = captured.length > 0
      ? +(captured.reduce((s, r) => s + (r.capture_turn || 0), 0) / captured.length).toFixed(1)
      : null

    // Visitor type breakdown
    const visitorTypes: Record<string, number> = {}
    all.forEach(r => {
      const vt = r.visitor_type || 'unknown'
      visitorTypes[vt] = (visitorTypes[vt] || 0) + 1
    })

    // Topic frequency
    const topicCounts: Record<string, number> = {}
    all.forEach(r => {
      (r.topics_discussed || []).forEach((t: string) => {
        topicCounts[t] = (topicCounts[t] || 0) + 1
      })
    })

    // Project frequency
    const projectCounts: Record<string, number> = {}
    all.forEach(r => {
      (r.projects_asked_about || []).forEach((p: string) => {
        projectCounts[p] = (projectCounts[p] || 0) + 1
      })
    })

    // Depth distribution
    const depthDist: Record<number, number> = { 1: 0, 2: 0, 3: 0 }
    all.forEach(r => {
      const d = r.max_depth_level || 1
      depthDist[d] = (depthDist[d] || 0) + 1
    })

    // Recent conversations (last 10)
    const recent = all.slice(0, 10).map(r => ({
      session_id: r.session_id,
      role: r.role,
      visitor_type: r.visitor_type,
      turn_count: r.turn_count,
      data_captured: r.data_captured,
      capture_turn: r.capture_turn,
      topics: r.topics_discussed,
      projects: r.projects_asked_about,
      max_depth: r.max_depth_level,
      started_at: r.started_at,
      last_active_at: r.last_active_at,
    }))

    return NextResponse.json({
      totalConversations,
      avgTurns,
      captureRate,
      avgCaptureTurn,
      visitorTypes,
      topicCounts,
      projectCounts,
      depthDistribution: depthDist,
      recent,
    })
  } catch (error) {
    console.error('Analytics API error:', error)
    return NextResponse.json({ error: 'Failed to fetch analytics' }, { status: 500 })
  }
}
