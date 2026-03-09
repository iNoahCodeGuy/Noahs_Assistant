import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

function getSupabase() {
  const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL || ''
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || ''
  return createClient(url, key)
}

export async function GET(request: NextRequest) {
  try {
    const sessionId = request.nextUrl.searchParams.get('session_id')
    if (!sessionId) {
      return NextResponse.json({ error: 'session_id required' }, { status: 400 })
    }

    const supabase = getSupabase()

    const { data: messages, error } = await supabase
      .from('conversation_messages')
      .select('turn_number, role, content, message_intent, created_at')
      .eq('session_id', sessionId)
      .order('turn_number', { ascending: true })
      .order('created_at', { ascending: true })

    if (error) throw error

    return NextResponse.json({ messages: messages || [] })
  } catch (error) {
    console.error('Transcript API error:', error)
    return NextResponse.json({ error: 'Failed to fetch transcript' }, { status: 500 })
  }
}
