import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

type ChatRequest = {
  query: string
  role: string
  session_id: string
  chat_history: Array<{ role: string; content: string }>
}

export async function POST(request: NextRequest) {
  try {
    const body: ChatRequest = await request.json()
    
    // Call your Python backend (update this URL based on your deployment)
    // Option 1: If Python backend is on same Vercel deployment
    // const pythonBackendUrl = `${request.nextUrl.origin}/api/python/chat`
    
    // Option 2: If Python backend is separate (Railway, Render, etc.)
    const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000/chat'
    
    // Option 3: Call Python functions directly (requires Python on Vercel)
    // For now, we'll simulate a response for development
    
    // TODO: Uncomment this when Python backend is deployed
    /*
    const response = await fetch(pythonBackendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body)
    })

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
    */

    // TEMPORARY: Mock response for development
    // Remove this and uncomment above when Python backend is ready
    await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate delay

    return NextResponse.json({
      response: `Thank you for your question! I'm currently being set up to provide detailed answers about Noah's background, experience, and technical skills.\n\nYour role: ${body.role}\nYour question: ${body.query}\n\n[This is a development response. The full AI assistant will be available once the Python backend is connected.]`,
      sources: [
        {
          doc_id: 'career_kb',
          section: 'Professional Background',
          similarity: 0.85
        }
      ],
      context: [],
      type: 'development_mode'
    })
    
  } catch (error) {
    console.error('Chat API error:', error)
    return NextResponse.json(
      { 
        error: 'Failed to process request',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    )
  }
}
