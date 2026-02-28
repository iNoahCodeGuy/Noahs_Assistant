-- Supabase Database Schema for Noah's AI Assistant
-- This migration replaces Google Cloud SQL with Supabase Postgres + pgvector
--
-- Run this file in your Supabase SQL Editor to set up the database
-- Dashboard → SQL Editor → New Query → Paste and Run
--
-- What this creates:
-- 1. pgvector extension for similarity search
-- 2. kb_chunks table for career knowledge base with embeddings
-- 3. messages table for chat logs and analytics
-- 4. retrieval_logs table for RAG pipeline tracking
-- 5. links table for external resources
-- 6. feedback table for user ratings and contact requests
-- 7. Indexes for performance optimization

-- ============================================================================
-- EXTENSION: Enable pgvector for similarity search
-- ============================================================================
-- pgvector adds vector data types and similarity search operators
-- This is the key technology that replaces Google Cloud Vertex AI
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- TABLE: kb_chunks
-- Purpose: Store chunks of career knowledge base with embeddings
-- Replaces: Local FAISS vector store
-- ============================================================================
CREATE TABLE IF NOT EXISTS kb_chunks (
    id BIGSERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,  -- Source document identifier (e.g., 'career_kb', 'resume')
    section TEXT NOT NULL,  -- Section name (e.g., 'education', 'experience', 'skills')
    content TEXT NOT NULL,  -- The actual text content
    embedding vector(1536),  -- OpenAI ada-002 embedding (1536 dimensions)
    metadata JSONB DEFAULT '{}'::jsonb,  -- Additional metadata (dates, tags, etc.)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for fast similarity search
-- IVFFLAT is faster than exact search for large datasets
-- lists=100 is good for ~10k vectors, adjust based on data size
CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx 
ON kb_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Index for filtering by document and section
CREATE INDEX IF NOT EXISTS kb_chunks_doc_section_idx 
ON kb_chunks (doc_id, section);

-- Enable Row Level Security (RLS) - Supabase best practice
ALTER TABLE kb_chunks ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role to manage all data
CREATE POLICY "Service role can manage kb_chunks"
ON kb_chunks FOR ALL
TO service_role
USING (true);

-- Policy: Anon users can only read (for future public API)
CREATE POLICY "Anon users can read kb_chunks"
ON kb_chunks FOR SELECT
TO anon
USING (true);

-- ============================================================================
-- TABLE: messages
-- Purpose: Log all chat interactions for analytics
-- Replaces: Google Cloud SQL user_interactions table + Pub/Sub events
-- ============================================================================
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,  -- Track conversation sessions
    role_mode TEXT NOT NULL,  -- User role (e.g., 'Software Developer')
    query TEXT NOT NULL,  -- User's question
    answer TEXT NOT NULL,  -- Assistant's response
    query_type TEXT,  -- Classified query type (e.g., 'technical', 'career')
    latency_ms INTEGER,  -- Response time in milliseconds
    tokens_prompt INTEGER,  -- OpenAI tokens used in prompt
    tokens_completion INTEGER,  -- OpenAI tokens used in completion
    success BOOLEAN DEFAULT true,  -- Whether the query succeeded
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS messages_session_id_idx ON messages (session_id);
CREATE INDEX IF NOT EXISTS messages_role_mode_idx ON messages (role_mode);
CREATE INDEX IF NOT EXISTS messages_created_at_idx ON messages (created_at DESC);

-- Enable RLS
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role can manage messages"
ON messages FOR ALL
TO service_role
USING (true);

-- ============================================================================
-- TABLE: retrieval_logs
-- Purpose: Track which KB chunks were retrieved for each message
-- This helps analyze RAG pipeline effectiveness
-- ============================================================================
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT REFERENCES messages(id) ON DELETE CASCADE,
    topk_ids BIGINT[] NOT NULL,  -- Array of kb_chunks IDs retrieved
    scores FLOAT[] NOT NULL,  -- Similarity scores for each chunk
    grounded BOOLEAN DEFAULT false,  -- Whether the response cited sources
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for joining with messages
CREATE INDEX IF NOT EXISTS retrieval_logs_message_id_idx 
ON retrieval_logs (message_id);

-- Enable RLS
ALTER TABLE retrieval_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role can manage retrieval_logs"
ON retrieval_logs FOR ALL
TO service_role
USING (true);

-- ============================================================================
-- TABLE: links
-- Purpose: Store external links (YouTube, LinkedIn, etc.)
-- Replaces: Hardcoded links in settings
-- ============================================================================
CREATE TABLE IF NOT EXISTS links (
    key TEXT PRIMARY KEY,  -- Link identifier (e.g., 'mma_fight', 'linkedin')
    url TEXT NOT NULL,  -- The actual URL
    description TEXT,  -- Human-readable description
    category TEXT,  -- Category (e.g., 'social', 'media', 'work')
    active BOOLEAN DEFAULT true,  -- Can be disabled without deletion
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE links ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role can manage links"
ON links FOR ALL
TO service_role
USING (true);

CREATE POLICY "Anon users can read active links"
ON links FOR SELECT
TO anon
USING (active = true);

-- ============================================================================
-- TABLE: feedback
-- Purpose: Store user ratings and contact requests
-- Enables: Email notifications via Resend, SMS via Twilio
-- ============================================================================
CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT REFERENCES messages(id) ON DELETE SET NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),  -- 1-5 star rating
    comment TEXT,  -- Optional user feedback
    contact_requested BOOLEAN DEFAULT false,  -- Did user request contact?
    email TEXT,  -- User's email (if provided)
    notification_sent BOOLEAN DEFAULT false,  -- Track if we sent Twilio SMS
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS feedback_message_id_idx ON feedback (message_id);
CREATE INDEX IF NOT EXISTS feedback_contact_requested_idx 
ON feedback (contact_requested) WHERE contact_requested = true;

-- Enable RLS
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role can manage feedback"
ON feedback FOR ALL
TO service_role
USING (true);

-- ============================================================================
-- VIEWS: Analytics helpers
-- ============================================================================

-- View: Recent messages with their retrieval logs
CREATE OR REPLACE VIEW messages_with_retrieval AS
SELECT 
    m.id,
    m.session_id,
    m.role_mode,
    m.query,
    m.answer,
    m.latency_ms,
    m.created_at,
    r.topk_ids,
    r.scores,
    r.grounded
FROM messages m
LEFT JOIN retrieval_logs r ON m.id = r.message_id;

-- View: Message analytics by role
CREATE OR REPLACE VIEW analytics_by_role AS
SELECT 
    role_mode,
    COUNT(*) as total_messages,
    AVG(latency_ms) as avg_latency_ms,
    SUM(tokens_prompt) as total_tokens_prompt,
    SUM(tokens_completion) as total_tokens_completion,
    COUNT(CASE WHEN success THEN 1 END)::FLOAT / COUNT(*) as success_rate
FROM messages
GROUP BY role_mode;

-- ============================================================================
-- FUNCTIONS: Helper functions for common operations
-- ============================================================================

-- Function: Search similar KB chunks using pgvector
CREATE OR REPLACE FUNCTION search_kb_chunks(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 3
)
RETURNS TABLE (
    id bigint,
    doc_id text,
    section text,
    content text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        kb_chunks.id,
        kb_chunks.doc_id,
        kb_chunks.section,
        kb_chunks.content,
        1 - (kb_chunks.embedding <=> query_embedding) AS similarity
    FROM kb_chunks
    WHERE 1 - (kb_chunks.embedding <=> query_embedding) > match_threshold
    ORDER BY kb_chunks.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function: Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Auto-update updated_at for kb_chunks
CREATE TRIGGER update_kb_chunks_updated_at
    BEFORE UPDATE ON kb_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger: Auto-update updated_at for links
CREATE TRIGGER update_links_updated_at
    BEFORE UPDATE ON links
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SEED DATA: Initial links
-- ============================================================================
INSERT INTO links (key, url, description, category) VALUES
    ('mma_fight', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'Noah''s MMA fight video', 'media'),
    ('linkedin', 'https://linkedin.com/in/noah-ai', 'Noah''s LinkedIn profile', 'social'),
    ('github', 'https://github.com/iNoahCodeGuy', 'Noah''s GitHub', 'social')
ON CONFLICT (key) DO NOTHING;

-- ============================================================================
-- COMPLETE! 
-- ============================================================================
-- Your database is now ready for the Supabase + Vercel architecture
-- 
-- Next steps:
-- 1. Run the data migration script to populate kb_chunks from career_kb.csv
-- 2. Update application code to use supabase_config.py
-- 3. Test vector similarity search
-- 4. Deploy to Vercel
--
-- Cost estimate: ~$25-50/month for Supabase Pro (includes 8GB database)
-- ============================================================================