-- ============================================================================
-- Supabase Migration for Noah's AI Assistant - CLEAN VERSION
-- This version drops existing objects first to avoid conflicts
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- DROP EXISTING POLICIES (to avoid conflicts)
-- ============================================================================
DROP POLICY IF EXISTS "service_role_all_kb_chunks" ON kb_chunks;
DROP POLICY IF EXISTS "service_role_all_messages" ON messages;
DROP POLICY IF EXISTS "service_role_all_retrieval_logs" ON retrieval_logs;
DROP POLICY IF EXISTS "service_role_all_links" ON links;
DROP POLICY IF EXISTS "service_role_all_feedback" ON feedback;
DROP POLICY IF EXISTS "authenticated_select_messages" ON messages;
DROP POLICY IF EXISTS "authenticated_insert_messages" ON messages;
DROP POLICY IF EXISTS "authenticated_select_retrieval_logs" ON retrieval_logs;
DROP POLICY IF EXISTS "authenticated_insert_feedback" ON feedback;
DROP POLICY IF EXISTS "authenticated_select_feedback" ON feedback;
DROP POLICY IF EXISTS "public_select_links" ON links;

-- ============================================================================
-- TABLES (CREATE IF NOT EXISTS - safe to run multiple times)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb_chunks (
    id BIGSERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    section TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL DEFAULT gen_random_uuid(),
    role_mode TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    query_type TEXT,
    latency_ms INTEGER,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    success BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT REFERENCES messages(id) ON DELETE CASCADE,
    topk_ids BIGINT[] NOT NULL,
    scores FLOAT[] NOT NULL,
    grounded BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS links (
    key TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    description TEXT,
    category TEXT,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT REFERENCES messages(id) ON DELETE SET NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    contact_requested BOOLEAN DEFAULT false,
    email TEXT,
    notification_sent BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================
CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx ON kb_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS kb_chunks_doc_section_idx ON kb_chunks (doc_id, section);
CREATE INDEX IF NOT EXISTS messages_session_id_idx ON messages (session_id);
CREATE INDEX IF NOT EXISTS messages_role_mode_idx ON messages (role_mode);
CREATE INDEX IF NOT EXISTS messages_created_at_idx ON messages (created_at DESC);
CREATE INDEX IF NOT EXISTS retrieval_logs_message_id_idx ON retrieval_logs (message_id);
CREATE INDEX IF NOT EXISTS feedback_message_id_idx ON feedback (message_id);
CREATE INDEX IF NOT EXISTS feedback_contact_requested_idx ON feedback (contact_requested) WHERE contact_requested = true;

-- ============================================================================
-- ENABLE RLS
-- ============================================================================
ALTER TABLE kb_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE retrieval_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE links ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- CREATE POLICIES (after dropping old ones)
-- ============================================================================
CREATE POLICY "service_role_all_kb_chunks" ON kb_chunks FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all_messages" ON messages FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all_retrieval_logs" ON retrieval_logs FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all_links" ON links FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all_feedback" ON feedback FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "authenticated_select_messages" ON messages FOR SELECT TO authenticated USING (true);
CREATE POLICY "authenticated_insert_messages" ON messages FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "authenticated_select_retrieval_logs" ON retrieval_logs FOR SELECT TO authenticated USING (true);
CREATE POLICY "authenticated_insert_feedback" ON feedback FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "authenticated_select_feedback" ON feedback FOR SELECT TO authenticated USING (true);
CREATE POLICY "public_select_links" ON links FOR SELECT TO anon USING (active = true);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================
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

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================
DROP TRIGGER IF EXISTS update_kb_chunks_updated_at ON kb_chunks;
DROP TRIGGER IF EXISTS update_links_updated_at ON links;

CREATE TRIGGER update_kb_chunks_updated_at BEFORE UPDATE ON kb_chunks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_links_updated_at BEFORE UPDATE ON links FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SEED DATA
-- ============================================================================
INSERT INTO links (key, url, description, category) VALUES
    ('mma_fight', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'Noah''s MMA fight video', 'media'),
    ('linkedin', 'https://linkedin.com/in/noah-ai', 'Noah''s LinkedIn profile', 'social'),
    ('github', 'https://github.com/iNoahCodeGuy', 'Noah''s GitHub', 'social')
ON CONFLICT (key) DO NOTHING;

-- ============================================================================
-- VERIFICATION
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE '✅ Migration complete!';
    RAISE NOTICE 'Tables: kb_chunks, messages, retrieval_logs, links, feedback';
    RAISE NOTICE 'Extensions: vector, pgcrypto';
    RAISE NOTICE 'Next: Run data migration to populate kb_chunks';
END $$;

SELECT 'SUCCESS: Tables created' as status;
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('kb_chunks', 'messages', 'retrieval_logs', 'links', 'feedback')
ORDER BY table_name;
