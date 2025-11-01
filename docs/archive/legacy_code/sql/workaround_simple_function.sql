-- Workaround: Create a simpler search function that PostgREST can handle
-- Run this in Supabase SQL Editor

-- Drop the old function
DROP FUNCTION IF EXISTS search_kb_chunks_simple;

-- Create a simplified version that just returns everything
CREATE OR REPLACE FUNCTION search_kb_chunks_simple(
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id bigint,
    doc_id text,
    section text,
    content text
)
LANGUAGE sql
AS $$
    SELECT
        id,
        doc_id,
        section,
        content
    FROM kb_chunks
    WHERE embedding IS NOT NULL
    ORDER BY id
    LIMIT match_count;
$$;

-- Test it
SELECT * FROM search_kb_chunks_simple(5);
