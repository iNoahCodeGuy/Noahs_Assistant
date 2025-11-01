-- Updated search function that should work
-- Run this in Supabase SQL Editor to replace the existing function

DROP FUNCTION IF EXISTS search_kb_chunks(vector, float, int);

CREATE OR REPLACE FUNCTION search_kb_chunks(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.3,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    id bigint,
    doc_id text,
    section text,
    content text,
    similarity float
)
LANGUAGE sql
AS $$
    SELECT
        kb_chunks.id,
        kb_chunks.doc_id,
        kb_chunks.section,
        kb_chunks.content,
        (1 - (kb_chunks.embedding <=> query_embedding))::float AS similarity
    FROM kb_chunks
    WHERE kb_chunks.embedding IS NOT NULL
    ORDER BY kb_chunks.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Test the function immediately
SELECT * FROM search_kb_chunks(
    (SELECT embedding FROM kb_chunks LIMIT 1),  -- Use an actual embedding from the table
    0.0,
    3
);
