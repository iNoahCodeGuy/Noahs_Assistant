-- ============================================================
-- QUICK FIX: Correct type mismatch in match_kb_chunks function
-- ============================================================
-- Issue: Function returned 'int' but kb_chunks.id is 'bigint'
-- Solution: Change return type from int to bigint
-- Run this in Supabase SQL Editor to fix immediately
-- ============================================================

CREATE OR REPLACE FUNCTION match_kb_chunks(
    query_embedding vector(1536),      -- Query embedding from OpenAI
    match_threshold float DEFAULT 0.60, -- Minimum similarity (0-1)
    match_count int DEFAULT 3,          -- Number of results to return
    filter_doc_id text DEFAULT NULL     -- Optional: filter by doc_id
)
RETURNS TABLE (
    id bigint,  -- Changed from 'int' to 'bigint' to match actual column type
    doc_id text,
    section text,
    content text,
    similarity float
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT
        kb_chunks.id,
        kb_chunks.doc_id,
        kb_chunks.section,
        kb_chunks.content,
        -- Convert cosine distance to cosine similarity
        -- pgvector's <=> returns distance (0-2), we want similarity (0-1)
        (1 - (kb_chunks.embedding <=> query_embedding))::float AS similarity
    FROM kb_chunks
    WHERE
        -- Apply similarity threshold filter
        (1 - (kb_chunks.embedding <=> query_embedding)) > match_threshold
        -- Apply doc_id filter if provided
        AND (filter_doc_id IS NULL OR kb_chunks.doc_id = filter_doc_id)
    -- Sort by similarity (highest first)
    -- Note: ORDER BY distance (ascending) = ORDER BY similarity (descending)
    ORDER BY kb_chunks.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;
