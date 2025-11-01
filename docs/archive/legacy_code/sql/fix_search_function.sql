-- Drop and recreate the search_kb_chunks function with better error handling
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

-- Test the function
SELECT search_kb_chunks(
    (SELECT embedding FROM kb_chunks LIMIT 1),  -- Use an actual embedding from the table
    0.0,
    3
);
