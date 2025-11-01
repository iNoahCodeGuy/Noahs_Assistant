-- Simple test to verify the search function works
-- Run this in Supabase SQL Editor

-- First, check if kb_chunks table has data
SELECT
    id,
    doc_id,
    LEFT(content, 50) as content_preview,
    embedding IS NOT NULL as has_embedding
FROM kb_chunks
LIMIT 5;

-- Test if we can compute similarity manually
-- (This uses a dummy embedding - just zeros)
SELECT
    id,
    content,
    1 - (embedding <=> ARRAY[0.0, 0.0]::vector(1536)) as similarity
FROM kb_chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> ARRAY[0.0, 0.0]::vector(1536)
LIMIT 3;
