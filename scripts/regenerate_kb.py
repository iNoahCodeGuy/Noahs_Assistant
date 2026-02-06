#!/usr/bin/env python3
"""
Regenerate Knowledge Base with High-Quality Content

This script:
1. Reads the markdown files in data/
2. Chunks them intelligently (by section, not by line)
3. Generates embeddings
4. Uploads to Supabase

Usage:
    python3 scripts/regenerate_kb.py --dry-run  # Preview without uploading
    python3 scripts/regenerate_kb.py            # Actually regenerate KB
"""

import os
import re
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
import argparse

load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def parse_markdown_sections(filepath: str) -> List[Dict[str, str]]:
    """Parse markdown file into logical sections.

    Returns list of chunks with:
    - section: Section title (h2 or h3 header)
    - content: Full section content (without code fragments)
    - doc_id: Derived from filename
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract doc_id from filename
    doc_id = os.path.basename(filepath).replace('.md', '').replace('noah_', '')

    chunks = []

    # Split by ## headers (h2)
    sections = re.split(r'\n## ', content)

    for i, section in enumerate(sections):
        if i == 0:
            # First section is title/intro, skip or handle specially
            continue

        # Extract section title (first line)
        lines = section.split('\n', 1)
        title = lines[0].strip()
        body = lines[1] if len(lines) > 1 else ""

        # Further split by ### headers (h3) if section is too long
        subsections = re.split(r'\n### ', body)

        if len(subsections) == 1:
            # No subsections, use as single chunk
            clean_content = clean_markdown(body)
            if len(clean_content) > 100:  # Only include substantial chunks
                chunks.append({
                    'doc_id': doc_id,
                    'section': title,
                    'content': f"{title}\n\n{clean_content}"
                })
        else:
            # Has subsections
            for j, subsection in enumerate(subsections):
                if j == 0:
                    # Text before first subsection
                    intro = subsection.strip()
                    if len(intro) > 100:
                        chunks.append({
                            'doc_id': doc_id,
                            'section': title,
                            'content': f"{title}\n\n{intro}"
                        })
                else:
                    # Subsection with title
                    sub_lines = subsection.split('\n', 1)
                    subtitle = sub_lines[0].strip()
                    sub_body = sub_lines[1] if len(sub_lines) > 1 else ""

                    clean_content = clean_markdown(sub_body)
                    if len(clean_content) > 100:
                        chunks.append({
                            'doc_id': doc_id,
                            'section': f"{title} - {subtitle}",
                            'content': f"{subtitle}\n\n{clean_content}"
                        })

    return chunks

def clean_markdown(text: str) -> str:
    """Remove markdown formatting but keep content readable."""
    # Remove code blocks (they're not useful for semantic search)
    text = re.sub(r'```[\s\S]*?```', '', text)

    # Remove links but keep link text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove images
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)

    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove bold/italic markers
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)

    # Remove headers markdown
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text

def generate_embedding(text: str) -> List[float]:
    """Generate embedding using OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def main():
    parser = argparse.ArgumentParser(description='Regenerate Knowledge Base')
    parser.add_argument('--dry-run', action='store_true', help='Preview chunks without uploading')
    parser.add_argument('--clear-existing', action='store_true', help='Delete existing chunks first')
    args = parser.parse_args()

    print("=" * 80)
    print("KNOWLEDGE BASE REGENERATION")
    print("=" * 80)

    # Find all markdown files in data/
    data_dir = 'data'
    md_files = [f for f in os.listdir(data_dir) if f.endswith('.md') and f.startswith('noah_')]

    print(f"\nFound {len(md_files)} markdown files:")
    for f in md_files:
        print(f"  - {f}")

    # Parse all files into chunks
    all_chunks = []
    for md_file in md_files:
        filepath = os.path.join(data_dir, md_file)
        chunks = parse_markdown_sections(filepath)
        all_chunks.extend(chunks)
        print(f"\n{md_file}: {len(chunks)} chunks")

    print(f"\n{'=' * 80}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"{'=' * 80}")

    # Preview chunks
    print("\nSAMPLE CHUNKS:")
    for i, chunk in enumerate(all_chunks[:3], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Doc ID: {chunk['doc_id']}")
        print(f"Section: {chunk['section']}")
        print(f"Content ({len(chunk['content'])} chars):")
        print(f"{chunk['content'][:300]}...")

    if args.dry_run:
        print("\n[DRY RUN] Not uploading to Supabase")
        print(f"Would create {len(all_chunks)} chunks")
        return

    # Confirm before proceeding
    print(f"\n{'=' * 80}")
    if args.clear_existing:
        print("⚠️  WARNING: This will DELETE existing KB and replace with new chunks")
    else:
        print("This will ADD new chunks to existing KB")
    print(f"{'=' * 80}")

    confirm = input("\nProceed? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("Aborted.")
        return

    # Clear existing if requested
    if args.clear_existing:
        print("\n🗑️  Deleting existing chunks...")
        # Get doc_ids we're replacing
        doc_ids = list(set(c['doc_id'] for c in all_chunks))
        for doc_id in doc_ids:
            result = supabase.table('kb_chunks').delete().eq('doc_id', doc_id).execute()
            print(f"   Deleted chunks for {doc_id}")

    # Generate embeddings and upload
    print(f"\n🔄 Generating embeddings and uploading...")
    for i, chunk in enumerate(all_chunks, 1):
        print(f"  [{i}/{len(all_chunks)}] {chunk['doc_id']} - {chunk['section'][:50]}...", end='')

        # Generate embedding
        embedding = generate_embedding(chunk['content'])

        # Upload to Supabase
        supabase.table('kb_chunks').insert({
            'doc_id': chunk['doc_id'],
            'section': chunk['section'],
            'content': chunk['content'],
            'embedding': embedding
        }).execute()

        print(" ✅")

    print(f"\n{'=' * 80}")
    print(f"✅ SUCCESS! Created {len(all_chunks)} high-quality KB chunks")
    print(f"{'=' * 80}")
    print("\nNext step: Test retrieval quality")
    print("  python3 test_rpc_performance.py")

if __name__ == '__main__':
    main()
