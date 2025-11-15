#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert technical_kb.csv from third person to first person.
Transforms "This AI assistant", "The product", "The system" -> "I", "My system", etc.
"""

import csv
import re
from pathlib import Path


def convert_to_first_person(text: str) -> str:
    """Convert third-person text to first-person perspective."""
    if not text:
        return text

    # Define replacement patterns (order matters - most specific first)
    replacements = [
        # Specific product references
        (r'\bThis AI assistant is\b', 'I\'m'),
        (r'\bThis AI assistant was\b', 'I was'),
        (r'\bThis AI assistant\b', 'I'),
        (r'\bthe AI assistant\b', 'I'),
        (r'\ban AI assistant\b', 'an assistant'),

        # Product references
        (r'\bThe product is\b', 'I\'m'),
        (r'\bThe product uses\b', 'I use'),
        (r'\bThe product was\b', 'I was'),
        (r'\bThe product\b', 'I'),
        (r'\bthis product is\b', 'I\'m'),
        (r'\bthis product uses\b', 'I use'),
        (r'\bthis product\b', 'I'),

        # System references
        (r'\bThe RAG system is\b', 'My RAG system is'),
        (r'\bThe RAG system follows\b', 'My RAG system follows'),
        (r'\bThe RAG system\b', 'My RAG system'),
        (r'\bThe system is\b', 'I\'m'),
        (r'\bThe system uses\b', 'I use'),
        (r'\bThe system follows\b', 'I follow'),
        (r'\bThe system implements\b', 'I implement'),
        (r'\bThe system tracks\b', 'I track'),
        (r'\bThe system works\b', 'I work'),
        (r'\bThe system includes\b', 'I include'),
        (r'\bThe system has\b', 'I have'),
        (r'\bThe system\b', 'I'),

        # Architecture references
        (r'\bThe architecture is\b', 'My architecture is'),
        (r'\bThe architecture follows\b', 'My architecture follows'),
        (r'\bThe architecture\b', 'My architecture'),

        # Backend/Database references
        (r'\bThe backend is\b', 'My backend is'),
        (r'\bThe backend uses\b', 'My backend uses'),
        (r'\bThe backend\b', 'My backend'),
        (r'\bThe database is\b', 'My database is'),
        (r'\bThe database uses\b', 'My database uses'),
        (r'\bThe database\b', 'My database'),

        # Code/Codebase references
        (r'\bThe codebase is\b', 'My codebase is'),
        (r'\bThe codebase includes\b', 'My codebase includes'),
        (r'\bThe codebase\b', 'My codebase'),

        # Portfolia references (when describing self)
        (r'\bPortfolia is\b', 'I\'m'),
        (r'\bPortfolia uses\b', 'I use'),
        (r'\bPortfolia implements\b', 'I implement'),
        (r'\bPortfolia tracks\b', 'I track'),
        (r'\bPortfolia works\b', 'I work'),
        (r'\bPortfolia\b', 'I'),

        # Noah references (keep third person for talking about Noah)
        # These are intentionally NOT converted - we want to preserve
        # "Noah built this system" structure

        # General "it" replacements when referring to the system
        # (Only safe when clearly referring to the assistant, not components)
    ]

    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Fix capitalization issues from replacements
    # "I'm" at start of sentence should stay capitalized
    # "i" in middle of sentence should stay lowercase (but after period should be "I")

    return result


def process_csv_file(input_path: Path, output_path: Path):
    """Read CSV, convert answers to first person, write back."""
    rows = []

    # Read all rows
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            # Convert the Answer field only (keep Question unchanged)
            if 'Answer' in row:
                row['Answer'] = convert_to_first_person(row['Answer'])
            rows.append(row)

    # Write converted rows
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Converted {len(rows)} entries from third person to first person")
    print(f"📄 Output: {output_path}")


def main():
    # Get script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # File paths
    input_file = project_root / "data" / "technical_kb.csv"
    output_file = project_root / "data" / "technical_kb.csv"  # Overwrite in place
    backup_file = project_root / "data" / "technical_kb_backup.csv"  # Safety backup

    # Create backup first
    if input_file.exists():
        import shutil
        shutil.copy(input_file, backup_file)
        print(f"📦 Created backup: {backup_file}")

    # Process file
    process_csv_file(input_file, output_file)

    print("\n🎯 Next steps:")
    print("1. Review the changes in data/technical_kb.csv")
    print("2. Run: python scripts/migrate_data_to_supabase.py")
    print("3. Restart LangGraph server to pick up new embeddings")


if __name__ == "__main__":
    main()
