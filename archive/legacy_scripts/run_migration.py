#!/usr/bin/env python3
"""
Simple wrapper to run migration with proper environment setup.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now run the migration
if __name__ == "__main__":
    # Import and run migration directly
    exec(open('scripts/migrate_data_to_supabase.py').read())
