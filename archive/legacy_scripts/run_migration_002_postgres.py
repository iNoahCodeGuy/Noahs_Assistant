#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Migration 002 using direct PostgreSQL connection.
Alternative method that uses psycopg2 instead of supabase-py.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")

def run_migration_via_postgres():
    """Execute migration using direct PostgreSQL connection."""
    print_header("Running Migration 002 via PostgreSQL")

    # Get Supabase connection details
    supabase_url = os.getenv('SUPABASE_URL')
    service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

    if not supabase_url or not service_key:
        print("ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
        return False

    # Extract database host from Supabase URL
    # Format: https://xxx.supabase.co
    project_ref = supabase_url.replace('https://', '').replace('.supabase.co', '')
    db_host = 'db.{}.supabase.co'.format(project_ref)

    print("Database host: {}".format(db_host))
    print("Project: {}".format(project_ref))

    # Read migration file
    try:
        with open('supabase/migrations/002_add_confessions_and_sms.sql', 'r') as f:
            sql_content = f.read()
        print("✓ Migration file loaded\n")
    except FileNotFoundError:
        print("ERROR: Migration file not found")
        return False

    # Try to import psycopg2
    try:
        import psycopg2
        print("✓ psycopg2 available")
    except ImportError:
        print("ERROR: psycopg2 not installed")
        print("\nInstall with:")
        print("  pip install psycopg2-binary")
        print("\nOr run migration manually in Supabase SQL Editor")
        return False

    # Database password - for Supabase, use the service role key
    db_password = service_key

    print("\nAttempting to connect to database...")
    print("Note: This requires database password (not just API key)")
    print("\nIf connection fails, please run migration manually:")
    print("  1. Go to https://app.supabase.com")
    print("  2. SQL Editor → New Query")
    print("  3. Copy/paste: supabase/migrations/002_add_confessions_and_sms.sql")
    print("  4. Click Run\n")

    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=db_host,
            port=5432,
            database='postgres',
            user='postgres',
            password=db_password
        )

        cur = conn.cursor()
        print("✓ Connected to database\n")

        # Execute migration
        print("Executing migration SQL...")
        print("-" * 60)

        cur.execute(sql_content)
        conn.commit()

        print("✓ Migration executed successfully!")

        cur.close()
        conn.close()

        return True

    except Exception as e:
        print("ERROR: {}".format(str(e)))
        print("\nPlease run migration manually (see instructions above)")
        return False

def main():
    """Main execution."""
    print_header("Automated Migration 002 Executor")

    print("This script will attempt to run the migration automatically.")
    print("If it fails, you can run it manually in Supabase SQL Editor.\n")

    input("Press Enter to continue or Ctrl+C to cancel...")

    success = run_migration_via_postgres()

    if success:
        print_header("SUCCESS! 🎉")
        print("Run verification: python3 scripts/verify_migration.py")
        return 0
    else:
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(130)
