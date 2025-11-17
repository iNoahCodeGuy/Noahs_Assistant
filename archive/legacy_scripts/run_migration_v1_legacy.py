#!/usr/bin/env python3
"""
Run the Supabase database migration directly from Python.
This creates all the necessary tables without needing to copy/paste SQL.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

print("=" * 70)
print("🚀 RUNNING SUPABASE DATABASE MIGRATION")
print("=" * 70)

# Connect to Supabase
supabase_url = os.getenv("SUPABASE_URL")
service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

print(f"\n📡 Connecting to: {supabase_url}")
supabase = create_client(supabase_url, service_key)

# Read the migration SQL file
print("\n📖 Reading migration file...")
with open("supabase/migrations/001_initial_schema.sql", "r", encoding="utf-8") as f:
    sql_content = f.read()

print(f"   ✅ Loaded {len(sql_content)} characters of SQL")

# Execute the migration using Supabase's RPC
print("\n⚙️  Executing migration...")
print("   This will create:")
print("   • pgvector extension")
print("   • kb_chunks table (for knowledge base)")
print("   • messages table (for chat logs)")
print("   • retrieval_logs table (for RAG tracking)")
print("   • links table (for external resources)")
print("   • feedback table (for ratings)")
print("   • Helper functions and indexes")

try:
    # Use the postgrest client to execute raw SQL
    # Note: Supabase Python client doesn't have direct SQL execution
    # We'll use the REST API endpoint
    import httpx

    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    # Supabase doesn't expose direct SQL execution via REST API for security
    # We need to create tables one by one using the Python client

    print("\n⚠️  Note: Python client cannot execute raw SQL for security.")
    print("   Let's create tables using the REST API instead...\n")

    # Alternative: Use psycopg2 to connect directly to Postgres
    print("📦 Installing psycopg2 if needed...")
    import subprocess
    subprocess.run(["pip", "install", "-q", "psycopg2-binary"], check=False)

    import psycopg2

    # Extract connection details from Supabase URL
    # Format: https://xxxproject.supabase.co
    project_ref = supabase_url.split("//")[1].split(".")[0]

    # Supabase Postgres connection string
    # Format: postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
    print("\n🔐 To connect directly to Postgres, we need the database password.")
    print("   You can find this in:")
    print("   Supabase Dashboard → Settings → Database → Connection String")
    print("   Look for 'Connection pooling' URI\n")

    db_password = input("   Enter your Supabase database password (or press Enter to skip): ").strip()

    if db_password:
        conn_string = f"postgresql://postgres.{project_ref}:{db_password}@aws-0-us-west-1.pooler.supabase.com:6543/postgres"

        print("\n   🔌 Connecting to Postgres...")
        conn = psycopg2.connect(conn_string)
        cur = conn.cursor()

        print("   ✅ Connected! Executing migration...")
        cur.execute(sql_content)
        conn.commit()

        cur.close()
        conn.close()

        print("\n" + "=" * 70)
        print("🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📋 Next Step:")
        print("   python scripts/migrate_data_to_supabase.py")
        print("=" * 70)
    else:
        print("\n   ⏭️  Skipped. Please run the SQL manually in Supabase SQL Editor.")
        print("\n   📋 Quick Copy:")
        print("   1. Open Notepad with the SQL file (already open)")
        print("   2. Select ALL (Ctrl+A) and Copy (Ctrl+C)")
        print("   3. In Supabase SQL Editor, clear everything")
        print("   4. Paste (Ctrl+V) and click Run")

except Exception as e:
    print(f"\n   ❌ Error: {e}")
    print("\n   💡 Alternative: Run the SQL manually in Supabase SQL Editor")
    print("   The SQL file is open in Notepad - copy from there!")
