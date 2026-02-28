#!/usr/bin/env python3
"""Create the crush_confessions table in Supabase"""

from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

print("=" * 80)
print("CREATING CRUSH_CONFESSIONS TABLE")
print("=" * 80)

# SQL to create the table
create_table_sql = """
CREATE TABLE IF NOT EXISTS crush_confessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    anonymous BOOLEAN NOT NULL DEFAULT true,
    name TEXT,
    contact TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_crush_confessions_session_id ON crush_confessions(session_id);
CREATE INDEX IF NOT EXISTS idx_crush_confessions_timestamp ON crush_confessions(timestamp DESC);
"""

try:
    # Execute the SQL using Supabase's RPC or direct query
    # Note: Supabase Python client doesn't have direct SQL execution
    # So we'll try using the postgrest API

    print("\nAttempting to create table...")
    print("(This requires database admin access via service role key)")

    # Try to query the table to see if it exists
    try:
        result = supabase.table('crush_confessions').select('id', count='exact').limit(1).execute()
        print(f"\n✅ Table 'crush_confessions' already exists with {result.count} rows")

    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg or "relation" in error_msg:
            print("\n⚠️  Table does not exist yet")
            print("\nTo create the table, run this SQL in Supabase SQL Editor:")
            print("\n" + "=" * 80)
            print(create_table_sql)
            print("=" * 80)
            print("\nOr use the Supabase CLI:")
            print("  supabase db push")
        else:
            print(f"\n❌ Error checking table: {e}")

except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "=" * 80)
print("INSTRUCTIONS")
print("=" * 80)
print("\nIf the table doesn't exist, create it by:")
print("1. Go to your Supabase project dashboard")
print("2. Navigate to SQL Editor")
print("3. Paste the SQL above")
print("4. Click 'Run'")
print("\nOr use the migration file:")
print("  File: supabase/migrations/003_crush_confessions_table.sql")
