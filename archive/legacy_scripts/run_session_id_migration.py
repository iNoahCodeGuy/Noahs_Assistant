"""
Session ID Migration - Simple Fix
Changes session_id from UUID to TEXT in messages table
"""

import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, 'src')
load_dotenv()

print("=" * 70)
print("🔧 SESSION ID TYPE MIGRATION")
print("=" * 70)

try:
    from config.supabase_config import get_supabase_client, supabase_settings
    import httpx

    supabase = get_supabase_client()

    print("\n📋 What This Does:")
    print("  Changes messages.session_id from UUID to TEXT")
    print("  Allows custom session ID formats like 'session_xxx_timestamp'")

    # Test current state
    print("\n🔍 Testing current state...")
    try:
        result = supabase.table('messages').select('id, session_id').limit(1).execute()
        if result.data:
            print(f"✅ Found {len(result.data)} existing message(s)")
        else:
            print("⚠️  No messages in table yet")
    except Exception as e:
        print(f"⚠️  Error reading table: {e}")

    # Since Supabase Python client can't execute raw SQL, use httpx
    print("\n🔄 Attempting migration via Supabase API...")

    # Prepare SQL statements
    sql_statements = [
        "ALTER TABLE messages ALTER COLUMN session_id TYPE TEXT;",
        "DROP INDEX IF EXISTS messages_session_id_idx;",
        "CREATE INDEX IF NOT EXISTS messages_session_id_idx ON messages (session_id);"
    ]

    print("\n⚠️  IMPORTANT: Supabase Python SDK doesn't support raw SQL execution")
    print("   You need to run this migration in the Supabase SQL Editor")
    print("\n📝 Copy this SQL to Supabase Dashboard → SQL Editor:\n")
    print("-" * 70)
    for sql in sql_statements:
        print(sql)
    print("-" * 70)

    print("\n🌐 Open this URL:")
    print("   https://supabase.com/dashboard/project/tjnlusesinzzlwvlbnnm/sql/new")

    print("\n✅ Steps:")
    print("   1. Open the link above (or go to Supabase Dashboard)")
    print("   2. Click 'SQL Editor' if not already there")
    print("   3. Copy the SQL statements above")
    print("   4. Paste into the editor")
    print("   5. Click 'RUN' button")
    print("   6. Come back here and press Enter when done!")

    input("\n⏸️  Press Enter after running the migration in Supabase...")

    # Verify migration worked
    print("\n✅ Testing if migration worked...")

    test_session_id = f"test_migration_{int(os.urandom(4).hex(), 16)}"

    try:
        test_insert = supabase.table('messages').insert({
            'session_id': test_session_id,  # Custom format (not UUID!)
            'role_mode': 'test',
            'query': 'migration verification test',
            'answer': 'testing if TEXT type works',
            'query_type': 'test',
            'latency_ms': 0,
            'success': True
        }).execute()

        print(f"✅ SUCCESS! Custom session_id accepted: {test_session_id}")

        # Clean up
        if test_insert.data:
            test_id = test_insert.data[0]['id']
            supabase.table('messages').delete().eq('id', test_id).execute()
            print("✅ Test record cleaned up")

        print("\n" + "=" * 70)
        print("🎉 MIGRATION SUCCESSFUL!")
        print("=" * 70)
        print("\n✅ Your Vercel app will now work correctly!")
        print("   Session IDs like 'session_4t1jo080e_1760203305367' are now accepted")
        print("\n🧪 Next: Test at noahsaiassistant.vercel.app")

    except Exception as e:
        error_str = str(e)
        if '22P02' in error_str or 'uuid' in error_str.lower():
            print(f"❌ Migration NOT applied yet!")
            print(f"   Error: {e}")
            print("\n⚠️  Please run the SQL in Supabase SQL Editor first")
        else:
            print(f"⚠️  Unexpected error: {e}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
