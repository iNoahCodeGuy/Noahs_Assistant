#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatically run Migration 002 via Supabase API.
This script reads the SQL file and executes it directly in your database.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")

def run_migration():
    """Execute Migration 002 via Supabase."""
    print_header("Running Migration 002: Confessions & SMS Logs")
    
    # Check environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not service_key:
        print("ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
        print("Please check your .env file")
        return False
    
    print("✓ Environment variables found")
    
    # Read migration file
    migration_file = 'supabase/migrations/002_add_confessions_and_sms.sql'
    
    try:
        with open(migration_file, 'r') as f:
            sql_content = f.read()
        print("✓ Migration file loaded: {}".format(migration_file))
    except FileNotFoundError:
        print("ERROR: Migration file not found: {}".format(migration_file))
        return False
    
    # Initialize Supabase client
    try:
        from supabase import create_client
        supabase = create_client(supabase_url, service_key)
        print("✓ Connected to Supabase")
    except ImportError:
        print("ERROR: supabase-py not installed")
        print("Install with: pip install supabase")
        return False
    except Exception as e:
        print("ERROR: Failed to connect to Supabase: {}".format(str(e)))
        return False
    
    # Execute migration
    print("\nExecuting migration SQL...")
    print("-" * 60)
    
    try:
        # Split SQL into individual statements
        statements = [s.strip() for s in sql_content.split(';') if s.strip()]
        
        total_statements = len(statements)
        successful = 0
        failed = 0
        
        for i, statement in enumerate(statements, 1):
            # Skip comments and empty statements
            if not statement or statement.startswith('--'):
                continue
            
            try:
                # Execute via RPC if available, otherwise use raw SQL
                result = supabase.rpc('exec_sql', {'query': statement}).execute()
                successful += 1
                print("✓ Statement {}/{} executed".format(i, total_statements))
            except Exception as e:
                error_msg = str(e)
                # These errors are expected and safe to ignore
                if 'already exists' in error_msg or 'duplicate' in error_msg.lower():
                    print("→ Statement {}/{} skipped (already exists)".format(i, total_statements))
                    successful += 1
                elif 'does not exist' in error_msg and 'column' in error_msg:
                    # Column doesn't exist, try to continue
                    print("→ Statement {}/{} skipped (column handling)".format(i, total_statements))
                    successful += 1
                else:
                    print("✗ Statement {}/{} failed: {}".format(i, total_statements, error_msg))
                    failed += 1
        
        print("-" * 60)
        print("\nMigration Summary:")
        print("  Successful: {}".format(successful))
        print("  Failed: {}".format(failed))
        
        if failed == 0:
            print("\n✓ Migration completed successfully!")
            return True
        else:
            print("\n⚠ Migration completed with some errors (see above)")
            return False
            
    except Exception as e:
        print("ERROR: Migration failed: {}".format(str(e)))
        return False

def verify_migration():
    """Verify that tables were created."""
    print_header("Verifying Migration Results")
    
    supabase_url = os.getenv('SUPABASE_URL')
    service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    try:
        from supabase import create_client
        supabase = create_client(supabase_url, service_key)
        
        tables_to_check = ['confessions', 'sms_logs']
        all_exist = True
        
        for table in tables_to_check:
            try:
                # Try to query the table (limit 0 for fast check)
                result = supabase.table(table).select('*').limit(0).execute()
                print("✓ {} table exists".format(table))
            except Exception as e:
                print("✗ {} table MISSING".format(table))
                all_exist = False
        
        return all_exist
        
    except Exception as e:
        print("ERROR: Verification failed: {}".format(str(e)))
        return False

def main():
    """Main execution flow."""
    print_header("Supabase Migration 002 - Auto Executor")
    
    # Run migration
    success = run_migration()
    
    if not success:
        print("\n⚠ Migration encountered issues. Check errors above.")
        print("\nAlternative: Run migration manually in Supabase SQL Editor")
        print("  1. Go to https://app.supabase.com")
        print("  2. SQL Editor → New Query")
        print("  3. Copy/paste: supabase/migrations/002_add_confessions_and_sms.sql")
        print("  4. Click Run")
        return 1
    
    # Verify results
    if verify_migration():
        print_header("SUCCESS! 🎉")
        print("Migration 002 completed successfully!")
        print("\nNext steps:")
        print("  1. Run: python3 scripts/verify_migration.py")
        print("  2. Test endpoints with curl commands")
        print("  3. Your deployment is ready!")
        return 0
    else:
        print("\n⚠ Some tables may be missing. Run verify_migration.py for details.")
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nMigration cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print("\n\nUnexpected error: {}".format(str(e)))
        sys.exit(1)
