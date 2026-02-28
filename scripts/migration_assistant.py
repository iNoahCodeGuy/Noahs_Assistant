#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper script to run Migration 002.
This will:
1. Show you the SQL to copy
2. Open your Supabase SQL Editor
3. Guide you through the process
"""

import os
import webbrowser
import sys

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")

def main():
    """Guide user through manual migration."""
    print_header("Migration 002 Setup Assistant")
    
    print("I'll help you run Migration 002 in 3 easy steps!\n")
    
    # Step 1: Read migration file
    migration_file = 'supabase/migrations/002_add_confessions_and_sms.sql'
    
    try:
        with open(migration_file, 'r') as f:
            sql_content = f.read()
    except FileNotFoundError:
        print("ERROR: Migration file not found: {}".format(migration_file))
        return 1
    
    print("Step 1: Open the migration file")
    print("-" * 60)
    print("File location: {}".format(migration_file))
    print("File size: {} lines".format(len(sql_content.splitlines())))
    print("\nOpening file in your default text editor...")
    
    # Try to open the file
    try:
        if sys.platform == 'darwin':  # macOS
            os.system('open "{}"'.format(migration_file))
        elif sys.platform == 'win32':  # Windows
            os.system('start "" "{}"'.format(migration_file))
        else:  # Linux
            os.system('xdg-open "{}"'.format(migration_file))
    except:
        pass
    
    input("\nPress Enter when ready for Step 2...")
    
    # Step 2: Open Supabase
    print("\n\nStep 2: Open Supabase SQL Editor")
    print("-" * 60)
    print("Opening: https://app.supabase.com")
    print("\nOnce there:")
    print("  1. Select your project")
    print("  2. Click 'SQL Editor' in left sidebar")
    print("  3. Click 'New Query' button")
    
    try:
        webbrowser.open('https://app.supabase.com')
        print("\n✓ Browser opened")
    except:
        print("\nPlease manually navigate to: https://app.supabase.com")
    
    input("\nPress Enter when you have SQL Editor open...")
    
    # Step 3: Instructions
    print("\n\nStep 3: Run the migration")
    print("-" * 60)
    print("Now:")
    print("  1. Copy ALL content from: {}".format(migration_file))
    print("  2. Paste into the Supabase SQL Editor")
    print("  3. Click 'Run' (or press Cmd/Ctrl + Enter)")
    print("\nYou should see: 'Success. No rows returned'")
    print("\n" + "="*60)
    
    input("\nPress Enter when migration is complete...")
    
    # Verify
    print("\n\nVerifying migration...")
    print("-" * 60)
    
    import subprocess
    try:
        result = subprocess.run(
            ['python3', 'scripts/verify_migration.py'],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print_header("SUCCESS! 🎉")
            print("Migration completed and verified!")
            print("\nYour deployment is now 100% ready!")
            print("\nTest your endpoints:")
            print("  curl -X POST https://noahsaiassistant.vercel.app/api/confess \\")
            print("    -H 'Content-Type: application/json' \\")
            print("    -d '{\"message\": \"test\", \"is_anonymous\": true}'")
            return 0
        else:
            print("\n⚠ Verification found issues. Run again:")
            print("  python3 scripts/verify_migration.py")
            return 1
            
    except Exception as e:
        print("Couldn't run verification automatically.")
        print("Please run: python3 scripts/verify_migration.py")
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(130)
