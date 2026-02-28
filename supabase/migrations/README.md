# Database Migrations

This directory contains SQL migration files for setting up the Supabase database schema.

## How to Run Migrations

1. **Go to your Supabase Dashboard**
   - Navigate to https://app.supabase.com
   - Select your project

2. **Open SQL Editor**
   - Click "SQL Editor" in the left sidebar
   - Click "New Query"

3. **Run migrations in order**:
   - `001_initial_schema.sql` - Core tables (kb_chunks, messages, feedback, etc.)
   - `002_add_confessions_and_sms.sql` - Confessions and SMS tracking tables

4. **Copy/paste and execute**
   - Copy the entire contents of each .sql file
   - Paste into the SQL Editor
   - Click "Run" or press Cmd/Ctrl + Enter

## Migration Files

### 001_initial_schema.sql
**Status**: ✅ Should be run first

Creates:
- `kb_chunks` - Vector embeddings for RAG
- `messages` - Chat interaction logs
- `retrieval_logs` - RAG pipeline tracking
- `links` - External URLs (LinkedIn, GitHub, etc.)
- `feedback` - User ratings and contact requests
- pgvector extension
- RLS policies
- Indexes for performance

### 002_add_confessions_and_sms.sql
**Status**: ⚠️ **REQUIRED** for `/api/confess` and `/api/feedback` endpoints

Creates:
- `confessions` - Anonymous crush confessions
- `sms_logs` - Twilio SMS notification tracking
- Adds missing columns to `feedback` table (`user_name`, `user_phone`, `user_email`)
- Analytics views (`recent_confessions`, `pending_contacts`)

**Current Issue**: If you're seeing `404 Not Found` errors on:
- `/api/confess` → Missing `confessions` table
- `/api/feedback` → Missing columns in `feedback` table

**Fix**: Run migration `002_add_confessions_and_sms.sql` immediately.

## Verifying Migrations

After running migrations, verify tables exist:

```sql
-- Check if all tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_type = 'BASE TABLE'
ORDER BY table_name;
```

Expected tables:
- `confessions` ✅
- `feedback` ✅
- `kb_chunks` ✅
- `links` ✅
- `messages` ✅
- `retrieval_logs` ✅
- `sms_logs` ✅

## Rollback (if needed)

To drop tables created by a migration:

```sql
-- Drop tables from migration 002
DROP TABLE IF EXISTS sms_logs CASCADE;
DROP TABLE IF EXISTS confessions CASCADE;

-- Drop tables from migration 001
DROP TABLE IF EXISTS retrieval_logs CASCADE;
DROP TABLE IF EXISTS feedback CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS links CASCADE;
DROP TABLE IF EXISTS kb_chunks CASCADE;
```

**Warning**: This will delete all data. Only use in development.

## Testing Migrations Locally

If you have Supabase CLI installed:

```bash
# Start local Supabase
supabase start

# Apply migrations
supabase db reset

# Check status
supabase db diff
```

## Production Deployment Checklist

Before deploying to Vercel:

- [ ] Run `001_initial_schema.sql` in production Supabase
- [ ] Run `002_add_confessions_and_sms.sql` in production Supabase
- [ ] Verify all tables exist (run verification query above)
- [ ] Run `python scripts/migrate_data_to_supabase.py` to populate kb_chunks
- [ ] Test API endpoints locally first
- [ ] Deploy to Vercel
- [ ] Test `/api/chat`, `/api/feedback`, `/api/confess` in production

## Troubleshooting

**Problem**: `relation "confessions" does not exist`
**Solution**: Run migration `002_add_confessions_and_sms.sql`

**Problem**: `column "user_name" does not exist in feedback`
**Solution**: Run migration `002_add_confessions_and_sms.sql` (it adds missing columns)

**Problem**: `permission denied for table confessions`
**Solution**: Check RLS policies, ensure you're using `service_role` key not `anon` key

**Problem**: Duplicate key errors when running migrations twice
**Solution**: Migrations use `IF NOT EXISTS` and `ON CONFLICT DO NOTHING`, safe to re-run
