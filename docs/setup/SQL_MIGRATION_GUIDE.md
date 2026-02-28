# SQL Migration Guide - Analytics Helpers

**IMPORTANT:** Run these SQL commands in Supabase Dashboard before testing the live analytics feature.

## Steps

### 1. Open Supabase Dashboard
Navigate to: https://supabase.com/dashboard

### 2. Go to SQL Editor
- Click on your project (noahs-ai-assistant or similar)
- Click "SQL Editor" in the left sidebar
- Click "New Query"

### 3. Copy and Paste SQL
Copy the entire content from:
```
supabase/migrations/003_analytics_helpers.sql
```

### 4. Run Migration
- Paste into the SQL Editor
- Click "Run" or press Ctrl+Enter
- Wait for success message

### 5. Verify Functions
Run this test query:
```sql
-- Test kb_coverage_summary
select * from kb_coverage_summary();

-- Test low_similarity_queries (should return empty if no low-similarity queries)
select * from low_similarity_queries(7, 5);

-- Test conversion_by_role
select * from conversion_by_role(30);

-- Test performance_summary
select * from performance_summary_7d();

-- Test tool stats (may error if table doesn't exist yet - that's okay)
select * from tool_invocation_stats(7);
```

## Expected Results

✅ **Success:**
```
5 functions created successfully
Permissions granted
```

❌ **Common Errors:**

### Error: "function already exists"
**Solution:** Functions were already created. Safe to ignore.

### Error: "permission denied"
**Solution:** Make sure you're logged in as the project owner or have sufficient permissions.

### Error: "table does not exist"
**Solution:** 
- For `tool_invocations` table, this is optional (will be created automatically on first use)
- For core tables (`messages`, `retrieval_logs`, `feedback`), run migrations 001 and 002 first

---

## Manual Function Creation (if needed)

If the migration file fails, you can create functions one by one:

### 1. KB Coverage Summary
```sql
create or replace function kb_coverage_summary()
returns table (source text, count int)
language sql stable as $$
  select section as source, count(*)::int
  from kb_chunks
  group by section
  order by count desc;
$$;

grant execute on function kb_coverage_summary() to authenticated, anon, service_role;
```

### 2. Low Similarity Queries
```sql
create or replace function low_similarity_queries(days_back int default 7, result_limit int default 20)
returns table (
  message_id uuid,
  user_query text,
  avg_similarity float,
  created_at timestamptz
)
language sql stable as $$
  select 
    m.id as message_id,
    m.user_query,
    avg(r.similarity_score) as avg_similarity,
    m.created_at
  from messages m
  join retrieval_logs r on r.message_id = m.id
  where m.created_at > now() - (days_back || ' days')::interval
  group by m.id, m.user_query, m.created_at
  having avg(r.similarity_score) < 0.60
  order by avg_similarity asc
  limit result_limit;
$$;

grant execute on function low_similarity_queries(int, int) to authenticated, anon, service_role;
```

### 3. Conversion by Role
```sql
create or replace function conversion_by_role(days_back int default 30)
returns table (
  role_mode text,
  sessions bigint,
  conversions bigint,
  conversion_rate numeric
)
language sql stable as $$
  select 
    m.role_mode,
    count(distinct m.session_id) as sessions,
    count(distinct case when f.contact_requested then m.session_id end) as conversions,
    round(
      100.0 * count(distinct case when f.contact_requested then m.session_id end) 
      / nullif(count(distinct m.session_id), 0), 
      1
    ) as conversion_rate
  from messages m
  left join feedback f on f.message_id = m.id
  where m.created_at > now() - (days_back || ' days')::interval
  group by m.role_mode
  order by conversion_rate desc;
$$;

grant execute on function conversion_by_role(int) to authenticated, anon, service_role;
```

### 4. Performance Summary
```sql
create or replace function performance_summary_7d()
returns table (
  metric text,
  value numeric
)
language sql stable as $$
  select 'total_messages' as metric, count(*)::numeric as value
  from messages
  where created_at > now() - interval '7 days'
  
  union all
  
  select 'p95_latency_ms', percentile_cont(0.95) within group (order by latency_ms)
  from messages
  where created_at > now() - interval '7 days' and latency_ms is not null
  
  union all
  
  select 'avg_latency_ms', avg(latency_ms)
  from messages
  where created_at > now() - interval '7 days' and latency_ms is not null
  
  union all
  
  select 'success_rate', round(100.0 * count(case when success then 1 end) / nullif(count(*), 0), 1)
  from messages
  where created_at > now() - interval '7 days'
  
  union all
  
  select 'grounded_rate', round(100.0 * count(case when grounded then 1 end) / nullif(count(*), 0), 1)
  from retrieval_logs r
  join messages m on m.id = r.message_id
  where m.created_at > now() - interval '7 days' and r.grounded is not null
  
  union all
  
  select 'avg_rating', avg(rating)
  from feedback f
  join messages m on m.id = f.message_id
  where m.created_at > now() - interval '7 days' and f.rating is not null;
$$;

grant execute on function performance_summary_7d() to authenticated, anon, service_role;
```

### 5. Tool Invocation Stats
```sql
create or replace function tool_invocation_stats(days_back int default 7)
returns table (
  tool text,
  invocations bigint,
  success_rate numeric,
  avg_duration_ms numeric
)
language sql stable as $$
  select 
    tool,
    count(*) as invocations,
    round(100.0 * count(case when status = 'success' then 1 end) / nullif(count(*), 0), 1) as success_rate,
    round(avg(duration_ms), 1) as avg_duration_ms
  from tool_invocations
  where created_at > now() - (days_back || ' days')::interval
  group by tool
  order by invocations desc;
$$;

grant execute on function tool_invocation_stats(int) to authenticated, anon, service_role;
```

---

## Testing the Migration

After running the migration, test the `/api/analytics` endpoint:

```bash
curl https://noahsaiassistant.vercel.app/api/analytics
```

Expected response:
```json
{
  "inventory": {
    "messages": 123,
    "retrieval_logs": 456,
    ...
  },
  "messages": {
    "data": [...]
  },
  "kb_coverage": [
    {"source": "career", "count": 12}
  ],
  "generated_at": "2025-01-30T..."
}
```

---

## Rollback (if needed)

To remove the functions:

```sql
drop function if exists kb_coverage_summary();
drop function if exists low_similarity_queries(int, int);
drop function if exists conversion_by_role(int);
drop function if exists performance_summary_7d();
drop function if exists tool_invocation_stats(int);
```

---

## Support

If you encounter issues:
1. Check Supabase logs (Dashboard → Logs)
2. Verify table schema matches expected columns
3. Ensure service role key is set in Vercel environment variables
4. Test individual functions with sample queries above
