-- SQL Helpers for Analytics Dashboard
-- Run these in Supabase SQL Editor to enable advanced analytics features

-- 1. KB Coverage Summary (groups chunks by section)
create or replace function kb_coverage_summary()
returns table (source text, count int)
language sql stable as $$
  select section as source, count(*)::int
  from kb_chunks
  group by section
  order by count desc;
$$;

-- 2. Low-Similarity Spotlight (identifies queries with poor retrieval)
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

-- 3. Conversion by Role (analyzes contact requests by role)
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

-- 4. Performance Summary (7-day aggregate metrics)
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

-- 5. Tool Invocation Stats (if tool_invocations table exists)
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

-- Grant execute permissions (adjust role name as needed)
grant execute on function kb_coverage_summary() to authenticated, anon, service_role;
grant execute on function low_similarity_queries(int, int) to authenticated, anon, service_role;
grant execute on function conversion_by_role(int) to authenticated, anon, service_role;
grant execute on function performance_summary_7d() to authenticated, anon, service_role;
grant execute on function tool_invocation_stats(int) to authenticated, anon, service_role;
