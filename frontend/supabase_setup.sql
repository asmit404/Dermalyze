-- ============================================================
-- Dermalyze — Supabase setup
-- Run this in the Supabase SQL Editor (project → SQL Editor → New query)
-- ============================================================


-- ── 1. analyses table ────────────────────────────────────────────────────────
create table if not exists public.analyses (
  id                  uuid primary key default gen_random_uuid(),
  user_id             uuid not null references auth.users(id) on delete cascade,
  created_at          timestamptz not null default now(),
  image_url           text,
  gradcam_image_url   text,                -- Grad-CAM explainability heatmap (nullable for backward compatibility)
  predicted_class_id  text not null,       -- e.g. "mel"
  predicted_class_name text not null,      -- e.g. "Melanoma"
  confidence          numeric(5,2) not null, -- e.g. 67.40
  all_scores          jsonb,               -- [{"id":"mel","name":"Melanoma","score":67.4}, ...]
  notes               text,                -- free-text clinician note (nullable)
  metadata            jsonb                -- patient metadata: {age_approx, sex, anatom_site} (nullable)
);

-- Index for fast per-user history queries
create index if not exists analyses_user_id_created_at_idx
  on public.analyses (user_id, created_at desc);

-- Migration: add notes column to existing databases (safe to re-run)
alter table public.analyses add column if not exists notes text;

-- Migration: add gradcam_image_url column for explainability heatmaps (safe to re-run)
alter table public.analyses add column if not exists gradcam_image_url text;

-- Migration: add metadata column for patient information captured at analysis time (safe to re-run)
alter table public.analyses add column if not exists metadata jsonb;


-- ── 2. Row Level Security ─────────────────────────────────────────────────────
alter table public.analyses enable row level security;

-- Users can only read their own records
create policy "Users can read own analyses"
  on public.analyses for select
  using (auth.uid() = user_id);

-- Users can insert only for themselves
create policy "Users can insert own analyses"
  on public.analyses for insert
  with check (auth.uid() = user_id);

-- Users can delete their own records (optional, for future "clear history")
create policy "Users can delete own analyses"
  on public.analyses for delete
  using (auth.uid() = user_id);

-- Users can update only their own records
create policy "Users can update own analyses"
  on public.analyses for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);


-- ── 3. Storage bucket ─────────────────────────────────────────────────────────
-- Run this AFTER creating the bucket in Storage → New bucket
-- Bucket name: analysis-images   (set to Private)

-- Allow authenticated users to upload their own images
create policy "Users can upload own images"
  on storage.objects for insert
  to authenticated
  with check (
    bucket_id = 'analysis-images'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

-- Allow authenticated users to read their own images
create policy "Users can read own images"
  on storage.objects for select
  to authenticated
  using (
    bucket_id = 'analysis-images'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

-- Allow authenticated users to delete their own images
create policy "Users can delete own images"
  on storage.objects for delete
  to authenticated
  using (
    bucket_id = 'analysis-images'
    and (storage.foldername(name))[1] = auth.uid()::text
  );


-- ── 4. Dashboard stats RPC ───────────────────────────────────────────────────
-- Returns aggregated stats for the current user in a single round-trip.
-- Uses auth.uid() internally — no user-id parameter needed, cannot be
-- called on behalf of another user.
create or replace function get_dashboard_stats()
returns json language sql stable security definer
set search_path = public, pg_temp as $$
  select json_build_object(
    'total',          count(*),
    'this_month',     count(*) filter (where created_at >= date_trunc('month', now())),
    'avg_confidence', round(avg(confidence)::numeric, 1),
    'needs_review',   count(*) filter (where predicted_class_id in ('mel','bcc','akiec')),
    'class_counts', (
      select json_agg(row_to_json(t))
      from (
        select
          predicted_class_id   as id,
          predicted_class_name as name,
          count(*)::int        as count
        from analyses
        where user_id = auth.uid()
        group by predicted_class_id, predicted_class_name
        order by count desc
      ) t
    )
  )
  from analyses
  where user_id = auth.uid();
$$;