-- Project Tohjo: Master SQL Schema (V1 Premium MVP)

-- 1) Profiles (User management & billing)
CREATE TABLE IF NOT EXISTS public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text unique not null,
  full_name text,
  avatar_url text,
  plan text default 'free' check (plan in ('free', 'pro', 'enterprise')),
  credits integer default 5,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- 2) Projects (Grouping of songs/projects)
CREATE TABLE IF NOT EXISTS public.projects (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  mood text,                             -- optional "mood/style" text
  created_at timestamptz default now()
);

-- 3) Job Status Enum
DO $$ BEGIN
    CREATE TYPE public.job_status AS ENUM (
        'queued', 
        'preprocessing', 
        'transcribing', 
        'aligning', 
        'hook_selecting', 
        'compositing', 
        'encoding', 
        'completed', 
        'failed', 
        'canceled'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 4) Asset Kind Enum
DO $$ BEGIN
    CREATE TYPE public.asset_kind AS ENUM (
        'input_audio', 
        'vocal_stem', 
        'cleaned_vocals', 
        'analysis_json', 
        'hook_audio', 
        'draft_video', 
        'final_video'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 5) Jobs (Execution tracking)
CREATE TABLE IF NOT EXISTS public.jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  project_id uuid references public.projects(id) on delete set null,

  status public.job_status not null default 'queued',
  stage text,                            -- transcribing, aligning, etc.
  progress int not null default 0 check (progress >= 0 and progress <= 100),

  -- Input
  input_audio_asset_id uuid REFERENCES public.job_assets(id),
  input_audio_url text,                  -- optional, if user pasted external link

  -- Render request parameters
  duration_sec int not null default 30,
  format text not null default 'horizontal_16_9',
  fps int not null default 30,
  resolution text not null default '1080p',
  style_id text not null default 'abstract_lyric_v1',

  -- Hook selection metadata
  hook_start_sec numeric,
  hook_end_sec numeric,
  hook_score numeric,

  -- Quality metrics
  asr_confidence numeric,
  align_score numeric,
  consensus_confidence numeric,

  -- Output pointers
  output_video_asset_id uuid REFERENCES public.job_assets(id),
  analysis_json_asset_id uuid REFERENCES public.job_assets(id),
  error_code text,
  error_message text,

  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- 6) Job Assets (Structured file pointers)
CREATE TABLE IF NOT EXISTS public.job_assets (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  job_id uuid references public.jobs(id) on delete cascade,

  bucket text not null,                  -- 'assets' or 'outputs'
  path text not null,                    -- storage path (e.g., 'jobs/uuid/audio.mp3')
  kind public.asset_kind not null,
  mime_type text,
  bytes bigint,
  checksum_sha256 text,

  created_at timestamptz default now(),
  unique(bucket, path)
);

-- 7) Job Events (Observability/SSE log)
CREATE TABLE IF NOT EXISTS public.job_events (
  id uuid primary key default gen_random_uuid(),
  job_id uuid not null references public.jobs(id) on delete cascade,
  at timestamptz default now(),
  level text not null default 'info',    -- info, warn, error
  event_type text not null,              -- stage, progress, asset, metric, error
  payload jsonb not null default '{}'::jsonb
);

-- 8) Usage Ledger (Billing/Cost accounting)
CREATE TABLE IF NOT EXISTS public.usage_ledger (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  job_id uuid references public.jobs(id) on delete set null,
  stage text,
  unit text not null,                    -- seconds, usd_cents, tokens
  quantity numeric not null default 0,
  cost_cents int not null default 0,
  meta jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);

-- Auto Updated At Trigger
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_jobs_updated_at ON public.jobs;
CREATE TRIGGER trg_jobs_updated_at
BEFORE UPDATE ON public.jobs
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- RLS (Row Level Security)
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.job_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.usage_ledger ENABLE ROW LEVEL SECURITY;

-- Basic Policies
CREATE POLICY "Profiles: Users see own" ON public.profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Profiles: Users update own" ON public.profiles FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Projects: Users manage own" ON public.projects FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Jobs: Users manage own" ON public.jobs FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Assets: Users manage own" ON public.job_assets FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Usage: Users view own" ON public.usage_ledger FOR SELECT USING (auth.uid() = user_id);
