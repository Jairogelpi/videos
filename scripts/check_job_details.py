
from supabase import create_client

url = "https://yimzyjsxlfrxzsbpqdma.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpbXp5anN4bGZyeHpzYnBxZG1hIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA3MzI0NjQsImV4cCI6MjA4NjMwODQ2NH0.kfP4XJBTmJLoyEQooWgKgUPcgcQjvnqn_O1aaCw_E2Y"
# Wait, I used the wrong key in the script? No, I need SERVICE_ROLE_KEY to query all tables.
# Let's use the SERVICE_ROLE_KEY from .env.local
srv_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpbXp5anN4bGZyeHpzYnBxZG1hIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDczMjQ2NCwiZXhwIjoyMDg2MzA4NDY0fQ.UXu0R4Ey4WzNo0MtB0UmY_8zOOckHghZiyXoQ14Dteg"

supabase = create_client(url, srv_key)

jid = "49b7fa81-fdee-4ce7-ac69-748902cc30ec"
print(f"--- Events for {jid} ---")
res = supabase.table("job_events").select("*").eq("job_id", jid).order("at", desc=True).limit(20).execute()
for e in res.data:
    print(f"[{e['at']}] {e['event_type']} | {e['payload']}")

print("\n--- Current Job Status ---")
job = supabase.table("jobs").select("*").eq("id", jid).single().execute()
print(job.data)
