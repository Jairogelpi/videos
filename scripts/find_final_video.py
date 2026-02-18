
from supabase import create_client

url = "https://yimzyjsxlfrxzsbpqdma.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpbXp5anN4bGZyeHpzYnBxZG1hIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDczMjQ2NCwiZXhwIjoyMDg2MzA4NDY0fQ.UXu0R4Ey4WzNo0MtB0UmY_8zOOckHghZiyXoQ14Dteg"

supabase = create_client(url, key)

print("--- RECENT PRODUCTION JOBS ---")
res = supabase.table("jobs").select("id, status, created_at").order("created_at", desc=True).limit(5).execute()
for r in res.data:
    print(f"[{r['id']}] {r['status']} ({r['created_at']})")
    assets = supabase.table("job_assets").select("bucket, path").eq("job_id", r['id']).eq("kind", "final_video").execute()
    if assets.data:
        for a in assets.data:
            print(f"  -> VIDEO: {a['bucket']}/{a['path']}")
            print(f"  -> LINK: {url}/storage/v1/object/public/{a['bucket']}/{a['path']}")
    else:
        # Check if the file exists in storage even if asset record is missing
        print(f"  (Checking storage for outputs/final.mp4...)")

print("\n--- RECENT ERRORS ---")
errs = supabase.table("job_events").select("job_id, payload").eq("level", "error").order("at", desc=True).limit(3).execute()
for e in errs.data:
    print(f"[ERR][{e['job_id']}] {e['payload'].get('message')}")
