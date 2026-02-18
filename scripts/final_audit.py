
from supabase import create_client

url = "https://yimzyjsxlfrxzsbpqdma.supabase.co"
srv_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpbXp5anN4bGZyeHpzYnBxZG1hIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDczMjQ2NCwiZXhwIjoyMDg2MzA4NDY0fQ.UXu0R4Ey4WzNo0MtB0UmY_8zOOckHghZiyXoQ14Dteg"

supabase = create_client(url, srv_key)

print("--- LATEST 5 JOBS ---")
res = supabase.table("jobs").select("id, status, progress, created_at").order("created_at", desc=True).limit(5).execute()
for r in res.data:
    print(f"[{r['id']}] Status: {r['status']} | Progress: {r['progress']}% | Created: {r['created_at']}")
    # Check if this job has any assets
    assets = supabase.table("job_assets").select("kind, bucket, path").eq("job_id", r['id']).execute()
    for a in assets.data:
        print(f"  - Asset: {a['kind']} ({a['bucket']}/{a['path']})")
        if a['kind'] in ['final_video', 'draft_video', 'cinematic_bg']: # Wait, kind names might be different
             try:
                 signed = supabase.storage.from_(a['bucket']).create_signed_url(a['path'], 3600)
                 print(f"    LINK: {signed['signedURL']}")
             except: pass

print("\n--- BUCKET LISTING (ROOT) ---")
for b in ["assets", "outputs"]:
    try:
        files = supabase.storage.from_(b).list()
        print(f"Bucket {b}: {[f['name'] for f in files]}")
    except: pass
