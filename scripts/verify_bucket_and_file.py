
from supabase import create_client

url = "https://yimzyjsxlfrxzsbpqdma.supabase.co"
srv_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpbXp5anN4bGZyeHpzYnBxZG1hIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDczMjQ2NCwiZXhwIjoyMDg2MzA4NDY0fQ.UXu0R4Ey4WzNo0MtB0UmY_8zOOckHghZiyXoQ14Dteg"

supabase = create_client(url, srv_key)

print("--- BUCKET STATUS ---")
try:
    buckets = supabase.storage.list_buckets()
    for b in buckets:
        print(f"Bucket: {b.id} | Public: {b.public}")
except Exception as e:
    print(f"Error listing buckets: {e}")

jid = "49b7fa81-fdee-4ce7-ac69-748902cc30ec"
print(f"\n--- Checking Storage for Job {jid} ---")
# Try listing files in outputs folder
try:
    files = supabase.storage.from_("outputs").list(jid)
    if files:
        for f in files:
            print(f"  - {f['name']} ({f['metadata'].get('size', 0)} bytes, Type: {f['metadata'].get('mimetype')})")
    else:
        print(f"  No files found in outputs/{jid}/")
except Exception as e:
    print(f"  Error listing files in outputs: {e}")

# Check job assets for this job specifically
print("\n--- Job Assets Entry ---")
assets = supabase.table("job_assets").select("*").eq("job_id", jid).execute()
for a in assets.data:
    print(f"  Kind: {a['kind']} | Bucket: {a['bucket']} | Path: {a['path']}")
