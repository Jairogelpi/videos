
from supabase import create_client

url = "https://yimzyjsxlfrxzsbpqdma.supabase.co"
srv_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpbXp5anN4bGZyeHpzYnBxZG1hIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDczMjQ2NCwiZXhwIjoyMDg2MzA4NDY0fQ.UXu0R4Ey4WzNo0MtB0UmY_8zOOckHghZiyXoQ14Dteg"

supabase = create_client(url, srv_key)

jid = "49b7fa81-fdee-4ce7-ac69-748902cc30ec"

for bucket_name in ["assets", "outputs"]:
    print(f"\n--- Bucket: {bucket_name} | Folder: {jid} ---")
    try:
        files = supabase.storage.from_(bucket_name).list(jid)
        if files:
            for f in files:
                print(f"  FOUND: {f['name']} (Size: {f.get('metadata', {}).get('size', 'unknown')})")
        else:
            print(f"  No files found in {bucket_name}/{jid}")
    except Exception as e:
        print(f"  Error listing {bucket_name}: {e}")

# Also check job status and error logs one last time
print("\n--- FINAL JOB STATUS ---")
job = supabase.table("jobs").select("*").eq("id", jid).single().execute()
print(job.data)
