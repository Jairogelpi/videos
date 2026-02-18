
from supabase import create_client

url = "https://yimzyjsxlfrxzsbpqdma.supabase.co"
srv_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpbXp5anN4bGZyeHpzYnBxZG1hIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDczMjQ2NCwiZXhwIjoyMDg2MzA4NDY0fQ.UXu0R4Ey4WzNo0MtB0UmY_8zOOckHghZiyXoQ14Dteg"

supabase = create_client(url, srv_key)

jid = "49b7fa81-fdee-4ce7-ac69-748902cc30ec"

# Try to check if the file exists in storage
try:
    path = f"{jid}/final.mp4"
    # We can't use .list() easily on specific folders with the python client in a one-liner
    # so we'll try to download 1 byte to check existence
    res = supabase.storage.from_("outputs").download(path, {"range": "bytes=0-1"})
    print(f"✅ VIDEO EXISTS in storage at: outputs/{path}")
    print(f"LINK: {url}/storage/v1/object/public/outputs/{path}")
except Exception as e:
    print(f"❌ VIDEO NOT FOUND in storage yet: {e}")

# Check latest status
job = supabase.table("jobs").select("status, progress, error_message").eq("id", jid).single().execute()
print(f"Current Status: {job.data['status']} | Progress: {job.data['progress']}%")
if job.data['error_message']:
    print(f"ERROR: {job.data['error_message']}")

# List all recent jobs to see if others exist
print("\nRecent jobs list:")
recent = supabase.table("jobs").select("id, status, created_at").order("created_at", desc=True).limit(5).execute()
for r in recent.data:
    print(f" - [{r['id']}] {r['status']} at {r['created_at']}")
