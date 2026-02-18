
from supabase import create_client

url = "https://yimzyjsxlfrxzsbpqdma.supabase.co"
srv_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpbXp5anN4bGZyeHpzYnBxZG1hIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDczMjQ2NCwiZXhwIjoyMDg2MzA4NDY0fQ.UXu0R4Ey4WzNo0MtB0UmY_8zOOckHghZiyXoQ14Dteg"

supabase = create_client(url, srv_key)

jid = "49b7fa81-fdee-4ce7-ac69-748902cc30ec"

# 1. Assets link (Cinematic Background)
asset_path = f"{jid}/cinematic_bg.mp4"
try:
    signed_asset = supabase.storage.from_("assets").create_signed_url(asset_path, 3600)
    print(f"CINEMATIC_BG_LINK: {signed_asset['signedURL']}")
except Exception as e:
    print(f"Error generating cinematic bg link: {e}")

# 2. Check if there are ANY final videos in outputs just in case I missed the ID
print("\n--- Recent files in 'outputs' root ---")
try:
    res = supabase.storage.from_("outputs").list()
    for item in res:
        print(f"  Folder: {item['name']}")
except Exception as e:
    print(f"Error listing outputs: {e}")

# 3. Check for specific error in the job record
job = supabase.table("jobs").select("error_message, status, stage").eq("id", jid).single().execute()
print(f"\nDB STATUS: {job.data}")
