
from supabase import create_client

url = "https://yimzyjsxlfrxzsbpqdma.supabase.co"
srv_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpbXp5anN4bGZyeHpzYnBxZG1hIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDczMjQ2NCwiZXhwIjoyMDg2MzA4NDY0fQ.UXu0R4Ey4WzNo0MtB0UmY_8zOOckHghZiyXoQ14Dteg"

supabase = create_client(url, srv_key)

jid = "49b7fa81-fdee-4ce7-ac69-748902cc30ec"
path = f"{jid}/final.mp4"

print(f"Generating signed URL for outputs/{path}...")
try:
    # Generate a signed URL valid for 1 hour
    signed_res = supabase.storage.from_("outputs").create_signed_url(path, 3600)
    print(f"SIGNED_URL: {signed_res['signedURL']}")
except Exception as e:
    print(f"Error generating signed URL: {e}")
