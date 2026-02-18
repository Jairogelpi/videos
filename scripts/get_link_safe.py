
from supabase import create_client

url = "https://yimzyjsxlfrxzsbpqdma.supabase.co"
srv_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpbXp5anN4bGZyeHpzYnBxZG1hIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDczMjQ2NCwiZXhwIjoyMDg2MzA4NDY0fQ.UXu0R4Ey4WzNo0MtB0UmY_8zOOckHghZiyXoQ14Dteg"

supabase = create_client(url, srv_key)
jid = "49b7fa81-fdee-4ce7-ac69-748902cc30ec"
path = f"{jid}/cinematic_bg.mp4"

res = supabase.storage.from_("assets").create_signed_url(path, 7200)
with open("C:/Users/jairo/Desktop/videos/final_url.txt", "w", encoding="utf-8") as f:
    f.write(res['signedURL'])
