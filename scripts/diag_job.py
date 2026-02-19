
import os
from supabase import create_client

url = "https://yimzyjsxlfrxzsbpqdma.supabase.co"
srv_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpbXp5anN4bGZyeHpzYnBxZG1hIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDczMjQ2NCwiZXhwIjoyMDg2MzA4NDY0fQ.UXu0R4Ey4WzNo0MtB0UmY_8zOOckHghZiyXoQ14Dteg"

supabase = create_client(url, srv_key)
jid = "5509bd26-a627-425b-a1b1-c6c493586bca"

print(f"--- JOB {jid} ASSETS ---")
res = supabase.table('job_assets').select('*').eq('job_id', jid).execute()
for asset in res.data:
    print(f"Kind: {asset['kind']} | Path: {asset['path']}")
    if asset['kind'] in ['draft_video', 'cinematic_bg', 'final_video']:
        try:
            signed = supabase.storage.from_("assets").create_signed_url(asset['path'], 3600)
            print(f"   URL: {signed['signedURL']}")
        except:
            pass

print(f"\n--- RECENT EVENTS ---")
res = supabase.table('job_events').select('*').eq('job_id', jid).order('at', desc=True).limit(5).execute()
for event in res.data:
    print(f"[{event['at']}] {event['event_type']} | {event['payload'].get('status', '')} {event['payload'].get('progress', '')}%")
