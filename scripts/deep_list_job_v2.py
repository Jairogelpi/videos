
from supabase import create_client

url = "https://yimzyjsxlfrxzsbpqdma.supabase.co"
srv_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpbXp5anN4bGZyeHpzYnBxZG1hIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDczMjQ2NCwiZXhwIjoyMDg2MzA4NDY0fQ.UXu0R4Ey4WzNo0MtB0UmY_8zOOckHghZiyXoQ14Dteg"

supabase = create_client(url, srv_key)

jid = "49b7fa81-fdee-4ce7-ac69-748902cc30ec"

for b in ["assets", "outputs"]:
    print(f"--- BUCKET: {b} ---")
    try:
        # List files in the ROOT of the bucket
        res = supabase.storage.from_(b).list()
        # Find the folder for the job
        folder_exists = False
        for f in res:
            if f['name'] == jid:
                folder_exists = True
                print(f"FOLDER {jid} EXISTS in {b}")
                # List inside the folder
                inner = supabase.storage.from_(b).list(jid)
                for item in inner:
                    print(f"  FILE: {item['name']} | SIZE: {item.get('metadata',{}).get('size')}")
        
        if not folder_exists:
            print(f"FOLDER {jid} NOT FOUND in {b}")
            # print all root items just in case
            print(f"ROOT ITEMS: {[i['name'] for i in res]}")

    except Exception as e:
        print(f"Error checking {b}: {e}")
