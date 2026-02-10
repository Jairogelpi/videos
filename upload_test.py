import os
from supabase import create_client, Client

url = "http://127.0.0.1:54321"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
supabase: Client = create_client(url, key)

with open("test_audio.mp3", "rb") as f:
    supabase.storage.from_("assets").upload(
        path="test/audio.mp3",
        file=f,
        file_options={"upsert": "true", "content-type": "audio/mpeg"}
    )
print("Uploaded test/audio.mp3 successfully")
