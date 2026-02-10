import requests
import json
import time

# The URL you got from the deployment output
MODAL_URL = "https://jairogelpi--tohjo-audio-ltx-worker-api-trigger.modal.run"

def test_webhook():
    print(f"🚀 Testing Modal Webhook: {MODAL_URL}")
    
    payload = {
        "job_id": f"test-webhook-{int(time.time())}",
        "asset_id": "test-asset-id",
        "user_id": "test-user-id",
        "prompt": "A futuristic city with neon lights, cinematic 4k",
        "style": "cinematic"
    }
    
    print(f"📤 Sending Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(MODAL_URL, json=payload, timeout=10)
        
        print(f"📥 Response Status: {response.status_code}")
        print(f"📥 Response Body: {response.text}")
        
        if response.status_code == 200:
            print("✅ SUCCESS: Webhook accepted the job!")
            data = response.json()
            if data.get("status") == "queued":
                print("✅ Job is effectively queued in the cloud.")
            else:
                print("⚠️ Job accepted but status is unexpected.")
        else:
            print("❌ FAILURE: Webhook rejected the request.")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_webhook()
