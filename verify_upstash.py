import redis
import sys

# Connection string provided by user
REDIS_URL = "rediss://default:AcrfAAIncDI1MDFhY2QyM2Y5NzE0OWEwYWM1YWU2ZjkyYzcwYzY4NXAyNTE5MzU@upward-parakeet-51935.upstash.io:6379"

def test_connection():
    print(f"Connecting to Upstash: {REDIS_URL.split('@')[1]}...")
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        # Test Set
        r.set("tohjo_cloud_test", "connected_successfully")
        # Test Get
        val = r.get("tohjo_cloud_test")
        
        if val == "connected_successfully":
            print("✅ SUCCESS: Connected to Upstash Redis and performed Read/Write.")
            print("This URL is ready for production.")
        else:
            print(f"❌ FAILURE: Write succeeded but read returned: {val}")
            
    except Exception as e:
        print(f"❌ CRITICAL FAILURE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_connection()
