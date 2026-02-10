import sys
import subprocess
import os

try:
    from dotenv import load_dotenv
except ImportError:
    print("❌ 'python-dotenv' not found. Please run: pip install python-dotenv")
    sys.exit(1)

def run_cmd(cmd):
    """Executes a command list properly on Windows."""
    print(f"\n[EXEC] {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        return False

def main():
    print("🚀 Starting Hybrid Cloud Deployment...\n")

    # 1. Load Local Secrets
    # We look for .env in the parent directory (workers/audio-cpu or root)
    env_path = os.path.join(os.path.dirname(__file__), "..", "audio-cpu", ".env")
    if os.path.exists(env_path):
        print(f"Loading secrets from: {env_path}")
        load_dotenv(env_path)
    else:
        print("⚠️ No .env file found in audio-cpu/.env - attempting current dir")
        load_dotenv()

    # 2. Deploy Secrets to Modal
    print("[1/2] Syncing Keys (Supabase, Google, Redis) to Cloud...")
    
    # We construct the key=value list explicitly so Modal gets them directly
    # This avoids issues where Modal's context doesn't see our environment
    keys_to_sync = [
        "SUPABASE_URL", 
        "SUPABASE_SERVICE_ROLE_KEY", 
        "GOOGLE_API_KEY", 
        "RUNWAYML_API_KEY",
        "REDIS_URL"
    ]
    
    secret_args = []
    for key in keys_to_sync:
        val = os.getenv(key)
        if val:
            # Escape value if needed (simple check)
            secret_args.append(f"{key}={val}")
        else:
            print(f"   ⚠️ Skipping {key} (Not found in .env)")

    if not secret_args:
        print("❌ CRITICAL: No secrets found to deploy!")
        sys.exit(1)

    # Command: python -m modal secret create tohjo-secrets KEY=VAL ...
    secret_cmd = [sys.executable, "-m", "modal", "secret", "create", "tohjo-secrets", "--force"] + secret_args
    
    run_cmd(secret_cmd)

    # 3. Deploy App
    print("\n[2/2] Deploying Worker App...")
    deploy_cmd = [sys.executable, "-m", "modal", "deploy", "modal_app.py"]
    
    if run_cmd(deploy_cmd):
        print("\n✅ SUCCESS! Your Cloud Engine is LIVE.")
        print("To verify, run: python -m modal run modal_app.py")
    else:
        print("\n❌ Deployment Failed.")

if __name__ == "__main__":
    main()
