import os
import modal

# Define Image with Node.js 20 + FFmpeg + Chromium
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "git", "ffmpeg") 
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g pnpm"
    )
    # Install Chromium dependencies for Remotion
    .run_commands(
        "apt-get install -y chromium libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2"
    )
    .pip_install("fastapi[standard]")
    # Mount local directories (Baked into Image for robust portability)
    .add_local_dir(
        local_path=os.path.dirname(os.path.abspath(__file__)),
        remote_path="/root/workers/render",
        ignore=["**/.venv/**", "**/node_modules/**", "**/.git/**", "**/__pycache__/**", "**/.turbo/**", "**/dist/**"]
    )
    .add_local_dir(
        local_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "apps", "render"),
        remote_path="/root/apps/render",
        ignore=["**/node_modules/**", "**/.next/**", "**/.turbo/**", "**/dist/**", "**/.git/**"]
    )
    .add_local_dir(
        local_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "packages"),
        remote_path="/root/packages",
        ignore=["**/node_modules/**", "**/.turbo/**", "**/dist/**", "**/.git/**"]
    )
)

app = modal.App("video-jobs-render", image=image)

# Secrets
secrets = [
    modal.Secret.from_dict({
        "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
        "SUPABASE_SERVICE_ROLE_KEY": os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
        "CALLBACK_URL": os.environ.get("CALLBACK_URL", ""),
        "CALLBACK_TOKEN": os.environ.get("CALLBACK_TOKEN", "")
    })
]

@app.function(
    gpu="any", # Remotion can use GPU? Chromium software render is usually fine, but GPU might crash if headless setup is wrong.
    # Let's start with CPU. Remotion is mostly CPU bounded for standard comps unless using WebGL heavy stuff.
    # Actually, keep it CPU to save cost unless needed.
    memory=4096,
    timeout=600,
    secrets=secrets,
    # mounts removed in favor of add_local_dir
)
@modal.web_endpoint(method="POST")
def render_video_webhook(payload: dict):
    """
    Triggers Remotion render via CLI.
    """
    import subprocess
    import json
    
    print(f"Received Render Job: {payload}")
    
    # 1. Install Dependencies (Lazy Install pattern)
    # In a real prod setup, we'd bake this into the image, but for monorepo speed:
    # We need to run pnpm install in the root? Or just in workers/render?
    # Our image has nodejs.
    
    # We are in /root.
    # We need to install dependencies for `workers/render`.
    
    # Setup pnpm workspace or just install locally
    # Hack: just install inside workers/render
    wd = "/root/workers/render"
    
    # Check if node_modules exists (Modal usually persists /tmp? No.)
    # We run npm install every time? That's slow.
    # Better: Use modal.Image.run_commands to pre-install.
    # But we need package.json for that.
    # For now, we do it at runtime (slow start) or rely on creating a custom image in a separate step.
    # Let's try runtime install (first run slow).
    
    # Actually, we need to install dependencies for apps/render too?
    # Monorepos are tricky in Modal.
    # Let's try to run `pnpm install` in /root/workers/render.
    
    print("Installing dependencies...")
    subprocess.run(["pnpm", "install"], cwd=wd, check=True)
    # We might need to install in apps/render too if it's not hoisted
    subprocess.run(["pnpm", "install"], cwd="/root/apps/render", check=True)
    
    # 2. Run Render
    # We write payload to json
    payload_path = "/tmp/job.json"
    with open(payload_path, "w") as f:
        json.dump(payload, f)
        
    print("Starting Render Process...")
    # Use tsx to run the typescript file directly
    cmd = ["npx", "tsx", "src/cli-render.ts", payload_path]
    
    # DYNAMIC CALLBACK: Pass to subprocess via env
    env = os.environ.copy()
    if payload.get("callback_url"):
        print(f"Overriding CALLBACK_URL to: {payload['callback_url']}")
        env["CALLBACK_URL"] = payload["callback_url"]
    
    res = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, env=env)
    
    print(res.stdout)
    if res.stderr:
        print(f"STDERR: {res.stderr}")
        
    if res.returncode != 0:
        return {"status": "error", "message": res.stderr}
        
    return {"status": "success", "message": "Render completed"}
