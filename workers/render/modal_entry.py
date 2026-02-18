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
    gpu="any", 
    memory=4096,
    timeout=600,
    secrets=secrets,
)
def render_video_kernel(payload: dict):
    """
    Internal kernel for Remotion render.
    """
    import subprocess
    import json
    import os
    
    print(f"Starting Render Kernel for Job: {payload.get('job_id')}")
    
    wd = "/root/workers/render"
    
    print("Installing dependencies...")
    subprocess.run(["pnpm", "install"], cwd=wd, check=True)
    subprocess.run(["pnpm", "install"], cwd="/root/apps/render", check=True)
    
    payload_path = "/tmp/job.json"
    with open(payload_path, "w") as f:
        json.dump(payload, f)
        
    print("Starting Render Process...")
    cmd = ["npx", "tsx", "src/cli-render.ts", payload_path]
    
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

@app.function(
    gpu="any",
    memory=4096,
    timeout=600,
    secrets=secrets,
)
@modal.web_endpoint(method="POST")
def render_video_webhook(payload: dict):
    """
    Public entrypoint for manual/API triggers.
    Handoff to kernel for async processing.
    """
    render_video_kernel.spawn(payload)
    return {"status": "accepted", "message": "Render job spawned in cloud"}
