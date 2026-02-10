import modal
from modal import App, Image, Volume, Secret
import os
import subprocess
import shutil
from typing import Dict, Any

# Define the Modal App
app = App("tohjo-audio-ltx-worker")

# Define Cloud Volume for large model weights (persisted across runs)
model_cache = Volume.from_name("tohjo-model-cache", create_if_missing=True)

# Define the Image for GPU tasks (Transcription + LTX)
image = (
    Image.from_registry("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "diffusers",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "librosa",
        "scipy",
        "soundfile",
        "supabase",
        "python-dotenv",
        "whisperx@git+https://github.com/m-bain/whisperX.git",
        "demucs",
        "google-generativeai",
        "fastapi",
        "pydantic"
    )
)

# Definir rutas absolutas para los montajes (robustez en Windows)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RENDER_LOCAL_PATH = os.path.join(BASE_DIR, "apps", "render")
SHARED_LOCAL_PATH = os.path.join(BASE_DIR, "packages", "shared")

# Define the Image for Remotion (Node.js + Chromium)
render_image = (
    Image.debian_slim()
    .apt_install("curl", "gnupg", "ffmpeg")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs"
    )
    .run_commands(
        "apt-get install -y libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0 libcairo2"
    )
    .pip_install("supabase")
    .add_local_dir(RENDER_LOCAL_PATH, remote_path="/root/render")
    .add_local_dir(SHARED_LOCAL_PATH, remote_path="/root/packages/shared")
)

# Secrets: Inject Supabase & Google keys
secrets = [
    Secret.from_name("tohjo-secrets") # User must create this in Modal dashboard
]

@app.function(
    image=image,
    gpu="L4", # Cheapest 24GB GPU ($0.40/hr)
    volumes={"/root/cache": model_cache}, # Mount cache
    secrets=secrets,
    timeout=900 # 15 minutes max
)
def process_audio_video_job(job_id: str, asset_id: str, user_id: str, prompt: str, style: str):
    import torch
    from diffusers import LTXPipeline, LTXVideoTransformer3DModel
    from transformers import BitsAndBytesConfig
    from supabase import create_client
    import librosa
    import soundfile as sf
    from google import genai

    print(f"[{job_id}] Starting Modal Cloud Processing...")
    
    # 1. Setup Clients
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    
    # 2. Download Audio
    print(f"[{job_id}] Downloading asset {asset_id}...")
    try:
        # Fetch the real path from Supabase (asset_id is a UUID)
        asset_res = supabase.table("job_assets").select("path").eq("id", asset_id).single().execute()
        asset_path = asset_res.data["path"]
        
        # Get public URL or signed URL
        res = supabase.storage.from_("assets").create_signed_url(asset_path, 600)
        url = res["signedURL"]
        
        # Download
        subprocess.run(["curl", "-o", "input.mp3", url], check=True)
        
        # Crop to 60s max (ffmpeg)
        subprocess.run([
            "ffmpeg", "-y", "-i", "input.mp3", "-t", "60", 
            "-ac", "2", "-ar", "44100", "input_segment.wav"
        ], check=True)
        
    except Exception as e:
        print(f"[{job_id}] Download failed: {e}")
        return {"status": "failed", "error": str(e)}

    # 3. Audio Intelligence (WhisperX & Profiling)
    print(f"[{job_id}] Analyzing audio with WhisperX...")
    import whisperx
    
    device = "cuda"
    compute_type = "float16" 
    
    try:
        audio_data = whisperx.load_audio("input_segment.wav")
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        result = model.transcribe(audio_data, batch_size=16)
        
        # Alignment for precise timestamps
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio_data, device, return_char_alignments=False)
        
        final_words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                if "start" in w and "end" in w:
                    final_words.append({
                        "t0": round(w["start"], 3),
                        "t1": round(w["end"], 3),
                        "w": w["word"]
                    })
        
        print(f"[{job_id}] Transcription complete: {len(final_words)} words.")
        
    except Exception as e:
        print(f"[{job_id}] Transcription failed: {e}")
        final_words = [{"t0": 0, "t1": 5, "w": "[Transcription Failed]"}]

    # 4. Load LTX-Video (Cached)
    cache_dir = "/root/cache/ltx-video"
    if not os.path.exists(cache_dir):
        print(f"[{job_id}] Downloading LTX model to cache...")
        LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.float16).save_pretrained(cache_dir)
        
    print(f"[{job_id}] Loading LTX Engine from cache...")
    pipe = LTXPipeline.from_pretrained(cache_dir, torch_dtype=torch.float16, local_files_only=True)
    pipe.to("cuda")
    
    # 5. Generate Video
    print(f"[{job_id}] Generating LTX-Video content...")
    
    video_path = "output.mp4"
    images = pipe(
        prompt=prompt, 
        negative_prompt="low quality, worst quality, deformed",
        width=768, 
        height=512, 
        num_frames=161, # ~6.7s @ 24fps
        num_inference_steps=50
    ).frames[0]
    
    from diffusers.utils import export_to_video
    export_to_video(images, video_path, fps=24)
    
    # 6. Upload Result
    print(f"[{job_id}] Uploading result...")
    video_key = f"{user_id}/{job_id}_video.mp4"
    with open(video_path, "rb") as f:
        supabase.storage.from_("assets").upload(
            video_key, 
            f, 
            {"content-type": "video/mp4"}
        )
    
    # 7. TRIGGER REMOTION RENDER (Serverless Chain)
    print(f"[{job_id}] IA complete. Triggering final Remotion render...")
    
    render_payload = {
        "job_id": job_id,
        "user_id": user_id,
        "video_bg_url": video_key,
        "audio_url": asset_path, # Use the actual path fetched earlier
        "words": final_words,
        "prompt": prompt,
        "styleId": style
    }
    
    render_final_video.spawn(render_payload)
    
    return {"status": "success", "job_id": job_id, "video_url": video_key}

@app.function(
    image=render_image,
    secrets=secrets,
    timeout=600
)
def render_final_video(payload: Dict[str, Any]):
    from supabase import create_client
    import subprocess
    import os
    import json

    job_id = payload["job_id"]
    user_id = payload["user_id"]
    
    print(f"[{job_id}] 🎬 Starting Cloud Remotion Render...")
    
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    
    # 1. Prepare Environment
    os.chdir("/root/render")
    
    # Install dependencies (only if node_modules doesn't exist)
    if not os.path.exists("node_modules"):
        print(f"[{job_id}] Installing npm dependencies...")
        subprocess.run(["npm", "install"], check=True)
    
    # 2. Render Media
    output_path = f"/tmp/{job_id}_final.mp4"
    
    input_props = json.dumps({
        "audioUrl": f"{os.environ['SUPABASE_URL']}/storage/v1/object/public/assets/{payload['audio_url']}",
        "videoBgUrl": f"{os.environ['SUPABASE_URL']}/storage/v1/object/public/assets/{payload['video_bg_url']}",
        "words": payload.get("words", []),
        "text": payload.get("prompt", "Tohjo Studio Generation"),
        "styleId": payload.get("styleId", "cinematic")
    })

    print(f"[{job_id}] Executing Remotion render (npx remotion render)...")
    try:
        # We use --props to pass the JSON
        subprocess.run([
            "npx", "remotion", "render", 
            "src/index.ts", "LyricVideo", 
            output_path,
            "--props", input_props,
            "--browser-executable", "/usr/bin/google-chrome" # If installed via apt? 
            # Actually Image.debian_slim + Chromium deps usually works with default remotion search
        ], check=True)
    except Exception as e:
        print(f"[{job_id}] Render failed: {e}")
        supabase.from_("jobs").update({"status": "failed", "error": str(e)}).eq("id", job_id).execute()
        return

    # 3. Upload Output
    final_key = f"{job_id}/final.mp4"
    print(f"[{job_id}] Rendering finished. Uploading to outputs/{final_key}...")
    with open(output_path, "rb") as f:
        supabase.storage.from_("outputs").upload(
            final_key,
            f,
            {"content-type": "video/mp4", "upsert": "true"}
        )
    
    # 4. Update Job Status & Asset Record
    # First create/update asset record
    asset_res = supabase.from_("job_assets").insert({
        "user_id": user_id,
        "bucket": "outputs",
        "path": final_key,
        "kind": "final_video"
    }).select().single().execute()
    
    asset_id = asset_res.data["id"]

    supabase.from_("jobs").update({
        "status": "completed", 
        "progress": 100,
        "output_video_asset_id": asset_id
    }).eq("id", job_id).execute()
    
    print(f"[{job_id}] ✅ Pipeline complete! Final video uploaded.")

@app.function(image=image)
@modal.web_endpoint(method="POST")
def api_trigger(item: Dict[str, Any]):
    """
    Public Endpoint for Vercel.
    Example body: {"job_id": "123", "asset_id": "abc", "user_id": "u1", "prompt": "foo", "style": "bar"}
    """
    print(f"Received Trigger: {item}")
    
    job_id = item.get("job_id")
    asset_id = item.get("asset_id")
    user_id = item.get("user_id")
    prompt = item.get("prompt")
    style = item.get("style", "cinematic")
    
    if not all([job_id, asset_id, user_id, prompt]):
        return {"error": "Missing required fields (job_id, asset_id, user_id, prompt)"}
    
    # SPAWN the GPU function asynchronously
    # This returns immediately so the HTTP request doesn't timeout
    process_audio_video_job.spawn(job_id, asset_id, user_id, prompt, style)
    
    return {"status": "queued", "job_id": job_id, "message": "GPU worker started"}

@app.local_entrypoint()
def main():
    print("Testing Modal function locally...")
    # This runs on your local machine and triggers the cloud function
    process_audio_video_job.remote(
        "test-job-local", 
        "asset-123", 
        "user-000", 
        "Cyberpunk city rain", 
        "cinematic"
    )
