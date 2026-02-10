import os
import sys
import json
import modal
import subprocess
from dotenv import load_dotenv

# Define the Modal Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch",
        "torchaudio",
        "numpy", # WhisperX compat
        "librosa",
        "scipy",
        "requests",
        "supabase",
        "python-dotenv",
        "google-genai",
        "diffusers",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "whisperx",
        "demucs",
        "deep-translator",
        "fastapi[standard]"
    )
    .add_local_dir(
        local_path=os.path.dirname(os.path.abspath(__file__)),
        remote_path="/root/workers/audio-cpu",
        ignore=["**/.venv/**", "**/node_modules/**", "**/.git/**", "**/__pycache__/**", "**/*.pyc"]
    )
)

app = modal.App("video-jobs-audio-cpu", image=image)

# Secrets (Make sure these are set in Modal dashboard or .env)
# We can also pass them explicitly if they are in the local .env
secrets = [
    modal.Secret.from_dict({
        "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
        "SUPABASE_SERVICE_ROLE_KEY": os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
        "CALLBACK_URL": os.environ.get("CALLBACK_URL", ""), # This might need to be the Next.js API URL
        "CALLBACK_TOKEN": os.environ.get("CALLBACK_TOKEN", ""),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
        "RUNWAYML_API_KEY": os.environ.get("RUNWAYML_API_KEY", ""),
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "0",
        "TORCHAUDIO_BACKEND": "soundfile"
    })
]

@app.function(
    gpu="T4", # T4 is sufficient for WhisperX and Demucs usually, A10G for LTX
    timeout=900, # 15 minutes
    secrets=secrets,
    # mounts removed in favor of add_local_dir
)
@modal.web_endpoint(method="POST")
def process_audio_job(payload: dict):
    """
    Webhook handler for Vercel -> Modal handoff.
    Receives JSON payload matching the Next.js API 'modalPayload'.
    """
    import sys
    sys.path.append("/root/workers/audio-cpu")
    
    # We need to adapt the payload to what process_job expects.
    # main.process_job expects a BullMQ-style 'job' object with .data
    
    from main import process_job
    
    class MockJob:
        def __init__(self, data):
            self.data = data
            self.id = data.get("job_id", "modal_unknown")
            
    try:
        print(f"received job payload: {payload}")
        
        # Check if environment is sane
        import torch
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        # Adapt keys if necessary (Next.js sends camelCase or snake_case?)
        # Next.js payload: { job_id, asset_id, user_id, prompt, style }
        # main.py expects: { jobId, inputAudioAssetId, userId, mood, ... }
        
        job_data = {
            "jobId": payload.get("job_id"),
            "inputAudioAssetId": payload.get("asset_id"),
            "userId": payload.get("user_id"),
            "bgPrompt": payload.get("prompt"),
            "styleId": payload.get("style"),
            # Add defaults that might be missing
            "startTime": 0,
            "endTime": 60,
            "targetLanguage": "en" 
        }
        
        # Run the processing logic
        # process_job is async in main.py, so run it
        import asyncio
        
        # DYNAMIC CALLBACK: Patch the global CALLBACK_URL in main.py if provided
        if payload.get("callback_url"):
            print(f"[{job_data['jobId']}] Overriding CALLBACK_URL to: {payload['callback_url']}")
            main.CALLBACK_URL = payload["callback_url"]
            
        asyncio.run(process_job(MockJob(job_data), None))
        
        # --- CHAINED EXECUTION: Hand off to Render Worker ---
        print(f"[{job_data['jobId']}] Audio done. Triggering Cloud Render...")
        try:
            # Look up the render function in the OTHER Modal app
            render_func = modal.Function.lookup("video-jobs-render", "render_video_webhook")
            
            # Use .spawn() for fire-and-forget (async background task)
            # Pass the same job payload, maybe enriched?
            # main.py updates specific fields in Supabase, Render reads them.
            # We just pass the ID.
            
            # render_video_webhook expects a dict payload
            render_func.spawn(payload) # Spawn runs in background
            print(f"[{job_data['jobId']}] Render triggered successfully.")
            
        except Exception as chain_error:
            print(f"[{job_data['jobId']}] Failed to trigger Render Worker: {chain_error}")
            # Do NOT fail the whole request, as audio part succeeded.
            # But the user might be stuck.
        
        return {"status": "success", "job_id": payload.get("job_id"), "render_triggered": True}
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return {"status": "error", "message": str(e)}
