import modal
import os
import subprocess

# Create a persistent volume for model weights
# This prevents the Docker image from growing to 50GB+
volume = modal.Volume.from_name("models-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("wan-downloader", image=image)

@app.function(volumes={"/models": volume}, timeout=3600) # 1 hour timeout
def download_model():
    from huggingface_hub import snapshot_download
    
    model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    local_dir = "/models/Wan2.1-T2V-14B-Diffusers"
    
    print(f"Starting download of {model_id} to {local_dir}...")
    
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        cache_dir="/models/cache",
        max_workers=8
    )
    
    print(f"Download complete to {path}!")
    
    # Verify files exist in the volume mount
    print("Verifying files in /models/Wan2.2-TI2V-5B-Diffusers:")
    subprocess.run(["ls", "-lh", local_dir])
    
    volume.commit()
    print("Volume committed.")

@app.local_entrypoint()
def main():
    download_model.remote()
