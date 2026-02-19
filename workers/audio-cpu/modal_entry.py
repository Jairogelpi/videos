# v1.1.2 - Plan Neutron-Start: FFmpeg ABI Compatibility Fix
import os
import sys
import json
import modal
import subprocess
from dotenv import load_dotenv

# Load .env file so secrets are available at deploy time
load_dotenv()

# --- CRITICAL SYSTEM CONFIG ---
# Disable PyTorch 2.6+ security checks for trusted legacy checkpoints
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ------------------------------

# Define the Modal Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg", "git", "build-essential", "libsndfile1",
        "libavutil-dev", "libavcodec-dev", "libavformat-dev", "libswscale-dev", "libavfilter-dev", "libavdevice-dev"
    )
    # v9 Force Refresh: 2026-02-16 00:08 - Kernel Handoff Refactor
    # 1. Unified AI Stack (CUDA 12.4)
    # We install ALL heavy AI libraries in a single step with the CUDA index to guarantee binary compatibility.
    # This specifically fixes the 'torchaudio' ABI mismatch by ensuring it matches the 'torch' version.
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "transformers", "diffusers", "accelerate", "peft", "bitsandbytes", "sentencepiece", "protobuf",
        "requests", "supabase", "python-dotenv", "google-genai", "deep-translator", "fastapi[standard]", 
        "huggingface_hub[hf_transfer]", "hf_transfer", "ftfy", "matplotlib", "opencv-python-headless", "imageio", "imageio-ffmpeg", "moviepy",
        "demucs", "librosa", "scipy", "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu124"
    )
    # WhisperX and LTX-Video
    .pip_install("git+https://github.com/m-bain/whisperX.git", "ltx-video")
    # 2. Legacy cuDNN 8 for WhisperX/CTranslate2 Support
    .pip_install("nvidia-cudnn-cu11==8.9.6.50", "nvidia-cublas-cu11==11.11.3.6")
    .env({"HF_HOME": "/models", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # (Plan Zero-Chill: Bakery removed from build to avoid timeouts; lazy imports in main.py handle the speed)
    .add_local_dir(
        local_path=os.path.dirname(os.path.abspath(__file__)),
        remote_path="/root/workers/audio-cpu",
        ignore=["**/.venv/**", "**/node_modules/**", "**/.git/**", "**/__pycache__/**", "**/*.pyc"]
    )
)

models_volume = modal.Volume.from_name("models-volume", create_if_missing=True)

app = modal.App("video-jobs-audio-cpu", image=image)


# Secrets (loaded from local .env at deploy time, injected into Modal container)
secrets = [
    modal.Secret.from_dict({
        "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
        "SUPABASE_SERVICE_ROLE_KEY": os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
        "CALLBACK_URL": os.environ.get("CALLBACK_URL", ""),
        "CALLBACK_TOKEN": os.environ.get("CALLBACK_TOKEN", ""),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
        "RUNWAYML_API_KEY": os.environ.get("RUNWAYML_API_KEY", ""),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "0",
        "TORCHAUDIO_BACKEND": "soundfile"
    })
]


@app.cls(
    gpu="L4",  # Value-First: Significantly cheaper than A100 (~70% savings)
    timeout=900,
    secrets=secrets,
    volumes={"/models": models_volume},
    min_containers=0,  # On-demand to save credits (B200 = ~$3-4/hr idle cost)
    scaledown_window=600,  # Keep container alive 10 min after last request
    enable_memory_snapshot=True,  # <--- CRITICAL: Snapshot memory state after startup for millisecond cold-starts
)
class AudioWorker:
    """
    Class-based Modal worker. The Wan 2.2 model is loaded ONCE when the
    container starts via @modal.enter(), then reused across all requests.
    With keep_warm=1, there's always a container ready with the model loaded.
    """

    @modal.enter()
    def startup(self):
        """Pre-load EVERYTHING when the container boots — not per-request."""
        import sys
        
        # 1. Linked Legacy cuDNN 8 for WhisperX
        # WhisperX (CTranslate2) needs cuDNN 8. We installed it via pip, but we need to ensure LD_LIBRARY_PATH helps it found.
        # However, the pip package usually puts libs in site-packages/nvidia/...
        import os
        import sys
        
        # Manually find and verify cuDNN 8
        try:
            import nvidia.cudnn
            cudnn_path = os.path.dirname(nvidia.cudnn.__file__)
            lib_path = os.path.join(cudnn_path, "lib")
            print(f"[STARTUP] 🔍 Searching for legacy cuDNN 8 libraries...")
            if os.path.exists(lib_path):
                 print(f"[STARTUP] ✅ Found cuDNN 8 at: {lib_path}")
                 # Add to LD_LIBRARY_PATH for this process
                 os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                 # Also symlink for good measure if we have permissions (Modal usually allows /usr/lib modifications in some scopes, but env var is safer)
                 # We will try to symlink to /usr/lib/x86_64-linux-gnu which covers most cases
                 try:
                     import shutil
                     dest_dir = "/usr/lib/x86_64-linux-gnu"
                     for file in os.listdir(lib_path):
                         if "libcudnn" in file:
                             src = os.path.join(lib_path, file)
                             dst = os.path.join(dest_dir, file)
                             if not os.path.exists(dst):
                                 os.symlink(src, dst)
                                 print(f"   -> Linked {file}")
                     print(f"[STARTUP] 🔗 Symlinking cuDNN 8 libs to {dest_dir}...")
                 except Exception as e:
                     print(f"[STARTUP] ⚠️ Could not symlink libs (Permission?): {e}")
            else:
                 print(f"[STARTUP] ⚠️ cuDNN 8 lib folder not found at {lib_path}")
        except ImportError:
            print("[STARTUP] ⚠️ nvidia.cudnn module not found. WhisperX might fail.")



        sys.path.append("/root/workers/audio-cpu")

        # 2. Parallel VRAM Loading - Plan Neutron-Start
        import threading
        import main
        self.main = main

        def load_wan():
            try:
                print("[STARTUP-P] Loading Wan 2.1 (FP16 + Offload)...")
                pipe = main.WanModelManager.get_pipe()
                if pipe:
                    print("[STARTUP-P] ✅ Wan 2.1 Loaded.")
                else:
                    print("[STARTUP-P] ❌ Wan Load Failed (Returned None)")
            except Exception as e:
                print(f"[STARTUP-P] ❌ Wan Load Error: {e}")


        def load_whisper():
            try:
                print("[STARTUP-P] Loading WhisperX...")
                # Use Singleton Manager to ensure cache coherence with process_job
                model = main.WhisperModelManager.get_model()
                if model:
                    print("[STARTUP-P] ✅ WhisperX Loaded.")
                else:
                    print("[STARTUP-P] ❌ WhisperX Load Failed (Returned None)")
            except Exception as e:
                print(f"[STARTUP-P] ❌ WhisperX Load Error: {e}")

        print("[STARTUP] Triggering Sequential Neutron-Start (Memory-Safe)...")
        load_wan()
        import torch
        torch.cuda.empty_cache()
        
        load_whisper()
        torch.cuda.empty_cache()
        

        print("[STARTUP] 🚀 Neutron-Start Complete. Memory Optimized.")

    @modal.method()
    def generate_scene(self, job_id, i, num_clips, scenes, scene_durations, fps=24, width=720, height=1280):
        """Worker method for parallel scene generation (Returns bytes)."""
        import os
        import tempfile
        worker_tmp = tempfile.mkdtemp()
        try:
            interp_path, raw_path = self.main.generate_single_scene(job_id, i, num_clips, scenes, scene_durations, worker_tmp, fps=fps, width=width, height=height)
            with open(interp_path, "rb") as f:
                interp_data = f.read()
            with open(raw_path, "rb") as f:
                raw_data = f.read()
            return {"interp": interp_data, "raw": raw_data}
        finally:
            import shutil
            shutil.rmtree(worker_tmp, ignore_errors=True)

    @modal.fastapi_endpoint(method="POST")
    async def process_audio_job(self, payload: dict):
        class MockJob:
            def __init__(self, data):
                self.data = data
                self.id = data.get("job_id", "modal_unknown")

        try:
            job_id = payload.get("job_id")
            print(f"[{job_id}] Received job payload")

            job_data = {
                "jobId": job_id,
                "inputAudioAssetId": payload.get("asset_id"),
                "userId": payload.get("user_id"),
                "bgPrompt": payload.get("prompt"),
                "styleId": payload.get("style"),
                "startTime": 0,
                "endTime": 60,
                "targetLanguage": payload.get("targetLanguage"),
                "callback_url": payload.get("callback_url")
            }

            # Parallel Generator Hook (Distributed Async)
            async def modal_parallel_generator(scenes, durations, tmp_dir, fps=24, width=720, height=1280):
                print(f"[{job_id}] ⚡ Modal Distributed: Launching {len(scenes)} parallel GPU workers...")
                
                # Starmap arguments with dynamic res/fps
                args = [(job_id, i, len(scenes), scenes, durations, fps, width, height) for i in range(len(scenes))]
                
                # Execute concurrently on remote GPUs using async generator
                results = []
                async for res in self.generate_scene.starmap.aio(args):
                    results.append(res)
                
                # Re-assemble bytes into local orchestrator filesystem
                clip_paths = []
                for i, res_dict in enumerate(results):
                    interp_path = os.path.join(tmp_dir, f"scene_{i}.mp4")
                    raw_path = os.path.join(tmp_dir, f"raw_scene_{i}.mp4")
                    
                    with open(interp_path, "wb") as f:
                        f.write(res_dict["interp"])
                    with open(raw_path, "wb") as f:
                        f.write(res_dict["raw"])
                    
                    clip_paths.append(interp_path)
                
                print(f"[{job_id}] Distributed assembly complete. {len(clip_paths)} clips ready.")
                return clip_paths

            main = self.main
            await main.process_job(MockJob(job_data), None, parallel_generator=modal_parallel_generator)

            # --- CHAINED EXECUTION ---
            print(f"[{job_id}] Audio done. Triggering Cloud Render...")
            try:
                render_func = modal.Function.from_name("video-jobs-render", "render_video_kernel")
                render_func.spawn(payload)
            except Exception as e:
                print(f"[{job_id}] Render trigger failed: {e}")

            return {"status": "success", "job_id": job_id, "render_triggered": True}

        except Exception as e:
            print(f"Processing failed: {e}")
            return {"status": "error", "message": str(e)}


@app.local_entrypoint()
def test_build():
    """Dummy entrypoint to trigger streaming build logs."""
    print("Build triggered successfully. If you see this, the image is valid.")

