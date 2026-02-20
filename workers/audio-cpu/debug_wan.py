import modal
import os
import sys

# Connect to the exact same image and volume used by audio-cpu
try:
    from workers.audio_cpu.modal_entry import image, models_volume
except ImportError:
    # Build it on the fly if import fails
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("ffmpeg", "git")
        .pip_install("torch==2.6.0", "torchvision", "torchaudio", extra_options="--index-url https://download.pytorch.org/whl/cu124")
        .pip_install("diffusers==0.32.2", "transformers==4.49.0", "accelerate==1.4.0", "sentencepiece", "protobuf")
        .pip_install("ftfy", "numpy", "scipy")
    )
    models_volume = modal.Volume.from_name("models-volume")

app = modal.App("wan-debug-texture", image=image)

@app.function(
    gpu="L4",
    timeout=1200,
    volumes={"/models": models_volume},
)
def debug_wan_generation():
    import torch
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video
    import time
    
    print("--- 🕵️‍♂️ WAN 2.1 TEXTURE DEBUGGER ---")
    
    # 1. Pipeline Setup (Matching our final main.py exactly)
    model_path = "/models/Wan2.1-T2V-1.3B-Diffusers"
    if not os.path.exists(model_path):
        model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        
    print(f"Loading from: {model_path}")
    pipe = WanPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16, # Pure float16
        local_files_only=os.path.exists(model_path)
    )
    
    # Simulate our exact CPU offload and VAE tiling setup
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
        print("VAE Tiling Enabled.")
    except Exception as e:
        print(f"VAE Tiling Failed: {e}")
        
    # 2. Test Prompts (1 Concrete, 1 Default)
    prompts = [
        "Photorealistic cinematic shot of Silas, reaching the mountain peak under a lighter grey sky, placing a small, smooth, deep indigo river stone on the summit, a sense of quiet resolution on his deeply lined face, static shot, highly detailed, 3d, realistic materials, volumetric lighting, sharp focus, masterpiece",
        "A highly detailed photograph of a cat sitting on a wooden table, volumetric lighting, sharp focus, masterpiece"
    ]
    
    negative_prompt = (
        "abstract textures, blurry, noise, static, low quality, cartoon, flat colors, "
        "distorted faces, messy patterns, wallpaper, glitch, flickering, incoherent objects, "
        "grainy, placeholder, grey blob"
    )
    
    results = {}
    
    # 3. Execution
    for i, prompt in enumerate(prompts):
        print(f"\nGenerando Test {i+1}...")
        print(f"Prompt: {prompt}")
        
        start_time = time.time()
        
        # Test 1: Our EXACT settings (480x832, 33 frames, 30 steps, 3.5 CFG)
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=480,
            height=832,
            num_frames=33,
            num_inference_steps=30,
            guidance_scale=3.5 
        )
        
        frames = output.frames[0]
        gen_time = time.time() - start_time
        print(f"Generation complete in {gen_time:.1f}s")
        
        # Save locally in the container
        file_path = f"/tmp/debug_raw_{i}.mp4"
        export_to_video(frames, file_path, fps=16)
        
        with open(file_path, "rb") as f:
            results[f"debug_raw_{i}.mp4"] = f.read()
            
    return results

@app.local_entrypoint()
def run_debug():
    print("Launching Remote WAN Debugger...")
    results = debug_wan_generation.remote()
    
    print("\n--- 📥 DOWNLOADING RESULTS ---")
    for filename, data in results.items():
        local_path = os.path.join(os.getcwd(), filename)
        with open(local_path, "wb") as f:
            f.write(data)
        print(f"Saved: {local_path}")
        
    print("✅ Debug complete. Please inspect the MP4 files.")
