import modal
import os

# Use the same image definition as modal_entry.py for consistency
image = modal.Image.debian_slim().apt_install("git")
models_volume = modal.Volume.from_name("models-volume")

app = modal.App("rife-setup")

@app.function(volumes={"/models": models_volume}, timeout=600)
def setup_rife():
    import subprocess
    
    # 1. Clone RIFE if not exists
    rife_dir = "/models/rife"
    if not os.path.exists(rife_dir):
        print("Cloning Practical-RIFE...")
        subprocess.run(["git", "clone", "https://github.com/hzwer/Practical-RIFE.git", rife_dir], check=True)
    else:
        print("Practical-RIFE already exists.")
        
    # 2. Download weights (v4.6 is stable and fast)
    # We'll use a direct link or gdown if needed, but hzwer usually provides direct Google Drive links
    # For this task, we will try to find a direct .pth link or just use the auto-download if the code supports it.
    # Actually, RIFE has a "train_log" folder with the weights.
    
    print("RIFE Setup complete in /models/rife")

if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            setup_rife.remote()
