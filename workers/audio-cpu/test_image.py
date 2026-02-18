import modal
from modal_entry import image

app = modal.App("test-build-v2", image=image)

@app.function(gpu="T4")
def remote_test():
    import torch
    import diffusers
    
    # Import main to verify no RecursionError in torch.load
    try:
        sys.path.append("/root/workers/audio-cpu")
        import main
        print("✅ Main module imported successfully (No RecursionError)")
    except Exception as e:
        print(f"❌ Main module import failed: {e}")
        raise
    
    print(f"--- REMOTE VERIFICATION ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Verify WhisperX (Critical)
    try:
        import whisperx
        print(f"✅ WhisperX imported successfully!")
        
        # Explicit ABI Check for libraries that often fail
        import torchaudio
        print(f"✅ Torchaudio imported successfully! Version: {torchaudio.__version__}")
        
        import pyannote.audio
        print(f"✅ Pyannote.audio imported successfully!")
        
    except ImportError as e:
        print(f"❌ WhisperX/Torchaudio import failed: {e}")
        raise
    except OSError as e:
        print(f"❌ ABI Mismatch detected (OSError): {e}")
        raise

    # Verify LTX-2
    try:
        from diffusers import LTX2Pipeline
        print(f"✅ LTX-2 Pipeline imported successfully!")
    except ImportError as e:
        print(f"❌ LTX-2 import failed: {e}")
        raise

    # Verify Security Bypass (Legacy Checkpoint Simulation)
    print("--- TESTING LEGACY CHECKPOINT LOAD ---")
    try:
        class LegacyObject:
            def __init__(self):
                self.data = "secret"
        
        obj = LegacyObject()
        # Save naturally (defaults to pickle)
        torch.save(obj, "legacy_test.pt")
        
        # Load without arguments - should rely on our patch to set weights_only=False
        # If patch misses, PyTorch 2.6 will block this as UnpicklingError (globals not allowed)
        loaded = torch.load("legacy_test.pt")
        print(f"✅ Security Bypass Verified! Loaded custom object: {loaded.data}")
        
    except Exception as e:
        print(f"❌ Security Bypass FAILED: {e}")
        # Identify if it's the specific error
        if "weights_only" in str(e) or "UnpicklingError" in str(e):
            print("   -> The patch is NOT active. PyTorch is blocking legacy pickles.")
        raise
        
    return True

@app.local_entrypoint()
def test():
    remote_test.remote()






