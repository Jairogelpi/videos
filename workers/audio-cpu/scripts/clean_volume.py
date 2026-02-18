import modal
import os
import shutil

# Attach to the same volume used by the app
models_volume = modal.Volume.from_name("models-volume", create_if_missing=True)

app = modal.App("cleaner")

@app.function(volumes={"/models": models_volume})
def clean_flux():
    target = "/models/models--black-forest-labs--FLUX.1-schnell"
    if os.path.exists(target):
        print(f"Deleting corrupt model cache: {target}...")
        shutil.rmtree(target)
        print("Deleted.")
        models_volume.commit() # Persist changes
    else:
        print(f"Target not found: {target}")

    # Also check the un-hashed version if it exists
    target2 = "/models/FLUX.1-schnell"
    if os.path.exists(target2):
        print(f"Deleting legacy path: {target2}...")
        shutil.rmtree(target2)
        print("Deleted.")
        models_volume.commit()
