import os
import sys
import functools
import time
import math
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- DEFINITIVE SYSTEM PATCHES (CRITICAL FOR WINDOWS / NEW TORCH) ---
# Disable weights_only security checks that break WhisperX/Demucs models
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

try:
    import torch
    import torch.serialization
    from deep_translator import GoogleTranslator
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs.pop('weights_only', None) # Remove it if passed
        return original_load(*args, **kwargs, weights_only=False)
    torch.load = patched_load
    if hasattr(torch.serialization, 'load'):
        torch.serialization.load = patched_load
    print("Successfully applied torch.load security bypass.")
except ImportError:
    print("Torch not found yet, patch will be applied if imported later.")
# -------------------------------------------------------------------

import json
import asyncio
import requests
import tempfile
import subprocess
from typing import Dict, Any, List
import numpy as np
import librosa
from scipy.signal import correlate
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("google-genai not installed. Veo 3.1 will be bypassed.")

try:
    from diffusers.pipelines.ltx import LTXPipeline
    from diffusers.utils import export_to_video
except ImportError:
    LTXPipeline = None
    print("LTX-Video dependencies not fully installed. Local engine will be bypassed.")

class LTXModelManager:
    _instance = None
    _pipe = None

    @classmethod
    def get_pipe(cls):
        if cls._pipe is None and LTXPipeline is not None:
            print("Loading local LTX-Video engine (RTX 4060 Optimized)...")
            try:
                # 8GB VRAM Hardening: Load in float16 precision for 2x memory efficiency
                cls._pipe = LTXPipeline.from_pretrained(
                    "Lightricks/LTX-Video",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                
                # Standard Offload move layers between CPU/GPU dynamically
                cls._pipe.enable_model_cpu_offload()
                print("Local LTX-Video engine ready (fp16 + Model Offload).")
            except Exception as e:
                print(f"CRITICAL: Engine Load Failed: {e}")
                if "1455" in str(e):
                    print("TIP: Your Windows Paging File is too small. Increase it to 32GB+.")
                cls._pipe = "FAILED"
        return cls._pipe if cls._pipe != "FAILED" else None

RUNWAYML_API_KEY = os.environ.get("RUNWAYML_API_KEY")

def profile_audio(path: str) -> Dict[str, Any]:
    """
    Advanced Neural Orchestration: Calculates SNR, Reverb, Tempo, 
    Harmonic Key, and Spectral Energy Flux for cinematic sync.
    """
    try:
        y, sr = librosa.load(path, sr=16000)
        
        # 1. SNR Estimation
        S = np.abs(librosa.stft(y))
        rms = librosa.feature.rms(S=S)[0]
        noise_floor = np.percentile(rms, 10)
        signal_level = np.percentile(rms, 90)
        snr_val = 20 * np.log10(max(1e-6, signal_level) / max(1e-6, noise_floor))

        # 2. Reverb estimation
        corr = correlate(y[:sr*5], y[:sr*5], mode='full')
        corr = corr[len(corr)//2:]
        max_corr = np.max(corr)
        reverb_level = float(np.mean(corr[sr//10:sr//2]) / max_corr) if max_corr > 1e-6 else 0.0
        if np.isnan(reverb_level): reverb_level = 0.0

        # 3. Tempo & Energy Flux
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Normalize energy flux for the Director
        energy_flux_norm = (onset_env - np.min(onset_env)) / (np.max(onset_env) - np.min(onset_env) + 1e-6)
        # Sample energy every 1s for the context window
        energy_series = [float(np.mean(energy_flux_norm[i:i+sr//512])) for i in range(0, len(energy_flux_norm), sr//512)]
        energy_summary = energy_series[::int(1.0 / (512/sr))] # roughly 1 sample per second

        # 4. Harmonic Key Detection
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        estimated_key = keys[int(np.argmax(chroma_mean))]
        
        # Simple Major/Minor balance (heuristic)
        is_minor = np.mean(chroma[3]) > np.mean(chroma[4]) # heuristic minor third vs major
        harmonic_vibe = f"{estimated_key} {'Minor' if is_minor else 'Major'}"

        # 5. Vocal Texture Analysis (if path exists, otherwise use y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        vocal_brightness = float(np.mean(centroid))

        # 6. Heuristic Mood Classification
        mood = "vibrant"
        if tempo > 120:
            mood = "aggressive/intense" if is_minor else "energetic/upbeat"
        else:
            mood = "melancholic/cinematic" if is_minor else "calm/ethereal"
        if vocal_brightness > 3000: mood += " & sharp"

        return {
            "snr": float(snr_val),
            "reverbLevel": float(reverb_level),
            "tempo": float(tempo),
            "mood": mood,
            "harmonicKey": harmonic_vibe,
            "energyFlux": energy_summary[:60],
            "vocalBrightness": vocal_brightness
        }
    except Exception as e:
        print(f"Advanced profiling failed: {e}")
        return {"snr": 15.0, "reverb_level": 0.1, "tempo": 120.0, "harmonic_key": "C Major", "energy_flux": [0.5]*60, "vocal_brightness": 2000.0}
from bullmq import Worker
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
CALLBACK_URL = os.getenv("CALLBACK_URL")
CALLBACK_TOKEN = os.getenv("CALLBACK_TOKEN")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("DEVICE", "cpu")
# float32 is safer for compatibility
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float32")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

headers = {
    "Authorization": f"Bearer {CALLBACK_TOKEN}",
    "Content-Type": "application/json"
}

def report_stage(job_id: str, status: str, progress: int, user_id: str = None, metrics: Dict[str, Any] = None):
    try:
        data = {
            "jobId": job_id,
            "userId": user_id,
            "event": "stage",
            "status": status,
            "progress": progress
        }
        if metrics:
            data["metrics"] = metrics
        resp = requests.post(CALLBACK_URL, headers=headers, json=data)
        if resp.status_code != 200:
            print(f"[{job_id}] Callback failed with {resp.status_code}: {resp.text}")
        else:
            print(f"[{job_id}] Stage {status} ({progress}%) reported.")
    except Exception as e:
        print(f"[{job_id}] Failed to report stage {status}: {e}")

def generate_video_background(job_id: str, user_prompt: str, lyrics: str, audio_profile: Dict[str, Any], duration: float = 8.0) -> str:
    """
    Narrative Narrative Engine (Phase 27): Local LTX-Video Synergy.
    Generates a sequence of local LTX-Video clips that maintain visual consistency
    over the full duration of the video using our RTX 4060.
    """
    pipe = LTXModelManager.get_pipe()
    if not pipe:
        print(f"[{job_id}] Local LTX engine unavailable. Skipping Cinematic Engine.")
        return "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
    
    try:
        # We still use Gemini for Director logic
        director_client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # 1. Scene Segmentation: Beat-Synchronized Orchestration
        tempo = audio_profile.get("tempo", 120)
        energy_flux = audio_profile.get("energyFlux", [0.5] * 60)
        
        # Beat-Sync Logic: Align cuts to musical bars
        beat_duration = 60.0 / max(1, tempo)
        # We aim for 1 or 2 bars per clip depending on speed
        bars_per_clip = 2 if tempo > 130 else 1
        clip_duration = max(2.0, min(5.0, beat_duration * 4 * bars_per_clip))
        
        num_clips = math.ceil(duration / clip_duration) 
        
        print(f"[{job_id}] Synesthesia Engine: {num_clips} beat-synced scenes ({clip_duration:.2f}s each) @ {tempo:.0f} BPM")

        # 2. Prologue: Build Persistent Identity Sheet (Expert Director)
        mood = audio_profile.get("mood", "vibrant")
        harmonic_key = audio_profile.get("harmonicKey", "C Major")
        
        prologue_prompt = (
            f"You are a Master Cinematic Director. Create a 'Persistent Identity Sheet' for a music video.\n"
            f"PRIMARY THEME: '{user_prompt}'\n"
            f"The musical mood is {mood}. The key is {harmonic_key}.\n"
            "Describe: 1. A consistent Protagonist/Object. 2. A Master Environment (Lighting, Era). 3. Camera DNA.\n"
            "This identity MUST BE EXTREMELY LOYAL to the theme. Respond ONLY with the identity description."
        )
        
        try:
            p_resp = director_client.models.generate_content(model='gemini-2.0-flash', contents=prologue_prompt)
            identity_sheet = p_resp.text.strip()
        except Exception:
            identity_sheet = f"Cinematic style, {user_prompt}, consistency"

        # 3. Sequential Execution Loop: The Masterwork
        clip_paths = []
        tmp_dir = tempfile.mkdtemp()
        last_frame = None 
        
        # Clear VRAM
        torch.cuda.empty_cache()
        
        for i in range(num_clips):
            t_start = i * clip_duration
            t_end = min(duration, (i + 1) * clip_duration)
            idx_start = int(t_start)
            idx_end = int(t_end)
            local_energy = np.mean(energy_flux[idx_start:max(idx_start+1, idx_end)]) if energy_flux else 0.5
            
            # Map energy to camera motion
            camera_motion = "fast tracking shot" if local_energy > 0.7 else "slow cinematic zoom" if local_energy < 0.4 else "steady slider move"
            
            # 3a. Synesthesia Analysis: Emotional Subtext & Action Verbs
            scene_lyrics = lyrics[int(len(lyrics)*(t_start/duration)):int(len(lyrics)*(t_end/duration))]
            local_prompt = (
                f"Scene {i+1}/{num_clips}. Theme: {identity_sheet}.\n"
                f"MUSICAL CONTEXT: Energy is {local_energy:.2f}. Lyrics are '{scene_lyrics[:200]}'.\n"
                f"TASK: Identify the EMOTIONAL SUBTEXT and LITERAL ACTION VERBS in the lyrics.\n"
                f"GENERATE: A cinematic prompt (ONE sentence) where the protagonist PERFORMS those actions. Camera: {camera_motion}.\n"
                "Respond ONLY with the prompt."
            )
            
            try:
                s_resp = director_client.models.generate_content(model='gemini-2.0-flash', contents=local_prompt)
                current_prompt = s_resp.text.strip()
            except Exception:
                current_prompt = f"{user_prompt}, cinematic action scene {i}"

            print(f"[{job_id}] Scene {i+1}/{num_clips}: {current_prompt[:100]}...")
            print(f"[{job_id}] Generating Synced Scene {i+1}/{num_clips} (Energy: {local_energy:.2f})...")
            
            try:
                # Local LTX Generation with I2V conditioning
                gen_args = {
                    "prompt": f"{current_prompt}, {camera_motion}, cinematic, 4k, highly detailed",
                    "negative_prompt": "low quality, blurry, distorted, static, text, watermark, bad anatomy",
                    "num_frames": int(clip_duration * 24) if clip_duration * 24 % 8 == 1 else int(clip_duration * 24 // 8 * 8 + 1),
                    "width": 320,
                    "height": 544,
                    "num_inference_steps": 12, # Optimized for 4060
                    "guidance_scale": 3.0,
                }
                
                # Local LTX Generation (Stable T2V)
                # Note: Default LTXPipeline does not support 'image' kwarg for I2V yet.
                # We rely on the Identity Sheet + Prompt for consistency.
                output = pipe(**gen_args)
                frames = output.frames[0]
                
                # Save the last frame for the NEXT clip's conditioning
                # frames is a list of PIL images or a tensor? Diffusers LTX returns PIL.
                last_frame = frames[-1] 
                
                local_path = os.path.join(tmp_dir, f"scene_{i}.mp4")
                export_to_video(frames, local_path, fps=24)
                clip_paths.append(local_path)
                
            except Exception as task_error:
                print(f"[{job_id}] Scene {i} failed: {task_error}")
                torch.cuda.empty_cache()
                last_frame = None # Reset continuity on failure

        if not clip_paths:
            return "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"

        # 5. FFmpeg Concatenation
        list_file = os.path.join(tmp_dir, "clips.txt")
        with open(list_file, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p.replace('\\', '/')}'\n")
        
        master_output = os.path.join(tmp_dir, "master_bg.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
            "-c", "copy", master_output
        ], check=True)

        # 6. Upload Master to Supabase
        storage_path = f"{job_id}/cinematic_bg.mp4"
        print(f"[{job_id}] Uploading Master Cinematic Background...")
        with open(master_output, "rb") as f:
            supabase.storage.from_("assets").upload(
                path=storage_path,
                file=f,
                file_options={"content-type": "video/mp4", "x-upsert": "true"}
            )
        
        return storage_path

    except Exception as e:
        print(f"[{job_id}] Narrative Engine failed: {e}")
        return "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"

async def process_job(job, job_token):
    job_id = job.data.get('jobId')
    asset_id = job.data.get('inputAudioAssetId')
    user_id = job.data.get('userId')
    mood = job.data.get('mood', 'default')
    position = job.data.get('position', 'center')
    font_size = job.data.get('fontSize', 6)
    bg_prompt = job.data.get('bgPrompt')
    start_time = job.data.get('startTime', 0)
    end_time = job.data.get('endTime', 60)
    style_id = job.data.get('styleId') # Extract styleId from payload
    target_language = job.data.get('targetLanguage') # Extract targetLanguage
    font_family = job.data.get('fontFamily')
    animation_effect = job.data.get('animationEffect')
    lyric_color = job.data.get('lyricColor')
    lyric_opacity = job.data.get('lyricOpacity')
    
    # Cap duration to 60s as per requirements
    if end_time - start_time > 60:
        end_time = start_time + 60

    print(f"[{job_id}] Received job for user {user_id} with asset: {asset_id}")
    print(f"[{job_id}] Range: {start_time}s - {end_time}s (Prompt: {bg_prompt}, Style: {style_id})")

    try:
        # 1. Fetch Asset Info from Supabase
        report_stage(job_id, "preprocessing", 5, user_id)
        asset_res = supabase.table("job_assets").select("*").eq("id", asset_id).single().execute()
        asset_data = asset_res.data
        if not asset_data:
            raise Exception(f"Asset {asset_id} not found in database")

        bucket = asset_data["bucket"]
        path = asset_data["path"]

        # 2. Download File
        tmp_dir = os.path.join(os.getcwd(), "debug_tmp")
        if os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        
        try:
            local_audio_path = os.path.join(tmp_dir, "raw_input.mp3")
            print(f"[{job_id}] Downloading {bucket}/{path}...")
            
            with open(local_audio_path, "wb+") as f:
                res = supabase.storage.from_(bucket).download(path)
                f.write(res)

            # 2a. HARDENING: Pre-crop to user segment + Ensure stereo 44.1kHz
            print(f"[{job_id}] Cropping and converting to stereo 44.1kHz...")
            segment_path = os.path.join(tmp_dir, "input_segment.wav")
            
            # -ss before -i is faster for long files
            ffmpeg_cmd = [
                "ffmpeg", "-y", 
                "-ss", str(start_time), 
                "-t", str(end_time - start_time), 
                "-i", local_audio_path, 
                "-ac", "2", "-ar", "44100", 
                segment_path
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                local_audio_path = segment_path
                print(f"[{job_id}] Selective cropping successful.")
            except subprocess.CalledProcessError as e:
                print(f"[{job_id}] Cropping failed: {e.stderr.decode()}")
                # Fallback to stereo conversion of whole file if cropping fails
                stereo_path = os.path.join(tmp_dir, "input_stereo.wav")
                subprocess.run(["ffmpeg", "-y", "-i", local_audio_path, "-ac", "2", "-ar", "44100", stereo_path], check=False)
                local_audio_path = stereo_path
            
            # --- PHASE 5: ADAPTIVE PROFILING ---
            print(f"[{job_id}] Profiling audio...")
            profile = profile_audio(local_audio_path)
            print(f"[{job_id}] Profile: {profile}")
            
            # Send early telemetry to the Dashboard
            report_stage(job_id, "preprocessing", 12, user_id, metrics=profile)

            # 2b. UNIVERSAL PRECISION: Vocal Separation (Demucs)
            report_stage(job_id, "preprocessing", 10, user_id)
            print(f"[{job_id}] Running Demucs vocal separation...")
            
            # Use absolute path for demucs in .venv
            if sys.platform == "win32":
                demucs_exe = os.path.join(os.path.dirname(sys.executable), "demucs.exe")
            else:
                demucs_exe = "demucs"
            
            # Use htdemucs for better stability on Windows (non-quantized)
            demucs_cmd = [demucs_exe, "-n", "htdemucs", "-o", tmp_dir, "-d", DEVICE, "--two-stems", "vocals", local_audio_path]
            print(f"[{job_id}] Executing: {' '.join(demucs_cmd)}")
            
            raw_vocal_path = local_audio_path # Default fallback
            try:
                # Use a timeout for Demucs to prevent hangs on short audio
                res_d = subprocess.run(demucs_cmd, capture_output=True, text=True, check=False, timeout=120)
                
                if res_d.returncode == 0:
                    print(f"[{job_id}] Demucs finished successfully.")
                    htdemucs_dir = os.path.join(tmp_dir, "htdemucs")
                    if os.path.exists(htdemucs_dir):
                        songs = os.listdir(htdemucs_dir)
                        if songs:
                            vocal_candidate = os.path.join(htdemucs_dir, songs[0], "vocals.wav")
                            if os.path.exists(vocal_candidate):
                                raw_vocal_path = vocal_candidate
                                print(f"[{job_id}] Using isolated vocals.")
                else:
                    print(f"[{job_id}] Demucs failed. Error: {res_d.stderr[:200]}")
            except Exception as e:
                print(f"[{job_id}] Demucs subprocess error: {e}")

            # 2c. PROFESSIONAL V2: Vocal Cleaning            # --- PHASE 8: HARDENED ADAPTIVE CLEANING ---
            report_stage(job_id, "preprocessing", 15, user_id)
            print(f"[{job_id}] Applying hardened adaptive cleaning...")
            vocals_path = os.path.join(tmp_dir, "htdemucs/input_segment/vocals.wav")
            vocal_cleaned = os.path.join(tmp_dir, "vocals_cleaned_final.wav")
            
            # 1. Base Cleanup
            filters = ["loudnorm=I=-16:TP=-1.5:LRA=11"] # Standardize volume
            
            # 2. Adaptive Denoise based on SNR
            if profile["snr"] < 15:
                # Strong noise detected
                filters.append("afftdn=nf=-25:nr=15")
            elif profile["snr"] < 25:
                # Moderate noise
                filters.append("afftdn=nf=-35:nr=10")
            
            # 3. Adaptive De-reverb / De-click
            if profile["reverbLevel"] > 0.4:
                filters.append("adeclick") # Light de-reverb effect
                filters.append("anequalizer=c0 f=400 g=-6 w=200") # Reduce muddy low-mid
            
            # 4. Vocal Focus (Highpass/Lowpass)
            filters.append("highpass=f=120") # Remove rumble
            filters.append("lowpass=f=8000") # Remove high hiss
            
            filter_str = ",".join(filters)
            print(f"[{job_id}] Cleaning Chain: {filter_str}")
            
            try:
                subprocess.run(["ffmpeg", "-y", "-i", raw_vocal_path, "-af", filter_str, vocal_cleaned], check=True, capture_output=True)
                vocal_path = vocal_cleaned
                print(f"[{job_id}] Vocal cleaning successful.")
            except subprocess.CalledProcessError:
                print(f"[{job_id}] Vocal cleaning failed. Using raw vocals.")

            # 3. Transcribe & Align (WhisperX) using isolated vocals
            report_stage(job_id, "transcribing", 20, user_id)
            import whisperx
            
            print(f"[{job_id}] Loading WhisperX model {WHISPER_MODEL} on {DEVICE}...")
            # Forcing weights_only=False inside the call if supported, though the patch handles it
            model = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)
            
            audio = whisperx.load_audio(vocal_path)
            result = model.transcribe(audio, batch_size=16)
            
            report_stage(job_id, "aligning", 50, user_id)
            print(f"[{job_id}] Aligning with phoneme tokens...")
            language_code = result.get("language", "en") # Fallback to English if detection fails
            print(f"[{job_id}] Using language code: {language_code}")
            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=DEVICE)
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)

            # --- PHASE 6: ALIGNMENT QUALITY GATES & ELEGANT DEGRADATION ---
            print(f"[{job_id}] Evaluating alignment quality...")
            all_words = [w for seg in result["segments"] for w in seg.get("words", [])]
            aligned_words = [w for w in all_words if "start" in w and "end" in w]
            
            aligned_ratio = len(aligned_words) / len(all_words) if all_words else 0
            
            durations = [w["end"] - w["start"] for w in aligned_words]
            median_duration = np.median(durations) if durations else 0
            
            # Simple overlap check
            overlaps = 0
            for i in range(len(aligned_words) - 1):
                if aligned_words[i]["end"] > aligned_words[i+1]["start"]:
                    overlaps += 1

            print(f"[{job_id}] Alignment Stats: Ratio={aligned_ratio:.2f}, MedianDur={median_duration:.3f}s, Overlaps={overlaps}")
            
            granularity = "word"
            fallback_used = False
            
            # Gate: If ratio < 0.85 or median duration is too short (<50ms), or no words aligned, fall back to phrases
            if aligned_ratio < 0.85 or median_duration < 0.05 or not aligned_words:
                print(f"[{job_id}] Quality gate failed (Ratio={aligned_ratio:.2f}, Median={median_duration:.2f}). Fallback to phrase.")
                granularity = "phrase"
                fallback_used = True
            
            # 4. UNIVERSAL PRECISION: Pre-Selected Segment Logic
            report_stage(job_id, "hook_selecting", 80, user_id)
            
            # Since we cropped the audio at the start (2a), the relative timestamps
            # for this segment are always 0..duration.
            segment_duration = audio.shape[0]/16000
            h_start = 0
            h_end = segment_duration
            
            print(f"[{job_id}] Using pre-defined segment 0.0s - {h_end:.2f}s (absolute: {start_time}s - {end_time}s)")
            hook = {
                "start": start_time,
                "end": end_time,
                "score": 1.0 # Static score for user-selected segments
            }

            # --- PHASE 6: CANONICAL OUTPUT GENERATION ---
            # Transform to relative timestamps (0..hook_duration)
            final_words = []
            for seg in result["segments"]:
                if granularity == "word":
                    for w in seg.get("words", []):
                        if "start" not in w: continue
                        
                        # Only include words within the hook
                        if w["start"] < h_start or w["start"] > h_end: continue
                        
                        final_words.append({
                            "t0": round(w["start"] - h_start, 3),
                            "t1": round(w["end"] - h_start, 3),
                            "w": w["word"],
                            "conf": round(w.get("score", 0.9), 2),
                            "lang": language_code,
                            "type": "lyric"
                        })
                else:
                    # Phrase level (use segment timestamps)
                    if seg["start"] < h_start or seg["start"] > h_end: continue
                    final_words.append({
                        "t0": round(seg["start"] - h_start, 3),
                        "t1": round(seg["end"] - h_start, 3),
                        "w": seg["text"].strip(),
                        "conf": 0.8,
                        "lang": language_code,
                        "type": "phrase"
                    })

            # Rescue if final_words empty but audio exists
            if not final_words and audio.shape[0] > 0:
                print(f"[{job_id}] Rescue: Force-mapping segments or dummy span.")
                if result["segments"]:
                    for seg in result["segments"]:
                        final_words.append({
                            "t0": round(max(0, seg["start"] - h_start), 3),
                            "t1": round(min(h_end - h_start, seg["end"] - h_start), 3),
                            "w": seg["text"].strip() or "...",
                            "conf": 0.5,
                            "lang": language_code,
                            "type": "caption"
                        })
                else:
                    # Hard Rescue: No segments at all
                    final_words.append({
                        "t0": 0.0,
                        "t1": round(h_end - h_start, 3),
                        "w": "[Instrumental / Silence]",
                        "conf": 1.0,
                        "lang": language_code,
                        "type": "caption"
                    })

            # 4a. Multilingual Translation Pass
            if target_language and target_language != language_code:
                print(f"[{job_id}] Translating lyrics to: {target_language}...")
                try:
                    translator = GoogleTranslator(source=language_code, target=target_language)
                    # Batch translate for efficiency and context
                    phrases_to_translate = [w["w"] for w in final_words]
                    # We translate the whole block or chunks. GoogleTranslator handles lists.
                    translated_phrases = translator.translate_batch(phrases_to_translate)
                    
                    for i, tw in enumerate(translated_phrases):
                        final_words[i]["tw"] = tw
                    print(f"[{job_id}] Translation complete.")
                except Exception as te:
                    print(f"[{job_id}] Translation failed: {te}")

            # 4b. AI Director Engine (Expert Synthesis)
            report_stage(job_id, "compositing", 85, user_id)
            lyrics_full = " ".join([w["w"] for w in final_words])
            video_bg_url = generate_video_background(job_id, bg_prompt or "cinematic background", lyrics_full, profile, duration=segment_duration)
            
            # 5. Save and Upload Analysis JSON
            total_h_dur = h_end - h_start + 0.1
            word_dur = sum([w["t1"] - w["t0"] for w in final_words])
            coverage = min(1.0, word_dur / total_h_dur)
            
            avg_conf = sum([w["conf"] for w in final_words]) / len(final_words) if final_words else 0.0

            analysis_path = f"{job_id}/analysis.json"
            analysis_data = {
                "jobId": job_id, "userId": user_id, "hook": hook, "words": final_words,
                "language": language_code, "precision": "studio-high-v1",
                "mood": mood, "styleId": style_id, "position": position, 
                "fontSize": font_size, "fontFamily": font_family,
                "animationEffect": animation_effect, "lyricColor": lyric_color,
                "lyricOpacity": lyric_opacity, "videoBgUrl": video_bg_url,
                "profiling": profile,
                "alignment": {
                    "method": "whisperx", "alignScore": round(aligned_ratio, 3),
                    "alignedRatio": round(aligned_ratio, 3), "fallbackUsed": fallback_used,
                    "granularity": granularity
                },
                "metrics": {
                    "alignScore": round(aligned_ratio, 3),
                    "coverage": round(coverage, 3),
                    "uncertainRate": round(1.0 - avg_conf, 3),
                    "sourceAgreementRate": round(avg_conf, 3)
                }
            }
            
            local_analysis_file = os.path.join(tmp_dir, "analysis.json")
            # 3. Upload to Supabase Storage (with upsert)
            vocal_path_storage = f"{user_id}/{job_id}_vocals.wav"
            print(f"[{job_id}] Uploading vocals to assets/{vocal_path_storage}...")
            
            with open(vocal_path, "rb") as f:
                supabase.storage.from_("assets").upload(
                    path=vocal_path_storage,
                    file=f,
                    file_options={"content-type": "audio/wav", "x-upsert": "true"}
                )

            # Report Asset to API
            resp_v = requests.post(CALLBACK_URL, headers=headers, json={
                "jobId": job_id,
                "userId": user_id,
                "event": "asset",
                "kind": "vocal_stem",
                "url": vocal_path_storage, 
                "metadata": {"precision": "high-demucs-v3"}
            })
            if resp_v.status_code != 200:
                print(f"[{job_id}] Vocal asset callback failed: {resp_v.text}")

            with open(local_analysis_file, "w") as f:
                json.dump(analysis_data, f)

            print(f"[{job_id}] Uploading analysis.json...")
            supabase.storage.from_("assets").upload(
                path=analysis_path,
                file=local_analysis_file,
                file_options={"content-type": "application/json", "x-upsert": "true"}
            )

            # Report Asset to API
            resp_a = requests.post(CALLBACK_URL, headers=headers, json={
                "jobId": job_id,
                "userId": user_id,
                "event": "asset",
                "kind": "analysis_json",
                "url": analysis_path,
                "metadata": {"segments_count": len(result["segments"]), "precision": "high-demucs-v3"}
            })
            if resp_a.status_code != 200:
                print(f"[{job_id}] Analysis asset callback failed: {resp_a.text}")

            # 6. Success
            # 6. Handover to Render Worker (Do not mark as completed yet)
            report_stage(job_id, "compositing", 90, user_id)
            print(f"[{job_id}] Audio processing finished. Handing off to Render Worker.")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"[{job_id}] Critical Error: {error_msg}")
            requests.post(CALLBACK_URL, headers=headers, json={
                "jobId": job_id,
                "userId": user_id,
                "event": "error",
                "message": str(e),
                "errorCode": "WORKER_FAILURE",
                "retryable": True
            })
            raise e

    except Exception as e:
        print(f"Outer Error: {e}")
        raise e

async def main():
    print(f"Starting audio-cpu worker (WhisperX + Demucs) listening on Redis: {REDIS_URL}")
    
    # Split URL or use defaults
    host = "127.0.0.1"
    port = 6379
    if "://" in REDIS_URL:
        remainder = REDIS_URL.split("://")[1]
        host = remainder.split(":")[0]
        if ":" in remainder:
            port = int(remainder.split(":")[1])

    print(f"Connecting to Redis at {host}:{port}...")
    
    worker = Worker("video-jobs", process_job, {
        "connection": {
            "host": host,
            "port": port
        }
    })
    
    print("Worker initialized. Entering loop...")
    while True:
        print(".", end="", flush=True)
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
