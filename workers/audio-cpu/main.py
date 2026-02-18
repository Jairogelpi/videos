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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Silence torchcodec / FFmpeg version noise
os.environ["TORCHCODEC_LOG_LEVEL"] = "ERROR"
# -------------------------------------------------------------------
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!! MAIN.PY LOADED V9 (KERNEL HANDOFF) !!!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

import json
import asyncio
import requests
import tempfile
import subprocess
from typing import Dict, Any, List, Optional
import numpy as np
import librosa
from scipy.signal import correlate
import ftfy  # Explicit import to fix 'name ftfy is not defined' in diffusers/transformers
# --- LAZY IMPORT INITIALIZATION ---
# We no longer import heavy AI libraries at the top level to ensure instant cold starts
WanPipeline = None
FluxPipeline = None
export_to_video = None
torch = None
DEVICE = "cuda"

def _ensure_ai_imports():
    global WanPipeline, FluxPipeline, export_to_video, torch, DEVICE, genai, types
    if torch is None:
        import torch as _torch
        import diffusers
        from diffusers import WanPipeline as _Wan, FluxPipeline as _Flux
        from diffusers.utils import export_to_video as _export
        from google import genai as _genai
        from google.genai import types as _types
        torch = _torch
        WanPipeline = _Wan
        FluxPipeline = _Flux
        export_to_video = _export
        genai = _genai
        types = _types
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        import ftfy


def get_gpu_memory_gb():
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except:
        pass
    return 0.0

def safe_clip_prompt(prompt, max_tokens=77, quality_tags=" cinematic movie still, masterwork, 8k, photorealistic, vertical 9:16"):
    """
    Intelligently ensures the prompt + quality_tags don't exceed CLIP token limits (approx 77).
    We assume ~1.3 tokens per word for a safe estimate.
    """
    # Conservatively allow ~40 words to safely fit within 77 tokens
    max_words = 40
    words = prompt.split()
    if len(words) > max_words:
        prompt = " ".join(words[:max_words])
    
    return prompt + quality_tags

class T5ModelManager:
    _model = None
    _tokenizer = None

    @classmethod
    def get_model(cls):
        _ensure_ai_imports()
        from transformers import T5EncoderModel, T5TokenizerFast, BitsAndBytesConfig
        if cls._model is None:
            print("Loading Shared T5-v1.1-XXL Encoder (8-bit Hypercharge)...")
            try:
                t5_path = "/models/Wan2.1-T2V-1.3B-Diffusers/text_encoder"
                if not os.path.exists(t5_path):
                    t5_path = "google/t5-v1_1-xxl"
                
                cls._tokenizer = T5TokenizerFast.from_pretrained(t5_path)
                
                # Zero-Latency: 8-bit quantization to fit everything in VRAM
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_enable_fp32_cpu_offload=False
                )
                
                cls._model = T5EncoderModel.from_pretrained(
                    t5_path,
                    quantization_config=quantization_config,
                    device_map="auto", # BitsAndBytes handles device mapping
                    torch_dtype=torch.float16,
                    variant="fp16" if "Wan2.1" in t5_path else None
                )
                print("Shared T5 Encoder (8-bit) Ready.")
            except Exception as e:
                print(f"CRITICAL: Shared T5 Load Failed: {e}")
                cls._model = "FAILED"
        return (cls._model, cls._tokenizer) if cls._model != "FAILED" else (None, None)

class WanModelManager:
    _pipe = None

    @classmethod
    def get_pipe(cls):
        _ensure_ai_imports()
        if cls._pipe is None and WanPipeline is not None:
            print("Loading Wan 2.1 (1.3B) [NEUTRON / LIGHT-SPEED]...")
            try:
                model_path = "/models/Wan2.1-T2V-1.3B-Diffusers"
                if not os.path.exists(model_path):
                    model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
                
                t5_model, t5_tokenizer = T5ModelManager.get_model()
                
                cls._pipe = WanPipeline.from_pretrained(
                    model_path,
                    text_encoder=t5_model,
                    tokenizer=t5_tokenizer,
                    torch_dtype=torch.float16,
                    local_files_only=os.path.exists(model_path)
                )
                # Zero-Latency: Stay on GPU!
                # Zero-Latency: Stay on GPU!
                cls._pipe.to("cuda")
                
                vram_gb = get_gpu_memory_gb()
                if vram_gb > 30:
                    print(f"[{vram_gb:.1f}GB VRAM] High-VRAM GPU Detected (A100/H100). Disabling VAE Tiling for Max Speed.")
                else:
                    # VAE Tiling (CRITICAL for 24GB VRAM)
                    try:
                        cls._pipe.vae.enable_tiling()
                        print("Wan 2.1 VAE Tiling Enabled (L4/Limited VRAM).")
                    except:
                        print("Wan 2.1 VAE Tiling NOT Supported (Risk of OOM).")

                torch.cuda.empty_cache()
                print(f"Wan 2.1 (GPU Resident) ready.")
            except Exception as e:
                print(f"CRITICAL: Wan Engine Load Failed: {e}")
                cls._pipe = "FAILED"
        return cls._pipe if cls._pipe != "FAILED" else None

class WhisperModelManager:
    _model = None
    
    @classmethod
    def get_model(cls):
        _ensure_ai_imports()
        import whisperx
        if cls._model is None:
            print(f"Loading WhisperX model {WHISPER_MODEL} [ZERO-CHILL] on {DEVICE}...")
            try:
                # Use baked models from /models/whisper if available
                cls._model = whisperx.load_model(
                    WHISPER_MODEL, 
                    DEVICE, 
                    compute_type=COMPUTE_TYPE,
                    download_root="/models/whisper"
                )
                print(f"WhisperX ready.")
            except Exception as e:
                print(f"CRITICAL: WhisperX Load Failed: {e}")
                cls._model = "FAILED"
        return cls._model if cls._model != "FAILED" else None

    @classmethod
    def offload(cls):
        if cls._model and cls._model != "FAILED":
            # WhisperX models are a bit complex, but we can try removing them or moving known parts
            # For now, we'll suggest manual GC. WhisperX often loads on device at init.
            # Best effort:
            pass 

RUNWAYML_API_KEY = os.environ.get("RUNWAYML_API_KEY")

def profile_audio(path: str) -> Dict[str, Any]:
    """
    HYPERCHARGED Audio Profiler (v2 - Neutron):
    - 2x faster via parallel metric computation (ThreadPoolExecutor)
    - 22050Hz sample rate for better frequency resolution  
    - 8 NEW metrics: valence, dynamic_range, rhythmic_complexity, 
      harmonic_movement, beat_positions, stereo_width, loudness_lufs, spectral_bandwidth
    """
    import concurrent.futures
    import time as _time

    try:
        t0 = _time.monotonic()
        
        # Load at 22050Hz (mono) for richer frequency resolution
        y, sr = librosa.load(path, sr=22050, mono=True)
        # Also load stereo for stereo_width analysis
        try:
            y_stereo, _ = librosa.load(path, sr=22050, mono=False)
            has_stereo = y_stereo.ndim == 2 and y_stereo.shape[0] == 2
        except:
            y_stereo = None
            has_stereo = False

        # Pre-compute shared transforms ONCE (used by multiple metric groups)
        S = np.abs(librosa.stft(y))
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # --- PARALLEL METRIC GROUPS ---
        def compute_dynamics():
            """SNR, Reverb, Dynamic Range, Loudness LUFS"""
            rms = librosa.feature.rms(S=S)[0]
            noise_floor = np.percentile(rms, 10)
            signal_level = np.percentile(rms, 90)
            snr_val = 20 * np.log10(max(1e-6, signal_level) / max(1e-6, noise_floor))

            # Reverb (autocorrelation)
            seg = y[:sr*3]
            corr = correlate(seg, seg, mode='full')
            corr = corr[len(corr)//2:]
            max_corr = np.max(corr)
            reverb_level = float(np.mean(corr[sr//10:sr//2]) / max_corr) if max_corr > 1e-6 else 0.0
            if np.isnan(reverb_level): reverb_level = 0.0

            # Dynamic Range (difference between loudest and quietest RMS in dB)
            dynamic_range = 20 * np.log10(max(1e-6, np.percentile(rms, 95)) / max(1e-6, np.percentile(rms, 5)))

            # Loudness LUFS approximation (EBU R128 simplified)
            mean_sq = float(np.mean(y**2))
            loudness_lufs = -0.691 + 10 * np.log10(max(1e-10, mean_sq))

            return {
                "snr": float(snr_val),
                "reverbLevel": float(reverb_level),
                "dynamicRange": float(dynamic_range),
                "loudnessLUFS": float(loudness_lufs)
            }

        def compute_rhythm():
            """Tempo, Beat Grid, Rhythmic Complexity, Energy Flux"""
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # Beat positions in seconds (for cinematic sync)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            beat_positions = [float(b) for b in beat_times[:120]]  # cap at 120 beats

            # Rhythmic Complexity: variance of inter-beat intervals
            if len(beat_times) > 2:
                ibis = np.diff(beat_times)
                rhythmic_complexity = float(np.std(ibis) / (np.mean(ibis) + 1e-6))
            else:
                rhythmic_complexity = 0.0

            # Energy Flux (normalized onset strength)
            if len(onset_env) > 0:
                onset_min, onset_max = np.min(onset_env), np.max(onset_env)
                energy_flux_norm = (onset_env - onset_min) / (onset_max - onset_min + 1e-6)
            else:
                energy_flux_norm = np.array([0.5])

            # Downsample to ~1 sample/second
            hop_per_sec = sr / 512
            step = max(1, int(hop_per_sec))
            energy_series = [float(np.mean(energy_flux_norm[i:i+step])) for i in range(0, len(energy_flux_norm), step)]

            return {
                "tempo": float(tempo),
                "beatPositions": beat_positions,
                "rhythmicComplexity": rhythmic_complexity,
                "energyFlux": energy_series[:60],
                "energySeries": energy_series
            }

        def compute_timbral():
            """MFCCs, Spectral Centroid, Rolloff, ZCR, Spectral Bandwidth, Flatness"""
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = [float(x) for x in np.mean(mfccs, axis=1)]

            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
            bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

            return {
                "timbralDNA": mfcc_mean,
                "vocalBrightness": spectral_centroid,
                "tonalWeight": rolloff,
                "audioGrain": zcr,
                "spectralFlatness": flatness,
                "spectralBandwidth": bandwidth
            }

        def compute_harmonic():
            """Key, Contrast, Percussion Ratio, Harmonic Movement, Valence"""
            # Key estimation
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            chroma_avg = np.mean(chroma, axis=1)
            dominant_key_idx = int(np.argmax(chroma_avg))

            # Major/Minor detection via Krumhansl-Schmuckler profiles
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            major_corr = np.corrcoef(np.roll(chroma_avg, -dominant_key_idx), major_profile)[0,1]
            minor_corr = np.corrcoef(np.roll(chroma_avg, -dominant_key_idx), minor_profile)[0,1]
            
            is_minor = minor_corr > major_corr
            mode_str = "Minor" if is_minor else "Major"
            harmonic_key = f"{key_names[dominant_key_idx]} {mode_str}"

            # Harmonic Movement: how much the chroma changes frame-to-frame
            chroma_diff = np.diff(chroma, axis=1)
            harmonic_movement = float(np.mean(np.abs(chroma_diff)))

            # Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            avg_contrast = float(np.mean(contrast))

            # Percussion Ratio (from pre-computed HPSS)
            perc_rms = float(np.mean(librosa.feature.rms(y=y_percussive)))
            harm_rms = float(np.mean(librosa.feature.rms(y=y_harmonic)))
            percussion_ratio = perc_rms / (harm_rms + 1e-6)

            # Emotional Valence (heuristic: major=positive, minor=negative, scaled by energy)
            energy_mean = float(np.mean(librosa.feature.rms(y=y)))
            valence = (1.0 if not is_minor else -1.0) * min(1.0, energy_mean * 20)

            return {
                "harmonicKey": harmonic_key,
                "harmonicMovement": harmonic_movement,
                "spectralContrast": avg_contrast,
                "percussionRatio": percussion_ratio,
                "emotionalValence": float(valence)
            }

        def compute_stereo():
            """Stereo Width analysis"""
            if has_stereo and y_stereo is not None:
                mid = (y_stereo[0] + y_stereo[1]) / 2
                side = (y_stereo[0] - y_stereo[1]) / 2
                mid_energy = float(np.mean(mid**2))
                side_energy = float(np.mean(side**2))
                stereo_width = side_energy / (mid_energy + 1e-6)
            else:
                stereo_width = 0.0
            return {"stereoWidth": float(min(stereo_width, 2.0))}

        def compute_deep_texture():
            """Frequency Band Energy, Peak Moments, Onset Density, Spectral Entropy, Energy Contour, Tempo Stability"""
            # Frequency Band Energy Distribution (5-band EQ profile)
            S_power = np.abs(librosa.stft(y))**2
            freqs = librosa.fft_frequencies(sr=sr)
            bands = {
                "sub_bass": (20, 60),
                "bass": (60, 250),
                "mid": (250, 2000),
                "high_mid": (2000, 6000),
                "high": (6000, sr//2)
            }
            band_energy = {}
            total_energy = float(np.sum(S_power)) + 1e-10
            for name, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs < hi)
                band_energy[name] = float(np.sum(S_power[mask, :]) / total_energy)

            # Peak / Climax Detection (timestamps of top 5 energy moments)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            if len(onset_env) > 10:
                peak_indices = np.argsort(onset_env)[-5:][::-1]
                peak_times = librosa.frames_to_time(peak_indices, sr=sr)
                peak_moments = sorted([float(t) for t in peak_times])
            else:
                peak_moments = []

            # Onset Density (events per second)
            onsets = librosa.onset.onset_detect(y=y, sr=sr)
            audio_duration = len(y) / sr
            onset_density = float(len(onsets) / max(0.1, audio_duration))

            # Spectral Entropy (unpredictability of the spectrum)
            S_norm = S_power / (np.sum(S_power, axis=0, keepdims=True) + 1e-10)
            spectral_entropy = float(np.mean(-np.sum(S_norm * np.log2(S_norm + 1e-10), axis=0)))

            # Energy Contour Classification
            rms_full = librosa.feature.rms(y=y)[0]
            if len(rms_full) > 4:
                quarters = np.array_split(rms_full, 4)
                q_means = [float(np.mean(q)) for q in quarters]
                if q_means[-1] > q_means[0] * 1.3:
                    energy_contour = "rising"
                elif q_means[0] > q_means[-1] * 1.3:
                    energy_contour = "falling"
                elif max(q_means[1], q_means[2]) > max(q_means[0], q_means[-1]) * 1.2:
                    energy_contour = "arch"
                elif np.std(q_means) < np.mean(q_means) * 0.15:
                    energy_contour = "steady"
                else:
                    energy_contour = "dynamic"
            else:
                energy_contour = "steady"

            # Tempo Stability (how consistent the tempo is)
            tempo_frames = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
            if len(tempo_frames) > 1:
                tempo_stability = 1.0 - float(np.std(tempo_frames) / (np.mean(tempo_frames) + 1e-6))
                tempo_stability = max(0.0, min(1.0, tempo_stability))
            else:
                tempo_stability = 1.0

            return {
                "frequencyBands": band_energy,
                "peakMoments": peak_moments,
                "onsetDensity": onset_density,
                "spectralEntropy": spectral_entropy,
                "energyContour": energy_contour,
                "tempoStability": tempo_stability
            }

        def compute_mood(tempo_val, energy_mean, is_minor, valence):
            """Advanced mood classification"""
            if valence > 0.5 and tempo_val > 120:
                return "euphoric"
            elif valence > 0.3 and tempo_val > 100:
                return "energetic"
            elif valence > 0 and tempo_val > 90:
                return "vibrant"
            elif is_minor and tempo_val < 90:
                return "melancholic"
            elif is_minor and energy_mean < 0.03:
                return "haunting"
            elif tempo_val < 80:
                return "ethereal"
            elif energy_mean > 0.04:
                return "atmospheric"
            else:
                return "contemplative"

        # --- EXECUTE ALL GROUPS IN PARALLEL ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            f_dynamics = executor.submit(compute_dynamics)
            f_rhythm = executor.submit(compute_rhythm)
            f_timbral = executor.submit(compute_timbral)
            f_harmonic = executor.submit(compute_harmonic)
            f_stereo = executor.submit(compute_stereo)
            f_deep = executor.submit(compute_deep_texture)

        # Collect
        dynamics = f_dynamics.result()
        rhythm = f_rhythm.result()
        timbral = f_timbral.result()
        harmonic = f_harmonic.result()
        stereo = f_stereo.result()
        deep = f_deep.result()

        # Derive mood from combined signals
        energy_mean = float(np.mean(librosa.feature.rms(y=y)))
        is_minor = "Minor" in harmonic["harmonicKey"]
        mood = compute_mood(rhythm["tempo"], energy_mean, is_minor, harmonic["emotionalValence"])

        elapsed = _time.monotonic() - t0
        print(f"[PROFILER] Hypercharged analysis complete in {elapsed:.2f}s ({len(y)/sr:.1f}s audio)")

        # Merge all metric groups
        result = {}
        result.update(dynamics)
        result.update(rhythm)
        result.update(timbral)
        result.update(harmonic)
        result.update(stereo)
        result.update(deep)
        result["mood"] = mood

        return result

    except Exception as e:
        print(f"Advanced profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "snr": 15.0, "reverbLevel": 0.1, "tempo": 120.0,
            "mood": "vibrant", "harmonicKey": "C Major",
            "energyFlux": [0.5]*60, "vocalBrightness": 2000.0,
            "energySeries": [0.5]*60, "spectralContrast": 20.0,
            "percussionRatio": 0.3, "spectralFlatness": 0.05,
            "timbralDNA": [0.0]*13, "tonalWeight": 4000.0, "audioGrain": 0.03,
            "dynamicRange": 30.0, "loudnessLUFS": -14.0,
            "beatPositions": [], "rhythmicComplexity": 0.0,
            "harmonicMovement": 0.0, "emotionalValence": 0.0,
            "stereoWidth": 0.0, "spectralBandwidth": 2000.0,
            "frequencyBands": {"sub_bass": 0.1, "bass": 0.2, "mid": 0.4, "high_mid": 0.2, "high": 0.1},
            "peakMoments": [], "onsetDensity": 2.0, "spectralEntropy": 5.0,
            "energyContour": "steady", "tempoStability": 0.9
        }

try:
    from bullmq import Worker
except ImportError:
    Worker = None
    print("BullMQ not available (running on Modal or missing dependency). Local queue disabled.")

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
CALLBACK_URL = os.getenv("CALLBACK_URL")
CALLBACK_TOKEN = os.getenv("CALLBACK_TOKEN")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
# DEVICE and COMPUTE_TYPE are now managed by _ensure_ai_imports or env defaults
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float32")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

headers = {
    "Authorization": f"Bearer {CALLBACK_TOKEN}",
    "Content-Type": "application/json"
}

def safe_callback(job_id: str, data: Dict[str, Any]):
    """
    Unified callback handler with safety checks for cloud environments.
    Prevents "Connection Refused" errors when workers attempt to callback to localhost.
    """
    try:
        if not CALLBACK_URL or "127.0.0.1" in CALLBACK_URL or "localhost" in CALLBACK_URL:
            # Special check: If running in Modal, localhost is guaranteed to fail if the backend is local.
            print(f"[{job_id}] ⚠️ Skipping callback to local/missing URL: {CALLBACK_URL}. (Define PUBLIC_CALLBACK_URL in your cloud provider settings)")
            return None

        resp = requests.post(CALLBACK_URL, headers=headers, json=data)
        if resp.status_code != 200:
            print(f"[{job_id}] Callback failed with {resp.status_code}: {resp.text}")
        return resp
    except Exception as e:
        print(f"[{job_id}] Callback skipping (Connectivity Error): {e}")
        return None

def report_stage(job_id: str, status: str, progress: int, user_id: str = None, metrics: Dict[str, Any] = None):
    data = {
        "jobId": job_id,
        "userId": user_id,
        "event": "stage",
        "status": status,
        "progress": progress
    }
    if metrics:
        data["metrics"] = metrics
    return safe_callback(job_id, data)

def compute_beat_sync_cuts(audio_profile: Dict[str, Any], total_duration: float, max_scenes_limit: int = 8, job_id: str = "") -> List[float]:
    """
    Intelligent Narrative Engine: Decides scene count and durations based on musical peaks.
    """
    beat_positions = audio_profile.get("beatPositions", [])
    peak_moments = audio_profile.get("peakMoments", [])
    
    # 1. Primary Candidates: Musical climaxes (peaks)
    # We want at least 4s between cuts to give the AI/WAN enough time to establish presence
    MIN_SCENE_DUR = 5.0 
    MAX_SCENE_DUR = 15.0
    
    # Filter/Select cut points based on peaks
    potential_cuts = sorted([p for p in peak_moments if 1.0 < p < (total_duration - 1.0)])
    
    intelligent_cuts = []
    last_cut = 0.0
    
    for p in potential_cuts:
        if (p - last_cut) >= MIN_SCENE_DUR:
            # Snap peak to nearest beat for rhythmic perfection
            nearest_beat = min(beat_positions, key=lambda b: abs(b - p)) if beat_positions else p
            snap_cut = nearest_beat if abs(nearest_beat - p) < 0.5 else p
            
            if (snap_cut - last_cut) >= MIN_SCENE_DUR:
                intelligent_cuts.append(snap_cut)
                last_cut = snap_cut
    
    # 2. Fill gaps: If a scene is too long (>MAX), find midpoints on beats
    final_cuts = []
    last_v = 0.0
    for c in (intelligent_cuts + [total_duration]):
        gap = c - last_v
        if gap > MAX_SCENE_DUR:
            # Split the gap
            num_splits = math.ceil(gap / MAX_SCENE_DUR)
            for s in range(1, num_splits):
                mid = last_v + (s * (gap / num_splits))
                # Try to find a beat near mid
                nearest_b = min(beat_positions, key=lambda b: abs(b - mid)) if beat_positions else mid
                final_cuts.append(nearest_b)
        if c < total_duration:
            final_cuts.append(c)
        last_v = c

    final_cuts = sorted(list(set([round(x, 2) for x in final_cuts])))
    
    # 3. Final Limits: Cap total scenes to prevent compute runaway
    if len(final_cuts) >= max_scenes_limit:
        final_cuts = final_cuts[:max_scenes_limit-1]

    # Convert to scene durations
    all_boundaries = [0.0] + final_cuts + [total_duration]
    scene_durations = [round(all_boundaries[i+1] - all_boundaries[i], 2) for i in range(len(all_boundaries)-1)]
    
    # Safety: ensure no scene is TOO small (FFmpeg/WAN crash risk)
    scene_durations = [max(1.5, d) for d in scene_durations]
    
    print(f"[{job_id}] Intelligent Segmentation: Created {len(scene_durations)} scenes based on musical narrative.")
    return scene_durations

def compute_segment_moods(audio_path: str, scene_durations: List[float], job_id: str = "") -> List[Dict[str, Any]]:
    """
    Per-Segment Mood Engine: Analyzes each beat-synced section independently.
    
    For each segment, computes:
    - Local tempo, energy, spectral centroid
    - Harmonic key (major/minor via Krumhansl-Schmuckler)
    - Emotional valence
    - Mood classification
    
    Returns: List of dicts, one per scene, with keys: mood, energy, tempo, key, valence, intensity
    """
    import librosa
    
    SR = 22050
    y_full, sr = librosa.load(audio_path, sr=SR, mono=True)
    total_duration = len(y_full) / sr
    
    # Krumhansl-Schmuckler key profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    segment_moods = []
    cursor = 0.0
    
    for i, dur in enumerate(scene_durations):
        seg_start = cursor
        seg_end = min(cursor + dur, total_duration)
        cursor = seg_end
        
        # Extract audio slice
        start_sample = int(seg_start * sr)
        end_sample = int(seg_end * sr)
        y_seg = y_full[start_sample:end_sample]
        
        if len(y_seg) < sr * 0.3:  # Less than 0.3s, too short for analysis
            segment_moods.append({
                "mood": "transitional", "energy": 0.5, "tempo": 120.0,
                "key": "C Major", "valence": 0.0, "intensity": "medium"
            })
            continue
        
        # Per-segment energy
        rms = librosa.feature.rms(y=y_seg)[0]
        energy_mean = float(np.mean(rms))
        energy_peak = float(np.max(rms))
        
        # Per-segment tempo
        try:
            seg_tempo, _ = librosa.beat.beat_track(y=y_seg, sr=sr)
            seg_tempo = float(np.atleast_1d(seg_tempo)[0])
        except Exception:
            seg_tempo = 120.0
        
        # Per-segment key detection (Krumhansl-Schmuckler)
        try:
            chroma = librosa.feature.chroma_cqt(y=y_seg, sr=sr)
            chroma_avg = np.mean(chroma, axis=1)
            
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            best_corr = -999
            best_key = "C Major"
            is_minor = False
            
            for shift in range(12):
                shifted = np.roll(chroma_avg, -shift)
                maj_corr = float(np.corrcoef(shifted, major_profile)[0, 1])
                min_corr = float(np.corrcoef(shifted, minor_profile)[0, 1])
                
                if maj_corr > best_corr:
                    best_corr = maj_corr
                    best_key = f"{key_names[shift]} Major"
                    is_minor = False
                if min_corr > best_corr:
                    best_corr = min_corr
                    best_key = f"{key_names[shift]} Minor"
                    is_minor = True
        except Exception:
            best_key = "C Major"
            is_minor = False
        
        # Per-segment spectral centroid (brightness proxy)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_seg, sr=sr)))
        
        # Emotional valence per segment
        valence = 0.0
        if not is_minor:
            valence += 0.3
        else:
            valence -= 0.3
        if seg_tempo > 120:
            valence += 0.2
        elif seg_tempo < 80:
            valence -= 0.2
        if centroid > 3000:
            valence += 0.1
        elif centroid < 1500:
            valence -= 0.1
        valence = max(-1.0, min(1.0, valence))
        
        # Intensity classification
        if energy_peak > 0.08:
            intensity = "explosive"
        elif energy_mean > 0.05:
            intensity = "high"
        elif energy_mean > 0.025:
            intensity = "medium"
        else:
            intensity = "low"
        
        # Mood classification per segment
        if valence > 0.4 and seg_tempo > 120:
            mood = "euphoric"
        elif valence > 0.2 and seg_tempo > 100:
            mood = "energetic"
        elif valence > 0 and energy_mean > 0.04:
            mood = "vibrant"
        elif is_minor and seg_tempo < 90:
            mood = "melancholic"
        elif is_minor and energy_mean < 0.025:
            mood = "haunting"
        elif seg_tempo < 80 and energy_mean < 0.03:
            mood = "ethereal"
        elif energy_mean > 0.04:
            mood = "atmospheric"
        else:
            mood = "contemplative"
        
        seg_info = {
            "mood": mood,
            "energy": round(energy_mean, 4),
            "tempo": round(seg_tempo, 1),
            "key": best_key,
            "valence": round(valence, 2),
            "intensity": intensity
        }
        segment_moods.append(seg_info)
        print(f"[{job_id}] Segment {i+1} ({dur:.1f}s): {mood} | {intensity} | {best_key} | {seg_tempo:.0f}BPM | E={energy_mean:.3f}")
    
    return segment_moods


def synthesize_audio_narrative(profile: Dict[str, Any]) -> str:
    """
    Converts raw audio metrics into a cinematic natural language summary
    that Gemini can interpret far better than raw JSON numbers.
    """
    # Core identity
    key = profile.get("harmonicKey", "C Major")
    tempo = profile.get("tempo", 120)
    mood = profile.get("mood", "vibrant")
    
    # Energy shape
    contour = profile.get("energyContour", "dynamic")
    contour_desc = {
        "rising": "builds from quiet to powerful",
        "falling": "starts strong and fades into silence",
        "arch": "rises to a peak in the middle then descends",
        "steady": "maintains consistent energy throughout",
        "dynamic": "shifts unpredictably between intensity levels"
    }.get(contour, "has a dynamic energy flow")
    
    # Frequency personality
    bands = profile.get("frequencyBands", {})
    dominant_band = max(bands, key=bands.get) if bands else "mid"
    band_desc = {
        "sub_bass": "extremely bass-heavy with deep, rumbling low-end that vibrates the body",
        "bass": "bass-dominant with warm, heavy low frequencies",
        "mid": "mid-range focused with strong melodic presence",
        "high_mid": "bright and cutting with prominent upper harmonics",
        "high": "airy and crystalline with dominant high-frequency shimmer"
    }.get(dominant_band, "balanced across the frequency spectrum")
    
    # Tempo feel
    stability = profile.get("tempoStability", 0.5)
    if stability > 0.85:
        tempo_feel = "metronomically steady"
    elif stability > 0.6:
        tempo_feel = "mostly consistent with subtle fluctuations"
    else:
        tempo_feel = "loose and freeform, with significant tempo variation"
    
    # Rhythmic character
    onset_density = profile.get("onsetDensity", 2.0)
    perc_ratio = profile.get("percussionRatio", 0.3)
    if onset_density > 6:
        rhythm_desc = "extremely dense and percussive, like a machine-gun of rhythmic events"
    elif onset_density > 3:
        rhythm_desc = "rhythmically active with frequent percussive hits"
    elif onset_density > 1.5:
        rhythm_desc = "moderately rhythmic with balanced percussion"
    else:
        rhythm_desc = "sparse and minimal, with few rhythmic events"
    
    # Spectral character
    entropy = profile.get("spectralEntropy", 5.0)
    flatness = profile.get("spectralFlatness", 0.05)
    if entropy > 7:
        spectral_desc = "chaotic and noisy, with unpredictable spectral content"
    elif entropy > 5:
        spectral_desc = "rich and complex, with varied spectral textures"
    elif entropy > 3:
        spectral_desc = "moderately complex, with clear tonal structure"
    else:
        spectral_desc = "pure and tonal, with clean, focused frequencies"
    
    # Spatial
    stereo = profile.get("stereoWidth", 0.0)
    if stereo > 1.0:
        spatial_desc = "extremely wide stereo image, immersive and enveloping"
    elif stereo > 0.5:
        spatial_desc = "wide stereo field with clear spatial separation"
    elif stereo > 0.2:
        spatial_desc = "moderate stereo width"
    else:
        spatial_desc = "narrow, mono-like presentation"
    
    # Peak moments
    peaks = profile.get("peakMoments", [])
    peaks_desc = f"Energy climaxes occur at {', '.join([f'{p:.1f}s' for p in peaks])}" if peaks else "No distinct energy peaks detected"
    
    # Dynamic range
    dr = profile.get("dynamicRange", 30)
    lufs = profile.get("loudnessLUFS", -14)
    if dr > 40:
        dynamics_desc = "very wide dynamic range (cinematic, orchestral feel)"
    elif dr > 25:
        dynamics_desc = "healthy dynamic range with contrast between loud and soft"
    elif dr > 15:
        dynamics_desc = "moderately compressed dynamics"
    else:
        dynamics_desc = "heavily compressed, wall-of-sound character"
    
    # Valence
    valence = profile.get("emotionalValence", 0.0)
    if valence > 0.3:
        valence_desc = "bright and uplifting emotional character"
    elif valence > 0:
        valence_desc = "slightly positive, neutral-warm emotional tone"
    elif valence > -0.3:
        valence_desc = "slightly dark, introspective emotional tone"
    else:
        valence_desc = "dark, melancholic emotional character"
    
    # Compose the narrative
    narrative = (
        f"AUDIO SOUL: This is a {mood} track in {key} at {tempo:.0f} BPM. "
        f"The energy {contour_desc}. "
        f"The sound is {band_desc}. "
        f"The rhythm is {rhythm_desc} (percussion ratio: {perc_ratio:.2f}). "
        f"Tempo is {tempo_feel}. "
        f"Spectrally, the track is {spectral_desc}. "
        f"Spatial character: {spatial_desc}. "
        f"{peaks_desc}. "
        f"Dynamics: {dynamics_desc} (loudness: {lufs:.1f} LUFS). "
        f"Emotional DNA: {valence_desc} (valence: {valence:+.2f})."
    )
    
    return narrative



def generate_single_scene(job_id, i, num_clips, scenes, scene_durations, tmp_dir):
    _ensure_ai_imports()
    wan_pipe = WanModelManager.get_pipe()
    
    scene_dur = scene_durations[i]
    current_prompt = scenes[i] if i < len(scenes) else scenes[-1]
    
    print(f"[{job_id}] Scene {i+1}/{num_clips} [Stage 1/2: WAN]")
    
    # Stage 1: Wan 2.1 Motion
    # Masterpiece Polish: 30 steps for high fidelity
    output = wan_pipe(
        prompt=current_prompt + ", realistic motion, cinematic",
        negative_prompt="cartoon, low quality, blurry, text, watermark",
        width=720,
        height=1280,
        num_frames=17,
        num_inference_steps=15, # Optimized: 15 steps is the "sweet spot" for L4 GPUs
        guidance_scale=5.0
    )
    frames = output.frames[0]
    
    raw_path = os.path.join(tmp_dir, f"raw_scene_{i}.mp4")
    export_to_video(frames, raw_path, fps=16)
    
    print(f"[{job_id}] Scene {i+1}/{num_clips} [Stage 3/3: MCI]")
    
    # Stage 3: Light-Speed FFmpeg MCI
    interp_path = os.path.join(tmp_dir, f"scene_{i}.mp4")
    pts_factor = scene_dur / (17/16)
    
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path,
            "-vf", (
                f"setpts={pts_factor}*PTS,"
                f"minterpolate=fps=24:mi_mode=mci:mc_mode=obmc,"
                f"scale=720:1280:force_original_aspect_ratio=decrease,"
                f"pad=720:1280:(ow-iw)/2:(oh-ih)/2,"
                f"noise=alls=5:allf=t+u," # Subtle Film Grain
                f"eq=contrast=1.05:saturation=1.1" # Gentle Cinematic Grade
            ),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", interp_path
        ], check=True)
    except Exception as e:
        print(f"[{job_id}] FFmpeg MCI failed: {e}. Falling back to raw export.")
        import shutil
        shutil.copy(raw_path, interp_path)
    
    return interp_path

    
    return interp_path

async def generate_video_background(
    job_id: str, user_id: str, audio_path: str, bg_prompt: str, 
    start_time: float, end_time: float, transcription_words: List[Dict], 
    style_id: Optional[str] = None, font_family: Optional[str] = None, animation_effect: Optional[str] = None, 
    lyric_opacity: Optional[float] = None, position: Optional[str] = None, font_size: Optional[int] = None,
    video_title: Optional[str] = None, title_font_family: Optional[str] = None,
    parallel_generator=None, audio_profile: Dict[str, Any] = None
) -> str:
    """
    Beat-Sync Narrative Engine (Plan Light-Speed).
    """
    try:
        director_client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Derive duration from start_time and end_time
        duration = end_time - start_time

        # 1. BEAT-SYNC ENGINE
        # Use the provided audio_profile, or compute if not provided (though it should be)
        if not audio_profile:
            # This case should ideally not happen if called from process_job
            print(f"[{job_id}] Warning: audio_profile not provided to generate_video_background. Computing now.")
            audio_profile = profile_audio(audio_path)

        target_scenes = max(4, math.ceil(duration / 15.0))
        print(f"[{job_id}] Dynamic Scene Target: {target_scenes} scenes for {duration:.1f}s")
        
        scene_durations = compute_beat_sync_cuts(audio_profile, duration, max_scenes_limit=target_scenes, job_id=job_id)
        num_clips = len(scene_durations)
        
        # 2. PER-SEGMENT MOOD ANALYSIS
        segment_moods = compute_segment_moods(audio_path, scene_durations, job_id=job_id)
        audio_narrative = synthesize_audio_narrative(audio_profile)

        # 3. Dynamic Creative Director (Blueprint)
        audio_mood = audio_profile.get("mood", "cinematic")
        harmonic_key = audio_profile.get("harmonicKey", "Unknown")
        tempo = audio_profile.get("tempo", "Unknown")
        
        # Derive full lyrics for context
        full_lyrics = " ".join([w["w"] for w in transcription_words]) if transcription_words else "Instrumental"
        
        vision_prompt = (
            f"You are a VISIONARY FILM AUTEUR (think Tarkovsky, Kubrick, Jodorowsky) creating a high-concept music video.\n"
            f"SONG DNA:\n"
            f"- MOOD: {audio_mood.upper()}\n"
            f"- NARRATIVE SUMMARY: {audio_narrative}\n"
            f"- USER VISUAL THEME: '{bg_prompt}'\n"
            f"- USER ARTISTIC STYLE: '{style_id or 'Director Choice'}'\n"
            f"- TOTAL SCENES: {num_clips} (Precisely aligned to musical transitions)\n"
            f"- LYRICS: {full_lyrics[:2000]}\n\n"
            f"TASK: Create a 'Cinematic Universe Blueprint' for a {num_clips}-scene visual masterpiece.\n"
            f"CRITICAL: Do NOT be literal. The scene transitions were calculated to hit MUSICAL PEAKS. "
            f"Your narrative must build tension and release in sync with the song's energy.\n"
            f"The video must be a cohesive, dream-like visual journey, not a series of stock clips.\n"
            f"Include:\n"
            f"1. 'protagonist': A SURREAL character consistent with the USER STYLE.\n"
            f"2. 'artistic_movement': Use '{style_id}' if provided, otherwise invent a bold style.\n"
            f"3. 'color_palette': 3 specific hex codes or complex color names.\n"
            f"4. 'lighting_mood': Atmospheric lighting.\n"
            f"5. 'narrative_arc': An abstract, metaphorical story summary that merges the USER THEME with the LYRICS.\n"
            f"Output ONLY the JSON."
        )
        
        try:
            vision_response = director_client.models.generate_content(model="gemini-2.0-flash", contents=vision_prompt)
            v_text = vision_response.text.strip()
            if "```json" in v_text: v_text = v_text.split("```json")[-1].split("```")[0].strip()
            elif "```" in v_text: v_text = v_text.split("```")[-1].split("```")[0].strip() # Handle unlabelled blocks
            blueprint = json.loads(v_text)
            
            # HARD OVERRIDE: If user provided a style_id (preset or custom), force it.
            if style_id:
                print(f"[{job_id}] User Style Override detected: {style_id}")
                blueprint["artistic_movement"] = style_id
                
            print(f"[{job_id}] 🎨 Blueprint Generated: {blueprint}")
        except Exception as blueprint_err:
            print(f"[{job_id}] Blueprint failed: {blueprint_err}. Raw: {v_text if 'v_text' in locals() else 'None'}")
            blueprint = {"protagonist": f"a heroic figure representing {bg_prompt}", "artistic_movement": "Cinematic Realism", "color_palette": "Vibrant"}

        # 4. Scene Synthesis
        # Map lyrics to scenes based on timestamps
        scene_lyrics_context = []
        current_time = 0.0
        
        for i, dur in enumerate(scene_durations):
            end_time_seg = current_time + dur
            
            # Find words in this time window
            scene_words = []
            if transcription_words:
                scene_words = [w["w"] for w in transcription_words if w["t0"] >= current_time and w["t0"] < end_time_seg]
            
            scene_lyric_text = " ".join(scene_words) if scene_words else "[Instrumental/Visual Build-Up]"
            scene_mood_val = segment_moods[i]['mood'].upper()
            
            context_str = (
                f"SCENE {i+1} ({dur:.1f}s) | Mood: {scene_mood_val}\n"
                f"   LYRICS SUNG: \"{scene_lyric_text}\"\n"
            )
            scene_lyrics_context.append(context_str)
            current_time = end_time_seg

        scene_context_block = "\n".join(scene_lyrics_context)
        
        scene_generation_prompt = (
            f"Using the Blueprint below, write exactly {num_clips} AVANT-GARDE scene descriptions.\n"
            f"BLUEPRINT: {json.dumps(blueprint)}\n\n"
            f"--- SCENE CONTEXT ---\n"
            f"{scene_context_block}\n"
            f"---------------------\n\n"
            f"INSTRUCTIONS:\n"
            f"1. NO LITERALISM: Interpret the lyrics METAPHORICALLY. If the lyric is about 'love', show a burning rose, not a couple hugging.\n"
            f"2. VISUAL CONTINUITY: Every scene MUST feature the '{blueprint.get('protagonist')}' in the style of '{blueprint.get('artistic_movement')}'.\n"
            f"3. DREAM LOGIC: Scenes should feel like a fever dream or a high-end art film. Unexpected angles, strange physics, surreal lighting.\n"
            f"4. LIGHTING/COLOR: Use the '{blueprint.get('lighting_mood')}' and palette ({blueprint.get('color_palette')}) to create ATMOSPHERE.\n"
            f"5. PROMPT FORMAT: 'Cinematic shot of [PROTAGONIST] [doing surreal action], [surreal environment], [lighting], [style], 8k, masterpiece'.\n"
            f"6. LIMITS: Keep each prompt under 50 words. Focus on VISUAL IMPACT.\n"
            f"Output JSON array of strings ONLY."
        )

        try:
            scene_response = director_client.models.generate_content(model="gemini-2.0-flash", contents=scene_generation_prompt)
            text = scene_response.text.strip()
            if "```json" in text: text = text.split("```json")[-1].split("```")[0].strip()
            elif "```" in text: text = text.split("```")[-1].split("```")[0].strip()
            scenes = json.loads(text)
            print(f"[{job_id}] 🎬 Scenes Generated: {scenes}")
        except Exception as e:
            print(f"[{job_id}] Scene generation failed: {e}. Raw: {text if 'text' in locals() else 'None'}")
            scenes = [f"cinematic {blueprint['artistic_movement']} shot of {blueprint['protagonist']}, highly detailed, 8k"] * num_clips

        # 5. EXECUTION: Parallel or Sequential
        tmp_dir = tempfile.mkdtemp()
        clip_paths = []

        if parallel_generator:
            print(f"[{job_id}] ⚡ Triggering Parallel Light-Speed Generation...")
            clip_paths = await parallel_generator(scenes, scene_durations, tmp_dir)
        else:
            print(f"[{job_id}] Sequential Generation (Light-Speed Optimizations active)...")
            for i in range(num_clips):
                try:
                    p = generate_single_scene(job_id, i, num_clips, scenes, scene_durations, tmp_dir)
                    clip_paths.append(p)
                except Exception as e:
                    print(f"[{job_id}] Scene {i} failed: {e}")

        if not clip_paths:
            raise Exception("No video clips were generated.")

        # 6. Final Merging
        master_output = os.path.join(tmp_dir, "master_bg.mp4")
        
        print(f"[{job_id}] Director's Cut: Assembly & Upscaling to 1080p...")
            
        # COMPLEX FFMPEG FILTER GRAPH FOR XFADE + UPSCALING
        # 1. Inputs
        input_args = []
        for cp in clip_paths:
            input_args.extend(["-i", cp])
        
        # 2. Filter Complex
        filter_complex = ""
        
        # A) Upscale & Pad each input to 1080x1920 (Base Resolution for High Quality)
        # We treat the 720x1280 source as the base, then upscale everything at the end.
        # Actually, proper xfade requires identical resolution inputs. 
        # The clips are already 720x1280 from Wan.
        
        # B) XFADE Chain
        # Stream 0 is [0:v]. 
        # We fade [0:v] and [1:v] -> v1
        # v1 and [2:v] -> v2 ...
        
        # Narrative transition duration (semi-dynamic based on tempo)
        tempo = audio_profile.get("tempo", 120)
        TR_DUR = 1.0 if tempo < 100 else 0.75
        
        if len(clip_paths) > 1:
            # Prepare inputs (just labels)
            filter_complex += f"[0:v]setsar=1[v0];"
            for i in range(1, len(clip_paths)):
                filter_complex += f"[{i}:v]setsar=1[v{i}];"
                
            # Chain
            curr_offset = 0
            for i in range(len(clip_paths) - 1):
                seg_dur = scene_durations[i]
                curr_offset += seg_dur - TR_DUR
                
                input_a = "v0" if i == 0 else f"vm{i}"
                input_b = f"v{i+1}"
                output_v = f"vm{i+1}"
                
                filter_complex += f"[{input_a}][{input_b}]xfade=transition=fade:duration={TR_DUR}:offset={curr_offset:.2f}[{output_v}];"
            
            last_label = f"vm{len(clip_paths)-1}"
        else:
            filter_complex += "[0:v]null[vm0];"
            last_label = "vm0"

        # C) Masterpiece Polish: Upscale + Sharpen
        filter_complex += (
            f"[{last_label}]"
            f"scale=1080:1920:flags=lanczos," # Smart Upscale
            f"unsharp=5:5:1.0:5:5:0.0" # Unsharp Mask against upscaling blur
            f"[vfinal]"
        )
        
        ffmpeg_cmd = ["ffmpeg", "-y"] + input_args + [
            "-filter_complex", filter_complex,
            "-map", "[vfinal]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "24", # Balance size (<50MB) and quality
            "-pix_fmt", "yuv420p",
            master_output
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        # Now merge with audio
        final_video_with_audio_path = os.path.join(tmp_dir, "master_bg_final.mp4")
        subprocess.run(["ffmpeg", "-y", "-i", master_output, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", final_video_with_audio_path], check=True)

        storage_path = f"{job_id}/cinematic_bg.mp4"
        with open(final_video_with_audio_path, "rb") as f:
            supabase.storage.from_("assets").upload(path=storage_path, file=f, file_options={"content-type": "video/mp4", "x-upsert": "true"})
        
        file_size = os.path.getsize(final_video_with_audio_path)
        print(f"[{job_id}] Final Video Size: {file_size / (1024*1024):.2f} MB")
        
        return storage_path, file_size

    except Exception as e:
        print(f"[{job_id}] Narrative Engine failed: {e}")
        raise e


async def process_job(job, job_token, parallel_generator=None):
    global CALLBACK_URL, headers
    
    # 0. Global Setup...
    job_id = job.data.get('jobId')
    
    # Allow per-job callback override (essential for remote workers calling back to local)
    dynamic_callback = job.data.get('callback_url')
    if dynamic_callback:
        print(f"[{job_id}] 🌐 Dynamic Callback Override: {dynamic_callback}")
        CALLBACK_URL = dynamic_callback
        headers["Authorization"] = f"Bearer {CALLBACK_TOKEN}" # Ensure token is synced
    asset_id = job.data.get('inputAudioAssetId')
    user_id = job.data.get('userId')
    mood = job.data.get('mood', 'default')
    position = job.data.get('position', 'center')
    font_size = job.data.get('fontSize', 6)
    bg_prompt = job.data.get('bgPrompt')
    start_time = job.data.get('startTime', 0)
    end_time = job.data.get('endTime', 60)
    # style_id removed as requested
    target_language = job.data.get('targetLanguage') # Extract targetLanguage
    font_family = job.data.get('fontFamily')
    animation_effect = job.data.get('animationEffect')
    lyric_opacity = job.data.get('lyricOpacity')
    video_title = job.data.get('video_title')
    title_font_family = job.data.get('title_font_family')
    lyric_color = job.data.get('lyricColor', '#ffffff')
    style_id = job.data.get('style') # Extracted from 'style' as per route.ts
    
    # Cap duration to 60s as per requirements
    if end_time - start_time > 60:
        end_time = start_time + 60

    print(f"[{job_id}] Received job for user {user_id} with asset: {asset_id}")
    print(f"[{job_id}] Range: {start_time}s - {end_time}s (Prompt: {bg_prompt})")

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
                res_d = subprocess.run(demucs_cmd, capture_output=True, text=True, check=False, timeout=300)
                
                if res_d.returncode == 0:
                    print(f"[{job_id}] Demucs finished successfully.")
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()
                    
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

            # --- NEW: Universal Motion Manifest Generation (Phase 46) ---
            print(f"[{job_id}] Synthesizing Universal Motion Manifest...")
            # We use the raw audio profile to drive the visual physics of the overlay
            # BUT we strictly respect User Constraints where provided.
            
            user_constraints = {
                "fontFamily": font_family,
                "animationEffect": animation_effect,
                "lyricOpacity": lyric_opacity,
                "position": position,
                "fontSize": font_size
            }
            
            motion_prompt = (
                f"You are a World-Class Motion Designer. Create a 'Universal Motion Manifest' (JSON) for a lyric video overlay.\n"
                f"AUDIO PROFILE:\n"
                f"- Tempo: {profile['tempo']} BPM\n"
                f"- Key: {profile['harmonicKey']}\n"
                f"- Energy Flux: {np.mean(profile['energyFlux']):.2f} (0-1)\n"
                f"- Grain (ZCR): {profile['audioGrain']:.3f}\n\n"
                f"USER CONSTRAINTS (MUST RESPECT):\n"
                f"- Font: {font_family if font_family else 'AI Decision (Choose a Google Font that perfectly matches the vibe)'}\n"
                f"- Effect: {animation_effect if animation_effect else 'AI Decision (Choose a Kinetic Effect: fade, pop, slide, typewriter, glitch, neon)'}\n"
                f"- Opacity: {lyric_opacity}\n"
                f"- Position: {position}\n\n"
                "TASK: Return a Valid JSON object. If User Constraints are present, use them. For 'AI Decision' fields, invent the perfect match for the audio.\n"
                "DESIGN GOAL: Create a 'Living Overlay'. The text should feel like it exists in the same physical space as the music. Use Physics-based animation parameters.\n"
                "JSON SCHEMA:\n"
                "{\n"
                '  "typography": { "fontFamily": "string", "fontWeight": 400-900, "tracking": -0.05 to 0.5, "fontSize": number },\n'
                '  "palette": { "primary": "#hex", "secondary": "#hex", "shadow": "#hex", "glow": "#hex" },\n'
                '  "layout": { "mode": "center/top/bottom", "offsetY": number },\n'
                '  "kinetics": { "effect": "string", "reactivity": 0.0-2.0, "jitter": 0.0-1.0, "physics": "spring/linear" }\n'
                "}\n"
                "Respond ONLY with the JSON."
            )
            
            motion_manifest = {}
            try:
                director_client = genai.Client(api_key=GOOGLE_API_KEY)
                m_resp = director_client.models.generate_content(model='gemini-2.0-flash', contents=motion_prompt)
                clean_json = m_resp.text.replace("```json", "").replace("```", "").strip()
                motion_manifest = json.loads(clean_json)
                
                # STRICT ENFORCEMENT: Overwrite AI with User Choices if they exist
                if font_family: motion_manifest.setdefault("typography", {})["fontFamily"] = font_family
                if font_size: motion_manifest.setdefault("typography", {})["fontSize"] = font_size
                if lyric_opacity: motion_manifest["typography"]["opacity"] = lyric_opacity
                if position: motion_manifest.setdefault("layout", {})["mode"] = position
                if animation_effect: motion_manifest.setdefault("kinetics", {})["effect"] = animation_effect
                if lyric_color: motion_manifest.setdefault("palette", {})["primary"] = lyric_color
                
                print(f"[{job_id}] Motion Manifest Synthesized (User-Guided): {motion_manifest.keys()}")
            except Exception as e:
                print(f"[{job_id}] Motion Manifest generation failed: {e}. Using safe defaults.")
                motion_manifest = {
                    "typography": { "fontFamily": font_family or "Inter", "fontWeight": 800, "fontSize": font_size or 6 },
                    "palette": { "primary": lyric_color or "#ffffff", "secondary": "#cccccc" },
                    "layout": { "mode": position or "center" },
                    "kinetics": { "effect": animation_effect or "fade", "reactivity": 1.0, "physics": "spring" }
                }
            


            # PHASE 5: WHISPERX TRANSCRIPTION (Singleton)
            model = WhisperModelManager.get_model()
            if not model:
                print(f"[{job_id}] WhisperX model failed to load. Skipping transcription.")
                return "TRANSCRIPTION_FAILED"
            
            import gc
            audio = whisperx.load_audio(vocal_path)
            result = model.transcribe(audio, batch_size=4) # Lowered from 16 to 4 for VRAM
            
            torch.cuda.empty_cache()
            gc.collect()
            
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
            # Pass local_audio_path (the pre-cropped segment) to preserve audio fidelity
            video_bg_url, video_file_size = await generate_video_background(
                job_id, user_id, local_audio_path, bg_prompt, 
                start_time, end_time, final_words,
                style_id=style_id, font_family=font_family, 
                animation_effect=animation_effect, 
                lyric_opacity=lyric_opacity, 
                position=position, 
                font_size=font_size,
                video_title=video_title,
                title_font_family=title_font_family,
                parallel_generator=parallel_generator,
                audio_profile=profile
            )
            
            # 5. Save and Upload Analysis JSON
            total_h_dur = h_end - h_start + 0.1
            word_dur = sum([w["t1"] - w["t0"] for w in final_words])
            coverage = min(1.0, word_dur / total_h_dur)
            
            avg_conf = sum([w["conf"] for w in final_words]) / len(final_words) if final_words else 0.0

            analysis_path = f"{job_id}/analysis.json"
            analysis_data = {
                "jobId": job_id, "userId": user_id, "hook": hook, "words": final_words,
                "language": language_code, "precision": "studio-high-v1",
                "mood": mood, "position": position, 
                "fontSize": font_size,
                "fontFamily": font_family,
                "styleId": style_id,
                "position": position,
                "videoTitle": video_title,
                "titleFontFamily": title_font_family,
                "animationEffect": animation_effect, "lyricColor": lyric_color,
                "lyricOpacity": lyric_opacity, "videoBgUrl": video_bg_url,
                "profiling": profile,
                "alignment": {
                    "method": "whisperx", "alignScore": round(aligned_ratio, 3),
                    "alignedRatio": round(aligned_ratio, 3), "fallbackUsed": fallback_used,
                    "granularity": granularity
                },
                "motion_manifest": motion_manifest, # NEW: Universal Motion Engine Manifest
                "metrics": {
                    "alignScore": round(aligned_ratio, 3),
                    "coverage": round(coverage, 3),
                    "uncertainRate": round(1.0 - avg_conf, 3),
                    "sourceAgreementRate": round(avg_conf, 3)
                }
            }
            
            local_analysis_file = os.path.join(tmp_dir, "analysis.json")
            # Cinematic BG already uploaded in generate_video_background
            storage_path = video_bg_url
            
            # Report Asset to API
            safe_callback(job_id, {
                "jobId": job_id,
                "userId": user_id,
                "event": "asset",
                "kind": "draft_video", 
                "url": storage_path, 
                "metadata": {"precision": "high-demucs-v3", "size_mb": round(video_file_size / (1024*1024), 2)}
            })

            # 3. Upload vocal stem (if not already done)
            vocal_path_storage = f"{user_id}/{job_id}_vocals.wav"
            print(f"[{job_id}] Uploading vocals to assets/{vocal_path_storage}...")
            
            with open(vocal_path, "rb") as f:
                supabase.storage.from_("assets").upload(
                    path=vocal_path_storage,
                    file=f,
                    file_options={"content-type": "audio/wav", "x-upsert": "true"}
                )

            # Report Asset to API
            safe_callback(job_id, {
                "jobId": job_id,
                "userId": user_id,
                "event": "asset",
                "kind": "vocal_stem",
                "url": vocal_path_storage, 
                "metadata": {"precision": "high-demucs-v3"}
            })

            with open(local_analysis_file, "w") as f:
                json.dump(analysis_data, f)

            print(f"[{job_id}] Uploading analysis.json...")
            # Use 'application/octet-stream' to bypass strict MIME type filters
            # and 'x-upsert' to allow overwriting.
            supabase.storage.from_("assets").upload(
                path=analysis_path,
                file=local_analysis_file,
                file_options={"content-type": "application/octet-stream", "x-upsert": "true"}
            )

            # Report Asset to API
            safe_callback(job_id, {
                "jobId": job_id,
                "userId": user_id,
                "event": "asset",
                "kind": "analysis_json",
                "url": analysis_path,
                "metadata": {"segments_count": len(result["segments"]), "precision": "high-demucs-v3"}
            })

            # 6. Success
            # 6. Handover to Render Worker (Do not mark as completed yet)
            report_stage(job_id, "compositing", 90, user_id)
            print(f"[{job_id}] Audio processing finished. Handing off to Render Worker.")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"[{job_id}] Critical Error: {error_msg}")
            safe_callback(job_id, {
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
