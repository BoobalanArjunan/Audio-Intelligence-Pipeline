import os
import glob
import json
import numpy as np
import librosa
import soundfile as sf
import argparse
import shutil
from pathlib import Path

def get_key(y, sr):
    if y.ndim > 1:
        y_mono = librosa.to_mono(y.T)
    else:
        y_mono = y
        
    chroma = librosa.feature.chroma_stft(y=y_mono, sr=sr)
    chroma_sum = np.sum(chroma, axis=1)
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return keys[np.argmax(chroma_sum)]

def get_vibe(y, sr):
    if y.ndim > 1:
        y_mono = librosa.to_mono(y.T)
    else:
        y_mono = y
        
    centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid < 1500:
        return "deep bass"
    elif mean_centroid > 3500:
        return "bright"
    else:
        return "balanced"

def get_bpm(y, sr):
    if y.ndim > 1:
        y_mono = librosa.to_mono(y.T)
    else:
        y_mono = y
        
    # Reverting to the highly stable beat_track algorithm
    # The tempogram algorithm was too sensitive to 16th-note transients (detecting 199 BPM)
    tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]
    return int(round(tempo))

def process_directory(input_dir, output_dir, quarantine_dir, dataset_name="BollyHood Beats", max_duration_s=45, fixed_bpm=None):
    """
    Generator function that processes a directory of WAV files.
    Yields log messages for the GUI console.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(quarantine_dir, exist_ok=True)
    
    wav_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    yield f"🔍 Found {len(wav_files)} WAV files in {input_dir}\n"
    
    if len(wav_files) == 0:
        yield "❌ No WAV files found. Check your input directory.\n"
        return

    success_count = 0
    quarantine_count = 0

    for i, file_path in enumerate(wav_files, 1):
        filename = os.path.basename(file_path)
        yield f"⏳ [{i}/{len(wav_files)}] Analyzing {filename}...\n"
        
        try:
            # 1. Load Audio
            y, sr = sf.read(file_path)
            if len(y) == 0:
                raise ValueError("Audio file is completely empty.")
                
            # 2. Dead-Silence Trimming
            if y.ndim > 1:
                y_mono = librosa.to_mono(y.T)
            else:
                y_mono = y
                
            non_silent_intervals = librosa.effects.split(y_mono, top_db=40)
            if len(non_silent_intervals) > 0:
                start_idx = non_silent_intervals[0][0]
                end_idx = non_silent_intervals[-1][1]
                y = y[start_idx:end_idx]
                y_mono = y_mono[start_idx:end_idx]
            else:
                raise ValueError("Audio file consists of pure silence.")

            # 3. Auto-Normalization (-1.0 dB peak)
            target_peak = 10 ** (-1.0 / 20)
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y * (target_peak / max_val)

            # 4. Intelligence Extraction
            duration_s = len(y) / sr
            
            # AI/Manual Hybrid BPM System
            if fixed_bpm and str(fixed_bpm).strip() != "":
                bpm = int(fixed_bpm)
            else:
                bpm = get_bpm(y_mono, sr)
                
            key = get_key(y_mono, sr)
            vibe = get_vibe(y_mono, sr)
            
            # 5. Advanced Audio Shaping (One-Shots vs Loops)
            max_samples = int(max_duration_s * sr)
            is_one_shot = duration_s < 3.0
            
            if not is_one_shot:
                if len(y) < max_samples:
                    repeats = int(np.ceil(max_samples / len(y)))
                    y = np.tile(y, (repeats, 1)) if y.ndim > 1 else np.tile(y, repeats)
                y = y[:max_samples]

            duration_s = len(y) / sr

            # 6. Auto-Captioning
            type_str = "one-shot" if is_one_shot else "audio loop"
            caption = f"{dataset_name} {type_str}, {bpm} BPM, {key}, {vibe}"

            # 7. Smart File Renaming
            safe_name = dataset_name.lower().replace(" ", "_").replace("-", "")
            key_safe = key.replace("#", "sharp")
            vibe_safe = vibe.replace(" ", "_")
            
            base_new_name = f"{safe_name}_{type_str.replace('-', '_')}_{bpm}bpm_{key_safe}_{vibe_safe}_{int(duration_s)}s_{Path(file_path).stem}"
            base_new_name = base_new_name.replace(" ", "_")
            
            out_wav = os.path.join(output_dir, f"{base_new_name}.wav")
            out_json = os.path.join(output_dir, f"{base_new_name}.json")
            out_txt = os.path.join(output_dir, f"{base_new_name}.txt")
            
            # 8. Save Files
            sf.write(out_wav, y, sr)
            
            metadata = {"caption": caption, "bpm": bpm, "key": key, "vibe": vibe, "duration_s": duration_s}
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
                
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(caption)

            success_count += 1
            yield f"✅ Success: {base_new_name}\n"

        except Exception as e:
            # Quarantine the file
            quarantine_count += 1
            quarantine_path = os.path.join(quarantine_dir, filename)
            shutil.copy2(file_path, quarantine_path)
            error_msg = str(e)
            yield f"⚠️ QUARANTINED: {filename} (Error: {error_msg})\n"
            
    yield f"\n🎉 Pipeline Complete! Successfully processed {success_count} files. Quarantined {quarantine_count} files.\n"

def main():
    parser = argparse.ArgumentParser(description="Automated Audio Intelligence Pipeline")
    parser.add_argument("--input-dir", required=True, help="Directory with original WAV files")
    parser.add_argument("--output-dir", required=True, help="Directory to save processed dataset")
    parser.add_argument("--quarantine-dir", required=True, help="Directory to save corrupted/bad audio files")
    parser.add_argument("--dataset-name", default="BollyHood Beats", help="Prefix for captions")
    parser.add_argument("--max-duration", type=float, default=45.0, help="Max duration in seconds")
    parser.add_argument("--fixed-bpm", type=str, default=None, help="Force a specific BPM (bypasses auto-detection)")
    args = parser.parse_args()

    for log in process_directory(args.input_dir, args.output_dir, args.quarantine_dir, args.dataset_name, args.max_duration, args.fixed_bpm):
        print(log, end="")

if __name__ == "__main__":
    main()
