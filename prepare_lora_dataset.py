#!/usr/bin/env python3
"""
prepare_lora_dataset.py
------------------------
Automated Audio Intelligence Pipeline for ACE-Step LoRA dataset prep.
Product Owner & Lead Architect: Boobalan Arjunan

Scans a folder of .wav files and, for each one:
  1. Trims dead silence from start/end
  2. Normalizes peak level to -1.0 dB
  3. Classifies as ONE-SHOT vs LOOP (based on onset count + duration)
     - one-shots: left untouched (no looping)
     - short loops (<45s): seamlessly looped (crossfaded) up to 45s
     - long loops (>45s): trimmed to exactly 45s
  4. Extracts BPM using librosa beat tracking
  5. Detects musical key using Krumhansl-Schmuckler profiles (Major/Minor)
  6. Detects spectral vibe (deep bass / balanced / bright)
  7. Renames the file to embed metadata
  8. Writes a .json sidecar and a .txt caption file (ACE-Step format)
  9. Quarantines corrupted or silent files to a dedicated folder
"""

import argparse
import json
import os
import sys
import re
import shutil
import glob
import numpy as np
from pathlib import Path

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("ERROR: missing dependencies. Run: pip install librosa soundfile numpy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_LOOP_SECONDS = 45.0
TARGET_PEAK_DB = -1.0
ONE_SHOT_MAX_DURATION = 3.0
ONE_SHOT_ONSET_THRESHOLD = 1
CROSSFADE_SECONDS = 0.05

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles — psychological pitch-class salience weights
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---------------------------------------------------------------------------
# DSP Helpers
# ---------------------------------------------------------------------------

def load_audio(path, sr=None):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def trim_silence(y, top_db=40):
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed if len(y_trimmed) > 0 else y


def normalize_peak(y, target_db=TARGET_PEAK_DB):
    peak = np.max(np.abs(y))
    if peak == 0:
        return y
    target_amp = 10 ** (target_db / 20.0)
    return y * (target_amp / peak)


def classify_one_shot(y, sr):
    """Classify using onset density — not just duration."""
    duration = len(y) / sr
    if duration <= ONE_SHOT_MAX_DURATION:
        return True
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=False)
    return len(onset_frames) <= ONE_SHOT_ONSET_THRESHOLD


def crossfade_concat(a, b, sr, fade_seconds=CROSSFADE_SECONDS):
    """Concatenate a -> b with an equal-power crossfade to avoid clicks."""
    fade_len = int(sr * fade_seconds)
    fade_len = min(fade_len, len(a) // 2, len(b) // 2)
    if fade_len <= 0:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, fade_len)
    fade_in  = np.linspace(0.0, 1.0, fade_len)
    mixed = (a[-fade_len:] * fade_out) + (b[:fade_len] * fade_in)
    return np.concatenate([a[:-fade_len], mixed, b[fade_len:]])


def loop_to_length(y, sr, target_seconds=TARGET_LOOP_SECONDS):
    """Repeat y with crossfades until it reaches target_seconds, then trim."""
    target_len = int(sr * target_seconds)
    if len(y) == 0:
        return y
    out = y.copy()
    while len(out) < target_len:
        out = crossfade_concat(out, y, sr)
    return out[:target_len]


def detect_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return round(float(np.atleast_1d(tempo)[0]), 1)


def detect_key(y, sr):
    """
    Krumhansl-Schmuckler key estimation.
    Correlates chroma against major and minor pitch-class salience profiles
    for all 12 root notes, returning the best-matching key.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    best_score = -np.inf
    best_key   = "C"
    best_mode  = "Major"

    for i in range(12):
        major_score = np.corrcoef(chroma_mean, np.roll(MAJOR_PROFILE, i))[0, 1]
        minor_score = np.corrcoef(chroma_mean, np.roll(MINOR_PROFILE, i))[0, 1]

        if major_score > best_score:
            best_score = major_score
            best_key   = NOTE_NAMES[i]
            best_mode  = "Major"
        if minor_score > best_score:
            best_score = minor_score
            best_key   = NOTE_NAMES[i]
            best_mode  = "Minor"

    return f"{best_key} {best_mode}", best_key, best_mode


def detect_vibe(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_c   = float(np.mean(centroid))
    if mean_c < 1500:
        return "deep bass", round(mean_c, 1)
    elif mean_c < 3000:
        return "balanced", round(mean_c, 1)
    else:
        return "bright", round(mean_c, 1)


def sanitize(text):
    text = re.sub(r"[^\w\s-]", "", text).strip()
    return re.sub(r"[\s]+", "_", text)


# ---------------------------------------------------------------------------
# Main Pipeline (Generator — yields log lines to the GUI)
# ---------------------------------------------------------------------------

def process_directory(input_dir, output_dir, quarantine_dir,
                      dataset_name="BollyHood Beats", max_duration_s=TARGET_LOOP_SECONDS):
    os.makedirs(output_dir,   exist_ok=True)
    os.makedirs(quarantine_dir, exist_ok=True)

    wav_files = sorted(glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True))
    yield f"🔍 Found {len(wav_files)} WAV files in {input_dir}\n"

    if not wav_files:
        yield "❌ No WAV files found. Check your input directory.\n"
        return

    success_count   = 0
    quarantine_count = 0

    for i, file_path in enumerate(wav_files, 1):
        filename = os.path.basename(file_path)
        yield f"⏳ [{i}/{len(wav_files)}] Analyzing {filename}...\n"

        try:
            # 1. Load
            y, sr = load_audio(file_path)
            if len(y) == 0:
                raise ValueError("Empty audio file.")

            # 2. Silence trim
            y = trim_silence(y)
            if len(y) == 0:
                raise ValueError("Audio is pure silence after trimming.")

            # 3. Peak normalize
            y = normalize_peak(y)

            # 4. Classify + shape
            one_shot = classify_one_shot(y, sr)
            duration  = len(y) / sr

            if one_shot:
                shape_tag = "oneshot"
            elif duration < max_duration_s:
                y = loop_to_length(y, sr, max_duration_s)
                shape_tag = "loop"
            else:
                y = y[:int(sr * max_duration_s)]
                shape_tag = "loop"

            final_duration = round(len(y) / sr, 2)

            # 5. Intelligence extraction
            bpm               = detect_bpm(y, sr)
            key_label, key_root, key_mode = detect_key(y, sr)
            vibe, centroid_hz = detect_vibe(y, sr)

            # 6. Caption
            caption = (f"{dataset_name} {shape_tag}, {int(round(bpm))} BPM, "
                       f"{key_root} {key_mode}, {vibe}")

            # 7. Smart rename
            stem = sanitize(Path(file_path).stem)
            safe_name = sanitize(dataset_name)
            key_safe  = key_root.replace("#", "sharp")
            new_name  = (f"{safe_name}_{shape_tag}_{int(round(bpm))}bpm_"
                         f"{key_safe}{key_mode}_{int(final_duration)}s_{stem}")

            out_wav  = os.path.join(output_dir, f"{new_name}.wav")
            out_json = os.path.join(output_dir, f"{new_name}.json")
            out_txt  = os.path.join(output_dir, f"{new_name}.txt")

            # 8. Write outputs
            sf.write(out_wav, y, sr)

            metadata = {
                "caption":            caption,
                "original_file":      filename,
                "renamed_file":       f"{new_name}.wav",
                "dataset_name":       dataset_name,
                "bpm":                bpm,
                "key":                key_label,
                "duration_seconds":   final_duration,
                "shape":              shape_tag,
                "spectral_vibe":      vibe,
                "spectral_centroid_hz": centroid_hz,
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(caption)

            success_count += 1
            yield f"✅ [{key_root} {key_mode} | {int(round(bpm))} BPM | {vibe}] → {new_name}\n"

        except Exception as e:
            quarantine_count += 1
            try:
                shutil.copy2(file_path, os.path.join(quarantine_dir, filename))
            except Exception:
                pass
            yield f"⚠️  QUARANTINED: {filename} — {e}\n"

    yield (f"\n🎉 Pipeline Complete! "
           f"Processed: {success_count} | Quarantined: {quarantine_count} | "
           f"Total: {len(wav_files)}\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ACE-Step LoRA — Automated Audio Intelligence Pipeline"
    )
    parser.add_argument("--input-dir",     required=True)
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--quarantine-dir",required=True)
    parser.add_argument("--dataset-name",  default="BollyHood Beats")
    parser.add_argument("--max-duration",  type=float, default=TARGET_LOOP_SECONDS)
    args = parser.parse_args()

    for line in process_directory(
        args.input_dir, args.output_dir, args.quarantine_dir,
        args.dataset_name, args.max_duration
    ):
        print(line, end="")


if __name__ == "__main__":
    main()
