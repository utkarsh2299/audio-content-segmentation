#!/usr/bin/env python3
"""
Background Music Detection using pyAudioAnalysis

"""

import os
import shutil
import io
import contextlib

import matplotlib
matplotlib.use("Agg")  #

import pyAudioAnalysis
from pyAudioAnalysis import audioSegmentation as aS


# ===================== CONFIG =====================

INPUT_DIR = "input_audios"
OUTPUT_DIR = "output"

WITH_MUSIC_DIR = os.path.join(OUTPUT_DIR, "with_music")
NO_MUSIC_DIR = os.path.join(OUTPUT_DIR, "no_music")
SEGMENT_DIR = os.path.join(OUTPUT_DIR, "segments")

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")

# ==================================================


def ensure_dirs():
    """Ensure output directories exist."""
    os.makedirs(WITH_MUSIC_DIR, exist_ok=True)
    os.makedirs(NO_MUSIC_DIR, exist_ok=True)
    os.makedirs(SEGMENT_DIR, exist_ok=True)


def get_model_path():
    """Locate pretrained pyAudioAnalysis model."""
    return os.path.join(
        os.path.dirname(pyAudioAnalysis.__file__),
        "data",
        "models",
        "svm_rbf_sm"
    )


MODEL_PATH = get_model_path()


def save_segmentation_txt(audio_path: str, out_txt: str) -> None:
    """
    First pass:
    Capture printed segmentation output and save to TXT.
    """
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        aS.mid_term_file_classification(
            audio_path,
            MODEL_PATH,
            "svm_rbf"
        )

    output = buffer.getvalue()
    if output.strip():
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(output)


def detect_music_from_api(audio_path: str):
    """
    Second pass:
    Use returned API values to detect music.
    """
    result = aS.mid_term_file_classification(
        audio_path,
        MODEL_PATH,
        "svm_rbf",
        True
    )

    flags = result[0]
    class_names = result[1]
    mt_step = result[2]

    segments = []
    has_music = False

    start_time = 0.0

    for i, label_idx in enumerate(flags):
        end_time = (i + 1) * mt_step
        label = class_names[int(label_idx)]

        segments.append(
            (round(start_time, 2), round(end_time, 2), label)
        )

        if label == "music":
            has_music = True

        start_time = end_time

    return segments, has_music


def process_audio(audio_path: str, out_txt: str):
    """
    Process a single audio file.

    Returns
    -------
    segments : list | None
    has_music : bool
    """
    try:
        save_segmentation_txt(audio_path, out_txt)
    except Exception as e:
        print(f"[ERROR] TXT generation failed for {audio_path}: {e}")

    try:
        return detect_music_from_api(audio_path)
    except Exception as e:
        print(f"[ERROR] Music detection failed for {audio_path}: {e}")
        return None, False


def main():
    ensure_dirs()
    if not os.path.isdir(INPUT_DIR):
        
        raise ValueError(f"Path is not a valid directory or does not exist: {INPUT_DIR}")
    if not any(os.scandir(INPUT_DIR)):
        print("Audio Directory is Empty.")
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(AUDIO_EXTENSIONS):
            continue

        audio_path = os.path.join(INPUT_DIR, fname)
        out_txt = os.path.join(
            SEGMENT_DIR,
            os.path.splitext(fname)[0] + ".txt"
        )

        print(f"Processing: {fname}")
        segments, has_music = process_audio(audio_path, out_txt)

        if segments is None:
            continue

        target_dir = WITH_MUSIC_DIR if has_music else NO_MUSIC_DIR
        shutil.copy2(audio_path, target_dir)

    print("Processing completed")


if __name__ == "__main__":
    main()
