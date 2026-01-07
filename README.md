# Background Music Detection using pyAudioAnalysis

This code detects speech and background sound/music in audio files using
pretrained models from **pyAudioAnalysis** and segments them in different folders.

## Features
- Speech / music / silence segmentation
- Saves segmentation output as `.txt`
- Separates audio into `with_music` and `no_music`
- Works across multiple pyAudioAnalysis versions

## Setup

```bash
pip install -r requirements.txt
