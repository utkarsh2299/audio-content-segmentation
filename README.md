# Background Music Detection using pyAudioAnalysis

The script processes all audio files in a given folder and automatically identifies files that
contain background sound or music using pretrained models from **pyAudioAnalysis** and segments them.


## Features
- Speech / music / silence segmentation
- Saves segmentation output as `.txt`
- Separates audio into `with_music` and `no_music`
- Works across multiple pyAudioAnalysis versions

## Setup

```bash
pip install -r requirements.txt
````
Ensure `ffmpeg` is installed for MP3 support.

## Usage

1. Place audio files in `input_audios/`
2. Run:

```bash
python detect_background_music.py
```

3. Results:

```
output/
├── with_music/
├── no_music/
└── segments/
```

## Output Format

Each `.txt` file contains:

```
start_time,end_time,label
```

Example:

```
0.01,9.90,speech
23.50,184.30,music
```
## Output Format

The project is Open to contribution. Thanks.

