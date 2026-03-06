# Social VLM Dataset Pipeline

A **bring-your-own-data** pipeline for building fine-tuning datasets for multimodal large language models (VLMs) on social scene understanding tasks. Each dataset is converted into a standardized Parquet format compatible with [MS-Swift](https://github.com/modelscope/ms-swift) for fine-tuning models such as Qwen3-Omni.

---

## Concept

Social scene understanding requires models to reason about people — who they are looking at, how they feel, and how they relate to each other. This pipeline unifies several datasets covering these tasks into a single consistent format, ready to use for supervised fine-tuning.

The **bring-your-own-data** approach means:
- You download the raw datasets yourself (links below) and place them in the `dataset/` folder
- You run the corresponding `*_parquet.py` script to build the Parquet file
- You verify the output with the corresponding `check_*.py` script
- The resulting Parquet files are self-contained (images and audio are embedded as bytes — no external file references at training time)

---

## Project Structure

```
data/
├── dataset/                        # Raw datasets (downloaded by the user)
│   ├── gazefollow/                 # GazeFollow JSONL files
│   ├── meld/                       # MELD JSONL files + audio
│   ├── PISC/                       # PISC images + annotation JSON files
│   ├── VocalSound_release_16k/     # VocalSound audio files
│   ├── dataset_proxemics/          # Proxemics images + annotation JSON files
│   ├── MUStARD/                    # MUStARD videos + annotation JSON
│   ├── videoattentiontarget/       # VideoAttentionTarget frames + annotations
│   └── EMOTIC/                     # EMOTIC images + Annotations.mat
│
├── builders/                       # Parquet builder scripts
│   ├── gazefollow_parquet.py
│   ├── meld_parquet.py
│   ├── pisc_parquet.py
│   ├── vocalsound_parquet.py
│   ├── proxemics_parquet.py
│   ├── mustard_parquet.py
│   ├── videoattentiontarget_parquet.py
│   └── emotic_parquet.py
│
├── checks/                         # Sanity check scripts
│   ├── check_gazefollow.py
│   ├── check_meld.py
│   ├── check_pisc.py
│   ├── check_vocalsound.py
│   ├── check_proxemics.py
│   ├── check_mustard.py
│   ├── check_videoattentiontarget.py
│   └── check_emotic.py
│
├── parquets/                       # Output: all generated parquet files
│   ├── gazefollow/
│   ├── meld/
│   ├── pisc/
│   ├── vocalsound/
│   ├── proxemics/
│   │   ├── skeleton/
│   │   └── no_skeleton/
│   ├── mustard/
│   │   ├── video_no_context/
│   │   └── video_context/
│   ├── videoattentiontarget/
│   │   ├── video/
│   │   └── frame/
│   └── emotic/
│       ├── discrete/
│       └── vad/
│
└── plugins/
    └── omni_dataset_plugin.py      # MS-Swift plugin to register all datasets
```

---

## Supported Datasets

### GazeFollow — Gaze Target Estimation
**Task:** Given an image and the bounding box of a person's head, predict where they are looking (gaze target bounding box).

| Split      | Examples |
|------------|----------|
| Train      | 113,001  |
| Validation | 12,556   |

**Download:** [GazeFollow Dataset](http://gazefollow.csail.mit.edu/)
Place files in `dataset/gazefollow/`.

**Build:**
```bash
python builders/gazefollow_parquet.py
python checks/check_gazefollow.py
```

---

### MELD — Multimodal Emotion Recognition
**Task:** Given a clip of a speaker in a conversation, classify their emotion into one of 7 categories: Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise.

Three modality variants are produced from the same clips:

| Variant | Input | Prompt |
|---|---|---|
| `video_audio` | 16fps video + audio | `<video>` + bbox + `<audio>` |
| `audio_only` | Audio only | `<audio>` |
| `video_transcript` | 16fps video (silent) + utterance text | `<video>` + bbox + transcript |

| Split      | Examples (per variant) |
|------------|------------------------|
| Train      | 21,405                 |
| Validation | 2,394                  |
| Test       | 5,413                  |

Videos are downsampled to 16 fps (H.264). The active speaker's face bounding box (from MELD-FAIR annotations) is included for video variants, normalized to Qwen's `[0, 1000]` format.

**Download:** Place the [MELD-FAIR](https://github.com/facebookresearch/MELD-FAIR) folder in `dataset/MELD-FAIR/`.

**Build:**
```bash
python builders/meld_parquet.py
python checks/check_meld.py
```

The builder caches downsampled videos in `parquets/meld/.video_cache/` to avoid re-encoding when building multiple variants.

---

### VideoAttentionTarget — Video Gaze Target Estimation
**Task:** Given a video clip and the bounding box of a person's head, predict where they are looking at a specified timestamp. If the gaze target is outside the frame, predict `out_of_frame`.

Two variants are produced:

| Variant | Input | Prompt |
|---|---|---|
| `vat-video` | 16fps reconstructed clip | `<video>` + head bbox + timestamp |
| `vat-frame` | Single frame (JPEG) | `<image>` + head bbox |

| Split | Video examples | Frame examples |
|-------|---------------|----------------|
| Train | ~5,165        | 132,563        |
| Test  | ~1,490        | 31,978         |

Video examples: 5 random frames sampled per tracked person per clip. Frame examples: all annotated frames. ~33% of frames have out-of-frame gaze targets. Official train/test split (40 shows train, 10 shows test).

**Answer format:**
```json
{"gaze_point": [x, y], "label": "gaze target"}
{"out_of_frame": true}
```
Coordinates normalized to `[0, 1000]`.

**Download:** Place files in `dataset/videoattentiontarget/`.

**Build:**
```bash
python builders/videoattentiontarget_parquet.py
python checks/check_videoattentiontarget.py
```

---

### PISC — People in Social Context (Relationship Recognition)
**Task:** Given an image and the bounding boxes of two individuals, classify their fine-grained social relationship into one of 6 categories: Friends, Family, Couple, Professional, Commercial, No relation.

| Split      | Examples |
|------------|----------|
| Train      | 55,400   |
| Validation | 1,505    |
| Test       | 3,961    |

**Download:** [PISC Dataset](https://zenodo.org/record/1059155)
Place files in `dataset/PISC/`. Extract the image archive with:
```bash
cat images-00 images-01 images-02 images-03 | tar xz -C dataset/PISC/
```

**Build:**
```bash
python builders/pisc_parquet.py
python checks/check_pisc.py
```

---

## Parquet Format

All datasets share a unified schema:

| Column     | Type                    | Description                            |
|------------|-------------------------|----------------------------------------|
| `messages` | `list[{role, content}]` | System / user / assistant conversation |
| `images`   | `list[Image]`           | PIL images embedded as bytes           |
| `audios`   | `list[binary]`          | Raw WAV bytes                          |
| `videos`   | `list[binary]` *(MELD video variants)* | Raw MP4 bytes (16 fps, H.264) |

The **bounding box format** follows Qwen's convention: `[x1, y1, x2, y2]` with coordinates normalized to `[0, 1000]`.

The **assistant response** is always a JSON string, e.g.:
```json
{"relationship": "Friends"}
{"emotion": "neutral"}
{"bbox_2d": [424, 614, 513, 681], "label": "gaze target"}
{"touching": ["Hand touch hand", "Shoulder touch shoulder"]}
{"vocal_sound": "Laughter"}
```

---

## MS-Swift Integration

A plugin file registers all datasets with MS-Swift under friendly aliases:

| Alias                        | HF dataset                       | Subsets / notes                        | Modality       |
|------------------------------|----------------------------------|----------------------------------------|----------------|
| `gazefollow-vlm`             | `AtwMaxime/gazefollow_swift`     | —                                      | Image          |
| `pisc-vlm`                   | `AtwMaxime/pisc_swift`           | —                                      | Image          |
| `vocalsound-omni`            | `AtwMaxime/vocalsound_swift`     | —                                      | Audio          |
| `rldd-video`                 | `AtwMaxime/rldd_swift`           | —                                      | Video          |
| `meld-omni/video-audio`      | `AtwMaxime/meld_swift`           | video + audio                          | Video + Audio  |
| `meld-omni/audio-only`       | `AtwMaxime/meld_swift`           | audio only                             | Audio          |
| `meld-omni/video-transcript` | `AtwMaxime/meld_swift`           | video + transcript text                | Video + Text   |
| `proxemics-omni/skeleton`    | `AtwMaxime/proxemics_swift`      | image + bbox + skeleton keypoints      | Image          |
| `proxemics-omni/no-skeleton` | `AtwMaxime/proxemics_swift`      | image + bbox only                      | Image          |
| `mustard-omni/video-no-context` | `AtwMaxime/mustard_swift`     | video + audio                          | Video + Audio  |
| `mustard-omni/video-context` | `AtwMaxime/mustard_swift`        | video + audio + dialogue context       | Video + Audio  |
| `vat-omni/video`             | `AtwMaxime/vat_swift`            | full reconstructed clip + timestamp    | Video          |
| `vat-omni/frame`             | `AtwMaxime/vat_swift`            | single JPEG frame                      | Image          |
| `mmew-omni/apex-au`          | `AtwMaxime/mmew_swift`           | apex frame → action units             | Image          |
| `mmew-omni/apex-emotion`     | `AtwMaxime/mmew_swift`           | apex frame → emotion label            | Image          |
| `mmew-omni/clip-emotion`     | `AtwMaxime/mmew_swift`           | 16-frame clip → emotion label         | Video          |
| `mmew-omni/clip-emotion-think` | `AtwMaxime/mmew_swift`         | 16-frame clip → CoT + emotion label   | Video          |
| `emotic-vlm/discrete`        | `AtwMaxime/emotic_swift`         | image + bbox → emotion list (26 cats) | Image          |
| `emotic-vlm/vad`             | `AtwMaxime/emotic_swift`         | image + bbox → VAD scores (1–9)       | Image          |

Pass the plugin to your Swift training command:
```bash
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset gazefollow-vlm pisc-vlm meld-omni/all proxemics-omni/all emotic-vlm/all \
  --custom_plugin plugins/omni_dataset_plugin.py \
  ...
```

---

### Proxemics — Body Contact Recognition
**Task:** Given an image and the bounding boxes of two individuals (optionally with skeleton keypoints), identify which body parts are touching between them. Multi-label classification across 6 contact categories: Hand touch hand, Hand touch shoulder, Shoulder touch shoulder, Hand touch elbow, Elbow touch shoulder, Hand touch torso.

Two variants are provided: **with skeleton** (10 labeled keypoints per person included in the prompt) and **without skeleton** (bounding boxes only).

| Split | Examples |
|-------|----------|
| Train | 471      |
| Test  | 118      |

**Download:** Place files in `dataset/dataset_proxemics/`.

**Build:**
```bash
python builders/proxemics_parquet.py
python checks/check_proxemics.py
```

---

### MUStARD — Multimodal Sarcasm Detection
**Task:** Given a video clip with audio of a speaker, determine whether their utterance is sarcastic. Binary classification: sarcastic or not.

Both variants always include video + audio (audio embedded in the video at 16fps). The difference is whether the preceding dialogue context is provided as text.

| Variant | Input | Prompt |
|---|---|---|
| `video_audio` | 16fps video + audio | `<video>` |
| `video_context` | 16fps video + audio + preceding dialogue | `<video>` + context text |

| Split | Examples |
|-------|----------|
| Train | 552      |
| Test  | 138      |

690 clips from TV shows (FRIENDS, BBT, Golden Girls), balanced (50% sarcastic). No official split — 80/20 with seed 42.

**Download:** Place the `MUStARD` folder in `dataset/MUStARD/`.

**Build:**
```bash
python builders/mustard_parquet.py
python checks/check_mustard.py
```

---

### VocalSound — Vocal Sound Classification
**Task:** Given an audio clip of a person producing a vocal sound, classify it into one of 6 categories: Laughter, Sigh, Cough, Throat clearing, Sneeze, Sniff.

| Split      | Examples |
|------------|----------|
| Train      | 15,531   |
| Validation | 1,855    |
| Test       | 3,591    |

**Download:** [VocalSound Dataset](https://github.com/yuangongnd/vocalsound)
Place files in `dataset/VocalSound_release_16k/`.

**Build:**
```bash
python builders/vocalsound_parquet.py
python checks/check_vocalsound.py
```

---

### EMOTIC — Context-Based Emotion Recognition
**Task:** Given a scene image and the bounding box of a person, recognize their emotional state from visual context, scene, and body language. Two supervision signals:

| Variant | Input | Answer |
|---|---|---|
| `discrete` | Image + person bbox | Multi-label emotion list (26 categories) |
| `vad` | Image + person bbox | Valence / Arousal / Dominance scores (1–9) |

The person description in the prompt includes gender and age when known (e.g. `"female adult"`, `"male teenager"`).

| Split | Images | Discrete examples | VAD examples |
|-------|--------|-------------------|--------------|
| Train | 17,077 | ~23,700 | ~23,600 |
| Val   | 2,088  | ~3,300  | ~3,300  |
| Test  | 4,389  | ~7,100  | ~7,000  |

Official split from the original dataset. For val/test (2–10 annotators), the first annotator's annotation is used — no combined labels.

**Download:** Place the `EMOTIC` folder in `dataset/EMOTIC/` (must contain `Annotations/Annotations.mat` and image subfolders: `mscoco/`, `ade20k/`, `framesdb/`, `emodb_small/`).

**Build:**
```bash
python builders/emotic_parquet.py
python checks/check_emotic.py
```

---

## Adding a New Dataset

1. Place raw data in `dataset/<dataset_name>/`
2. Create `<dataset_name>_parquet.py` following the same structure as the existing builders
3. Create `check_<dataset_name>.py` to verify a sample
4. Add a `register_dataset` entry in `plugins/omni_dataset_plugin.py`
5. Upload the parquet to Hugging Face Hub and reference it via `hf_dataset_id`

---

## Requirements

```bash
pip install datasets soundfile pillow pyarrow
```
