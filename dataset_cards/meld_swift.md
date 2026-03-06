---
license: cc-by-4.0
task_categories:
  - video-classification
  - audio-classification
language:
  - en
tags:
  - emotion-recognition
  - multimodal
  - video
  - audio
  - social-scene-understanding
  - ms-swift
  - qwen
  - bounding-box
size_categories:
  - 10K<n<100K
---

# MELD — Multimodal Emotion Recognition (Unified, MS-Swift Format)

This dataset is a reformatted version of [MELD-FAIR](https://github.com/facebookresearch/MELD-FAIR) for supervised fine-tuning of multimodal large language models, packaged in the [MS-Swift](https://github.com/modelscope/ms-swift) Parquet format.

**Task:** Given a conversational clip, classify the speaker's emotion into one of 7 categories:
`Anger` · `Disgust` · `Fear` · `Joy` · `Neutral` · `Sadness` · `Surprise`

This repository contains **3 subsets** (modality variants) of the same task:

| Subset | CLI name | Modalities |
|---|---|---|
| `video_audio` | `meld-omni/video-audio` | Video clip (16 fps, H.264) + Audio (WAV 16 kHz) |
| `audio_only` | `meld-omni/audio-only` | Audio only (WAV 16 kHz) |
| `video_transcript` | `meld-omni/video-transcript` | Video clip + speech transcript in text |

---

## Dataset Structure

### Shared columns

| Column | Type | Description |
|---|---|---|
| `messages` | `list[{role, content}]` | System / user / assistant conversation |
| `videos` | `list[binary]` | Raw MP4 bytes — 16 frames, H.264, no audio track (empty for `audio_only`) |
| `audios` | `list[binary]` | Raw WAV bytes at 16 kHz (empty for `video_transcript`) |

### Data Splits

| Split | Examples |
|---|---|
| Train | 21,405 |
| Validation | 2,394 |
| Test | 5,413 |

---

## Subset Details

### `video_audio`

**System:**
```
You are an expert in multimodal emotion recognition.
Given a video clip and an audio recording of a speaker in a conversation,
classify their emotion into one of: ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"].
Provide your answer as a valid JSON object: {"emotion": "Category"}.
```

**User:**
```
<video>
Speaker face bounding box: [x1, y1, x2, y2]
<audio>
What is the emotion of the speaker?
```

**Assistant:** `{"emotion": "Joy"}`

---

### `audio_only`

**System:**
```
You are an expert in emotion recognition from speech.
Given an audio recording of a speaker in a conversation,
classify their emotion into one of: ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"].
Provide your answer as a valid JSON object: {"emotion": "Category"}.
```

**User:**
```
<audio>
What is the emotion of the speaker?
```

**Assistant:** `{"emotion": "Neutral"}`

---

### `video_transcript`

**System:**
```
You are an expert in multimodal emotion recognition.
Given a video clip and the transcript of a speaker in a conversation,
classify their emotion into one of: ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"].
Provide your answer as a valid JSON object: {"emotion": "Category"}.
```

**User:**
```
<video>
Speaker face bounding box: [x1, y1, x2, y2]
Transcript: "..."
What is the emotion of the speaker?
```

**Assistant:** `{"emotion": "Sadness"}`

---

## Usage with MS-Swift

```bash
# Video + Audio variant
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset meld-omni/video-audio \
  --custom_plugin plugins/omni_dataset_plugin.py

# Audio only variant
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset meld-omni/audio-only \
  --custom_plugin plugins/omni_dataset_plugin.py

# Video + Transcript variant
swift sft \
  --model Qwen/Qwen2.5-VL-7B \
  --dataset meld-omni/video-transcript \
  --custom_plugin plugins/omni_dataset_plugin.py

# All subsets at once
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset meld-omni/all \
  --custom_plugin plugins/omni_dataset_plugin.py
```

## Notes

- The speaker's face bounding box (from MELD-FAIR annotations) is normalized to Qwen's `[0, 1000]` format and corresponds to the middle frame of the clip. It is omitted when not available.
- Videos contain exactly 16 frames sampled evenly across the full clip duration.

## Source Dataset

- **Original dataset:** [MELD](https://affective-meld.github.io/) — Poria et al., ACL 2019
- **FAIR annotations:** [MELD-FAIR](https://github.com/facebookresearch/MELD-FAIR) — face bounding boxes and realigned splits
- Clips are sourced from the TV show *Friends*.

## Citation

```bibtex
@inproceedings{poria2019meld,
  title={MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations},
  author={Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  booktitle={ACL},
  year={2019}
}
```
