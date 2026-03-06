---
license: cc-by-nc-4.0
task_categories:
  - image-classification
  - image-to-text
language:
  - en
tags:
  - emotion-recognition
  - context-based-emotion
  - valence-arousal-dominance
  - multimodal
  - image
  - social-scene-understanding
  - vlm
  - ms-swift
  - qwen
  - bounding-box
size_categories:
  - 10K<n<100K
---

# EMOTIC — Context-Based Emotion Recognition (Unified, MS-Swift Format)

This dataset is a reformatted version of [EMOTIC](http://sunai.uoc.edu/emotic/) for supervised fine-tuning of multimodal large language models (VLMs), packaged in the [MS-Swift](https://github.com/modelscope/ms-swift) Parquet format.

**Task:** Given a scene image and the bounding box of a person, recognize their emotional state from visual context, scene, and body language.

This repository contains **2 subsets** differing in the supervision signal:

| Subset | CLI name | Description |
|---|---|---|
| `discrete` | `emotic-vlm/discrete` | Predict list of active emotion categories (multi-label, 26 classes) |
| `vad` | `emotic-vlm/vad` | Predict valence, arousal, dominance scores (integers 1–9) |

---

## Dataset Structure

### Shared columns

| Column | Type | Description |
|---|---|---|
| `messages` | `list[{role, content}]` | System / user / assistant conversation |
| `images` | `list[Image]` | Scene image embedded as bytes |

### Data Splits

Official train/val/test split from the original dataset.

| Split | Images | `discrete` examples | `vad` examples |
|---|---|---|---|
| Train | 17,077 | ~23,700 | ~23,600 |
| Val | 2,088 | ~3,300 | ~3,300 |
| Test | 4,389 | ~7,100 | ~7,000 |

Multiple persons per image generate multiple examples. For val/test (which have 2–10 annotators per image), the first annotator's annotation is used — no aggregated/combined labels.

Images are sourced from MSCOCO, ADE20K, framesdb, and emodb_small.

---

## Subset Details

### `discrete`

**System:**
```
You are an expert in context-based emotion recognition.
Given an image and the bounding box of a person, identify their emotional state
from visual context, scene, and body language.
Choose from: [Affection, Anger, Annoyance, Anticipation, Aversion, Confidence,
Disapproval, Disconnection, Disquietment, Doubt/Confusion, Embarrassment,
Engagement, Esteem, Excitement, Fatigue, Fear, Happiness, Pain, Peace,
Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning].
Provide your answer as a valid JSON object: {"emotions": ["Emotion1", "Emotion2"]}.
The list must contain at least one emotion.
```

**User:**
```
<image>
Bounding box: [134, 58, 564, 628]
What are the emotions of the female adult in this bounding box?
```

**Assistant:** `{"emotions": ["Happiness", "Engagement", "Affection"]}`

The person description uses gender and age when known: `"female adult"`, `"male teenager"`, `"adult"` (unknown gender), `"person"` (both unknown).

---

### `vad`

**System:**
```
You are an expert in context-based emotion recognition.
Given an image and the bounding box of a person, predict their emotional state
as valence, arousal, and dominance scores.
Each score is an integer from 1 (very low) to 9 (very high).
Provide your answer as a valid JSON object: {"valence": x, "arousal": x, "dominance": x}.
```

**User:**
```
<image>
Bounding box: [134, 58, 564, 628]
What are the valence, arousal, and dominance scores of the female adult in this bounding box?
```

**Assistant:** `{"valence": 6, "arousal": 4, "dominance": 7}`

All bounding box coordinates are normalized to Qwen's `[0, 1000]` format.

---

## Usage with MS-Swift

```bash
# Discrete emotion prediction
swift sft \
  --model Qwen/Qwen3-VL-7B \
  --dataset emotic-vlm/discrete \
  --custom_plugin plugins/omni_dataset_plugin.py

# Valence / Arousal / Dominance regression
swift sft \
  --model Qwen/Qwen3-VL-7B \
  --dataset emotic-vlm/vad \
  --custom_plugin plugins/omni_dataset_plugin.py

# Both subsets at once
swift sft \
  --model Qwen/Qwen3-VL-7B \
  --dataset emotic-vlm/all \
  --custom_plugin plugins/omni_dataset_plugin.py
```

## Source Dataset

- **Original dataset:** [EMOTIC](http://sunai.uoc.edu/emotic/) — Kosti et al., CVPR 2017
- Images sourced from MSCOCO, ADE20K, framesdb, and emodb_small.
- 23,571 images, 34,320 annotated persons across train/val/test splits.
- Each person annotated with 26 discrete emotion categories and continuous VAD scores.

## Citation

```bibtex
@inproceedings{kosti2017emotion,
  title={Emotion recognition in context},
  author={Kosti, Ronak and Alvarez, Jose M and Recasens, Adria and Lapedriza, Agata},
  booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```
