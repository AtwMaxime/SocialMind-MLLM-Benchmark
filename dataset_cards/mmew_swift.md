---
license: cc-by-nc-4.0
task_categories:
  - image-classification
  - video-classification
language:
  - en
tags:
  - micro-expression
  - facial-expression
  - action-units
  - emotion-recognition
  - multimodal
  - social-scene-understanding
  - ms-swift
  - qwen
  - chain-of-thought
size_categories:
  - 1K<n<10K
---

# MMEW — Micro & Macro Expression Wildness Database (MS-Swift Format)

This dataset is a reformatted version of [MMEW](https://ieeexplore.ieee.org/document/9774929) for supervised fine-tuning of vision-language models, packaged in the [MS-Swift](https://github.com/modelscope/ms-swift) Parquet format.

MMEW contains both **macro-expressions** (still face images, 30 subjects × 6 emotions × 5 poses) and **micro-expressions** (short frame sequences, 300 annotated clips with onset/apex/offset and Action Unit labels).

This repository contains **4 subsets** covering two levels of granularity (image and clip) and two types of output (AU detection and emotion classification):

| Subset | CLI name | Input | Output |
|---|---|---|---|
| `apex_au` | `mmew-omni/apex-au` | Apex frame image | Active Action Units |
| `apex_emotion` | `mmew-omni/apex-emotion` | Apex/still frame image | Emotion label |
| `clip_emotion` | `mmew-omni/clip-emotion` | 16-frame video clip | Emotion label |
| `clip_emotion_think` | `mmew-omni/clip-emotion-think` | 16-frame video clip | AU reasoning + Emotion label |

---

## Dataset Structure

### Data Splits

Subject-based split: subjects S01–S24 → train, S25–S30 → val.

| Subset | Train | Val |
|---|---|---|
| `apex_au` | 244 | 56 |
| `apex_emotion` | 964 (244 micro + 720 macro) | 236 (56 micro + 180 macro) |
| `clip_emotion` | 244 | 56 |
| `clip_emotion_think` | 244 | 56 |

---

## Subset Details

### `apex_au` — Action Unit detection from apex frame

**Input:** Single JPEG image of the apex frame (peak muscular intensity of the micro-expression).

**System:**
```
You are an expert in facial action unit analysis.
Given an apex frame of a micro-expression, identify the active facial action units.
Provide your answer as a valid JSON object: {"action_units": ["AU6", "AU12"]}
or {"action_units": []} if none are active.
```

**User:** `<image>\nWhich facial action units are active in this apex frame?`

**Assistant:** `{"action_units": ["AU4", "AU7"]}`

---

### `apex_emotion` — Emotion classification from a single frame

**Input:** Either a micro-expression apex frame or a macro-expression still face image.

Emotion labels: `happiness` · `surprise` · `disgust` · `fear` · `sadness` · `anger` · `others`

**System:**
```
You are an expert in facial expression recognition.
Given a face image, classify the emotion into one of: [...].
Provide your answer as a valid JSON object: {"emotion": "happiness"}.
```

**User:** `<image>\nWhat emotion is expressed in this face image?`

**Assistant:** `{"emotion": "disgust"}`

---

### `clip_emotion` — Emotion classification from a video clip

**Input:** 16-frame MP4 clip (H.264, 16 fps) extracted from the onset-to-offset range of a micro-expression sequence, with the apex frame always included among the 16 frames.

**System:**
```
You are an expert in micro-expression recognition.
Given a video clip of a face, classify the micro-expression emotion into one of: [...].
Provide your answer as a valid JSON object: {"emotion": "happiness"}.
```

**User:** `<video>\nWhat micro-expression emotion is shown in this clip?`

**Assistant:** `{"emotion": "surprise"}`

---

### `clip_emotion_think` — Chain-of-thought emotion classification

Same video input as `clip_emotion`, but the answer includes explicit AU reasoning in a `<think>` block before the emotion label — designed for training models with chain-of-thought supervision.

**System:**
```
You are an expert in micro-expression recognition.
Given a video clip of a face, first identify the active facial action units at the apex
frame inside <think> tags, then classify the micro-expression emotion.
Use the format: <think>AU reasoning</think>{"emotion": "happiness"}.
```

**User:** `<video>\nWhat micro-expression emotion is shown in this clip?`

**Assistant:**
```
<think>
1. Apex Spotting: The movement begins from a neutral state (Onset), reaches its maximum muscular intensity during the sequence (Apex), and then fades (Offset).
2. Action Units: At the peak intensity (Apex), the activated Action Units are AU6+12.
3. Deduction: The dynamic activation of these specific Action Units is a physical signature characteristic of happiness.
</think>
{"emotion": "happiness"}
```

---

## Usage with MS-Swift

```bash
# Action unit detection from apex frame
swift sft \
  --model Qwen/Qwen2.5-VL-7B \
  --dataset mmew-omni/apex-au \
  --custom_plugin plugins/omni_dataset_plugin.py

# Emotion from single frame (micro + macro)
swift sft \
  --model Qwen/Qwen2.5-VL-7B \
  --dataset mmew-omni/apex-emotion \
  --custom_plugin plugins/omni_dataset_plugin.py

# Emotion from clip
swift sft \
  --model Qwen/Qwen2.5-VL-7B \
  --dataset mmew-omni/clip-emotion \
  --custom_plugin plugins/omni_dataset_plugin.py

# Chain-of-thought with AU reasoning
swift sft \
  --model Qwen/Qwen2.5-VL-7B \
  --dataset mmew-omni/clip-emotion-think \
  --custom_plugin plugins/omni_dataset_plugin.py

# All subsets at once
swift sft \
  --model Qwen/Qwen2.5-VL-7B \
  --dataset mmew-omni/all \
  --custom_plugin plugins/omni_dataset_plugin.py
```

## Source Dataset

- **Original dataset:** [MMEW](https://ieeexplore.ieee.org/document/9774929) — Ben et al., IEEE TAFFC 2022
- 30 subjects, lab-controlled setting, annotated with onset/apex/offset frames and FACS Action Units.

## Citation

```bibtex
@article{ben2022video,
  title={Video-based Facial Micro-Expression Analysis: A Survey of Datasets, Features and Algorithms},
  author={Ben, Xianye and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022}
}
```
