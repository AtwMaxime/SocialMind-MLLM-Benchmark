---
license: cc-by-4.0
task_categories:
  - visual-question-answering
language:
  - en
tags:
  - body-contact
  - physical-interaction
  - social-scene-understanding
  - vlm
  - ms-swift
  - qwen
  - bounding-box
  - skeleton
  - keypoints
size_categories:
  - n<1K
---

# Proxemics — Body Contact Recognition (Unified, MS-Swift Format)

This dataset is a reformatted version of the Proxemics dataset for supervised fine-tuning of multimodal large language models (VLMs), packaged in the [MS-Swift](https://github.com/modelscope/ms-swift) Parquet format.

**Task:** Given an image of two individuals with their bounding boxes (and optionally skeleton keypoints), identify which body parts are touching between them. Multi-label classification.

This repository contains **2 subsets** differing only in whether skeleton keypoints are provided:

| Subset | CLI name | Description |
|---|---|---|
| `skeleton` | `proxemics-omni/skeleton` | Image + bounding boxes + skeleton keypoints |
| `no_skeleton` | `proxemics-omni/no-skeleton` | Image + bounding boxes only |

---

## Dataset Structure

### Shared columns

| Column | Type | Description |
|---|---|---|
| `messages` | `list[{role, content}]` | System / user / assistant conversation |
| `images` | `list[Image]` | Scene image (224×224) embedded as PIL bytes |

### Contact Categories (multi-label)

`Hand touch hand` · `Hand touch shoulder` · `Shoulder touch shoulder` · `Hand touch elbow` · `Elbow touch shoulder` · `Hand touch torso`

The output list may be empty if no body parts are touching.

### Keypoints (10 per person, `skeleton` subset only)

`Left Head` · `Right Head` · `Left Shoulder` · `Right Shoulder` · `Left Elbow` · `Right Elbow` · `Left Hand` · `Right Hand` · `Left Torso` · `Right Torso`

### Data Splits

No official split — 80/20 random split with seed 42.

| Split | Examples |
|---|---|
| Train | 471 |
| Test | 118 |

---

## Subset Details

### `skeleton`

**System:**
```
You are an expert in analyzing physical contact between people.
Given an image of two individuals and their body information, identify which body
parts are touching between them. Choose from:
["Hand touch hand", "Hand touch shoulder", "Shoulder touch shoulder",
"Hand touch elbow", "Elbow touch shoulder", "Hand touch torso"].
Provide your answer as a valid JSON string with a list of touching body part pairs.
The list may be empty if no body parts are touching.
```

**User:**
```
<image>
Here are the bounding boxes of the 2 persons:
Person 1 bounding box: [x1, y1, x2, y2]
Person 2 bounding box: [x1, y1, x2, y2]

Person 1 skeleton:
  - Left Head: [x, y]
  - Right Head: [x, y]
  ...

Person 2 skeleton:
  - Left Head: [x, y]
  ...

Evaluate which body parts are touching between Person 1 and Person 2.
```

**Assistant:** `{"touching": ["Hand touch hand", "Shoulder touch shoulder"]}`

---

### `no_skeleton`

Same as above but without the skeleton keypoints section in the user message.

**User:**
```
<image>
Here are the bounding boxes of the 2 persons:
Person 1 bounding box: [x1, y1, x2, y2]
Person 2 bounding box: [x1, y1, x2, y2]

Evaluate which body parts are touching between Person 1 and Person 2.
```

**Assistant:** `{"touching": ["Hand touch hand"]}`

All coordinates are normalized to Qwen's `[0, 1000]` format.

---

## Usage with MS-Swift

```bash
# With skeleton keypoints
swift sft \
  --model Qwen/Qwen3-VL-7B \
  --dataset proxemics-omni/skeleton \
  --custom_plugin plugins/omni_dataset_plugin.py

# Bounding boxes only
swift sft \
  --model Qwen/Qwen3-VL-7B \
  --dataset proxemics-omni/no-skeleton \
  --custom_plugin plugins/omni_dataset_plugin.py

# Both subsets at once
swift sft \
  --model Qwen/Qwen3-VL-7B \
  --dataset proxemics-omni/all \
  --custom_plugin plugins/omni_dataset_plugin.py
```

## Source Dataset

Images depict pairs of people in social interactions, annotated for physical contact type and body keypoint locations.
