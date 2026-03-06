---
license: cc-by-4.0
task_categories:
  - video-classification
  - image-classification
language:
  - en
tags:
  - gaze-estimation
  - multimodal
  - video
  - image
  - social-scene-understanding
  - ms-swift
  - qwen
size_categories:
  - 1K<n<10K
---

# VideoAttentionTarget — Gaze Target Estimation (Unified, MS-Swift Format)

This dataset is a reformatted version of [VideoAttentionTarget](https://github.com/ejcgt/attention-target-detection) for supervised fine-tuning of multimodal large language models, packaged in the [MS-Swift](https://github.com/modelscope/ms-swift) Parquet format.

**Task:** Given a person's head bounding box, predict where they are looking. If the gaze target is outside the frame, predict `out_of_frame`.

This repository contains **2 subsets** differing in the input modality:

| Subset | CLI name | Description |
|---|---|---|
| `video` | `vat-omni/video` | Full reconstructed video clip + timestamp |
| `frame` | `vat-omni/frame` | Single JPEG frame |

---

## Dataset Structure

### Shared columns

| Column | Type | Description |
|---|---|---|
| `messages` | `list[{role, content}]` | System / user / assistant conversation |
| `videos` | `list[binary]` | Raw MP4 bytes — `video` subset only |
| `images` | `list[Image]` | JPEG frames — `frame` subset only |

### Data Splits

Official train/test split preserved (40 shows train, 10 shows test).

| Split | video examples | frame examples |
|---|---|---|
| Train | ~5,165 | ~7,445 |
| Test | ~1,490 | ~1,569 |

---

## Subset Details

### `video`

**System:**
```
You are an expert in gaze target estimation in video.
Given a video clip and the bounding box of a person's head, predict where they
are looking at the specified timestamp.
Coordinates are normalized to [0, 1000].
If the gaze target is within the frame provide: {"gaze_point": [x, y], "label": "gaze target"}.
If the target is outside the frame provide: {"out_of_frame": true}.
```

**User:**
```
<video>
Person head bounding box: [312, 145, 398, 237]
At t=0.17s, where is this person looking?
```

**Assistant:** `{"gaze_point": [724, 382], "label": "gaze target"}`

Videos are reconstructed from the original JPEG frames (30 fps source) and downsampled to 16 fps (H.264, no audio). 5 frames are randomly sampled per tracked person per clip (seed 42); the prompt specifies the timestamp derived from the original frame index at 30 fps.

---

### `frame`

**System:**
```
You are an expert in gaze target estimation.
Given an image and the bounding box of a person's head, predict where they are looking.
Coordinates are normalized to [0, 1000].
If the gaze target is within the frame provide: {"gaze_point": [x, y], "label": "gaze target"}.
If the target is outside the frame provide: {"out_of_frame": true}.
```

**User:**
```
<image>
Person head bounding box: [312, 145, 398, 237]
Where is this person looking?
```

**Assistant:** `{"gaze_point": [724, 382], "label": "gaze target"}`

A spatial diversity filter is applied per sequence: a frame is only included if its gaze point is at least 25 units away (in [0, 1000] space) from all previously selected gaze points in the same sequence. At most one out-of-frame example is kept per sequence.

---

All coordinates are normalized to Qwen's `[0, 1000]` format. ~33% of examples have out-of-frame gaze targets.

---

## Usage with MS-Swift

```bash
# Video clip variant
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset vat-omni/video \
  --custom_plugin plugins/omni_dataset_plugin.py

# Single frame variant
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset vat-omni/frame \
  --custom_plugin plugins/omni_dataset_plugin.py

# Both subsets at once
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset vat-omni/all \
  --custom_plugin plugins/omni_dataset_plugin.py
```

## Source Dataset

- **Original dataset:** [VideoAttentionTarget](https://github.com/ejcgt/attention-target-detection) — Chong et al., CVPR 2020
- Clips sourced from 50 US TV shows (40 train / 10 test).
- Annotations provide per-frame head bounding boxes and gaze targets for tracked persons across sequences.

## Citation

```bibtex
@inproceedings{chong2020detecting,
  title={Detecting Attended Visual Targets in Video},
  author={Chong, Eunji and Wang, Yonglong and Ruber, Nataniel and Shi, Yun and Rehg, James M},
  booktitle={CVPR},
  year={2020}
}
```
