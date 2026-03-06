---
license: other
task_categories:
  - video-classification
language:
  - en
tags:
  - facial-expression
  - action-units
  - valence-arousal
  - affective-computing
  - multimodal
  - video
  - social-scene-understanding
  - ms-swift
  - qwen
size_categories:
  - 100K<n<1M
---

# AffWild2 — Facial Expression, VA & Action Unit Recognition (Unified, MS-Swift Format)

This dataset is a reformatted version of [AffWild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/) for supervised fine-tuning of multimodal large language models, packaged in the [MS-Swift](https://github.com/modelscope/ms-swift) Parquet format.

**Task:** Given a 16-frame video clip, predict the affective state at the center frame (frame 8). Four tasks are provided from the same in-the-wild video footage.

This repository contains **4 subsets**:

| Subset | CLI name | Description |
|---|---|---|
| `expr` | `affwild2-omni/expr` | Facial expression classification (8 classes) |
| `va` | `affwild2-omni/va` | Valence & arousal regression (continuous, [-1, 1]) |
| `au` | `affwild2-omni/au` | Action unit detection (12 AUs, multi-label) |
| `expr_think` | `affwild2-omni/expr-think` | Expression with CoT: VA + AU in `<think>`, then label |

---

## Dataset Structure

### Shared columns

| Column | Type | Description |
|---|---|---|
| `messages` | `list[{role, content}]` | System / user / assistant conversation |
| `videos` | `list[binary]` | Raw MP4 bytes (16 frames, H.264, no audio) |
| `audios` | `list[binary]` | Empty |

### Sliding window approach

Videos are processed with a **stride-16 sliding window**: 16 contiguous frames are extracted per window, and the annotation at the **center frame (index 8)** is used as the prediction target. A diversity filter is applied per task to avoid near-duplicate windows:

| Task | Diversity filter |
|---|---|
| `expr` | Max 5 windows per expression label per video |
| `va` | Min Euclidean distance 0.05 in [-1,1]² between selected VA points |
| `au` | Min Hamming distance 1 between selected AU vectors |

### Data Splits

Official ABAW train/val split. For `expr_think`, 20 train videos are borrowed into the val set to compensate for the small number of videos with all three annotations available.

| Split | `expr` | `va` | `au` | `expr_think` |
|---|---|---|---|---|
| Train | ~200K | ~200K | ~100K | ~10K |
| Val | ~50K | ~50K | ~25K | ~2K |

*(Approximate — exact counts depend on video availability and diversity filtering.)*

---

## Subset Details

### `expr`

**System:**
```
You are an expert in facial expression recognition.
Given a 16-frame video clip, predict the facial expression at the center frame.
Classify into one of: ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Other"].
Provide your answer as a valid JSON object: {"label": "Expression"}.
```

**User:** `<video>\nWhat is the facial expression at the center frame of this clip?`

**Assistant:** `{"label": "Happiness"}`

---

### `va`

**System:**
```
You are an expert in affective computing.
Given a 16-frame video clip, predict the valence and arousal at the center frame.
Both values are continuous in [-1, 1].
Provide your answer as a valid JSON object: {"valence": x.xxx, "arousal": x.xxx}.
```

**User:** `<video>\nWhat are the valence and arousal values at the center frame of this clip?`

**Assistant:** `{"valence": 0.312, "arousal": -0.145}`

---

### `au`

**System:**
```
You are an expert in facial action unit detection.
Given a 16-frame video clip, predict which action units are active at the center frame.
Possible action units: ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"].
Provide your answer as a valid JSON object: {"action_units": ["AU1", ...]} or {"action_units": []} if none are active.
```

**User:** `<video>\nWhich action units are active at the center frame of this clip?`

**Assistant:** `{"action_units": ["AU6", "AU12"]}`

---

### `expr_think`

Chain-of-thought variant requiring all three annotations (VA + AU + Expr) to be available for the same video. VA and AU are provided inside `<think>` tags as intermediate reasoning, followed by the expression label.

**User:** `<video>\nFor the center frame of this clip, provide the valence/arousal and active action units, then give the expression label.`

**Assistant:**
```
<think>
{"valence": 0.312, "arousal": 0.187}
{"action_units": ["AU6", "AU12"]}
</think>
{"label": "Happiness"}
```

---

## Usage with MS-Swift

```bash
# Expression classification
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset affwild2-omni/expr \
  --custom_plugin plugins/omni_dataset_plugin.py

# Valence / Arousal regression
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset affwild2-omni/va \
  --custom_plugin plugins/omni_dataset_plugin.py

# Action unit detection
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset affwild2-omni/au \
  --custom_plugin plugins/omni_dataset_plugin.py

# Chain-of-thought expression
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset affwild2-omni/expr-think \
  --custom_plugin plugins/omni_dataset_plugin.py

# All subsets at once
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset affwild2-omni/all \
  --custom_plugin plugins/omni_dataset_plugin.py
```

## Source Dataset

- **Original dataset:** [AffWild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/) — Kollias et al., CVPR 2019 & IJCV 2021
- ~2.8M frames from in-the-wild videos annotated for expression, VA, and action units as part of the ABAW challenge series.
- Access requires registration at the official website.

## Citation

```bibtex
@article{kollias2021affect,
  title={Affect analysis in-the-wild: Valence-arousal, expressions, action units and a unified framework},
  author={Kollias, Dimitrios and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:2103.15792},
  year={2021}
}
```
