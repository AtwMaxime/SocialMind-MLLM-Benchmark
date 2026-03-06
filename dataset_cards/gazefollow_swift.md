---
license: cc-by-4.0
task_categories:
  - visual-question-answering
language:
  - en
tags:
  - gaze-estimation
  - social-scene-understanding
  - vlm
  - ms-swift
  - qwen
  - bounding-box
size_categories:
  - 100K<n<1M
---

# GazeFollow — Gaze Target Estimation (VLM Format)

This dataset is a reformatted version of [GazeFollow](http://gazefollow.csail.mit.edu/) for supervised fine-tuning of multimodal large language models (VLMs), packaged in the [MS-Swift](https://github.com/modelscope/ms-swift) Parquet format.

**Task:** Given an image and the bounding box of a person's head, predict where they are looking (gaze target bounding box).

## Dataset Structure

### Features

| Column | Type | Description |
|--------|------|-------------|
| `messages` | `list[{role, content}]` | System / user / assistant conversation |
| `images` | `list[Image]` | Scene image embedded as PIL bytes |

### Prompt Format

**System:**
```
You are a vision-language model specialized in gaze target estimation.
Given an image and the bounding box of a person's head, predict the bounding box
of what they are looking at. Bounding boxes use [x1, y1, x2, y2] format with
coordinates normalized to [0, 1000]. Provide your answer as a valid JSON object.
```

**User:**
```
<image>
Head bounding box: [x1, y1, x2, y2]
Where is this person looking?
```

**Assistant:**
```json
{"bbox_2d": [424, 614, 513, 681], "label": "gaze target"}
```

Bounding box coordinates follow Qwen's convention: `[x1, y1, x2, y2]` normalized to `[0, 1000]`.

### Data Splits

| Split | Examples |
|-------|----------|
| Train | 113,001 |
| Validation | 12,556 |

## Usage with MS-Swift

```python
from swift.llm.dataset import register_dataset, DatasetMeta, MessagesPreprocessor

register_dataset(
    DatasetMeta(
        dataset_name='gazefollow-vlm',
        hf_dataset_id='AtwMaxime/gazefollow_swift',
        preprocess_func=MessagesPreprocessor()
    )
)
```

```bash
swift sft \
  --model Qwen/Qwen3-VL-7B \
  --dataset gazefollow-vlm \
  --custom_plugin plugins/omni_dataset_plugin.py
```

## Source Dataset

- **Original dataset:** [GazeFollow](http://gazefollow.csail.mit.edu/) — Recasens et al., NeurIPS 2015
- Images depict people in everyday scenes; each example contains one annotated person and their gaze target.

## Citation

```bibtex
@inproceedings{recasens2015they,
  title={Where are they looking?},
  author={Recasens, Adria and Khosla, Aditya and Vondrick, Carl and Torralba, Antonio},
  booktitle={NeurIPS},
  year={2015}
}
```
