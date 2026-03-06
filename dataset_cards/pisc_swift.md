---
license: cc-by-4.0
task_categories:
  - visual-question-answering
language:
  - en
tags:
  - social-relationship-recognition
  - social-scene-understanding
  - vlm
  - ms-swift
  - qwen
  - bounding-box
size_categories:
  - 10K<n<100K
---

# PISC — People in Social Context (VLM Format)

This dataset is a reformatted version of [PISC](https://zenodo.org/record/1059155) for supervised fine-tuning of multimodal large language models (VLMs), packaged in the [MS-Swift](https://github.com/modelscope/ms-swift) Parquet format.

**Task:** Given an image and the bounding boxes of two individuals, classify their fine-grained social relationship.

## Dataset Structure

### Features

| Column | Type | Description |
|--------|------|-------------|
| `messages` | `list[{role, content}]` | System / user / assistant conversation |
| `images` | `list[Image]` | Scene image embedded as PIL bytes |

### Relationship Categories

`Friends` · `Family` · `Couple` · `Professional` · `Commercial` · `No relation`

### Prompt Format

**System:**
```
You are a vision-language model specialized in social relationship recognition.
Given an image and the bounding boxes of two individuals, classify their
fine-grained social relationship into one of:
["Friends", "Family", "Couple", "Professional", "Commercial", "No relation"].
Provide your answer as a valid JSON string.
```

**User:**
```
<image>
Two individuals are visible in this image.
Person 1 bounding box: [x1, y1, x2, y2]
Person 2 bounding box: [x1, y1, x2, y2]
What is the social relationship between Person 1 and Person 2?
```

**Assistant:**
```json
{"relationship": "Friends"}
```

Bounding box coordinates follow Qwen's convention: `[x1, y1, x2, y2]` normalized to `[0, 1000]`.

### Data Splits

| Split | Examples |
|-------|----------|
| Train | 55,400 |
| Validation | 1,505 |
| Test | 3,961 |

## Usage with MS-Swift

```python
from swift.llm.dataset import register_dataset, DatasetMeta, MessagesPreprocessor

register_dataset(
    DatasetMeta(
        dataset_name='pisc-vlm',
        hf_dataset_id='AtwMaxime/pisc_swift',
        preprocess_func=MessagesPreprocessor()
    )
)
```

```bash
swift sft \
  --model Qwen/Qwen3-VL-7B \
  --dataset pisc-vlm \
  --custom_plugin plugins/omni_dataset_plugin.py
```

## Source Dataset

- **Original dataset:** [PISC](https://zenodo.org/record/1059155) — Li et al., ICCV 2017
- Images are collected from social media and photo-sharing platforms; each image may contain multiple people and multiple annotated pairs.

## Citation

```bibtex
@inproceedings{li2017dual,
  title={Dual-glance model for deciphering social relationships},
  author={Li, Junnan and Wong, Yongkang and Zhao, Qi and Kankanhalli, Mohan S},
  booktitle={ICCV},
  year={2017}
}
```
