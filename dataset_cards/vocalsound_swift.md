---
license: cc-by-4.0
task_categories:
  - audio-classification
language:
  - en
tags:
  - vocal-sound
  - speech
  - audio
  - social-scene-understanding
  - ms-swift
  - qwen
size_categories:
  - 10K<n<100K
---

# VocalSound — Vocal Sound Classification (Omni Format)

This dataset is a reformatted version of [VocalSound](https://github.com/yuangongnd/vocalsound) for supervised fine-tuning of multimodal large language models (Omni-LLMs), packaged in the [MS-Swift](https://github.com/modelscope/ms-swift) Parquet format.

**Task:** Given an audio clip of a person producing a vocal sound, classify it into one of 6 categories.

## Dataset Structure

### Features

| Column | Type | Description |
|--------|------|-------------|
| `messages` | `list[{role, content}]` | System / user / assistant conversation |
| `audios` | `list[binary]` | Raw WAV bytes (16 kHz) |

### Sound Categories

`Laughter` · `Sigh` · `Cough` · `Throat clearing` · `Sneeze` · `Sniff`

### Prompt Format

**System:**
```
You are an expert in vocal sound recognition.
Given an audio clip of a person producing a vocal sound, classify it into one of:
["Laughter", "Sigh", "Cough", "Throat clearing", "Sneeze", "Sniff"].
Provide your answer as a valid JSON string.
```

**User:**
```
<audio>
For this given audio, identify the vocal sound and give the corresponding class
out of the following: ["Laughter", "Sigh", "Cough", "Throat clearing", "Sneeze", "Sniff"].
```

**Assistant:**
```json
{"vocal_sound": "Laughter"}
```

### Data Splits

| Split | Examples |
|-------|----------|
| Train | 15,531 |
| Validation | 1,855 |
| Test | 3,591 |

## Usage with MS-Swift

```bash
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset vocalsound-omni \
  --custom_plugin plugins/omni_dataset_plugin.py
```

## Source Dataset

- **Original dataset:** [VocalSound](https://github.com/yuangongnd/vocalsound) — Gong et al., ICASSP 2022
- Audio clips are recorded at 16 kHz, mono, from over 3,000 Zoom meeting participants.

## Citation

```bibtex
@inproceedings{gong2022vocalsound,
  title={Vocalsound: A dataset for improving human vocal sounds recognition},
  author={Gong, Yuan and Yu, Jin and Glass, James},
  booktitle={ICASSP},
  year={2022}
}
```
