---
license: cc-by-4.0
task_categories:
  - video-classification
language:
  - en
tags:
  - deception-detection
  - multimodal
  - video
  - audio
  - social-scene-understanding
  - ms-swift
  - qwen
size_categories:
  - n<1K
---

# Real-Life Deception Detection — Video + Audio Variant (Omni Format)

This dataset is a reformatted version of the [Real-Life Deception Detection 2016](http://web.eecs.umich.edu/~mihalcea/downloads.html#RealLifeDeception) dataset for supervised fine-tuning of multimodal large language models (Omni-LLMs), packaged in the [MS-Swift](https://github.com/modelscope/ms-swift) Parquet format.

**Task:** Given a video clip with audio of a person making a statement, determine whether they are being deceptive or truthful. Binary classification.

## Dataset Structure

### Features

| Column | Type | Description |
|--------|------|-------------|
| `messages` | `list[{role, content}]` | System / user / assistant conversation |
| `videos` | `list[binary]` | Raw MP4 bytes (16 frames, H.264, variable fps, with audio track) |
| `audios` | `list[binary]` | Empty (audio is embedded in the video) |

### Prompt Format

**System:**
```
You are an expert at detecting deception in video.
Given a video clip and audio recording of a person, determine whether
they are being deceptive or truthful.
Provide your answer as a valid JSON object: {"label": "deceptive"} or {"label": "truthful"}.
```

**User:**
```
<video>
Is this person being deceptive or truthful?
```

**Assistant:**
```json
{"label": "deceptive"}
```

- Videos contain exactly 16 frames sampled evenly across the full clip duration (H.264, variable fps = 16 / duration), audio track kept (downmixed to stereo).
- The dataset is perfectly balanced: 61 deceptive clips, 60 truthful clips (121 total).
- Clips are sourced from real courtroom testimonies and filmed depositions.

### Data Splits

No official split — 80/20 random split with seed 42.

| Split | Examples |
|-------|----------|
| Train | 96 |
| Test  | 25 |

## Usage with MS-Swift

```bash
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset rldd-video \
  --custom_plugin plugins/omni_dataset_plugin.py
```

## Source Dataset

- **Original dataset:** [Real-Life Deception Detection 2016](http://web.eecs.umich.edu/~mihalcea/downloads.html#RealLifeDeception) — Pérez-Rosas et al., 2015
- 121 video clips of real-life high-stakes situations (courtroom testimonies, filmed depositions).
- Balanced between deceptive and truthful statements.

## Citation

```bibtex
@inproceedings{perez2015deception,
  title={Deception Detection using Real-life Trial Data},
  author={P{\'e}rez-Rosas, Ver{\'o}nica and Abouelenien, Mohamed and Mihalcea, Rada and Burzo, Mihai},
  booktitle={ICMI},
  year={2015}
}
```
