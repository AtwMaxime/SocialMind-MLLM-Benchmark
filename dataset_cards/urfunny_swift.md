---
license: cc-by-4.0
task_categories:
  - video-classification
language:
  - en
tags:
  - humor-detection
  - multimodal
  - video
  - audio
  - punchline
  - social-scene-understanding
  - ms-swift
  - qwen
size_categories:
  - 1K<n<10K
---

# UR-FUNNY V2 — Multimodal Humor Detection (Unified, MS-Swift Format)

This dataset is a reformatted version of [UR-FUNNY V2](https://github.com/ROC-HCI/UR-FUNNY) for supervised fine-tuning of multimodal large language models, packaged in the [MS-Swift](https://github.com/modelscope/ms-swift) Parquet format.

**Task:** Given a video clip and audio of a speaker delivering a punchline, determine whether it is funny. Binary classification: `{"funny": 1}` or `{"funny": 0}`.

This repository contains **2 subsets** differing only in whether the preceding dialogue context is provided:

| Subset | CLI name | Description |
|---|---|---|
| `video_audio` | `urfunny-omni/video-audio` | Video + audio only |
| `video_context` | `urfunny-omni/video-context` | Video + audio + preceding context sentences |

---

## Dataset Structure

### Shared columns

| Column | Type | Description |
|---|---|---|
| `messages` | `list[{role, content}]` | System / user / assistant conversation |
| `videos` | `list[binary]` | Raw MP4 bytes (16 frames, H.264, audio track embedded) |
| `audios` | `list[binary]` | Empty (audio is embedded in the video) |

### Data Splits

Official train/dev/test split from the original dataset.

| Split | Examples | Funny | Not Funny |
|---|---|---|---|
| Train | 7,614 | 3,810 | 3,804 |
| Dev | 980 | 494 | 486 |
| Test | 994 | 490 | 504 |

The dataset is near-perfectly balanced (~50% funny, ~50% not funny).

---

## Subset Details

### `video_audio`

**System:**
```
You are an expert at detecting humor in video.
Given a video clip and audio recording of a speaker delivering a punchline,
determine whether it is funny.
Provide your answer as a valid JSON object: {"funny": 1} or {"funny": 0}.
```

**User:**
```
<video>
Is the punchline funny?
```

**Assistant:** `{"funny": 1}`

---

### `video_context`

Same as above but with the preceding dialogue context prepended to the user message.

**System:**
```
You are an expert at detecting humor in video.
Given a video clip and audio recording of a speaker delivering a punchline,
along with the preceding context sentences, determine whether it is funny.
Provide your answer as a valid JSON object: {"funny": 1} or {"funny": 0}.
```

**User:**
```
<video>
Context:
  - "the mathematics of quantum mechanics very accurately describes how our universe works"
  - "and it tells us our reality is continually branching into different possibilities just like a coral"
  - "it's a weird thing for us humans to wrap our minds around since we only ever get to experience one possibility"
Is the punchline funny?
```

**Assistant:** `{"funny": 0}`

The context contains the sentences preceding the punchline clip, providing the setup needed to judge the humor.

---

## Usage with MS-Swift

```bash
# Video + audio only
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset urfunny-omni/video-audio \
  --custom_plugin plugins/omni_dataset_plugin.py

# Video + audio + context
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset urfunny-omni/video-context \
  --custom_plugin plugins/omni_dataset_plugin.py

# Both subsets at once
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset urfunny-omni/all \
  --custom_plugin plugins/omni_dataset_plugin.py
```

## Source Dataset

- **Original dataset:** [UR-FUNNY V2](https://github.com/ROC-HCI/UR-FUNNY) — Hasan et al., EMNLP 2019
- Punchline clips from TED talks, annotated for humor by crowd-sourced workers.
- ~9,588 total clips, near-perfectly balanced between funny and not funny.

## Citation

```bibtex
@inproceedings{hasan2019ur,
  title={UR-FUNNY: A Multimodal Language Dataset for Understanding Humor},
  author={Hasan, Md Kamrul and Rahman, Wasifur and Zadeh, AmirAli Bagher and Zhong, Jianyuan and Tanveer, Md Iftekhar and Morency, Louis-Philippe and Hossain, Ehsan},
  booktitle={EMNLP},
  year={2019}
}
```
