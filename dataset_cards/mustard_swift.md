---
license: cc-by-4.0
task_categories:
  - video-classification
language:
  - en
tags:
  - sarcasm-detection
  - multimodal
  - video
  - audio
  - dialogue
  - social-scene-understanding
  - ms-swift
  - qwen
size_categories:
  - n<1K
---

# MUStARD — Multimodal Sarcasm Detection (Unified, MS-Swift Format)

This dataset is a reformatted version of [MUStARD](https://github.com/soujanyaporia/MUStARD) for supervised fine-tuning of multimodal large language models, packaged in the [MS-Swift](https://github.com/modelscope/ms-swift) Parquet format.

**Task:** Given a video clip with audio of a speaker, determine whether their utterance is sarcastic. Binary classification: `{"sarcasm": true}` or `{"sarcasm": false}`.

This repository contains **2 subsets** differing only in whether the preceding dialogue context is provided:

| Subset | CLI name | Description |
|---|---|---|
| `video_no_context` | `mustard-omni/video-no-context` | Video + audio only |
| `video_context` | `mustard-omni/video-context` | Video + audio + preceding dialogue context |

---

## Dataset Structure

### Shared columns

| Column | Type | Description |
|---|---|---|
| `messages` | `list[{role, content}]` | System / user / assistant conversation |
| `videos` | `list[binary]` | Raw MP4 bytes (16 frames, H.264, variable fps, audio track embedded) |
| `audios` | `list[binary]` | Empty (audio is embedded in the video) |

### Data Splits

No official split — 80/20 random split with seed 42.

| Split | Examples |
|---|---|
| Train | 552 |
| Test | 138 |

The dataset is perfectly balanced: 50% sarcastic, 50% not sarcastic.

---

## Subset Details

### `video_no_context`

**System:**
```
You are an expert at detecting sarcasm in multimodal communication.
Given a video clip and audio recording of a speaker, determine whether
their utterance is sarcastic.
Provide your answer as a valid JSON object: {"sarcasm": true} or {"sarcasm": false}.
```

**User:**
```
<video>
Is the speaker being sarcastic?
```

**Assistant:** `{"sarcasm": true}`

---

### `video_context`

Same as above but with the preceding dialogue turns prepended to the user message.

**System:**
```
You are an expert at detecting sarcasm in multimodal communication.
Given a video clip and audio recording of a speaker, along with the preceding
dialogue context, determine whether their utterance is sarcastic.
Provide your answer as a valid JSON object: {"sarcasm": true} or {"sarcasm": false}.
```

**User:**
```
<video>
Dialogue context:
  - LEONARD: "I never would have identified the fingerprints of string theory in the Big Bang."
  - SHELDON: "My apologies. What's your plan?"
Is the speaker being sarcastic?
```

**Assistant:** `{"sarcasm": false}`

The dialogue context contains 1–11 preceding utterances with their speaker labels.

---

## Usage with MS-Swift

```bash
# Video + audio only
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset mustard-omni/video-no-context \
  --custom_plugin plugins/omni_dataset_plugin.py

# Video + audio + dialogue context
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset mustard-omni/video-context \
  --custom_plugin plugins/omni_dataset_plugin.py

# Both subsets at once
swift sft \
  --model Qwen/Qwen3-Omni-7B \
  --dataset mustard-omni/all \
  --custom_plugin plugins/omni_dataset_plugin.py
```

## Source Dataset

- **Original dataset:** [MUStARD](https://github.com/soujanyaporia/MUStARD) — Castro et al., ACL 2019
- Clips sourced from TV shows: *The Big Bang Theory*, *Friends*, *The Golden Girls*, and *Sarcasmaholics Anonymous*.
- 690 total clips, balanced between sarcastic and non-sarcastic utterances.

## Citation

```bibtex
@inproceedings{castro2019towards,
  title={Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)},
  author={Castro, Santiago and Hazarika, Devamanyu and P{\'e}rez-Rosas, Ver{\'o}nica and Zimmermann, Roger and Mihalcea, Rada and Poria, Soujanya},
  booktitle={ACL},
  year={2019}
}
```
