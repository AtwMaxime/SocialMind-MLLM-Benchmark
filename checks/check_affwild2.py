import json
import os
import random
import subprocess
import tempfile

from datasets import load_dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUETS_DIR = os.path.join(ROOT_DIR, "parquets", "affwild2")

EXPR_LABELS = [
    "Neutral",
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Sadness",
    "Surprise",
    "Other",
]


def play_video(video_bytes):
    """Play video via ffplay."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        print("▶️  Playing video... [close window to continue]")
        subprocess.run(["ffplay", "-autoexit", "-loglevel", "quiet", tmp_path])
    finally:
        os.unlink(tmp_path)


def print_expr_stats(ds):
    counts = {label: 0 for label in EXPR_LABELS}
    for ex in ds:
        answer = json.loads(ex["messages"][-1]["content"])
        label = answer.get("label", "")
        if label in counts:
            counts[label] += 1
    for label, count in counts.items():
        print(f"   {label:<12} {count:>6}")


def print_va_stats(ds):
    valences = []
    arousals = []
    for ex in ds:
        answer = json.loads(ex["messages"][-1]["content"])
        valences.append(answer["valence"])
        arousals.append(answer["arousal"])
    print(
        f"   valence  min={min(valences):.3f}  max={max(valences):.3f}  mean={sum(valences)/len(valences):.3f}"
    )
    print(
        f"   arousal  min={min(arousals):.3f}  max={max(arousals):.3f}  mean={sum(arousals)/len(arousals):.3f}"
    )


def print_au_stats(ds):
    au_counts = {}
    for ex in ds:
        answer = json.loads(ex["messages"][-1]["content"])
        for au in answer.get("action_units", []):
            au_counts[au] = au_counts.get(au, 0) + 1
    none_count = sum(
        1
        for ex in ds
        if json.loads(ex["messages"][-1]["content"])["action_units"] == []
    )
    print(f"   (no AUs active): {none_count}")
    for au, count in sorted(au_counts.items()):
        print(f"   {au:<6} {count:>6}")


for task in ["expr", "va", "au"]:
    for split in ["train", "val"]:
        parquet_file = os.path.join(
            PARQUETS_DIR, f"affwild2_{task}_{split}.parquet"
        )

        print(f"\n{'='*60}")
        print(f"📂 {task} / {split}")

        if not os.path.exists(parquet_file):
            print(f"❌ Not found: {parquet_file}")
            continue

        ds = load_dataset(
            "parquet", data_files={"train": parquet_file}, split="train"
        )
        print(f"✅ {len(ds)} examples")

        if task == "expr":
            print_expr_stats(ds)
        elif task == "va":
            print_va_stats(ds)
        else:
            print_au_stats(ds)

        idx = random.randint(0, len(ds) - 1)
        example = ds[idx]
        msgs = {m["role"]: m["content"] for m in example["messages"]}

        print(f"\n🎲 Sample #{idx}")
        print(f"\n--- 📝 User ---\n{msgs['user']}")
        print(f"\n--- 💬 Assistant ---\n{msgs['assistant']}\n")

        play_video(example["videos"][0])
