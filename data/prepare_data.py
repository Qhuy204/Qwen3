"""
Data Preparation Ultra-Stable: Batched Parallel Processing.
Tối ưu cho RAM lớn và GPU khủng, tránh nghẽn I/O khi xử lý ảnh.

Usage:
    python data/prepare_data.py --config configs/model_config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_conversations(conversations_str: str) -> list[dict[str, str]]:
    if isinstance(conversations_str, list) or conversations_str is None:
        return conversations_str
    return json.loads(conversations_str)


def _process_batch_fn(batch: dict[str, list], max_qa: int = 5) -> dict[str, list]:
    """Xử lý theo cụm (Batch) để tăng tốc độ và tránh nghẽn I/O."""
    all_new_messages = []
    
    # Duyệt qua từng ảnh trong batch
    for i in range(len(batch["image"])):
        img = batch["image"][i]
        convs = _parse_conversations(batch["conversations"][i])
        
        if not convs: continue

        current_msgs = []
        qa_counter = 0

        for j in range(0, len(convs) - 1, 2):
            user_turn = convs[j]
            assistant_turn = convs[j + 1]

            # Đính kèm ảnh vào lượt user đầu tiên của mỗi sample mới
            if len(current_msgs) == 0:
                user_content = [
                    {"type": "text", "text": user_turn["content"]},
                    {"type": "image", "image": img},
                ]
            else:
                user_content = [{"type": "text", "text": user_turn["content"]}]

            current_msgs.append({"role": "user", "content": user_content})
            current_msgs.append({
                "role": "assistant", 
                "content": [{"type": "text", "text": assistant_turn["content"]}]
            })
            
            qa_counter += 1
            
            # Nếu đạt giới hạn QA hoặc hết lượt hội thoại
            if qa_counter >= max_qa:
                all_new_messages.append(current_msgs)
                current_msgs = []
                qa_counter = 0

        if current_messages:
            all_new_messages.append(current_msgs)
            
    return {"messages": all_new_messages}


def prepare_and_save(config_path: str | Path) -> None:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    dataset_name = data_cfg["dataset_name"]
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    max_qa_limit = data_cfg.get("max_qa_per_image", 0)
    train_ratio = data_cfg.get("train_ratio", 0.85)
    seed = data_cfg.get("seed", 42)

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🚀 Stable Batched Preparation Pipeline")
    print(f"📦 Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name, split="train")
    
    # Đối với ảnh, dùng 4-6 cores là "điểm ngọt" (sweet spot)
    num_cpus = min(6, os.cpu_count() or 4)
    print(f"⚙️ Using {num_cpus} CPU cores with Batched Processing...")

    # ─── Step 1: Batched Map (Cực nhanh và ổn định) ──────────────────
    limit = 5 if max_qa_limit == 0 else min(5, max_qa_limit)
    
    print(f"🔄 Processing and grouping into multi-turn...")
    processed_ds = raw_dataset.map(
        _process_batch_fn,
        fn_kwargs={"max_qa": limit},
        batched=True,
        batch_size=100,             # Xử lý 100 ảnh mỗi cụm
        num_proc=num_cpus,
        remove_columns=raw_dataset.column_names,
        desc="Batched Processing"
    )

    print(f"✅ Total samples: {len(processed_ds)}")

    # ─── Step 2: Shuffle & Split ──────────────────────────────────────
    print(f"🔀 Shuffling and splitting (train_ratio={train_ratio})...")
    processed_ds = processed_ds.shuffle(seed=seed)
    ds_split = processed_ds.train_test_split(test_size=(1 - train_ratio), seed=seed)

    # ─── Step 3: Save ────────────────────────────────────────────────
    print(f"\n💾 Saving to disk: {processed_dir}/")
    ds_split["train"].save_to_disk(str(processed_dir / "train"))
    ds_split["test"].save_to_disk(str(processed_dir / "val"))

    print(f"\n{'=' * 60}")
    print("✅ Xong! Dataset đã sẵn sàng.")
    print(f"   Train samples: {len(ds_split['train'])}")
    print(f"   Val samples:   {len(ds_split['test'])}")
    print(f"👉 Chạy training: python training/train.py {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_pos", type=str, nargs="?", help="Path to YAML config")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    config_path = args.config_pos if args.config_pos else args.config
    prepare_and_save(config_path)
