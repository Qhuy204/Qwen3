"""
Data Preparation Super-Fast Parallel: Batched + Multi-processing + Resizing.
Tối ưu hóa tuyệt đối cho RAM 167GB và đa nhân CPU.

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
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_conversations(conversations_str: str) -> list[dict[str, str]]:
    if isinstance(conversations_str, list) or conversations_str is None:
        return conversations_str
    return json.loads(conversations_str)


def _resize_image(img: Image.Image, size: int = 512) -> Image.Image:
    """Resize ảnh về max_dimension=size, giữ nguyên tỷ lệ."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) <= size:
        return img
    
    if w > h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
        
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _process_batch_fn(batch: dict[str, list], max_qa: int = 5, image_size: int = 512) -> dict[str, list]:
    """Hàm xử lý batch song song."""
    all_messages = []
    
    for i in range(len(batch["image"])):
        raw_img = batch["image"][i]
        convs = _parse_conversations(batch["conversations"][i])
        
        if not convs: continue
        
        # Resize ảnh
        img = _resize_image(raw_img, size=image_size)
        
        current_msgs = []
        qa_counter = 0

        for j in range(0, len(convs) - 1, 2):
            user_turn = convs[j]
            assistant_turn = convs[j + 1]

            if len(current_msgs) == 0:
                user_content = [
                    {"type": "text", "text": user_turn["content"]},
                    {"type": "image", "image": img},
                ]
            else:
                user_content = [{"type": "text", "text": user_turn["content"]}]

            current_msgs.append({"role": "user", "content": user_content})
            current_msgs.append({"role": "assistant", "content": [{"type": "text", "text": assistant_turn["content"]}]})
            
            qa_counter += 1
            if qa_counter >= max_qa:
                all_messages.append(current_msgs)
                current_msgs = []
                qa_counter = 0

        if current_msgs:
            all_messages.append(current_msgs)
            
    return {"messages": all_messages}


def prepare_and_save(config_path: str | Path) -> None:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    dataset_name = data_cfg["dataset_name"]
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    max_qa_limit = data_cfg.get("max_qa_per_image", 5)
    train_ratio = data_cfg.get("train_ratio", 0.85)
    seed = data_cfg.get("seed", 42)
    image_size = data_cfg.get("image_resize", 512)

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🚀 Super-Fast Parallel Preparation")
    print(f"📦 Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name, split="train")
    
    # Dùng khoảng 8 cores để vừa nhanh vừa không nghẽn I/O
    num_cpus = min(8, os.cpu_count() or 4)
    print(f"⚙️ Using {num_cpus} CPU cores for parallel processing & resizing...")

    # ─── Step 1: Parallel Mapping ─────────────────────────────────────
    limit = 5 if max_qa_limit == 0 else max_qa_limit
    
    # .map với batched=True và num_proc là cách nhanh nhất để tạo dataset
    full_dataset = raw_dataset.map(
        _process_batch_fn,
        fn_kwargs={"max_qa": limit, "image_size": image_size},
        batched=True,
        batch_size=50, # Mỗi batch 50 ảnh để nhân CPU xử lý
        num_proc=num_cpus,
        remove_columns=raw_dataset.column_names,
        desc="Parallel Processing & Resizing"
    )

    print(f"\n✅ Total samples created: {len(full_dataset)}")

    # ─── Step 2: Shuffle & Split ──────────────────────────────────────
    print(f"🔀 Shuffling and splitting...")
    full_dataset = full_dataset.shuffle(seed=seed)
    ds_split = full_dataset.train_test_split(test_size=(1 - train_ratio), seed=seed)

    # ─── Step 3: Save ────────────────────────────────────────────────
    print(f"\n💾 Saving to disk (Multi-shard save)...")
    ds_split["train"].save_to_disk(str(processed_dir / "train"))
    ds_split["test"].save_to_disk(str(processed_dir / "val"))

    print(f"\n{'=' * 60}")
    print("✅ Xong! Dataset đã sẵn sàng với tốc độ siêu tốc.")
    print(f"   Train: {len(ds_split['train'])} samples")
    print(f"   Val:   {len(ds_split['test'])} samples")
    print(f"👉 Chạy training: python training/train.py {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_pos", type=str, nargs="?", help="Path to YAML config")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    config_path = args.config_pos if args.config_pos else args.config
    prepare_and_save(config_path)
