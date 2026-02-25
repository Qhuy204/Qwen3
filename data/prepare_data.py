"""
Data Preparation Metadata-Only: Xử lý siêu tốc (Dưới 1 phút).
Chỉ lưu văn bản và chỉ số ảnh, không lưu/nén ảnh để tránh nghẽn I/O.

Usage:
    python data/prepare_data.py --config configs/model_config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_conversations(conversations_str: str) -> list[dict[str, str]]:
    if isinstance(conversations_str, list) or conversations_str is None:
        return conversations_str
    return json.loads(conversations_str)


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

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🚀 Metadata-Only Preparation (Lightning Speed)")
    print(f"📦 Loading dataset structure: {dataset_name}")
    raw_dataset = load_dataset(dataset_name, split="train")
    
    print(f"\n🔄 Extracting text metadata (Image processing skipped for speed)...")
    
    all_metadata = []
    limit = 5 if max_qa_limit == 0 else max_qa_limit

    for idx, item in enumerate(raw_dataset):
        if idx % 5000 == 0:
            print(f"   Progress: {idx}/{len(raw_dataset)} images...")
            
        convs = _parse_conversations(item["conversations"])
        if not convs: continue
        
        current_qa = []
        qa_counter = 0

        for j in range(0, len(convs) - 1, 2):
            user_text = convs[j]["content"]
            assistant_text = convs[j+1]["content"]

            current_qa.append({"u": user_text, "a": assistant_text})
            qa_counter += 1
            
            if qa_counter >= limit:
                all_metadata.append({"idx": idx, "qa": current_qa})
                current_qa = []
                qa_counter = 0

        if current_qa:
            all_metadata.append({"idx": idx, "qa": current_qa})

    # ─── Step 2: Shuffle & Split ──────────────────────────────────────
    print(f"\n🔀 Shuffling and splitting ({len(all_metadata)} samples)...")
    random.seed(seed)
    random.shuffle(all_metadata)

    n_train = int(len(all_metadata) * train_ratio)
    train_meta = all_metadata[:n_train]
    val_meta = all_metadata[n_train:]

    # ─── Step 3: Save Metadata (JSONL) ────────────────────────────────
    print(f"💾 Saving metadata to {processed_dir}/*.jsonl")
    
    with open(processed_dir / "train_meta.jsonl", "w", encoding="utf-8") as f:
        for entry in train_meta:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open(processed_dir / "val_meta.jsonl", "w", encoding="utf-8") as f:
        for entry in val_meta:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print("✅ Xong! Quá trình chuẩn bị metadata chỉ mất vài chục giây.")
    print("👉 Bây giờ chạy: python training/train.py configs/model_config.yaml")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_pos", type=str, nargs="?", help="Path to YAML config")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    config_path = args.config_pos if args.config_pos else args.config
    prepare_and_save(config_path)
