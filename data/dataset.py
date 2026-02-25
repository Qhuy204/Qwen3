"""
Dataset Loader: Kết hợp Metadata với Ảnh gốc từ HF Cache.
Đây là cách tối ưu nhất để train với Vision Dataset lớn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from datasets import Dataset, load_dataset
from PIL import Image


def load_processed_dataset(
    config_path: str | Path,
    split: Optional[str] = None,
) -> dict[str, Dataset]:
    """Load metadata từ jsonl và gán ảnh từ HF dataset."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    dataset_name = data_cfg["dataset_name"]
    image_resize = data_cfg.get("image_resize", 512)

    # 1. Load Original Dataset (Lấy ảnh từ cache)
    print(f"📦 Connecting to original image cache: {dataset_name}")
    raw_images = load_dataset(dataset_name, split="train")

    # 2. Hàm gom metadata + image
    def _create_hf_dataset(meta_file: Path) -> Dataset:
        print(f"   📖 Reading {meta_file.name}...")
        meta_data = []
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                meta_data.append(json.loads(line))
        
        def gen_fn():
            for item in meta_data:
                img_idx = item["idx"]
                qa_list = item["qa"]
                
                # Resize ảnh tại đây (trên CPU của nhân train)
                img = raw_images[img_idx]["image"]
                if image_resize > 0:
                    w, h = img.size
                    if max(w, h) > image_resize:
                        scale = image_resize / max(w, h)
                        img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)

                # Format thành Unsloth messages
                messages = []
                for j, qa in enumerate(qa_list):
                    user_content = [{"type": "text", "text": qa["u"]}]
                    if j == 0: # Chỉ đính kèm ảnh vào lượt đầu
                        user_content.append({"type": "image", "image": img})
                    
                    messages.append({"role": "user", "content": user_content})
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": qa["a"]}]})
                
                yield {"messages": messages}

        return Dataset.from_generator(gen_fn)

    # 3. Load các split
    datasets = {}
    if split in [None, "train"]:
        datasets["train"] = _create_hf_dataset(processed_dir / "train_meta.jsonl")
    if split in [None, "val"]:
        datasets["val"] = _create_hf_dataset(processed_dir / "val_meta.jsonl")
        
    return datasets
