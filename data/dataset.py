"""
Dataset Loader Optimized: Kết hợp Metadata với AutoProcessor.
Sử dụng AutoProcessor để đảm bảo ảnh khớp hoàn toàn với positional embeddings của model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from datasets import load_dataset, IterableDataset
from PIL import Image
from transformers import AutoProcessor


def load_processed_dataset(
    config_path: str | Path,
    split: Optional[str] = None,
) -> dict[str, Any]:
    """Load metadata và trả về IterableDataset sử dụng AutoProcessor."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    dataset_name = data_cfg["dataset_name"]
    
    # Load Processor chuẩn của model
    processor = AutoProcessor.from_pretrained(model_cfg["name"])

    # 1. Load Original Dataset (Image Cache)
    print(f"📦 Connecting to image cache and processor: {dataset_name}")
    raw_images = load_dataset(dataset_name, split="train")

    # 2. Generator function
    def _gen_fn(meta_file: Path):
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                img_idx = item["idx"]
                qa_list = item["qa"]
                
                # Để Processor tự lo việc resize và normalize
                img = raw_images[img_idx]["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Format Unsloth messages
                messages = []
                for j, qa in enumerate(qa_list):
                    user_content = [{"type": "text", "text": qa["u"]}]
                    if j == 0:
                        # Gửi nguyên object Image sang cho DataCollator xử lý
                        user_content.append({"type": "image", "image": img})
                    messages.append({"role": "user", "content": user_content})
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": qa["a"]}]})
                
                yield {"messages": messages}

    datasets = {}
    if split in [None, "train"]:
        datasets["train"] = IterableDataset.from_generator(
            _gen_fn, gen_kwargs={"meta_file": processed_dir / "train_meta.jsonl"}
        )
    if split in [None, "val"]:
        datasets["val"] = IterableDataset.from_generator(
            _gen_fn, gen_kwargs={"meta_file": processed_dir / "val_meta.jsonl"}
        )
        
    return datasets
