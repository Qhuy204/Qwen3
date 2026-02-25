"""
Dataset Loader: Sử dụng IterableDataset để bắt đầu train NGAY LẬP TỨC.
Không cần chờ đợi tạo 1.1 triệu samples, dữ liệu sẽ được load theo kiểu "vừa train vừa load".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from datasets import load_dataset, IterableDataset
from PIL import Image


def load_processed_dataset(
    config_path: str | Path,
    split: Optional[str] = None,
) -> dict[str, Any]:
    """Load metadata và trả về IterableDataset để train ngay."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    dataset_name = data_cfg["dataset_name"]
    image_resize = data_cfg.get("image_resize", 512)

    # 1. Load Original Dataset (Image Cache)
    print(f"📦 Connecting to original image cache: {dataset_name}")
    raw_images = load_dataset(dataset_name, split="train")

    # 2. Generator function
    def _gen_fn(meta_file: Path):
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                img_idx = item["idx"]
                qa_list = item["qa"]
                # Resize ảnh: Bắt buộc phải là bội số của 28 cho Qwen-VL
                img = raw_images[img_idx]["image"]
                if image_resize > 0:
                    w, h = img.size
                    scale = image_resize / max(w, h)
                    # Làm tròn về bội số của 28 gần nhất
                    new_w = max(28, (int(w * scale) // 28) * 28)
                    new_h = max(28, (int(h * scale) // 28) * 28)
                    
                    if new_w != w or new_h != h:
                        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # Format Unsloth
                messages = []
                for j, qa in enumerate(qa_list):
                    user_content = [{"type": "text", "text": qa["u"]}]
                    if j == 0:
                        user_content.append({"type": "image", "image": img})
                    messages.append({"role": "user", "content": user_content})
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": qa["a"]}]})
                
                yield {"messages": messages}

    # 3. Trả về IterableDataset (Không tốn thời gian generate trước)
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
