"""
Dataset Loader: Custom PyTorch Dataset cho Qwen3-VL training.
Load ảnh lazy từ HF cache, trả về đúng format Unsloth messages.
Khởi tạo ngay lập tức, không cần chờ serialize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from datasets import load_dataset
from PIL import Image


class VQADataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset cho vision fine-tuning.
    
    Load metadata (text) từ JSONL, load ảnh lazy từ HF cache.
    Trả về dict {"messages": [...]} đúng format Unsloth.
    """
    
    def __init__(
        self,
        meta_file: str | Path,
        raw_dataset: Any,
        max_image_size: int = 512,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.max_image_size = max_image_size
        
        # Load metadata (chỉ text, cực nhanh)
        self.metadata: list[dict] = []
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize ảnh giữ tỷ lệ, dimensions là bội số của 28."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        w, h = img.size
        if self.max_image_size > 0 and max(w, h) > self.max_image_size:
            scale = self.max_image_size / max(w, h)
            new_w = max(28, (int(w * scale) // 28) * 28)
            new_h = max(28, (int(h * scale) // 28) * 28)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return img
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.metadata[idx]
        img_idx = item["idx"]
        qa_list = item["qa"]
        
        # Load & resize ảnh từ HF cache
        img = self._resize_image(self.raw_dataset[img_idx]["image"])
        
        # Format Unsloth messages
        messages = []
        for j, qa in enumerate(qa_list):
            if j == 0:
                # Lượt đầu tiên: đính kèm ảnh
                user_content = [
                    {"type": "image", "image": img},
                    {"type": "text", "text": qa["u"]},
                ]
            else:
                user_content = [
                    {"type": "text", "text": qa["u"]},
                ]
            
            messages.append({"role": "user", "content": user_content})
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": qa["a"]}],
            })
        
        return {"messages": messages}


def load_processed_dataset(
    config_path: str | Path,
) -> dict[str, VQADataset]:
    """Load metadata từ jsonl và trả về VQADataset (PyTorch Dataset)."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    dataset_name = data_cfg["dataset_name"]
    image_resize = data_cfg.get("image_resize", 512)

    # Kiểm tra metadata đã tồn tại chưa
    train_meta = processed_dir / "train_meta.jsonl"
    if not train_meta.exists():
        raise FileNotFoundError(
            f"❌ File '{train_meta}' không tồn tại.\n"
            f"   👉 Chạy trước: python data/prepare_data.py --config {config_path}"
        )

    # Load Original Dataset (lấy ảnh từ cache, KHÔNG download lại)
    print(f"📦 Loading image cache: {dataset_name}")
    raw_images = load_dataset(dataset_name, split="train")

    datasets: dict[str, VQADataset] = {}
    
    datasets["train"] = VQADataset(train_meta, raw_images, max_image_size=image_resize)
    print(f"   ✅ Train: {len(datasets['train'])} samples")
    
    val_meta = processed_dir / "val_meta.jsonl"
    if val_meta.exists():
        datasets["val"] = VQADataset(val_meta, raw_images, max_image_size=image_resize)
        print(f"   ✅ Val: {len(datasets['val'])} samples")

    return datasets
