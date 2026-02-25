"""
Dataset Utilities: Load processed dataset from disk for training.

Flow:
    1. Chạy: python data/prepare_data.py   (download + process + save)
    2. Chạy: python training/train.py      (load processed data + train)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from datasets import Dataset, load_from_disk


def load_processed_dataset(
    config_path: str | Path,
    split: Optional[str] = None,
) -> dict[str, Dataset]:
    """
    Load pre-processed dataset từ disk (đã được prepare_data.py tạo ra).

    Args:
        config_path: Path to YAML config file.
        split: If provided, only load that split ("train", "val", "test").

    Returns:
        Dict with keys "train", "val", "test" → HuggingFace Datasets.

    Raises:
        FileNotFoundError: nếu chưa chạy prepare_data.py.
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["data"].get("processed_dir", "data/processed"))

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"❌ Thư mục '{processed_dir}' không tồn tại.\n"
            f"   👉 Chạy trước: python data/prepare_data.py --config {config_path}"
        )

    split_names = [split] if split else ["train", "val", "test"]
    datasets: dict[str, Dataset] = {}

    for name in split_names:
        split_path = processed_dir / name
        if not split_path.exists():
            raise FileNotFoundError(
                f"❌ Split '{name}' không tồn tại tại '{split_path}'.\n"
                f"   👉 Chạy lại: python data/prepare_data.py --config {config_path}"
            )
        datasets[name] = load_from_disk(str(split_path))
        print(f"   📂 Loaded {name}: {len(datasets[name])} samples")

    return datasets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test loading processed dataset")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()

    datasets = load_processed_dataset(args.config)
    print(f"\n✅ All splits loaded successfully!")
    for name, ds in datasets.items():
        print(f"   {name}: {len(ds)} samples")

    # Print a sample
    sample = datasets["train"][0]
    print("\n📝 Sample message structure:")
    for msg in sample["messages"]:
        role = msg["role"]
        for content in msg["content"]:
            if content["type"] == "text":
                print(f"  [{role}] {content['text'][:100]}...")
            elif content["type"] == "image":
                print(f"  [{role}] <image>")
