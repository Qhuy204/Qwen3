"""
Data Pipeline: Load & prepare Qhuy204/VQA_VN_Destination for Qwen3-VL fine-tuning.

Input:  HuggingFace dataset with columns (id, image, conversations)
Output: HuggingFace Dataset with "messages" column (Unsloth-compatible)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import yaml
from datasets import Dataset, load_dataset


def _parse_conversations(conversations_str: str) -> list[dict[str, str]]:
    """Parse conversations JSON string → list of {role, content}."""
    if isinstance(conversations_str, list):
        return conversations_str
    return json.loads(conversations_str)


def _convert_sample_to_messages(
    sample: dict[str, Any],
    max_qa_per_image: int = 5,
) -> list[dict[str, Any]]:
    """
    Convert a single HF dataset sample → list of Unsloth message dicts.

    Each QA pair becomes one training sample with format:
    [
        {"role": "user", "content": [{"type": "text", ...}, {"type": "image", ...}]},
        {"role": "assistant", "content": [{"type": "text", ...}]}
    ]
    """
    conversations = _parse_conversations(sample["conversations"])
    image = sample["image"]

    results: list[dict[str, Any]] = []
    qa_count = 0

    for i in range(0, len(conversations) - 1, 2):
        if max_qa_per_image > 0 and qa_count >= max_qa_per_image:
            break

        user_turn = conversations[i]
        assistant_turn = conversations[i + 1]

        if user_turn.get("role") != "user" or assistant_turn.get("role") != "assistant":
            continue

        messages = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_turn["content"]},
                        {"type": "image", "image": image},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_turn["content"]},
                    ],
                },
            ]
        }
        results.append(messages)
        qa_count += 1

    return results


def load_and_prepare_dataset(
    config_path: str | Path,
    split: Optional[str] = None,
) -> dict[str, Dataset]:
    """
    Load dataset from HuggingFace Hub and prepare for Unsloth training.

    Args:
        config_path: Path to YAML config file.
        split: If provided, only return that split ("train", "val", "test").

    Returns:
        Dict with keys "train", "val", "test" → HuggingFace Datasets.
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    dataset_name: str = data_cfg["dataset_name"]
    max_qa_per_image: int = data_cfg.get("max_qa_per_image", 5)
    train_ratio: float = data_cfg.get("train_ratio", 0.85)
    val_ratio: float = data_cfg.get("val_ratio", 0.10)
    seed: int = data_cfg.get("seed", 42)

    print(f"📦 Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name, split="train")
    print(f"   → {len(raw_dataset)} images loaded")

    # Convert each image → multiple QA samples
    print(f"🔄 Converting to messages format (max {max_qa_per_image} QA/image)...")
    all_samples: list[dict[str, Any]] = []
    for sample in raw_dataset:
        qa_samples = _convert_sample_to_messages(sample, max_qa_per_image)
        all_samples.extend(qa_samples)

    print(f"   → {len(all_samples)} total QA samples")

    # Shuffle & split
    random.seed(seed)
    random.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": Dataset.from_list(all_samples[:n_train]),
        "val": Dataset.from_list(all_samples[n_train : n_train + n_val]),
        "test": Dataset.from_list(all_samples[n_train + n_val :]),
    }

    print(f"✅ Split sizes — Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    if split is not None:
        return {split: splits[split]}
    return splits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test data pipeline")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()

    datasets = load_and_prepare_dataset(args.config)
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
