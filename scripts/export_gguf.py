"""
Export Pipeline: Merge LoRA adapter + Export to GGUF (q4_k_m).

Optimized for inference on RTX 3060 12GB.

Usage:
    python scripts/export_gguf.py --config configs/model_config.yaml
    python scripts/export_gguf.py --lora-path outputs/final_lora --quantization q4_k_m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(
    config_path: str = "configs/model_config.yaml",
    lora_path: str | None = None,
    quantization: str | None = None,
) -> None:
    """
    Merge LoRA adapter into base model and export as GGUF.

    Steps:
        1. Load base model + LoRA adapter
        2. Save merged model to 16-bit
        3. Export GGUF with specified quantization (default: q4_k_m)
    """
    config = load_config(config_path)
    export_cfg = config.get("export", {})
    model_cfg = config["model"]

    if lora_path is None:
        lora_path = str(Path(config["training"]["output_dir"]) / "final_lora")
    if quantization is None:
        quantization = export_cfg.get("quantization_method", "q4_k_m")

    output_dir = export_cfg.get("output_dir", "exported_model")
    merge_16bit_dir = export_cfg.get("merge_16bit_dir", "merged_model_16bit")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(merge_16bit_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📦 Qwen3-VL-2B → GGUF Export Pipeline")
    print("=" * 60)
    print(f"   Base model:    {model_cfg['name']}")
    print(f"   LoRA adapter:  {lora_path}")
    print(f"   Quantization:  {quantization}")
    print(f"   Output dir:    {output_dir}")

    # ─── Step 1: Load model + LoRA ────────────────────────────────────
    print("\n🔧 Step 1: Loading model + LoRA adapter...")

    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        lora_path,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )

    # ─── Step 2: Save merged 16-bit ──────────────────────────────────
    print(f"\n💾 Step 2: Saving merged model to 16-bit → {merge_16bit_dir}")
    model.save_pretrained_merged(
        merge_16bit_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    print("   ✅ 16-bit merge complete!")

    # ─── Step 3: Export GGUF ──────────────────────────────────────────
    print(f"\n📤 Step 3: Exporting GGUF ({quantization}) → {output_dir}")
    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method=quantization,
    )

    print("\n" + "=" * 60)
    print("✅ Export complete!")
    print(f"   GGUF file location: {output_dir}/")
    print(f"   Quantization: {quantization}")
    print(f"\n   🚀 RTX 3060 ready! Use with Ollama or llama.cpp:")
    print(f"      ollama create qwen3vl-2b -f {output_dir}/Modelfile")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-VL-2B to GGUF")
    parser.add_argument("config_pos", type=str, nargs="?", help="Path to YAML config (positional)")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to YAML config (optional flag)")
    parser.add_argument("--lora-path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16", "q4_0", "q4_k_s"],
        help="GGUF quantization method",
    )
    args = parser.parse_args()
    
    config_path = args.config_pos if args.config_pos else args.config
    main(config_path, args.lora_path, args.quantization)
