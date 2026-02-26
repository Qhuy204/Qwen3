"""
Utility Script: Zipping and Downloading results from Google Colab.
Chỉ dành cho môi trường Google Colab.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import yaml

def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def zip_and_download(config_path: str):
    config = load_config(config_path)
    
    # ─── 1. Determine Paths ──────────────────────────────────────────
    # Lấy path từ config
    lora_path = Path(config["training"]["output_dir"]) / "final_lora"
    export_path = Path(config["export"]["output_dir"])
    
    # ─── 2. Try to Import Colab ──────────────────────────────────────
    try:
        from google.colab import files
    except ImportError:
        print("❌ Script này chỉ chạy được trong môi trường Google Colab.")
        sys.exit(1)

    print("=" * 60)
    print("📦 Colab Results Downloader")
    print("=" * 60)

    # ─── 3. Zip and Download LoRA ────────────────────────────────────
    if lora_path.exists():
        print(f"📁 Zipping LoRA adapter: {lora_path}...")
        os.system(f"zip -r final_lora.zip {lora_path}")
        print("🔥 Downloading final_lora.zip to your computer...")
        files.download("final_lora.zip")
    else:
        print(f"⚠️ LoRA path not found: {lora_path}")

    # ─── 4. Zip and Download Exported Model (GGUF) ───────────────────
    if export_path.exists():
        print(f"\n📁 Zipping Exported Models (GGUF): {export_path}...")
        os.system(f"zip -r exported_model.zip {export_path}")
        print("🔥 Downloading exported_model.zip (This may take a while)...")
        files.download("exported_model.zip")
    else:
        print(f"⚠️ Export path not found: {export_path}")

    print("\n✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    zip_and_download(args.config)
