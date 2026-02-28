"""
Inference: Test fine-tuned Qwen3-VL-8B model on a single image + question.

Usage:
    python inference/predict.py --image path/to/image.jpg --question "Mô tả bức ảnh này"
    python inference/predict.py --image path/to/image.jpg  # Uses default question
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model(
    config_path: str = "configs/model_config.yaml",
    lora_path: Optional[str] = None,
):
    """
    Load fine-tuned Qwen3-VL model for inference.

    Args:
        config_path: Path to config YAML.
        lora_path: Path to LoRA adapter. If None, uses outputs/final_lora.

    Returns:
        Tuple of (model, processor).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]

    if lora_path is None:
        lora_path = str(Path(config["training"]["output_dir"]) / "final_lora")

    print(f"📦 Loading base model: {model_name}")
    print(f"🔧 Loading LoRA from: {lora_path}")

    from unsloth import FastVisionModel
    from transformers import AutoProcessor

    model, tokenizer = FastVisionModel.from_pretrained(
        lora_path,
        load_in_4bit=config["model"].get("load_in_4bit", True),
    )
    FastVisionModel.for_inference(model)

    # Force load processor from base model since LoRA's tokenizer might be just text
    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor

    return model, processor


def predict(
    model,
    processor,
    image_path: str,
    question: str = "Mô tả chi tiết địa điểm du lịch trong bức ảnh này.",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Run inference on a single image + question.

    Args:
        model: Fine-tuned model.
        processor: Processor (AutoProcessor).
        image_path: Path to input image.
        question: Question about the image.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Generated answer string.
    """
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    # Apply chat template
    input_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[input_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    from transformers import TextStreamer

    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"\n🖼️ Image: {image_path}")
    print(f"❓ Question: {question}")
    print(f"💬 Answer: ", end="")

    outputs = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        use_cache=True,
    )

    # Decode (without streamer for return value)
    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer


def main() -> None:
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(description="Qwen3-VL-8B Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument(
        "--question",
        type=str,
        default="Mô tả chi tiết địa điểm du lịch trong bức ảnh này.",
        help="Question about the image",
    )
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--lora-path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, processor = load_model(args.config, args.lora_path)
    answer = predict(
        model,
        processor,
        args.image,
        args.question,
        args.max_tokens,
        args.temperature,
    )

    print(f"\n\n{'=' * 40}")
    print(f"📝 Full answer:\n{answer}")


if __name__ == "__main__":
    main()
