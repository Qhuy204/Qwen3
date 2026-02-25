# Qwen3-VL-2B: Vietnamese Travel VQA

Fine-tune **Qwen3-VL-2B-Instruct** trên dataset [Qhuy204/VQA_VN_Destination](https://huggingface.co/datasets/Qhuy204/VQA_VN_Destination), quantize sang GGUF `q4_k_m` để inference nhẹ trên RTX 3060.

## Tổng quan

| Component | Chi tiết |
|:---|:---|
| **Base Model** | `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit` |
| **Method** | QLoRA 4-bit + LoRA (r=16) |
| **Dataset** | ~29,759 images × ≤5 QA/image |
| **Train** | A100 80GB (bf16, batch=8, eff_batch=16) |
| **Inference** | RTX 3060 12GB (GGUF q4_k_m) |

## Project Structure

```
Qwen3/
├── configs/model_config.yaml    # Hyperparameters
├── data/dataset.py              # HF dataset loader
├── training/train.py            # Fine-tuning pipeline
├── inference/predict.py         # Inference test
├── scripts/export_gguf.py       # GGUF export
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python training/train.py --config configs/model_config.yaml
```

### 3. Export GGUF (cho RTX 3060)

```bash
python scripts/export_gguf.py --config configs/model_config.yaml
```

### 4. Inference

```bash
# Dùng LoRA adapter (Unsloth)
python inference/predict.py --image path/to/image.jpg --question "Mô tả bức ảnh này"

# Dùng GGUF + Ollama (sau khi export)
ollama create qwen3vl-2b -f exported_model/Modelfile
ollama run qwen3vl-2b
```

## Config

Tất cả hyperparameters nằm trong `configs/model_config.yaml`. Các setting quan trọng:

```yaml
training:
  per_device_train_batch_size: 8   # A100 80GB
  max_steps: 500                    # Tăng cho full training
  bf16: true                        # A100 supports bf16

data:
  max_qa_per_image: 5               # Giới hạn QA/image

export:
  quantization_method: "q4_k_m"     # RTX 3060 optimized
```

## License

MIT
