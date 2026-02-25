# Qwen3-VL-2B: Vietnamese Travel VQA

Fine-tune **Qwen3-VL-2B-Instruct** trên dataset [Qhuy204/VQA_VN_Destination](https://huggingface.co/datasets/Qhuy204/VQA_VN_Destination), quantize sang GGUF `q4_k_m` để inference nhẹ trên RTX 3060.

## Tổng quan

| Component | Chi tiết |
|:---|:---|
| **Base Model** | `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit` |
| **Method** | QLoRA 4-bit + LoRA (r=16) |
| **Dataset** | ~29,759 images × ~39 QA/image (Total ~1.16M samples) |
| **Train** | A100 80GB (bf16, batch=8, eff_batch=16) |
| **Inference** | RTX 3060 12GB (GGUF q4_k_m) |

## Project Structure

```
Qwen3/
├── configs/model_config.yaml    # Hyperparameters
├── data/prepare_data.py         # [BƯỚC 1] Download & Process data
├── data/dataset.py              # Loader cho data đã xử lý
├── training/train.py            # [BƯỚC 2] Fine-tuning pipeline
├── inference/predict.py         # Inference test
├── scripts/export_gguf.py       # GGUF export cho RTX 3060
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Prepare Data (Chạy một lần)

Tải dataset từ HuggingFace, xử lý sang format tin nhắn và lưu vào disk:

```bash
python data/prepare_data.py --config configs/model_config.yaml
```

### 3. Train (A100)

Sử dụng dữ liệu đã xử lý để train model:

```bash
python training/train.py --config configs/model_config.yaml
```

### 4. Export GGUF (Cho RTX 3060)

Merge LoRA và xuất file GGUF `q4_k_m`:

```bash
python scripts/export_gguf.py --config configs/model_config.yaml
```

### 5. Inference

```bash
# Dùng LoRA adapter (Unsloth)
python inference/predict.py --image path/to/image.jpg --question "Mô tả bức ảnh này"

# Dùng GGUF + Ollama (sau khi export)
# Modelfile được tự động tạo trong exported_model/
ollama create qwen3vl-2b -f exported_model/Modelfile
ollama run qwen3vl-2b
```

## Config

Tất cả hyperparameters nằm trong `configs/model_config.yaml`. Các setting quan trọng:

```yaml
training:
  per_device_train_batch_size: 8   # Tối ưu cho A100 80GB
  num_train_epochs: 1               # 1 epoch cho dataset lớn (~1.16M samples)
  bf16: true                        # A100 supports bf16

data:
  max_qa_per_image: 0               # 0 = Dùng toàn bộ QA pairs (~39/ảnh)
  processed_dir: "data/processed"   # Thư mục lưu data sau xử lý

export:
  quantization_method: "q4_k_m"     # RTX 3060 optimized (4-bit)
```

## License

MIT
