# Qwen3-VL-8B: Vietnamese Travel VQA

Fine-tune **Qwen3-VL-8B-Instruct** trên dataset [Qhuy204/VQA_VN_Destination](https://huggingface.co/datasets/Qhuy204/VQA_VN_Destination), quantize sang GGUF `q4_k_m` để inference nhẹ trên RTX 3060.

## Tổng quan

| Component | Chi tiết |
|:---|:---|
| **Base Model** | `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` |
| **Method** | QLoRA 4-bit + **Hybrid Data Strategy** (Strategy C) |
| **Dataset** | ~30k images (5 single-turn + 1 multi-turn chunk per image) |
| **Train Time** | ~7-10h (A100 80GB) / ~24-28h (L4 24GB) |
| **Recovery** | **Auto-Resume** from latest checkpoint |
| **Inference** | RTX 3060 12GB (GGUF q4_k_m ~7.2GB VRAM) |

## GGUF Export Note
Dự án bao gồm cơ chế **Fallback Export**. Nếu Unsloth không thể xuất GGUF trực tiếp (lỗi vision projector), script sẽ tự động tải và build `llama.cpp` để convert thủ công, đảm bảo luôn có file `.gguf` cuối cùng.

## Google Colab Integration
Để đảm bảo an toàn khi treo máy (tránh mất checkpoint khi session bị ngắt), dự án đã được cấu hình mặc định lưu vào Google Drive:
1. **Mount Drive** ở đầu Notebook: `from google.colab import drive; drive.mount('/content/drive')`
2. **Auto-Resume**: Script training sẽ tự động tìm checkpoint mới nhất trong `/content/drive/MyDrive/Qwen3_Backup/` để chạy tiếp.

## Project Structure

```
Qwen3/
├── configs/model_config.yaml    # Config 8B, LR 2e-4, Grad Clip, Drive Paths
├── data/prepare_data.py         # [BƯỚC 1] Hybrid sampling (Single + Multi-turn)
├── data/dataset.py              # Loader cho data đã xử lý
├── training/train.py            # [BƯỚC 2] Training pipeline with Auto-Resume
├── inference/predict.py         # Inference test
├── scripts/export_gguf.py       # GGUF export with CMake fallback
├── scripts/download_results.py  # Công cụ nén & download nhanh từ Colab
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Prepare Data (Chạy một lần)

Tải dataset từ HuggingFace, áp dụng chiến lược Hybrid và lưu metadata:

```bash
python data/prepare_data.py --config configs/model_config.yaml
```

### 3. Train (A100/L4)

Sử dụng dữ liệu đã xử lý để train model. Tự động tìm checkpoint để resume:

```bash
python training/train.py --config configs/model_config.yaml
```

### 4. Export & Download (Cho RTX 3060)

Sau khi train xong, xuất ra GGUF và tải về máy tính:

```bash
# Export
python scripts/export_gguf.py --config configs/model_config.yaml

# Download (Colab only)
python scripts/download_results.py --config configs/model_config.yaml
```

### 5. Inference

```bash
# Dùng LoRA adapter (Unsloth)
python inference/predict.py --image path/to/image.jpg --question "Bức ảnh này ở đâu?"

# Dùng GGUF + Ollama (sau khi export)
ollama create qwen3vl-8b -f /content/drive/MyDrive/Qwen3_Backup/exported_model/Modelfile
ollama run qwen3vl-8b
```

## Config

Tất cả hyperparameters quan trọng nằm trong `configs/model_config.yaml`:

```yaml
training:
  learning_rate: 2e-4          # Phù hợp cho 8B, tránh overfit
  max_grad_norm: 1.0           # Clipping để ổn định training
  resume_from_checkpoint: true # Tự động chạy tiếp từ checkpoint
  bf16: true                   # A100/L4 support
```

## License

MIT
