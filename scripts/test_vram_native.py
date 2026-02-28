import subprocess
import time
import threading
import torch

def get_gpu_memory():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        return int(output.strip())
    except:
        return 0

vram_log = []
stop_event = threading.Event()

def monitor():
    while not stop_event.is_set():
        vram_log.append(get_gpu_memory())
        time.sleep(0.1)

# Khởi động monitor
initial_vram = get_gpu_memory()
monitor_thread = threading.Thread(target=monitor)
monitor_thread.start()

print(f"📊 VRAM ban đầu: {initial_vram} MiB")
print("🚀 Đang chạy inference (LoRA Native)...")

try:
    # Chạy lại script predict đã test thành công của bạn
    cmd = [
        "python", "inference/predict.py",
        "--image", "/home/qhuy/Qwen3/test_landmark.jpg",
        "--question", "Bức ảnh này ở đâu?",
        "--lora-path", "/home/qhuy/Qwen3/outputs/Qwen3-VL8B"
    ]
    subprocess.run(cmd, check=True)
except Exception as e:
    print(f"❌ Lỗi khi chạy inference: {e}")

# Dừng monitor
stop_event.set()
monitor_thread.join()

peak_vram = max(vram_log) if vram_log else initial_vram
print("\n" + "="*50)
print(f"🔥 PEAK VRAM (Native LoRA): {peak_vram} MiB (~{peak_vram/1024:.2f} GB)")
print(f"📈 Dung lượng sử dụng thêm: {peak_vram - initial_vram} MiB")
print(f"✅ Trạng thái: RTX 3060 (12GB) còn dư {(12288 - peak_vram)/1024:.2f} GB")
print("="*50)
