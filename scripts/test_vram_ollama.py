import requests
import base64
import json
import subprocess
import time
import threading

def get_gpu_memory():
    """Lấy lượng VRAM đang sử dụng hiện tại (MiB)."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        return int(output.strip())
    except Exception:
        return 0

def monitor_vram(vram_log, stop_event):
    """Tiến trình chạy ngầm để ghi lại Peak VRAM."""
    while not stop_event.is_set():
        vram_log.append(get_gpu_memory())
        time.sleep(0.1)

def test_vision_vram(model_name, image_path, prompt):
    # Chuyển ảnh sang base64
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [img_base64],
        "stream": False
    }

    print(f"🚀 Gửi yêu cầu tới Ollama (Model: {model_name})...")
    
    vram_log = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_vram, args=(vram_log, stop_event))
    
    initial_vram = get_gpu_memory()
    monitor_thread.start()
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        end_time = time.time()
        
        stop_event.set()
        monitor_thread.join()
        
        if response.status_code == 200:
            result = response.json()
            peak_vram = max(vram_log) if vram_log else initial_vram
            
            print("\n" + "="*50)
            print(f"💬 Câu trả lời: {result.get('response', '')}")
            print(f"⏱️ Thời gian xử lý: {end_time - start_time:.2f}s")
            print(f"📊 VRAM ban đầu: {initial_vram} MiB")
            print(f"🔥 PEAK VRAM: {peak_vram} MiB (~{peak_vram/1024:.2f} GB)")
            print("="*50)
        else:
            print(f"❌ Lỗi API Ollama: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
        stop_event.set()

if __name__ == "__main__":
    test_vision_vram(
        model_name="qwen3vl-8b", 
        image_path="/home/qhuy/Qwen3/test_landmark.jpg", 
        prompt="Bức ảnh này ở đâu?"
    )
