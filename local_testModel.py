# train.py
import torch
from torch.utils.data import DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from dataset_qwen import load_llava_for_qwen
from dataloader_qwen import QwenVLDataset

# === Đường dẫn ===
JSON_PATH = "./llava-data/llava_instruct_150k.json"
IMAGE_ROOT = "./llava-data/coco2017/train2017"

# === Bước 1: Load dataset ===
hf_dataset = load_llava_for_qwen(JSON_PATH, IMAGE_ROOT)

# === Bước 2: Khởi tạo processor cho Qwen2.5-VL ===
model_name = "Qwen/Qwen2-VL-2B-Instruct"  # hoặc "Qwen/Qwen2-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Qwen2-VL không cần set pad_token thủ công vì đã có sẵn
print(f"✅ Processor loaded for {model_name}")
print(f"   Pad token: {processor.tokenizer.pad_token}")
print(f"   Vocab size: {len(processor.tokenizer)}")

# === Bước 3: Tạo Dataset và DataLoader ===
dataset = QwenVLDataset(hf_dataset, processor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# === Bước 4: Kiểm thử batch đầu ===
print("\n🔄 Testing dataloader...")
batch = next(iter(dataloader))
print("Batch keys:", batch.keys())
for k, v in batch.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: {v.shape}")
    else:
        print(f"  {k}: {type(v)}")