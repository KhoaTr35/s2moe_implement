# dataset_qwen.py
import json
from pathlib import Path
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import Dataset as Dataset

def load_llava_for_qwen(json_path, image_root):
    """
    Convert LLaVA-Instruct JSON into Qwen-VL friendly format.
    Each sample: { "image_path", "query", "response" }
    """
    json_path = Path(json_path)
    image_root = Path(image_root)

    # DEBUG: Check paths
    print(f"📂 JSON path: {json_path} (exists: {json_path.exists()})")
    print(f"📂 Image root: {image_root} (exists: {image_root.exists()})")

    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"📊 Total items in JSON: {len(data)}")

    samples = []
    missing_images = 0
    
    for item in data:
        convs = item.get("conversations", [])
        
        if not convs or len(convs) < 2:
            continue
        
        # Lấy turn đầu tiên
        first_human = None
        first_gpt = None
        
        for i, conv in enumerate(convs):
            if conv["from"] == "human" and first_human is None:
                first_human = conv["value"].replace("<image>\n", "").replace("\n<image>", "").strip()
            elif conv["from"] == "gpt" and first_gpt is None and first_human is not None:
                first_gpt = conv["value"].strip()
                break
        
        if first_human and first_gpt:
            img_path = image_root / item["image"]
            if img_path.exists():
                samples.append(
                    {
                        "image_path": str(img_path),
                        "query": f"<image>\n{first_human}",
                        "response": first_gpt,
                    }
                )
            else:
                missing_images += 1
                if missing_images <= 3:
                    print(f"❌ Missing image: {img_path}")

    print(f"⚠️  Missing images: {missing_images}")
    print(f"✅ Loaded {len(samples)} valid pairs from {json_path.name}")
    
    return Dataset.from_list(samples)

def load_llava_for_qwen(base_dir: str):
    """Chuyển dữ liệu LLaVA-Instruct JSON thành format cho Qwen-VL"""
    base = Path(base_dir)
    json_path = base / "llava_instruct_150k.json"
    image_root = base / "coco/train2017"

    print(f"JSON path: {json_path} (exists: {json_path.exists()})")
    print(f"Image root: {image_root} (exists: {image_root.exists()})")

    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"Total items in JSON: {len(data)}")

    samples, missing = [], 0
    for item in data:
        convs = item.get("conversations", [])
        if not convs or len(convs) < 2:
            continue

        first_human, first_gpt = None, None
        for c in convs:
            if c["from"] == "human" and first_human is None:
                first_human = c["value"].replace("<image>\n", "").replace("\n<image>", "").strip()
            elif c["from"] == "gpt" and first_gpt is None and first_human is not None:
                first_gpt = c["value"].strip()
                break

        if first_human and first_gpt:
            img_path = image_root / item["image"]
            if img_path.exists():
                samples.append({
                    "image_path": str(img_path),
                    "query": f"<image>\n{first_human}",
                    "response": first_gpt,
                })
            else:
                missing += 1
                if missing <= 3:
                    print(f"Missing image: {img_path}")

    print(f"Loaded {len(samples)} valid pairs, {missing} missing images.")
    return HFDataset.from_list(samples)