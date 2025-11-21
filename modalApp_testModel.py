# train_modal_infer.py
import modal

# === 1️⃣ Tạo app Modal ===
app = modal.App("test-qwen-infer")

# === 2️⃣ Mount volume chứa dữ liệu + model ===
volume = modal.Volume.from_name("llava-data", create_if_missing=False)

# === 3️⃣ Tạo môi trường image GPU-ready ===
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "ffmpeg")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "transformers", "datasets", "Pillow",
        "accelerate", "bitsandbytes", "sentencepiece",
        "safetensors", "qwen-vl-utils"
    )
    .add_local_file("dataloader_qwen.py", remote_path="/root/dataloader_qwen.py", copy=True)
    .add_local_file("dataset_qwen.py", remote_path="/root/dataset_qwen.py", copy=True)
    .add_local_file("S2MOE_LORA.py", remote_path="/root/S2MOE_LORA.py", copy=True)
    .add_local_file("utils.py", remote_path="/root/utils.py", copy=True)

)

# === 4️⃣ Các import dùng trong function ===
from torch.utils.data import Dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
import json
from pathlib import Path
from datasets import Dataset as HFDataset



# ----------------------------
# Dataset class
# ----------------------------
from dataloader_qwen import QwenVLDataset

# ----------------------------
# Data loading function
# ----------------------------
from dataset_qwen import load_llava_for_qwen

# Load lora classes
from S2MOE_LORA import LoRALayer, LoRA_MOE_LM, S2MoE_LoRA_MLP

from utils import replace_mlp


# ----------------------------
# Inference function chạy trên GPU
# ----------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 24 * 60,
    volumes={"/root/llava-data": volume},
)

def run_test():
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, AutoModelForVision2Seq
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    BASE = "/root/llava-data"
    MODEL_PATH = f"{BASE}/qwen_vl_7b"  # checkpoint đã lưu sẵn

    print(f"📂 Using model from: {MODEL_PATH}")

    # === Load dataset ===
    hf_dataset = load_llava_for_qwen(BASE)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dataset = QwenVLDataset(hf_dataset, processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # === Load model (từ volume) ===
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("✅ Model loaded successfully.")

    #=== Inference 1 batch ===
    batch = next(iter(dataloader))
    batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
    print(f"🔍 Batch keys: {list(batch.keys())}")

    with torch.inference_mode():
        output_ids = model.generate(**batch, max_new_tokens=64)
        generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    print("\n🧠 Inference output:")
    print(generated_text[:500])

    return generated_text

    # # ==== Check model parameters ====
    # print("\n=== Parameter names and shapes ===")
    # count = 0
    # for name, param in model.named_parameters():
    #     print(f"{name:70s} {tuple(param.shape)}")
    #     count += 1
    # print(f"\nTotal parameter tensors: {count}")

    # # Total parameter count
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters (elements): {total_params:,}")

    # for n, p in model.named_parameters():
    #     p.requires_grad = False

    # print("✅ All parameters frozen.")

    # replace_mlp(model, is_s2moe=True)
    
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 24 * 60,
    volumes={"/root/llava-data": volume},
)
def test_forward_backward():
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, AutoModelForVision2Seq
    import torch.nn as nn
    import torch.nn.functional as F

    BASE = "/root/llava-data"
    MODEL_PATH = f"{BASE}/qwen_vl_7b"

    print(f"📂 Using model from: {MODEL_PATH}")

    # === Load dataset ===
    print("\n🔄 Loading dataset...")
    hf_dataset = load_llava_for_qwen(BASE)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dataset = QwenVLDataset(hf_dataset, processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"✅ Dataset loaded: {len(dataset)} samples")

    # === Load model ===
    print("\n🔄 Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # === Freeze all parameters ===
    for p in model.parameters():
        p.requires_grad = False

    # === Replace MLPs with S2MoE_LoRA_MLP ===
    print("\n🔄 Replacing MLPs with S2MoE_LoRA_MLP...")
    replace_mlp(model, is_s2moe=True)

    # ✅ FIX: Move model về GPU sau khi replace
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔄 Moving model to {device}...")
    model = model.to(device)
    
    # Hoặc dùng device_map nếu muốn tự động phân bổ:
    # from accelerate import dispatch_model, infer_auto_device_map
    # device_map = infer_auto_device_map(model, max_memory={0: "70GiB"})
    # model = dispatch_model(model, device_map=device_map)

    # === Verify trainable parameters ===
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    # === Verify all parameters are on GPU ===
    print("\n🔍 Verifying device placement...")
    devices = set()
    for name, param in model.named_parameters():
        devices.add(str(param.device))
        if param.requires_grad and 'cpu' in str(param.device):
            print(f"⚠️  WARNING: Trainable param on CPU: {name}")
    
    print(f"✅ Model devices: {devices}")

    # === Get one batch ===
    print("\n🔄 Getting one batch from dataloader...")
    batch = next(iter(dataloader))
    
    # Move batch to same device as model
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    
    print(f"✅ Batch loaded. Keys: {list(batch.keys())}")
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape} on {v.device}")

    # === Test Forward Pass ===
    print("\n" + "="*60)
    print("🧪 TESTING FORWARD PASS")
    print("="*60)
    
    model.train()
    
    try:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
        
        print(f"✅ Forward pass successful!")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Loss shape: {loss.shape}")
        print(f"   Loss dtype: {loss.dtype}")
        
        # Collect auxiliary losses from S2MoE_LoRA_MLP
        aux_losses = {}
        for name, module in model.named_modules():
            if isinstance(module, S2MoE_LoRA_MLP):
                if hasattr(module, '_aux_losses'):
                    for loss_name, loss_val in module._aux_losses.items():
                        if loss_name not in aux_losses:
                            aux_losses[loss_name] = []
                        aux_losses[loss_name].append(loss_val.item())
        
        if aux_losses:
            print("\n📊 Auxiliary losses from S2MoE layers:")
            for loss_name, vals in aux_losses.items():
                avg_val = sum(vals) / len(vals)
                print(f"   {loss_name}: {avg_val:.6f} (avg over {len(vals)} layers)")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # === Test Backward Pass ===
    print("\n" + "="*60)
    print("🧪 TESTING BACKWARD PASS")
    print("="*60)
    
    try:
        # Compute total loss (main + auxiliary)
        total_loss = loss
        
        if aux_losses:
            # Get alpha_bal and beta_unc from first S2MoE layer
            for module in model.modules():
                if isinstance(module, S2MoE_LoRA_MLP):
                    alpha_bal = module.alpha_bal
                    beta_unc = module.beta_unc
                    break
            
            # Add weighted auxiliary losses
            for name, module in model.named_modules():
                if isinstance(module, S2MoE_LoRA_MLP) and hasattr(module, '_aux_losses'):
                    if 'Lb' in module._aux_losses:
                        total_loss = total_loss + alpha_bal * module._aux_losses['Lb']
                    if 'Lu' in module._aux_losses:
                        total_loss = total_loss + beta_unc * module._aux_losses['Lu']
        
        print(f"📊 Total loss: {total_loss.item():.4f}")
        
        # Backward pass
        total_loss.backward()
        
        print("✅ Backward pass successful!")
        
        # Check gradients
        print("\n📊 Gradient statistics:")
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_type = "S2MoE" if "moe_" in name or "router" in name or "merge_gate" in name else "other"
                
                if param_type not in grad_norms:
                    grad_norms[param_type] = []
                grad_norms[param_type].append(grad_norm)
        
        for param_type, norms in grad_norms.items():
            avg_norm = sum(norms) / len(norms)
            max_norm = max(norms)
            min_norm = min(norms)
            print(f"   {param_type:10s}: avg={avg_norm:.6f}, max={max_norm:.6f}, min={min_norm:.6f}, count={len(norms)}")
        
        # Print sample gradients
        print("\n📋 Sample gradients (first 5 trainable params):")
        count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"   {name:60s} grad_norm={param.grad.norm().item():.6f}")
                count += 1
                if count >= 5:
                    break
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    
    return {
        "loss": loss.item(),
        "total_loss": total_loss.item(),
        "aux_losses": {k: sum(v)/len(v) for k, v in aux_losses.items()},
        "trainable_params": trainable_params,
        "total_params": total_params,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 24 * 60,
    volumes={"/root/llava-data": volume},
)
def verify_dataset():
    """Verify dataset is loading images correctly"""
    from transformers import AutoProcessor
    from pathlib import Path
    
    BASE = "/root/llava-data"
    MODEL_PATH = f"{BASE}/qwen_vl_7b"
    
    # Load dataset
    hf_dataset = load_llava_for_qwen(BASE)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dataset = QwenVLDataset(hf_dataset, processor)
    
    print(f"\n📊 Dataset size: {len(dataset)}")
    
    # Test load first 3 samples
    for i in range(min(3, len(dataset))):
        try:
            sample = dataset[i]
            print(f"\n✅ Sample {i}:")
            print(f"   image_path: {hf_dataset[i]['image_path']}")
            print(f"   image exists: {Path(hf_dataset[i]['image_path']).exists()}")
            print(f"   pixel_values shape: {sample['pixel_values'].shape}")
            print(f"   input_ids shape: {sample['input_ids'].shape}")
        except Exception as e:
            print(f"\n❌ Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    return "Dataset verification complete"


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 24 * 60,
    volumes={"/root/llava-data": volume},
    secrets=[modal.Secret.from_dotenv()],   # để load HF_TOKEN nếu cần
)
def hf_infer_test(
    image_path="coco/train2017/000000149669.jpg",
    prompt="Describe the image in detail."
):
    """
    LOAD base model or fine-tuned model from HF Hub
    and run inference giống như training pipeline.
    Không dùng chat template sai.
    Không dùng <|vision_start|> token.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq

    model_id = "KhTran35/s2moe-qwen-finetuned1-20251121-143845"

    print(f"🔍 Loading model from HF Hub: {model_id}")

    # === Load processor ===
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # === Load model ===
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("✅ Model loaded successfully.")

    # === Load image ===
    img_full_path = f"/root/llava-data/{image_path}"
    print(f"🖼️ Loading image: {img_full_path}")
    image = Image.open(img_full_path).convert("RGB")

    # === Encode image + prompt (training-style encoding) ===
    print("🔄 Encoding image + text using processor...")
    inputs = processor(
        images=image,
        text=prompt,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    print(f"🎨 pixel_values shape = {inputs['pixel_values'].shape}")  
    # Expect: [1, 3, H, W]

    # === Generate ===
    print("🧠 Running inference...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

    # === Decode ===
    output_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    print("\n====================")
    print("🧠 MODEL OUTPUT")
    print("====================")
    print(output_text)
    print("====================\n")

    return output_text

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    with app.run():
        hf_infer_test.remote(
            image_path="coco/train2017/000000149669.jpg",
            prompt="Describe the image in detail."
        )
        #verify_dataset.remote()
        # run_test.remote()
        #test_forward_backward.remote()
