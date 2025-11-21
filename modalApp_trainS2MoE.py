import modal

# === Tạo app Modal ===
app = modal.App("qwen2.5-s2moe-train-subset")
# app = modal.App("qwen2.5-s2moe-train-full")

# === Mount volume chứa dữ liệu + model ===
volume = modal.Volume.from_name("llava-data", create_if_missing=False)

# === Tạo môi trường image GPU-ready ===
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "ffmpeg")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "transformers", "datasets", "Pillow",
        "accelerate", "bitsandbytes", "sentencepiece",
        "safetensors", "qwen-vl-utils", "wandb", "huggingface-hub",
    )
    # Mount all local Python files
    .add_local_file("dataloader_qwen.py", remote_path="/root/dataloader_qwen.py", copy=True)
    .add_local_file("dataset_qwen.py", remote_path="/root/dataset_qwen.py", copy=True)
    .add_local_file("S2MOE_LORA.py", remote_path="/root/S2MOE_LORA.py", copy=True)
    .add_local_file("utils.py", remote_path="/root/utils.py", copy=True)
    .add_local_file("trainer.py", remote_path="/root/trainer.py", copy=True)
)

# === Training function ===
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 24 * 60,  # 3 days
    volumes={"/root/llava-data": volume},
    secrets=[modal.Secret.from_dotenv()],
)
def train_s2moe_qwen(subset_size=None):
    """
    Train S2MoE Qwen VL model on Modal
    
    Args:
        subset_size: Optional int, use subset of data for testing
    """
    import sys
    sys.path.insert(0, "/root")
    
    # Import training function
    from trainer import train_s2moe_model
    
    BASE = "/root/llava-data"
    MODEL_PATH = f"{BASE}/qwen_vl_7b"
    
    # Run training
    result = train_s2moe_model(
        base_dir=BASE,
        model_path=MODEL_PATH,
        volume=volume,
        subset_size=subset_size,
    )
    
    return result


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    with app.run():
        
        # 3. Train with subset (for testing)
        # train_s2moe_qwen.remote(subset_size=1000)
        
        # 4. Full training
        train_s2moe_qwen.remote()

        # Commands:
        # full data
        # modal run modalApp_trainS2MoE.py::train_s2moe_qwen

        # subset
        # modal run modalApp_trainS2MoE.py::train_s2moe_qwen --subset-size 1000