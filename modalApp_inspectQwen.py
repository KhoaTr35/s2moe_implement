import modal
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from pathlib import Path

app = modal.App("inspect-qwen-model")
volume = modal.Volume.from_name("llava-data", create_if_missing=False)
image = modal.Image.debian_slim().pip_install("transformers", "torch", "huggingface_hub")

@app.function(
        image=image,
        volumes={"/root/llava-data": volume},
        timeout = 60 * 60,
        gpu="A100-80GB"
        )
def inspect_model():
    model_path = Path("/root/llava-data/qwen_vl_7b")

    print(f"🔍 Loading model from: {model_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Total parameters: {total_params:,}")

    # In ra 20 layer đầu tiên
    print("\n=== Parameter names and shapes (first 20) ===")
    for i, (name, param) in enumerate(model.named_parameters()):
        print(f"{name:70s} {tuple(param.shape)}")
        if i >= 20:
            print("... truncated ...")
            break

    print(f"\nDevice: {next(model.parameters()).device}, dtype: {next(model.parameters()).dtype}")

if __name__ == "__main__":
    with app.run():
        inspect_model.remote()
