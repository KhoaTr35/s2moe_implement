import modal
from pathlib import Path

# === Khai báo app và volume mount ===
app = modal.App("download-qwen-vl-model")
volume = modal.Volume.from_name("llava-data", create_if_missing=False)

# === Tạo image với dependencies ===
image = modal.Image.debian_slim().pip_install("huggingface_hub")

@app.function(
    image=image,
    volumes={"/root/llava-data": volume}, 
    timeout=60 * 60
)
def download_qwen_vl_model():
    """
    Tải model Qwen-VL-2.5-7B-Instruct từ HuggingFace và lưu vào volume llava-data.
    """
    import os
    from huggingface_hub import snapshot_download

    model_repo = "Qwen/Qwen2.5-VL-7B-Instruct"
    save_dir = Path("/root/llava-data/qwen_vl_7b")

    # Tạo thư mục nếu chưa có
    os.makedirs(save_dir, exist_ok=True)

    print(f"Bắt đầu tải model {model_repo} vào {save_dir} ...")
    snapshot_download(repo_id=model_repo, local_dir=save_dir, local_dir_use_symlinks=False)

    print("Tải model hoàn tất. Liệt kê nội dung:")
    for path in list(save_dir.glob("*"))[:10]:
        print("  •", path.name)

    print(f"\n Model đã được lưu trong volume llava-data tại: {save_dir}")

# === 3️⃣ Chạy app ===
if __name__ == "__main__":
    with app.run():
        download_qwen_vl_model.remote()