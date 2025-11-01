import modal

app = modal.App("build-gpu-base")

# ============================================================
# Define base image
# ============================================================
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "bash",
        "git",
        "wget",
        "curl",
        "vim",
        "build-essential",
        "tesseract-ocr",
        "libgl1",
        "libglib2.0-0",
        "python3-pip",
        "ffmpeg",
    )
    # Copy requirements.txt to image build context
    .add_local_file(
        "requirements.txt",  # file nằm cùng thư mục khoa-moe
        remote_path="/root/requirements.txt",
        copy=True,
    )
    # Upgrade pip + install dependencies
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging",
        "pip install -r /root/requirements.txt",
    )
    # Optional tools
    .run_commands("pip install jupyter notebook ipywidgets")
)

# ============================================================
# Test environment: Check GPU, CUDA, PyTorch, Transformers
# ============================================================
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=600,
)
def test_env():
    import torch, transformers
    print(" CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(" Device:", torch.cuda.get_device_name(0))
    print(" Torch version:", torch.__version__)
    print(" Transformers version:", transformers.__version__)
    print(" Environment ready for GPU fine-tuning 🚀")