import modal
import shutil
from pathlib import Path

app = modal.App("upload-llava-dataset")
volume = modal.Volume.from_name("llava-data", create_if_missing=True)

# Tạo image KHÔNG chứa toàn bộ local folder (tránh limit 125k files)
image = modal.Image.debian_slim().pip_install("tqdm")

# === Mount chỉ một phần nhỏ để test ===
@app.function(
    image=image,
    volumes={"/root/llava-data": volume},
    timeout=60 * 120,  # 2 giờ
)
def upload_small_files():
    """Upload các file nhỏ trước (annotations, llava json, val2017)"""
    # Mount riêng cho từng folder nhỏ
    pass

# === Giải pháp: Chia train2017 thành nhiều batch nhỏ ===

def create_batch_folders():
    """Chia train2017 thành các batch 50k files"""
    import os
    from pathlib import Path
    
    source = Path("llava-data/coco/train2017")
    batch_dir = Path("llava-data-batches")
    batch_dir.mkdir(exist_ok=True)
    
    all_images = sorted(source.glob("*.jpg"))
    batch_size = 50000  # Dưới limit 125k
    
    print(f"📦 Total images: {len(all_images):,}")
    print(f"📊 Creating batches of {batch_size:,} images each...")
    
    for i in range(0, len(all_images), batch_size):
        batch_num = i // batch_size + 1
        batch_path = batch_dir / f"batch{batch_num}"
        batch_path.mkdir(exist_ok=True)
        
        batch_images = all_images[i:i+batch_size]
        print(f"\nBatch {batch_num}: {len(batch_images):,} images")
        
        for img in batch_images:
            # Tạo symlink thay vì copy (tiết kiệm disk)
            dst = batch_path / img.name
            if not dst.exists():
                os.link(img, dst)  # Hard link
        
        print(f"  ✅ Created {batch_path}")
    
    print(f"\n✅ Done! Upload each batch with:")
    print(f"   modal run upload_llava_dataset.py batch1")
    print(f"   modal run upload_llava_dataset.py batch2")
    print(f"   modal run upload_llava_dataset.py batch3")


# === Upload từng batch ===

@app.function(
    image=modal.Image.debian_slim()
        .add_local_dir("llava-data-batches/batch1", remote_path="/mnt/batch"),
    volumes={"/root/llava-data": volume},
    timeout=60 * 90,
)
def upload_batch1():
    """Upload batch 1 (0-50k)"""
    src = Path("/mnt/batch")
    dst = Path("/root/llava-data/coco/train2017")
    dst.mkdir(parents=True, exist_ok=True)
    
    print(f"📤 Uploading batch 1...")
    count = 0
    
    for img in src.glob("*.jpg"):
        shutil.copy2(img, dst / img.name)
        count += 1
        if count % 5000 == 0:
            print(f"  Progress: {count:,} files...")
    
    print(f"✅ Batch 1 complete: {count:,} files")
    volume.commit()


@app.function(
    image=modal.Image.debian_slim()
        .add_local_dir("llava-data-batches/batch2", remote_path="/mnt/batch"),
    volumes={"/root/llava-data": volume},
    timeout=60 * 90,
)
def upload_batch2():
    """Upload batch 2 (50k-100k)"""
    src = Path("/mnt/batch")
    dst = Path("/root/llava-data/coco/train2017")
    dst.mkdir(parents=True, exist_ok=True)
    
    print(f"📤 Uploading batch 2...")
    count = 0
    
    for img in src.glob("*.jpg"):
        shutil.copy2(img, dst / img.name)
        count += 1
        if count % 5000 == 0:
            print(f"  Progress: {count:,} files...")
    
    print(f"✅ Batch 2 complete: {count:,} files")
    volume.commit()


@app.function(
    image=modal.Image.debian_slim()
        .add_local_dir("llava-data-batches/batch3", remote_path="/mnt/batch"),
    volumes={"/root/llava-data": volume},
    timeout=60 * 90,
)
def upload_batch3():
    """Upload batch 3 (100k-118k)"""
    src = Path("/mnt/batch")
    dst = Path("/root/llava-data/coco/train2017")
    dst.mkdir(parents=True, exist_ok=True)
    
    print(f"📤 Uploading batch 3...")
    count = 0
    
    for img in src.glob("*.jpg"):
        shutil.copy2(img, dst / img.name)
        count += 1
        if count % 5000 == 0:
            print(f"  Progress: {count:,} files...")
    
    print(f"✅ Batch 3 complete: {count:,} files")
    volume.commit()


# === Upload các file khác (nhỏ hơn) ===

@app.function(
    image=modal.Image.debian_slim()
        .add_local_dir("llava-data/coco/val2017", remote_path="/mnt/val")
        .add_local_dir("llava-data/coco/annotations", remote_path="/mnt/ann")
        .add_local_file("llava-data/llava_instruct_150k.json", remote_path="/mnt/llava.json"),
    volumes={"/root/llava-data": volume},
    timeout=60 * 30,
)
def upload_other_files():
    """Upload val2017, annotations, llava json"""
    import shutil
    from pathlib import Path
    
    # Val2017
    print("📤 Uploading val2017...")
    shutil.copytree("/mnt/val", "/root/llava-data/coco/val2017", dirs_exist_ok=True)
    print(f"  ✅ val2017: {len(list(Path('/root/llava-data/coco/val2017').glob('*')))} files")
    
    # Annotations
    print("📤 Uploading annotations...")
    shutil.copytree("/mnt/ann", "/root/llava-data/coco/annotations", dirs_exist_ok=True)
    print(f"  ✅ annotations: {len(list(Path('/root/llava-data/coco/annotations').glob('*')))} files")
    
    # LLaVA JSON
    print("📤 Uploading llava_instruct_150k.json...")
    shutil.copy2("/mnt/llava.json", "/root/llava-data/llava_instruct_150k.json")
    size = Path("/root/llava-data/llava_instruct_150k.json").stat().st_size / 1e6
    print(f"  ✅ llava_instruct_150k.json: {size:.2f} MB")
    
    volume.commit()
    print("✅ All other files uploaded!")


@app.function(volumes={"/root/llava-data": volume})
def check_status():
    """Kiểm tra trạng thái upload"""
    from pathlib import Path
    
    base = Path("/root/llava-data")
    
    print("📊 Upload Status:\n")
    print("=" * 60)
    
    # Train2017
    train_dir = base / "coco/train2017"
    if train_dir.exists():
        train_count = len(list(train_dir.glob("*.jpg")))
        expected = 118287
        progress = (train_count / expected) * 100
        print(f"📸 train2017: {train_count:,}/{expected:,} ({progress:.1f}%)")
    else:
        print(f"📸 train2017: Not started")
    
    # Val2017
    val_dir = base / "coco/val2017"
    if val_dir.exists():
        val_count = len(list(val_dir.glob("*.jpg")))
        print(f"📸 val2017: {val_count:,} images")
    else:
        print(f"📸 val2017: Not uploaded")
    
    # Annotations
    ann_dir = base / "coco/annotations"
    if ann_dir.exists():
        ann_count = len(list(ann_dir.glob("*.json")))
        print(f"📄 annotations: {ann_count} files")
    else:
        print(f"📄 annotations: Not uploaded")
    
    # LLaVA
    llava_file = base / "llava_instruct_150k.json"
    if llava_file.exists():
        size = llava_file.stat().st_size / 1e6
        print(f"📄 llava_instruct_150k.json: {size:.2f} MB")
    else:
        print(f"📄 llava_instruct_150k.json: Not uploaded")
    
    print("=" * 60)


if __name__ == "__main__":
    upload_batch2.remote()
    # import sys
    
    # if len(sys.argv) > 1:
    #     cmd = sys.argv[1]
        
    #     if cmd == "prepare":
    #         # Tạo batch folders trên local
    #         create_batch_folders()
        
    #     elif cmd == "batch1":
    #         with app.run():
    #             upload_batch1.remote()
        
    #     elif cmd == "batch2":
    #         with app.run():
    #             upload_batch2.remote()
        
    #     elif cmd == "batch3":
    #         with app.run():
    #             upload_batch3.remote()
        
    #     elif cmd == "other":
    #         with app.run():
    #             upload_other_files.remote()
        
    #     elif cmd == "check":
    #         with app.run():
    #             check_status.remote()
        
    #     else:
    #         print("Usage:")
    #         print("  python upload_llava_dataset.py prepare  - Tạo batch folders")
    #         print("  modal run upload_llava_dataset.py batch1  - Upload batch 1 (0-50k)")
    #         print("  modal run upload_llava_dataset.py batch2  - Upload batch 2 (50k-100k)")
    #         print("  modal run upload_llava_dataset.py batch3  - Upload batch 3 (100k-118k)")
    #         print("  modal run upload_llava_dataset.py other   - Upload val, annotations, llava json")
    #         print("  modal run upload_llava_dataset.py check   - Check upload status")
    # else:
        # print("❌ Missing command. Run 'python upload_llava_dataset.py prepare' first")