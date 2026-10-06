import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image

def run_audit():
    csv_path = Path("data/raw/train.csv")
    img_dir = Path("data/raw/train_images")

    print(f"=== PATH CHECKS ===")
    print(f"CSV exists: {csv_path.exists()}")
    print(f"Img dir exists: {img_dir.exists()}")

    if not csv_path.exists():
        print("ERROR: CSV does not exist!")
        return

    df = pd.read_csv(csv_path)
    print(f"\n=== CSV AUDIT ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nDuplicate image_id count: {df.duplicated(subset=['image_id']).sum()}")

    print(f"\nLabel counts:\n{df['label'].value_counts()}")
    print(f"\nTotal classes: {df['label'].nunique()}")

    print(f"\nVariety counts ({df['variety'].nunique()} unique):\n{df['variety'].value_counts()}")
    print(f"\nAge statistics:\n{df['age'].describe()}")
    print(f"Age value counts:\n{df['age'].value_counts().sort_index()}")

    print(f"\n=== IMAGE REPOSITORY AUDIT ===")
    subdirs = [d for d in img_dir.iterdir() if d.is_dir()]
    print(f"Number of subdirectories in train_images: {len(subdirs)}")
    
    total_images_found = 0
    folder_counts = {}
    invalid_extensions = []
    
    # Track all images found on disk: map image_id -> list of paths where found
    disk_images = {}
    
    for d in sorted(subdirs):
        files = list(d.glob("*"))
        folder_counts[d.name] = len(files)
        total_images_found += len(files)
        for f in files:
            ext = f.suffix.lower()
            if ext not in [".jpg", ".jpeg", ".png"]:
                invalid_extensions.append(str(f))
            disk_images.setdefault(f.name, []).append(f)

    # Check root of train_images in case flat files exist
    root_files = [f for f in img_dir.iterdir() if f.is_file()]
    print(f"Files directly in train_images root: {len(root_files)}")
    for f in root_files:
        disk_images.setdefault(f.name, []).append(f)
        total_images_found += 1

    print(f"Total image files found across all subdirectories: {total_images_found}")
    print(f"Counts per folder:\n{folder_counts}")
    print(f"Invalid extensions found: {len(invalid_extensions)}")

    print(f"\n=== IMAGE TO CSV CORRESPONDENCE AUDIT ===")
    csv_image_ids = set(df["image_id"].tolist())
    disk_image_ids = set(disk_images.keys())

    missing_on_disk = csv_image_ids - disk_image_ids
    extra_on_disk = disk_image_ids - csv_image_ids

    print(f"CSV unique image IDs: {len(csv_image_ids)}")
    print(f"Disk unique image IDs: {len(disk_image_ids)}")
    print(f"Images in CSV but missing on disk: {len(missing_on_disk)}")
    if missing_on_disk:
        print(f"Sample missing: {list(missing_on_disk)[:10]}")
    print(f"Images on disk but not in CSV: {len(extra_on_disk)}")
    if extra_on_disk:
        print(f"Sample extra: {list(extra_on_disk)[:10]}")

    # Duplicate image files across different folders
    duplicate_disk_files = {k: v for k, v in disk_images.items() if len(v) > 1}
    print(f"Image IDs appearing in multiple folders: {len(duplicate_disk_files)}")

    # Check label vs folder matching
    mismatched_folders = []
    for idx, row in df.iterrows():
        img_id = row["image_id"]
        label = row["label"]
        expected_path = img_dir / label / img_id
        if not expected_path.exists():
            # Check where it actually is
            actual_paths = disk_images.get(img_id, [])
            mismatched_folders.append((img_id, label, [str(p) for p in actual_paths]))

    print(f"Label to folder mismatches: {len(mismatched_folders)}")
    if mismatched_folders:
        print(f"Sample mismatches (first 5): {mismatched_folders[:5]}")

    print(f"\n=== IMAGE INTEGRITY / CORRUPTION CHECK ===")
    corrupted = []
    dims = set()
    sample_count = 0
    
    # Check all images for corruption
    for img_id in df["image_id"]:
        label = df.loc[df["image_id"] == img_id, "label"].values[0]
        path = img_dir / label / img_id
        if not path.exists():
            # Try alternate path from disk_images
            if img_id in disk_images and disk_images[img_id]:
                path = disk_images[img_id][0]
            else:
                continue
        try:
            with Image.open(path) as im:
                im.verify()
            # reopen to check size/mode
            with Image.open(path) as im:
                dims.add(im.size)
                if im.mode != "RGB":
                    print(f"Non-RGB image {img_id}: mode={im.mode}")
            sample_count += 1
        except Exception as e:
            corrupted.append((img_id, str(e)))

    print(f"Verified {sample_count} images.")
    print(f"Corrupted images count: {len(corrupted)}")
    if corrupted:
        print(f"Corrupted samples: {corrupted[:10]}")
    print(f"Image dimensions detected (W x H): {dims}")

if __name__ == "__main__":
    run_audit()
