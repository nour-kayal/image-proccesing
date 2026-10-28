import cv2
import os
import random
import shutil
import yaml
import numpy as np

# 1. Paths Setup
base_dir = r"C:\Users\Noor\Documents"
output_dir = os.path.join(base_dir, "dataset_augmented")

# Create train/val/test directories for images and labels
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# List original images (assuming star_render_01 to star_render_50 or similar)
original_files = [f for f in os.listdir(base_dir) if f.startswith("star_render_") and f.endswith(".png")]

print(f"Found {len(original_files)} base images. Starting augmentation pipeline...")

# Helper function to apply augmentations and update YOLO boxes
def augment_image_and_box(img, box_data):
    """
    Applies photometric changes and geometric flips/rotations, 
    and updates normalized YOLO bounding box coordinates accordingly.
    """
    class_id, x_c, y_c, w, h = box_data
    augmented_variants = []

    # --- Variant 1: Original/Resized (Base) ---
    img_resized = cv2.resize(img, (640, 640))
    augmented_variants.append((img_resized, [class_id, x_c, y_c, w, h], "base"))

    # --- Variant 2: Photometric (Brightness/Contrast & Color Shift - Box doesn't change) ---
    img_photo = cv2.resize(img, (640, 640))
    # Random brightness/contrast
    alpha = random.uniform(0.8, 1.2) # Contrast
    beta = random.randint(-20, 20)   # Brightness
    img_photo = cv2.convertScaleAbs(img_photo, alpha=alpha, beta=beta)
    augmented_variants.append((img_photo, [class_id, x_c, y_c, w, h], "photometric"))

    # --- Variant 3: Horizontal Flip (Updates x_center) ---
    img_hflip = cv2.resize(img, (640, 640))
    img_hflip = cv2.flip(img_hflip, 1) # 1 = horizontal flip
    new_x_c = 1.0 - x_c
    augmented_variants.append((img_hflip, [class_id, new_x_c, y_c, w, h], "hflip"))

    # --- Variant 4: Vertical Flip (Updates y_center) ---
    img_vflip = cv2.resize(img, (640, 640))
    img_vflip = cv2.flip(img_vflip, 0) # 0 = vertical flip
    new_y_c = 1.0 - y_c
    augmented_variants.append((img_vflip, [class_id, x_c, new_y_c, w, h], "vflip"))

    # --- Variant 5: 90 Degree Clockwise Rotation ---
    img_rot90 = cv2.resize(img, (640, 640))
    img_rot90 = cv2.rotate(img_rot90, cv2.ROTATE_90_CLOCKWISE)
    # Rotating 90 deg clockwise: x_new = 1.0 - y_old, y_new = x_old, width/height swapped
    new_x_c_rot = 1.0 - y_c
    new_y_c_rot = x_c
    new_w_rot = h
    new_h_rot = w
    augmented_variants.append((img_rot90, [class_id, new_x_c_rot, new_y_c_rot, new_w_rot, new_h_rot], "rot90"))

    return augmented_variants

# 2. Main Processing Loop across the Dataset
all_samples = []

for file_name in original_files:
    base_name = os.path.splitext(file_name)[0]
    img_path = os.path.join(base_dir, f"{base_name}.png")
    txt_path = os.path.join(base_dir, f"{base_name}.txt")

    if not os.path.exists(txt_path):
        continue

    img = cv2.imread(img_path)
    
    # Read YOLO annotation
    with open(txt_path, "r") as f:
        line = f.readline().strip()
        parts = line.split()
        box_data = [int(parts[0])] + [float(x) for x in parts[1:]]

    # Generate variants
    variants = augment_image_and_box(img, box_data)
    
    for idx, (aug_img, aug_box, tag) in enumerate(variants):
        sample_name = f"{base_name}_{tag}"
        all_samples.append((aug_img, aug_box, sample_name))

# 3. Dataset Splitting (70% Train, 20% Val, 10% Test)
# Crucial: Keep augmented variants of the same base image in the same split to prevent data leakage!
random.shuffle(original_files)
total_base = len(original_files)

train_split_idx = int(0.7 * total_base)
val_split_idx = int(0.9 * total_base)

train_base_names = set(os.path.splitext(f)[0] for f in original_files[:train_split_idx])
val_base_names = set(os.path.splitext(f)[0] for f in original_files[train_split_idx:val_split_idx])
test_base_names = set(os.path.splitext(f)[0] for f in original_files[val_split_idx:])

for aug_img, aug_box, sample_name in all_samples:
    # Determine parent base name (e.g., star_render_01 from star_render_01_hflip)
    parent_base = "_".join(sample_name.split("_")[:3])
    
    if parent_base in train_base_names:
        split = "train"
    elif parent_base in val_base_names:
        split = "val"
    else:
        split = "test"

    # Save Image
    save_img_path = os.path.join(output_dir, "images", split, f"{sample_name}.png")
    cv2.imwrite(save_img_path, aug_img)

    # Save Label TXT
    save_txt_path = os.path.join(output_dir, "labels", split, f"{sample_name}.txt")
    with open(save_txt_path, "w") as f:
        f.write(f"{aug_box[0]} {aug_box[1]:.6f} {aug_box[2]:.6f} {aug_box[3]:.6f} {aug_box[4]:.6f}\n")

print(f"Dataset successfully augmented and split into {output_dir}!")

# 4. Generate data.yaml configuration file
yaml_data = {
    'path': output_dir,
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'names': {
        0: 'star'  # Change class name if necessary
    }
}

yaml_path = os.path.join(base_dir, "data.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f"Generated data.yaml at {yaml_path}")