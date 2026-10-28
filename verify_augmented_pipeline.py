import cv2
import os
import random
import numpy as np

# Paths to your augmented dataset
augmented_images_dir = r"C:\Users\Noor\Documents\dataset_augmented\images\train"
augmented_labels_dir = r"C:\Users\Noor\Documents\dataset_augmented\labels\train"
output_dir = r"C:\Users\Noor\Documents\dataset_augmented\part2_verification_grids"

os.makedirs(output_dir, exist_ok=True)

# Get all augmented image filenames
image_files = [f for f in os.listdir(augmented_images_dir) if f.endswith(".png")]

# Ensure we process at least 10 images (or all if fewer than 10 exist)
sample_count = min(10, len(image_files))
selected_samples = random.sample(image_files, sample_count)

print(f"Running Part 1 visualization on {sample_count} augmented images...")

for file_name in selected_samples:
    base_name = os.path.splitext(file_name)[0]
    img_path = os.path.join(augmented_images_dir, f"{base_name}.png")
    txt_path = os.path.join(augmented_labels_dir, f"{base_name}.txt")

    if not os.path.exists(txt_path):
        continue

    # 1. Load augmented image ($640\times640$)
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    display_img = img.copy()

    # 2. Run Part 1 Pipeline stages for internal verification
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.bilateralFilter(img, 9, 75, 75)
    hsv_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2HSV)

    # Thresholding & Masking (Inverted so star is white, background is black)
    lower_bound = np.array([10, 30, 100])
    upper_bound = np.array([30, 255, 255])
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    mask = cv2.bitwise_not(mask)

    # Morphological Operations
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # Find Contours & Draw Classical Detection Box (Red)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cx, cy, cw, ch = cv2.boundingRect(largest_contour)
        cv2.rectangle(display_img, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)
        cv2.putText(display_img, "Classical CV", (cx, max(20, cy - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 3. Read the Recalculated Augmented YOLO Annotation File & Draw Box (Green)
    with open(txt_path, "r") as f:
        line = f.readline().strip()
        parts = line.split()
        
        class_id = int(parts[0])
        x_center, y_center, box_w, box_h = map(float, parts[1:])

        x_min = int((x_center * img_width) - ((box_w * img_width) / 2))
        y_min = int((y_center * img_height) - ((box_h * img_height) / 2))
        x_max = int((x_center * img_width) + ((box_w * img_width) / 2))
        y_max = int((y_center * img_height) + ((box_h * img_height) / 2))

        cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(display_img, "Aug YOLO GT", (x_min, max(20, y_min - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 4. Convert single-channel stages for visual grid generation
    gray_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

    # Horizontal stack: Original -> Grayscale -> Bilateral -> Mask -> Final Box Verification
    grid_row = np.hstack((img, gray_bgr, filtered_img, mask_bgr, display_img))
    grid_resized = cv2.resize(grid_row, (0, 0), fx=0.4, fy=0.4)

    # 5. Save verification grid
    out_path = os.path.join(output_dir, f"verified_grid_{base_name}.png")
    cv2.imwrite(out_path, grid_resized)
    print(f"Generated Part 2 verification grid -> {out_path}")

print(f"Successfully verified 10 augmented images! Saved in: {output_dir}")