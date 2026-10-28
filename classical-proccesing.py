import cv2
import os
import numpy as np

# Define the folder where your files are located
folder_path = r"C:\Users\Noor\Documents"

# Loop through 5 scenes (star_render_01 to star_render_05)
for i in range(1, 6):
    file_name = f"star_render_0{i}"
    img_path = os.path.join(folder_path, f"{file_name}.png")
    txt_path = os.path.join(folder_path, f"{file_name}.txt")

    if not os.path.exists(img_path) or not os.path.exists(txt_path):
        print(f"Skipping {file_name}: Files not found.")
        continue

    # 1. Load the image
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    # Keep a clean copy for final comparison drawing
    display_img = img.copy()

    # 2. Color Space Conversions & Filtering
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.bilateralFilter(img, 9, 75, 75)
    hsv_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2HSV)

    # 3. Thresholding & Masking (Inverted so star is white, background is black)
    lower_bound = np.array([10, 30, 100])
    upper_bound = np.array([30, 255, 255])
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    mask = cv2.bitwise_not(mask)

    # 4. Morphological Operations to clean noise and holes
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # 5. Find Contours & Draw Classical Detection Box (Red)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cx, cy, cw, ch = cv2.boundingRect(largest_contour)
        
        cv2.rectangle(display_img, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)
        cv2.putText(display_img, "Classical CV", (cx, cy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 6. Read YOLO Annotation File & Draw Ground-Truth Box (Green)
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
        cv2.putText(display_img, "YOLO GT", (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 7. Convert single-channel stages to 3-channel BGR for consistent stacking
    gray_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

    # 8. Concatenate in exact sequential order: Original -> Grayscale -> Bilateral -> Mask -> Comparison Grid
    grid_row = np.hstack((img, gray_bgr, filtered_img, mask_bgr, display_img))

    # Resize the grid down so it fits comfortably on your screen/viewer
    grid_resized = cv2.resize(grid_row, (0, 0), fx=0.4, fy=0.4)

    # 9. Save the side-by-side comparison grid automatically
    output_filename = os.path.join(folder_path, f"{file_name}_pipeline_grid.png")
    cv2.imwrite(output_filename, grid_resized)
    print(f"Saved comparison grid for {file_name} -> {output_filename}")

print("All 5 scenes processed and re-ordered grids saved successfully!")