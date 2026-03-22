import cv2
import json
import os
from pathlib import Path

FRAMES_DIR = "backend/data/raw/frames/m7"
LAYOUT_PATH = "backend/data/layouts.json"
TOURNAMENT_ID = "M7_World"
OUTPUT_DIR = "backend/data/crops/m7_test"

# Load layout
with open(LAYOUT_PATH) as f:
    layout = json.load(f)[TOURNAMENT_ID]

slots = layout["slots"]
ref_w, ref_h = layout["resolution"]

# Process every frame
frame_paths = sorted(Path(FRAMES_DIR).glob("*.jpg"))
print(f"Processing {len(frame_paths)} frames...")

for frame_path in frame_paths:
    img = cv2.imread(str(frame_path))
    if img is None:
        continue

    h, w = img.shape[:2]
    scale_x, scale_y = w / ref_w, h / ref_h

    for slot_name, (x, y, sw, sh) in slots.items():
        # Scale to actual resolution
        x_scaled  = int(x  * scale_x)
        y_scaled  = int(y  * scale_y)
        sw_scaled = int(sw * scale_x)
        sh_scaled = int(sh * scale_y)

        # Crop the slot
        crop = img[y_scaled:y_scaled+sh_scaled, x_scaled:x_scaled+sw_scaled]

        # Resize to standard size for classifier
        crop = cv2.resize(crop, (64, 64))

        # Save as frames_dir/slot_name/frame_00001.jpg
        slot_output_dir = Path(OUTPUT_DIR) / slot_name
        slot_output_dir.mkdir(parents=True, exist_ok=True)
        out_path = slot_output_dir / frame_path.name
        cv2.imwrite(str(out_path), crop)

print(f"Done! Crops saved to {OUTPUT_DIR}")