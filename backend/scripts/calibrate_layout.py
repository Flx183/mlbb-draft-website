# backend/scripts/calibrate_layout.py
import cv2
import json
from pathlib import Path

# --- CONFIG ---
FRAME_PATH = "backend/data/raw/frames/m7/frame_00001.jpg"  # replace with your clearest draft frame
OUTPUT_PATH = "backend/data/layouts.json"
TOURNAMENT_ID = "M7_World"

clicks = []
slot_names = [
    "blue_ban1", "blue_ban2", "blue_ban3", "blue_ban4", "blue_ban5",
    "red_ban1",  "red_ban2",  "red_ban3",  "red_ban4",  "red_ban5",
    "blue_pick1","blue_pick2","blue_pick3","blue_pick4","blue_pick5",
    "red_pick1", "red_pick2", "red_pick3", "red_pick4", "red_pick5",
]

img_original = None
img = None

def get_current_slot():
    slot_idx = len(clicks) // 2
    click_in_slot = len(clicks) % 2  # 0 = waiting for top-left, 1 = waiting for bottom-right
    return slot_idx, click_in_slot

def redraw():
    global img
    img = img_original.copy()
    slot_idx, click_in_slot = get_current_slot()

    # Draw all completed slots in green
    for i in range(slot_idx):
        x1, y1 = clicks[i * 2]
        x2, y2 = clicks[i * 2 + 1]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
        cv2.putText(img, slot_names[i], (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

    # Draw the first click of current slot in yellow (if placed)
    if click_in_slot == 1:
        x1, y1 = clicks[-1]
        cv2.circle(img, (x1, y1), 4, (0, 255, 255), -1)

    # Status bar at top
    if slot_idx < len(slot_names):
        current_name = slot_names[slot_idx]
        corner = "TOP-LEFT" if click_in_slot == 0 else "BOTTOM-RIGHT"
        status = f"Slot {slot_idx + 1}/{len(slot_names)}: [{current_name}] — click {corner} corner"
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, status, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    else:
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 150, 0), -1)
        cv2.putText(img, "All slots marked! Press any key to save.", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("Calibrate", img)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        slot_idx, _ = get_current_slot()
        if slot_idx >= len(slot_names):
            return
        clicks.append((x, y))
        print(f"  {slot_names[min(slot_idx, len(slot_names)-1)]}: click {len(clicks) % 2 or 2} → ({x}, {y})")
        redraw()

    # Right click to undo last click
    if event == cv2.EVENT_RBUTTONDOWN and clicks:
        clicks.pop()
        print(f"  Undid last click")
        redraw()

img_original = cv2.imread(FRAME_PATH)
if img_original is None:
    raise FileNotFoundError(f"Could not load frame: {FRAME_PATH}")

h, w = img_original.shape[:2]
print(f"Frame resolution: {w}x{h}")
print("Instructions:")
print("  LEFT CLICK  — mark top-left then bottom-right corner of each slot")
print("  RIGHT CLICK — undo last click")
print(f"  {len(slot_names)} slots to mark ({len(slot_names)*2} clicks total)\n")

img = img_original.copy()
redraw()
cv2.setMouseCallback("Calibrate", mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Process clicks into bounding boxes
slots = {}
for i, name in enumerate(slot_names):
    x1, y1 = clicks[i * 2]
    x2, y2 = clicks[i * 2 + 1]
    slots[name] = [x1, y1, x2 - x1, y2 - y1]
    print(f"  {name}: {slots[name]}")

# Save to layouts.json
layout = {
    TOURNAMENT_ID: {
        "resolution": [w, h],
        "slots": slots
    }
}

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

if Path(OUTPUT_PATH).exists():
    with open(OUTPUT_PATH) as f:
        existing = json.load(f)
    existing.update(layout)
    layout = existing

with open(OUTPUT_PATH, "w") as f:
    json.dump(layout, f, indent=2)

print(f"\nLayout saved to {OUTPUT_PATH}")