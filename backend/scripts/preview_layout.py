import cv2
import json

FRAME_PATH = "backend/data/raw/frames/m7/frame_00001.jpg"
LAYOUT_PATH = "backend/data/layouts.json"
TOURNAMENT_ID = "M7_World"

# Load layout
with open(LAYOUT_PATH) as f:
    layouts = json.load(f)

layout = layouts[TOURNAMENT_ID]
slots = layout["slots"]

# Load frame
img = cv2.imread(FRAME_PATH)
h, w = img.shape[:2]

# Scale if resolution differs
ref_w, ref_h = layout["resolution"]
scale_x, scale_y = w / ref_w, h / ref_h

# Draw each slot
for name, (x, y, sw, sh) in slots.items():
    x = int(x * scale_x)
    y = int(y * scale_y)
    sw = int(sw * scale_x)
    sh = int(sh * scale_y)

    color = (0, 200, 0) if "blue" in name else (0, 0, 200)  # green=blue team, red=red team
    cv2.rectangle(img, (x, y), (x + sw, y + sh), color, 1)
    cv2.putText(img, name, (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

cv2.imshow("Layout Preview", img)
cv2.waitKey(0)
cv2.destroyAllWindows()