# backend/scripts/save_empty_template.py
import cv2
from backend.services.liquipedia.slot_classifier import save_empty_template

# Point this at any known empty slot crop
crop = cv2.imread("backend/data/crops/m7_test/blue_ban1/frame_00001.jpg")
save_empty_template(crop, "backend/data/templates/empty_slot.jpg")

#TODO