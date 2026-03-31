
import cv2

def save_empty_template(crop, output_path):
    """
    Call this once manually on a known empty slot crop to save it as a template.
    Run this from a script pointing at one of your empty slot crops.
    """
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, crop)
    print(f"Saved empty template to {output_path}")

def is_empty_slot(crop, template, threshold=0.75):
    """
    Returns True if the slot matches the empty placeholder template.
    Works regardless of brightness — detects the swirl/logo pattern.
    """
    crop_gray     = cv2.cvtColor(crop,     cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Resize template to match crop size just in case
    template_gray = cv2.resize(template_gray, 
                               (crop_gray.shape[1], crop_gray.shape[0]))

    result = cv2.matchTemplate(crop_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    score = result.max()
    return score >= threshold, score