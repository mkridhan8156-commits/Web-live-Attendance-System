"""
Correct backlit/underexposed photos for better face recognition.
Run: python correct_photo.py <input_image> [output_image]
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def correct_backlit(img_path: str, output_path: str = None) -> str:
    """Enhance backlit photos: brighten shadows, improve contrast."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Convert to LAB to adjust brightness without affecting color much
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel - enhances local contrast and brightens shadows
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Slight gamma correction to brighten midtones
    gamma = 1.15
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)

    out = Path(output_path) if output_path else Path(img_path).parent / f"{Path(img_path).stem}_corrected.jpg"
    cv2.imwrite(str(out), enhanced)
    return str(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python correct_photo.py <input_image> [output_image]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    result = correct_backlit(inp, out)
    print(f"Saved corrected image: {result}")
