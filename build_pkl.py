# build_pkl.py
# Builds images.pkl and caption.txt from the cleaned images + labels_final.txt
# Run from TAMER repo root.

import pickle
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

IMAGES_DIR   = Path(r"C:\Users\samee\Documents\GitHub_Repos\TAMER\data\output_images")
LABELS_FILE  = Path(r"C:\Users\samee\Documents\GitHub_Repos\TAMER\math_writing_labels_final.txt")
OUTPUT_DIR   = Path(r"C:\Users\samee\Documents\GitHub_Repos\TAMER\data\mathwriting\train")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read labels
lines = LABELS_FILE.read_text(encoding="utf-8").splitlines()

images_dict = {}
caption_lines = []

for line in tqdm(lines, desc="Loading images"):
    parts = line.split("\t", maxsplit=1)
    if len(parts) < 2:
        continue
    filename, token_str = parts[0].strip(), parts[1].strip()
    img_path = IMAGES_DIR / filename

    if not img_path.exists():
        print(f"Missing image: {filename}, skipping")
        continue

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read: {filename}, skipping")
        continue

    images_dict[filename] = img
    # caption.txt format: filename tok1 tok2 ... (space separated, no tab)
    caption_lines.append(f"{filename} {token_str}")

# Write images.pkl
pkl_path = OUTPUT_DIR / "images.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(images_dict, f)
print(f"Written {len(images_dict)} images → {pkl_path}")

# Write caption.txt
caption_path = OUTPUT_DIR / "caption.txt"
caption_path.write_text("\n".join(caption_lines) + "\n", encoding="utf-8")
print(f"Written {len(caption_lines)} captions → {caption_path}")