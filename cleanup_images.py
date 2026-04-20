# cleanup_images.py
# Deletes images from mathwriting_images/ that have no entry in labels_final.txt
# Run from TAMER repo root.

from pathlib import Path

IMAGES_DIR   = Path(r"C:\Users\samee\Documents\GitHub_Repos\TAMER\data\output_images")
LABELS_FILE  = Path(r"C:\Users\samee\Documents\GitHub_Repos\TAMER\math_writing_labels_final.txt")

# Build set of filenames that survived filtering
kept_filenames = set()
with open(LABELS_FILE, encoding="utf-8") as f:
    for line in f:
        parts = line.split("\t", maxsplit=1)
        if len(parts) < 2:
            continue
        kept_filenames.add(parts[0].strip())

# Delete any image not in the kept set
deleted, kept = 0, 0
for img_path in sorted(IMAGES_DIR.glob("*.jpg")):
    if img_path.name not in kept_filenames:
        img_path.unlink()
        deleted += 1
    else:
        kept += 1

print(f"Kept:    {kept}")
print(f"Deleted: {deleted}")