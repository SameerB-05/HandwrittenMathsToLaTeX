import os
from pathlib import Path

OUTPUT_ROOT = Path(r"C:\Users\samee\Documents\GitHub_Repos\TAMER\data\output_images_test")

def rename_images():
    renamed = 0
    errors = 0

    for i in range(1, 7645):
        old_name = OUTPUT_ROOT / f"train_{str(i).zfill(2)}.jpg"

        # Handle both zero-padded and non-zero-padded names
        if not old_name.exists():
            old_name = OUTPUT_ROOT / f"train_{i}.jpg"

        new_name = OUTPUT_ROOT / f"test_{i}.jpg"

        if old_name.exists():
            os.rename(old_name, new_name)
            renamed += 1
        else:
            print(f"Not found: {old_name.name}")
            errors += 1

    print(f"\nDone! Renamed: {renamed}, Not found: {errors}")

if __name__ == "__main__":
    rename_images()