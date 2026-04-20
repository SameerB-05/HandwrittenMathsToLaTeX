# check_prediction.py
import pickle
from PIL import Image as PILImage
import numpy as np

vocab_path = "data/CROHME/crohme/dictionary.txt"
pkl_path = "data/CROHME/crohme/2014/images.pkl"
caption_path = "data/CROHME/crohme/2014/caption.txt"

# get ground truth for RIT_2014_293
with open(caption_path) as f:
    for line in f:
        parts = line.strip().split()
        if parts[0] == "RIT_2014_59":
            gt = " ".join(parts[1:])
            print(f"Ground truth:  {gt}")
            break

print(f"Prediction:    + m ( 2 - 2 - 3 ) = \\frac {{ 2 - 1 - 1 }} {{ n + i }} )")

# save the image so you can see it
with open(pkl_path, "rb") as f:
    images = pickle.load(f)
img = images["RIT_2014_59"]
PILImage.fromarray(img).save("test_image3.png")
print(f"\nImage saved to test_image3.png — open it to see the handwritten equation")
print(f"Image shape: {img.shape}")