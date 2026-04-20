# check_image_sizes.py
import pickle
import numpy as np

with open("data/CROHME/crohme/2014/images.pkl", "rb") as f:
    images = pickle.load(f)

sizes = [v.shape for v in images.values()]
areas = [s[0]*s[1] for s in sizes]
print(f"Max size: {max(sizes)}")
print(f"Min size: {min(sizes)}")
print(f"Mean area: {np.mean(areas):.0f}")
print(f"Max area: {max(areas)}")
print(f"Images > 32000px: {sum(1 for a in areas if a > 32000)}")
print(f"Images > 50000px: {sum(1 for a in areas if a > 50000)}")