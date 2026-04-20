# single_inference.py
import torch
from PIL import Image as PILImage
import numpy as np
import pickle
from tamer.datamodule.vocab import vocab

# init vocab first
vocab.init("data/CROHME/crohme/dictionary.txt")

# load one image
with open("data/CROHME/crohme/2014/images.pkl", "rb") as f:
    images = pickle.load(f)

img_name = list(images.keys())[27]
img = images[img_name]
print(f"Testing with: {img_name}, shape: {img.shape}")
print(f"GPU free before model load: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

from tamer.lit_tamer import LitTAMER
model = LitTAMER.load_from_checkpoint(
    "lightning_logs/version_0/checkpoints/epoch=315-step=118815-val_ExpRate=0.6113.ckpt"
)
model.eval().cuda()
print(f"GPU free after model load: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

from torchvision import transforms
transform = transforms.ToTensor()

pil = PILImage.fromarray(img).resize((256, 64), PILImage.LANCZOS)
img_tensor = transform(pil).unsqueeze(0).cuda()
mask = torch.zeros(1, 64, 256, dtype=torch.bool).cuda()

print(f"GPU free before inference: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

with torch.no_grad():
    result = model.approximate_joint_search(img_tensor, mask)

print(f"GPU free after inference: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
print(f"Predicted tokens: {result}")

# decode to latex
from tamer.datamodule.vocab import vocab as v
predicted_latex = " ".join(v.indices2words(result[0].seq))
print(f"Predicted LaTeX: {predicted_latex}")