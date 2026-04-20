"""with open("data/CROHME/crohme/2014/caption.txt") as f:
    lines = f.readlines()
lengths = [len(line.strip().split()) - 1 for line in lines]  # -1 for filename
print(f"Max tokens: {max(lengths)}")
print(f"Mean tokens: {sum(lengths)/len(lengths):.1f}")
print(f"Captions > 50 tokens: {sum(1 for l in lengths if l > 50)}")
print(f"Captions > 100 tokens: {sum(1 for l in lengths if l > 100)}")"""

"""from tamer.datamodule.vocab import vocab
vocab.init('lightning_logs/version_4/dictionary.txt')
print(repr('\\\\' in vocab.word2idx))
print(repr('\\\\\\\\' in vocab.word2idx))"""

import torch
import pickle
import cv2
from tamer.datamodule.vocab import vocab
from tamer.model.tamer import TAMER

vocab.init('lightning_logs/version_4/dictionary.txt')

model = TAMER.load_from_checkpoint(
    'lightning_logs/version_4/checkpoints/epoch=51-step=162967_vocab334.ckpt',
    map_location='cpu'
)
model.eval()

# Make a tiny synthetic batch
imgs  = torch.ones(1, 1, 64, 256) * 0.5   # mid-grey image
masks = torch.zeros(1, 64, 256, dtype=torch.bool)  # all valid
tgt   = torch.tensor([[1, 5, 6, 2], [2, 6, 5, 1]], dtype=torch.long)  # [2b, l] fake tokens

with torch.no_grad():
    # Step 1 — encoder
    feature, mask = model.encoder(imgs, masks)
    print('encoder feature nan:', torch.isnan(feature).any().item())
    print('encoder mask nan:', torch.isnan(mask.float()).any().item())

    # Step 2 — double for bidirectional
    feature2 = torch.cat([feature, feature], dim=0)
    mask2    = torch.cat([mask, mask], dim=0)

    # Step 3 — decoder
    out, sim = model.decoder(feature2, mask2, tgt)
    print('decoder out nan:', torch.isnan(out).any().item())
    print('decoder out min/max:', out.min().item(), out.max().item())