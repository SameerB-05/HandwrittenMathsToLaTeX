# finetune.py
# Fine-tunes the version_4 (vocab334) checkpoint on MathWriting data.
# Encoder frozen, decoder only trained. Run from TAMER repo root.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pickle
import cv2
from pathlib import Path
from tamer.datamodule.vocab import vocab
from tamer.model.tamer import TAMER

# =========================
# CONFIG
# =========================
CKPT_PATH   = "lightning_logs/version_4/checkpoints/epoch=51-step=162967_vocab334.ckpt"
DICT_PATH   = "lightning_logs/version_4/dictionary.txt"
IMAGES_PKL  = "data/mathwriting/train/images.pkl"
CAPTION_TXT = "data/mathwriting/train/caption.txt"

NUM_EPOCHS  = 5
BATCH_SIZE  = 32
LR          = 1e-5
H_HI, W_HI = 128, 512
MAX_LEN     = 200
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR    = Path("lightning_logs/version_4/checkpoints")


# =========================
# DATASET
# =========================
class MathWritingDataset(Dataset):
    def __init__(self, images_dict, captions):
        self.samples = []
        for line in captions:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            filename = parts[0]
            tokens   = parts[1:]
            if filename not in images_dict:
                continue
            self.samples.append((images_dict[filename], tokens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, tokens = self.samples[idx]
        h, w = img.shape
        scale = min(H_HI / h, W_HI / w, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        token_ids = (
            [vocab.SOS_IDX]
            + [vocab.word2idx[t] for t in tokens if t in vocab.word2idx]
            + [vocab.EOS_IDX]
        )
        return img, torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch):
    imgs, seqs = zip(*batch)
    max_h = max(img.shape[0] for img in imgs)
    max_w = max(img.shape[1] for img in imgs)

    img_tensors = torch.ones(len(imgs), 1, max_h, max_w)
    img_masks   = torch.ones(len(imgs), max_h, max_w, dtype=torch.bool)
    for i, img in enumerate(imgs):
        h, w = img.shape
        img_tensors[i, 0, :h, :w] = img
        img_masks[i, :h, :w] = False

    max_len     = max(s.shape[0] for s in seqs)
    seq_tensors = torch.full((len(seqs), max_len), vocab.PAD_IDX, dtype=torch.long)
    for i, seq in enumerate(seqs):
        seq_tensors[i, :seq.shape[0]] = seq

    return img_tensors, img_masks, seq_tensors


# =========================
# MAIN
# =========================
def main():
    vocab.init(DICT_PATH)

    print("Loading images.pkl ...")
    with open(IMAGES_PKL, "rb") as f:
        images_dict = pickle.load(f)

    captions = Path(CAPTION_TXT).read_text(encoding="utf-8").splitlines()
    dataset  = MathWritingDataset(images_dict, captions)
    print(f"Dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Loading checkpoint: {CKPT_PATH} ...")
    model = TAMER.load_from_checkpoint(CKPT_PATH, map_location=DEVICE)
    model.hparams.beam_size = 5
    model.hparams.max_len   = MAX_LEN
    model = model.to(DEVICE)

    for param in model.encoder.parameters():
        param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Encoder frozen. Trainable parameters: {sum(p.numel() for p in trainable):,}\n")

    optimizer = torch.optim.Adam(trainable, lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss    = 0.0
        total_batches = 0

        for batch_idx, (imgs, masks, seqs) in enumerate(dataloader):
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            seqs  = seqs.to(DEVICE)

            tgt_in  = seqs[:, :-1]
            tgt_out = seqs[:, 1:]

            tgt_in_2b  = torch.cat([tgt_in,  tgt_in],  dim=0)
            tgt_out_2b = torch.cat([tgt_out, tgt_out], dim=0)

            optimizer.zero_grad()

            with torch.no_grad():
                feature, mask = model.encoder(imgs, masks)
            feature2 = torch.cat([feature, feature], dim=0)
            mask2    = torch.cat([mask,    mask],    dim=0)

            logits, _ = model.decoder(feature2, mask2, tgt_in_2b)

            B2, L, V = logits.shape
            loss = criterion(logits.reshape(B2 * L, V), tgt_out_2b.reshape(B2 * L))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            total_loss    += loss.item()
            total_batches += 1

            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss {loss.item():.4f}")

        avg_loss = total_loss / total_batches
        save_path = SAVE_DIR / f"finetune_epoch{epoch}_loss{avg_loss:.4f}.ckpt"
        torch.save(model.state_dict(), save_path)
        print(f"\nEpoch {epoch} complete — avg loss: {avg_loss:.4f} | Saved → {save_path}\n")

    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()