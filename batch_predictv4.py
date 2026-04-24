# batch_predict_v4.py
# ============================================================
# Root cause identified:
# beam_search in generation_utils.py uses len(vocab) internally
# for index arithmetic. After ~4 images the CUDA allocator
# accumulates enough fragmentation from beam search's internal
# tensors that it triggers a hard OOM that poisons the context.
#
# Fix: set PYTORCH_NO_CUDA_MEMORY_CACHING=1 to disable the
# caching allocator, forcing immediate release after each op.
# Also set CUDA_LAUNCH_BLOCKING=1 so errors are caught at the
# right call site instead of cascading.
# ============================================================

import os
import sys
import re
import gc
import torch
from PIL import Image
from torchvision import transforms
from tamer.lit_tamer import LitTAMER
from tamer.datamodule import vocab

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── MUST be set before any CUDA call ────────────────────────
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"]           = "1"

# ── Config ──────────────────────────────────────────────────
V1_CKPT      = r"C:\Users\samee\Documents\GitHub_Repos\TAMER\lightning_logs\version_1\checkpoints\epoch=51-step=162967-val_ExpRate=0.6851.ckpt"
V4_WEIGHTS   = r"C:\Users\samee\Documents\GitHub_Repos\TAMER\lightning_logs\version_5\checkpoints\finetune_epoch10_loss0.6182.ckpt"
DICT_PATH    = r"C:\Users\samee\Documents\GitHub_Repos\TAMER\lightning_logs\version_5\dictionary.txt"
IMAGE_FOLDER = r"C:\Users\samee\Documents\GitHub_Repos\TAMER\data\output_images_test"
OUTPUT_FILE  = os.path.join(IMAGE_FOLDER, "predicted_labels_v4_e10.txt")
VOCAB_SIZE   = 334
D_MODEL      = 256
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY    = 50
TARGET_H     = 128
MAX_W        = 800
# ────────────────────────────────────────────────────────────

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def load_and_resize(img_path):
    img  = Image.open(img_path).convert("L")
    w, h = img.size
    scale = TARGET_H / h
    new_w = min(int(w * scale), MAX_W)
    return img.resize((new_w, TARGET_H), Image.LANCZOS)

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7931], std=[0.1738])
])

# ── Load vocab ───────────────────────────────────────────────
print("Loading vocab...")
vocab.init(DICT_PATH)
print(f"Vocab size: {len(vocab)}")

# ── Load model ───────────────────────────────────────────────
print(f"\nLoading model on {DEVICE}...")
model = LitTAMER.load_from_checkpoint(V1_CKPT, map_location="cpu")

print(f"  Resizing vocab layers: {model.hparams.vocab_size} → {VOCAB_SIZE}")
model.tamer_model.decoder.word_embed[0] = torch.nn.Embedding(VOCAB_SIZE, D_MODEL)
model.tamer_model.decoder.proj          = torch.nn.Linear(D_MODEL, VOCAB_SIZE, bias=True)

print("  Loading finetuned weights...")
raw     = torch.load(V4_WEIGHTS, map_location="cpu")
full_sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
tamer_sd = (
    {k[len("tamer_model."):]: v for k, v in full_sd.items() if k.startswith("tamer_model.")}
    if any(k.startswith("tamer_model.") for k in full_sd)
    else full_sd
)
missing, unexpected = model.tamer_model.load_state_dict(tamer_sd, strict=False)
print(f"  Missing: {len(missing)}   Unexpected: {len(unexpected)}")

# ── Critical: patch hparams so beam_search uses correct vocab_size ──
model._hparams["vocab_size"] = VOCAB_SIZE

model.eval()
model.to(DEVICE)

if DEVICE == "cuda":
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e6
    used_vram  = torch.cuda.memory_allocated() / 1e6
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {used_vram:.0f}MB used / {total_vram:.0f}MB total")
    print(f"  CUDA caching allocator: DISABLED (prevents OOM accumulation)")

print(f"  Model ready on {DEVICE}!\n")

# ── Collect images ───────────────────────────────────────────
exts = {'.jpg', '.jpeg', '.png', '.bmp'}
all_files = sorted(
    [f for f in os.listdir(IMAGE_FOLDER)
     if os.path.splitext(f)[1].lower() in exts],
    key=natural_sort_key
)

# ── Check for already-completed predictions (resume support) ─
completed = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                completed.add(parts[0])
    if completed:
        print(f"Resuming: {len(completed)} already done, "
              f"{len(all_files) - len(completed)} remaining.\n")

print(f"Found {len(all_files)} images in : {IMAGE_FOLDER}")
print(f"Output will be written to        : {OUTPUT_FILE}\n")

# ── Inference loop ───────────────────────────────────────────
failed = []
done   = len(completed)

# Open in append mode so resumed runs don't overwrite existing results
write_mode = "a" if completed else "w"

with open(OUTPUT_FILE, write_mode, encoding="utf-8") as f_out:
    for idx, fname in enumerate(all_files, 1):

        # Skip already-completed
        if fname in completed:
            continue

        img_path   = os.path.join(IMAGE_FOLDER, fname)
        img_tensor = None
        mask       = None
        hyps       = None

        try:
            image      = load_and_resize(img_path)
            img_tensor = normalize(image).unsqueeze(0).to(DEVICE)
            mask       = torch.zeros(
                1, img_tensor.shape[2], img_tensor.shape[3],
                dtype=torch.bool
            ).to(DEVICE)

            with torch.no_grad():
                hyps = model.tamer_model.beam_search(
                    img_tensor, mask, **model.hparams
                )

            latex = vocab.indices2label(hyps[0].seq)

            f_out.write(f"{fname}\t{latex}\n")
            f_out.flush()
            done += 1

            if done % LOG_EVERY == 0 or idx == len(all_files):
                pct  = 100 * idx / len(all_files)
                vram = torch.cuda.memory_allocated() / 1e6 if DEVICE == "cuda" else 0
                print(f"[{idx}/{len(all_files)}] ({pct:.1f}%)  VRAM:{vram:.0f}MB  "
                      f"{fname} -> {latex[:50]}...")

        except Exception as e:
            print(f"  ERROR on {fname}: {e}")
            f_out.write(f"{fname}\tERROR\n")
            f_out.flush()
            failed.append(fname)

        finally:
            del img_tensor, mask, hyps
            gc.collect()

print(f"\nDone!  {done} succeeded,  {len(failed)} failed.")
print(f"Saved : {OUTPUT_FILE}")
if failed:
    print(f"Failed ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")