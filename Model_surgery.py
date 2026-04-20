# model_surgery.py
# Expands the three vocab-dependent tensors in the version_1 checkpoint
# from vocab_size=248 to vocab_size=334, zero-initialising new rows.
# Run from TAMER repo root.

import torch
from pathlib import Path

# =========================
# CONFIG
# =========================
SRC_CKPT   = Path("lightning_logs/version_1/checkpoints/epoch=51-step=162967-val_ExpRate=0.6851.ckpt")
DST_DIR    = Path("lightning_logs/version_4/checkpoints")
OLD_VOCAB  = 248
NEW_VOCAB  = 334
D_MODEL    = 256  # embedding dimension, unchanged

DST_DIR.mkdir(parents=True, exist_ok=True)
DST_CKPT = DST_DIR / "epoch=51-step=162967_vocab334.ckpt"


# =========================
# LOAD
# =========================
print(f"Loading {SRC_CKPT} ...")
ckpt = torch.load(SRC_CKPT, map_location="cpu")
sd = ckpt["state_dict"]

# Sanity check — confirm original shapes
# CHANGE THESE THREE KEY NAMES
we = sd["tamer_model.decoder.word_embed.0.weight"]
pw = sd["tamer_model.decoder.proj.weight"]
pb = sd["tamer_model.decoder.proj.bias"]

print(f"  word_embed.0.weight : {tuple(we.shape)}")
print(f"  proj.weight         : {tuple(pw.shape)}")
print(f"  proj.bias           : {tuple(pb.shape)}")

assert we.shape == (OLD_VOCAB, D_MODEL), f"Unexpected word_embed shape: {we.shape}"
assert pw.shape == (OLD_VOCAB, D_MODEL), f"Unexpected proj.weight shape: {pw.shape}"
assert pb.shape == (OLD_VOCAB,),         f"Unexpected proj.bias shape:   {pb.shape}"

print(f"\nExpanding {OLD_VOCAB} → {NEW_VOCAB} ...")


# =========================
# EXPAND
# =========================
def expand_rows(tensor, new_rows):
    """Append zero-initialised rows to a 2D tensor."""
    extra = torch.zeros(new_rows, tensor.shape[1], dtype=tensor.dtype)
    return torch.cat([tensor, extra], dim=0)

n_new = NEW_VOCAB - OLD_VOCAB  # 86 new rows

# AND THE THREE ASSIGNMENT LINES
sd["tamer_model.decoder.word_embed.0.weight"] = expand_rows(we, n_new)
sd["tamer_model.decoder.proj.weight"]         = expand_rows(pw, n_new)
sd["tamer_model.decoder.proj.bias"]           = torch.cat([pb, torch.zeros(n_new, dtype=pb.dtype)])

# Confirm new shapes
print(f"  word_embed.0.weight : {tuple(sd['tamer_model.decoder.word_embed.0.weight'].shape)}")
print(f"  proj.weight         : {tuple(sd['tamer_model.decoder.proj.weight'].shape)}")
print(f"  proj.bias           : {tuple(sd['tamer_model.decoder.proj.bias'].shape)}")


# =========================
# UPDATE HPARAMS INSIDE CHECKPOINT
# =========================
if "hyper_parameters" in ckpt:
    old_vs = ckpt["hyper_parameters"].get("vocab_size", "?")
    ckpt["hyper_parameters"]["vocab_size"] = NEW_VOCAB
    print(f"\nUpdated hyper_parameters.vocab_size: {old_vs} → {NEW_VOCAB}")
else:
    print("\nWarning: no hyper_parameters key found in checkpoint — skipping")


# =========================
# SAVE
# =========================

# Strip tamer_model. prefix so Lightning can load correctly
fixed_sd = {}
for k, v in sd.items():
    if k.startswith("tamer_model."):
        fixed_sd[k[len("tamer_model."):]] = v
    else:
        fixed_sd[k] = v


ckpt["state_dict"] = fixed_sd
torch.save(ckpt, DST_CKPT)
print(f"\nSaved → {DST_CKPT}")
print("Surgery complete. Ready for fine-tuning.")