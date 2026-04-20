"""
build_extended_dict.py  —  Step 1 of TAMER vocab expansion

Creates lightning_logs/version_4/ and populates it with:
  - dictionary.txt       (extended vocab, new tokens appended at end)
  - hparams.yaml         (copy of version_1 hparams, vocab_size updated)

Existing version_1 files are never touched.

Usage (run from TAMER repo root):
    python build_extended_dict.py
"""

import os
import re

# ── paths ─────────────────────────────────────────────────────────────────────
VERSION_1_DIR  = r"lightning_logs/version_1"
VERSION_4_DIR  = r"lightning_logs/version_4"

V1_DICT_PATH   = os.path.join(VERSION_1_DIR, "dictionary.txt")
V1_HPARAMS     = os.path.join(VERSION_1_DIR, "hparams.yaml")

V4_DICT_PATH   = os.path.join(VERSION_4_DIR, "dictionary.txt")
V4_HPARAMS     = os.path.join(VERSION_4_DIR, "hparams.yaml")
V4_CKPT_DIR    = os.path.join(VERSION_4_DIR, "checkpoints")
# ─────────────────────────────────────────────────────────────────────────────

# MathWriting vocab (252 tokens — treated as complete for now)
# Note: \mathbb{X} is NOT atomic in your tokenizer — it splits to
#       \mathbb { X }, so only \mathbb itself needs to be added.
# Note: matrix environment produces tokens: \begin  \end  matrix  \\
MATHWRITING_VOCAB = set([
    # punctuation / misc single chars
    "!", "#", "%", "&", "(", ")", "*", "+", ",", "-", ".", "/",
    ":", ";", "<", "=", ">", "?", "[", "]", "{", "}", "|",
    # digits
    "0","1","2","3","4","5","6","7","8","9",
    # uppercase latin
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    # lowercase latin
    "a","b","c","d","e","f","g","h","i","j","k","l","m",
    "n","o","p","q","r","s","t","u","v","w","x","y","z",
    # structural
    "^","_",
    # Greek lowercase
    "\\alpha","\\beta","\\gamma","\\delta","\\epsilon","\\zeta","\\eta",
    "\\theta","\\vartheta","\\iota","\\kappa","\\lambda","\\mu","\\nu",
    "\\xi","\\pi","\\rho","\\sigma","\\varsigma","\\tau","\\upsilon",
    "\\phi","\\varphi","\\chi","\\psi","\\omega",
    # Greek uppercase
    "\\Gamma","\\Delta","\\Theta","\\Lambda","\\Xi","\\Pi",
    "\\Sigma","\\Upsilon","\\Phi","\\Psi","\\Omega",
    # core math
    "\\frac","\\sqrt","\\sum","\\prod","\\int","\\iint","\\oint",
    "\\nabla","\\partial","\\infty","\\prime","\\vdots",
    # arrows
    "\\rightarrow","\\leftarrow","\\leftrightarrow",
    "\\Rightarrow","\\Leftrightarrow","\\rightleftharpoons",
    "\\mapsto","\\longrightarrow","\\hookrightarrow","\\iff",
    # set / logic
    "\\in","\\notin","\\ni",
    "\\subset","\\subseteq","\\subsetneq",
    "\\supset","\\supseteq",
    "\\cup","\\cap","\\bigcup","\\bigcap",
    "\\bigoplus","\\bigvee","\\bigwedge",
    "\\vee","\\wedge",
    "\\oplus","\\ominus","\\otimes","\\odot",
    "\\forall","\\exists","\\neg","\\top","\\perp",
    "\\vdash","\\Vdash","\\models","\\emptyset","\\aleph",
    # relational / operators
    "\\pm","\\mp","\\times","\\div","\\cdot","\\circ","\\bullet",
    "\\angle","\\triangleq","\\equiv","\\cong","\\approx",
    "\\simeq","\\sim","\\propto",
    "\\le","\\ge","\\ll","\\gg","\\ne",
    # accents / decorators
    "\\dot","\\hat","\\tilde","\\vec","\\overline","\\underline",
    # blackboard bold (command only)
    "\\mathbb",
    # delimiters
    "\\langle","\\rangle","\\lceil","\\rceil","\\lfloor","\\rfloor",
    "\\backslash","\\|",
    # misc
    "\\dagger","\\hbar","\\varpi","\\triangle","\\bigcirc",
    # matrix environment (from your tokenizer)
    "\\begin","\\end","matrix","\\\\",
])


# ── load version_1 dictionary ─────────────────────────────────────────────────
hme_tokens = []
with open(V1_DICT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        tok = line.rstrip("\n")
        if tok:
            hme_tokens.append(tok)

hme_set = set(hme_tokens)

# ── compute new tokens ────────────────────────────────────────────────────────
# sort: \commands first (alphabetically), then single chars
new_tokens = sorted(
    [t for t in MATHWRITING_VOCAB if t not in hme_set],
    key=lambda t: (not t.startswith("\\"), t)
)

old_vocab_size = len(hme_tokens) + 3   # +3 for <pad>/<sos>/<eos>
new_vocab_size = old_vocab_size + len(new_tokens)

# ── report ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TAMER VOCAB EXPANSION  —  version_1 → version_4")
print("=" * 60)
print(f"  version_1 dictionary tokens : {len(hme_tokens)}")
print(f"  version_1 vocab_size        : {old_vocab_size}  (incl. pad/sos/eos)")
print(f"  MathWriting vocab size      : {len(MATHWRITING_VOCAB)}")
print(f"  New tokens to add           : {len(new_tokens)}")
print(f"  version_4 vocab_size        : {new_vocab_size}")
print()
print("New tokens (with their new indices):")
for i, tok in enumerate(new_tokens):
    print(f"  [{old_vocab_size + i:>3}]  {repr(tok)}")
print()

# ── already present (sanity check) ───────────────────────────────────────────
already_present = [t for t in MATHWRITING_VOCAB if t in hme_set]
print(f"Already in HME100K dict    : {len(already_present)} tokens")
print()

# ── create version_4 folder structure ────────────────────────────────────────
os.makedirs(V4_CKPT_DIR, exist_ok=True)
print(f"Created : {VERSION_4_DIR}/")
print(f"Created : {V4_CKPT_DIR}/   (empty — surgery script fills this)")

# ── write extended dictionary.txt ────────────────────────────────────────────
with open(V4_DICT_PATH, "w", encoding="utf-8") as f:
    for tok in hme_tokens:
        f.write(tok + "\n")
    for tok in new_tokens:
        f.write(tok + "\n")

print(f"Written : {V4_DICT_PATH}")
print(f"          {len(hme_tokens)} original + {len(new_tokens)} new "
      f"= {len(hme_tokens) + len(new_tokens)} lines")

# ── write updated hparams.yaml ────────────────────────────────────────────────
with open(V1_HPARAMS, "r", encoding="utf-8") as f:
    hparams = f.read()

hparams = re.sub(r"vocab_size:\s*\d+",
                 f"vocab_size: {new_vocab_size}", hparams)
hparams = re.sub(r"beam_size:\s*\d+",
                 "beam_size: 5", hparams)

with open(V4_HPARAMS, "w", encoding="utf-8") as f:
    f.write(hparams)

print(f"Written : {V4_HPARAMS}")
print(f"          vocab_size : {old_vocab_size} → {new_vocab_size}")
print(f"          beam_size  : 10 → 5  (laptop-friendly)")

print()
print("=" * 60)
print("Step 1 complete. version_4/ structure is ready.")
print("Next step: run model_surgery.py to expand the checkpoint.")
print("=" * 60)