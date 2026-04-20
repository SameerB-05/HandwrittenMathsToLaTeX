# filter_oov.py
# Removes any samples containing OOV tokens from labels_v2.txt
# Run from TAMER repo root.

from pathlib import Path
from tamer.datamodule.vocab import vocab

DICT_PATH     = "lightning_logs/version_4/dictionary.txt"
INPUT_LABELS  = Path(r"C:\Users\samee\Documents\GitHub_Repos\TAMER\math_writing_labels_v2.txt")
OUTPUT_LABELS = Path(r"C:\Users\samee\Documents\GitHub_Repos\TAMER\math_writing_labels_final.txt")

vocab.init(DICT_PATH)

lines = INPUT_LABELS.read_text(encoding="utf-8").splitlines()

kept, removed = 0, 0
with open(OUTPUT_LABELS, "w", encoding="utf-8") as f_out:
    for line in lines:
        parts = line.split("\t", maxsplit=1)
        if len(parts) < 2:
            continue
        tokens = parts[1].split()
        if any(tok not in vocab.word2idx for tok in tokens):
            removed += 1
        else:
            f_out.write(line + "\n")
            kept += 1

print(f"Kept:    {kept}")
print(f"Removed: {removed}")
print(f"Output:  {OUTPUT_LABELS}")