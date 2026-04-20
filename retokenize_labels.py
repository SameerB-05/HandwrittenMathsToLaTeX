# retokenize_labels.py
# Fixes OOV tokens in labels.txt by splitting only the broken tokens,
# leaving already-valid vocab tokens untouched.
# Run from TAMER repo root.

import re
from pathlib import Path
from tqdm import tqdm
from tamer.datamodule.vocab import vocab

# =========================
# CONFIG
# =========================
DICT_PATH     = "lightning_logs/version_4/dictionary.txt"
INPUT_LABELS  = Path(r"C:\Users\samee\Documents\GitHub_Repos\TAMER\math_writing_labels.txt")
OUTPUT_LABELS = Path(r"C:\Users\samee\Documents\GitHub_Repos\TAMER\math_writing_labels_v2.txt")


# =========================
# SPLIT A SINGLE BAD TOKEN
# =========================
def split_token(tok):
    """Called only for tokens not in vocab. Splits into known sub-tokens."""
    return re.findall(r'\\[a-zA-Z]+|matrix|[a-zA-Z]|[0-9]|[^\s]', tok)


# =========================
# RETOKENIZE ONE LABEL LINE
# =========================
def retokenize_line(label):
    result = []
    for tok in label.split():
        tok = tok.replace('\\\\\\\\', '\\\\')  # fix 4-char → 2-char backslash
        if tok in vocab.word2idx:
            result.append(tok)
        else:
            result.extend(split_token(tok))
    return " ".join(result)


# =========================
# MAIN
# =========================
def main():
    vocab.init(DICT_PATH)

    lines = INPUT_LABELS.read_text(encoding="utf-8").splitlines()

    with open(OUTPUT_LABELS, "w", encoding="utf-8") as f_out:
        for line in tqdm(lines, desc="Retokenizing"):
            parts = line.split("\t", maxsplit=1)
            if len(parts) < 2:
                f_out.write(line + "\n")
                continue
            filename, label = parts
            f_out.write(f"{filename}\t{retokenize_line(label)}\n")

    print(f"\nDone. Written to {OUTPUT_LABELS}")


if __name__ == "__main__":
    main()