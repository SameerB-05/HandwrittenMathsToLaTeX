import re
from pathlib import Path
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_LABELS  = Path(r"C:\Users\samee\Documents\GitHub_Repos\TAMER\math_writing_labels.txt")
OUTPUT_LABELS = Path(r"C:\Users\samee\Documents\GitHub_Repos\TAMER\math_writing_labels_v2.txt")


# =========================
# TOKENIZER (FIXED)
# =========================
def tokenize_latex(expr):
    if expr is None:
        return ""
    tokens = re.findall(
        r'\\[a-zA-Z]+|matrix|[a-zA-Z]|[0-9]|[^\s]',
        expr
    )
    return " ".join(tokens)


# =========================
# MAIN
# =========================
def retokenize():
    lines = INPUT_LABELS.read_text(encoding="utf-8").splitlines()

    with open(OUTPUT_LABELS, "w", encoding="utf-8") as f_out:
        for line in tqdm(lines, desc="Retokenizing"):
            parts = line.split("\t", maxsplit=1)
            if len(parts) < 2:
                f_out.write(line + "\n")
                continue

            filename, raw_label = parts[0], parts[1]

            # The existing labels.txt already has some tokenization applied.
            # We need to work on the RAW label, not re-tokenize already-spaced tokens.
            # So we first strip all spaces to recover the original expression,
            # then apply the new tokenizer.
            collapsed = re.sub(r'\s+', '', raw_label)
            new_label = tokenize_latex(collapsed)

            f_out.write(f"{filename}\t{new_label}\n")

    print(f"Done. Written to {OUTPUT_LABELS}")


if __name__ == "__main__":
    retokenize()