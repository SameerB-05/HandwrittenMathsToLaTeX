# check_oov.py  — run from TAMER repo root
from tamer.datamodule.vocab import vocab

DICT_PATH   = "lightning_logs/version_4/dictionary.txt"
LABELS_PATH = r"C:\Users\samee\Documents\GitHub_Repos\TAMER\math_writing_labels_final.txt"

vocab.init(DICT_PATH)

oov = set()
oov_sample_count = 0
total_samples = 0

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        total_samples += 1
        tokens = parts[1].split()
        sample_has_oov = False
        for tok in tokens:
            if tok not in vocab.word2idx:
                oov.add(tok)
                sample_has_oov = True
        if sample_has_oov:
            oov_sample_count += 1

print(f"Total samples: {total_samples}")
print(f"Samples with at least one OOV token: {oov_sample_count}  ({100*oov_sample_count/total_samples:.1f}%)")
print(f"Samples that would survive filtering: {total_samples - oov_sample_count}  ({100*(total_samples - oov_sample_count)/total_samples:.1f}%)")
print(f"\nOut-of-vocabulary unique tokens: {len(oov)}")
for tok in sorted(oov):
    print(f"  {repr(tok)}")