"""
evaluate_v3.py
==============
Evaluation script with filename-based matching.
  - Metrics are computed ONLY for filenames present in predicted_labels.txt
  - GT entries are looked up by filename, not by line index
  - All v2 metrics retained:
      * Standard ExpRate (exact match)
      * Edit distance thresholds (<=1 to <=5)
      * Token accuracy (LCS-based)
      * Structural accuracy (LaTeX structure tokens)
      * Prefix accuracy
      * Edit distance breakdown (insert/delete/substitute)

File format (tab or space separated):
    test_1.jpg\tx + y
    test_1.jpg x + y

Usage:
    python evaluate_v3.py --gt labels.txt --pred predicted_labels.txt

Or with defaults:
    python evaluate_v3.py
"""

import argparse
import os
from difflib import SequenceMatcher
import editdistance

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_GT   = r"C:\Users\samee\Documents\GitHub_Repos\TAMER\data\output_images_test\labels.txt"
DEFAULT_PRED = r"C:\Users\samee\Documents\GitHub_Repos\TAMER\results\predicted_labels_v1.txt"

# ── Load labels as ORDERED LIST ───────────────────────────────────────────────
def load_labels_ordered(path):
    """
    Returns a list of (filename, tokens) in file order.
    Supports both tab-separated and space-separated formats.
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                fname, label = parts
            else:
                parts = line.split(None, 1)
                fname = parts[0]
                label = parts[1] if len(parts) > 1 else ""
            entries.append((fname.strip(), label.strip().split()))
    return entries

# ── Load labels as DICT keyed by filename ─────────────────────────────────────
def load_labels_dict(path):
    """
    Returns an OrderedDict: { filename: token_list }
    Preserves insertion order (i.e., file order).
    Warns on duplicate filenames.
    """
    from collections import OrderedDict
    entries = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                fname, label = parts
            else:
                parts = line.split(None, 1)
                fname = parts[0]
                label = parts[1] if len(parts) > 1 else ""
            fname = fname.strip()
            if fname in entries:
                print(f"  WARNING: Duplicate filename '{fname}' at line {lineno} — overwriting earlier entry.")
            entries[fname] = label.strip().split()
    return entries

# ── LCS-based token accuracy ─────────────────────────────────────────────────
def lcs_length(a, b):
    matcher = SequenceMatcher(None, a, b)
    return sum(block.size for block in matcher.get_matching_blocks())

def token_accuracy(gt_tokens, pred_tokens):
    if not gt_tokens and not pred_tokens:
        return 1.0
    lcs = lcs_length(gt_tokens, pred_tokens)
    return lcs / max(len(gt_tokens), len(pred_tokens))

# ── Prefix accuracy ───────────────────────────────────────────────────────────
def prefix_correct(gt_tokens, pred_tokens):
    n = min(len(gt_tokens), len(pred_tokens))
    for i in range(n):
        if gt_tokens[i] != pred_tokens[i]:
            return i
    return n

# ── Structural token check ────────────────────────────────────────────────────
STRUCTURAL_TOKENS = {
    '{', '}', '^', '_', '\\frac', '\\sqrt', '\\sum', '\\int',
    '\\left', '\\right', '\\begin', '\\end', '\\over',
    '(', ')', '[', ']', '\\{', '\\}', '\\|'
}

def structural_accuracy(gt_tokens, pred_tokens):
    gt_struct   = [t for t in gt_tokens   if t in STRUCTURAL_TOKENS]
    pred_struct = [t for t in pred_tokens if t in STRUCTURAL_TOKENS]
    if not gt_struct:
        return 1.0, 0
    lcs = lcs_length(gt_struct, pred_struct)
    return lcs / len(gt_struct), len(gt_struct)

# ── Edit distance breakdown ───────────────────────────────────────────────────
def edit_ops(gt_tokens, pred_tokens):
    matcher = SequenceMatcher(None, gt_tokens, pred_tokens)
    inserts = deletes = substitutes = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            substitutes += min(i2 - i1, j2 - j1)
            if i2 - i1 > j2 - j1:
                deletes += (i2 - i1) - (j2 - j1)
            elif j2 - j1 > i2 - i1:
                inserts += (j2 - j1) - (i2 - i1)
        elif tag == 'delete':
            deletes += i2 - i1
        elif tag == 'insert':
            inserts += j2 - j1
    return inserts, deletes, substitutes

# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate(gt_path, pred_path):
    gt_dict    = load_labels_dict(gt_path)
    pred_dict  = load_labels_dict(pred_path)

    total_gt   = len(gt_dict)
    total_pred = len(pred_dict)

    # Only evaluate filenames present in predictions, in prediction file order
    evaluated_fnames = list(pred_dict.keys())

    # Check which predicted filenames are missing from GT
    missing_from_gt = [f for f in evaluated_fnames if f not in gt_dict]
    valid_fnames    = [f for f in evaluated_fnames if f in gt_dict]

    print(f"\n{'='*65}")
    print(f"  File Summary")
    print(f"{'='*65}")
    print(f"  Ground truth file           : {gt_path}")
    print(f"  Predicted file              : {pred_path}")
    print(f"  Ground truth entries        : {total_gt}")
    print(f"  Predicted entries           : {total_pred}")
    print(f"  Evaluated (filename-matched): {len(valid_fnames)}")
    print(f"  GT entries skipped (not in pred): {total_gt - len(valid_fnames)}")
    if missing_from_gt:
        print(f"  Pred filenames not in GT    : {len(missing_from_gt)}  (skipped)")
        for f in missing_from_gt[:10]:
            print(f"    - {f}")
        if len(missing_from_gt) > 10:
            print(f"    ... and {len(missing_from_gt) - 10} more")

    if not valid_fnames:
        print("\nERROR: No matching filenames found between GT and predictions.")
        return

    # ── Evaluate over matched filenames ───────────────────────────────────────
    exact           = 0
    le1 = le2 = le3 = le4 = le5 = 0
    total_tok_acc   = 0.0
    total_struct_acc= 0.0
    total_struct_n  = 0
    total_prefix    = 0
    total_gt_len    = 0
    total_inserts   = 0
    total_deletes   = 0
    total_subs      = 0
    errors          = []

    for fname in valid_fnames:
        gt_tok   = gt_dict[fname]
        pred_tok = pred_dict[fname]

        dist = editdistance.eval(gt_tok, pred_tok)
        ins, dels, subs = edit_ops(gt_tok, pred_tok)
        tok_acc = token_accuracy(gt_tok, pred_tok)
        struct_acc, struct_n = structural_accuracy(gt_tok, pred_tok)
        prefix = prefix_correct(gt_tok, pred_tok)

        if dist == 0: exact += 1
        if dist <= 1: le1   += 1
        if dist <= 2: le2   += 1
        if dist <= 3: le3   += 1
        if dist <= 4: le4   += 1
        if dist <= 5: le5   += 1

        total_tok_acc    += tok_acc
        total_struct_acc += struct_acc * struct_n
        total_struct_n   += struct_n
        total_prefix     += prefix
        total_gt_len     += len(gt_tok)
        total_inserts    += ins
        total_deletes    += dels
        total_subs       += subs

        if dist > 0:
            errors.append({
                "fname"   : fname,
                "dist"    : dist,
                "tok_acc" : tok_acc,
                "ins"     : ins,
                "dels"    : dels,
                "subs"    : subs,
                "gt"      : " ".join(gt_tok),
                "pred"    : " ".join(pred_tok),
            })

    errors.sort(key=lambda x: x["dist"], reverse=True)

    total = len(valid_fnames)
    avg_tok_acc    = total_tok_acc / total * 100
    avg_struct_acc = (total_struct_acc / total_struct_n * 100) if total_struct_n > 0 else 100.0
    avg_prefix     = total_prefix / total
    avg_gt_len     = total_gt_len / total

    print(f"\n{'='*65}")
    print(f"  STANDARD METRICS  (denominator = {total} matched samples)")
    print(f"{'='*65}")
    print(f"  ExpRate (exact match)       : {exact/total*100:.2f}%  ({exact}/{total})")
    print(f"  Edit dist <= 1              : {le1/total*100:.2f}%  ({le1}/{total})")
    print(f"  Edit dist <= 2              : {le2/total*100:.2f}%  ({le2}/{total})")
    print(f"  Edit dist <= 3              : {le3/total*100:.2f}%  ({le3}/{total})")
    print(f"  Edit dist <= 4              : {le4/total*100:.2f}%  ({le4}/{total})")
    print(f"  Edit dist <= 5              : {le5/total*100:.2f}%  ({le5}/{total})")

    print(f"\n{'='*65}")
    print(f"  PARTIAL CREDIT METRICS")
    print(f"{'='*65}")
    print(f"  Token accuracy (LCS-based)  : {avg_tok_acc:.2f}%")
    print(f"    → Correct tokens regardless of position shifts")
    print(f"  Structural token accuracy   : {avg_struct_acc:.2f}%")
    print(f"    → \\frac {{}} ^_ \\sqrt etc. correctly predicted")
    print(f"  Avg prefix correct          : {avg_prefix:.1f} tokens")
    print(f"    → Avg tokens correct from start before first error")
    print(f"  Avg GT sequence length      : {avg_gt_len:.1f} tokens")

    print(f"\n{'='*65}")
    print(f"  ERROR BREAKDOWN")
    print(f"{'='*65}")
    print(f"  Total wrong predictions     : {len(errors)}/{total}")
    print(f"  Total insertions            : {total_inserts}  (extra tokens predicted)")
    print(f"  Total deletions             : {total_deletes}  (tokens missed)")
    print(f"  Total substitutions         : {total_subs}   (wrong token predicted)")
    if total_inserts + total_deletes + total_subs > 0:
        total_ops = total_inserts + total_deletes + total_subs
        print(f"  Insert  share               : {total_inserts/total_ops*100:.1f}%")
        print(f"  Delete  share               : {total_deletes/total_ops*100:.1f}%")
        print(f"  Subst.  share               : {total_subs/total_ops*100:.1f}%")
    print(f"{'='*65}\n")

    # ── Save full report ──────────────────────────────────────────────────────
    report_path = os.path.join(os.path.dirname(pred_path), "mismatch_report_v3.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Evaluation Report v3  (filename-matched, pred-scoped)\n")
        f.write(f"GT file   : {gt_path}\n")
        f.write(f"Pred file : {pred_path}\n")
        f.write(f"GT total  : {total_gt}  |  Pred total: {total_pred}  |  Evaluated: {total}\n\n")
        f.write(f"ExpRate            : {exact/total*100:.2f}%\n")
        f.write(f"Token accuracy     : {avg_tok_acc:.2f}%\n")
        f.write(f"Structural acc     : {avg_struct_acc:.2f}%\n")
        f.write(f"Avg prefix correct : {avg_prefix:.1f} tokens\n")
        f.write(f"Insertions         : {total_inserts}\n")
        f.write(f"Deletions          : {total_deletes}\n")
        f.write(f"Substitutions      : {total_subs}\n")
        f.write("=" * 80 + "\n\n")
        f.write("WRONG PREDICTIONS (sorted by edit distance, worst first):\n\n")
        for e in errors:
            f.write(f"File     : {e['fname']}\n")
            f.write(f"EditDist : {e['dist']}  |  TokenAcc: {e['tok_acc']*100:.1f}%  |  ins={e['ins']} del={e['dels']} sub={e['subs']}\n")
            f.write(f"GT       : {e['gt']}\n")
            f.write(f"PRED     : {e['pred']}\n\n")

    print(f"Full report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",   default=DEFAULT_GT,   help="Ground truth labels file")
    parser.add_argument("--pred", default=DEFAULT_PRED, help="Predicted labels file")
    args = parser.parse_args()
    evaluate(args.gt, args.pred)