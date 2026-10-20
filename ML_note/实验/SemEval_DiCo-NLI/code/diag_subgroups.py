"""Exploratory split of the word-pair diagnostic (not pre-registered): substitutions of content words vs
"substitutions" where one side is a function word (article, preposition ...), which are word-order or structure
shifts rather than lexical substitutions. Reads the saved out-of-fold logits; no model needed.

Usage:
    python3 diag_subgroups.py <official_repo_dir> <diagnostics_dir> <out_json>
"""
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SWAP = [0, 2, 1, 3]
TOKEN = re.compile(r"\w+")
FUNCTION_WORDS = {"a", "an", "the", "with", "on", "in", "of", "for", "at", "to", "from", "by", "as", "and", "is",
                  "are", "s", "after", "over", "into", "near", "his", "her", "its", "their"}
SEEDS, FOLDS, BOOTSTRAP = (42, 43, 44), 5, 2000


def substitution(a, b):
    """Same rule as notebooks/dico_diagnostics.py: exactly one extra \\w+ token on each side, at least one shared."""
    ta, tb = TOKEN.findall(a), TOKEN.findall(b)
    A, B = Counter(t.lower() for t in ta), Counter(t.lower() for t in tb)
    da, db = A - B, B - A
    if not (A & B) or sum(da.values()) != 1 or sum(db.values()) != 1:
        return None
    wa, wb = next(iter(da)), next(iter(db))
    return [t for t in ta if t.lower() == wa][-1], [t for t in tb if t.lower() == wb][-1]


def softmax(row):
    m = max(row)
    e = [math.exp(x - m) for x in row]
    return [x / sum(e) for x in e]


def sym_label(lm, a, b):
    pa, pb = softmax(lm[(a, b)]), softmax(lm[(b, a)])
    return LABELS[max(range(4), key=lambda i: pa[i] + pb[SWAP[i]])]


def main():
    repo, diag, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    with open(repo / "final_data/train/dico_nli_train_track1_reference.csv", encoding="utf-8") as f:
        train = list(csv.DictReader(f))
    split = json.loads((diag / "families.json").read_text(encoding="utf-8"))
    fam_of, fold_of = split["family_of_pair"], split["fold_of_family"]
    fam_list = sorted(set(fam_of.values()))
    fam_idx = {f: i for i, f in enumerate(fam_list)}
    rng = np.random.default_rng(0)
    draws = [np.bincount(rng.integers(0, len(fam_list), len(fam_list)), minlength=len(fam_list)) for _ in range(BOOTSTRAP)]
    subs = [(r, s) for r in train for s in [substitution(r["text1"], r["text2"])] if s]
    groups = {"content_words": [x for x in subs if x[1][0].lower() not in FUNCTION_WORDS and x[1][1].lower() not in FUNCTION_WORDS],
              "function_word_on_one_side": [x for x in subs if x[1][0].lower() in FUNCTION_WORDS or x[1][1].lower() in FUNCTION_WORDS]}
    result = {"note": "exploratory, defined after seeing the pre-registered result", "function_words": sorted(FUNCTION_WORDS)}
    for recipe in ("e6", "e12"):
        lms = {}
        for seed in SEEDS:
            by_fold = []
            for fold in range(FOLDS):
                d = json.loads((diag / "oof" / recipe / f"seed{seed}" / f"fold{fold}" / "logits.json").read_text(encoding="utf-8"))
                by_fold.append({tuple(k): v for k, v in zip(d["inputs"], d["logits"])})
            lms[seed] = by_fold
        for name, items in groups.items():
            full, pair = [], []
            for seed in SEEDS:
                lm_rows = [lms[seed][fold_of[fam_of[r["pair_id"]]]] for r, _ in items]
                full.append([sym_label(lm, r["text1"], r["text2"]) == r["label"] for lm, (r, _) in zip(lm_rows, items)])
                pair.append([sym_label(lm, wa, wb) == r["label"] for lm, (r, (wa, wb)) in zip(lm_rows, items)])
            full, pair = np.array(full, dtype=float), np.array(pair, dtype=float)
            diff = (pair - full).mean(axis=0)
            fidx = np.array([fam_idx[fam_of[r["pair_id"]]] for r, _ in items])
            boot = sorted(float((diff * w[fidx]).sum() / w[fidx].sum()) for w in draws)
            result[f"{recipe}/{name}"] = {
                "rows": len(items), "families": len({fam_of[r["pair_id"]] for r, _ in items}),
                "labels": dict(Counter(r["label"] for r, _ in items)),
                "agree_full": round(float(full.mean()), 4), "agree_pair": round(float(pair.mean()), 4),
                "delta": round(float(diff.mean()), 4),
                "interval_family_resampling": [round(boot[int(0.025 * (BOOTSTRAP - 1))], 4), round(boot[int(0.975 * (BOOTSTRAP - 1))], 4)]}
            print(recipe, name, result[f"{recipe}/{name}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
