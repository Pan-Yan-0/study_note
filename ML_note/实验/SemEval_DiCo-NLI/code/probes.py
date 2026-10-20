"""Shallow probes (logistic regression) on tracks 1-3, scored with the official scorer.

Usage:
    python3 probes.py <official_repo_dir> <out_dir>
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SWAP = [0, 2, 1, 3]


def tok(s):
    return re.findall(r"\w+", s.lower())


def surface(a, b):
    A, B = set(tok(a)), set(tok(b))
    i, la, lb = len(A & B), len(tok(a)), len(tok(b))
    return [la, lb, la - lb, abs(la - lb), i / max(1, len(A | B)), i / max(1, len(A)),
            i / max(1, len(B)), float(A <= B), float(B <= A), float(a.lower() == b.lower()), float(A == B)]


def read_rows(repo, split, track):
    path = repo / "final_data" / split / f"dico_nli_{split}_track{track}_reference.csv"
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score(repo, track, rows, labels, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = out_dir / "pred.csv"
    with open(pred, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "label"])
        w.writerows((r["instance_id"], l) for r, l in zip(rows, labels))
    gold = repo / "final_data" / "dev" / f"dico_nli_dev_track{track}_reference.csv"
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", str(gold),
                    "--predictions", str(pred), "--output-dir", str(out_dir)],
                   cwd=repo, check=True, capture_output=True)
    s = json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))
    return {k: round(s[k], 4) for k in ("weighted_f1", "soft_cons", "hard_cons")}


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)


def features(tr, dv):
    feats = {}
    feats["surface"] = tuple(np.array([surface(r["text1"], r["text2"]) for r in rs]) for rs in (tr, dv))
    for side in ("text1", "text2"):
        v = char_tfidf().fit([r[side] for r in tr])
        feats[f"only_{side}"] = tuple(v.transform([r[side] for r in rs]) for rs in (tr, dv))
    v = char_tfidf().fit([r["text1"] for r in tr] + [r["text2"] for r in tr])

    def pair(rs):
        s = csr_matrix(np.array([surface(r["text1"], r["text2"]) for r in rs]))
        return hstack([v.transform([r["text1"] for r in rs]), v.transform([r["text2"] for r in rs]), s]).tocsr()

    feats["bow_pair_surface"] = (pair(tr), pair(dv))
    return feats


def symmetrize(rows, probs):
    idx = {r["instance_id"]: k for k, r in enumerate(rows)}
    out = probs.copy()
    for k, r in enumerate(rows):
        j = idx.get(r.get("reverse_pair_id") or "")
        if j is not None:
            out[k] = (probs[k] + probs[j][SWAP]) / 2
    return out


def main():
    repo, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    results = []
    for track in (1, 2, 3):
        tr, dv = read_rows(repo, "train", track), read_rows(repo, "dev", track)
        y = [LABELS.index(r["label"]) for r in tr]
        for name, (xtr, xdv) in features(tr, dv).items():
            p = LogisticRegression(max_iter=3000).fit(xtr, y).predict_proba(xdv)
            for mode, q in (("plain", p), ("sym", symmetrize(dv, p))):
                s = score(repo, track, dv, [LABELS[i] for i in q.argmax(1)], out / f"track{track}" / name / mode)
                results.append({"track": track, "probe": name, "mode": mode, **s})
                print(results[-1])
    (out / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
