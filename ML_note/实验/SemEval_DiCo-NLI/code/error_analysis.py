"""Error analysis of the encoder runs: error types, edit shapes, length rule, word overlap.

Usage:
    python3 error_analysis.py <official_repo_dir> <encoder_metrics_dir> <out_json>
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SHORT = {"EQUIVALENCE": "EQ", "FORWARD_ENTAILMENT": "FE", "BACKWARD_ENTAILMENT": "BE", "NEGATIVE_OTHER": "NEG"}
MODELS, SEEDS = ("mbert", "mdeberta"), (42, 43, 44)


def tok(s):
    return re.findall(r"\w+", s.lower())


def shape(a, b):
    A, B = Counter(tok(a)), Counter(tok(b))
    da, db = sum((A - B).values()), sum((B - A).values())
    if not (A & B):
        return "disjoint"
    if da == 0 and db == 0:
        return "same_bag"
    if da == 0 or db == 0:
        return "insert"
    if da == 1 and db == 1:
        return "sub1"
    return "other"


def length_rule(a, b):
    return "FORWARD_ENTAILMENT" if len(tok(a)) > len(tok(b)) else "BACKWARD_ENTAILMENT"


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def preds(runs, model, track, mode):
    out = defaultdict(list)
    for s in SEEDS:
        for r in read_csv(runs / model / f"track{track}" / f"seed{s}" / f"dev_pred_{mode}.csv"):
            out[r["instance_id"]].append(r["label"])
    return out


def accuracy_by(rows, pred, key):
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        for p in pred[r["instance_id"]]:
            a = acc[key(r)]
            a[0] += 1
            a[1] += p == r["label"]
    return {k: {"rows": n // len(SEEDS), "acc": round(c / n, 3)} for k, (n, c) in sorted(acc.items())}


def data_profile(repo):
    neg = re.compile(r"\b(no|not|n't|never|without|none|nobody|nothing|neither|nor)\b", re.I)
    quant = re.compile(r"\b(all|every|everyone|some|someone|each|any|most|few|many|several)\b", re.I)
    prof = {}
    for split in ("train", "dev"):
        rows = [r for r in read_csv(repo / "final_data" / split / f"dico_nli_{split}_track1_reference.csv")
                if not r["instance_id"].endswith("flipped")]
        lens = [len(tok(r[s])) for r in rows for s in ("text1", "text2")]
        shapes = defaultdict(Counter)
        for r in rows:
            shapes[shape(r["text1"], r["text2"])][SHORT[r["label"]]] += 1
        ins = [r for r in rows if shape(r["text1"], r["text2"]) == "insert"]
        prof[split] = {
            "pairs": len(rows),
            "mean_words": round(sum(lens) / len(lens), 2),
            "share_le2_words": round(sum(l <= 2 for l in lens) / len(lens), 3),
            "pairs_with_negation": sum(bool(neg.search(r["text1"] + " " + r["text2"])) for r in rows),
            "pairs_with_quantifier": sum(bool(quant.search(r["text1"] + " " + r["text2"])) for r in rows),
            "shapes": {k: dict(v) for k, v in shapes.items()},
            "insert_length_rule_acc": round(sum(length_rule(r["text1"], r["text2"]) == r["label"] for r in ins) / len(ins), 3),
        }
    return prof


def main():
    repo, runs, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    result = {"data_track1": data_profile(repo), "tracks": {}}
    for track in (1, 2, 3, 4):
        rows = read_csv(repo / "final_data" / "dev" / f"dico_nli_dev_track{track}_reference.csv")
        orig = [r for r in rows if not r["instance_id"].endswith("flipped")]
        per = {}
        for m in MODELS:
            plain, sym = preds(runs, m, track, "plain"), preds(runs, m, track, "sym")
            entry = {"acc_by_shape_sym": accuracy_by(orig, sym, lambda r: shape(r["text1"], r["text2"]))}
            ins = [r for r in rows if shape(r["text1"], r["text2"]) == "insert"]
            entry["insert_plain"] = accuracy_by(
                ins, plain, lambda r: "rule_holds" if length_rule(r["text1"], r["text2"]) == r["label"] else "rule_fails")
            fails = [(p, length_rule(r["text1"], r["text2"])) for r in ins
                     if length_rule(r["text1"], r["text2"]) != r["label"] for p in plain[r["instance_id"]]]
            entry["rule_fails_follow_rule"] = round(sum(p == q for p, q in fails) / max(1, len(fails)), 3)
            errors = Counter()
            for r in orig:
                maj = Counter(sym[r["instance_id"]]).most_common(1)[0][0]
                if maj != r["label"]:
                    errors[f"{SHORT[r['label']]}->{SHORT[maj]}"] += 1
            entry["errors_majority_sym"] = dict(errors.most_common())
            per[m] = entry
        sym_all = {m: preds(runs, m, track, "sym") for m in MODELS}
        per["wrong_in_all_runs"] = sum(
            all(p != r["label"] for m in MODELS for p in sym_all[m][r["instance_id"]]) for r in orig)
        result["tracks"][f"track{track}"] = per
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["data_track1"], ensure_ascii=False))
    print(json.dumps(result["tracks"]["track1"], ensure_ascii=False))


if __name__ == "__main__":
    main()
