"""Slice oracle: set predictions inside one slice to gold, keep the rest, rescore with the official scorer.

It shows how much a slice can move the official metrics; it is not the gain of any real method.

Usage:
    python3 slice_oracle.py <official_repo_dir> <encoder_metrics_dir> <work_dir> <out_json>
"""
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from error_analysis import length_rule, read_csv, shape

SLICES = {
    "insert_rule_fails": lambda r: shape(r["text1"], r["text2"]) == "insert" and length_rule(r["text1"], r["text2"]) != r["label"],
    "sub1": lambda r: shape(r["text1"], r["text2"]) == "sub1",
}
METRICS = ("weighted_f1", "soft_cons", "hard_cons")


def score(repo, track, pred, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pred.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "label"])
        w.writerows(pred.items())
    gold = repo / "final_data" / "dev" / f"dico_nli_dev_track{track}_reference.csv"
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", str(gold), "--predictions", str(path),
                    "--output-dir", str(out_dir)], cwd=repo, check=True, capture_output=True)
    s = json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))
    return [s[m] for m in METRICS]


def main():
    repo, runs, work, out = (Path(a).resolve() for a in sys.argv[1:5])
    result = []
    for track in (1, 2, 3):
        gold = read_csv(repo / "final_data" / "dev" / f"dico_nli_dev_track{track}_reference.csv")
        for mode in ("plain", "sym"):
            for name, in_slice in SLICES.items():
                fix = {r["instance_id"]: r["label"] for r in gold if in_slice(r)}
                deltas = []
                for seed in (42, 43, 44):
                    pred = {r["instance_id"]: r["label"]
                            for r in read_csv(runs / "mdeberta" / f"track{track}" / f"seed{seed}" / f"dev_pred_{mode}.csv")}
                    tag = f"track{track}_{mode}_seed{seed}"
                    base = score(repo, track, pred, work / tag / "base")
                    oracle = score(repo, track, {**pred, **fix}, work / tag / name)
                    deltas.append([o - b for o, b in zip(oracle, base)])
                row = {"track": track, "mode": mode, "slice": name, "rows": len(fix),
                       **{f"delta_{m}": round(statistics.mean(d[i] for d in deltas), 4) for i, m in enumerate(METRICS)}}
                result.append(row)
                print(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
