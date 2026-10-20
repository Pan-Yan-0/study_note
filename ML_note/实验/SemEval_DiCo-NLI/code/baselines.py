"""random / majority baselines on the dev split, scored with the official scorer.

Usage:
    python3 baselines.py <official_repo_dir> <out_dir>
"""
import csv
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SEEDS = [2027, 2028, 2029]


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_predictions(path, ids, labels):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "label"])
        w.writerows(zip(ids, labels))


def score(repo, gold, pred, out_dir):
    subprocess.run(
        [sys.executable, "-m", "evaluation_functions", "--gold", str(gold),
         "--predictions", str(pred), "--output-dir", str(out_dir)],
        cwd=repo, check=True, capture_output=True,
    )
    s = json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))
    return {k: s[k] for k in ("weighted_f1", "soft_cons", "hard_cons")} if "weighted_f1" in s else s


def main():
    repo, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = repo / "final_data"
    summary = {}
    for track in [1, 2, 3, 4]:
        train = read_rows(data / f"train/dico_nli_train_track{track}_participant_labeled.csv")
        gold = data / f"dev/dico_nli_dev_track{track}_reference.csv"
        ids = [r["instance_id"] for r in read_rows(gold)]
        counts = Counter(r["label"] for r in train)
        majority = max(LABELS, key=lambda l: (counts[l], -LABELS.index(l)))  # 并列时取 LABELS 里靠前的

        systems = {"majority": [[majority] * len(ids)]}
        rngs = [random.Random(s * 10 + track) for s in SEEDS]
        systems["random"] = [[rng.choice(LABELS) for _ in ids] for rng in rngs]

        for name, runs in systems.items():
            scores = []
            for i, labels in enumerate(runs):
                pred = out / f"dev_track{track}_{name}_{i}.csv"
                write_predictions(pred, ids, labels)
                scores.append(score(repo, gold, pred, out / f"dev_track{track}_{name}_{i}_scores"))
            mean = {k: sum(s[k] for s in scores) / len(scores) for k in scores[0]}
            summary[f"track{track}/{name}"] = {"runs": scores, "mean": mean,
                                               **({"label": majority} if name == "majority" else {})}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in summary.items():
        print(k, {m: round(x, 4) for m, x in v["mean"].items()})


if __name__ == "__main__":
    main()
