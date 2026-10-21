"""Analysis of the third group (Qwen instruction prompting), from the saved results folder only; no model needed.

1. Rebuild every plain / sym prediction from cache.jsonl and rescore with the official scorer; must match done.json.
2. English training set, select subset: paired differences of symmetrized WF1 (few - zero, zero_rules - zero),
   approximate 95% interval by resampling families (2,000 draws). The prompt choice itself follows the rule fixed
   in the notebook; this only says how far apart the prompts are.
3. cot on dev: plain accuracy split by how generation ended (wrote "Answer:" / hit the 320-token limit / stopped).
   Descriptive only: harder items may also produce longer reasoning.
4. English dev: how often a gold BACKWARD row is predicted FORWARD (direction confusion), per prompt and mode.

Usage:
    python3 llm_instruct_analysis.py <results_dir> <official_repo_dir> <notebooks/dico_llm_instruct.py> <out_json>
"""
import csv
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict


def rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[(d["a"], d["b"])] = d
    return out


def main(res, dico, nb_path, out_path):
    spec = importlib.util.spec_from_file_location("nb", nb_path)
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    nb.DICO_DIR = dico
    labels, swap = nb.LABELS, nb.SWAP
    out = {"recheck_mismatches": 0}

    # 1. 从缓存重建预测并重新评分
    for prompt in nb.PROMPTS:
        for split, track in [("train", 1)] + [("dev", t) for t in nb.TRACKS]:
            d = os.path.join(res, prompt, f"{split}_track{track}")
            if not os.path.isdir(d):
                continue
            cache, done = jsonl(os.path.join(d, "cache.jsonl")), json.load(open(os.path.join(d, "done.json")))
            for sub in (["all"] if split == "dev" else ["select", "rule_source"]):
                ref = (nb.data_file(track, "dev", "reference") if split == "dev"
                       else os.path.join(d, f"reference_{sub}.csv"))
                for mode in ("plain", "sym"):
                    pred = {}
                    for r in rows(ref):
                        p = cache[(r["text1"], r["text2"])]["p"]
                        if mode == "sym":
                            p = nb.sym(p, cache[(r["text2"], r["text1"])]["p"])
                        pred[r["instance_id"]] = nb.argmax_label(p)
                    saved = {r["instance_id"]: r["label"] for r in rows(os.path.join(d, f"pred_{sub}_{mode}.csv"))}
                    with tempfile.TemporaryDirectory() as tmp:
                        pf = os.path.join(tmp, "pred.csv")
                        with open(pf, "w", encoding="utf-8", newline="") as f:
                            w = csv.writer(f)
                            w.writerow(["instance_id", "label"])
                            w.writerows(pred.items())
                        subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", os.path.abspath(ref),
                                        "--predictions", pf, "--output-dir", tmp], cwd=dico, check=True,
                                       capture_output=True)
                        sc = json.load(open(os.path.join(tmp, "scores.json")))
                    old = done["scores"][sub][mode]
                    if pred != saved or any(abs(sc[k] - old[k]) > 1e-9 for k in old):
                        out["recheck_mismatches"] += 1

    # 2. 训练集 select 子集上的配对差值
    d = os.path.join(res, "zero", "train_track1")
    ref = rows(os.path.join(d, "reference_select.csv"))
    fam = nb.train_families()
    by_fam = defaultdict(list)
    for i, r in enumerate(ref):
        by_fam[fam[r["pair_id"]]].append(i)
    fams = sorted(by_fam)
    gold = [r["label"] for r in ref]
    pred = {p: [x["label"] for x in rows(os.path.join(res, p, "train_track1", "pred_select_sym.csv"))]
            for p in nb.TRAIN_PROMPTS}
    for p in nb.TRAIN_PROMPTS:  # 预测文件和 reference 的行顺序相同
        assert [x["instance_id"] for x in rows(os.path.join(res, p, "train_track1", "pred_select_sym.csv"))] == \
               [r["instance_id"] for r in ref]
    wf1 = lambda idx, p: nb.weighted_f1([gold[i] for i in idx], [pred[p][i] for i in idx])
    point = {p: wf1(range(len(ref)), p) for p in nb.TRAIN_PROMPTS}
    rng, draws = random.Random(2027), defaultdict(list)
    for _ in range(2000):
        idx = [i for f in (rng.choice(fams) for _ in fams) for i in by_fam[f]]
        z = wf1(idx, "zero")
        for p in ("few", "zero_rules"):
            draws[p].append(wf1(idx, p) - z)
    out["select_sym_wf1"] = {p: round(v, 4) for p, v in point.items()}
    out["select_paired_vs_zero"] = {
        p: {"delta": round(point[p] - point["zero"], 4),
            "family_bootstrap_95": [round(sorted(draws[p])[50], 4), round(sorted(draws[p])[1949], 4)]}
        for p in ("few", "zero_rules")}
    out["select_families"], out["select_rows"] = len(fams), len(ref)

    # 3. cot：按生成结束的原因分开的原方向准确率
    out["cot_accuracy_by_end"] = {}
    for t in nb.TRACKS:
        d = os.path.join(res, "cot", f"dev_track{t}")
        gen, cache = jsonl(os.path.join(d, "gen.jsonl")), jsonl(os.path.join(d, "cache.jsonl"))
        acc = defaultdict(lambda: [0, 0])
        for r in rows(nb.data_file(t, "dev", "participant_labeled")):
            k = (r["text1"], r["text2"])
            e = gen[k]["end"]
            acc[e][0] += nb.argmax_label(cache[k]["p"]) == r["label"]
            acc[e][1] += 1
        out["cot_accuracy_by_end"][f"track{t}"] = {e: {"correct": a, "rows": n, "accuracy": round(a / n, 3)}
                                                   for e, (a, n) in acc.items()}

    # 4. 英语 dev：金标 BACKWARD 被判成 FORWARD 的行数
    out["en_dev_gold_backward"] = {}
    for p in nb.PROMPTS:
        for mode in ("plain", "sym"):
            sc = json.load(open(os.path.join(res, p, "dev_track1", f"scores_all_{mode}", "scores.json")))
            row = sc["classification"]["confusion_matrix"]["BACKWARD_ENTAILMENT"]
            out["en_dev_gold_backward"][f"{p}_{mode}"] = {k: row[k] for k in labels}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
