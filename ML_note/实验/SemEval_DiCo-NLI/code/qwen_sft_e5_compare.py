"""第四组"最多 5 轮"和"3 轮"配方的比较（英语 fold 0，同种子配对）。

1. 从 5 轮结果的缓存重建预测、和保存的预测比较，用官方脚本重新评分；
2. 零样本在两次运行之间是否逐条相同（检查平台差异）；
3. 同一种子 sym WF1 的差（5 轮 − 3 轮），种子平均，按家族配对重抽样 2,000 次的近似 95% 区间（每次所有种子用同一批家族）。

用法：python3 qwen_sft_e5_compare.py <5 轮结果目录> <3 轮结果目录> <官方仓库> <第二组 metrics/diagnostics>
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_sft_analysis import rows, jsonl, predict, official, weighted_f1, METRICS
new, old, dico, diag = sys.argv[1:5]
fam = json.load(open(os.path.join(diag, "families.json")))
train_ref = rows(os.path.join(dico, "final_data/train/dico_nli_train_track1_reference.csv"))
dev_ref = rows(os.path.join(dico, "final_data/dev/dico_nli_dev_track1_reference.csv"))
fold0 = [r for r in train_ref if fam["fold_of_family"][fam["family_of_pair"][r["pair_id"]]] == 0]
assert len(fold0) == 608
parts = {"fold0": fold0, "dev": dev_ref}
SEEDS = sorted(n[4:] for n in os.listdir(os.path.join(new, "track1")) if n.startswith("seed")
               and os.path.exists(os.path.join(new, "track1", n, "done.json")))
mism = 0
for run in ["zero_shot"] + [f"seed{s}" for s in SEEDS]:
    d = os.path.join(new, "track1", run)
    P = jsonl(os.path.join(d, "cache.jsonl"))
    saved = json.load(open(os.path.join(d, "done.json")))["scores"]
    for name, rs in parts.items():
        for mode in ("plain", "sym"):
            lab = predict(P, rs, mode)
            fp = {r["instance_id"]: r["label"] for r in rows(os.path.join(d, f"pred_{name}_{mode}.csv"))}
            mism += sum(fp[r["instance_id"]] != l for r, l in zip(rs, lab))
            sc = official(dico, rs, lab)
            assert all(abs(sc[k] - saved[name][mode][k]) < 1e-12 for k in METRICS), (run, name, mode)
print("种子：", SEEDS, "；重建预测和保存的不同：", mism, "；官方脚本重算一致")
# zero-shot 跨平台是否一致
zn = jsonl(os.path.join(new, "track1/zero_shot/cache.jsonl")); zo = jsonl(os.path.join(old, "track1/zero_shot/cache.jsonl"))
common = [k for k in zn if k in zo]
print("零样本：共同输入", len(common), "，概率最大差", max(abs(a - b) for k in common for a, b in zip(zn[k], zo[k])),
      "，argmax 不同", sum(max(range(4), key=lambda i: zn[k][i]) != max(range(4), key=lambda i: zo[k][i]) for k in common))
gold = [r["label"] for r in fold0]
by = {}
for i, r in enumerate(fold0):
    by.setdefault(fam["family_of_pair"][r["pair_id"]], []).append(i)
fams = sorted(by)
out = {"seeds": SEEDS, "per_seed": {}}
lab = {}
for s in SEEDS:
    P5 = jsonl(os.path.join(new, f"track1/seed{s}/cache.jsonl")); P3 = jsonl(os.path.join(old, f"track1/seed{s}/cache.jsonl"))
    e5 = json.load(open(os.path.join(new, f"track1/seed{s}/done.json")))["best_epoch"]
    e3 = json.load(open(os.path.join(old, f"track1/seed{s}/done.json")))["best_epoch"]
    rec = {"best_epoch_5_vs_3": [e5, e3]}
    for mode in ("plain", "sym"):
        l5, l3 = predict(P5, fold0, mode), predict(P3, fold0, mode)
        s5, s3 = official(dico, fold0, l5), official(dico, fold0, l3)
        rec[mode] = {k: (round(s5[k], 4), round(s3[k], 4), round(s5[k] - s3[k], 4)) for k in METRICS}
        rec[mode]["same_pred"] = round(sum(a == b for a, b in zip(l5, l3)) / len(l5), 4)
        if mode == "sym":
            lab[s] = (l5, l3)
    a, b = official(dico, dev_ref, predict(P5, dev_ref, "sym")), official(dico, dev_ref, predict(P3, dev_ref, "sym"))
    rec["dev_sym"] = {k: (round(a[k], 4), round(b[k], 4)) for k in METRICS}
    out["per_seed"][s] = rec


def mean_diff(idx):  # 同种子 sym WF1 的差（5 轮 − 3 轮），种子平均
    g = [gold[i] for i in idx]
    return sum(weighted_f1(g, [l5[i] for i in idx]) - weighted_f1(g, [l3[i] for i in idx]) for l5, l3 in lab.values()) / len(lab)


rng = random.Random(20261021)  # 和 registry.py、qwen_sft_analysis.py 同一写法：每次所有种子用同一批家族
diffs = sorted(mean_diff([i for f in rng.choices(fams, k=len(fams)) for i in by[f]]) for _ in range(2000))
out["sym_wf1_mean_diff"] = round(mean_diff(list(range(len(fold0)))), 4)
out["sym_wf1_mean_diff_ci"] = (round(diffs[int(.025 * 2000)], 4), round(diffs[int(.975 * 2000) - 1], 4))
print(json.dumps(out, ensure_ascii=False, indent=1))
