"""第四组（Qwen3-4B LoRA SFT）的复算和分析。

1. 从每个种子的 cache.jsonl 重建 plain / sym 预测，和保存的预测文件逐条比较，并用官方脚本重新评分。
2. 英语 fold 0 上和 mDeBERTa 12 轮配方（第二组折外模型，同一批训练 / 验证 / 留出数据）按种子配对比较，
   算事先定好的两条蒸馏门槛：
   - 门槛 1：每个种子 sym WF1 的差（Qwen − mDeBERTa），三个种子平均；
   - 门槛 2：同一种子、同一有序输入，两个模型的四类概率按 0.5 / 0.5 平均，再对称化、取最大，
     "混合 − mDeBERTa"按种子求差再平均。
   区间：按家族配对重抽样 2,000 次，每次抽同一批家族，三个种子都在这批家族上重算差值再平均。
3. 每类 F1、方向混淆。

用法：python3 qwen_sft_analysis.py <结果目录 dico_qwen_sft_results> <官方仓库> <第二组 metrics/diagnostics> <输出 json>
"""
import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SWAP = [0, 2, 1, 3]
SEEDS = [42, 43, 44]
METRICS = ["weighted_f1", "soft_cons", "hard_cons"]
B, BOOT_SEED = 2000, 20261021


def rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jsonl(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        out[(d["a"], d["b"])] = d["p"]
    return out


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [x / sum(e) for x in e]


def top(p):
    return LABELS[max(range(4), key=lambda i: p[i])]


def sym(p_ab, p_ba):
    return [(p_ab[i] + p_ba[SWAP[i]]) / 2 for i in range(4)]


def weighted_f1(gold, pred):
    total, out = len(gold), 0.0
    for l in LABELS:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        out += (2 * tp / (2 * tp + fp + fn) if tp else 0.0) * sum(g == l for g in gold) / total
    return out


def class_f1(gold, pred):
    out = {}
    for l in LABELS:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        out[l] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return out


def official(dico, ref_rows, labels):
    d = tempfile.mkdtemp()
    with open(f"{d}/ref.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["instance_id", "pair_id", "reverse_pair_id", "label"])
        w.writeheader()
        for r in ref_rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    with open(f"{d}/pred.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "label"])
        for r, l in zip(ref_rows, labels):
            w.writerow([r["instance_id"], l])
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", f"{d}/ref.csv", "--predictions",
                    f"{d}/pred.csv", "--output-dir", f"{d}/out"], cwd=dico, check=True, capture_output=True)
    s = json.load(open(f"{d}/out/scores.json"))
    return {k: s[k] for k in METRICS}


def predict(P, rs, mode):
    return [top(P[(r["text1"], r["text2"])] if mode == "plain" else sym(P[(r["text1"], r["text2"])], P[(r["text2"], r["text1"])]))
            for r in rs]


def main(res, dico, diag, out_path):
    fam = json.load(open(os.path.join(diag, "families.json")))
    train_ref = rows(os.path.join(dico, "final_data/train/dico_nli_train_track1_reference.csv"))
    dev_ref = rows(os.path.join(dico, "final_data/dev/dico_nli_dev_track1_reference.csv"))
    fold0 = [r for r in train_ref if fam["fold_of_family"][fam["family_of_pair"][r["pair_id"]]] == 0]
    assert len(fold0) == 608
    parts = {"fold0": fold0, "dev": dev_ref}
    out = {"rescore": {}, "mismatches": 0}

    # 1. 从缓存重建预测、重新评分
    qwen = {}
    for run in ["zero_shot"] + [f"seed{s}" for s in SEEDS]:
        d = os.path.join(res, "track1", run)
        P = jsonl(os.path.join(d, "cache.jsonl"))
        qwen[run] = P
        saved = json.load(open(os.path.join(d, "done.json")))["scores"]
        for name, rs in parts.items():
            for mode in ("plain", "sym"):
                labels = predict(P, rs, mode)
                file_pred = {r["instance_id"]: r["label"] for r in rows(os.path.join(d, f"pred_{name}_{mode}.csv"))}
                out["mismatches"] += sum(file_pred[r["instance_id"]] != l for r, l in zip(rs, labels))
                sc = official(dico, rs, labels)
                assert all(abs(sc[k] - saved[name][mode][k]) < 1e-12 for k in METRICS), (run, name, mode)
                out["rescore"].setdefault(run, {}).setdefault(name, {})[mode] = sc

    # 2. mDeBERTa e12 的 fold 0（第二组折外模型）
    deb = {}
    for s in SEEDS:
        d = json.load(open(os.path.join(diag, f"oof/e12/seed{s}/fold0/logits.json")))
        deb[s] = {tuple(x): softmax(z) for x, z in zip(d["inputs"], d["logits"])}
    mix = {s: {k: [(a + b) / 2 for a, b in zip(qwen[f"seed{s}"][k], deb[s][k])] for k in qwen[f"seed{s}"] if k in deb[s]} for s in SEEDS}
    systems = {"qwen": {s: qwen[f"seed{s}"] for s in SEEDS}, "mdeberta": deb, "mix": mix}
    gold = [r["label"] for r in fold0]
    labels = {name: {s: {m: predict(P[s], fold0, m) for m in ("plain", "sym")} for s in SEEDS} for name, P in systems.items()}
    out["fold0"] = {name: {str(s): {m: official(dico, fold0, labels[name][s][m]) for m in ("plain", "sym")} for s in SEEDS}
                    for name in systems}

    by_fam = {}
    for i, r in enumerate(fold0):
        by_fam.setdefault(fam["family_of_pair"][r["pair_id"]], []).append(i)
    fams = sorted(by_fam)
    rng = random.Random(BOOT_SEED)
    draws = [[i for f in rng.choices(fams, k=len(fams)) for i in by_fam[f]] for _ in range(B)]

    def gate(a, b):
        per_seed = {s: weighted_f1(gold, labels[a][s]["sym"]) - weighted_f1(gold, labels[b][s]["sym"]) for s in SEEDS}
        mean = sum(per_seed.values()) / len(SEEDS)
        boot = []
        for idx in draws:
            g = [gold[i] for i in idx]
            boot.append(sum(weighted_f1(g, [labels[a][s]["sym"][i] for i in idx]) - weighted_f1(g, [labels[b][s]["sym"][i] for i in idx])
                            for s in SEEDS) / len(SEEDS))
        boot.sort()
        return {"per_seed": {str(s): round(v, 5) for s, v in per_seed.items()}, "mean": round(mean, 5),
                "ci95": [round(boot[int(0.025 * B)], 5), round(boot[int(0.975 * B) - 1], 5)]}

    out["gates"] = {"gate1_qwen_minus_mdeberta": gate("qwen", "mdeberta"), "gate2_mix_minus_mdeberta": gate("mix", "mdeberta"),
                    "mix_minus_qwen": gate("mix", "qwen"),
                    "n_families": len(fams), "bootstrap": {"draws": B, "seed": BOOT_SEED}}

    # 3. 每类 F1（sym，三个种子平均）和方向混淆（plain，金标 B 判成 F）
    out["class_f1_sym_mean"] = {name: {l: round(sum(class_f1(gold, labels[name][s]["sym"])[l] for s in SEEDS) / 3, 4) for l in LABELS}
                                for name in systems}
    out["confusion_sym_seed_sum"] = {name: {f"{g}->{p}": n for (g, p), n in sorted(Counter(
        (g, p) for s in SEEDS for g, p in zip(gold, labels[name][s]["sym"]) if g != p).items(), key=lambda x: -x[1])[:8]}
        for name in ("qwen", "mdeberta")}
    out["backward_to_forward_plain_per_seed"] = {name: [sum(g == LABELS[2] and p == LABELS[1] for g, p in zip(gold, labels[name][s]["plain"]))
                                                       for s in SEEDS] for name in ("qwen", "mdeberta")}
    out["gold_backward_fold0"] = sum(g == LABELS[2] for g in gold)
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("mismatches", "gates", "class_f1_sym_mean", "backward_to_forward_plain_per_seed",
                                          "confusion_sym_seed_sum")}, ensure_ascii=False, indent=1))
    for name in systems:
        print(name, {s: {m: " / ".join(f"{v[k]:.3f}" for k in METRICS) for m, v in d.items()} for s, d in out["fold0"][name].items()})


if __name__ == "__main__":
    main(*sys.argv[1:5])
