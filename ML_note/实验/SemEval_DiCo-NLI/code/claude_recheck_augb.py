"""Claude 对实验 B +1（B0 / B1-Haiku / step+1，英语五折三种子，Kaggle T4 fp16）的独立复核。

不复用训练脚本的汇总代码，只读每个任务保存的文件：
1. 完整性：45 个任务都在；done.json 记录的预测哈希和文件一致；task.json 里的代码、模型、数据、环境三组相同，
   只有组名、训练行数和计划步数不同；每折留出 / 内层验证和仓库的家族五折（families.json）、split_manifest 一致。
2. 训练记录：每个任务实际更新 + fp16 跳步 = 计划步数；选中轮 = 内层验证原方向 WF1 最高、同分取早的那一轮；
   保存的内层验证预测重算的 WF1 等于记录的选轮分数；B1-Haiku 和 step+1 每折计划步数相同。
3. 预测：从四类概率重新取 argmax；反方向概率 r_p_* 等于反方向那一行的 p_*；对称化 = (p + r 交换 F/B) / 2。
4. 分数：拼成 3,042 行折外预测，用官方评分模块算原方向和对称化的 WF1 / SoftCons / HardCons。
5. 比较：同种子 sym WF1 差的均值，按家族配对重抽样（和 registry.py 同样的约定：2,000 次、种子 20261021、
   各种子各组共用同一批家族）。判断规则（冻结计划第 1 节）：均值 > 0 且区间下限 > 0。
6. 次指标（只报告）：结构规则下的分数、三种子概率平均后的分数、逐类 F1、预测分布。

用法：python3 -I code/claude_recheck_augb.py <官方仓库目录>
输出：metrics/augment_b/train_plus1/oof/、claude_recheck.json
"""
import csv, hashlib, json, os, random, subprocess, sys, tempfile
from collections import Counter

OFF = os.path.abspath(sys.argv[1])
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{SE}/metrics/augment_b/train_plus1"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
SWAP = [0, 2, 1, 3]
SEEDS, FOLDS = [42, 43, 44], [0, 1, 2, 3, 4]
GROUPS = {"B0": "B0", "B1-Haiku": "B1-Haiku", "step+1": "stepplus1"}
B, BOOT_SEED = 2000, 20261021
problems = []


def check(cond, msg):
    if not cond:
        problems.append(msg)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rj(p):
    return json.load(open(p, encoding="utf-8"))


def rcsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def wf1(gold, pred):
    n, out = len(gold), 0.0
    for l in LABELS:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        out += (2 * tp / (2 * tp + fp + fn) if tp else 0.0) * sum(g == l for g in gold) / n
    return out


def argmax(v):
    return LABELS[max(range(4), key=lambda k: v[k])]


# ---------- 官方数据和家族五折 ----------
ref_rows = rcsv(f"{OFF}/final_data/train/dico_nli_train_track1_reference.csv")
gold = {r["instance_id"]: r["label"] for r in ref_rows}
reverse_of = {r["instance_id"]: r["reverse_pair_id"] for r in ref_rows}
inst_of_pair = {r["pair_id"]: r["instance_id"] for r in ref_rows}
fam = rj(f"{SE}/metrics/diagnostics/families.json")
pair_of = {r["instance_id"]: r["pair_id"] for r in ref_rows}
family = {i: fam["family_of_pair"][pair_of[i]] for i in gold}
fold_of = {i: int(fam["fold_of_family"][str(family[i])]) for i in gold}
split = rj(f"{R}/input_data/split_manifest.json")["folds"]
check(sha(f"{R}/input_data/split_manifest.json").startswith("f9363290"), "split_manifest 哈希不对")
for k in FOLDS:
    check(set(split[str(k)]["test"]) == {i for i in gold if fold_of[i] == k}, f"fold {k} 留出和 families.json 不同")
    check(not (set(split[str(k)]["val"]) & set(split[str(k)]["test"])), f"fold {k} 验证和留出重叠")
    fams_fit = {family[i] for i in split[str(k)]["fit"]}
    check(not (fams_fit & {family[i] for i in split[str(k)]["val"] + split[str(k)]["test"]}), f"fold {k} 家族泄漏")
check(sorted(sum((split[str(k)]["test"] for k in FOLDS), [])) == sorted(gold), "五折留出拼起来不是 3,042 行")
a1 = rj(f"{SE}/metrics/qwen8b_a1/formal/seed42/fold0/task.json")["split_ids"]
check(set(a1["fit"]) == set(split["0"]["fit"]) and set(a1["val"]) == set(split["0"]["val"]), "fold 0 划分和实验 A 不同")

# ---------- 逐任务核对 ----------
base_task, tasks, oof, oof_prob, updates = None, {}, {}, {}, []
for g, d in GROUPS.items():
    for s in SEEDS:
        plain, sym, prob = {}, {}, {}
        for k in FOLDS:
            td = f"{R}/tasks/{d}/seed{s}/fold{k}"
            t, done = rj(f"{td}/task.json"), rj(f"{td}/done.json")
            tasks[(g, s, k)] = (t, done)
            core = {x: t[x] for x in t if x not in ("group", "train_rows", "steps", "seed", "fold", "val_rows", "test_rows")}
            if base_task is None:
                base_task = core
            check(core == base_task, f"{g} {s} {k} 的 task.json 和其他任务不只差组、行数、步数")
            check(t["group"] == g and t["seed"] == s and t["fold"] == k and not t["smoke"], f"{td} 身份不对")
            check(done.get("status") == "pass", f"{td} 状态 {done.get('status')}")
            check(t["val_rows"] == len(split[str(k)]["val"]) and t["test_rows"] == len(split[str(k)]["test"]), f"{td} 行数不对")
            check(sha(f"{td}/predictions.csv") == done["predictions_sha256"], f"{td} predictions 哈希不对")
            check(sha(f"{td}/val_predictions.csv") == done["val_predictions_sha256"], f"{td} val 哈希不对")
            check(done["actual_optimizer_updates"] + done["skipped_fp16_updates"] == done["planned_updates"] == t["steps"],
                  f"{td} 更新 + 跳步 ≠ 计划")
            ep = done["epochs"]
            check(len(ep) == 12, f"{td} 轮数 {len(ep)}")
            vals = [e["val_original_wf1"] for e in ep]
            check(done["best_epoch"] == vals.index(max(vals)) + 1, f"{td} 选轮不是最早的最高分")
            check(abs(done["best_val_original_wf1"] - max(vals)) < 1e-12, f"{td} 选轮分数不一致")
            vrows = rcsv(f"{td}/val_predictions.csv")
            check([r["instance_id"] for r in vrows] == split[str(k)]["val"] or
                  sorted(r["instance_id"] for r in vrows) == sorted(split[str(k)]["val"]), f"{td} 内层验证行不对")
            vp = [argmax([float(r["p_" + c]) for c in "EFBO"]) for r in vrows]
            check(all(r["pred"] == p for r, p in zip(vrows, vp)), f"{td} 内层验证 pred 和概率不一致")
            vw = wf1([gold[r["instance_id"]] for r in vrows], vp)
            check(abs(vw - done["best_val_original_wf1"]) < 1e-9, f"{td} 内层验证重算 {vw} ≠ 记录")
            updates.append({"group": g, "seed": s, "fold": k, "planned": done["planned_updates"],
                            "actual": done["actual_optimizer_updates"], "skipped": done["skipped_fp16_updates"],
                            "best_epoch": done["best_epoch"], "train_rows": t["train_rows"], "runtime_sec": done["runtime_sec"]})
            rows = rcsv(f"{td}/predictions.csv")
            byid = {r["instance_id"]: r for r in rows}
            check(sorted(byid) == sorted(split[str(k)]["test"]), f"{td} 留出行不对")
            for r in rows:
                i = r["instance_id"]
                check(r["label"] == gold[i], f"{i} 金标不对")
                p = [float(r["p_" + c]) for c in "EFBO"]
                q = [float(r["r_p_" + c]) for c in "EFBO"]
                rev = inst_of_pair.get(reverse_of[i]) if reverse_of[i] else None
                if rev:
                    check(rev in byid, f"{i} 反方向不在同一折")
                    check(max(abs(a - float(byid[rev]["p_" + c])) for a, c in zip(q, "EFBO")) < 1e-9, f"{i} r_p 不等于反方向的 p")
                sp = [(p[j] + q[SWAP[j]]) / 2 for j in range(4)]
                check(max(abs(a - float(r["sym_p_" + c])) for a, c in zip(sp, "EFBO")) < 2e-9, f"{i} sym 概率不对")
                plain[i], sym[i], prob[i] = argmax(p), argmax(sp), sp
                check(r["plain"] == plain[i] and r["sym"] == sym[i], f"{i} 保存的标签和概率不一致")
        oof[(g, s)] = {"plain": plain, "sym": sym}
        oof_prob[(g, s)] = prob
for s in SEEDS:
    for k in FOLDS:
        a, b = tasks[("B1-Haiku", s, k)][0], tasks[("step+1", s, k)][0]
        check(a["train_rows"] == b["train_rows"] == 2 * tasks[("B0", s, k)][0]["train_rows"], f"{s} {k} 行数不对")
        check(a["steps"] == b["steps"], f"{s} {k} B1 和 step+1 计划步数不同")


# ---------- 官方评分 ----------
def official(pred):
    d = tempfile.mkdtemp()
    with open(f"{d}/g.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["instance_id", "pair_id", "reverse_pair_id", "label"])
        for r in ref_rows: w.writerow([r["instance_id"], r["pair_id"], r["reverse_pair_id"], r["label"]])
    with open(f"{d}/p.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["instance_id", "label"])
        for r in ref_rows: w.writerow([r["instance_id"], pred[r["instance_id"]]])
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", f"{d}/g.csv", "--predictions", f"{d}/p.csv",
                    "--output-dir", f"{d}/o"], cwd=OFF, check=True, capture_output=True)
    sc = rj(f"{d}/o/scores.json")
    c = sc["classification"]
    return {"wf1": sc["weighted_f1"], "softcons": sc["soft_cons"], "hardcons": sc["hard_cons"],
            "per_class_f1": {l: c["per_label"][l]["f1"] for l in LABELS}, "_raw": sc}


os.makedirs(f"{R}/oof", exist_ok=True)
scores = {}
for (g, s), m in oof.items():
    scores.setdefault(g, {})[str(s)] = {}
    for mode in ("plain", "sym"):
        with open(f"{R}/oof/{GROUPS[g]}_seed{s}_{mode}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["instance_id", "label"])
            for r in ref_rows: w.writerow([r["instance_id"], m[mode][r["instance_id"]]])
        o = official(m[mode])
        check(abs(o["wf1"] - wf1([gold[i] for i in gold], [m[mode][i] for i in gold])) < 1e-12, f"{g} {s} {mode} WF1 自算不一致")
        o.pop("_raw")
        scores[g][str(s)][mode] = o


# ---------- 次指标 ----------
pair_has_reverse = {r["instance_id"]: bool(r["reverse_pair_id"]) for r in ref_rows}


def structure(prob):
    out = {}
    for i, p in prob.items():
        out[i] = "NEGATIVE_OTHER" if not pair_has_reverse[i] else LABELS[max(range(3), key=lambda k: p[k])]
    return out


secondary = {}
for g in GROUPS:
    avg = {i: [sum(oof_prob[(g, s)][i][j] for s in SEEDS) / 3 for j in range(4)] for i in gold}
    secondary[g] = {"seed_average_sym": official({i: argmax(v) for i, v in avg.items()}),
                    "structure_rule_sym_mean_wf1": sum(wf1([gold[i] for i in gold], [structure(oof_prob[(g, s)])[i] for i in gold]) for s in SEEDS) / 3,
                    "prediction_distribution_sym": {str(s): dict(Counter(oof[(g, s)]["sym"].values())) for s in SEEDS}}
    secondary[g]["seed_average_sym"].pop("_raw")


# ---------- 配对比较 ----------
fams = sorted({family[i] for i in gold})
by_fam = {}
for i in gold:
    by_fam.setdefault(family[i], []).append(i)


def compare(a, b, mode="sym"):
    import numpy as np
    ids = sorted(gold)
    per_seed = {str(s): wf1([gold[i] for i in ids], [oof[(a, s)][mode][i] for i in ids]) -
                        wf1([gold[i] for i in ids], [oof[(b, s)][mode][i] for i in ids]) for s in SEEDS}
    counts = np.zeros((len(fams), len(SEEDS), 2, 4, 4), dtype=np.int64)
    for fi, f in enumerate(fams):
        for si, s in enumerate(SEEDS):
            for i in by_fam[f]:
                gi = L2I[gold[i]]
                counts[fi, si, 0, gi, L2I[oof[(a, s)][mode][i]]] += 1
                counts[fi, si, 1, gi, L2I[oof[(b, s)][mode][i]]] += 1
    idx = {f: n for n, f in enumerate(fams)}
    rng, boot = random.Random(BOOT_SEED), []
    for _ in range(B):
        wts = np.bincount([idx[f] for f in rng.choices(fams, k=len(fams))], minlength=len(fams))
        cm = np.tensordot(wts, counts, axes=(0, 0))
        sup, prd = cm.sum(-1), cm.sum(-2)
        tp = np.diagonal(cm, axis1=-2, axis2=-1)
        den = sup + prd
        f1 = np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den != 0)
        w = (f1 * sup).sum(-1) / sup.sum(-1)
        boot.append(float((w[:, 0] - w[:, 1]).mean()))
    boot.sort()
    mean = sum(per_seed.values()) / len(SEEDS)
    ci = [boot[int(.025 * B)], boot[int(.975 * B) - 1]]
    return {"per_seed": per_seed, "mean": mean, "ci95": ci, "improvement": mean > 0 and ci[0] > 0,
            "families": len(fams), "rows": len(ids)}


comparisons = {"B1-Haiku_minus_B0": compare("B1-Haiku", "B0"), "B1-Haiku_minus_step+1": compare("B1-Haiku", "step+1"),
               "step+1_minus_B0（描述）": compare("step+1", "B0"),
               "B1-Haiku_minus_B0_plain（描述）": compare("B1-Haiku", "B0", "plain"),
               "B1-Haiku_minus_step+1_plain（描述）": compare("B1-Haiku", "step+1", "plain")}
agree = {f"{a}_vs_{b}_seed{s}": sum(oof[(a, s)]["sym"][i] == oof[(b, s)]["sym"][i] for i in gold) / len(gold)
         for a, b in [("B1-Haiku", "B0"), ("B1-Haiku", "step+1"), ("step+1", "B0")] for s in SEEDS}

# ---------- 和训练脚本的汇总对照 ----------
gs = rj(f"{R}/summary.json")
vs_gpt = {}
for g in GROUPS:
    for s in SEEDS:
        for mode in ("plain", "sym"):
            mine = scores[g][str(s)][mode]["wf1"]
            theirs = gs["seed_by_group"][str(s)][g][f"weighted_f1_{mode}"]
            check(abs(mine - theirs) < 1e-12, f"{g} {s} {mode} 和训练脚本汇总不同：{mine} 对 {theirs}")
for key, gk in [("B1-Haiku_minus_B0", "B1_minus_B0"), ("B1-Haiku_minus_step+1", "B1_minus_step1")]:
    check(abs(comparisons[key]["mean"] - gs["paired_comparison"][gk]["mean_difference"]) < 1e-12, f"{key} 均值和训练脚本不同")
    vs_gpt[key] = {"claude_ci95": comparisons[key]["ci95"], "train_script_ci95": gs["paired_comparison"][gk]["family_bootstrap_95pct_interval"],
                   "note": "两边重抽样的随机种子不同（registry 约定 20261021 对训练脚本 20261106），区间端点差一点是预期的"}

mean_scores = {g: {mode: {m: sum(scores[g][str(s)][mode][m] for s in SEEDS) / 3 for m in ("wf1", "softcons", "hardcons")}
                   for mode in ("plain", "sym")} for g in GROUPS}
out = {"problems": problems, "tasks": len(tasks), "scores_by_seed": scores, "scores_mean": mean_scores,
       "comparisons": comparisons, "prediction_agreement_sym": agree, "secondary": secondary,
       "updates": updates, "vs_train_script": vs_gpt,
       "rule": "均值 > 0 且家族重抽样区间下限 > 0（冻结计划第 1 节）；区间不含重新训练随机性"}
json.dump(out, open(f"{R}/claude_recheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("问题：", problems[:20] if problems else "无")
for g in GROUPS:
    print(g, {mode: {m: round(v * 100, 2) for m, v in mean_scores[g][mode].items()} for mode in ("plain", "sym")})
for k, v in comparisons.items():
    print(k, {s: round(x * 100, 2) for s, x in v["per_seed"].items()}, round(v["mean"] * 100, 2),
          [round(x * 100, 2) for x in v["ci95"]], "提升" if v["improvement"] else "不算提升")
print("和训练脚本区间对照：", vs_gpt)
