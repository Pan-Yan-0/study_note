"""Claude 对实验 B Commit 3（Focal / Dice loss 对照，英语五折三种子，Kaggle T4 fp16，30 项）的独立复核。

不复用训练脚本的代码，只读每个任务保存的文件：
1. B0 复用：v4 的 B0 SMOKE 和 Commit 1 归档的 B0 SMOKE 三个预测文件逐字节相同，done.json 只差指纹和耗时；
   结果包的 b0_reuse.json 是 pass。通过后 B0 直接用 Commit 1 的 15 项。
2. 完整性：30 个任务状态 pass、代码哈希是 v4.1（d16e182e…）；预测和额外 logits 的哈希与 done.json 一致；
   task.json 和 Commit 1 的 B0 只差代码身份、loss、group、group_spec（config 只差 code_version、loss_variants 两条说明）；训练行数 = fit、计划步数 = 同种子同折的 B0。
3. 训练记录：实际更新 + fp16 跳步 = 计划步数；选中轮 = 内层验证原方向 WF1 最高、同分取早；内层验证预测重算等于记录。
4. 分数：拼成 3,042 行折外预测，用官方评分模块算原方向和对称化的 WF1 / SoftCons / HardCons；O 的 AP 用对称化 p_O。
5. 比较（冻结计划变更记录 11-10）：Focal − B0、Dice − B0，同种子 sym WF1 差的均值，按家族配对重抽样
   （2,000 次、种子 20261021），Bonferroni 97.5% 区间；均值 > 0 且下限 > 0 才算提升。另报 95% 区间、plain、
   O 的 AP / F1 / P / R、预测分布、更新次数，以及对 W2 / W4 的描述。

用法：python3 -I code/claude_recheck_augb_c3.py <官方仓库目录>
输入：metrics/augment_b/train_c3/（结果包原样解压）、train_c1/
输出：metrics/augment_b/train_c3/oof/、claude_recheck.json
"""
import csv, hashlib, json, os, random, subprocess, sys, tempfile
from collections import Counter

OFF = os.path.abspath(sys.argv[1])
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = f"{SE}/metrics/augment_b"
R, C1 = f"{AB}/train_c3", f"{AB}/train_c1"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
SWAP = [0, 2, 1, 3]
SEEDS, FOLDS = [42, 43, 44], [0, 1, 2, 3, 4]
NEW = {"Focal": "focal_softmax_a1_g2", "Dice": "soft_dice_macro4_batch"}
OLD = {"B0": f"{C1}/tasks/B0", "W2": f"{C1}/tasks/W2", "W4": f"{C1}/tasks/W4"}
B, BOOT_SEED = 2000, 20261021
CODE_SHA = "d16e182e29556d3ea03c9eb9546b0497419ac0ab9bcf90b03cb256c86dd6fc06"
VARY = ("code_sha256", "loss", "group", "group_spec", "seed", "fold", "steps", "train_rows", "val_rows", "test_rows", "config")
CFG_VARY = ("code_version", "loss_variants")  # config 里只允许差代码说明
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


def ap(y, s):
    order = sorted(range(len(s)), key=lambda i: -s[i])
    npos, tp, fp, out, j = sum(y), 0, 0, 0.0, 0
    while j < len(order):
        k = j
        while k < len(order) and s[order[k]] == s[order[j]]:
            k += 1
        dtp = sum(y[order[t]] for t in range(j, k))
        tp += dtp; fp += (k - j) - dtp
        out += dtp / npos * tp / (tp + fp)
        j = k
    return out


# ---------- 官方数据和家族五折 ----------
ref_rows = rcsv(f"{OFF}/final_data/train/dico_nli_train_track1_reference.csv")
gold = {r["instance_id"]: r["label"] for r in ref_rows}
reverse_of = {r["instance_id"]: r["reverse_pair_id"] for r in ref_rows}
inst_of_pair = {r["pair_id"]: r["instance_id"] for r in ref_rows}
pair_of = {r["instance_id"]: r["pair_id"] for r in ref_rows}
fam = rj(f"{SE}/metrics/diagnostics/families.json")
family = {i: fam["family_of_pair"][pair_of[i]] for i in gold}
split = rj(f"{R}/input_data/split_manifest.json")["folds"]
check(sha(f"{R}/input_data/split_manifest.json").startswith("f9363290"), "split_manifest 哈希不对")

# ---------- B0 复用 ----------
reuse = rj(f"{R}/b0_reuse.json")
check(reuse["status"] == "pass" and not reuse["differences"], "b0_reuse.json 不是 pass")
for f_ in ("predictions.csv", "val_predictions.csv", "extra_logits.csv"):
    check(sha(f"{R}/smoke/B0/{f_}") == sha(f"{C1}/smoke/B0/{f_}"), f"B0 SMOKE 的 {f_} 和 Commit 1 归档不同")
a, b = rj(f"{R}/smoke/B0/done.json"), rj(f"{C1}/smoke/B0/done.json")
check({k for k in set(a) | set(b) if a.get(k) != b.get(k)} <= {"task_fingerprint", "runtime_sec"}, "B0 SMOKE 的 done.json 不只差指纹和耗时")
check(rj(f"{R}/loss_checks.json")["status"] == "pass", "loss_checks 不是 pass")
for g in ("B0", "Focal", "Dice"):
    sm = rj(f"{R}/smoke/{g}/smoke.json")
    check(sum(x["status"] == "updated" for x in sm["grad_trace"]) == 2, f"SMOKE {g} 实际更新不是 2 次")

# ---------- 逐任务 ----------
base_task = {x: v for x, v in rj(f"{C1}/tasks/B0/seed42/fold0/task.json").items() if x not in VARY}
base_cfg = {x: v for x, v in rj(f"{C1}/tasks/B0/seed42/fold0/task.json")["config"].items() if x not in CFG_VARY}
tasks, oof, oof_prob, updates = {}, {}, {}, []


def load(g, root, verify):
    for s in SEEDS:
        plain, sym, prob = {}, {}, {}
        for k in FOLDS:
            td = f"{root}/seed{s}/fold{k}"
            t, done = rj(f"{td}/task.json"), rj(f"{td}/done.json")
            tasks[(g, s, k)] = (t, done)
            rows = rcsv(f"{td}/predictions.csv")
            byid = {r["instance_id"]: r for r in rows}
            if verify:
                check({x: v for x, v in t.items() if x not in VARY} == base_task, f"{td} task.json 和 Commit 1 的 B0 不只差代码身份和 loss")
                check({x: v for x, v in t["config"].items() if x not in CFG_VARY} == base_cfg, f"{td} config 和 Commit 1 的 B0 不只差代码说明")
                check(t["code_sha256"] == CODE_SHA, f"{td} 代码哈希不对")
                check(t["loss"] == NEW[g] and t["group_spec"].get("loss") == NEW[g] and t["group_spec"]["kind"] == "orig", f"{td} loss 不对")
                check(t["group"] == g and t["seed"] == s and t["fold"] == k and not t["smoke"], f"{td} 身份不对")
                check(done.get("status") == "pass", f"{td} 状态 {done.get('status')}")
                for f_, key in (("predictions.csv", "predictions_sha256"), ("val_predictions.csv", "val_predictions_sha256"),
                                ("extra_logits.csv", "extra_logits_sha256")):
                    check(sha(f"{td}/{f_}") == done[key], f"{td} {f_} 哈希不对")
                b0t = tasks[("B0", s, k)][0]
                check(t["train_rows"] == len(split[str(k)]["fit"]) == b0t["train_rows"] and t["steps"] == b0t["steps"],
                      f"{td} 行数或计划步数和 B0 不同")
                check(t["class_weights"] is None and not done["repeat_counts"], f"{td} 不应有类别权重或重复行")
                check(done["actual_optimizer_updates"] + done["skipped_fp16_updates"] == done["planned_updates"] == t["steps"],
                      f"{td} 更新 + 跳步 ≠ 计划")
                ep = done["epochs"]
                check(len(ep) == 12, f"{td} 轮数 {len(ep)}")
                vals = [e["val_original_wf1"] for e in ep]
                check(done["best_epoch"] == vals.index(max(vals)) + 1, f"{td} 选轮不是最早的最高分")
                vrows = rcsv(f"{td}/val_predictions.csv")
                check(sorted(r["instance_id"] for r in vrows) == sorted(split[str(k)]["val"]), f"{td} 内层验证行不对")
                vp = [argmax([float(r["p_" + c]) for c in "EFBO"]) for r in vrows]
                check(all(r["pred"] == p for r, p in zip(vrows, vp)), f"{td} 内层验证 pred 和概率不一致")
                vw = wf1([gold[r["instance_id"]] for r in vrows], vp)
                check(abs(vw - done["best_val_original_wf1"]) < 1e-9, f"{td} 内层验证重算 {vw} ≠ 记录")
                check(sorted(byid) == sorted(split[str(k)]["test"]), f"{td} 留出行不对")
                updates.append({"group": g, "seed": s, "fold": k, "planned": done["planned_updates"],
                                "actual": done["actual_optimizer_updates"], "skipped": done["skipped_fp16_updates"],
                                "best_epoch": done["best_epoch"], "runtime_sec": done["runtime_sec"],
                                "final_train_loss": ep[-1]["train_loss"]})
            for r in rows:
                i = r["instance_id"]
                p = [float(r["p_" + c]) for c in "EFBO"]
                q = [float(r["r_p_" + c]) for c in "EFBO"]
                sp = [(p[j] + q[SWAP[j]]) / 2 for j in range(4)]
                if verify:
                    check(r["label"] == gold[i], f"{i} 金标不对")
                    rev = inst_of_pair.get(reverse_of[i]) if reverse_of[i] else None
                    if rev:
                        check(max(abs(a - float(byid[rev]["p_" + c])) for a, c in zip(q, "EFBO")) < 1e-9, f"{i} r_p 不等于反方向的 p")
                    check(max(abs(a - float(r["sym_p_" + c])) for a, c in zip(sp, "EFBO")) < 2e-9, f"{i} sym 概率不对")
                    check(r["plain"] == argmax(p) and r["sym"] == argmax(sp), f"{i} 保存的标签和概率不一致")
                plain[i], sym[i], prob[i] = r["plain"], r["sym"], sp
        oof[(g, s)] = {"plain": plain, "sym": sym}
        oof_prob[(g, s)] = prob


for g, d in OLD.items():
    load(g, d, False)
for g in NEW:
    load(g, f"{R}/tasks/{g}", True)


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
            **{f"per_class_{x}": {l: c["per_label"][l][y] for l in LABELS} for x, y in (("f1", "f1"), ("p", "precision"), ("r", "recall"))}}


os.makedirs(f"{R}/oof", exist_ok=True)
ALL = list(NEW) + list(OLD)
ids = sorted(gold)
scores = {}
for g in ALL:
    scores[g] = {}
    for s in SEEDS:
        scores[g][str(s)] = {}
        for mode in ("plain", "sym"):
            m = oof[(g, s)][mode]
            if g in NEW:
                with open(f"{R}/oof/{g}_seed{s}_{mode}.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f); w.writerow(["instance_id", "label"])
                    for r in ref_rows: w.writerow([r["instance_id"], m[r["instance_id"]]])
            o = official(m)
            check(abs(o["wf1"] - wf1([gold[i] for i in ids], [m[i] for i in ids])) < 1e-12, f"{g} {s} {mode} WF1 自算不一致")
            o["pred_dist"] = dict(Counter(m.values()))
            scores[g][str(s)][mode] = o
        scores[g][str(s)]["O_AP_sym"] = ap([gold[i] == "NEGATIVE_OTHER" for i in ids], [oof_prob[(g, s)][i][3] for i in ids])

# ---------- 配对比较 ----------
fams = sorted({family[i] for i in gold})
by_fam = {}
for i in gold:
    by_fam.setdefault(family[i], []).append(i)


def diff(a, b, mode="sym"):
    import numpy as np
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
    ci95 = [boot[int(.025 * B)], boot[int(.975 * B) - 1]]
    ci975 = [boot[int(.0125 * B)], boot[int(.9875 * B) - 1]]
    return {"per_seed": per_seed, "mean": mean, "ci95": ci95, "ci97_5_bonferroni": ci975,
            "improvement": mean > 0 and ci975[0] > 0}


main = {f"{g}_minus_B0": diff(g, "B0") for g in NEW}
desc = {**{f"{g}_minus_B0_plain": diff(g, "B0", "plain") for g in NEW},
        "Focal_minus_W4": diff("Focal", "W4"), "Dice_minus_W4": diff("Dice", "W4"),
        "Focal_minus_W2": diff("Focal", "W2"), "Dice_minus_W2": diff("Dice", "W2"), "Dice_minus_Focal": diff("Dice", "Focal")}
mean_scores = {g: {mode: {m: sum(scores[g][str(s)][mode][m] for s in SEEDS) / 3 for m in ("wf1", "softcons", "hardcons")}
                   for mode in ("plain", "sym")} for g in ALL}
per_class = {g: {x: {l: sum(scores[g][str(s)]["sym"][f"per_class_{x}"][l] for s in SEEDS) / 3 for l in LABELS}
                 for x in ("f1", "p", "r")} for g in ALL}
o_ap = {g: {"by_seed": {str(s): scores[g][str(s)]["O_AP_sym"] for s in SEEDS},
            "mean": sum(scores[g][str(s)]["O_AP_sym"] for s in SEEDS) / 3} for g in ALL}
pred_o = {g: {str(s): scores[g][str(s)]["sym"]["pred_dist"].get("NEGATIVE_OTHER", 0) for s in SEEDS} for g in ALL}
out = {"problems": problems, "new_tasks": len(updates), "b0_reuse": "pass（SMOKE 三个预测文件和 Commit 1 归档逐字节相同）",
       "scores_by_seed": scores, "scores_mean": mean_scores, "per_class_sym_mean": per_class, "O_AP_sym": o_ap,
       "pred_O_sym": pred_o, "main_comparisons": main, "descriptive_comparisons": desc, "updates": updates,
       "rule": "Focal − B0、Dice − B0：均值 > 0 且 Bonferroni 97.5% 家族重抽样区间下限 > 0；区间不含重新训练随机性"}
json.dump(out, open(f"{R}/claude_recheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("问题：", problems[:20] if problems else "无", len(problems))
for g in ALL:
    print(f"{g:6s}", {mode: {m: round(v * 100, 2) for m, v in mean_scores[g][mode].items()} for mode in ("plain", "sym")},
          "O F1/P/R", [round(per_class[g][x]["NEGATIVE_OTHER"] * 100, 1) for x in ("f1", "p", "r")],
          "O-AP", round(o_ap[g]["mean"] * 100, 2), {s: round(v * 100, 2) for s, v in o_ap[g]["by_seed"].items()}, "predO", pred_o[g])
for k, v in list(main.items()) + list(desc.items()):
    print(k, {s: round(x * 100, 2) for s, x in v["per_seed"].items()}, round(v["mean"] * 100, 2),
          "95%", [round(x * 100, 2) for x in v["ci95"]], "97.5%", [round(x * 100, 2) for x in v["ci97_5_bonferroni"]],
          "提升" if v["improvement"] else "-")
