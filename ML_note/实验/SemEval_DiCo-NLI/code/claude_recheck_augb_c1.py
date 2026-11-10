"""Claude 对实验 B Commit 1（B0 / W2 / W4 / step+2 / step+4 / Dup×5，英语五折三种子，Kaggle T4 fp16，150 项）的独立复核。

不复用训练脚本的代码，只读每个任务保存的文件：
1. 完整性：150 个任务都在、状态 pass；预测和额外 logits 的哈希与 done.json 一致；task.json 除组相关字段外完全相同；
   留出 / 内层验证和家族五折一致。
2. 训练记录：实际更新 + fp16 跳步 = 计划步数；选中轮 = 内层验证原方向 WF1 最高、同分取早；保存的内层验证预测重算等于记录。
3. 各组的数据：B0 / W = fit；step+k = (k+1)·fit，每行多重复 k 次；Dup = 2·fit，重复次数和 Commit 1 包里的 mix_lists.json 逐行相同；
   W 的类别权重 = w_O = λ/Z、其余 1/Z、Z = 1 + (λ−1)·p_O(fit)。
4. 分数：拼成 3,042 行折外预测，用官方评分模块算原方向和对称化的 WF1 / SoftCons / HardCons。
5. 比较：同种子 sym WF1 差的均值，按家族配对重抽样（2,000 次、种子 20261021）。判断：均值 > 0 且区间下限 > 0。
   - 事先登记的主比较：W2 − B0、W4 − B0；解析偏置对照 W2(−log 2)、W4(−log 4) − B0（O 的 logit 减 log λ，不需选择）。
   - 描述：step+2 / step+4 − B0；Dup_c − B0；Dup_c − Dup_R（Dup_R = +1 的 step+1，v2 代码，跨代码，只作参考）；
     v3 的 B0 和 +1（v2）的 B0 逐条对照。

用法：python3 -I code/claude_recheck_augb_c1.py <官方仓库目录>
输入：metrics/augment_b/train_c1/（结果包原样解压的 tasks/、input_data/，加 Commit 1 包里的 mix_lists.json）
输出：metrics/augment_b/train_c1/oof/、claude_recheck.json
"""
import csv, hashlib, json, math, os, random, subprocess, sys, tempfile
from collections import Counter

OFF = os.path.abspath(sys.argv[1])
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{SE}/metrics/augment_b/train_c1"
P1 = f"{SE}/metrics/augment_b/train_plus1"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
SWAP = [0, 2, 1, 3]
SEEDS, FOLDS = [42, 43, 44], [0, 1, 2, 3, 4]
GROUPS = {"B0": "B0", "W2": "W2", "W4": "W4", "step+2": "stepplus2", "step+4": "stepplus4",
          "Dup-U4": "Dup-U4", "Dup-BAL": "Dup-BAL", "Dup-E1": "Dup-E1", "Dup-D1": "Dup-D1", "Dup-O1": "Dup-O1"}
LAMBDA = {"W2": 2.0, "W4": 4.0}
B, BOOT_SEED = 2000, 20261021
CODE_SHA = "c8ff912cf3f587e4d7d5b06f658946aaa8c5e79659be2ce8e992441db4d7d36c"
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


def debias(p, lam):
    """O 的 logit 减 log λ：等于 p_O 除以 λ 再归一化。"""
    q = [p[0], p[1], p[2], p[3] / lam]
    s = sum(q)
    return [x / s for x in q]


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
    fams_fit = {family[i] for i in split[str(k)]["fit"]}
    check(not (fams_fit & {family[i] for i in split[str(k)]["val"] + split[str(k)]["test"]}), f"fold {k} 家族泄漏")
mix = rj(f"{R}/input_data/mix_lists.json")
check(sha(f"{R}/input_data/mix_lists.json").startswith("f4d39799"), "mix_lists 不是 Commit 1 包里的那份")

# ---------- 逐任务核对 ----------
base_task, tasks, oof, oof_prob, oof_raw, updates = None, {}, {}, {}, {}, []
VARY = ("group", "group_spec", "class_weights", "train_rows", "steps", "seed", "fold", "val_rows", "test_rows", "epochs")
for g, d in GROUPS.items():
    for s in SEEDS:
        plain, sym, prob, raw = {}, {}, {}, {}
        for k in FOLDS:
            td = f"{R}/tasks/{d}/seed{s}/fold{k}"
            t, done = rj(f"{td}/task.json"), rj(f"{td}/done.json")
            tasks[(g, s, k)] = (t, done)
            core = {x: t[x] for x in t if x not in VARY}
            if base_task is None:
                base_task = core
            check(core == base_task, f"{g} {s} {k} 的 task.json 和其他任务不只差组相关字段")
            check(t["code_sha256"] == CODE_SHA, f"{td} 代码哈希不对")
            check(t["group"] == g and t["seed"] == s and t["fold"] == k and not t["smoke"], f"{td} 身份不对")
            check(done.get("status") == "pass", f"{td} 状态 {done.get('status')}")
            check(t["val_rows"] == len(split[str(k)]["val"]) and t["test_rows"] == len(split[str(k)]["test"]), f"{td} 行数不对")
            for f_, key in (("predictions.csv", "predictions_sha256"), ("val_predictions.csv", "val_predictions_sha256"),
                            ("extra_logits.csv", "extra_logits_sha256")):
                check(sha(f"{td}/{f_}") == done[key], f"{td} {f_} 哈希不对")
            check(done["actual_optimizer_updates"] + done["skipped_fp16_updates"] == done["planned_updates"] == t["steps"],
                  f"{td} 更新 + 跳步 ≠ 计划")
            # 各组的数据
            fit = split[str(k)]["fit"]
            nfit = len(fit)
            rc = done["repeat_counts"]
            spec = t["group_spec"]
            if spec["kind"] == "orig":
                check(t["train_rows"] == nfit and not rc, f"{td} 行数应等于 fit")
                lam = spec.get("lambda_O", 1.0)
                if lam == 1.0:
                    check(t["class_weights"] is None, f"{td} B0 不应带类别权重（λ=1 走原 CE 分支）")
                else:
                    po = sum(gold[i] == "NEGATIVE_OTHER" for i in fit) / nfit
                    z = 1 + (lam - 1) * po
                    exp = [1 / z] * 3 + [lam / z]
                    check(max(abs(a - b) for a, b in zip(t["class_weights"], exp)) < 1e-8, f"{td} 类别权重 {t['class_weights']} ≠ {exp}")
            elif spec["kind"] == "step":
                kk = spec["k"]
                check(t["train_rows"] == (kk + 1) * nfit, f"{td} step 行数不对")
                check(set(rc) == set(fit) and all(v == kk for v in rc.values()), f"{td} step 每行重复次数不是 {kk}")
            else:
                check(t["train_rows"] == 2 * nfit, f"{td} Dup 行数不是 2·fit")
                want = {i: c for i, c in mix["arms"][g]["folds"][str(k)]["dup"]}
                check(rc == want, f"{td} Dup 重复次数和 mix_lists 不同")
                check(set(rc) <= set(fit), f"{td} Dup 用了 fit 以外的行")
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
                plain[i], sym[i], prob[i], raw[i] = argmax(p), argmax(sp), sp, (p, q)
                check(r["plain"] == plain[i] and r["sym"] == sym[i], f"{i} 保存的标签和概率不一致")
        oof[(g, s)] = {"plain": plain, "sym": sym}
        oof_prob[(g, s)] = prob
        oof_raw[(g, s)] = raw
for s in SEEDS:
    for k in FOLDS:
        st = {g: tasks[(g, s, k)][0]["steps"] for g in GROUPS}
        check(len({st[g] for g in GROUPS if g.startswith("Dup")}) == 1, f"{s} {k} Dup 各臂计划步数不同")
        check(st["B0"] == st["W2"] == st["W4"], f"{s} {k} B0 和 W 的计划步数不同")

# ---------- 解析偏置对照 ----------
for g, lam in LAMBDA.items():
    for s in SEEDS:
        plain, sym, prob = {}, {}, {}
        for i, (p, q) in oof_raw[(g, s)].items():
            pd, qd = debias(p, lam), debias(q, lam)
            sp = [(pd[j] + qd[SWAP[j]]) / 2 for j in range(4)]
            plain[i], sym[i], prob[i] = argmax(pd), argmax(sp), sp
        oof[(f"{g}-bias", s)] = {"plain": plain, "sym": sym}
        oof_prob[(f"{g}-bias", s)] = prob

# ---------- +1 的 B0 和 step+1（v2 代码） ----------
for g, d in (("B0(v2)", "B0"), ("Dup_R=step+1(v2)", "stepplus1")):
    for s in SEEDS:
        plain, sym = {}, {}
        for k in FOLDS:
            for r in rcsv(f"{P1}/tasks/{d}/seed{s}/fold{k}/predictions.csv"):
                plain[r["instance_id"]], sym[r["instance_id"]] = r["plain"], r["sym"]
        oof[(g, s)] = {"plain": plain, "sym": sym}


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
            "per_class_f1": {l: c["per_label"][l]["f1"] for l in LABELS}}


os.makedirs(f"{R}/oof", exist_ok=True)
ALL = list(GROUPS) + [f"{g}-bias" for g in LAMBDA] + ["B0(v2)", "Dup_R=step+1(v2)"]
scores = {}
for g in ALL:
    scores[g] = {}
    for s in SEEDS:
        scores[g][str(s)] = {}
        for mode in ("plain", "sym"):
            m = oof[(g, s)][mode]
            if g in GROUPS or g.endswith("-bias"):
                with open(f"{R}/oof/{GROUPS.get(g, g)}_seed{s}_{mode}.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f); w.writerow(["instance_id", "label"])
                    for r in ref_rows: w.writerow([r["instance_id"], m[r["instance_id"]]])
            o = official(m)
            check(abs(o["wf1"] - wf1([gold[i] for i in gold], [m[i] for i in gold])) < 1e-12, f"{g} {s} {mode} WF1 自算不一致")
            o["pred_dist"] = dict(Counter(m.values()))
            scores[g][str(s)][mode] = o

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
    return {"per_seed": per_seed, "mean": mean, "ci95": ci, "improvement": mean > 0 and ci[0] > 0}


main = {f"{a}_minus_B0": compare(a, "B0") for a in ("W2", "W4", "W2-bias", "W4-bias")}
desc = {}
for a in ("step+2", "step+4", "Dup-U4", "Dup-BAL", "Dup-E1", "Dup-D1", "Dup-O1"):
    desc[f"{a}_minus_B0"] = compare(a, "B0")
for a in ("Dup-U4", "Dup-BAL", "Dup-E1", "Dup-D1", "Dup-O1"):
    desc[f"{a}_minus_Dup_R（跨代码）"] = compare(a, "Dup_R=step+1(v2)")
desc["Dup_R(v2)_minus_B0(v2)（+1 已报）"] = compare("Dup_R=step+1(v2)", "B0(v2)")
desc["B0(v3)_minus_B0(v2)（代码路径对照）"] = compare("B0", "B0(v2)")
for a in ("W2", "W4"):
    desc[f"{a}_minus_B0_plain"] = compare(a, "B0", "plain")
agree = {f"B0_v3_vs_v2_seed{s}": sum(oof[("B0", s)]["sym"][i] == oof[("B0(v2)", s)]["sym"][i] for i in gold) / len(gold) for s in SEEDS}
mean_scores = {g: {mode: {m: sum(scores[g][str(s)][mode][m] for s in SEEDS) / 3 for m in ("wf1", "softcons", "hardcons")}
                   for mode in ("plain", "sym")} for g in ALL}
per_class = {g: {l: sum(scores[g][str(s)]["sym"]["per_class_f1"][l] for s in SEEDS) / 3 for l in LABELS} for g in ALL}
out = {"problems": problems, "tasks": len(tasks), "scores_by_seed": scores, "scores_mean": mean_scores, "per_class_f1_sym_mean": per_class,
       "main_comparisons": main, "descriptive_comparisons": desc, "b0_v3_v2_agreement_sym": agree, "updates": updates,
       "rule": "均值 > 0 且家族重抽样区间下限 > 0；区间不含重新训练随机性"}
json.dump(out, open(f"{R}/claude_recheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("问题：", problems[:20] if problems else "无", len(problems))
for g in ALL:
    print(g, {mode: {m: round(v * 100, 2) for m, v in mean_scores[g][mode].items()} for mode in ("plain", "sym")},
          "O-F1", round(per_class[g]["NEGATIVE_OTHER"] * 100, 1))
for k, v in list(main.items()) + list(desc.items()):
    print(k, {s: round(x * 100, 2) for s, x in v["per_seed"].items()}, round(v["mean"] * 100, 2),
          [round(x * 100, 2) for x in v["ci95"]], "提升" if v["improvement"] else "-")
print("B0 v3 和 v2 的对称化预测一致率：", {k: round(v, 4) for k, v in agree.items()})
