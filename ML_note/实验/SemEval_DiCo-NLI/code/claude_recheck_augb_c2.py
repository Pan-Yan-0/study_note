"""Claude 对实验 B Commit 2（B2-Haiku / B4-Haiku / G-U4 / BAL / E1 / D1 / O1，英语五折三种子，Kaggle T4 fp16，105 项）的独立复核。

不复用训练脚本的代码，只读每个任务保存的文件：
1. 完整性：105 个任务都在、状态 pass、代码哈希是 v3（c8ff912c…）；预测和额外 logits 的哈希与 done.json 一致；
   task.json 除组相关字段外和 Commit 1 的 B0 完全相同；v3 数据哈希等于仓库里归档的规模轮文件。
2. 训练记录：实际更新 + fp16 跳步 = 计划步数；选中轮 = 内层验证原方向 WF1 最高、同分取早；保存的内层验证预测重算等于记录。
3. 各组的数据：B2 = fit + 第 1、2 层，B4 = fit + 第 1、2、4 层（fold_selection_scale.csv），G_c = fit + mix_lists.json 里
   该臂的 pair_uid 的全部行；生成行在该折可用、上下文父样本都在该折 fit 里；G_c 每类新增行数等于 Dup_c；
   计划步数 B2 = step+2、B4 = step+4、G_c = Dup_c（同种子同折）。
4. 分数：拼成 3,042 行折外预测，用官方评分模块算原方向和对称化的 WF1 / SoftCons / HardCons；O 的 AP 用对称化 p_O。
5. 比较：同种子 sym WF1 差（或差的线性组合）的均值，按家族配对重抽样（2,000 次、种子 20261021）。判断：均值 > 0 且区间下限 > 0。
   - 冻结计划 3.5：B2 − B0、B2 − step+2、B4 − B0、B4 − step+4，+k 两条都过才算扩充有效。
   - 类别构成 v2 第 4 节：G_c − Dup_c（同代码）、G_c − G_R（G_R = +1 的 B1-Haiku，v2 代码）、
     Δ_c = (G_c − Dup_c) − (G_R − Dup_R)（Dup_R = +1 的 step+1，v2 代码）；"构成 c 值得生成"要求前两条都过。
   - 描述：B2 − B1、B4 − B2、规模曲线。

用法：python3 -I code/claude_recheck_augb_c2.py <官方仓库目录>
输入：metrics/augment_b/train_c2/（结果包原样解压，加 Commit 2 包里的 mix_lists.json）、train_c1/、train_plus1/、haiku_scale/merged/
输出：metrics/augment_b/train_c2/oof/、claude_recheck.json
"""
import csv, hashlib, json, os, random, subprocess, sys, tempfile
from collections import Counter

OFF = os.path.abspath(sys.argv[1])
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = f"{SE}/metrics/augment_b"
R, C1, P1, SCALE = f"{AB}/train_c2", f"{AB}/train_c1", f"{AB}/train_plus1", f"{AB}/haiku_scale/merged"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
SWAP = [0, 2, 1, 3]
SEEDS, FOLDS = [42, 43, 44], [0, 1, 2, 3, 4]
POINTS = ["U4", "BAL", "E1", "D1", "O1"]
NEW = {"B2-Haiku": "B2-Haiku", "B4-Haiku": "B4-Haiku", **{f"G-{p}": f"G-{p}" for p in POINTS}}
OLD = {"B0": f"{C1}/tasks/B0", "step+2": f"{C1}/tasks/stepplus2", "step+4": f"{C1}/tasks/stepplus4",
       **{f"Dup-{p}": f"{C1}/tasks/Dup-{p}" for p in POINTS},
       "B1-Haiku(v2)": f"{P1}/tasks/B1-Haiku", "step+1(v2)": f"{P1}/tasks/stepplus1", "B0(v2)": f"{P1}/tasks/B0"}
MATCH_STEPS = {"B2-Haiku": "step+2", "B4-Haiku": "step+4", **{f"G-{p}": f"Dup-{p}" for p in POINTS}}
B, BOOT_SEED = 2000, 20261021
CODE_SHA = "c8ff912cf3f587e4d7d5b06f658946aaa8c5e79659be2ce8e992441db4d7d36c"
VARY = ("group", "group_spec", "class_weights", "train_rows", "steps", "seed", "fold", "val_rows", "test_rows", "epochs",
        "v3_data_sha256")
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


def pset(s):
    s = (s or "").strip()
    return set(x for x in s.replace(";", "|").split("|") if x) if s else set()


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
    """二分类 average precision（sklearn 的定义：按分数降序，每个正例处的 precision 取平均；并列一起算）。"""
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
fold_of = {i: int(fam["fold_of_family"][str(family[i])]) for i in gold}
split = rj(f"{R}/input_data/split_manifest.json")["folds"]
check(sha(f"{R}/input_data/split_manifest.json").startswith("f9363290"), "split_manifest 哈希不对")
for k in FOLDS:
    check(set(split[str(k)]["test"]) == {i for i in gold if fold_of[i] == k}, f"fold {k} 留出和 families.json 不同")

# ---------- 规模轮数据 ----------
mix = rj(f"{R}/input_data/mix_lists.json")
check(sha(f"{R}/input_data/mix_lists.json").startswith("ff21b1c3"), "mix_lists 不是 Commit 2 包里的那份")
c1mix = rj(f"{C1}/input_data/mix_lists.json")
for p in POINTS:
    check(mix["arms"][f"Dup-{p}"] == c1mix["arms"][f"Dup-{p}"], f"Dup-{p} 在两个包里不同")
check(mix["pool_order_sha256"] == sha(f"{SCALE}/pool_order.json"), "mix_lists 的 pool_order 哈希和归档不同")
man = rcsv(f"{SCALE}/manifest.csv")
by_sid = {r["sample_id"]: r for r in man}
by_uid = {}
for r in man:
    by_uid.setdefault(r["pair_uid"], []).append(r)
scale_sel = rcsv(f"{SCALE}/fold_selection_scale.csv")
tier_rows = {}
for s in scale_sel:
    tier_rows.setdefault((int(s["fold"]), int(s["tier"])), []).append(by_sid[s["sample_id"]])
plus1 = {(int(r["fold"]), r["sample_id"]) for r in rcsv(f"{R}/input_data/fold_selection.csv")}
check({(k, r["sample_id"]) for (k, t), rs in tier_rows.items() if t == 1 for r in rs} == plus1, "第 1 层和 +1 的选用不同")
check(len({(k, r["sample_id"]) for (k, t), rs in tier_rows.items() for r in rs}) == len(scale_sel), "各层之间有重复")


def gen_rows(group, k):
    spec = {"B2-Haiku": (1, 2), "B4-Haiku": (1, 2, 4)}
    if group in spec:
        return [r for t in spec[group] for r in tier_rows[(k, t)]]
    return [r for u in mix["arms"][group]["folds"][str(k)]["pair_uids"] for r in by_uid[u]]


# ---------- 逐任务核对（新任务）和读入（全部组） ----------
base_task = {x: v for x, v in rj(f"{C1}/tasks/B0/seed42/fold0/task.json").items() if x not in VARY}
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
                check({x: v for x, v in t.items() if x not in VARY} == base_task, f"{td} task.json 和 Commit 1 的 B0 不只差组相关字段")
                check(t["code_sha256"] == CODE_SHA, f"{td} 代码哈希不对")
                check(t["group"] == g and t["seed"] == s and t["fold"] == k and not t["smoke"], f"{td} 身份不对")
                check(done.get("status") == "pass", f"{td} 状态 {done.get('status')}")
                vh = t["v3_data_sha256"]
                for n in ("manifest.csv", "fold_selection_scale.csv", "pool_order.json"):
                    check(vh[n] == sha(f"{SCALE}/{n}"), f"{td} {n} 哈希和归档不同")
                check(vh["mix_lists.json"] == sha(f"{R}/input_data/mix_lists.json"), f"{td} mix_lists 哈希不对")
                for f_, key in (("predictions.csv", "predictions_sha256"), ("val_predictions.csv", "val_predictions_sha256"),
                                ("extra_logits.csv", "extra_logits_sha256")):
                    check(sha(f"{td}/{f_}") == done[key], f"{td} {f_} 哈希不对")
                check(done["actual_optimizer_updates"] + done["skipped_fp16_updates"] == done["planned_updates"] == t["steps"],
                      f"{td} 更新 + 跳步 ≠ 计划")
                fit = split[str(k)]["fit"]
                gr = gen_rows(g, k)
                check(t["train_rows"] == len(fit) + len(gr), f"{td} 行数 {t['train_rows']} ≠ fit + 生成 {len(fit) + len(gr)}")
                check(done["source_count"] == {"claude-haiku-5-5": len(gr), "original": len(fit)}, f"{td} source_count 不对")
                check(not done["repeat_counts"], f"{td} 不应有重复行")
                fitset = set(fit)
                for r in gr:
                    check(str(k) in pset(r["usable_folds"]), f"{td} {r['sample_id']} 在本折不可用")
                    check(pset(r["context_parent_ids"]) <= fitset, f"{td} {r['sample_id']} 上下文父样本不在 fit")
                    check(r["source"] == "claude-haiku-5-5" and r["generator_model"] == "haiku", f"{td} {r['sample_id']} 来源不对")
                if g.startswith("G-"):
                    want = Counter()
                    for i, n in mix["arms"]["Dup-" + g[2:]]["folds"][str(k)]["dup"]:
                        want[gold[i]] += n
                    check(Counter(r["label"] for r in gr) == want, f"{td} 新增的类别构成和 Dup 不同")
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
                                "best_epoch": done["best_epoch"], "train_rows": t["train_rows"], "runtime_sec": done["runtime_sec"]})
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


for g, d in NEW.items():
    load(g, f"{R}/tasks/{d}", True)
for g, d in OLD.items():
    load(g, d, False)
for g, m in MATCH_STEPS.items():
    for s in SEEDS:
        for k in FOLDS:
            check(tasks[(g, s, k)][0]["steps"] == tasks[(m, s, k)][0]["steps"], f"{g} {s} {k} 计划步数和 {m} 不同")
            check(tasks[(g, s, k)][0]["train_rows"] == tasks[(m, s, k)][0]["train_rows"], f"{g} {s} {k} 行数和 {m} 不同")


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
            "per_class_f1": {l: c["per_label"][l]["f1"] for l in LABELS},
            "per_class_p": {l: c["per_label"][l]["precision"] for l in LABELS},
            "per_class_r": {l: c["per_label"][l]["recall"] for l in LABELS}}


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

# ---------- 配对比较（差的线性组合） ----------
fams = sorted({family[i] for i in gold})
by_fam = {}
for i in gold:
    by_fam.setdefault(family[i], []).append(i)


def compare(terms, mode="sym"):
    """terms = [(系数, 组), ...]；统计量 = Σ 系数 · WF1(组)，按种子平均。"""
    import numpy as np
    per_seed = {str(s): sum(c * wf1([gold[i] for i in ids], [oof[(g, s)][mode][i] for i in ids]) for c, g in terms) for s in SEEDS}
    counts = np.zeros((len(fams), len(SEEDS), len(terms), 4, 4), dtype=np.int64)
    for fi, f in enumerate(fams):
        for si, s in enumerate(SEEDS):
            for i in by_fam[f]:
                gi = L2I[gold[i]]
                for ti, (_, g) in enumerate(terms):
                    counts[fi, si, ti, gi, L2I[oof[(g, s)][mode][i]]] += 1
    coef = np.array([c for c, _ in terms])
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
        boot.append(float((w * coef).sum(-1).mean()))
    boot.sort()
    mean = sum(per_seed.values()) / len(SEEDS)
    ci = [boot[int(.025 * B)], boot[int(.975 * B) - 1]]
    return {"terms": terms, "per_seed": per_seed, "mean": mean, "ci95": ci, "improvement": mean > 0 and ci[0] > 0}


def diff(a, b, mode="sym"):
    return compare([(1, a), (-1, b)], mode)


main_scale = {}
for k_, st in (("B2-Haiku", "step+2"), ("B4-Haiku", "step+4")):
    for mode in ("sym", "plain"):
        sfx = "" if mode == "sym" else "_plain"
        main_scale[f"{k_}_minus_B0{sfx}"] = diff(k_, "B0", mode)
        main_scale[f"{k_}_minus_{st}{sfx}"] = diff(k_, st, mode)
mixture = {}
for p in POINTS:
    for mode in ("sym", "plain"):
        sfx = "" if mode == "sym" else "_plain"
        mixture[f"G-{p}_minus_Dup-{p}{sfx}"] = diff(f"G-{p}", f"Dup-{p}", mode)
        mixture[f"G-{p}_minus_G_R（B1，跨代码）{sfx}"] = diff(f"G-{p}", "B1-Haiku(v2)", mode)
        mixture[f"Delta_{p}（交互，跨代码）{sfx}"] = compare([(1, f"G-{p}"), (-1, f"Dup-{p}"), (-1, "B1-Haiku(v2)"), (1, "step+1(v2)")], mode)
desc = {"B1(v2)_minus_step+1(v2)（G_R − Dup_R，+1 已报）": diff("B1-Haiku(v2)", "step+1(v2)"),
        "B2_minus_B1（跨代码）": diff("B2-Haiku", "B1-Haiku(v2)"),
        "B4_minus_B2": diff("B4-Haiku", "B2-Haiku"),
        "B4_minus_B1（跨代码）": diff("B4-Haiku", "B1-Haiku(v2)")}
mean_scores = {g: {mode: {m: sum(scores[g][str(s)][mode][m] for s in SEEDS) / 3 for m in ("wf1", "softcons", "hardcons")}
                   for mode in ("plain", "sym")} for g in ALL}
per_class = {g: {x: {l: sum(scores[g][str(s)]["sym"][f"per_class_{x}"][l] for s in SEEDS) / 3 for l in LABELS}
                 for x in ("f1", "p", "r")} for g in ALL}
o_ap = {g: {"by_seed": {str(s): scores[g][str(s)]["O_AP_sym"] for s in SEEDS},
            "mean": sum(scores[g][str(s)]["O_AP_sym"] for s in SEEDS) / 3} for g in ALL}
out = {"problems": problems, "new_tasks": len(updates), "scores_by_seed": scores, "scores_mean": mean_scores,
       "per_class_sym_mean": per_class, "O_AP_sym": o_ap, "scale_comparisons": main_scale,
       "mixture_comparisons": mixture, "descriptive_comparisons": desc, "updates": updates,
       "rule": "均值 > 0 且家族重抽样区间下限 > 0；区间不含重新训练随机性；B1 / step+1 / B0(v2) 来自 +1 的 v2 代码"}
json.dump(out, open(f"{R}/claude_recheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("问题：", problems[:20] if problems else "无", len(problems))
for g in ALL:
    print(f"{g:14s}", {mode: {m: round(v * 100, 2) for m, v in mean_scores[g][mode].items()} for mode in ("plain", "sym")},
          "O-F1", round(per_class[g]["f1"]["NEGATIVE_OTHER"] * 100, 1), "O-AP", round(o_ap[g]["mean"] * 100, 2))
for k, v in list(main_scale.items()) + list(mixture.items()) + list(desc.items()):
    print(k, {s: round(x * 100, 2) for s, x in v["per_seed"].items()}, round(v["mean"] * 100, 2),
          [round(x * 100, 2) for x in v["ci95"]], "提升" if v["improvement"] else "-")
