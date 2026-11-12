"""Claude 对实验 B Commit 4（B0 数据上的三臂：OLat4 / NLIinit / BiBCE，英语五折三种子，Kaggle T4 fp16）的独立复核。
运行前写定（11-12，冻结计划变更记录 Commit 4 条目和 Codex 11-12 复核），看结果后不改。

有问题时仍写出诊断，但以非零状态退出，主比较只在 0 个问题时才能引用。
不复用训练脚本的代码，只读每个任务保存的文件：
1. 回执：arm_checks.json、olat1_check.json 都是 pass；b0_reuse.json 决定 B0 的来源——pass 时用 Commit 1 归档的 15 项，
   并核对 v5.1 的 B0 SMOKE 三个预测文件和归档逐字节相同；不 pass 时用本次结果包里重跑的 B0（同样逐项核对）。
2. 完整性：45 个臂任务（以及重跑时的 15 个 B0，对照 Commit 1 归档的同种子同折 B0）状态 pass、代码哈希是 v5.2；
   有效损失两边都是 CE（缺 loss 字段按 CE）；预测和额外 logits 的哈希和 done.json 一致；
   task.json 和 B0 只差代码身份、group、group_spec（config 只差 code_version、loss_variants、arm_variants）；
   训练行数 = fit，计划步数 = 同种子同折的 B0；GPU、精度、torch、cuda、transformers 和 B0 的任务相同。
3. 训练记录：实际更新 + fp16 跳步 = 计划步数；选中轮 = 内层验证原方向 WF1 最高、同分取早；内层验证预测重算等于记录。
4. 分数：拼成 3,042 行折外预测，用官方评分模块算原方向和对称化的 WF1 / SoftCons / HardCons。
5. 主比较：OLat4 − B0、NLIinit − B0、BiBCE − B0，用 sym WF1。每次重抽样从 1,725 个家族有放回地抽，同一组抽样在
   四组和三个种子之间共用（抽样方式和 Commit 3 复核相同：random.Random(20261021).choices，每次抽 1,725 个）；
   先算每个种子的差，再取三个种子的均值。重抽样 20,000 次，双侧 Bonferroni 98.33% 区间（三条比较），取第
   ⌊B·0.05/6⌋ 个和第 B−1−⌊B·0.05/6⌋ 个次序统计量（从 0 数）。均值和区间下限都严格大于 0 才算提升，用未四舍五入的值判。
6. 另报（描述）：逐种子差、95% 区间、plain 方向的同样比较；O 的 AP（对称化 p_O，逐种子）、O 的漏判（金标 O 判成非 O）
   和误判（金标非 O 判成 O）、预测为 O 的行数、逐类 F1；OLat4 各分量的责任度；NLIinit 的载入记录；更新次数。

用法：
  python3 -I code/claude_recheck_augb_c4.py <官方仓库目录>
  python3 -I code/claude_recheck_augb_c4.py <官方仓库目录> --selftest-c3   用同一个重抽样函数重算 Commit 3 的两条主比较
                                                                          （B = 2,000、m = 2），必须和已归档的 claude_recheck.json 相同
输入：metrics/augment_b/train_c4/（结果包原样解压）、train_c1/
输出：metrics/augment_b/train_c4/oof/、claude_recheck.json
"""
import csv, hashlib, json, os, random, subprocess, sys, tempfile
from collections import Counter

import numpy as np

OFF = os.path.abspath(sys.argv[1])
SELFTEST = "--selftest-c3" in sys.argv
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = f"{SE}/metrics/augment_b"
R, C1, C3 = f"{AB}/train_c4", f"{AB}/train_c1", f"{AB}/train_c3"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
SWAP = [0, 2, 1, 3]
SEEDS, FOLDS = [42, 43, 44], [0, 1, 2, 3, 4]
B, BOOT_SEED, M = 20000, 20261021, 3
CODE_SHA = "7f3e01b36b8bfe1c92e22f1ef460117b7f359bf6d029df5cd8bbf638094fa4f1"  # train_augb_v5.py v5.2
NEW = {"OLat4": {"kind": "orig", "head": "olat", "K": 4, "perturb_std": 0.002, "perturb_seed_offset": 20000},
       "NLIinit": {"kind": "orig", "init": "nli", "init_model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                   "init_revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "init_prefixes": ["deberta.", "pooler."]},
       "BiBCE": {"kind": "orig", "aux": "bibce_mean_of_two_directions", "bibce_lambda": 0.25}}
VARY = ("code_sha256", "group", "group_spec", "seed", "fold", "steps", "train_rows", "val_rows", "test_rows", "config", "loss")
eff_loss = lambda t: t.get("loss", "ce")  # 归档的 Commit 1 task.json 没有 loss 字段，按训练入口的口径视为 CE；显式非 CE 拒绝
CFG_VARY = ("code_version", "loss_variants", "arm_variants")
ENV_KEYS = ("gpu", "precision", "torch", "cuda", "transformers")
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


# ---------- 官方数据和家族 ----------
ref_rows = rcsv(f"{OFF}/final_data/train/dico_nli_train_track1_reference.csv")
gold = {r["instance_id"]: r["label"] for r in ref_rows}
reverse_of = {r["instance_id"]: r["reverse_pair_id"] for r in ref_rows}
inst_of_pair = {r["pair_id"]: r["instance_id"] for r in ref_rows}
pair_of = {r["instance_id"]: r["pair_id"] for r in ref_rows}
fam = rj(f"{SE}/metrics/diagnostics/families.json")
family = {i: fam["family_of_pair"][pair_of[i]] for i in gold}
ids = sorted(gold)
fams = sorted({family[i] for i in gold})
fidx = {f: n for n, f in enumerate(fams)}
gold_idx = np.array([L2I[gold[i]] for i in ids])
fam_of_row = np.array([fidx[family[i]] for i in ids])


# ---------- 共用的家族重抽样 ----------
def boot_wf1(preds, n_boot, chunk=500):
    """preds: {group: [seed -> list of label indices aligned with ids]}. Returns {group: array (n_boot, seeds)} of WF1,
    every resample drawing families exactly as the Commit 3 recheck did (one rng.choices of len(fams) per resample)."""
    groups = list(preds)
    S = len(next(iter(preds.values())))
    counts = np.zeros((len(fams), len(groups), S, 4, 4))
    for gi, g in enumerate(groups):
        for si in range(S):
            np.add.at(counts, (fam_of_row, gi, si, gold_idx, np.asarray(preds[g][si])), 1)
    flat = counts.reshape(len(fams), -1)
    rng, res = random.Random(BOOT_SEED), np.zeros((n_boot, len(groups), S))
    for start in range(0, n_boot, chunk):
        n = min(chunk, n_boot - start)
        wts = np.stack([np.bincount([fidx[f] for f in rng.choices(fams, k=len(fams))], minlength=len(fams)) for _ in range(n)])
        cm = (wts.astype(float) @ flat).reshape(n, len(groups), S, 4, 4)
        sup, prd = cm.sum(-1), cm.sum(-2)
        tp = np.diagonal(cm, axis1=-2, axis2=-1)
        den = sup + prd
        f1 = np.divide(2 * tp, den, out=np.zeros_like(den), where=den != 0)
        res[start:start + n] = (f1 * sup).sum(-1) / sup.sum(-1)
    return {g: res[:, gi, :] for gi, g in enumerate(groups)}


def compare(boot, point, a, b, n_boot, m):
    per_seed = point[a] - point[b]
    d = np.sort((boot[a] - boot[b]).mean(axis=1))
    lo = int(n_boot * 0.05 / (2 * m))
    lo95 = int(n_boot * 0.025)
    mean = float(per_seed.mean())
    ci = [float(d[lo]), float(d[n_boot - 1 - lo])]
    return {"per_seed": {str(s): float(x) for s, x in zip(SEEDS, per_seed)}, "mean": mean,
            "ci95": [float(d[lo95]), float(d[n_boot - 1 - lo95])], "ci_bonferroni": ci,
            "bonferroni_level": 1 - 0.05 / m, "improvement": mean > 0 and ci[0] > 0}


def point_wf1(preds):
    return {g: np.array([wf1([gold[i] for i in ids], [LABELS[k] for k in v]) for v in seeds]) for g, seeds in preds.items()}


def read_oof(path):
    m = {r["instance_id"]: r["label"] for r in rcsv(path)}
    return [L2I[m[i]] for i in ids]


if SELFTEST:
    preds = {g: [read_oof(f"{(C1 if g == 'B0' else C3)}/oof/{g}_seed{s}_sym.csv") for s in SEEDS] for g in ("B0", "Focal", "Dice")}
    boot = boot_wf1(preds, 2000)
    pt = point_wf1(preds)
    ref = rj(f"{C3}/claude_recheck.json")["main_comparisons"]
    worst = 0.0
    for g in ("Focal", "Dice"):
        c = compare(boot, pt, g, "B0", 2000, 2)
        r = ref[f"{g}_minus_B0"]
        worst = max([worst, abs(c["mean"] - sum(r["per_seed"].values()) / 3)] + [abs(x - y) for x, y in zip(c["ci_bonferroni"], r["ci97_5_bonferroni"])]
                    + [abs(c["per_seed"][s] - r["per_seed"][s]) for s in r["per_seed"]] + [abs(x - y) for x, y in zip(c["ci95"], r["ci95"])])
        print(g, "per-seed", {s: round(v * 100, 4) for s, v in c["per_seed"].items()}, "97.5%", [round(x * 100, 4) for x in c["ci_bonferroni"]],
              "archived", [round(x * 100, 4) for x in r["ci97_5_bonferroni"]])
    print("selftest max abs difference vs archived Commit 3 recheck:", worst)
    sys.exit(0 if worst <= 1e-12 else 1)

# ---------- 回执和 B0 的来源 ----------
for f_ in ("arm_checks.json", "olat1_check.json", "b0_reuse.json"):
    check(os.path.exists(f"{R}/{f_}"), f"缺 {f_}")
check(rj(f"{R}/arm_checks.json")["status"] == "pass" and rj(f"{R}/arm_checks.json")["code_sha256"] == CODE_SHA, "arm_checks.json 不是 v5.1 的 pass")
check(rj(f"{R}/olat1_check.json")["status"] == "pass", "olat1_check.json 不是 pass")
reuse = rj(f"{R}/b0_reuse.json")["status"] == "pass"
if reuse:
    for f_ in ("predictions.csv", "val_predictions.csv", "extra_logits.csv"):
        check(sha(f"{R}/smoke/B0/{f_}") == sha(f"{C1}/smoke/B0/{f_}"), f"B0 SMOKE 的 {f_} 和 Commit 1 归档不同")
for g in ("B0", "OLat1", "OLat4", "NLIinit", "BiBCE"):
    sm = rj(f"{R}/smoke/{g}/smoke.json")
    check(sm["status"] == "pass" and sum(x["status"] == "updated" for x in sm["grad_trace"]) == 2, f"SMOKE {g} 不是 2 次实际更新")
B0_ROOT = f"{C1}/tasks/B0" if reuse else f"{R}/tasks/B0"
split = rj(f"{R}/input_data/split_manifest.json")["folds"]
check(sha(f"{R}/input_data/split_manifest.json").startswith("f9363290"), "split_manifest 哈希不对")

# ---------- 逐任务 ----------
tasks, oof, oof_prob, updates, extras = {}, {}, {}, [], {g: [] for g in NEW}


def load(g, root, verify, spec):
    for s in SEEDS:
        plain, sym, prob = {}, {}, {}
        for k in FOLDS:
            td = f"{root}/seed{s}/fold{k}"
            t, done = rj(f"{td}/task.json"), rj(f"{td}/done.json")
            tasks[(g, s, k)] = (t, done)
            rows = rcsv(f"{td}/predictions.csv")
            byid = {r["instance_id"]: r for r in rows}
            if verify:
                b0t, b0d = arch[(s, k)] if g == "B0" else tasks[("B0", s, k)]
                check({x: v for x, v in t.items() if x not in VARY} == {x: v for x, v in b0t.items() if x not in VARY}, f"{td} task.json 和 B0 不只差代码身份和组")
                check({x: v for x, v in t["config"].items() if x not in CFG_VARY} == {x: v for x, v in b0t["config"].items() if x not in CFG_VARY}, f"{td} config 和 B0 不只差代码说明")
                check(t["code_sha256"] == CODE_SHA, f"{td} 代码哈希不对")
                check(t["group"] == g and t["group_spec"] == spec, f"{td} 组或 group_spec 不对")
                check(eff_loss(t) == "ce" and eff_loss(b0t) == "ce", f"{td} 有效损失不是 CE")
                check(t["seed"] == s and t["fold"] == k and not t["smoke"], f"{td} 身份不对")
                if g != "B0":  # 重跑的 B0 本来就和归档环境不同；三臂必须和所用的 B0 同环境
                    check(all(done["environment"].get(e) == b0d["environment"].get(e) for e in ENV_KEYS), f"{td} 环境和 B0 不同")
                check(done.get("status") == "pass", f"{td} 状态 {done.get('status')}")
                for f_, key in (("predictions.csv", "predictions_sha256"), ("val_predictions.csv", "val_predictions_sha256"),
                                ("extra_logits.csv", "extra_logits_sha256")):
                    check(sha(f"{td}/{f_}") == done[key], f"{td} {f_} 哈希不对")
                check(t["train_rows"] == len(split[str(k)]["fit"]) == b0t["train_rows"] and t["steps"] == b0t["steps"], f"{td} 行数或计划步数和 B0 不同")
                check(t["class_weights"] is None and not done["repeat_counts"], f"{td} 不应有类别权重或重复行")
                check(done["actual_optimizer_updates"] + done["skipped_fp16_updates"] == done["planned_updates"] == t["steps"], f"{td} 更新 + 跳步 ≠ 计划")
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
                updates.append({"group": g, "seed": s, "fold": k, "planned": done["planned_updates"], "actual": done["actual_optimizer_updates"],
                                "skipped": done["skipped_fp16_updates"], "best_epoch": done["best_epoch"], "runtime_sec": done["runtime_sec"]})
                if g == "OLat4":
                    oc = done.get("olat_components") or {}
                    check(oc.get("val", {}).get("gold_O") and oc.get("test", {}).get("gold_O"), f"{td} 缺 olat_components")
                    extras[g].append({"seed": s, "fold": k, **{p: oc.get(p, {}).get("gold_O") for p in ("val", "test")}})
                if g == "NLIinit":
                    ni = (done.get("arm_report") or {}).get("nli_init") or {}
                    check(ni.get("revision") == spec["init_revision"] and ni.get("tensors_loaded") == 200 and ni.get("parameters_loaded") == 278809344,
                          f"{td} NLI 载入记录不对：{ni.get('tensors_loaded')} / {ni.get('parameters_loaded')}")
                    check(ni.get("kept_from_b0") == ["classifier.bias", "classifier.weight"], f"{td} 保留 B0 的键不是只有 classifier")
                    extras[g].append({"seed": s, "fold": k, "tensors_loaded": ni.get("tensors_loaded"), "parameters_loaded": ni.get("parameters_loaded")})
            for r in rows:
                i = r["instance_id"]
                p = [float(r["p_" + c]) for c in "EFBO"]
                q = [float(r["r_p_" + c]) for c in "EFBO"]
                sp = [(p[j] + q[SWAP[j]]) / 2 for j in range(4)]
                if verify:
                    check(r["label"] == gold[i], f"{i} 金标不对")
                    rv = inst_of_pair.get(reverse_of[i]) if reverse_of[i] else None
                    if rv:
                        check(max(abs(a - float(byid[rv]["p_" + c])) for a, c in zip(q, "EFBO")) < 1e-9, f"{i} r_p 不等于反方向的 p")
                    check(max(abs(a - float(r["sym_p_" + c])) for a, c in zip(sp, "EFBO")) < 2e-9, f"{i} sym 概率不对")
                    check(r["plain"] == argmax(p) and r["sym"] == argmax(sp), f"{i} 保存的标签和概率不一致")
                plain[i], sym[i], prob[i] = r["plain"], r["sym"], sp
        oof[(g, s)] = {"plain": plain, "sym": sym}
        oof_prob[(g, s)] = prob


arch = {(s, k): (rj(f"{C1}/tasks/B0/seed{s}/fold{k}/task.json"), rj(f"{C1}/tasks/B0/seed{s}/fold{k}/done.json")) for s in SEEDS for k in FOLDS}
load("B0", B0_ROOT, not reuse, {"kind": "orig"})  # 重跑的 B0 走和三臂相同的逐任务核对（对照归档的 B0）
for g, spec in NEW.items():
    load(g, f"{R}/tasks/{g}", True, spec)


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
ALL = ["B0"] + list(NEW)
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
        sym = oof[(g, s)]["sym"]
        scores[g][str(s)]["O_AP_sym"] = ap([gold[i] == "NEGATIVE_OTHER" for i in ids], [oof_prob[(g, s)][i][3] for i in ids])
        scores[g][str(s)]["O_missed"] = sum(gold[i] == "NEGATIVE_OTHER" and sym[i] != "NEGATIVE_OTHER" for i in ids)
        scores[g][str(s)]["O_false_alarm"] = sum(gold[i] != "NEGATIVE_OTHER" and sym[i] == "NEGATIVE_OTHER" for i in ids)
        scores[g][str(s)]["pred_O"] = sum(sym[i] == "NEGATIVE_OTHER" for i in ids)

# ---------- 配对比较 ----------
res = {}
for mode in ("sym", "plain"):
    preds = {g: [[L2I[oof[(g, s)][mode][i]] for i in ids] for s in SEEDS] for g in ALL}
    boot, pt = boot_wf1(preds, B), point_wf1(preds)
    res[mode] = {f"{g}_minus_B0": compare(boot, pt, g, "B0", B, M) for g in NEW}
main = res["sym"]
mean_scores = {g: {mode: {m: sum(scores[g][str(s)][mode][m] for s in SEEDS) / 3 for m in ("wf1", "softcons", "hardcons")}
                   for mode in ("plain", "sym")} for g in ALL}
o_desc = {g: {k: {str(s): scores[g][str(s)][k] for s in SEEDS} for k in ("O_AP_sym", "O_missed", "O_false_alarm", "pred_O")} for g in ALL}
per_class = {g: {l: sum(scores[g][str(s)]["sym"]["per_class_f1"][l] for s in SEEDS) / 3 for l in LABELS} for g in ALL}
out = {"problems": problems, "b0_source": "Commit 1 归档（B0 复用通过）" if reuse else "本次重跑", "new_tasks": len(updates),
       "scores_by_seed": scores, "scores_mean": mean_scores, "per_class_f1_sym_mean": per_class, "O_descriptive": o_desc,
       "main_comparisons": main, "descriptive_plain": res["plain"], "olat_components": extras["OLat4"], "nli_init": extras["NLIinit"],
       "updates": updates,
       "rule": "臂 − B0（sym WF1）：均值 > 0 且双侧 Bonferroni 98.33% 区间下限 > 0；20,000 次家族重抽样，四组三种子共用；区间不含重新训练的随机性"}
check(len(updates) == (45 if reuse else 60), f"新任务数 {len(updates)} 不对")
json.dump(out, open(f"{R}/claude_recheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("问题：", problems[:20] if problems else "无", len(problems), "| 核对的新任务", len(updates), "（复用 B0 应为 45，重跑应为 60）")
for g in ALL:
    print(f"{g:8s}", {mode: {m: round(v * 100, 2) for m, v in mean_scores[g][mode].items()} for mode in ("plain", "sym")},
          "O-AP", {s: round(v * 100, 2) for s, v in o_desc[g]["O_AP_sym"].items()}, "漏判", o_desc[g]["O_missed"], "误判", o_desc[g]["O_false_alarm"])
for k, v in main.items():
    print(k, {s: round(x * 100, 2) for s, x in v["per_seed"].items()}, round(v["mean"] * 100, 3),
          "98.33%", [round(x * 100, 2) for x in v["ci_bonferroni"]], "提升" if v["improvement"] else "-")
if problems:
    sys.exit(2)
