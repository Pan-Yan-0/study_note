"""Claude 对实验 B Commit 5（B2 的数据上换 NLI 检查点初始化：B2-NLIinit 对同一次运行重跑的 B2-Haiku，英语五折三种子，Kaggle T4 fp16）的独立复核。
运行前写定（11-12，冻结计划变更记录 Commit 5 条目），看结果后不改。

有问题时仍写出诊断，但以非零状态退出，主比较只在 0 个问题时才能引用。
不复用训练脚本的代码，只读每个任务保存的文件：
1. 完整性：30 个任务（B2-Haiku、B2-NLIinit 各 15）状态 pass、代码哈希是 v6；有效损失是 CE；预测和额外 logits 的哈希
   和 done.json 一致；SMOKE 两组各 2 次实际更新。
2. 和 Commit 2 归档的 B2（同种子同折）对照：task.json 只差代码身份、group、group_spec、loss（config 只差 code_version、
   loss_variants、arm_variants）；训练行数、计划步数、v3 数据哈希、输入数据哈希都相同。
3. B2-NLIinit 和本次 B2-Haiku（同种子同折）：task.json 只差 group 和 group_spec，group_spec = B2 的加上 Commit 4 NLIinit
   的四个 init 字段；GPU、精度、torch、cuda、transformers 相同；NLI 载入 200 个张量、278,809,344 个参数，只保留 classifier。
   两组计划更新次数相同；实际更新和 fp16 跳步逐 fold / seed 报告两组的差（描述，跳步时 scheduler 也不前进）。
4. 训练记录：实际更新 + fp16 跳步 = 计划步数；选中轮 = 内层验证原方向 WF1 最高、同分取早；内层验证预测重算等于记录。
5. 分数：拼成 3,042 行折外预测，用官方评分模块算原方向和对称化的 WF1 / SoftCons / HardCons。
6. 主比较（只有一条）：B2-NLIinit − B2-Haiku（本次重跑），sym WF1。每次重抽样从 1,725 个家族有放回地抽
   （random.Random(20261021).choices，每次抽 1,725 个，和 Commit 3 / 4 复核相同），同一组抽样在所有组和三个种子之间共用；
   先算每个种子的差，再取三个种子的均值。重抽样 20,000 次，双侧 95% 区间，取第 ⌊B·0.025⌋ 个和第 B−1−⌊B·0.025⌋ 个次序统计量
   （从 0 数）。均值和区间下限都严格大于 0 才算提升，用未四舍五入的值判。
7. 另报（描述，不作判定）：逐种子差；plain 方向的同样比较；O 的 AP、漏判、误判、预测为 O 的行数、逐类 F1；
   本次 B2-Haiku 和 Commit 2 的 B2（预测文件逐字节相同的任务数、逐种子 sym WF1 差）；B2-NLIinit 和 Commit 4 的 NLIinit、
   Commit 1 的 B0 的差（不同运行，只作参照）。

用法：
  python3 -I code/claude_recheck_augb_c5.py <官方仓库目录>
  python3 -I code/claude_recheck_augb_c5.py <官方仓库目录> --selftest-c4   用同一个重抽样函数重算 Commit 4 的三条主比较
                                                                          （B = 20,000、m = 3），必须和已归档的 claude_recheck.json 相同
输入：metrics/augment_b/train_c5/（结果包原样解压）、train_c1/、train_c2/、train_c4/
输出：metrics/augment_b/train_c5/oof/、claude_recheck.json
"""
import csv, hashlib, json, os, random, subprocess, sys, tempfile
from collections import Counter

import numpy as np

OFF = os.path.abspath(sys.argv[1])
SELFTEST = "--selftest-c4" in sys.argv
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = f"{SE}/metrics/augment_b"
R, C1, C2, C4 = f"{AB}/train_c5", f"{AB}/train_c1", f"{AB}/train_c2", f"{AB}/train_c4"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
SWAP = [0, 2, 1, 3]
SEEDS, FOLDS = [42, 43, 44], [0, 1, 2, 3, 4]
B, BOOT_SEED, M = 20000, 20261021, 1
CODE_SHA = "50c800620fbc9d5f53dcad5dffba3c8821b83aee28f7d567554dd887488e60ad"  # train_augb_v6.py v6.1
B2_SPEC = {"kind": "gen", "k": 2}
NLI_INIT = {"init": "nli", "init_model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            "init_revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "init_prefixes": ["deberta.", "pooler."]}
NEW = {"B2-Haiku": B2_SPEC, "B2-NLIinit": {**B2_SPEC, **NLI_INIT}}
VARY_ARCH = ("code_sha256", "group", "group_spec", "loss", "config")
CFG_VARY = ("code_version", "loss_variants", "arm_variants")
ENV_KEYS = ("gpu", "precision", "torch", "cuda", "transformers")
eff_loss = lambda t: t.get("loss", "ce")
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


# ---------- 共用的家族重抽样（和 Commit 4 复核相同） ----------
def boot_wf1(preds, n_boot, chunk=500):
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
    preds = {g: [read_oof(f"{(C1 if g == 'B0' else C4)}/oof/{g}_seed{s}_sym.csv") for s in SEEDS] for g in ("B0", "OLat4", "NLIinit", "BiBCE")}
    boot = boot_wf1(preds, 20000)
    pt = point_wf1(preds)
    ref = rj(f"{C4}/claude_recheck.json")["main_comparisons"]
    worst = 0.0
    for g in ("OLat4", "NLIinit", "BiBCE"):
        c, r = compare(boot, pt, g, "B0", 20000, 3), ref[f"{g}_minus_B0"]
        worst = max([worst, abs(c["mean"] - r["mean"])] + [abs(x - y) for x, y in zip(c["ci_bonferroni"], r["ci_bonferroni"])]
                    + [abs(c["per_seed"][s] - r["per_seed"][s]) for s in r["per_seed"]] + [abs(x - y) for x, y in zip(c["ci95"], r["ci95"])])
        print(g, round(c["mean"] * 100, 4), "98.33%", [round(x * 100, 4) for x in c["ci_bonferroni"]], "archived", [round(x * 100, 4) for x in r["ci_bonferroni"]])
    print("selftest max abs difference vs archived Commit 4 recheck:", worst)
    sys.exit(0 if worst <= 1e-12 else 1)

# ---------- SMOKE 和输入 ----------
for g in NEW:
    sm = rj(f"{R}/smoke/{g}/smoke.json")
    check(sm["status"] == "pass" and sm["code_sha256"] == CODE_SHA and sum(x["status"] == "updated" for x in sm["grad_trace"]) == 2,
          f"SMOKE {g} 不是 v6 的 2 次实际更新")
split = rj(f"{R}/input_data/split_manifest.json")["folds"]
check(sha(f"{R}/input_data/split_manifest.json").startswith("f9363290"), "split_manifest 哈希不对")
for n in ("manifest.csv", "fold_selection.csv", "split_manifest.json"):
    check(sha(f"{R}/input_data/{n}") == sha(f"{C2}/input_data/{n}"), f"input_data/{n} 和 Commit 2 不同")

# ---------- 逐任务 ----------
tasks, oof, oof_prob, updates, nli_rec, repro = {}, {}, {}, [], [], []
arch = {(s, k): (rj(f"{C2}/tasks/B2-Haiku/seed{s}/fold{k}/task.json"), rj(f"{C2}/tasks/B2-Haiku/seed{s}/fold{k}/done.json")) for s in SEEDS for k in FOLDS}
strip = lambda t, keys: {x: v for x, v in t.items() if x not in keys}


def load_new(g, spec):
    for s in SEEDS:
        plain, sym, prob = {}, {}, {}
        for k in FOLDS:
            td = f"{R}/tasks/{g}/seed{s}/fold{k}"
            t, done = rj(f"{td}/task.json"), rj(f"{td}/done.json")
            tasks[(g, s, k)] = (t, done)
            at, ad = arch[(s, k)]
            check(strip(t, VARY_ARCH) == strip(at, VARY_ARCH), f"{td} task.json 和 Commit 2 的 B2 不只差代码身份、组和 config")
            check(strip(t["config"], CFG_VARY) == strip(at["config"], CFG_VARY), f"{td} config 和 Commit 2 的 B2 不只差代码说明")
            check(t["code_sha256"] == CODE_SHA and t["group"] == g and t["group_spec"] == spec, f"{td} 代码、组或 group_spec 不对")
            check(eff_loss(t) == "ce" and eff_loss(at) == "ce", f"{td} 有效损失不是 CE")
            check(t["seed"] == s and t["fold"] == k and not t["smoke"], f"{td} 身份不对")
            if g == "B2-NLIinit":
                bt, bd = tasks[("B2-Haiku", s, k)]
                check(strip(t, ("group", "group_spec")) == strip(bt, ("group", "group_spec")), f"{td} task.json 和本次 B2-Haiku 不只差组")
                check(all(done["environment"].get(e) == bd["environment"].get(e) for e in ENV_KEYS), f"{td} 环境和本次 B2-Haiku 不同")
                ni = (done.get("arm_report") or {}).get("nli_init") or {}
                check(ni.get("revision") == NLI_INIT["init_revision"] and ni.get("tensors_loaded") == 200 and ni.get("parameters_loaded") == 278809344,
                      f"{td} NLI 载入记录不对：{ni.get('tensors_loaded')} / {ni.get('parameters_loaded')}")
                check(ni.get("kept_from_b0") == ["classifier.bias", "classifier.weight"], f"{td} 保留 B0 的键不是只有 classifier")
                nli_rec.append({"seed": s, "fold": k, "tensors_loaded": ni.get("tensors_loaded"), "parameters_loaded": ni.get("parameters_loaded"),
                                "forward_config_differences_kept_b0": ni.get("forward_config_differences_kept_b0")})
            else:
                same = {f_: sha(f"{td}/{f_}") == sha(f"{C2}/tasks/B2-Haiku/seed{s}/fold{k}/{f_}") for f_ in ("predictions.csv", "val_predictions.csv", "extra_logits.csv")}
                repro.append({"seed": s, "fold": k, **same, "best_epoch": [done["best_epoch"], ad["best_epoch"]],
                              "env_same_as_c2": all(done["environment"].get(e) == ad["environment"].get(e) for e in ENV_KEYS)})
            check(done.get("status") == "pass", f"{td} 状态 {done.get('status')}")
            for f_, key in (("predictions.csv", "predictions_sha256"), ("val_predictions.csv", "val_predictions_sha256"),
                            ("extra_logits.csv", "extra_logits_sha256")):
                check(sha(f"{td}/{f_}") == done[key], f"{td} {f_} 哈希不对")
            check(t["train_rows"] > len(split[str(k)]["fit"]), f"{td} 训练行数没有生成数据")
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
            rows = rcsv(f"{td}/predictions.csv")
            byid = {r["instance_id"]: r for r in rows}
            check(sorted(byid) == sorted(split[str(k)]["test"]), f"{td} 留出行不对")
            updates.append({"group": g, "seed": s, "fold": k, "planned": done["planned_updates"], "actual": done["actual_optimizer_updates"],
                            "skipped": done["skipped_fp16_updates"], "best_epoch": done["best_epoch"], "runtime_sec": done["runtime_sec"]})
            for r in rows:
                i = r["instance_id"]
                p = [float(r["p_" + c]) for c in "EFBO"]
                q = [float(r["r_p_" + c]) for c in "EFBO"]
                sp = [(p[j] + q[SWAP[j]]) / 2 for j in range(4)]
                check(r["label"] == gold[i], f"{i} 金标不对")
                rv = inst_of_pair.get(reverse_of[i]) if reverse_of[i] else None
                if rv:
                    check(max(abs(a - float(byid[rv]["p_" + c])) for a, c in zip(q, "EFBO")) < 1e-9, f"{i} r_p 不等于反方向的 p")
                check(max(abs(a - float(r["sym_p_" + c])) for a, c in zip(sp, "EFBO")) < 2e-9, f"{i} sym 概率不对")
                check(r["plain"] == argmax(p) and r["sym"] == argmax(sp), f"{i} 保存的标签和概率不一致")
                plain[i], sym[i], prob[i] = r["plain"], r["sym"], sp
        oof[(g, s)] = {"plain": plain, "sym": sym}
        oof_prob[(g, s)] = prob


for g, spec in NEW.items():  # B2-Haiku 先读，B2-NLIinit 和它逐任务对照
    load_new(g, spec)


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
scores = {}
for g in NEW:
    scores[g] = {}
    for s in SEEDS:
        scores[g][str(s)] = {}
        for mode in ("plain", "sym"):
            m = oof[(g, s)][mode]
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

# ---------- 比较 ----------
REFS = {"B2-Haiku(C2)": f"{C2}/oof/B2-Haiku_seed{{s}}_{{mode}}.csv", "NLIinit(C4)": f"{C4}/oof/NLIinit_seed{{s}}_{{mode}}.csv",
        "B0(C1)": f"{C1}/oof/B0_seed{{s}}_{{mode}}.csv"}
res = {}
for mode in ("sym", "plain"):
    preds = {g: [[L2I[oof[(g, s)][mode][i]] for i in ids] for s in SEEDS] for g in NEW}
    preds.update({n: [read_oof(p.format(s=s, mode=mode)) for s in SEEDS] for n, p in REFS.items()})
    boot, pt = boot_wf1(preds, B), point_wf1(preds)
    res[mode] = {"B2-NLIinit_minus_B2-Haiku": compare(boot, pt, "B2-NLIinit", "B2-Haiku", B, M),
                 "descriptive": {f"{a}_minus_{b}": compare(boot, pt, a, b, B, M) for a, b in
                                 (("B2-Haiku", "B2-Haiku(C2)"), ("B2-NLIinit", "NLIinit(C4)"), ("B2-NLIinit", "B0(C1)"), ("B2-Haiku", "B0(C1)"))},
                 "point": {g: [float(x) for x in v] for g, v in pt.items()}}
main = res["sym"]["B2-NLIinit_minus_B2-Haiku"]
mean_scores = {g: {mode: {m: sum(scores[g][str(s)][mode][m] for s in SEEDS) / 3 for m in ("wf1", "softcons", "hardcons")}
                   for mode in ("plain", "sym")} for g in NEW}
o_desc = {g: {k: {str(s): scores[g][str(s)][k] for s in SEEDS} for k in ("O_AP_sym", "O_missed", "O_false_alarm", "pred_O")} for g in NEW}
per_class = {g: {l: sum(scores[g][str(s)]["sym"]["per_class_f1"][l] for s in SEEDS) / 3 for l in LABELS} for g in NEW}
check(len(updates) == 30, f"任务数 {len(updates)} 不对")
by = {(u["group"], u["seed"], u["fold"]): u for u in updates}
update_diff = [{"seed": s, "fold": k, "planned": by[("B2-Haiku", s, k)]["planned"],
                "actual_B2-Haiku": by[("B2-Haiku", s, k)]["actual"], "actual_B2-NLIinit": by[("B2-NLIinit", s, k)]["actual"],
                "actual_diff": by[("B2-NLIinit", s, k)]["actual"] - by[("B2-Haiku", s, k)]["actual"],
                "skipped_B2-Haiku": by[("B2-Haiku", s, k)]["skipped"], "skipped_B2-NLIinit": by[("B2-NLIinit", s, k)]["skipped"]}
               for s in SEEDS for k in FOLDS if ("B2-Haiku", s, k) in by and ("B2-NLIinit", s, k) in by]
for u in update_diff:
    check(by[("B2-NLIinit", u["seed"], u["fold"])]["planned"] == u["planned"], f"seed{u['seed']} fold{u['fold']} 两组计划更新次数不同")
out = {"problems": problems, "tasks": len(updates), "scores_by_seed": scores, "scores_mean": mean_scores, "per_class_f1_sym_mean": per_class,
       "O_descriptive": o_desc, "main_comparison": main, "plain": res["plain"], "descriptive_sym": res["sym"]["descriptive"],
       "point_sym": res["sym"]["point"], "b2_reproduction_vs_c2": repro, "update_diff": update_diff, "nli_init": nli_rec, "updates": updates,
       "rule": "B2-NLIinit − B2-Haiku（本次重跑，sym WF1）：均值 > 0 且双侧 95% 区间下限 > 0；20,000 次家族重抽样，各组三种子共用；区间不含重新训练的随机性"}
json.dump(out, open(f"{R}/claude_recheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("问题：", problems[:20] if problems else "无", len(problems), "| 核对的任务", len(updates), "（应为 30）")
for g in NEW:
    print(f"{g:11s}", {mode: {m: round(v * 100, 2) for m, v in mean_scores[g][mode].items()} for mode in ("plain", "sym")},
          "O-AP", {s: round(v * 100, 2) for s, v in o_desc[g]["O_AP_sym"].items()}, "漏判", o_desc[g]["O_missed"], "误判", o_desc[g]["O_false_alarm"])
print("主比较 B2-NLIinit − B2-Haiku", {s: round(x * 100, 2) for s, x in main["per_seed"].items()}, round(main["mean"] * 100, 3),
      "95%", [round(x * 100, 2) for x in main["ci95"]], "提升" if main["improvement"] else "-")
for k, v in res["sym"]["descriptive"].items():
    print("  描述", k, round(v["mean"] * 100, 2), [round(x * 100, 2) for x in v["ci95"]])
print("B2 重跑和 Commit 2 逐字节相同的预测文件：", sum(r["predictions.csv"] for r in repro), "/ 15")
print("实际更新差（B2-NLIinit − B2-Haiku）逐 seed / fold：", [u["actual_diff"] for u in update_diff],
      "| 跳步 B2-Haiku", [u["skipped_B2-Haiku"] for u in update_diff], "B2-NLIinit", [u["skipped_B2-NLIinit"] for u in update_diff])
if problems:
    sys.exit(2)
