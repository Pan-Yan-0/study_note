"""Claude 对实验 B Commit 6（v7.2）（DirEnt：把四分类拆成两个方向的蕴含打分；DEC / DEC-H 对 Commit 4 的 NLIinit，英语五折三种子，
Kaggle T4 fp16）的独立复核。运行前写定（11-12，冻结计划变更记录 Commit 6 条目），看结果后不改。

有问题时仍写出诊断，但以非零状态退出，主比较只在 0 个问题时才能引用。
不复用训练脚本的代码，只读每个任务保存的文件：
1. 完整性：DEC、DEC-H 各 15 个任务状态 pass、代码哈希是 v7；有效损失是 dec_bce；预测、内层验证和额外 logits 的哈希和
   done.json 一致；SMOKE 三组各 2 次实际更新（DEC / DEC-H 之前最多 2 次 fp16 跳步，NLIinit 不许跳步）；dec_checks.json 通过。
2. 对照组：SMOKE 的 NLIinit 和 Commit 4 归档逐字节相同（nli_reuse.json）且 15 个 Commit 4 任务的身份核对通过时，用 Commit 4
   的 NLIinit；否则用本次重跑的 NLIinit（这时 15 个任务必须都在）。
3. DEC / DEC-H 和对照组同种子同折：task.json 只差代码身份、group、group_spec、loss 和 config 里的代码说明；group_spec =
   NLIinit 的加上 head（和 DEC 的 head_seed_offset）、loss；训练行数、计划步数、输入数据哈希、v3 数据哈希、模型 revision 相同；
   GPU、精度、torch、cuda、transformers 相同；NLI 载入 200 个张量、278,809,344 个参数；方向头的记录（DEC 新建 1 维，
   DEC-H 是检查点的原三类头、载入时逐位相同）；训练后头的参数有变化。
4. 训练记录：实际更新 + fp16 跳步 = 计划步数；选中轮 = 逐轮内层验证 WF1 最高、同分取早（DEC 用两个顺序合成的四类预测）；
   内层验证按保存的 s_ab / s_ba 用"s > 0 才算蕴含"重算，标签、四类概率和选轮分数都和记录一致。
5. 留出预测：按 s_ab / s_ba 重算四类概率和标签，plain = sym = 规则标签；反向行的 s_ab 和本行的 s_ba 只差分批的浮点误差（报最大差）。
6. 分数：拼成 3,042 行折外预测，用官方评分模块算原方向和对称化的 WF1 / SoftCons / HardCons。
7. 主比较（两条）：DEC − NLIinit、DEC-H − NLIinit，sym WF1。每次重抽样从 1,725 个家族有放回地抽
   （random.Random(20261021).choices，每次抽 1,725 个，和 Commit 3–5 复核相同），同一组抽样在所有组和三个种子之间共用；
   先算每个种子的差，再取三个种子的均值。重抽样 20,000 次，Bonferroni m = 2：双侧 97.5% 区间，取第 ⌊B·0.0125⌋ 个和第
   B−1−⌊B·0.0125⌋ 个次序统计量（从 0 数）。均值和区间下限都严格大于 0 才算提升，用未四舍五入的值判。
   两条都过时，只把对 NLIinit 点估计增益更大的那一臂带进 +2 叠加（同分取 DEC）。
8. 另报（描述，不作判定）：逐种子差、95% 区间；plain 方向的同样比较；SoftCons、HardCons；换序映射违例数（一对可逆行的两个预测
   不是 F ↔ B 镜像）；金标可逆、两个方向都预测成 O 的对数；O → F/B、E → F/B 的行数；逐类 F1；O 的漏判、误判、预测为 O 的行数；
   sym 概率的 ECE（top-label，15 个等宽箱；命中按实际评分的标签，置信度取该标签的概率）；零样本 DEC-H（训练前的 NLI 检查点，3,042 行）的分数；DEC-H − DEC；
   训练、逐轮验证、末轮 logits、最终预测的耗时；和 Commit 1 B0 的差（不同运行，只作参照）。

用法：
  python3 -I code/claude_recheck_augb_c6.py <官方仓库目录>
  python3 -I code/claude_recheck_augb_c6.py <官方仓库目录> --selftest-c4   用同一个重抽样函数重算 Commit 4 的三条主比较
                                                                          （B = 20,000、m = 3），必须和已归档的 claude_recheck.json 相同
输入：metrics/augment_b/train_c6/（结果包原样解压）、train_c1/、train_c4/
输出：metrics/augment_b/train_c6/oof/、claude_recheck.json
"""
import csv, hashlib, json, math, os, random, subprocess, sys, tempfile
from collections import Counter

import numpy as np

OFF = os.path.abspath(sys.argv[1])
SELFTEST = "--selftest-c4" in sys.argv
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = f"{SE}/metrics/augment_b"
R, C1, C4 = f"{AB}/train_c6", f"{AB}/train_c1", f"{AB}/train_c4"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
SWAP = [0, 2, 1, 3]
SEEDS, FOLDS = [42, 43, 44], [0, 1, 2, 3, 4]
B, BOOT_SEED, M = 20000, 20261021, 2
CODE_SHA = "80396fd93f2a910a19407e3e3dd4fd7703c86abe97de8499e5f52fce36659e71"  # train_augb_v7.py v7.2
NLI_SPEC = {"kind": "orig", "init": "nli", "init_model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            "init_revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "init_prefixes": ["deberta.", "pooler."]}
NEW = {"DEC": {**NLI_SPEC, "head": "dec", "head_seed_offset": 30000, "loss": "dec_bce"},
       "DEC-H": {**NLI_SPEC, "head": "dec_h", "loss": "dec_bce"}}
HEAD = {"DEC": ("linear", 1), "DEC-H": ("nli_head", 3)}
VARY = ("code_sha256", "group", "group_spec", "loss", "config")
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


def sig(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def dec_probs(a, b):
    pa, pb = sig(a), sig(b)
    return [pa * pb, pa * (1 - pb), (1 - pa) * pb, (1 - pa) * (1 - pb)]


def dec_label(a, b):
    """An order entails iff its score is > 0 (exactly 0 = non-entailment)."""
    return LABELS[0 if a > 0 and b > 0 else 1 if a > 0 else 2 if b > 0 else 3]


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


def ece(gold_l, probs, preds, bins=15):
    """Top-label ECE: the hit is the label actually scored (for DEC the s > 0 rule, which differs from the four-class argmax at
    ties), the confidence is that label's probability (Codex v7 review)."""
    tot, n = 0.0, len(gold_l)
    edges = [i / bins for i in range(bins + 1)]
    conf = [p[L2I[q]] for p, q in zip(probs, preds)]
    hit = [q == g for q, g in zip(preds, gold_l)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (lo == 0 and c == 0)]
        if idx:
            tot += len(idx) / n * abs(sum(hit[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return tot


# ---------- 官方数据和家族 ----------
ref_rows = rcsv(f"{OFF}/final_data/train/dico_nli_train_track1_reference.csv")
gold = {r["instance_id"]: r["label"] for r in ref_rows}
reverse_of = {r["instance_id"]: r["reverse_pair_id"] for r in ref_rows}
pair_of = {r["instance_id"]: r["pair_id"] for r in ref_rows}
# reverse_pair_id holds the reverse row's instance_id (the c1–c5 rechecks mapped it through pair_id, so their reverse-row
# check never ran; on the Commit 4 / 5 archives r_p equals the reverse row's p exactly in all 38,460 rows)
rev_inst = {i: reverse_of[i] if reverse_of[i] in gold else None for i in gold}
check(sum(v is not None for v in rev_inst.values()) == 2564 and all(rev_inst[v] == i for i, v in rev_inst.items() if v), "反向行映射不对")
fam = rj(f"{SE}/metrics/diagnostics/families.json")
family = {i: fam["family_of_pair"][pair_of[i]] for i in gold}
ids = sorted(gold)
fams = sorted({family[i] for i in gold})
fidx = {f: n for n, f in enumerate(fams)}
gold_idx = np.array([L2I[gold[i]] for i in ids])
fam_of_row = np.array([fidx[family[i]] for i in ids])


# ---------- 共用的家族重抽样（和 Commit 4 / 5 复核相同） ----------
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

# ---------- SMOKE、回执和对照组来源 ----------
for g in ("NLIinit", "DEC", "DEC-H"):
    sm = rj(f"{R}/smoke/{g}/smoke.json")
    sk = sm["skips"]  # v7.2: DEC / DEC-H may skip up to 2 fp16 back-off steps before their 2 actual updates; NLIinit none
    check(sm["status"] == "pass" and sm["code_sha256"] == CODE_SHA and sm["updates"] == 2 and 0 <= sk <= (0 if g == "NLIinit" else 2)
          and len(sm["grad_trace"]) == 2 + sk and sum(x["status"] == "updated" for x in sm["grad_trace"]) == 2,
          f"SMOKE {g} 不是 v7.2 的 2 次实际更新")
dc = rj(f"{R}/dec_checks.json")
check(dc.get("status") == "pass" and dc.get("code_sha256") == CODE_SHA and (dc.get("numeric") or {}).get("status") == "pass"
      and (dc.get("init") or {}).get("status") == "pass", "dec_checks.json 没通过或不是 v7")
check(sha(f"{R}/zeroshot_dech.csv") == (dc.get("zeroshot") or {}).get("sha256"), "zeroshot_dech.csv 哈希和回执不同")
reuse = rj(f"{R}/nli_reuse.json")
ctrl_ok = (dc.get("control_identity") or {}).get("status") == "pass"
rerun = os.path.isdir(f"{R}/tasks/NLIinit")
CTRL = R if rerun else C4
check(rerun == (reuse["status"] != "pass" or not ctrl_ok), f"对照组来源不一致：复用检查 {reuse['status']}、身份核对 {ctrl_ok}、本次重跑 {rerun}")
split = rj(f"{R}/input_data/split_manifest.json")["folds"]
for n in ("manifest.csv", "fold_selection.csv", "split_manifest.json"):
    check(sha(f"{R}/input_data/{n}") == sha(f"{C4}/input_data/{n}"), f"input_data/{n} 和 Commit 4 不同")
check(sha(f"{R}/input_data/split_manifest.json").startswith("f9363290"), "split_manifest 哈希不对")

strip = lambda t, keys: {x: v for x, v in t.items() if x not in keys}


def task_files(td, done):
    for f_, key in (("predictions.csv", "predictions_sha256"), ("val_predictions.csv", "val_predictions_sha256"),
                    ("extra_logits.csv", "extra_logits_sha256")):
        check(sha(f"{td}/{f_}") == done[key], f"{td} {f_} 哈希不对")


def train_record(td, t, done, k):
    check(done.get("status") == "pass", f"{td} 状态 {done.get('status')}")
    check(t["seed"] in SEEDS and t["fold"] == k and not t["smoke"], f"{td} 身份不对")
    check(t["train_rows"] == len(split[str(k)]["fit"]) and t["class_weights"] is None and not done["repeat_counts"], f"{td} 训练行不是原始 fit")
    check(done["actual_optimizer_updates"] + done["skipped_fp16_updates"] == done["planned_updates"] == t["steps"], f"{td} 更新 + 跳步 ≠ 计划")
    ep = done["epochs"]
    check(len(ep) == 12, f"{td} 轮数 {len(ep)}")
    vals = [e["val_original_wf1"] for e in ep]
    check(done["best_epoch"] == vals.index(max(vals)) + 1, f"{td} 选轮不是最早的最高分")


# ---------- 对照组（NLIinit） ----------
tasks, oof, oof_prob, timing = {}, {}, {}, []
viol = {}
for s in SEEDS:
    plain, sym, prob = {}, {}, {}
    for k in FOLDS:
        td = f"{CTRL}/tasks/NLIinit/seed{s}/fold{k}"
        t, done = rj(f"{td}/task.json"), rj(f"{td}/done.json")
        tasks[("NLIinit", s, k)] = (t, done)
        check(t["group"] == "NLIinit" and t["group_spec"] == NLI_SPEC and t.get("loss", "ce") == "ce", f"{td} 不是 NLIinit")
        task_files(td, done)
        train_record(td, t, done, k)
        rows = rcsv(f"{td}/predictions.csv")
        check(sorted(r["instance_id"] for r in rows) == sorted(split[str(k)]["test"]), f"{td} 留出行不对")
        byid = {r["instance_id"]: r for r in rows}
        for r in rows:
            rv = rev_inst[r["instance_id"]]
            if rv:
                check(rv in byid and max(abs(float(r["r_p_" + c]) - float(byid[rv]["p_" + c])) for c in "EFBO") < 1e-6,
                      f"{td} {r['instance_id']} r_p 不等于反向行的 p")
            p = [float(r["p_" + c]) for c in "EFBO"]
            q = [float(r["r_p_" + c]) for c in "EFBO"]
            sp = [(p[j] + q[SWAP[j]]) / 2 for j in range(4)]
            check(r["plain"] == argmax(p) and r["sym"] == argmax(sp), f"{r['instance_id']} NLIinit 保存的标签和概率不一致")
            plain[r["instance_id"]], sym[r["instance_id"]], prob[r["instance_id"]] = r["plain"], r["sym"], sp
        timing.append({"group": "NLIinit", "seed": s, "fold": k, "runtime_sec": done["runtime_sec"], **(done.get("timing") or {})})
    oof[("NLIinit", s)] = {"plain": plain, "sym": sym}
    oof_prob[("NLIinit", s)] = prob

# ---------- DEC / DEC-H ----------
score_gap = []
for g, spec in NEW.items():
    for s in SEEDS:
        plain, prob = {}, {}
        s_pair = {}
        for k in FOLDS:
            td = f"{R}/tasks/{g}/seed{s}/fold{k}"
            t, done = rj(f"{td}/task.json"), rj(f"{td}/done.json")
            tasks[(g, s, k)] = (t, done)
            ct, cd = tasks[("NLIinit", s, k)]
            check(strip(t, VARY) == strip(ct, VARY), f"{td} task.json 和对照组不只差代码身份、组、损失和 config")
            check(strip(t["config"], CFG_VARY) == strip(ct["config"], CFG_VARY), f"{td} config 和对照组不只差代码说明")
            check(t["code_sha256"] == CODE_SHA and t["group"] == g and t["group_spec"] == spec and t["loss"] == "dec_bce", f"{td} 代码、组、group_spec 或损失不对")
            check(all(done["environment"].get(e) == cd["environment"].get(e) for e in ENV_KEYS), f"{td} 环境和对照组不同")
            task_files(td, done)
            train_record(td, t, done, k)
            ar = done.get("arm_report") or {}
            ni, di = ar.get("nli_init") or {}, ar.get("dec_init") or {}
            check(ar.get("global_rng_restored") is True and ni.get("revision") == NLI_SPEC["init_revision"] and ni.get("tensors_loaded") == 200
                  and ni.get("parameters_loaded") == 278809344 and ni.get("kept_from_b0") == ["classifier.bias", "classifier.weight"],
                  f"{td} NLI 载入记录不对")
            check((di.get("mode"), di.get("n_out")) == HEAD[g] and (g == "DEC" or di.get("head_equal_to_checkpoint") is True), f"{td} 方向头记录不对")
            hc = done.get("dec_head_change") or {}
            check(hc.get("proj.weight", 0) > 0, f"{td} 训练后头的参数没有变化")
            # inner validation: rule labels, probabilities and the selection score from the saved scores
            vrows = rcsv(f"{td}/val_predictions.csv")
            check(sorted(r["instance_id"] for r in vrows) == sorted(split[str(k)]["val"]), f"{td} 内层验证行不对")
            vp = []
            for r in vrows:
                a, b = float(r["s_ab"]), float(r["s_ba"])
                lab = dec_label(a, b)
                vp.append(lab)
                check(r["pred"] == lab, f"{td} 内层验证 pred 不是 s > 0 规则")
                check(max(abs(x - float(r["p_" + c])) for x, c in zip(dec_probs(a, b), "EFBO")) < 2e-9, f"{td} 内层验证概率和 s 不一致")
            vw = wf1([gold[r["instance_id"]] for r in vrows], vp)
            check(abs(vw - done["best_val_original_wf1"]) < 1e-9, f"{td} 内层验证重算 {vw} ≠ 记录")
            rows = rcsv(f"{td}/predictions.csv")
            check(sorted(r["instance_id"] for r in rows) == sorted(split[str(k)]["test"]), f"{td} 留出行不对")
            for r in rows:
                i = r["instance_id"]
                a, b = float(r["s_ab"]), float(r["s_ba"])
                lab, p = dec_label(a, b), dec_probs(a, b)
                check(r["label"] == gold[i], f"{i} 金标不对")
                check(r["plain"] == lab and r["sym"] == lab, f"{td} {i} 保存的标签不是 s > 0 规则")
                for pre in ("p_", "sym_p_"):
                    check(max(abs(x - float(r[pre + c])) for x, c in zip(p, "EFBO")) < 2e-9, f"{td} {i} {pre} 概率和 s 不一致")
                check(max(abs(x - float(r["r_p_" + c])) for x, c in zip(dec_probs(b, a), "EFBO")) < 2e-9, f"{td} {i} 反方向概率和 s 不一致")
                plain[i], prob[i], s_pair[i] = lab, p, (a, b)
            timing.append({"group": g, "seed": s, "fold": k, "runtime_sec": done["runtime_sec"], **(done.get("timing") or {})})
        for i, (a, b) in s_pair.items():  # the reverse row's own s_ab is the same score as this row's s_ba (up to batching noise)
            rv = rev_inst[i]
            if rv:
                score_gap.append(abs(s_pair[rv][0] - b))
        oof[(g, s)] = {"plain": plain, "sym": dict(plain)}
        oof_prob[(g, s)] = prob
check(not score_gap or max(score_gap) < 1e-3, f"反向行的 s_ab 和本行的 s_ba 最大差 {max(score_gap) if score_gap else 0}，超过分批误差")
check(len(tasks) == 45, f"任务数 {len(tasks)} 不对（应为 45：对照组 15 + 两臂各 15）")


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


def structure(m):
    """Swap-mapping violations, reversible gold pairs predicted O in both orders, and O / E -> F / B rows."""
    pairs = {tuple(sorted((i, rev_inst[i]))) for i in ids if rev_inst[i]}
    mirror = {LABELS[j]: LABELS[SWAP[j]] for j in range(4)}
    return {"reversible_pairs": len(pairs),
            "swap_mapping_violations": sum(m[b] != mirror[m[a]] for a, b in pairs),
            "reversible_gold_both_O": sum(m[a] == m[b] == "NEGATIVE_OTHER" for a, b in pairs),
            "O_to_F_or_B": sum(gold[i] == "NEGATIVE_OTHER" and m[i] in LABELS[1:3] for i in ids),
            "E_to_F_or_B": sum(gold[i] == "EQUIVALENCE" and m[i] in LABELS[1:3] for i in ids),
            "single_O_rows_predicted_F_or_B": sum(rev_inst[i] is None and m[i] in LABELS[1:3] for i in ids)}


os.makedirs(f"{R}/oof", exist_ok=True)
GROUPS = ["NLIinit", "DEC", "DEC-H"]
scores = {}
for g in GROUPS:
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
            o["structure"] = structure(m)
            scores[g][str(s)][mode] = o
        sym, pr = oof[(g, s)]["sym"], oof_prob[(g, s)]
        scores[g][str(s)]["O_AP_sym"] = ap([gold[i] == "NEGATIVE_OTHER" for i in ids], [pr[i][3] for i in ids])
        scores[g][str(s)]["O_missed"] = sum(gold[i] == "NEGATIVE_OTHER" and sym[i] != "NEGATIVE_OTHER" for i in ids)
        scores[g][str(s)]["O_false_alarm"] = sum(gold[i] != "NEGATIVE_OTHER" and sym[i] == "NEGATIVE_OTHER" for i in ids)
        scores[g][str(s)]["pred_O"] = sum(sym[i] == "NEGATIVE_OTHER" for i in ids)
        scores[g][str(s)]["ECE_sym"] = ece([gold[i] for i in ids], [pr[i] for i in ids], [sym[i] for i in ids])
if not rerun:  # the reused control's OOF must equal the Commit 4 archive
    for s in SEEDS:
        for mode in ("plain", "sym"):
            check(read_oof(f"{C4}/oof/NLIinit_seed{s}_{mode}.csv") == [L2I[oof[("NLIinit", s)][mode][i]] for i in ids],
                  f"复用的 NLIinit seed{s} {mode} 和 Commit 4 归档的 OOF 不同")

# ---------- 零样本 DEC-H（训练前的 NLI 检查点） ----------
zs = {r["instance_id"]: (float(r["s_ab"]), float(r["s_ba"])) for r in rcsv(f"{R}/zeroshot_dech.csv")}
check(sorted(zs) == ids, "zeroshot_dech.csv 不是 3,042 行")
zm = {i: dec_label(*zs[i]) for i in ids}
zero = official(zm)
zero.update(structure=structure(zm), pred_dist=dict(Counter(zm.values())), ECE=ece([gold[i] for i in ids], [dec_probs(*zs[i]) for i in ids], [zm[i] for i in ids]))
zero_gap = max(abs(zs[rev_inst[i]][0] - zs[i][1]) for i in ids if rev_inst[i])

# ---------- 比较 ----------
res = {}
for mode in ("sym", "plain"):
    preds = {g: [[L2I[oof[(g, s)][mode][i]] for i in ids] for s in SEEDS] for g in GROUPS}
    preds["B0(C1)"] = [read_oof(f"{C1}/oof/B0_seed{s}_{mode}.csv") for s in SEEDS]
    boot, pt = boot_wf1(preds, B), point_wf1(preds)
    res[mode] = {"main": {f"{a}_minus_NLIinit": compare(boot, pt, a, "NLIinit", B, M) for a in ("DEC", "DEC-H")},
                 "descriptive": {f"{a}_minus_{b}": compare(boot, pt, a, b, B, 1) for a, b in
                                 (("DEC-H", "DEC"), ("DEC", "B0(C1)"), ("DEC-H", "B0(C1)"), ("NLIinit", "B0(C1)"))},
                 "point": {g: [float(x) for x in v] for g, v in pt.items()}}
main = res["sym"]["main"]
passed = [a for a in ("DEC", "DEC-H") if main[f"{a}_minus_NLIinit"]["improvement"]]
pick = None
if passed:
    pick = max(passed, key=lambda a: (main[f"{a}_minus_NLIinit"]["mean"], a == "DEC"))
mean_scores = {g: {mode: {m: sum(scores[g][str(s)][mode][m] for s in SEEDS) / 3 for m in ("wf1", "softcons", "hardcons")}
                   for mode in ("plain", "sym")} for g in GROUPS}
per_class = {g: {l: sum(scores[g][str(s)]["sym"]["per_class_f1"][l] for s in SEEDS) / 3 for l in LABELS} for g in GROUPS}
desc_keys = ("O_AP_sym", "O_missed", "O_false_alarm", "pred_O", "ECE_sym")
o_desc = {g: {k: {str(s): scores[g][str(s)][k] for s in SEEDS} for k in desc_keys} for g in GROUPS}
struct = {g: {mode: {k: {str(s): scores[g][str(s)][mode]["structure"][k] for s in SEEDS} for k in scores[g]["42"][mode]["structure"]}
              for mode in ("plain", "sym")} for g in GROUPS}
tmean = {}
for g in GROUPS:
    rows_ = [t for t in timing if t["group"] == g]
    tmean[g] = {k: sum(t.get(k, 0) for t in rows_) / len(rows_) for k in ("runtime_sec", "train_sec", "epoch_validation_sec",
                                                                             "last_epoch_logits_sec", "final_prediction_sec") if all(k in t for t in rows_)}
upd = [{"group": g, "seed": s, "fold": k, "planned": d["planned_updates"], "actual": d["actual_optimizer_updates"],
        "skipped": d["skipped_fp16_updates"], "best_epoch": d["best_epoch"]} for (g, s, k), (t, d) in sorted(tasks.items())]
out = {"problems": problems, "tasks": len(tasks), "control_source": "Commit 6 rerun" if rerun else "Commit 4 (reused)",
       "nli_reuse": {"status": reuse["status"], "differences": reuse["differences"]}, "control_identity": dc.get("control_identity"),
       "scores_by_seed": scores, "scores_mean": mean_scores, "per_class_f1_sym_mean": per_class, "O_descriptive": o_desc,
       "structure": struct, "main_comparisons": main, "passed": passed, "pick_for_plus2": pick,
       "plain": res["plain"], "descriptive_sym": res["sym"]["descriptive"], "point_sym": res["sym"]["point"],
       "zero_shot_DEC-H": zero, "reverse_score_gap_max": {"trained": max(score_gap) if score_gap else None, "zero_shot": zero_gap},
       "timing_mean_sec": tmean, "updates": upd,
       "rule": "DEC − NLIinit、DEC-H − NLIinit（sym WF1）：均值 > 0 且 Bonferroni m = 2 的双侧 97.5% 区间下限 > 0；20,000 次家族重抽样，"
               "各组三种子共用；区间不含重新训练的随机性；两条都过时只把点估计增益更大的一臂带进 +2（同分取 DEC）"}
json.dump(out, open(f"{R}/claude_recheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("问题：", problems[:20] if problems else "无", len(problems), "| 核对的任务", len(tasks), "（应为 45）| 对照组", out["control_source"])
for g in GROUPS:
    print(f"{g:8s}", {mode: {m: round(v * 100, 2) for m, v in mean_scores[g][mode].items()} for mode in ("plain", "sym")},
          "O-AP", {s: round(v * 100, 2) for s, v in o_desc[g]["O_AP_sym"].items()}, "漏判", o_desc[g]["O_missed"], "误判", o_desc[g]["O_false_alarm"])
for k, v in main.items():
    print("主比较", k, {s: round(x * 100, 2) for s, x in v["per_seed"].items()}, round(v["mean"] * 100, 3),
          "97.5%", [round(x * 100, 2) for x in v["ci_bonferroni"]], "提升" if v["improvement"] else "-")
print("进入 +2 的一臂：", pick)
for k, v in res["sym"]["descriptive"].items():
    print("  描述", k, round(v["mean"] * 100, 2), [round(x * 100, 2) for x in v["ci95"]])
print("零样本 DEC-H：WF1 %.2f SoftCons %.2f HardCons %.2f" % (zero["wf1"] * 100, zero["softcons"] * 100, zero["hardcons"] * 100))
print("结构（sym，seed 42）：", {g: scores[g]["42"]["sym"]["structure"] for g in GROUPS})
print("耗时均值（秒）：", {g: {k: round(v, 1) for k, v in t.items()} for g, t in tmean.items()})
if problems:
    sys.exit(2)
