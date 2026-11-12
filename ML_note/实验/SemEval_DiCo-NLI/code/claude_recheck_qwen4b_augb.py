"""Claude 对 Qwen 4B +1 扩充阶段 1（Q0 / Q-B1，英语五折三种子，A10 bf16，qwen4b_augb v1.1）的独立复核。
判定按 code/qwen4b_augb/README.md "事先登记的判定" 和 "阶段 1 结果的分析"（运行前写定）：
- 主比较 Q-B1 − Q0（sym WF1）：同种子差的均值；1,725 个家族重抽样 2,000 次（random.Random(20261021).choices），
  两组和三个种子共用同一组抽到的家族；双侧 95% 区间；均值 > 0 且下限 > 0（未四舍五入）才算阶段 1 通过。
- 描述：plain 方向、SoftCons / HardCons、逐类 F1、O 的 AP（对称化 p_O，每个种子完整 OOF）、steps / fit_tokens / best_epoch；
  Q0 − JOINT-QWEN（同配方重跑）；Q-B1 和 mDeBERTa 的 B1 / B2（系统比较）。
核对：结果包 MANIFEST 哈希；任务身份和代码身份；留出 = 家族划分第 k 折、训练和留出不共享家族；Q-B1 的 fit = Q0 的 fit
加上该折 fold_selection 的生成行（顺序也相同）；从 cache.jsonl 重建留出和 dev 的预测，和保存的逐折预测、OOF CSV 逐条比对，
OOF 分数和保存的分数一致。
用法：python3 -I code/claude_recheck_qwen4b_augb.py <官方仓库目录>；有问题时写出诊断并以状态 2 退出。
"""
import csv, hashlib, json, os, random, subprocess, sys, tempfile
from collections import Counter, defaultdict

OFF = os.path.abspath(sys.argv[1])
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{SE}/metrics/qwen4b_augb_s1"
J = f"{SE}/metrics/joint_validation"
AB = f"{SE}/metrics/augment_b"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SWAP = [0, 2, 1, 3]
SEEDS, FOLDS, GROUPS = [42, 43, 44], range(5), ["Q0", "Q-B1"]
PIPELINE = "adc19fa00971106514ee9c16e5901d76a78d9bac5fe68438aa4454a4e7748541"
B, BOOT_SEED = 2000, 20261021
problems = []


def check(c, m):
    if not c:
        problems.append(m)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rj(p):
    return json.load(open(p, encoding="utf-8"))


def rcsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def read_pred(p):
    return {r["instance_id"]: r["label"] for r in rcsv(p)}


rows = rcsv(f"{OFF}/final_data/train/dico_nli_train_track1_participant_labeled.csv")
dev_rows = rcsv(f"{OFF}/final_data/dev/dico_nli_dev_track1_participant_labeled.csv")
fam = rj(f"{SE}/metrics/diagnostics/families.json")
family = {r["instance_id"]: fam["family_of_pair"][r["pair_id"]] for r in rows}
fold_of = {i: int(fam["fold_of_family"][str(f)]) for i, f in family.items()}
gold = {r["instance_id"]: r["label"] for r in rows}
byid = {r["instance_id"]: r for r in rows}
dev_byid = {r["instance_id"]: r for r in dev_rows}
ids_all = sorted(gold)
fams = sorted(set(family.values()))
ids_of = defaultdict(list)
for i in ids_all:
    ids_of[family[i]].append(i)

# ---------- 结果包完整性 ----------
man = rj(f"{R}/MANIFEST.json")
check(man["pipeline"] == PIPELINE, "MANIFEST 代码身份不对")
bad = [f for f, h in man["files"].items() if not os.path.exists(f"{R}/{f}") or sha(f"{R}/{f}") != h]
check(not bad, f"MANIFEST 哈希不符 {bad[:5]}")
sel = defaultdict(list)
for r in rcsv(f"{R}/source/augb_data/fold_selection.csv"):
    sel[int(r["fold"])].append("gen:" + r["sample_id"])


def predict(prob, ids, table):
    plain, sym, psym = {}, {}, {}
    for i in ids:
        r = table[i]
        p, q = prob[(r["text1"], r["text2"])], prob[(r["text2"], r["text1"])]
        s = [(p[k] + q[SWAP[k]]) / 2 for k in range(4)]
        plain[i] = LABELS[max(range(4), key=lambda k: p[k])]
        sym[i] = LABELS[max(range(4), key=lambda k: s[k])]
        psym[i] = s
    return plain, sym, psym


def official(pred, gold_csv):
    d = tempfile.mkdtemp()
    g = rcsv(gold_csv)
    with open(f"{d}/g.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["instance_id", "pair_id", "reverse_pair_id", "label"])
        for r in g: w.writerow([r["instance_id"], r["pair_id"], r["reverse_pair_id"], r["label"]])
    with open(f"{d}/p.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["instance_id", "label"])
        for r in g: w.writerow([r["instance_id"], pred[r["instance_id"]]])
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", f"{d}/g.csv", "--predictions", f"{d}/p.csv",
                    "--output-dir", f"{d}/o"], cwd=OFF, check=True, capture_output=True)
    s = rj(f"{d}/o/scores.json")
    return {"wf1": s["weighted_f1"], "soft_cons": s["soft_cons"], "hard_cons": s["hard_cons"],
            "per_class_f1": {l: s["classification"]["per_label"][l]["f1"] for l in LABELS}}


def wf1(ids, pred):
    c, tp, fp, fn = Counter(), Counter(), Counter(), Counter()
    for i in ids:
        g, p = gold[i], pred[i]; c[g] += 1
        if g == p: tp[g] += 1
        else: fp[p] += 1; fn[g] += 1
    tot, out = sum(c.values()), 0.0
    for l in LABELS:
        if c[l]:
            pr = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
            rc = tp[l] / c[l]
            out += c[l] / tot * (2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return out


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


# ---------- 逐任务 ----------
oof, oof_p, stats, fits = {}, {}, [], {}
for g in GROUPS:
    for s in SEEDS:
        plain_all, sym_all, p_all = {}, {}, {}
        for k in FOLDS:
            d = f"{R}/formal/{g}/seed{s}/fold{k}"
            t, done, act = rj(f"{d}/task.json"), rj(f"{d}/done.json"), rj(f"{d}/actual_split.json")
            check(t["pipeline"] == PIPELINE and t["group"] == g and t["seed"] == s and t["fold"] == k and not t["smoke"], f"{d} 任务身份不对")
            check(done["group"] == g and done["seed"] == s and done["fold"] == k, f"{d} done 身份不对")
            check(sorted(act["test"]) == sorted(i for i in ids_all if fold_of[i] == k), f"{d} 留出和家族划分不符")
            orig = [i for i in act["fit"] if not i.startswith("gen:")]
            gen = [i for i in act["fit"] if i.startswith("gen:")]
            check(not ({family[i] for i in orig + act["val"]} & {family[i] for i in act["test"]}), f"{d} 训练和留出共享家族")
            check(not (set(act["fit"]) & set(act["val"])), f"{d} fit 和 val 重叠")
            check(len(act["fit"]) == done["fit_rows"] and len(act["val"]) == done["val_rows"], f"{d} 行数和 done 不符")
            if g == "Q0":
                check(not gen, f"{d} Q0 含生成行")
                fits[(s, k)] = act["fit"]
            else:
                check(orig == fits[(s, k)] and gen == sel[k], f"{d} Q-B1 的 fit 不是 Q0 fit + 本折选用的生成行")
                check(len(gen) == len(orig), f"{d} 生成行数 ≠ fit 行数")
            prob = {}
            for line in open(f"{d}/cache.jsonl", encoding="utf-8"):
                x = json.loads(line); prob[(x["a"], x["b"])] = x["p"]
            pl, sy, ps = predict(prob, act["test"], byid)
            check(sorted(read_pred(f"{d}/reference_fold0.csv")) == sorted(act["test"]), f"{d} 留出参考文件和划分不符")
            for mode, mine in (("plain", pl), ("sym", sy)):
                check(read_pred(f"{d}/pred_fold0_{mode}.csv") == mine, f"{d} {mode} 留出预测和概率重建不一致")
            dpl, dsy, _ = predict(prob, list(dev_byid), dev_byid)
            for mode, mine in (("plain", dpl), ("sym", dsy)):
                check(read_pred(f"{d}/pred_dev_{mode}.csv") == mine, f"{d} dev {mode} 预测不一致")
            plain_all.update(pl); sym_all.update(sy); p_all.update(ps)
            stats.append({"group": g, "seed": s, "fold": k, "steps": done["steps"], "fit_rows": done["fit_rows"],
                          "fit_tokens": done["fit_tokens"], "best_epoch": done["best_epoch"], "minutes": round(done["seconds"] / 60, 1)})
        check(sorted(sym_all) == ids_all, f"{g} seed{s} OOF 不是 3,042 行")
        for mode, mine in (("plain", plain_all), ("sym", sym_all)):
            check(read_pred(f"{R}/oof/{g}/seed{s}_{mode}.csv") == mine, f"{g} seed{s} OOF {mode} 和概率重建不一致")
        oof[(g, s)] = {"plain": plain_all, "sym": sym_all}
        oof_p[(g, s)] = p_all

# ---------- 分数 ----------
REF = f"{OFF}/final_data/train/dico_nli_train_track1_reference.csv"
scores = {}
for g in GROUPS:
    scores[g] = {}
    for s in SEEDS:
        sc = {m: official(oof[(g, s)][m], REF) for m in ("plain", "sym")}
        for m in ("plain", "sym"):
            saved = rj(f"{R}/oof/{g}/seed{s}_{m}_scores/scores.json")
            check(abs(saved["weighted_f1"] - sc[m]["wf1"]) < 1e-12 and abs(saved["soft_cons"] - sc[m]["soft_cons"]) < 1e-12
                  and abs(saved["hard_cons"] - sc[m]["hard_cons"]) < 1e-12, f"{g} seed{s} {m} 分数和保存的不同")
            check(abs(sc[m]["wf1"] - wf1(ids_all, oof[(g, s)][m])) < 1e-12, f"{g} seed{s} {m} WF1 自算不一致")
        sym = oof[(g, s)]["sym"]
        sc["O_AP_sym"] = ap([gold[i] == "NEGATIVE_OTHER" for i in ids_all], [oof_p[(g, s)][i][3] for i in ids_all])
        sc["O_missed"] = sum(gold[i] == "NEGATIVE_OTHER" and sym[i] != "NEGATIVE_OTHER" for i in ids_all)
        sc["O_false_alarm"] = sum(gold[i] != "NEGATIVE_OTHER" and sym[i] == "NEGATIVE_OTHER" for i in ids_all)
        scores[g][str(s)] = sc


# ---------- 比较（家族重抽样，两组和三个种子共用同一组抽样） ----------
def resample_draws():
    rng = random.Random(BOOT_SEED)
    return [rng.choices(fams, k=len(fams)) for _ in range(B)]


DRAWS = resample_draws()


def compare(a, b):
    """a, b: {seed: {id: label}}; per-seed difference, mean, and 95% interval of the seed-mean difference."""
    per_seed = [wf1(ids_all, a[s]) - wf1(ids_all, b[s]) for s in SEEDS]
    out = []
    for draw in DRAWS:
        ids = [i for f in draw for i in ids_of[f]]
        out.append(sum(wf1(ids, a[s]) - wf1(ids, b[s]) for s in SEEDS) / len(SEEDS))
    out.sort()
    lo, hi = out[int(B * 0.025)], out[B - 1 - int(B * 0.025)]
    mean = sum(per_seed) / len(per_seed)
    return {"per_seed": dict(zip(map(str, SEEDS), per_seed)), "mean": mean, "ci95": [lo, hi], "improvement": mean > 0 and lo > 0}


ext = {"JOINT-QWEN": {m: {s: read_pred(f"{J}/oof/qwen/seed{s}/pred_{m}.csv") for s in SEEDS} for m in ("plain", "sym")},
       "mDeBERTa-B1": {m: {s: read_pred(f"{AB}/train_plus1/oof/B1-Haiku_seed{s}_{m}.csv") for s in SEEDS} for m in ("plain", "sym")},
       "mDeBERTa-B2": {m: {s: read_pred(f"{AB}/train_c2/oof/B2-Haiku_seed{s}_{m}.csv") for s in SEEDS} for m in ("plain", "sym")}}
mine = {g: {m: {s: oof[(g, s)][m] for s in SEEDS} for m in ("plain", "sym")} for g in GROUPS}
main = compare(mine["Q-B1"]["sym"], mine["Q0"]["sym"])
desc = {"plain_QB1_minus_Q0": compare(mine["Q-B1"]["plain"], mine["Q0"]["plain"]),
        "sym_Q0_minus_JOINT-QWEN": compare(mine["Q0"]["sym"], ext["JOINT-QWEN"]["sym"]),
        "sym_QB1_minus_mDeBERTa-B1": compare(mine["Q-B1"]["sym"], ext["mDeBERTa-B1"]["sym"]),
        "sym_QB1_minus_mDeBERTa-B2": compare(mine["Q-B1"]["sym"], ext["mDeBERTa-B2"]["sym"])}
mean_scores = {g: {m: {x: sum(scores[g][str(s)][m][x] for s in SEEDS) / 3 for x in ("wf1", "soft_cons", "hard_cons")} for m in ("plain", "sym")} for g in GROUPS}
per_class = {g: {l: sum(scores[g][str(s)]["sym"]["per_class_f1"][l] for s in SEEDS) / 3 for l in LABELS} for g in GROUPS}
check(len(stats) == 30, f"任务数 {len(stats)}")
out = {"problems": problems, "tasks": len(stats), "main_QB1_minus_Q0_sym_wf1": main, "descriptive": desc,
       "scores_mean": mean_scores, "per_class_f1_sym_mean": per_class, "scores_by_seed": scores, "stats": stats,
       "rule": "Q-B1 − Q0（sym WF1）：同种子差均值 > 0 且 1,725 个家族重抽样 2,000 次（种子 20261021，两组三种子共用）的双侧 95% 区间下限 > 0；区间不含重新训练的随机性"}
json.dump(out, open(f"{R}/claude_recheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("问题：", problems[:10] if problems else "无", len(problems), "| 任务", len(stats))
for g in GROUPS:
    print(g, {m: {x: round(v * 100, 2) for x, v in mean_scores[g][m].items()} for m in ("plain", "sym")},
          "O-AP", [round(scores[g][str(s)]["O_AP_sym"] * 100, 1) for s in SEEDS],
          "漏判", [scores[g][str(s)]["O_missed"] for s in SEEDS], "误判", [scores[g][str(s)]["O_false_alarm"] for s in SEEDS])
print("主比较 Q-B1 − Q0 sym", {s: round(v * 100, 2) for s, v in main["per_seed"].items()}, round(main["mean"] * 100, 3),
      "95%", [round(x * 100, 2) for x in main["ci95"]], "通过" if main["improvement"] else "不通过")
for k, v in desc.items():
    print("  描述", k, round(v["mean"] * 100, 2), [round(x * 100, 2) for x in v["ci95"]], {s: round(x * 100, 2) for s, x in v["per_seed"].items()})
print("逐类 F1（sym）", {g: {l[:3]: round(v * 100, 1) for l, v in per_class[g].items()} for g in GROUPS})
print("best_epoch", {g: [x["best_epoch"] for x in stats if x["group"] == g] for g in GROUPS})
if problems:
    sys.exit(2)
