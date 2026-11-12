"""O 的阈值诊断（离线、事后描述，不训练；Codex 11-12 合并审阅提议）。

问题：O 的错误是排序不好（O 的概率排不到前面），还是只是判定位置不对？
做法：
  - 规则：p_O ≥ τ 判 O，否则在 E / F / B 里取最大。τ 在每个 (模型, 种子, 折) 的内层验证上选：
    网格 0.02–0.98（步长 0.01）上内层验证四类 WF1 最高，同分取离 argmax 等价点最近的（先取 WF1 最高，再取 |τ − 0.5| 最小）。
    内层验证不属于留出集，τ 不看 OOF 金标。
  - 留出集上用该折的 τ，报 WF1、O 的精确率 / 召回 / 误报 / 漏报，和 argmax 判定比较（家族配对重抽样 2,000 次，95% 区间）。
  - O 的 AP（与阈值无关，只看排序）同时报告。
模型：
  - mDeBERTa B2（Commit 2）、NLIinit（Commit 4）：内层验证和留出都用对称化概率（两个方向的 logits 都已保存）。
  - Qwen 4B（JOINT-QWEN）：内层验证只存了原方向概率，所以内层验证和留出都用原方向（plain）。
用法：python3 -I code/o_threshold_check.py <官方仓库目录>
输出：metrics/diagnostics/boundary/o_threshold.json
"""
import csv, json, math, os, random, sys

OFF = os.path.abspath(sys.argv[1])
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{SE}/metrics"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
O = "NEGATIVE_OTHER"
SWAP = [0, 2, 1, 3]
SEEDS = [42, 43, 44]

rows = list(csv.DictReader(open(f"{OFF}/final_data/train/dico_nli_train_track1_participant_labeled.csv", encoding="utf-8")))
byid = {r["instance_id"]: r for r in rows}
gold = {r["instance_id"]: r["label"] for r in rows}
fam = json.load(open(f"{M}/diagnostics/families.json"))
family = {i: fam["family_of_pair"][byid[i]["pair_id"]] for i in gold}
fold_of = {i: int(fam["fold_of_family"][str(family[i])]) for i in gold}
ids = sorted(gold)
fams = sorted(set(family.values()))
by_fam = {}
for i in ids:
    by_fam.setdefault(family[i], []).append(i)


def soft(z):
    m = max(z); e = [math.exp(x - m) for x in z]; s = sum(e); return [x / s for x in e]


def norm(p):
    s = sum(p); return [x / s for x in p]


def symp(p, q):
    return [(p[k] + q[SWAP[k]]) / 2 for k in range(4)]


def wf1(idl, gl, pred):
    out, n = 0.0, len(idl)
    for l in LABELS:
        tp = sum(gl[i] == l and pred[i] == l for i in idl)
        fp = sum(gl[i] != l and pred[i] == l for i in idl)
        fn = sum(gl[i] == l and pred[i] != l for i in idl)
        out += (2 * tp / (2 * tp + fp + fn) if tp else 0.0) * sum(gl[i] == l for i in idl) / n
    return out


def argmax(p):
    return LABELS[max(range(4), key=lambda k: p[k])]


def thr(p, t):
    return O if p[3] >= t else LABELS[max(range(3), key=lambda k: p[k])]


def ap(idl, probs):
    s = sorted(idl, key=lambda i: -probs[i][3])
    pos = sum(gold[i] == O for i in idl)
    hit, out = 0, 0.0
    for r, i in enumerate(s, 1):
        if gold[i] == O:
            hit += 1; out += hit / r
    return out / pos


GRID = [round(0.02 + 0.01 * k, 2) for k in range(97)]


def pick_tau(val):
    """val: {id: (label, probs)}"""
    gl = {i: v[0] for i, v in val.items()}
    idl = list(val)
    best = None
    for t in GRID:
        f = wf1(idl, gl, {i: thr(v[1], t) for i, v in val.items()})
        key = (f, -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, t)
    return best[1], best[0][0], wf1(idl, gl, {i: argmax(v[1]) for i, v in val.items()})


def md(root, s, k):
    d = f"{root}/seed{s}/fold{k}"
    rev = {r["instance_id"]: [float(r["best_rev_" + c]) for c in "EFBO"] for r in csv.DictReader(open(f"{d}/extra_logits.csv", encoding="utf-8")) if r["part"] == "val"}
    val = {}
    for r in csv.DictReader(open(f"{d}/val_predictions.csv", encoding="utf-8")):
        val[r["instance_id"]] = (r["label"], symp(soft([float(r["logit_" + c]) for c in "EFBO"]), soft(rev[r["instance_id"]])))
    oof = {r["instance_id"]: [float(r["sym_p_" + c]) for c in "EFBO"] for r in csv.DictReader(open(f"{d}/predictions.csv", encoding="utf-8"))}
    return val, oof


SPLIT_VAL = {k: [r["instance_id"] for r in csv.DictReader(open(f"{M}/augment_b/train_c2/tasks/B2-Haiku/seed42/fold{k}/val_predictions.csv", encoding="utf-8"))] for k in range(5)}


def qw(s, k):
    run = f"{M}/joint_validation/formal/qwen/fold{k}/dico_qwen_ce_results/track1/seed{s}"
    best = json.load(open(f"{run}/trained.json"))["best_epoch"]
    vp = json.load(open(f"{run}/val_probs/epoch{best}.json"))
    vids = SPLIT_VAL[k]
    inputs = list(dict.fromkeys((byid[i]["text1"], byid[i]["text2"]) for i in vids))
    by = dict(zip(inputs, vp))
    val = {i: (gold[i], norm(by[(byid[i]["text1"], byid[i]["text2"])])) for i in vids}
    prob = {}
    for line in open(f"{run}/cache.jsonl", encoding="utf-8"):
        x = json.loads(line); prob[(x["a"], x["b"])] = norm(x["p"])
    oof = {i: prob[(byid[i]["text1"], byid[i]["text2"])] for i in ids if fold_of[i] == k}
    return val, oof


AB = f"{M}/augment_b"
MODELS = {"mDeBERTa B2 (sym)": lambda s, k: md(f"{AB}/train_c2/tasks/B2-Haiku", s, k),
          "mDeBERTa NLIinit (sym)": lambda s, k: md(f"{AB}/train_c4/tasks/NLIinit", s, k),
          "Qwen 4B (plain)": qw}


def o_stats(pred):
    tp = sum(gold[i] == O and pred[i] == O for i in ids)
    fp = sum(gold[i] != O and pred[i] == O for i in ids)
    fn = sum(gold[i] == O and pred[i] != O for i in ids)
    return {"pred_O": tp + fp, "false_alarm": fp, "missed": fn, "precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn)}


def boot(fn, B=2000, seed=20261112):
    rng, out = random.Random(seed), []
    for _ in range(B):
        idl = [i for f in rng.choices(fams, k=len(fams)) for i in by_fam[f]]
        out.append(fn(idl))
    out.sort()
    return [out[int(.025 * B)], out[int(.975 * B) - 1]]


res = {"note": __doc__.split("用法")[0].strip(), "models": {}}
for m, load in MODELS.items():
    per = {"seeds": {}}
    arg_all, thr_all = {}, {}
    for s in SEEDS:
        probs, pa, pt, taus = {}, {}, {}, {}
        for k in range(5):
            val, oof = load(s, k)
            t, fv, fa = pick_tau(val)
            taus[k] = {"tau": t, "val_wf1_tau": fv, "val_wf1_argmax": fa}
            for i, p in oof.items():
                probs[i] = p; pa[i] = argmax(p); pt[i] = thr(p, t)
        assert set(probs) == set(ids)
        arg_all[s], thr_all[s] = pa, pt
        per["seeds"][str(s)] = {"taus": taus, "wf1_argmax": wf1(ids, gold, pa), "wf1_tau": wf1(ids, gold, pt),
                                "O_argmax": o_stats(pa), "O_tau": o_stats(pt), "O_AP": ap(ids, probs)}
    d = lambda idl: sum(wf1(idl, gold, thr_all[s]) - wf1(idl, gold, arg_all[s]) for s in SEEDS) / 3
    per["diff_wf1_tau_minus_argmax"] = d(ids)
    per["ci95"] = boot(d)
    per["tau_all"] = [per["seeds"][str(s)]["taus"][k]["tau"] for s in SEEDS for k in range(5)]
    res["models"][m] = per
    sd = per["seeds"]
    mean = lambda f: sum(f(sd[str(s)]) for s in SEEDS) / 3
    print(m, "tau", sorted(per["tau_all"]))
    print("  WF1 argmax %.2f  tau %.2f  diff %+.2f [%+.2f, %+.2f]" % (100 * mean(lambda x: x["wf1_argmax"]), 100 * mean(lambda x: x["wf1_tau"]),
          100 * per["diff_wf1_tau_minus_argmax"], 100 * per["ci95"][0], 100 * per["ci95"][1]))
    for tag in ("O_argmax", "O_tau"):
        print("  %-8s pred_O %.0f  false_alarm %.0f  missed %.0f  P %.1f  R %.1f  F1 %.1f" % (tag, mean(lambda x: x[tag]["pred_O"]), mean(lambda x: x[tag]["false_alarm"]),
              mean(lambda x: x[tag]["missed"]), 100 * mean(lambda x: x[tag]["precision"]), 100 * mean(lambda x: x[tag]["recall"]), 100 * mean(lambda x: x[tag]["f1"])))
    print("  O AP %.1f" % (100 * mean(lambda x: x["O_AP"])))
json.dump(res, open(f"{M}/diagnostics/boundary/o_threshold.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
