"""先校准再集成（离线、事后描述，不训练）。

问题：mDeBERTa B2 严重过度自信，Qwen 校准较好；直接把概率平均时，B2 的高置信度会压过 Qwen。
先在每个 (模型, 种子, 折) 的内层验证上拟合一个温度 T，再平均概率，集成会不会更好？

做法：
  - 温度：在内层验证的原方向四类 log 概率上，最小化 NLL 求 T（对数空间黄金分割，T ∈ [0.05, 20]）。
    内层验证不属于留出集，T 不看 OOF 金标。
  - 应用：OOF 两个方向的四类 log 概率都除以 T 后 softmax，再按 F↔B 对调对称化。
  - 集成：同一组模型、三个种子的对称化概率等权相加取最大类（与 boundary_analysis.py 的集成相同，只多了温度）。
  - 比较：校准后集成 − 直接集成，按家族配对重抽样 2,000 次的 95% 区间。组合是看过结果后挑的，只作描述。
  - 同时报每个模型校准前后的 ECE（15 等宽箱，按预测标签判对）和 NLL，作实现检查。
模型：mDeBERTa B2（Commit 2）、Qwen3-4B JOINT-QWEN、Qwen3-8B A1，与边界分析相同。
用法：python3 -I code/calibrated_ensemble.py <官方仓库目录>
输出：metrics/diagnostics/boundary/calibrated_ensemble.json
"""
import csv, json, math, os, random, sys

OFF = os.path.abspath(sys.argv[1])
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{SE}/metrics"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {l: k for k, l in enumerate(LABELS)}
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


def logs(p):
    s = sum(p)
    return [math.log(max(x / s, 1e-300)) for x in p]


def soft(z, T=1.0):
    z = [x / T for x in z]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def nll(zs, ys, T):
    out = 0.0
    for z, y in zip(zs, ys):
        zz = [x / T for x in z]
        m = max(zz)
        out += m + math.log(sum(math.exp(x - m) for x in zz)) - zz[y]
    return out / len(ys)


def fit_T(zs, ys):
    lo, hi = math.log(0.05), math.log(20.0)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(zs, ys, math.exp(a)), nll(zs, ys, math.exp(b))
    for _ in range(80):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo); fa = nll(zs, ys, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo); fb = nll(zs, ys, math.exp(b))
    return math.exp((lo + hi) / 2)


def wf1(idl, pred):
    out, n = 0.0, len(idl)
    for l in LABELS:
        tp = sum(gold[i] == l and pred[i] == l for i in idl)
        fp = sum(gold[i] != l and pred[i] == l for i in idl)
        fn = sum(gold[i] == l and pred[i] != l for i in idl)
        out += (2 * tp / (2 * tp + fp + fn) if tp else 0.0) * sum(gold[i] == l for i in idl) / n
    return out


def argmax(p):
    return LABELS[max(range(4), key=lambda k: p[k])]


def sym(zp, zq, T):
    p, q = soft(zp, T), soft(zq, T)
    return [(p[k] + q[SWAP[k]]) / 2 for k in range(4)]


def ece(probs, bins=15):
    acc = [[0, 0, 0.0] for _ in range(bins)]
    for i, p in probs.items():
        c = max(p); k = min(int(c * bins), bins - 1)
        acc[k][0] += 1; acc[k][1] += argmax(p) == gold[i]; acc[k][2] += c
    n = len(probs)
    return sum(abs(b[1] / b[0] - b[2] / b[0]) * b[0] / n for b in acc if b[0])


def mean_nll(probs):
    return sum(-math.log(max(p[L2I[gold[i]]], 1e-300)) for i, p in probs.items()) / len(probs)


# ---------- 读取：每个 (种子, 折) 给出内层验证 (z, y) 和 OOF {id: (z_ab, z_ba)} ----------
def md_unit(root, s, k):
    d = f"{root}/seed{s}/fold{k}"
    val = list(csv.DictReader(open(f"{d}/val_predictions.csv", encoding="utf-8")))
    vz = [[float(r["logit_" + c]) for c in "EFBO"] for r in val]
    vy = [L2I[r["label"]] for r in val]
    oof = {}
    for r in csv.DictReader(open(f"{d}/predictions.csv", encoding="utf-8")):
        oof[r["instance_id"]] = (logs([float(r["p_" + c]) for c in "EFBO"]), logs([float(r["r_p_" + c]) for c in "EFBO"]))
    return vz, vy, oof, [r["instance_id"] for r in val]


SPLIT_VAL = {k: [r["instance_id"] for r in csv.DictReader(open(f"{M}/augment_b/train_c2/tasks/B2-Haiku/seed42/fold{k}/val_predictions.csv", encoding="utf-8"))] for k in range(5)}


def qw_unit(run, s, k):
    tr = json.load(open(f"{run}/trained.json"))
    hist = json.load(open(f"{run}/history.json"))
    best = tr["best_epoch"]
    vp = json.load(open(f"{run}/val_probs/epoch{best}.json"))
    vids = SPLIT_VAL[k]
    inputs = list(dict.fromkeys((byid[i]["text1"], byid[i]["text2"]) for i in vids))
    assert len(inputs) == len(vp), (run, len(inputs), len(vp))
    by = dict(zip(inputs, vp))
    vz = [logs(by[(byid[i]["text1"], byid[i]["text2"])]) for i in vids]
    vy = [L2I[gold[i]] for i in vids]
    rec = [h["val_weighted_f1"] for h in hist if h["epoch"] == best][0]
    got = wf1(vids, {i: argmax(soft(z)) for i, z in zip(vids, vz)})
    assert abs(rec - got) < 1e-9, (run, rec, got)  # 顺序重建正确的检查
    prob = {}
    for line in open(f"{run}/cache.jsonl", encoding="utf-8"):
        x = json.loads(line)
        prob[(x["a"], x["b"])] = x["p"]
    oof = {}
    for i in ids:
        if fold_of[i] == k:
            r = byid[i]
            oof[i] = (logs(prob[(r["text1"], r["text2"])]), logs(prob[(r["text2"], r["text1"])]))
    return vz, vy, oof, vids


AB = f"{M}/augment_b"
MODELS = {
    "mDeBERTa B2": lambda s, k: md_unit(f"{AB}/train_c2/tasks/B2-Haiku", s, k),
    "Qwen 4B": lambda s, k: qw_unit(f"{M}/joint_validation/formal/qwen/fold{k}/dico_qwen_ce_results/track1/seed{s}", s, k),
    "Qwen 8B": lambda s, k: qw_unit(f"{M}/qwen8b_a1/formal/seed{s}/fold{k}", s, k),
}

raw, cal, temps = {}, {}, {}
for m, load in MODELS.items():
    raw[m], cal[m], temps[m] = {}, {}, {}
    for s in SEEDS:
        rp, cp = {}, {}
        for k in range(5):
            vz, vy, oof, vids = load(s, k)
            assert vids == SPLIT_VAL[k] and not set(vids) & set(oof)
            T = fit_T(vz, vy)
            temps[m][f"seed{s}/fold{k}"] = {"T": T, "val_nll_T1": nll(vz, vy, 1.0), "val_nll_T": nll(vz, vy, T)}
            for i, (zp, zq) in oof.items():
                rp[i], cp[i] = sym(zp, zq, 1.0), sym(zp, zq, T)
        assert set(rp) == set(ids)
        raw[m][s], cal[m][s] = rp, cp


def ens(src, names):
    return {i: argmax([sum(src[m][s][i][k] for m in names for s in SEEDS) for k in range(4)]) for i in ids}


def boot(fn, B=2000, seed=20261112):
    rng, out = random.Random(seed), []
    for _ in range(B):
        idl = [i for f in rng.choices(fams, k=len(fams)) for i in by_fam[f]]
        out.append(fn(idl))
    out.sort()
    return [out[int(.025 * B)], out[int(.975 * B) - 1]]


res = {"note": __doc__.split("用法")[0].strip(), "per_model": {}, "ensemble": {}}
for m in MODELS:
    Ts = [v["T"] for v in temps[m].values()]
    pm = {"T_min": min(Ts), "T_median": sorted(Ts)[7], "T_max": max(Ts), "units": temps[m]}
    for tag, src in (("raw", raw), ("calibrated", cal)):
        pm[tag] = {"single_seed_wf1_mean": sum(wf1(ids, {i: argmax(src[m][s][i]) for i in ids}) for s in SEEDS) / 3,
                   "seed_ensemble_wf1": wf1(ids, ens(src, [m])),
                   "ece_seed_mean": sum(ece(src[m][s]) for s in SEEDS) / 3,
                   "nll_seed_mean": sum(mean_nll(src[m][s]) for s in SEEDS) / 3}
    res["per_model"][m] = pm
for names in (["Qwen 4B", "mDeBERTa B2"], ["Qwen 4B", "Qwen 8B"], ["Qwen 4B", "Qwen 8B", "mDeBERTa B2"]):
    er, ec = ens(raw, names), ens(cal, names)
    d = lambda idl: wf1(idl, ec) - wf1(idl, er)
    res["ensemble"][" + ".join(names)] = {"raw_wf1": wf1(ids, er), "calibrated_wf1": wf1(ids, ec), "diff": d(ids), "ci95": boot(d),
                                          "rows_changed": sum(er[i] != ec[i] for i in ids),
                                          "changed_now_right": sum(er[i] != ec[i] and ec[i] == gold[i] for i in ids),
                                          "changed_was_right": sum(er[i] != ec[i] and er[i] == gold[i] for i in ids)}
out = f"{M}/diagnostics/boundary/calibrated_ensemble.json"
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for m, pm in res["per_model"].items():
    print(m, "T median %.3f [%.3f, %.3f]" % (pm["T_median"], pm["T_min"], pm["T_max"]),
          {t: {k: round(v * (100 if "wf1" in k or "ece" in k else 1), 2) for k, v in pm[t].items()} for t in ("raw", "calibrated")})
for n, e in res["ensemble"].items():
    print(n, {k: (round(v * 100, 2) if isinstance(v, float) else [round(x * 100, 2) for x in v] if isinstance(v, list) else v) for k, v in e.items()})
