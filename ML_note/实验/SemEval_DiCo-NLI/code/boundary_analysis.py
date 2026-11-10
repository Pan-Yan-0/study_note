"""边界与难 O 的离线检查（只读已有的 OOF 概率，不训练）。

问题 1：错误是不是集中在"靠近决策边界"的预测上？
  - 概率间隔 margin = 对称化后最大类概率 − 第二大类概率；按 margin 分箱，看每箱的样本占比、错误率、占全部错误的比例。
  - 同一条在三个种子里错几次：三次都错（稳定错误）和一两次错（不稳定）分开，各自的平均 margin。
  - 跨模型：mDeBERTa B2、Qwen 4B、Qwen 8B 的 9 个 OOF 全错的"顽固错误"，看金标和预测的分布、置信度。
  - 校准：ECE（15 等宽箱），看高置信度区间是不是过度自信。
问题 2：生成的 O 有没有教到"难的 O"？
  - 金标 O 按字面关系分组：一侧的词是另一侧的子集（看起来像蕴含）/ 实词有重叠 / 没有共同实词；
    比较各组 O 的召回，以及金标 F / B 被判成 O 的比例（同样分组）。

问题 3：三个种子的概率平均（种子集成）和几个模型的概率平均能修掉多少错误。

全部是事后的描述性分析，没有事先登记的检验；不改任何已有结论。
用法：python3 -I code/boundary_analysis.py <官方仓库目录>
输出：metrics/diagnostics/boundary/boundary_analysis.json（不含文本），顽固错误样例打印到屏幕供人工抽查。
"""
import csv, json, os, re, sys
from collections import Counter, defaultdict

OFF = os.path.abspath(sys.argv[1])
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{SE}/metrics"
OUT = f"{M}/diagnostics/boundary"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SHORT = dict(zip(LABELS, "EFBO"))
SWAP = [0, 2, 1, 3]
SEEDS = [42, 43, 44]

rows = list(csv.DictReader(open(f"{OFF}/final_data/train/dico_nli_train_track1_participant_labeled.csv", encoding="utf-8")))
byid = {r["instance_id"]: r for r in rows}
gold = {r["instance_id"]: r["label"] for r in rows}
fam = json.load(open(f"{M}/diagnostics/families.json"))
fold_of = {i: int(fam["fold_of_family"][str(fam["family_of_pair"][byid[i]["pair_id"]])]) for i in gold}
ids = sorted(gold)


def norm(p):
    s = sum(p)
    return [x / s for x in p]


def sym_from_pq(p, q):
    return [(p[k] + q[SWAP[k]]) / 2 for k in range(4)]


def load_mdeberta(root):
    """root/seed{s}/fold{k}/predictions.csv 里的 p_* 和 r_p_*（反方向）。"""
    out = {}
    for s in SEEDS:
        d = {}
        for k in range(5):
            for r in csv.DictReader(open(f"{root}/seed{s}/fold{k}/predictions.csv", encoding="utf-8")):
                p = [float(r["p_" + c]) for c in "EFBO"]
                q = [float(r["r_p_" + c]) for c in "EFBO"]
                d[r["instance_id"]] = sym_from_pq(p, q)
        assert set(d) == set(ids), root
        out[s] = d
    return out


def load_qwen(path_of):
    """cache.jsonl：每行 a、b 两段文本和四类字母的概率；按本折留出行查两个方向。"""
    out = {}
    for s in SEEDS:
        d = {}
        for k in range(5):
            prob = {}
            for line in open(path_of(s, k), encoding="utf-8"):
                x = json.loads(line)
                prob[(x["a"], x["b"])] = norm(x["p"])
            for i in ids:
                if fold_of[i] == k:
                    r = byid[i]
                    d[i] = sym_from_pq(prob[(r["text1"], r["text2"])], prob[(r["text2"], r["text1"])])
        assert set(d) == set(ids)
        out[s] = d
    return out


AB = f"{M}/augment_b"
MODELS = {
    "mDeBERTa B0": load_mdeberta(f"{AB}/train_c1/tasks/B0"),
    "mDeBERTa B1（+1）": load_mdeberta(f"{AB}/train_plus1/tasks/B1-Haiku"),
    "mDeBERTa B2（+2）": load_mdeberta(f"{AB}/train_c2/tasks/B2-Haiku"),
    "mDeBERTa G-E1": load_mdeberta(f"{AB}/train_c2/tasks/G-E1"),
    "mDeBERTa G-O1": load_mdeberta(f"{AB}/train_c2/tasks/G-O1"),
    "Qwen 4B": load_qwen(lambda s, k: f"{M}/joint_validation/formal/qwen/fold{k}/dico_qwen_ce_results/track1/seed{s}/cache.jsonl"),
    "Qwen 8B": load_qwen(lambda s, k: f"{M}/qwen8b_a1/formal/seed{s}/fold{k}/cache.jsonl"),
}
MAIN = ["mDeBERTa B0", "mDeBERTa B2（+2）", "Qwen 4B", "Qwen 8B"]


def top2(p):
    o = sorted(range(4), key=lambda k: -p[k])
    return o[0], o[1], p[o[0]] - p[o[1]]


BINS = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001]
res = {"note": "事后的描述性分析，所有比例按三个种子合并（每条样本算 3 次）", "models": {}}

# ---------- 问题 1 ----------
for name, P in MODELS.items():
    n, err = 0, 0
    bins = [Counter() for _ in BINS[:-1]]
    low_pairs, conf_pairs = Counter(), Counter()
    ece_bins = [[0, 0.0, 0] for _ in range(15)]  # 条数、置信度和、对的条数
    wrong_count = Counter()
    margins_by_wrong = defaultdict(list)
    for s in SEEDS:
        for i in ids:
            a, b, m = top2(P[s][i])
            ok = LABELS[a] == gold[i]
            n += 1; err += not ok
            j = next(t for t in range(len(BINS) - 1) if BINS[t] <= m < BINS[t + 1])
            bins[j]["n"] += 1; bins[j]["err"] += not ok
            if not ok:
                wrong_count[i] += 1
                g = L = LABELS.index(gold[i])
                key = "".join(sorted(SHORT[LABELS[x]] for x in (a, b)))
                (low_pairs if m < 0.2 else conf_pairs if m >= 0.6 else Counter())[f"金标{SHORT[gold[i]]}→判{SHORT[LABELS[a]]}"] += 1
            c = P[s][i][a]
            e = min(int(c * 15), 14)
            ece_bins[e][0] += 1; ece_bins[e][1] += c; ece_bins[e][2] += ok
    for i in ids:
        for s in SEEDS:
            a, b, m = top2(P[s][i])
            if LABELS[a] != gold[i]:
                margins_by_wrong[wrong_count[i]].append(m)
    ece = sum(b[0] / n * abs(b[1] / b[0] - b[2] / b[0]) for b in ece_bins if b[0])
    hi = [b for b in ece_bins[13:] if b[0]]  # 置信度 ≥ 0.867
    res["models"][name] = {
        "error_rate": err / n,
        "margin_bins": [{"range": f"[{BINS[t]:.1f}, {min(BINS[t + 1], 1):.1f})", "share_of_items": bins[t]["n"] / n,
                         "error_rate": bins[t]["err"] / bins[t]["n"] if bins[t]["n"] else None,
                         "share_of_errors": bins[t]["err"] / err} for t in range(len(BINS) - 1)],
        "errors_margin_lt_0.2": sum(bins[t]["err"] for t in range(2)) / err,
        "errors_margin_ge_0.6": sum(bins[t]["err"] for t in range(4, 6)) / err,
        "low_margin_error_types": dict(low_pairs.most_common(6)),
        "confident_error_types": dict(conf_pairs.most_common(6)),
        "seed_stability": {f"wrong_in_{k}_of_3_seeds": {"items": sum(1 for i in ids if wrong_count[i] == k),
                                                         "mean_margin_when_wrong": (sum(margins_by_wrong[k]) / len(margins_by_wrong[k])) if margins_by_wrong[k] else None}
                           for k in (1, 2, 3)},
        "ece_15bins": ece,
        "confidence_ge_0.867": {"share": sum(b[0] for b in hi) / n, "mean_conf": sum(b[1] for b in hi) / sum(b[0] for b in hi),
                                "accuracy": sum(b[2] for b in hi) / sum(b[0] for b in hi)},
    }

# 跨模型的顽固错误：MAIN 里后三个（B2、Qwen 4B、8B）9 个 OOF 全错
STRONG = ["mDeBERTa B2（+2）", "Qwen 4B", "Qwen 8B"]
stub = [i for i in ids if all(LABELS[top2(MODELS[m][s][i])[0]] != gold[i] for m in STRONG for s in SEEDS)]
stub_pred = Counter()
for i in stub:
    votes = Counter(LABELS[top2(MODELS[m][s][i])[0]] for m in STRONG for s in SEEDS)
    stub_pred[f"金标{SHORT[gold[i]]}→多数判{SHORT[votes.most_common(1)[0][0]]}"] += 1
stub_margin = [top2(MODELS[m][s][i])[2] for i in stub for m in STRONG for s in SEEDS]
anyerr = [i for i in ids if any(LABELS[top2(MODELS[m][s][i])[0]] != gold[i] for m in STRONG for s in SEEDS)]
res["stubborn_errors"] = {"definition": "B2、Qwen 4B、Qwen 8B 的 9 个 OOF 全错", "items": len(stub),
                          "share_of_items_with_any_error": len(stub) / len(anyerr), "items_with_any_error": len(anyerr),
                          "types": dict(stub_pred.most_common()), "mean_margin": sum(stub_margin) / len(stub_margin),
                          "share_margin_ge_0.6": sum(m >= 0.6 for m in stub_margin) / len(stub_margin),
                          "gold_dist": dict(Counter(SHORT[gold[i]] for i in stub))}

# ---------- 问题 2：难 O ----------
STOP = set("a an the of in on at to for with by and or is are was were be been being as from into onto that this these those "
           "it its his her their there here up down out over under near s".split())


def toks(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def relation(r):
    a, b = toks(r["text1"]), toks(r["text2"])
    sa, sb = set(a), set(b)
    if sa and sb and (sa <= sb or sb <= sa):
        return "一侧是另一侧的子集"
    ca, cb = sa - STOP, sb - STOP
    if ca & cb:
        return "实词有重叠"
    return "没有共同实词"


rel = {i: relation(byid[i]) for i in ids}
GROUPS = ["一侧是另一侧的子集", "实词有重叠", "没有共同实词"]
o_ids = [i for i in ids if gold[i] == "NEGATIVE_OTHER"]
fb_ids = [i for i in ids if gold[i] in ("FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT")]
hardo = {"groups": {g: {"gold_O_items": sum(rel[i] == g for i in o_ids), "gold_FB_items": sum(rel[i] == g for i in fb_ids)} for g in GROUPS},
         "O_recall": {}, "FB_predicted_as_O": {}}
for name, P in MODELS.items():
    hardo["O_recall"][name], hardo["FB_predicted_as_O"][name] = {}, {}
    for g in GROUPS:
        og = [i for i in o_ids if rel[i] == g]
        fg = [i for i in fb_ids if rel[i] == g]
        per = [sum(LABELS[top2(P[s][i])[0]] == "NEGATIVE_OTHER" for i in og) / len(og) for s in SEEDS]
        fper = [sum(LABELS[top2(P[s][i])[0]] == "NEGATIVE_OTHER" for i in fg) / len(fg) for s in SEEDS] if fg else [None] * 3
        hardo["O_recall"][name][g] = {"mean": sum(per) / 3, "by_seed": per}
        hardo["FB_predicted_as_O"][name][g] = {"mean": (sum(fper) / 3) if fg else None, "by_seed": fper}
res["hard_O"] = hardo

# ---------- 问题 3：种子集成能修掉多少（不稳定的错误）----------
import random
family = {i: fam["family_of_pair"][byid[i]["pair_id"]] for i in ids}
fams = sorted(set(family.values()))
by_fam = defaultdict(list)
for i in ids:
    by_fam[family[i]].append(i)


def wf1(idl, pred):
    out, n = 0.0, len(idl)
    for l in LABELS:
        tp = sum(gold[i] == l and pred[i] == l for i in idl)
        fp = sum(gold[i] != l and pred[i] == l for i in idl)
        fn = sum(gold[i] == l and pred[i] != l for i in idl)
        out += (2 * tp / (2 * tp + fp + fn) if tp else 0.0) * sum(gold[i] == l for i in idl) / n
    return out


def argmax_label(p):
    return LABELS[max(range(4), key=lambda k: p[k])]


def ens(names):
    return {i: argmax_label([sum(MODELS[m][s][i][k] for m in names for s in SEEDS) for k in range(4)]) for i in ids}


def boot(fn, B=500, seed=20261021):
    rng, out = random.Random(seed), []
    for _ in range(B):
        idl = [i for f in rng.choices(fams, k=len(fams)) for i in by_fam[f]]
        out.append(fn(idl))
    out.sort()
    return [out[int(.025 * B)], out[int(.975 * B) - 1]]


ensemble = {"note": "三个种子的对称化概率直接相加再取最大类；区间按家族重抽样 500 次，只作描述（组合是看过结果后挑的）", "seed_ensemble": {}, "cross_model": {}}
for m in MAIN:
    e = ens([m]); singles = [{i: argmax_label(MODELS[m][s][i]) for i in ids} for s in SEEDS]
    d = lambda idl: wf1(idl, e) - sum(wf1(idl, x) for x in singles) / 3
    ensemble["seed_ensemble"][m] = {"single_seed_mean": sum(wf1(ids, x) for x in singles) / 3, "ensemble": wf1(ids, e), "diff": d(ids), "ci95": boot(d)}
q4 = ens(["Qwen 4B"])
for names in (["Qwen 4B", "Qwen 8B"], ["Qwen 4B", "mDeBERTa B2（+2）"], ["Qwen 4B", "Qwen 8B", "mDeBERTa B2（+2）"]):
    e = ens(names)
    d = lambda idl: wf1(idl, e) - wf1(idl, q4)
    ensemble["cross_model"][" + ".join(names)] = {"ensemble": wf1(ids, e), "minus_qwen4b_seed_ensemble": d(ids), "ci95": boot(d)}
res["ensemble"] = ensemble

os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/boundary_analysis.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- 打印 ----------
for name in MODELS:
    r = res["models"][name]
    print(f"\n== {name}  错误率 {r['error_rate']*100:.1f}%  ECE {r['ece_15bins']*100:.1f}  "
          f"置信度≥0.867 的占比 {r['confidence_ge_0.867']['share']*100:.0f}%、平均置信 {r['confidence_ge_0.867']['mean_conf']*100:.1f}、准确 {r['confidence_ge_0.867']['accuracy']*100:.1f}")
    for b in r["margin_bins"]:
        print(f"  margin {b['range']:12s} 样本 {b['share_of_items']*100:5.1f}%  错误率 {b['error_rate']*100 if b['error_rate'] is not None else float('nan'):5.1f}%  占错误 {b['share_of_errors']*100:5.1f}%")
    print(f"  错误里 margin<0.2: {r['errors_margin_lt_0.2']*100:.0f}%   margin≥0.6: {r['errors_margin_ge_0.6']*100:.0f}%")
    print("  低间隔错误类型", r["low_margin_error_types"]); print("  高置信错误类型", r["confident_error_types"])
    print("  种子稳定性", {k: (v["items"], round(v["mean_margin_when_wrong"], 2) if v["mean_margin_when_wrong"] else None) for k, v in r["seed_stability"].items()})
print("\n== 顽固错误", {k: v for k, v in res["stubborn_errors"].items()})
print("\n== 难 O 分组", hardo["groups"])
for name in MODELS:
    print(f"  {name:18s} O 召回", {g: round(v["mean"] * 100, 1) for g, v in hardo["O_recall"][name].items()},
          " F/B 判成 O", {g: (round(v["mean"] * 100, 1) if v["mean"] is not None else None) for g, v in hardo["FB_predicted_as_O"][name].items()})
print("\n== 集成", json.dumps(ensemble, ensure_ascii=False, indent=1))
print("\n== 顽固错误样例（人工抽查用，不写进文件）")
for i in stub[:40]:
    r = byid[i]
    votes = Counter(SHORT[LABELS[top2(MODELS[m][s][i])[0]]] for m in STRONG for s in SEEDS)
    pm = [round(x, 2) for x in MODELS["Qwen 4B"][42][i]]
    print(f"  金标 {SHORT[gold[i]]} 判 {dict(votes)} Qwen4B p={pm} | {r['text1']}  ||  {r['text2']}")
