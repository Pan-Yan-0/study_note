"""A0：Q0 教师在 B2 生成行上的改动统计和盲审抽样（冻结计划 11-12「研究对齐 v4」A0 条目，抽样规则在看统计之前写定）。

  python3 -I code/a0_analysis.py stats <A0 结果目录> <输出目录>    教师目标、改动统计 → teacher_targets.jsonl、a0_stats.json
  python3 -I code/a0_analysis.py sheet <输出目录>                  抽 40 + 20 行盲审，写 audit_sheet.csv、audit_key.json（封存）
  python3 -I code/a0_analysis.py score <输出目录> <复审标注.csv>    计票，写 audit_result.json

教师目标：生成行 (a, b) 在其所在折的 sym 概率 (p(a, b) + swapFB(p(b, a))) / 2，三个种子平均（先各自归一）。
改动统计：硬标签改动比例 = 教师 argmax ≠ 生成标签的有向生成行 / B2 实际送训的有向生成行（逐折、逐生成标签，给分子分母）；
  另报教师给生成标签的平均概率、目标熵、生成标签 → 教师标签的迁移矩阵。
盲审抽样（种子 SEED）：
  先把有向行合并成无序短语对单位：同一无序对的正反向和跨折重复是一个单位，展示方向取字典序较小的 (text1, text2)，
  两个候选标签都映射到展示方向；一个单位的教师标签取它所有出现处教师 sym 概率的平均（映射到展示方向后）。
  分歧单位（教师 argmax ≠ 生成标签）按"生成标签 → 教师标签"格子分层，共 40 个：每个非空格子先给 1 个，余下按格子大小比例
  分配（最大余数法，同余数按格子名排序），格子不够就全取；一致单位抽 20 个：按教师给共同标签的概率分成 < 0.9 和 ≥ 0.9 两层，各 10 个。
  审阅人只看编号和两个短语，对照参考卡（metrics/augment_b/e_recheck/参考卡.md）填 E / F / B / O / U 和理由。
计票（按 Codex 复审后的标签）：分歧行记"教师较对"（审阅 = 教师）/"生成标签较对"（审阅 = 生成）/"两者都错"/"无法确定"（U）；
  一致行记"一致且审阅同意"/"一致但审阅不同意"/U。入队规则在冻结计划里（RL：硬改动 ≥ 5% 且 教师较对 > 生成较对；SL：教师较对 ≥ 生成较对）。
"""
import csv, json, math, os, random, sys
from collections import Counter, defaultdict

SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED = f"{SE}/metrics/augment_b/haiku_scale/merged"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SHORT = dict(zip(LABELS, "EFBO"))
SWAP = [0, 2, 1, 3]
SEED, N_DIS, N_AGR = 20261113, 40, 20
SEEDS = (42, 43, 44)


def rj(p): return json.load(open(p, encoding="utf-8"))
def wj(p, x):
    with open(p, "w", encoding="utf-8") as f: json.dump(x, f, ensure_ascii=False, indent=1)
def sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
def norm(p): s = sum(p); return [x / s for x in p]
def swap(p): return [p[SWAP[k]] for k in range(4)]
def argmax(p): return max(range(4), key=lambda k: p[k])
def ent(p): return -sum(x * math.log(x) for x in p if x > 0)


def b2_rows():
    man = {r["sample_id"]: r for r in csv.DictReader(open(f"{MERGED}/manifest.csv", encoding="utf-8"))}
    sel = list(csv.DictReader(open(f"{MERGED}/fold_selection_scale.csv", encoding="utf-8")))
    return [(int(s["fold"]), man[s["sample_id"]]) for s in sel if int(s["tier"]) in (1, 2)]


def stats(res, out):
    os.makedirs(out, exist_ok=True)
    rows = b2_rows()
    prob = {}
    for f in range(5):
        for s in SEEDS:
            for l in open(f"{res}/teacher/fold{f}/seed{s}.jsonl", encoding="utf-8"):
                x = json.loads(l); prob[(f, s, x["a"], x["b"])] = norm(x["p"])
    targets, cnt = [], defaultdict(Counter)
    trans = Counter(); pg, en = defaultdict(list), defaultdict(list)
    for f, r in rows:
        a, b, g = r["text1"], r["text2"], LABELS.index(r["label"])
        ps = [[(u + v) / 2 for u, v in zip(prob[(f, s, a, b)], swap(prob[(f, s, b, a)]))] for s in SEEDS]
        t = [sum(p[k] for p in ps) / 3 for k in range(4)]
        k = argmax(t)
        targets.append({"fold": f, "sample_id": r["sample_id"], "pair_uid": r["pair_uid"], "text1": a, "text2": b,
                        "gen": LABELS[g], "teacher": LABELS[k], "p": t, "p_seeds": ps})
        for key in (f"fold{f}", "all"):
            cnt[key]["rows"] += 1; cnt[key]["changed"] += k != g
            cnt[f"{key}|{SHORT[LABELS[g]]}"]["rows"] += 1; cnt[f"{key}|{SHORT[LABELS[g]]}"]["changed"] += k != g
        trans[f"{SHORT[LABELS[g]]}->{SHORT[LABELS[k]]}"] += 1
        pg[SHORT[LABELS[g]]].append(t[g]); en[SHORT[LABELS[g]]].append(ent(t))
    with open(f"{out}/teacher_targets.jsonl", "w", encoding="utf-8") as fo:
        for t in targets: fo.write(json.dumps(t, ensure_ascii=False) + "\n")
    st = {"version": "a0-analysis-v1", "source_summary_sha256": sha(f"{res}/A0_SUMMARY.json"),
          "hard_change": {k: {"changed": v["changed"], "rows": v["rows"], "rate": v["changed"] / v["rows"]} for k, v in sorted(cnt.items())},
          "transition": dict(sorted(trans.items())),
          "mean_p_gen": {k: sum(v) / len(v) for k, v in sorted(pg.items())},
          "mean_entropy": {k: sum(v) / len(v) for k, v in sorted(en.items())},
          "seed_agreement": sum(len({argmax(p) for p in t["p_seeds"]}) == 1 for t in targets) / len(targets),
          "teacher_targets_sha256": sha(f"{out}/teacher_targets.jsonl")}
    wj(f"{out}/a0_stats.json", st)
    hc = st["hard_change"]["all"]
    print(f"硬改动 {hc['changed']} / {hc['rows']} = {hc['rate']:.4f}；三种子 argmax 一致 {st['seed_agreement']:.4f}")
    for k in "EFBO": print(k, hc and st["hard_change"][f"all|{k}"], f"p_gen {st['mean_p_gen'][k]:.3f} H {st['mean_entropy'][k]:.3f}")
    print(st["transition"])


def units(out):
    """有向行 → 无序短语对单位，标签和概率都映射到展示方向。"""
    u = {}
    for l in open(f"{out}/teacher_targets.jsonl", encoding="utf-8"):
        t = json.loads(l)
        a, b = t["text1"], t["text2"]
        flip = (b, a) < (a, b)
        d1, d2 = (b, a) if flip else (a, b)
        g = LABELS.index(t["gen"]); p = t["p"]
        if flip: g, p = SWAP[g], swap(p)
        x = u.setdefault((d1, d2), {"text1": d1, "text2": d2, "gens": set(), "ps": [], "sources": []})
        x["gens"].add(LABELS[g]); x["ps"].append(p); x["sources"].append({"fold": t["fold"], "sample_id": t["sample_id"]})
    res = []
    for k, x in sorted(u.items()):
        p = [sum(q[i] for q in x["ps"]) / len(x["ps"]) for i in range(4)]
        res.append({"text1": x["text1"], "text2": x["text2"], "gens": sorted(x["gens"]), "p": p, "teacher": LABELS[argmax(p)], "sources": x["sources"]})
    return res


def sheet(out):
    us = units(out)
    ok = [x for x in us if len(x["gens"]) == 1]
    conflict = len(us) - len(ok)  # 同一无序对在不同地方的生成标签不一致（映射后），不抽
    dis = defaultdict(list); agr = {"low": [], "high": []}
    for x in ok:
        g = x["gens"][0]
        if x["teacher"] != g: dis[f"{SHORT[g]}->{SHORT[x['teacher']]}"].append(x)
        else: agr["low" if x["p"][LABELS.index(g)] < 0.9 else "high"].append(x)
    rng = random.Random(SEED)
    cells = sorted(dis)
    alloc = {c: min(1, len(dis[c])) for c in cells}
    left = N_DIS - sum(alloc.values())
    tot = sum(len(dis[c]) for c in cells)
    quota = {c: left * len(dis[c]) / tot for c in cells}
    base = {c: int(quota[c]) for c in cells}
    for c in cells: alloc[c] += base[c]
    rem = N_DIS - sum(alloc.values())
    for c in sorted(cells, key=lambda c: (-(quota[c] - base[c]), c))[:rem]: alloc[c] += 1
    alloc = {c: min(alloc[c], len(dis[c])) for c in cells}
    rows = []
    for c in cells:
        for x in rng.sample(dis[c], alloc[c]): rows.append(dict(x, kind="DIS", cell=c))
    for h in ("low", "high"):
        for x in rng.sample(agr[h], min(N_AGR // 2, len(agr[h]))): rows.append(dict(x, kind="AGR", cell=h))
    rng.shuffle(rows)
    with open(f"{out}/audit_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["review_id", "text1", "text2", "label", "evidence"])
        for n, r in enumerate(rows, 1): w.writerow([f"T{n:03d}", r["text1"], r["text2"], "", ""])
    key = {"version": "a0-audit-v1", "seed": SEED, "units": len(us), "label_conflict_units": conflict,
           "cell_sizes": {c: len(v) for c, v in dis.items()}, "alloc": alloc, "agree_sizes": {h: len(v) for h, v in agr.items()},
           "teacher_targets_sha256": sha(f"{out}/teacher_targets.jsonl"),
           "items": {f"T{n:03d}": {"kind": r["kind"], "cell": r["cell"], "gen": r["gens"][0], "teacher": r["teacher"], "p": r["p"],
                                   "sources": r["sources"]} for n, r in enumerate(rows, 1)}}
    wj(f"{out}/audit_key.json", key)
    print(f"单位 {len(us)}（生成标签冲突 {conflict}）；分歧格子 {key['cell_sizes']}；分配 {alloc}；一致 {key['agree_sizes']}")
    print("audit_sheet.csv", sha(f"{out}/audit_sheet.csv")[:12], "audit_key.json", sha(f"{out}/audit_key.json")[:12])


def score(out, labels):
    key = rj(f"{out}/audit_key.json")["items"]
    lab = {r["review_id"]: r["label"].strip().upper() for r in csv.DictReader(open(labels, encoding="utf-8"))}
    assert set(lab) == set(key) and all(v in "EFBOU" for v in lab.values()), "标注表不完整或有非法标签"
    c = Counter()
    for rid, x in key.items():
        l = lab[rid]; g, t = SHORT[x["gen"]], SHORT[x["teacher"]]
        if x["kind"] == "DIS":
            c["DIS|" + ("U" if l == "U" else "teacher" if l == t else "gen" if l == g else "both_wrong")] += 1
        else:
            c[f"AGR|{x['cell']}|" + ("U" if l == "U" else "agree" if l == g else "disagree")] += 1
    st = rj(f"{out}/a0_stats.json")["hard_change"]["all"]
    tb, gb = c["DIS|teacher"], c["DIS|gen"]
    res = {"labels_sha256": sha(labels), "counts": dict(sorted(c.items())), "hard_change_rate": st["rate"],
           "RL": {"rule": "硬改动 ≥ 5% 且 教师较对 > 生成较对", "pass": st["rate"] >= 0.05 and tb > gb},
           "SL": {"rule": "教师较对 ≥ 生成较对（两者都是 0 时另议）", "pass": (tb >= gb) if (tb or gb) else None}}
    wj(f"{out}/audit_result.json", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 3: raise SystemExit(__doc__)
    if a[1] == "stats": stats(os.path.abspath(a[2]), os.path.abspath(a[3]))
    elif a[1] == "sheet": sheet(os.path.abspath(a[2]))
    elif a[1] == "score": score(os.path.abspath(a[2]), os.path.abspath(a[3]))
    else: raise SystemExit(__doc__)
