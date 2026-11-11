"""顽固错误与 Other 边界的带依据盲审（Claude 起草，待和 Codex 对齐字段和抽样；判定规则在任何人标注之前写定，看结果后不改）。

  python3 -I code/boundary_review.py build <官方仓库目录> <输出目录>        生成 review_sheet.csv（只有文本）和 review_key.json（封存）
  python3 -I code/boundary_review.py score <输出目录> <标注A.csv> <标注B.csv>

抽样
  单位是短语对家族（metrics/diagnostics/families.json）。每个家族取 pair_id 最小的 __original 实例作代表，按它原本的方向展示。
  按 9 份 OOF（mDeBERTa B2、Qwen 4B、Qwen 8B 各 3 个种子，读法同 boundary_analysis.py）给代表实例分层：
    S1 顽固错误：9 份 OOF 全错。抽 16 个家族。
    S2 Other 边界混淆：不在 S1；Qwen 4B 三种子多数票判错，且错在 O 和 E/F/B 之间。
       S2-O→EFB（金标 O、判成 E/F/B）8 个，S2-EFB→O（金标 E/F/B、判成 O）8 个。
    S3 判对对照：9 份 OOF 全对。给 S1 的前 8 个、S2 两个方向各前 4 个，各配一个"金标相同、实词 Jaccard 同一档
       （0 / (0, 0.5) / ≥ 0.5）"的家族；同档没有就只要求金标相同。抽 16 个。
  共 48 个家族，展示顺序随机（种子 SEED）。
  排除 Claude 之前看过金标的家族：已有盲审和复审的答案文件里出现过的原数据 id（单词替换盲审、E 类对照、E 复审、P1 等），
  boundary_analysis.py 打印过的顽固错误样例（stub 前 40 条），字面近邻检查里"最近邻标签 = 金标"的全部顽固错误
  （当时打印了其中 12 条，打印顺序不可复现，所以整类排除）。更早的会话里零散看过的样例无法逐条追溯，这一点在报告里写明。

标注（每行一个家族，按 text1 → text2 的方向）
  label        E / F / B / O / U（U = 拿不准）
  o_subtype    只在 label = O 时填：SIM（同类、相似但不蕴含）/ REL（相关）/ OPP（对立）/ UNR（无关）
  phenomena    可多选，用 ; 分隔：SYN（近义、同指、不同称呼）HYP（上下位）MOD（一侧多或少修饰语）NUM（数量）
               PART（部分与整体）TENSE（时态、体、词形）PREP（介词、方位）WK（需要世界知识）OTHER
  evidence     一句话依据
  审阅人对照参考卡判断（E 复审和规模复审用的同一张，metrics/augment_b/e_recheck/参考卡.md；卡上 40 个家族已排除），
  拿不准时和卡上最像的例子比；不查训练集，不看模型预测和本脚本的答案文件。

事先定的读法（score 输出）
  每个家族：两人都给确定标签（非 U）时，归为 一致且等于金标 / 一致但不等于金标 / 两人不一致；有 U 的单列。
  R1 顽固错误的来源。记 a1、a3 为 S1、S3 可判定家族中"一致但不等于金标"的比例，c1 为 S1 中"一致且等于金标"的比例。
     a1 ≥ 50% 且 a1 − a3 ≥ 30 个百分点：顽固错误多是对照参考卡的人也复现不了的标法（个别化的口径或噪声），
        超出了判对样本里也有的分歧；定向增广和结构改动不以这些样本为目标。
     c1 ≥ 50%：对照参考卡的人能判出金标、9 份 OOF 全错，属于可学但没学到，作为后续改动的目标。
     都不满足：只报告。S3 的 a3 本身也报告（判对样本上审阅人和金标的基线分歧）。
  R2 定向增广候选。某个现象标签在 S1 ∪ S2 里被两人同时标出的家族 ≥ 6 个，且其中"一致且等于金标"≥ 70%：
     列为候选（仍要单独登记方案，按 v6 的质检和停止条件生成，和同规模同标签配额的对照比较）。
  R3（描述）金标为 O 的家族里两人给的 o_subtype 分布，S2 和 S3 分开，供 Other 潜在分量头解读。
  另报：每层 U 的比例、两人标签一致率、每个审阅人和 Qwen 4B 多数票一致的比例。
  48 个家族只够看大的差别，所有比例附计数；这是诊断，不改任何已登记的结论。
"""
import contextlib, csv, glob, hashlib, io, json, os, random, re, runpy, sys
from collections import Counter, defaultdict

SEED = 20261112
SHORT = {"E": "EQUIVALENCE", "F": "FORWARD_ENTAILMENT", "B": "BACKWARD_ENTAILMENT", "O": "NEGATIVE_OTHER", "U": "UNSURE"}
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRONG = ["mDeBERTa B2（+2）", "Qwen 4B", "Qwen 8B"]
OSUB = {"SIM", "REL", "OPP", "UNR"}
PHEN = {"SYN", "HYP", "MOD", "NUM", "PART", "TENSE", "PREP", "WK", "OTHER"}
O = "NEGATIVE_OTHER"
STOP = set("a an the of in on at to for with by and or is are was were be been being as from into onto that this these those "
           "it its his her their there here up down out over under near s".split())


def load_analysis(off):
    sys.argv = ["boundary_analysis.py", off]
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(f"{SE}/code/boundary_analysis.py")


def knn_label_matches(g, subset):
    """字面近邻检查（11-11）的同一算法：候选来自其他四折，两侧 token Jaccard 平均，允许对调（对调时 F/B 互换）。"""
    ids, byid, gold, fold_of = g["ids"], g["byid"], g["gold"], g["fold_of"]
    toks = lambda t: set(re.findall(r"[a-z0-9]+", t.lower()))
    T = {i: (toks(byid[i]["text1"]), toks(byid[i]["text2"])) for i in ids}
    J = lambda a, b: len(a & b) / len(a | b) if a | b else 0
    sw = {"FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT", "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT"}
    out = set()
    for i in subset:
        a1, a2 = T[i]; best = (-1, None, False)
        for j in ids:
            if fold_of[j] == fold_of[i]: continue
            b1, b2 = T[j]; s = (J(a1, b1) + J(a2, b2)) / 2; r = (J(a1, b2) + J(a2, b1)) / 2
            if s > best[0]: best = (s, j, False)
            if r > best[0]: best = (r, j, True)
        lab = gold[best[1]]; lab = sw.get(lab, lab) if best[2] else lab
        if lab == gold[i]: out.add(i)
    return out


def seen_by_claude(g):
    pairs = set()
    files = glob.glob(f"{SE}/metrics/diagnostics/blind_review/*.json") + glob.glob(f"{SE}/metrics/augment_b/**/*key*.json", recursive=True)
    for p in files:
        pairs |= set(re.findall(r"dico_phrasis_\d{7}", open(p, encoding="utf-8").read()))
    printed = list(g["stub"][:40]) + sorted(knn_label_matches(g, g["stub"]))
    pairs |= {g["byid"][i]["pair_id"] for i in printed}
    return pairs, len(files), len(printed)


def overlap_bin(r):
    a = set(re.findall(r"[a-z0-9]+", r["text1"].lower())) - STOP
    b = set(re.findall(r"[a-z0-9]+", r["text2"].lower())) - STOP
    j = len(a & b) / len(a | b) if a | b else 0
    return "0" if j == 0 else ("lt0.5" if j < 0.5 else "ge0.5")


def build(off, out):
    g = load_analysis(off)
    MODELS, gold, LABELS, SEEDS, byid, top2, fam = g["MODELS"], g["gold"], g["LABELS"], g["SEEDS"], g["byid"], g["top2"], g["fam"]
    family = {i: fam["family_of_pair"][byid[i]["pair_id"]] for i in gold}
    rep = {}
    for i in sorted(gold):
        if i.endswith("__original") and (family[i] not in rep or byid[i]["pair_id"] < byid[rep[family[i]]]["pair_id"]):
            rep[family[i]] = i
    seen_pairs, n_files, n_printed = seen_by_claude(g)
    seen_fam = {family[i] for i in gold if byid[i]["pair_id"] in seen_pairs}
    pred = lambda m, s, i: LABELS[top2(MODELS[m][s][i])[0]]
    nine = lambda i: [pred(m, s, i) for m in STRONG for s in SEEDS]
    qmaj = lambda i: Counter(pred("Qwen 4B", s, i) for s in SEEDS).most_common(1)[0][0]
    S1, S2o, S2e, S3 = [], [], [], []
    for f, i in sorted(rep.items(), key=lambda kv: kv[1]):
        if f in seen_fam: continue
        v = nine(i)
        if all(x != gold[i] for x in v): S1.append(i)
        elif all(x == gold[i] for x in v): S3.append(i)
        elif qmaj(i) != gold[i] and gold[i] == O: S2o.append(i)
        elif qmaj(i) != gold[i] and qmaj(i) == O: S2e.append(i)
    rng = random.Random(SEED)
    s1 = rng.sample(S1, 16); s2o = rng.sample(S2o, 8); s2e = rng.sample(S2e, 8)
    pool = list(S3); rng.shuffle(pool); s3 = []
    for t in s1[:8] + s2o[:4] + s2e[:4]:
        cand = [i for i in pool if gold[i] == gold[t] and overlap_bin(byid[i]) == overlap_bin(byid[t])] or [i for i in pool if gold[i] == gold[t]]
        s3.append(cand[0]); pool.remove(cand[0])
    items = [(i, "S1") for i in s1] + [(i, "S2-O→EFB") for i in s2o] + [(i, "S2-EFB→O") for i in s2e] + [(i, "S3") for i in s3]
    rng.shuffle(items)
    os.makedirs(out, exist_ok=True)
    key = {"version": "boundary-review-v1", "seed": SEED, "rules": "见 code/boundary_review.py 开头",
           "pool_sizes": {"S1": len(S1), "S2-O→EFB": len(S2o), "S2-EFB→O": len(S2e), "S3": len(S3)},
           "families_total": len(rep), "excluded_families_seen_by_claude": len(seen_fam),
           "exclusion_sources": {"answer_files": n_files, "printed_instances": n_printed}, "items": {}}
    with open(f"{out}/review_sheet.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["review_id", "text1", "text2", "label", "o_subtype", "phenomena", "evidence"])
        for n, (i, st) in enumerate(items, 1):
            rid = f"R{n:03d}"; r = byid[i]
            w.writerow([rid, r["text1"], r["text2"], "", "", "", ""])
            key["items"][rid] = {"instance_id": i, "family": family[i], "stratum": st, "gold": gold[i], "overlap_bin": overlap_bin(r),
                                 "nine_oof": dict(Counter(nine(i))), "qwen4b_majority": qmaj(i)}
    with open(f"{out}/review_key.json", "w", encoding="utf-8") as fh:
        json.dump(key, fh, ensure_ascii=False, indent=1)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    print("家族", len(items), dict(Counter(st for _, st in items)), "| 候选池", key["pool_sizes"], "| 全部家族", len(rep), "排除", len(seen_fam))
    print("review_sheet.csv SHA-256", sha(f"{out}/review_sheet.csv"))
    print("review_key.json  SHA-256", sha(f"{out}/review_key.json"))


def read_ann(p, ids):
    out = {}
    with open(p, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rid = r["review_id"].strip(); lab = r["label"].strip().upper(); sub = r["o_subtype"].strip().upper()
            ph = {x.strip().upper() for x in r["phenomena"].split(";") if x.strip()}
            if lab not in SHORT: raise ValueError(f"{p} {rid}：label 非法 {lab!r}")
            if lab == "O" and sub not in OSUB: raise ValueError(f"{p} {rid}：label = O 但 o_subtype 非法 {sub!r}")
            if lab != "O" and sub: raise ValueError(f"{p} {rid}：label ≠ O 却填了 o_subtype")
            if not ph <= PHEN: raise ValueError(f"{p} {rid}：phenomena 非法 {sorted(ph - PHEN)}")
            if not r["evidence"].strip(): raise ValueError(f"{p} {rid}：缺 evidence")
            if rid in out: raise ValueError(f"{p} {rid}：重复")
            out[rid] = {"label": SHORT[lab], "sub": sub, "ph": ph}
    if set(out) != set(ids): raise ValueError(f"{p}：行不全或多出 {sorted(set(ids) ^ set(out))[:5]}")
    return out


def score(out, pa, pb):
    key = json.load(open(f"{out}/review_key.json", encoding="utf-8"))["items"]
    A, B = read_ann(pa, key), read_ann(pb, key)
    cat, by = {}, defaultdict(Counter)
    for rid, k in key.items():
        a, b = A[rid]["label"], B[rid]["label"]
        c = "U" if "UNSURE" in (a, b) else "两人不一致" if a != b else "一致且等于金标" if a == k["gold"] else "一致但不等于金标"
        cat[rid] = c
        for s in {k["stratum"], k["stratum"].split("-")[0]}: by[s][c] += 1
    share = lambda s, c: (by[s][c], sum(v for x, v in by[s].items() if x != "U"))
    a1, a3, c1 = share("S1", "一致但不等于金标"), share("S3", "一致但不等于金标"), share("S1", "一致且等于金标")
    # 阈值用整数交叉相乘判断，避免 0.7 − 0.4 = 0.2999… 这类浮点误差（Codex 11-12 指出，阈值和分支顺序不变）
    (A1, N1), (A3, N3), (C1, _) = a1, a3, c1
    if N1 and N3 and 2 * A1 >= N1 and 10 * (A1 * N3 - A3 * N1) >= 3 * N1 * N3: verdict = "顽固错误多是对照参考卡也复现不了的标法"
    elif N1 and 2 * C1 >= N1: verdict = "可学但没学到，作为后续改动的目标"
    else: verdict = "只报告"
    res = {"by_stratum": {s: dict(v) for s, v in sorted(by.items())},
           "R1": {"a1": a1, "a3": a3, "c1": c1, "读法": verdict}}
    tag = defaultdict(list)
    for rid, k in key.items():
        if k["stratum"].startswith(("S1", "S2")):
            for t in A[rid]["ph"] & B[rid]["ph"]: tag[t].append(rid)
    res["R2"] = {t: {"families": len(v), "一致且等于金标": sum(cat[x] == "一致且等于金标" for x in v),
                     "候选": len(v) >= 6 and 10 * sum(cat[x] == "一致且等于金标" for x in v) >= 7 * len(v)} for t, v in sorted(tag.items())}
    sub = defaultdict(Counter)
    for rid, k in key.items():
        if k["gold"] == O:
            for x in (A[rid]["sub"], B[rid]["sub"]):
                if x: sub[k["stratum"].split("-")[0]][x] += 1
    res["R3_o_subtype"] = {s: dict(v) for s, v in sorted(sub.items())}
    res["label_agreement"] = (sum(A[x]["label"] == B[x]["label"] for x in key), len(key))
    res["U_by_stratum"] = {s: (sum(cat[x] == "U" for x, k in key.items() if k["stratum"].startswith(s)), sum(k["stratum"].startswith(s) for k in key.values())) for s in ("S1", "S2", "S3")}
    res["agree_with_qwen4b_majority"] = {w: (sum(X[x]["label"] == k["qwen4b_majority"] for x, k in key.items()), len(key)) for w, X in (("A", A), ("B", B))}
    json.dump(res, open(f"{out}/review_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build": build(os.path.abspath(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "score": score(sys.argv[2], sys.argv[3], sys.argv[4])
    else: raise SystemExit(__doc__)
