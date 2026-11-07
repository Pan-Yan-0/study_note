"""类别构成补充实验（线 2）的配额、重复对照抽样和生成池缺口计算。只用 Python 标准库。

用法：
    python3 mix_quota.py <官方仓库目录> <输出 JSON>

划分和第五组 / 联合验证逐行相同（FOLD_SEED 2027，内层验证 10%，种子 FOLD_SEED*100+fold+1），
函数照抄自 code/joint_validation/dico_mdeberta_methods.py，并在 selftest 里核对 fold 0 的已知计数。
"""
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter

FOLDS, FOLD_SEED, INNER_FRACTION = 5, 2027, 0.1
QUOTA_VERSION = "mixquota-v1"
ORDER_SEED = 20261028          # 每折每类固定顺序的种子（生成池和原数据都用它，按名字派生）
CLASSES = ("E", "D", "O")      # D = 单向短语对（一行 F、一行 B）
LABEL_CLASS = {"EQUIVALENCE": "E", "FORWARD_ENTAILMENT": "D", "BACKWARD_ENTAILMENT": "D", "NEGATIVE_OTHER": "O"}
ROWS_PER_PAIR = {"E": 2, "D": 2, "O": 1}


# ---------- 划分（照抄，勿改） ----------

def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm(s):
    return " ".join(s.lower().split())


def build_families(rows):
    parent, first = {}, {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        p = r["pair_id"]
        parent.setdefault(p, p)
        k = tuple(sorted((norm(r["text1"]), norm(r["text2"]))))
        if k in first:
            a, b = find(p), find(first[k])
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            first[k] = p
    return {p: find(p) for p in parent}


def assign_folds(rows, fam_of):
    size = Counter(fam_of[r["pair_id"]] for r in rows)
    fams = sorted(size)
    random.Random(FOLD_SEED).shuffle(fams)
    load, fold_of = [0] * FOLDS, {}
    for f in fams:
        k = min(range(FOLDS), key=lambda i: (load[i], i))
        fold_of[f] = k
        load[k] += size[f]
    return fold_of


def pick_families(families, fraction, seed):
    fams = sorted(set(families))
    random.Random(seed).shuffle(fams)
    return set(fams[: max(1, int(len(fams) * fraction))])


def load_split(repo):
    path = os.path.join(repo, "final_data", "train", "dico_nli_train_track1_reference.csv")
    rows = read_rows(path)
    fam_of = build_families(rows)
    fold_of = assign_folds(rows, fam_of)
    inner = {}
    for k in range(FOLDS):
        rest = {fam_of[r["pair_id"]] for r in rows if fold_of[fam_of[r["pair_id"]]] != k}
        inner[k] = pick_families(rest, INNER_FRACTION, FOLD_SEED * 100 + k + 1)
    part = {}
    for k in range(FOLDS):
        for r in rows:
            f = fam_of[r["pair_id"]]
            part[(k, r["instance_id"])] = "test" if fold_of[f] == k else "val" if f in inner[k] else "fit"
    return rows, fam_of, fold_of, inner, part, digest(split_manifest(rows, part))


def digest(obj):
    """和 GPT 的 Qwen8B 队列（common.py）相同的 JSON 哈希。"""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def split_manifest(rows, part):
    """和 metrics/qwen8b_acceptance/split_manifest.json 同口径（不含 dev），可以逐项比对。"""
    keys = ("instance_id", "pair_id", "text1", "text2", "label")
    out = {}
    for k in range(FOLDS):
        out[str(k)] = {}
        for s in ("fit", "val", "test"):
            rs = [{c: r[c] for c in keys} for r in rows if part[(k, r["instance_id"])] == s]
            out[str(k)][s] = {"rows": len(rs), "sha256": digest(rs)}
    return out


def signature(fam, fold_of, inner):
    """家族在哪些折里属于拟合部分（不在留出，也不在内层验证）。"""
    return tuple(k for k in range(FOLDS) if fold_of[fam] != k and fam not in inner[k])


# ---------- 原数据按短语对分组 ----------

def original_pairs(rows, fam_of, part, k):
    """第 k 折拟合部分的完整短语对：{类: [(pair_key, [instance_id, ...]), ...]}，按 pair_key 排序。
    两个方向共用 pair_id（reverse_pair_id 是另一方向的 instance_id）；E / D 一对两行，O 一对一行。"""
    groups = {}
    for r in rows:
        groups.setdefault(r["pair_id"], []).append(r)
    out = {c: [] for c in CLASSES}
    for pid, members in groups.items():
        where = {part[(k, m["instance_id"])] for m in members}
        assert len(where) == 1, pid  # 同一对的两个方向必须在同一部分
        if where != {"fit"}:
            continue
        cls = {LABEL_CLASS[m["label"]] for m in members}
        assert len(cls) == 1, pid
        c = cls.pop()
        assert len(members) == ROWS_PER_PAIR[c], (pid, len(members))
        if c == "D":
            assert sorted(m["label"] for m in members) == ["BACKWARD_ENTAILMENT", "FORWARD_ENTAILMENT"], pid
        if c != "O":
            ids = {m["instance_id"] for m in members}
            assert all(m["reverse_pair_id"] in ids for m in members), pid
        out[c].append((pid, sorted(m["instance_id"] for m in members)))
    for c in CLASSES:
        out[c].sort()
    return out


def fit_counts(pairs):
    return {c: sum(len(m) for _, m in pairs[c]) for c in CLASSES}


# ---------- 设计点和整数化 ----------

def design_targets(fit_rows, n_add):
    """返回 {点: (E, D, O) 新增行的实数目标}。R 用本折拟合部分自己的比例。"""
    n_fit = sum(fit_rows.values())
    r = tuple(fit_rows[c] / n_fit for c in CLASSES)
    pts = {"R": r, "U4": (0.25, 0.50, 0.25)}
    for i, c in enumerate(CLASSES):
        for j, t in ((1, 1 / 3), (2, 2 / 3), (3, 1.0)):
            pts[f"{c}{j}"] = tuple((1 - t) * r[m] + t * (1.0 if m == i else 0.0) for m in range(3))
    tgt = {p: tuple(s * n_add for s in v) for p, v in pts.items()}
    total = n_fit + n_add  # BAL：最终训练集四个标签各 1/4，即 E 1/4、D 1/2、O 1/4
    tgt["BAL"] = (total / 4 - fit_rows["E"], total / 2 - fit_rows["D"], total / 4 - fit_rows["O"])
    return tgt


def integerize(target, n_add):
    """E、D 取偶数（整对），O = n_add − E − D ≥ 0，总数恰为 n_add；最小化各类与目标的平方误差。
    并列时取 E 较小、再取 D 较小的解（确定性）。n_add 必须是偶数，否则 E3 / D3 无解。"""
    assert n_add % 2 == 0
    tE, tD, tO = target
    best = None
    for e in range(0, n_add + 1, 2):
        cont = (tD + (n_add - e) - tO) / 2  # 固定 e 后 D 的连续最优
        cands = {2 * math.floor(cont / 2), 2 * math.ceil(cont / 2)}
        for d in sorted(cands):
            d = min(max(d, 0), n_add - e)
            d -= d % 2
            o = n_add - e - d
            err = (e - tE) ** 2 + (d - tD) ** 2 + (o - tO) ** 2
            key = (round(err, 9), e, d)
            if best is None or key < best[0]:
                best = (key, (e, d, o))
    return best[1]


def fold_quotas(fit_rows):
    n_fit = sum(fit_rows.values())
    n_add = n_fit - n_fit % 2     # 所有组共用的新增行数（偶数，保证顶点可行）
    out = {}
    for p, t in design_targets(fit_rows, n_add).items():
        e, d, o = integerize(t, n_add)
        assert e + d + o == n_add and e % 2 == 0 and d % 2 == 0 and min(e, d, o) >= 0
        final = {"E": fit_rows["E"] + e, "D": fit_rows["D"] + d, "O": fit_rows["O"] + o}
        T = sum(final.values())
        out[p] = {
            "target_rows": [round(x, 3) for x in t],
            "rows": {"E": e, "D": d, "O": o},
            "pairs": {"E": e // 2, "D": d // 2, "O": o},
            "rows_by_label": {"E": e, "F": d // 2, "B": d // 2, "O": o},
            "added_share": {c: round(v / n_add, 4) for c, v in zip(CLASSES, (e, d, o))},
            "pi_full": {"E": round(final["E"] / T, 4), "F": round(final["D"] / 2 / T, 4),
                        "B": round(final["D"] / 2 / T, 4), "O": round(final["O"] / T, 4)},
        }
    return n_add, out


# ---------- 重复对照的抽样列表 ----------

def fixed_order(items, name):
    seed = int(hashlib.sha256(f"{ORDER_SEED}|{name}".encode()).hexdigest()[:16], 16)
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


def dup_list(pairs, quota, k):
    """按每类固定顺序循环取 quota 对，返回 [(instance_id, 重复次数)]（每对的行都重复同样次数）。"""
    reps = Counter()
    for c in CLASSES:
        order = fixed_order([key for key, _ in pairs[c]], f"dup|fold{k}|{c}")
        members = dict(pairs[c])
        need = quota["pairs"][c]
        for i in range(need):
            for iid in members[order[i % len(order)]]:
                reps[iid] += 1
    return sorted(reps.items())


# ---------- 生成池需求和缺口 ----------

def pool_need(quotas_by_fold, main_line_k=(1, 2, 4), fit_by_fold=None):
    """每折每类需要的生成短语对数：线 2 各点的最大值，与主线 +k（按本折比例）取大。"""
    need = {}
    for k, (n_add, q) in quotas_by_fold.items():
        m = {c: max(v["pairs"][c] for v in q.values()) for c in CLASSES}
        if fit_by_fold is not None:
            for mult in main_line_k:
                _, qq = fold_quotas_for_scale(fit_by_fold[k], mult)
                for c in CLASSES:
                    m[c] = max(m[c], qq["pairs"][c])
        need[k] = m
    return need


def fold_quotas_for_scale(fit_rows, mult):
    n_fit = sum(fit_rows.values())
    n_add = mult * (n_fit - n_fit % 2)
    t = design_targets(fit_rows, n_add)["R"]
    e, d, o = integerize(t, n_add)
    return n_add, {"pairs": {"E": e // 2, "D": d // 2, "O": o}}


def pool_gap(need, pool_rows):
    """pool_rows: 生成池清单（每行至少含 pair_uid、label、usable_folds）。返回每折每类的缺口（对）。"""
    have = {k: Counter() for k in need}
    seen = set()
    for r in pool_rows:
        if r["pair_uid"] in seen:
            continue
        seen.add(r["pair_uid"])
        c = LABEL_CLASS[r["label"]]
        for k in r["usable_folds"]:
            have[k][c] += 1
    return {k: {c: max(0, need[k][c] - have[k][c]) for c in CLASSES} for k in need}


def plan_supplement_uniform(gap_O, sig_parents):
    """第一步：补生成在全部父家族上均匀分配（和主线相同的父样本分布，不集中到少数签名）。
    每个家族多产出 λ 对可用 O，第 k 折得到 λ × (第 k 折拟合部分的家族数)；取刚好盖住最大缺口的 λ。
    返回 (λ, 预计补生成的 O 对数, 每折预计新增)。实际生成后用 pool_gap 复核，残差再用 plan_supplement 补。"""
    fit_fams = {k: sum(n for s, n in sig_parents.items() if k in s) for k in gap_O}
    lam = max(gap_O[k] / fit_fams[k] for k in gap_O if gap_O[k] > 0) if any(v > 0 for v in gap_O.values()) else 0.0
    total = lam * sum(n for s, n in sig_parents.items() if s)
    return lam, total, {k: lam * fit_fams[k] for k in gap_O}


def plan_supplement(gap_O, sig_parents, o_per_batch):
    """第二步（只补残差）：每次选能填上最多缺口的签名生成一批（混合配额批次，只取其中的 O）。
    gap_O: {fold: 缺多少对 O}；sig_parents: {签名: 父家族数}；o_per_batch: 试点实测的每批过滤后 O 对数。
    返回 {签名: 批数}。贪心，不保证最少批数，但每一步都确定、可复现。"""
    gap = dict(gap_O)
    plan = Counter()
    sigs = sorted(s for s in sig_parents if s)
    while any(v > 0 for v in gap.values()):
        s = max(sigs, key=lambda s: (sum(min(gap[k], o_per_batch) for k in s if gap[k] > 0), len(s), s))
        if sum(min(gap[k], o_per_batch) for k in s if gap[k] > 0) == 0:
            raise RuntimeError("剩余缺口无法由任何签名覆盖")
        plan[s] += 1
        for k in s:
            gap[k] = max(0, gap[k] - o_per_batch)
    return dict(plan)


# ---------- 主程序 ----------

def main(repo, out_path):
    rows, fam_of, fold_of, inner, part, split_hash = load_split(repo)
    fams = sorted(set(fam_of.values()))
    sig_parents = Counter(signature(f, fold_of, inner) for f in fams)
    result = {"quota_version": QUOTA_VERSION, "order_seed": ORDER_SEED, "split_hash": split_hash,
              "n_signatures": len(sig_parents),
              "signatures": {",".join(map(str, s)): n for s, n in sorted(sig_parents.items())}, "folds": {}}
    quotas_by_fold, fit_by_fold = {}, {}
    for k in range(FOLDS):
        pairs = original_pairs(rows, fam_of, part, k)
        fr = fit_counts(pairs)
        n_add, q = fold_quotas(fr)
        quotas_by_fold[k], fit_by_fold[k] = (n_add, q), fr
        dup = {p: dup_list(pairs, q[p], k) for p in q}
        for p in q:
            assert sum(n for _, n in dup[p]) == n_add, (k, p)
        n_fit = sum(fr.values())
        result["folds"][k] = {
            "fit_rows": fr, "fit_rows_by_label": {"E": fr["E"], "F": fr["D"] // 2, "B": fr["D"] // 2, "O": fr["O"]},
            "pi_fit": {c: round(fr[c] / n_fit, 4) for c in CLASSES},
            "n_add": n_add, "points": q,
            "dup_unique_rows": {p: len(dup[p]) for p in q},
            "dup_max_repeat": {p: max((n for _, n in dup[p]), default=0) for p in q},
            "parts": dict(Counter(part[(k, r["instance_id"])] for r in rows)),
        }
    need = pool_need(quotas_by_fold, fit_by_fold=fit_by_fold)
    result["pool_need_pairs"] = need
    # 生成前的预估：主线按父家族均匀生成，池子正好够主线 +4；线 2 还要补多少 O
    main4 = {k: fold_quotas_for_scale(fit_by_fold[k], 4)[1]["pairs"] for k in range(FOLDS)}
    result["main_plus4_pairs"] = main4
    result["o_gap_after_main_plus4"] = {k: need[k]["O"] - main4[k]["O"] for k in range(FOLDS)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    return result


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
