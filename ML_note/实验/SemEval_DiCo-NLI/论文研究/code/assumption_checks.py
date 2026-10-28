"""检查论文假设在英语训练集上是否成立：只读已有数据和预测，CPU 运行。

用法:
    python3 assumption_checks.py <官方仓库目录> <joint_validation 目录> <WordNet 3.0 dict 目录> <输出 json>

三项检查：
1. 自然逻辑（MacCartney & Manning 2009）：单词替换（实词）对里，WordNet 关系能否预测金标；
2. 数据地图（Swayamdipta et al. 2020）的替代口径：用 CE、R-Drop 各 5 个种子的 OOF 对错，
   看"多数运行都错"的样本占多少、标签怎么分布（这里没有逐轮训练动态，只是跨种子的粗口径）；
3. 加权软标签（Zhou et al. 2021）的前提：教师在学生训练行上是否几乎总是对且很自信。
"""
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

E, F, B, O = "EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"
FUNC = set("a an the of in on at to for with by from and or is are was were be 's".split())
POS = {"n": "noun", "v": "verb", "a": "adj"}


def tok(s):
    return re.findall(r"\w+", s.lower())


# ---------- 极简 WordNet 3.0 读取（不依赖 nltk） ----------
class WN:
    def __init__(self, d):
        d = Path(d)
        self.index, self.data, self.exc = {}, {}, {}
        for p, name in POS.items():
            for line in open(d / f"index.{name}", encoding="utf-8"):
                if line.startswith(" "):
                    continue
                f = line.split()
                n_ptr = int(f[3])
                n_syn = int(f[2])
                offs = f[4 + n_ptr + 2:4 + n_ptr + 2 + n_syn]
                self.index[(f[0], p)] = [(p, o) for o in offs]
            for line in open(d / f"data.{name}", encoding="utf-8"):
                if line.startswith(" "):
                    continue
                f = line.split()
                w = int(f[3], 16)
                i = 4 + 2 * w
                pc = int(f[i])
                ptrs = []
                for k in range(pc):
                    sym, off, pp = f[i + 1 + 4 * k:i + 4 + 4 * k]
                    pp = "a" if pp in "as" else pp
                    ptrs.append((sym, (pp, off)))
                self.data[(p, f[0])] = ptrs
            for line in open(d / f"{name}.exc", encoding="utf-8") if (d / f"{name}.exc").exists() else []:
                a, *b = line.split()
                self.exc.setdefault((a, p), b[0])

    def synsets(self, w):
        out = []
        for p in POS:
            cands = [w, self.exc.get((w, p), w)]
            if p == "n":
                cands += [w[:-1]] if w.endswith("s") else []
                cands += [w[:-2]] if w.endswith("es") else []
            if p == "v":
                cands += [w[:-1], w[:-2], w[:-3]] if w.endswith(("s", "ed", "ing")) else []
            for c in cands:
                if (c, p) in self.index:
                    out += self.index[(c, p)]
                    break
        return out

    def hypernyms(self, s, depth=12):
        seen, frontier = set(), {s}
        for _ in range(depth):
            nxt = set()
            for x in frontier:
                for sym, t in self.data.get(x, []):
                    if sym in ("@", "@i") and t not in seen:
                        nxt.add(t)
            seen |= nxt
            frontier = nxt
        return seen

    def relation(self, w1, w2):
        s1, s2 = self.synsets(w1), self.synsets(w2)
        if not s1 or not s2:
            return "WordNet没有"
        if set(s1) & set(s2):
            return "同义"
        for a in s1:
            for sym, t in self.data.get(a, []):
                if sym == "!" and t in s2:
                    return "反义"
        h1 = set().union(*(self.hypernyms(a) for a in s1))
        h2 = set().union(*(self.hypernyms(b) for b in s2))
        if set(s2) & h1:
            return "左是右的下位"
        if set(s1) & h2:
            return "左是右的上位"
        direct1 = {t for a in s1 for sym, t in self.data.get(a, []) if sym in ("@", "@i")}
        direct2 = {t for b in s2 for sym, t in self.data.get(b, []) if sym in ("@", "@i")}
        if direct1 & direct2:
            return "同位(共享直接上位)"
        return "无直接关系"


EXPECT = {"同义": E, "左是右的下位": F, "左是右的上位": B, "反义": O, "同位(共享直接上位)": O}


def read(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def main(off, jv, wndir, out):
    rows = read(Path(off) / "final_data/train/dico_nli_train_track1_participant_labeled.csv")
    jv = Path(jv)
    res = {}

    # 各模型各种子的 OOF sym 预测
    runs = {}
    for m in ("ce", "rdrop"):
        for d in sorted((jv / "oof/mdeberta" / m).glob("seed*")):
            runs[f"{m}/{d.name}"] = {r["instance_id"]: r["label"] for r in read(d / "pred_sym.csv")}
    for d in sorted((jv / "oof/qwen").glob("seed*")):
        runs[f"qwen/{d.name}"] = {r["instance_id"]: r["label"] for r in read(d / "pred_sym.csv")}

    # 1. 自然逻辑的单词替换检查
    wn = WN(wndir)
    table = defaultdict(Counter)
    acc = defaultdict(lambda: [0, 0, 0, 0])
    examples = defaultdict(list)
    for r in rows:
        if not r["instance_id"].endswith("original"):
            continue  # 每个源短语对只看一个方向；单条的 O 也是 original
        A, Bc = Counter(tok(r["text1"])), Counter(tok(r["text2"]))
        da, db = A - Bc, Bc - A
        if not (A & Bc) or sum(da.values()) != 1 or sum(db.values()) != 1:
            continue
        w1, w2 = next(iter(da)), next(iter(db))
        if w1 in FUNC or w2 in FUNC or w1.isdigit() or w2.isdigit():
            continue
        rel = wn.relation(w1, w2)
        table[rel][r["label"]] += 1
        ce = [runs[k][r["instance_id"]] == r["label"] for k in runs if k.startswith("ce/")]
        qw = [runs[k][r["instance_id"]] == r["label"] for k in runs if k.startswith("qwen/")]
        a = acc[rel]
        a[0] += len(ce); a[1] += sum(ce); a[2] += len(qw); a[3] += sum(qw)
        if len(examples[rel]) < 6:
            examples[rel].append(f"{r['text1']} ‖ {r['text2']} → {r['label'][:3]}")
    tot = sum(sum(c.values()) for c in table.values())
    agree = sum(table[k][v] for k, v in EXPECT.items())
    covered = sum(sum(table[k].values()) for k in EXPECT)
    res["natural_logic_sub1"] = {
        "pairs": tot,
        "relation_x_gold": {k: dict(v) for k, v in table.items()},
        "with_definite_relation": covered,
        "gold_matches_natural_logic_mapping": agree,
        "acc_ce_qwen_by_relation": {k: [round(v[1] / v[0], 3), round(v[3] / v[2], 3)] for k, v in acc.items()},
        "examples": examples,
    }

    # 2. 跨种子的"数据地图"替代口径（CE+R-Drop 共 10 次运行）
    ks = [k for k in runs if not k.startswith("qwen/")]
    buckets = Counter()
    by_gold = defaultdict(Counter)
    hard = []
    for r in rows:
        c = sum(runs[k][r["instance_id"]] == r["label"] for k in ks)
        b = "0/10 全错" if c == 0 else ("10/10 全对" if c == len(ks) else ("1-4/10 多数错" if c < 5 else "5-9/10 多数对"))
        buckets[b] += 1
        by_gold[b][r["label"]] += 1
        if c == 0:
            hard.append(r)
    res["cross_seed_map"] = {"runs": len(ks), "buckets": dict(buckets), "by_gold": {k: dict(v) for k, v in by_gold.items()}}
    # 全错样本里：10 次运行给的是不是同一个错误标签
    same = sum(len({runs[k][r["instance_id"]] for k in ks}) == 1 for r in hard)
    qwen_also = sum(all(runs[k][r["instance_id"]] != r["label"] for k in runs if k.startswith("qwen/")) for r in hard)
    res["cross_seed_map"]["all_wrong_same_label"] = same
    res["cross_seed_map"]["all_wrong_and_qwen_3of3_wrong"] = qwen_also

    # 3. 教师在学生训练行（fit）上的表现：teacher.jsonl 的 p 顺序按 E/F/B/O（由准确率核对）
    gold = {(r["text1"], r["text2"]): r["label"] for r in rows}
    order = [E, F, B, O]
    stats = []
    for t in sorted((jv / "formal/teachers").glob("fold*/seed*/teacher.jsonl")):
        n = c = conf = 0
        pg, ent2 = [], []
        for line in open(t, encoding="utf-8"):
            x = json.loads(line)
            if x.get("part") != "fit" or (x["a"], x["b"]) not in gold:
                continue
            g = gold[(x["a"], x["b"])]
            p = x["p"]
            n += 1
            c += order[max(range(4), key=lambda i: p[i])] == g
            pg.append(p[order.index(g)])
            q = [v ** 0.5 for v in p]  # T=2 的软化
            s = sum(q)
            q = [v / s for v in q]
            ent2.append(-sum(v * math.log(v) for v in q if v > 0))
        pg.sort()
        stats.append({"file": str(t.relative_to(jv)), "fit_rows": n, "acc": round(c / n, 4),
                      "gold_prob_median": round(pg[len(pg) // 2], 4),
                      "share_gold_prob_gt_0.9": round(sum(v > .9 for v in pg) / n, 4),
                      "T2_entropy_mean_nats": round(sum(ent2) / n, 3)})
    res["teacher_fit"] = stats
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
