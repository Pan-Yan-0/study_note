"""英语五折 OOF 的错误分类：只读已有预测，不训练、不改结果。

用法:
    python3 error_taxonomy.py <官方仓库目录> <joint_validation/oof 目录> <输出 json>

口径：每条记录（3042 行，两个方向都算）取各种子的多数票（平票取字母序最小），
看 sym 预测。错误按"金标 -> 预测"归成四大类，再按编辑形状、短语长度切开。
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

E, F, B, O = "EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"
FUNC = set("a an the of in on at to for with by from and or is are was were be 's".split())


def tok(s):
    return re.findall(r"\w+", s.lower())


def shape(a, b):
    A, Bc = Counter(tok(a)), Counter(tok(b))
    da, db = sum((A - Bc).values()), sum((Bc - A).values())
    if not (A & Bc):
        return "无共同词"
    if da == 0 and db == 0:
        return "词相同"
    if da == 0 or db == 0:
        return "插入"
    if da == 1 and db == 1:
        diff = set((A - Bc) + (Bc - A))
        return "单词替换(含虚词)" if diff & FUNC else "单词替换(实词)"
    return "其他"


def kind(gold, pred):
    if gold == pred:
        return "对"
    if gold == O:
        return "漏判O(金标O→判成E/F/B)"
    if pred == O:
        return "错判O(金标E/F/B→判成O)"
    if E in (gold, pred):
        return "粒度(E↔单向)"
    return "方向(F↔B)"


def load_majority(dirs):
    votes = defaultdict(list)
    for d in dirs:
        with open(d / "pred_sym.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                votes[r["instance_id"]].append(r["label"])
    return {k: sorted(Counter(v).items(), key=lambda x: (-x[1], x[0]))[0][0] for k, v in votes.items()}, \
        {k: Counter(v) for k, v in votes.items()}


def main(off, oof, out):
    rows = list(csv.DictReader(open(Path(off) / "final_data/train/dico_nli_train_track1_participant_labeled.csv", encoding="utf-8")))
    oof = Path(oof)
    models = {
        "CE": sorted((oof / "mdeberta/ce").glob("seed*")),
        "R-Drop": sorted((oof / "mdeberta/rdrop").glob("seed*")),
        "Qwen": sorted((oof / "qwen").glob("seed*")),
        "KD": sorted((oof / "kd").glob("seed*")),
        "CE3": [oof / "mdeberta/ce" / f"seed{s}" for s in (42, 43, 44)],
    }
    maj = {m: load_majority(d)[0] for m, d in models.items()}
    res = {"rows": len(rows), "gold": Counter(r["label"] for r in rows)}

    for m in ("CE", "R-Drop", "Qwen", "KD"):
        p = maj[m]
        kinds = Counter(kind(r["label"], p[r["instance_id"]]) for r in rows)
        err = sum(v for k, v in kinds.items() if k != "对")
        res[m] = {"errors": err, "by_kind": {k: [v, round(v / err, 3)] for k, v in kinds.most_common() if k != "对"}}

    # 编辑形状 × 错误类型（CE 与 Qwen）
    for m in ("CE", "Qwen"):
        t = defaultdict(Counter)
        for r in rows:
            t[shape(r["text1"], r["text2"])][kind(r["label"], maj[m][r["instance_id"]])] += 1
        res[f"{m}_by_shape"] = {s: {"rows": sum(c.values()), "acc": round(c["对"] / sum(c.values()), 3),
                                    "errors": {k: v for k, v in c.items() if k != "对"}} for s, c in t.items()}

    # 共同错误：CE、R-Drop、Qwen 多数票都错
    common = [r for r in rows if all(maj[m][r["instance_id"]] != r["label"] for m in ("CE", "R-Drop", "Qwen"))]
    res["common_errors"] = {
        "rows": len(common),
        "by_kind_CE": Counter(kind(r["label"], maj["CE"][r["instance_id"]]) for r in common),
        "by_gold": Counter(r["label"] for r in common),
        "by_shape": Counter(shape(r["text1"], r["text2"]) for r in common),
        "same_wrong_label_all3": sum(len({maj[m][r["instance_id"]] for m in ("CE", "R-Drop", "Qwen")}) == 1 for r in common),
    }
    # 只看金标 O：错误里有多少是"两侧共享词多"的（主题相关却不蕴含）
    def overlap(r):
        a, b = set(tok(r["text1"])) - FUNC, set(tok(r["text2"])) - FUNC
        return len(a & b) / max(1, len(a | b))
    o_rows = [r for r in rows if r["label"] == O]
    bins = defaultdict(lambda: [0, 0])
    for r in o_rows:
        k = "无实词重合" if overlap(r) == 0 else ("重合<0.5" if overlap(r) < .5 else "重合≥0.5")
        bins[k][0] += 1
        bins[k][1] += maj["CE"][r["instance_id"]] == O
    res["O_by_overlap_CE"] = {k: {"rows": n, "recall": round(c / n, 3)} for k, (n, c) in bins.items()}
    bins = defaultdict(lambda: [0, 0])
    for r in o_rows:
        k = "无实词重合" if overlap(r) == 0 else ("重合<0.5" if overlap(r) < .5 else "重合≥0.5")
        bins[k][0] += 1
        bins[k][1] += maj["Qwen"][r["instance_id"]] == O
    res["O_by_overlap_Qwen"] = {k: {"rows": n, "recall": round(c / n, 3)} for k, (n, c) in bins.items()}
    # 金标 E/F/B 被判成 O 的比例，按实词重合度分（检验试点论文"词汇重合多时偏向 Similar"的说法）
    for m in ("CE", "Qwen"):
        bins = defaultdict(lambda: [0, 0])
        for r in rows:
            if r["label"] == O:
                continue
            k = "无实词重合" if overlap(r) == 0 else ("重合<0.5" if overlap(r) < .5 else "重合≥0.5")
            bins[k][0] += 1
            bins[k][1] += maj[m][r["instance_id"]] == O
        res[f"EFB_predicted_O_by_overlap_{m}"] = {k: {"rows": n, "rate": round(c / n, 3)} for k, (n, c) in bins.items()}

    # KD 相对同三种子 CE：修正与新增按错误类型
    fix, new = Counter(), Counter()
    for r in rows:
        g, c, k = r["label"], maj["CE3"][r["instance_id"]], maj["KD"][r["instance_id"]]
        if c != g and k == g:
            fix[kind(g, c)] += 1
        if c == g and k != g:
            new[kind(g, k)] += 1
    res["KD_vs_CE3_majority"] = {"fixed": fix, "introduced": new}

    # 同一归一化无序文本、标签不一致（只看可对调方向前的原始行，判断标注噪声的下限）
    lab = defaultdict(set)
    for r in rows:
        lab[(r["text1"].lower().strip(), r["text2"].lower().strip())].add(r["label"])
    res["ordered_inputs_with_conflicting_labels"] = sum(len(v) > 1 for v in lab.values())

    # 抽样：共同错误按类型各取前 8 条（按 id 排序，描述性）
    ex = defaultdict(list)
    for r in sorted(common, key=lambda r: r["instance_id"]):
        k = kind(r["label"], maj["CE"][r["instance_id"]])
        if len(ex[k]) < 8 and r["instance_id"].endswith("original") or (len(ex[k]) < 8 and r["label"] == O):
            ex[k].append([r["text1"], r["text2"], r["label"][:3], maj["CE"][r["instance_id"]][:3], maj["Qwen"][r["instance_id"]][:3]])
    res["common_examples"] = ex
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
