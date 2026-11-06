"""实验 B：E 类审核标准对照（11-06，Pan-Yan-0 定；在看到对照结果之前写定）。不生成，只抽原训练集。

  python e_control.py build <官方仓库目录> <输出目录>          生成 control_sheet.csv（隐藏金标）和 control_key.json
  python e_control.py score <输出目录> <标注.csv>              计分并按事先的规则给出 v7 的处理

- 抽样：英语训练集，每个短语对家族只取一个代表（pair_id 字典序最小的 original 方向），跳过只差数量的对。
  E 20 对；非 E 20 对作干扰项（单向 12、O 8），单向随机展示方向。种子 SEED。
- 标注流程和 Haiku 的 P1C 相同（GPT 先标、Pan-Yan-0 审核），这样两边可以比。
- 只看 E：m_orig = 原数据 E 里可判定项（非 U）中，标注和金标不同的比例。
- 事先的规则：
  m_orig < 30%：审核标准基本接受数据集的 E，生成 E 的问题算生成质量。v7 对 E 禁止任何一侧多出信息，
               包括全名对姓、多出头衔或身份、多出修饰语。
  m_orig ≥ 30%：门槛本身会拒掉原数据的 E。v7 只收紧修饰语和具体程度，人名变体保留；跑 v7 之前先把门槛问题交回 Pan-Yan-0。
"""
import csv, json, os, random, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from label_conventions import number_pair  # noqa: E402

SEED = 20261108
N = {"EQUIVALENCE": 20, "FORWARD_ENTAILMENT": 12, "NEGATIVE_OTHER": 8}
SHORT = {"E": "EQUIVALENCE", "EQ": "EQUIVALENCE", "F": "FORWARD_ENTAILMENT", "FE": "FORWARD_ENTAILMENT",
         "B": "BACKWARD_ENTAILMENT", "BE": "BACKWARD_ENTAILMENT", "O": "NEGATIVE_OTHER", "U": "UNSURE"}
THRESHOLD = 0.30


def build(off, out):
    with open(f"{off}/final_data/train/dico_nli_train_track1_participant_labeled.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_text = {}
    for r in rows:
        if not r["instance_id"].endswith("__original"): continue
        key = frozenset((r["text1"].strip().lower(), r["text2"].strip().lower()))
        by_text.setdefault(key, []).append(r)
    reps = [sorted(v, key=lambda r: r["pair_id"])[0] for v in by_text.values()]
    pool = {}
    for r in reps:
        t1, t2, lab = r["text1"], r["text2"], r["label"]
        if lab == "BACKWARD_ENTAILMENT": t1, t2, lab = t2, t1, "FORWARD_ENTAILMENT"
        if number_pair(t1, t2): continue
        pool.setdefault(lab, []).append((r["pair_id"], t1, t2))
    rng = random.Random(SEED)
    items = []
    for lab, n in N.items():
        for pid, t1, t2 in rng.sample(sorted(pool[lab]), n):
            flip = rng.random() < 0.5
            a, b = (t2, t1) if flip else (t1, t2)
            gold = lab if lab != "FORWARD_ENTAILMENT" else ("BACKWARD_ENTAILMENT" if flip else "FORWARD_ENTAILMENT")
            items.append({"pair_id": pid, "text1": a, "text2": b, "gold": gold, "group": lab})
    rng.shuffle(items)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/control_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["review_id", "text1", "text2", "label", "note"])
        for i, x in enumerate(items, 1): w.writerow([f"C{i:03d}", x["text1"], x["text2"], "", ""])
    with open(f"{out}/control_key.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "items": {f"C{i:03d}": x for i, x in enumerate(items, 1)}}, f, ensure_ascii=False, indent=1)
    print(len(items), dict(Counter(x["group"] for x in items)))


def score(out, ann_path):
    key = json.load(open(f"{out}/control_key.json", encoding="utf-8"))["items"]
    with open(ann_path, encoding="utf-8") as f:
        ann = {r["review_id"]: SHORT[r["label"].strip().upper()] for r in csv.DictReader(f)}
    assert set(ann) == set(key), "标注要覆盖全部 40 项"
    res = {}
    for g in N:
        c = Counter()
        for rid, k in key.items():
            if k["group"] != g: continue
            x = ann[rid]
            if x == "UNSURE": c["unsure"] += 1; continue
            c["decided"] += 1; c["mismatch"] += x != k["gold"]
        c = dict(c); c["mismatch_rate"] = c.get("mismatch", 0) / c["decided"] if c.get("decided") else None
        res[g] = c
    m = res["EQUIVALENCE"]["mismatch_rate"]
    rule = ("m_orig < 30%：v7 对 E 禁止任何一侧多出信息（含全名对姓、头衔、修饰语）" if m < THRESHOLD else
            "m_orig ≥ 30%：v7 只收紧修饰语和具体程度，人名变体保留；先把门槛问题交回 Pan-Yan-0")
    result = {"by_group": res, "m_orig": m, "threshold": THRESHOLD, "decision": rule}
    with open(f"{out}/control_result.json", "w", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "build": build(os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]))
    elif sys.argv[1] == "score": score(os.path.abspath(sys.argv[2]), sys.argv[3])
    else: raise SystemExit(__doc__)
