"""蒸馏第 1 步（老师导出）的复核：从导出的 teacher.jsonl 独立重算核对项和统计。

1. 每个有序输入只出现一次，四个概率的和为 1；
2. 内层验证部分和第四组训练时保存的 val_probs/epoch<选中轮>.json 逐条相同；
3. 用 families.json 从验证部分的文本反推出内层验证家族，fit 行 = fold 0 以外、不在内层验证家族里的行，应为 2,190 行，每行都有老师概率；
4. 在原始的 2,190 行上（保留重复）重算：老师的准确率、最大概率的中位数、超过 .999 的比例、T = 2 后最大概率的中位数。

用法：python3 qwen_teacher_check.py <导出目录 dico_qwen_teacher_results> <第四组 metrics/qwen_sft> <官方仓库> <第二组 metrics/diagnostics>
"""
import csv, json, math, os, statistics as st, sys
teacher, sft, dico, diag = sys.argv[1:5]
LAB = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
fam = json.load(open(os.path.join(diag, "families.json")))
tr = list(csv.DictReader(open(os.path.join(dico, "final_data/train/dico_nli_train_track1_reference.csv"), encoding="utf-8")))
nonf0 = [r for r in tr if fam["fold_of_family"][fam["family_of_pair"][r["pair_id"]]] != 0]
out = {}
for s in (42, 43, 44):
    L = [json.loads(l) for l in open(os.path.join(teacher, f"track1/seed{s}/teacher.jsonl"))]
    assert len({(x["a"], x["b"]) for x in L}) == len(L), "有重复的输入"
    assert max(abs(sum(x["p"]) - 1) for x in L) < 1e-5
    val = [x for x in L if x["part"] == "val"]
    fit = {(x["a"], x["b"]): x["p"] for x in L if x["part"] == "fit"}
    best = json.load(open(os.path.join(sft, f"track1/seed{s}/done.json")))["best_epoch"]
    vp = json.load(open(os.path.join(sft, f"track1/seed{s}/val_probs/epoch{best}.json")))
    assert len(val) == len(vp) and all(x["p"] == y for x, y in zip(val, vp)), "验证部分和训练时保存的不同"
    valkeys = {(x["a"], x["b"]) for x in val}
    valfams = {fam["family_of_pair"][r["pair_id"]] for r in nonf0 if (r["text1"], r["text2"]) in valkeys}
    rows = [r for r in nonf0 if fam["family_of_pair"][r["pair_id"]] not in valfams]
    assert len(rows) == 2190 and all((r["text1"], r["text2"]) in fit for r in rows)
    P = [fit[(r["text1"], r["text2"])] for r in rows]
    t2 = []
    for p in P:
        q = [math.exp(math.log(x) / 2) for x in p]
        t2.append(max(q) / sum(q))
    out[s] = {"fit_rows": len(rows), "fit_inputs": len(fit), "val_inputs": len(val), "best_epoch": best,
              "teacher_accuracy_on_fit_rows": round(sum(LAB[p.index(max(p))] == r["label"] for p, r in zip(P, rows)) / len(rows), 4),
              "max_prob_median": round(st.median(max(p) for p in P), 4),
              "max_prob_above_0.999": round(sum(max(p) > 0.999 for p in P) / len(P), 4),
              "max_prob_median_T2": round(st.median(t2), 4)}
print(json.dumps(out, ensure_ascii=False, indent=1))
