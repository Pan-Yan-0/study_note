"""Compare the two blind reviews of the word-substitution sample (Claude, GPT), then unblind against gold and the model.

Both reviewers labelled the same 45 families twice: sheet A shows only the two extracted words, sheet B the full
phrases, in one direction per family. Labels: E / F / B / O (= neither direction) or 无法确定 (undecided, kept
separate and never counted as O). The key maps sheet ids to families; review_sample.csv holds the gold label, the
per-seed predictions (e6 recipe, symmetrized, out-of-fold) and the cell each family was sampled from.

Usage:
    python3 blind_review.py <diagnostics_dir> [<official train track1 reference.csv>]

Writes <diagnostics_dir>/blind_review/对照表.csv and <diagnostics_dir>/blind_review/summary.json.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REL = "关系（E/F/B/O/无法确定）"
WHY = "依据"
CONF = "把握（高/中/低）"
PHEN = "语义现象（词汇关系/搭配或结构/歧义，可多选，可空）"
TO_LABEL = {"E": "EQUIVALENCE", "F": "FORWARD_ENTAILMENT", "B": "BACKWARD_ENTAILMENT", "O": "NEGATIVE_OTHER",
            "无法确定": None}
SHORT = {v: k for k, v in TO_LABEL.items() if v}
# Claude saw the gold labels and predictions of these 6 families before building the sheets (disclosed in the
# annotation guide); GPT had seen the text (not the labels) of one family in Claude's earlier message.
CLAUDE_NOT_BLIND = {"A06", "A11", "A13", "A24", "A30", "A44", "B03", "B07", "B13", "B14", "B41", "B42"}
GPT_TEXT_SEEN = {"A14", "B35"}
CELLS = ["去掉其余词后变错", "去掉其余词后变对", "都错"]
SEEDS = (42, 43, 44)


def read_sheet(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
    for r in rows:
        assert r[REL] in TO_LABEL, (path, r["编号"], r[REL])
    return {r["编号"]: r for r in rows}


def majority(labels):
    label, n = Counter(labels).most_common(1)[0]
    return label if n >= 2 else None


def kappa(pairs):
    """Cohen's kappa for two raters over the given (label_1, label_2) pairs."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    c1, c2 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / n / n
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def agreement(x, y, ids):
    """Raw agreement between two sheets; 无法确定 is its own category, then the both-decided subset."""
    both = [(TO_LABEL[x[i][REL]], TO_LABEL[y[i][REL]]) for i in ids
            if TO_LABEL[x[i][REL]] and TO_LABEL[y[i][REL]]]
    return {
        "items": len(ids),
        "same_label_incl_undecided": sum(x[i][REL] == y[i][REL] for i in ids),
        "both_undecided": sum(x[i][REL] == y[i][REL] == "无法确定" for i in ids),
        "one_undecided": sum((x[i][REL] == "无法确定") != (y[i][REL] == "无法确定") for i in ids),
        "both_decided": len(both),
        "both_decided_same": sum(a == b for a, b in both),
        "kappa_both_decided": kappa(both),
        "decided_disagreements": sorted(f"{SHORT[a]}/{SHORT[b]}" for a, b in both if a != b),
    }


def main(diag_dir, reference=None):
    diag = Path(diag_dir)
    br = diag / "blind_review"
    sheets = {
        ("Claude", "A"): read_sheet(br / "claude" / "Claude_A_词对_标注.csv"),
        ("Claude", "B"): read_sheet(br / "claude" / "Claude_B_短语_标注.csv"),
        ("GPT", "A"): read_sheet(br / "gpt" / "A_词对_GPT_冻结.csv"),
        ("GPT", "B"): read_sheet(br / "gpt" / "B_短语_GPT_冻结.csv"),
    }
    key = json.load(open(br / "key_不要发给标注者.json", encoding="utf-8"))
    sample = {r["instance_id"]: r for r in csv.DictReader(open(diag / "review_sample.csv", encoding="utf-8-sig"))}

    # ---- integrity: same ids and text in both reviewers' sheets, key consistent with the sample ----
    for s, cols in (("A", ("词1", "词2")), ("B", ("短语1", "短语2"))):
        c, g = sheets[("Claude", s)], sheets[("GPT", s)]
        assert list(c) == list(g) and len(c) == 45, s
        for i in c:
            assert all(c[i][k] == g[i][k] for k in cols), (s, i)
    fam = {}
    for sid, v in key.items():
        fam.setdefault(v["family"], {})[sid[0]] = sid
        r = sample[v["instance_id"]]
        assert r["家族"] == v["family"]
        if sid[0] == "A":
            assert (sheets[("Claude", "A")][sid]["词1"], sheets[("Claude", "A")][sid]["词2"]) == (r["被替换的词"], r["替换成的词"]), sid
        else:
            assert (sheets[("Claude", "B")][sid]["短语1"], sheets[("Claude", "B")][sid]["短语2"]) == (r["text1"], r["text2"]), sid
    assert len(fam) == 45 and all(set(d) == {"A", "B"} for d in fam.values())
    assert all(key[d["A"]]["instance_id"] == key[d["B"]]["instance_id"] for d in fam.values())
    if reference:
        gold = {r["instance_id"]: r["label"] for r in csv.DictReader(open(reference, encoding="utf-8"))}
        assert all(gold[r["instance_id"]] == r["金标"] for r in sample.values())

    # ---- one row per family ----
    rows = []
    for f, d in fam.items():
        a, b = d["A"], d["B"]
        r = sample[key[a]["instance_id"]]
        gold = r["金标"]
        full = [r[f"完整_seed{s}"] for s in SEEDS]
        pair = [r[f"只给词对_seed{s}"] for s in SEEDS]
        row = {
            "家族": f, "instance_id": r["instance_id"], "A编号": a, "B编号": b,
            "词1": r["被替换的词"], "词2": r["替换成的词"], "短语1": r["text1"], "短语2": r["text2"],
            "金标": SHORT[gold], "格子": r["格子（多数票）"],
            "模型_完整短语（3个种子）": "/".join(SHORT[x] for x in full),
            "模型_只给词对（3个种子）": "/".join(SHORT[x] for x in pair),
            "模型_完整多数票": SHORT.get(majority(full), "无"), "模型_词对多数票": SHORT.get(majority(pair), "无"),
        }
        for who in ("Claude", "GPT"):
            for s, sid in (("A", a), ("B", b)):
                x = sheets[(who, s)][sid]
                row[f"{who}_{s}"] = x[REL]
                row[f"{who}_{s}_把握"] = x[CONF]
            row[f"{who}_B_语义现象"] = sheets[(who, "B")][b][PHEN]
        row["Claude_非盲"] = int(a in CLAUDE_NOT_BLIND)
        row["GPT_见过文本"] = int(a in GPT_TEXT_SEEN)
        # how many reviewers' phrase label equals gold, and whether the two reviewers agree
        gb = [TO_LABEL[row["Claude_B"]], TO_LABEL[row["GPT_B"]]]
        row["B_同意金标的人数"] = sum(x == gold for x in gb)
        row["B_双方一致"] = int(row["Claude_B"] == row["GPT_B"])
        ga = [TO_LABEL[row["Claude_A"]], TO_LABEL[row["GPT_A"]]]
        cons_a = ga[0] if ga[0] and ga[0] == ga[1] else None
        row["A_双方一致的关系"] = SHORT[cons_a] if cons_a else ""
        row["A_共识与短语金标"] = "" if not cons_a else ("相同" if cons_a == gold else "不同")
        rows.append(row)
    rows.sort(key=lambda x: (CELLS.index(x["格子"]), x["家族"]))

    # ---- summary ----
    ids = {s: list(sheets[("Claude", s)]) for s in "AB"}
    summary = {"families": len(rows), "agreement": {}, "vs_gold": {}, "by_cell": {}}
    for s in "AB":
        summary["agreement"][s] = agreement(sheets[("Claude", s)], sheets[("GPT", s)], ids[s])
        blind = [i for i in ids[s] if i not in CLAUDE_NOT_BLIND]
        summary["agreement"][s + "_excluding_claude_not_blind"] = agreement(
            sheets[("Claude", s)], sheets[("GPT", s)], blind)
    gold_of = {sid: sample[key[sid]["instance_id"]]["金标"] for sid in key}
    for who in ("Claude", "GPT"):
        for s in "AB":
            for scope, keep in (("all", lambda i: True), ("excluding_claude_not_blind", lambda i: i not in CLAUDE_NOT_BLIND)):
                sel = [i for i in ids[s] if keep(i)]
                dec = [i for i in sel if TO_LABEL[sheets[(who, s)][i][REL]]]
                summary["vs_gold"][f"{who}_{s}_{scope}"] = {
                    "items": len(sel), "decided": len(dec),
                    "equal_gold": sum(TO_LABEL[sheets[(who, s)][i][REL]] == gold_of[i] for i in dec),
                }
    for cell in CELLS:
        rs = [x for x in rows if x["格子"] == cell]
        summary["by_cell"][cell] = {
            "families": len(rs),
            "model_full_majority_correct": sum(x["模型_完整多数票"] == x["金标"] for x in rs),
            "model_pair_majority_correct": sum(x["模型_词对多数票"] == x["金标"] for x in rs),
            "B_reviewers_agreeing_with_gold": dict(sorted(Counter(x["B_同意金标的人数"] for x in rs).items())),
            "B_both_decided_same_but_not_gold": [x["B编号"] for x in rs if x["B_双方一致"] and x["Claude_B"] != "无法确定"
                                                 and x["B_同意金标的人数"] == 0],
            "A_consensus_vs_phrase_gold": dict(Counter(x["A_共识与短语金标"] or "无共识" for x in rs)),
        }
    summary["B_both_decided_same_but_not_gold"] = [
        {k: x[k] for k in ("B编号", "家族", "短语1", "短语2", "金标", "Claude_B", "格子", "Claude_非盲")}
        for x in rows if x["B_双方一致"] and x["Claude_B"] != "无法确定" and x["B_同意金标的人数"] == 0]
    with open(br / "对照表.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(summary, open(br / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
