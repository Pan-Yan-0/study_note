"""实验 B：E 类按原数据校准的复审（11-06，Pan-Yan-0 定；在看到复审结果之前写定）。

  python e_recheck.py build <官方仓库目录> <Haiku 合并目录> <E 对照目录> <输出目录>
  python e_recheck.py score <输出目录> <Haiku 合并目录> <最终标注.csv>

原则：比赛数据视为绝对准确；审核人的判断要和比赛数据的标法对齐，生成数据和原数据用同一把尺子量。
- 审核表 60 对，隐藏来源：Haiku E 20、原数据 E 20（计分用）；干扰项 Haiku 单向 5、原数据单向 5、Haiku O 5、原数据 O 5（只报告）。
  Haiku 的对取自被选用的对，排除已经审过的 60 对；原数据每个家族只取一个代表，排除 E 对照用过的 40 对和只差数量的对。
  单向随机展示方向。种子 SEED。
- 参考卡：E 对照的 40 对连同金标一起给审核人，作为"比赛数据怎么标"的例子。
- 流程：Claude 先标 → Pan-Yan-0 审核修改 → GPT 对照参考卡再审；计分用最后一步的标签。三步不独立，报告写明。
- 放行规则：生成 E 的不一致率 ≤ 原数据 E 的不一致率 + MARGIN，两边可判定项都 ≥ 10。
  通过时，合并目录的审核结果改为 P1C-R：E 用这次复审，单向和 O 沿用 P1C 的结果（已经按绝对门槛通过）。
"""
import csv, json, os, random, shutil, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import process_outputs as po  # noqa: E402
from label_conventions import number_pair  # noqa: E402

SEED, MARGIN, MIN_DECIDED = 20261109, 0.15, 10
PLAN = [("haiku", "EQUIVALENCE", 20), ("orig", "EQUIVALENCE", 20), ("haiku", "FORWARD_ENTAILMENT", 5),
        ("orig", "FORWARD_ENTAILMENT", 5), ("haiku", "NEGATIVE_OTHER", 5), ("orig", "NEGATIVE_OTHER", 5)]
LETTER = {"EQUIVALENCE": "EQ", "FORWARD_ENTAILMENT": "FE", "BACKWARD_ENTAILMENT": "BE", "NEGATIVE_OTHER": "O"}


def build(off, merged, control, out, stages=None, seed=SEED, exclude_keys=()):
    """stages：只从这些阶段抽 Haiku 对（规模轮复审，第 15 节）；exclude_keys：之前复审的 recheck_key.json，其中的原数据对不再抽。
    都不给时和 11-06 的复审完全相同。"""
    reviewed = {v["pair_uid"] for v in po.load(f"{merged}/review_key.json")["items"].values()}
    sel_file = f"{merged}/fold_selection_scale.csv" if stages is not None else f"{merged}/fold_selection.csv"  # 规模轮：任一规模被选用
    selected = {r["pair_uid"] for r in po.read_csv(sel_file)}
    haiku = {}
    for r in po.read_csv(f"{merged}/manifest.csv"):
        if r["direction"] != "original" or r["pair_uid"] in reviewed or r["pair_uid"] not in selected: continue
        if stages is not None and r["batch_id"].rsplit("_b", 1)[0] not in stages: continue
        haiku.setdefault(r["label"], []).append((r["pair_uid"], r["text1"], r["text2"]))
    ctrl = po.load(f"{control}/control_key.json")["items"]
    used = {v["pair_id"] for v in ctrl.values()}
    for kp in exclude_keys:
        used |= {v["id"] for v in po.load(kp)["items"].values() if v["source"] == "orig"}
    reps = {}
    for r in po.read_csv(f"{off}/final_data/train/dico_nli_train_track1_participant_labeled.csv"):
        if not r["instance_id"].endswith("__original"): continue
        reps.setdefault(frozenset((r["text1"].strip().lower(), r["text2"].strip().lower())), []).append(r)
    orig = {}
    for v in reps.values():
        r = sorted(v, key=lambda r: r["pair_id"])[0]
        t1, t2, lab = r["text1"], r["text2"], r["label"]
        if lab == "BACKWARD_ENTAILMENT": t1, t2, lab = t2, t1, "FORWARD_ENTAILMENT"
        if r["pair_id"] in used or number_pair(t1, t2): continue
        orig.setdefault(lab, []).append((r["pair_id"], t1, t2))
    rng = random.Random(seed)
    items = []
    for src, lab, n in PLAN:
        pool = sorted((haiku if src == "haiku" else orig)[lab])
        for uid, t1, t2 in rng.sample(pool, n):
            flip = lab == "FORWARD_ENTAILMENT" and rng.random() < 0.5
            a, b = (t2, t1) if flip else (t1, t2)
            exp = "BACKWARD_ENTAILMENT" if flip else lab
            items.append({"source": src, "id": uid, "group": lab, "expected": exp, "text1": a, "text2": b})
    rng.shuffle(items)
    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(out)
    key = {f"Q{i:03d}": x for i, x in enumerate(items, 1)}
    po.write_csv(f"{out}/recheck_sheet.csv", [{"review_id": k, "text1": x["text1"], "text2": x["text2"], "label": "", "note": ""}
                                               for k, x in key.items()], ["review_id", "text1", "text2", "label", "note"])
    meta = {} if stages is None and not exclude_keys else {"stages": sorted(stages or []), "exclude_keys_sha256": [po.sha_file(k) for k in exclude_keys]}
    po.dump(f"{out}/recheck_key.json", {"seed": seed, "margin": MARGIN, **meta, "items": key})
    lines = ["# 参考卡：比赛数据怎么标（E 对照的 40 对，带金标）", "",
             "比赛数据视为绝对准确。拿不准时，和下面最像的例子比，按它的标法标。", "",
             "| 编号 | text1 | text2 | 金标 |", "|---|---|---|---|"]
    for rid, k in sorted(ctrl.items()):
        lines.append(f"| {rid} | {k['text1']} | {k['text2']} | {LETTER[k['gold']]} |")
    with open(f"{out}/参考卡.md", "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")
    print(len(items), dict(Counter((x["source"], x["group"][:3]) for x in items)))


def score(out, merged, final_path):
    key = po.load(f"{out}/recheck_key.json")["items"]
    alias = {"EQ": "E", "FE": "F", "BE": "B"}  # 审核表用的是 EQ / FE / BE 写法
    norm_path = f"{out}/final_normalized.csv"
    rows = po.read_csv(final_path)
    for r in rows: r["label"] = alias.get(r["label"].strip().upper(), r["label"])
    po.write_csv(norm_path, rows, list(rows[0]))
    ann, errs = po.read_ann(norm_path, list(key))
    if errs: raise SystemExit(f"标注不合法：{errs[:5]}")
    res = {}
    for src in ("haiku", "orig"):
        for g in po.GROUPS:
            c = Counter()
            for rid, k in key.items():
                if k["source"] != src or k["group"] != g: continue
                x = ann[rid]
                if x == "UNSURE": c["unsure"] += 1; continue
                c["decided"] += 1; c["mismatch"] += x != k["expected"]
            c = dict(c); c["mismatch_rate"] = c.get("mismatch", 0) / c["decided"] if c.get("decided") else None
            res[f"{src}-{g}"] = c
    h, o = res["haiku-EQUIVALENCE"], res["orig-EQUIVALENCE"]
    if min(h.get("decided", 0), o.get("decided", 0)) < MIN_DECIDED: e_status = "insufficient"
    else: e_status = "pass" if h["mismatch_rate"] <= o["mismatch_rate"] + MARGIN else "fail"
    recheck = {"by_source_group": res, "e_rule": f"haiku E 不一致率 ≤ 原数据 E 不一致率 + {MARGIN}，两边可判定 ≥ {MIN_DECIDED}",
               "e_status": e_status, "inputs_sha256": {"final": po.sha_file(final_path)}}
    po.dump(f"{out}/recheck_result.json", recheck)
    # 合并目录：E 用复审，单向和 O 沿用 P1C
    p1c_path = f"{merged}/review_result_P1C.json"
    if not os.path.exists(p1c_path): shutil.copyfile(f"{merged}/review_result.json", p1c_path)
    p1c = po.load(p1c_path)
    assert p1c["protocol"] == "P1C"
    by = {g: p1c["by_group"][g] for g in ("FORWARD_ENTAILMENT", "NEGATIVE_OTHER")}
    by["EQUIVALENCE"] = {"recheck": res["haiku-EQUIVALENCE"], "original_reference": res["orig-EQUIVALENCE"], "status": e_status}
    status = "pass" if all(v["status"] == "pass" for v in by.values()) else (
        "fail" if any(v["status"] == "fail" for v in by.values()) else "insufficient")
    po.dump(f"{merged}/review_result.json", {"protocol": "P1C-R", "status": status, "errors": [], "by_group": by,
                                             "recheck_result_sha256": po.sha_file(f"{out}/recheck_result.json"),
                                             "p1c_result_sha256": po.sha_file(p1c_path), "bound_to": po.bound_files(merged)})
    gate = po.write_gate(merged)
    print(json.dumps({"recheck": recheck, "review_status": status, "ready_for_training": gate["ready_for_training"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "build":
        opt = lambda k: next((x.split("=", 1)[1] for x in a if x.startswith(f"--{k}=")), None)
        st, sd, ex = opt("stages"), opt("seed"), opt("exclude-keys")
        build(*map(os.path.abspath, a[2:6]), stages=st.split(",") if st else None, seed=int(sd) if sd else SEED,
              exclude_keys=tuple(os.path.abspath(p) for p in ex.split(",")) if ex else ())
    elif a[1] == "score": score(os.path.abspath(a[2]), os.path.abspath(a[3]), a[4])
    else: raise SystemExit(__doc__)
