"""反事实最小对 B1 测试集（冻结计划 11-12「研究对齐 v4」B 部分的 B1 诊断；11-13 登记，规则在生成和审核之前写定）。

  python3 -I code/cf_b1.py build   <官方仓库目录> <输出目录>              选父样本和对照，写 batches/*.json、b1_key.json（封存）
  python3 -I code/cf_b1.py prep    <输出目录> <运行目录>                  写 prompts/<batch>.txt（逐字节等于批次文件的 prompt）
  python3 -I code/cf_b1.py harvest <输出目录> <运行目录> <subagents 目录> 从 transcript 登记每次派发（同试点）
  python3 -I code/cf_b1.py sheet   <输出目录> <运行目录> <官方仓库目录>   合并合格编辑，写 review_sheet.csv 和 review_key.json
  python3 -I code/cf_b1.py lock    <输出目录> <盲标.csv>                   按事先规则锁定测试集，写 b1_testset.json

父样本（只用金标和文本，不用任何模型预测）
  每个 E 家族只在一折里是留出：家族的 B1 编辑对只用那一折的 Q0 评测。条件和试点相同：金标 E 的家族，取 pair_id 最小的 __original
  实例；两段 1–6 个词、至少共享一个实词、两段不同；排除 Claude 看过金标的家族（cf_pilot.seen_families）和试点的 60 个父样本家族。
  符合条件的家族全部使用，每个家族做两种编辑：按种子 SEED 打乱后第 n 个家族，CF-SPEC 改 text1（n 偶数，候选 F）或 text2（候选 B），
  CF-HEAD 改另一侧（候选 O）。提示词模板、格式检查和试点逐字相同（cf_pilot.PROMPTS、cf_pilot.check_item），每批 10 项。

审核（单阶段盲标，审阅人 Codex；Claude 不标）
  审核表 = B1 的全部合格编辑 + 试点里合格但没进试点审核表的 13 条编辑 + 20 条同现象的原数据对照（SPEC 型 F/B 各 5、HEAD 型 O 10；
  排除看过金标的家族、试点父样本和对照、B1 父样本），顺序随机（种子 SEED + 1）。只给编号和两个短语，对照参考卡判 E / F / B / O / U。
  这一轮不揭示答案、不做第二阶段：锁定靠"规则给的候选标签"和"审阅人的盲标"两个独立来源一致。

锁定规则（lock 输出）
  一条编辑进入测试集，当且仅当审阅人的盲标等于候选标签；U 和不一致都排除，按编辑和折计数。
  试点里已审的 40 条编辑用 Codex 的第一阶段盲标（看答案前，`fa6aea37…`）按同一规则锁定，记为 subset = pilot；
  新生成的记为 subset = b1。两者一起评测，结果按 subset 分开报和合并报。对照只用来报告审阅人和金标的分歧，不进测试集。

评测（推理包另写；这里先登记口径）
  每条锁定的编辑对 (t1', t2') 和它的父样本 (t1, t2) 都按两个方向推理，用该家族留出所在折的 Q0 三个种子，T4 fp16，
  先按归档复现该折 seed 42 的预测（同 A0 的门槛），再评编辑对。预测用对称化：argmax((p(a, b) + swapFB(p(b, a))) / 2)。
  每种编辑报：编辑后准确率（预测 = 锁定标签）；联合准确率（父样本预测 E 且编辑对预测 = 锁定标签）；条件比例（联合 / 父样本预测 E，
  给分母）。每个种子单独算，再报三种子平均；区间用按家族的 2,000 次重抽样（三种子共用同一组抽样）。另报原方向（不对称化）的同样三项、
  编辑对被判成的类别分布。只是诊断，不训练、不做放行判定。
"""
import csv, json, os, random, sys
from collections import Counter

SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{SE}/code")
sys.path.insert(0, f"{SE}/code/augment_b")
import cf_pilot as P  # noqa: E402

SEED = 20261114
BATCH = 10
N_CTRL = {"SPEC_F": 5, "SPEC_B": 5, "HEAD": 10}
PILOT = f"{SE}/metrics/cf_pilot"
STAGE1_SHA = "fa6aea372b4763fca886706162230a9fc79e6954e28bf10374f8d38467ade1b8"
sha, rj, wj = P.sha, P.rj, P.wj


def data(off):
    rows = list(csv.DictReader(open(f"{off}/final_data/train/dico_nli_train_track1_participant_labeled.csv", encoding="utf-8")))
    byid = {r["instance_id"]: r for r in rows}
    fam = rj(f"{SE}/metrics/diagnostics/families.json")
    family = {i: fam["family_of_pair"][r["pair_id"]] for i, r in byid.items()}
    fold = {i: int(fam["fold_of_family"][str(family[i])]) for i in byid}
    return byid, family, fold


def build(off, out):
    import boundary_review as br
    byid, family, fold = data(off)
    seen, seen_src = P.seen_families(off)
    pk = rj(f"{PILOT}/pilot_key.json")
    pilot_par = {family[x["parent"]] for x in pk["items"]}
    pilot_ctrl = {c["family"] for v in pk["controls"].values() for c in v}
    content = lambda t: {x for x in P.toks(t) if x.isalpha() and x not in br.STOP}
    rep = {}
    for i, r in byid.items():
        if i.endswith("__original") and (family[i] not in rep or r["pair_id"] < byid[rep[family[i]]]["pair_id"]): rep[family[i]] = i
    cand = []
    for f, i in sorted(rep.items(), key=lambda kv: kv[1]):
        r = byid[i]
        if r["label"] != P.SHORT["E"] or f in seen or f in pilot_par: continue
        n1, n2 = len(r["text1"].split()), len(r["text2"].split())
        if not (1 <= n1 <= 6 and 1 <= n2 <= 6) or r["text1"].strip().lower() == r["text2"].strip().lower(): continue
        if not content(r["text1"]) & content(r["text2"]): continue
        cand.append(i)
    rng = random.Random(SEED)
    order = cand[:]; rng.shuffle(order)
    items = []
    for n, i in enumerate(order):
        s_side = "text1" if n % 2 == 0 else "text2"
        h_side = "text2" if s_side == "text1" else "text1"
        items.append({"op": "SPEC", "parent": i, "family": family[i], "fold": fold[i], "edit_side": s_side,
                      "target": P.SHORT["F"] if s_side == "text1" else P.SHORT["B"]})
        items.append({"op": "HEAD", "parent": i, "family": family[i], "fold": fold[i], "edit_side": h_side, "target": P.SHORT["O"]})
    os.makedirs(f"{out}/batches", exist_ok=True)
    batches = {}
    for op in ("SPEC", "HEAD"):
        its = [x for x in items if x["op"] == op]
        for bn in range(0, len(its), BATCH):
            bid = f"cf_b1_{op.lower()}_b{bn // BATCH + 1:02d}"
            chunk = its[bn:bn + BATCH]
            lines = [f'{k}. P1: "{byid[x["parent"]]["text1"]}" | P2: "{byid[x["parent"]]["text2"]}" | edit: {"P1" if x["edit_side"] == "text1" else "P2"}'
                     for k, x in enumerate(chunk, 1)]
            prompt = P.PROMPTS[op].format(items="\n".join(lines), batch_id=bid, n=len(chunk))
            b = {"batch_id": bid, "op": op, "prompt": prompt, "prompt_sha256": sha(prompt.encode("utf-8")),
                 "items": [{"k": k, "parent": x["parent"], "edit_side": x["edit_side"], "orig": byid[x["parent"]][x["edit_side"]],
                            "other": byid[x["parent"]]["text2" if x["edit_side"] == "text1" else "text1"]} for k, x in enumerate(chunk, 1)]}
            wj(f"{out}/batches/{bid}.json", b); batches[bid] = b
    used = {family[i] for i in cand} | pilot_par | pilot_ctrl | seen
    pools = {"SPEC_F": [], "SPEC_B": [], "HEAD": []}
    for f in sorted(set(family.values())):
        if f in used: continue
        for i in sorted(i for i in byid if family[i] == f):
            r = byid[i]; l = r["label"]
            a1, a2 = P.insertion(r["text2"], r["text1"]), P.insertion(r["text1"], r["text2"])
            if l == P.SHORT["F"] and a1 and len(a1) <= 3: pools["SPEC_F"].append(i); break
            if l == P.SHORT["B"] and a2 and len(a2) <= 3: pools["SPEC_B"].append(i); break
            if l == P.SHORT["O"] and P.substitution(r["text1"], r["text2"]): pools["HEAD"].append(i); break
    ctrl = {k: rng.sample(v, N_CTRL[k]) for k, v in sorted(pools.items())}
    key = {"version": "cf-b1-v1", "seed": SEED, "rules": "见 code/cf_b1.py 开头", "parents": len(cand),
           "parents_per_fold": dict(sorted(Counter(fold[i] for i in cand).items())),
           "excluded": {"seen_families": len(seen), "pilot_parents": len(pilot_par)}, "seen_sources": seen_src,
           "items": items, "controls": {k: [{"instance_id": i, "family": family[i], "gold": byid[i]["label"]} for i in v] for k, v in ctrl.items()},
           "control_pools": {k: len(v) for k, v in pools.items()}, "batches": {k: v["prompt_sha256"] for k, v in batches.items()},
           "prompt_templates_sha256": {op: sha(P.PROMPTS[op].encode("utf-8")) for op in P.PROMPTS}}
    wj(f"{out}/b1_key.json", key)
    print(f"父样本 {len(cand)}（每折 {key['parents_per_fold']}）；编辑 {len(items)}；批次 {len(batches)}；对照池 {key['control_pools']}")


def sheet(out, run, off):
    byid, family, fold = data(off)
    key = rj(f"{out}/b1_key.json")
    target = {(x["parent"], x["op"]): x for x in key["items"]}
    res, stats = P.collect(out, run)
    rows = []
    for x in res:
        t1, t2 = (x["new_phrase"], x["other"]) if x["edit_side"] == "text1" else (x["other"], x["new_phrase"])
        it = target[(x["parent"], x["op"])]
        rows.append({"kind": "CF", "subset": "b1", "op": x["op"], "text1": t1, "text2": t2, "expected": it["target"], "parent": x["parent"],
                     "family": it["family"], "fold": it["fold"], "edit_side": x["edit_side"], "batch_id": x["batch_id"], "k": x["k"]})
    # 试点里合格但没进试点审核表的编辑
    prk = rj(f"{PILOT}/review_key.json")["items"]
    reviewed = {(x["parent"], x["op"]) for x in prk.values() if x["kind"] == "CF"}
    ptarget = {(x["parent"], x["op"]): x["target"] for x in rj(f"{PILOT}/pilot_key.json")["items"]}
    for x in rj(f"{PILOT}/valid_edits.json"):
        if (x["parent"], x["op"]) in reviewed: continue
        t1, t2 = (x["new_phrase"], x["other"]) if x["edit_side"] == "text1" else (x["other"], x["new_phrase"])
        rows.append({"kind": "CF", "subset": "pilot", "op": x["op"], "text1": t1, "text2": t2, "expected": ptarget[(x["parent"], x["op"])],
                     "parent": x["parent"], "family": family[x["parent"]], "fold": fold[x["parent"]], "edit_side": x["edit_side"],
                     "batch_id": x["batch_id"], "k": x["k"]})
    for k, v in key["controls"].items():
        for c in v:
            r = byid[c["instance_id"]]
            rows.append({"kind": "CTRL", "op": k.split("_")[0], "text1": r["text1"], "text2": r["text2"], "expected": c["gold"],
                         "instance_id": c["instance_id"], "family": c["family"]})
    rng = random.Random(SEED + 1)
    rng.shuffle(rows)
    with open(f"{out}/review_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["review_id", "text1", "text2", "label", "evidence"])
        for n, r in enumerate(rows, 1): w.writerow([f"D{n:03d}", r["text1"], r["text2"], "", ""])
    rk = {"version": "cf-b1-review-v1", "seed": SEED + 1, "generation_stats": stats, "valid_b1": len(res),
          "items": {f"D{n:03d}": r for n, r in enumerate(rows, 1)}}
    wj(f"{out}/review_key.json", rk)
    wj(f"{out}/valid_edits.json", res)
    print("生成统计", stats, "| 审核表", len(rows), "行", dict(Counter((r["kind"], r.get("subset", ""), r["op"]) for r in rows)))
    print("review_sheet.csv", sha(open(f"{out}/review_sheet.csv", "rb").read())[:12], "review_key.json",
          sha(open(f"{out}/review_key.json", "rb").read())[:12])


def labels(path, ids):
    lab = {r["review_id"]: r["label"].strip().upper() for r in csv.DictReader(open(path, encoding="utf-8"))}
    assert set(lab) == set(ids) and all(v in P.SHORT for v in lab.values()), f"{path} 标注表不完整或有非法标签"
    return lab


def lock(out, blind):
    rk = rj(f"{out}/review_key.json")["items"]
    lab = labels(blind, rk)
    test, tally = [], Counter()
    for rid, x in sorted(rk.items()):
        if x["kind"] != "CF": continue
        ok = lab[rid] != "U" and P.SHORT[lab[rid]] == x["expected"]
        tally[(x["subset"], x["op"], "locked" if ok else ("U" if lab[rid] == "U" else "disagree"))] += 1
        if ok: test.append({k: x[k] for k in ("subset", "op", "text1", "text2", "parent", "family", "fold", "edit_side")} | {"label": x["expected"], "review_id": rid})
    # 试点已审的 40 条：用第一阶段盲标
    s1 = f"{PILOT}/review/review_sheet_Codex_stage1.csv"
    assert sha(open(s1, "rb").read()) == STAGE1_SHA, "试点盲标文件的哈希不对"
    prk = rj(f"{PILOT}/review_key.json")["items"]
    plab = labels(s1, prk)
    fam = rj(f"{SE}/metrics/diagnostics/families.json")
    for rid, x in sorted(prk.items()):
        if x["kind"] != "CF": continue
        ok = plab[rid] != "U" and P.SHORT[plab[rid]] == x["expected"]
        tally[("pilot", x["op"], "locked" if ok else ("U" if plab[rid] == "U" else "disagree"))] += 1
        if ok:
            test.append({"subset": "pilot", "op": x["op"], "text1": x["text1"], "text2": x["text2"], "parent": x["parent"],
                         "edit_side": x["edit_side"], "label": x["expected"], "review_id": "pilot:" + rid})
    # 补家族和折
    off = os.environ.get("DICO_OFFICIAL")
    assert off, "需要 DICO_OFFICIAL（官方仓库目录）"
    byid, family, fold = data(off)
    for t in test:
        t["family"], t["fold"] = family[t["parent"]], fold[t["parent"]]
    ctrl = Counter()
    for rid, x in rk.items():
        if x["kind"] == "CTRL":
            ctrl[(x["op"], "U" if lab[rid] == "U" else ("agree" if P.SHORT[lab[rid]] == x["expected"] else "disagree"))] += 1
    res = {"version": "cf-b1-test-v1", "rule": "盲标 = 候选标签才锁定；U 和不一致排除",
           "labels_sha256": sha(open(blind, "rb").read()), "pilot_stage1_sha256": STAGE1_SHA,
           "tally": {"|".join(k): v for k, v in sorted(tally.items())}, "controls": {"|".join(k): v for k, v in sorted(ctrl.items())},
           "per_fold": {f"{s}|{op}|fold{k}": v for (s, op, k), v in sorted(Counter((t["subset"], t["op"], t["fold"]) for t in test).items())},
           "items": test}
    wj(f"{out}/b1_testset.json", res)
    print("锁定", len(test), "条；", res["tally"]); print("对照", res["controls"])


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 2: raise SystemExit(__doc__)
    ab = lambda k: os.path.abspath(a[k])
    if a[1] == "build": build(ab(2), ab(3))
    elif a[1] == "prep": P.prep(ab(2), ab(3))
    elif a[1] == "harvest": P.harvest(ab(2), ab(3), ab(4))
    elif a[1] == "sheet": sheet(ab(2), ab(3), ab(4))
    elif a[1] == "lock": lock(ab(2), ab(3))
    else: raise SystemExit(__doc__)
