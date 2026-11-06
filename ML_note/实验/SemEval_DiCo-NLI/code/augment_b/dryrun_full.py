"""全量 +1（方案 B）的预算干跑：主轮之后每折四类还缺多少，补充规划器会下达多少，20% 预算够不够。只用 CPU，不生成。

  python dryrun_full.py <官方仓库目录> <v6 试点包目录> <v6 试点处理结果目录> <主轮批次包目录> <输出 json>

- 需求：每折 fit 部分的 E / F / B / O 行数（+1 = 每折新增和原 fit 一样多的行）。
- 供给：试点已放行的 manifest（实际行，按 usable_folds 计入各折）+ 主轮每个父样本预计产出的一对
  （E → 2 行 E；单向 → 1 行 F + 1 行 B；O → 1 行 O），乘以预计保留率。
- 两种情形：keep = 1（最好情况，下界）；keep = 试点各组的实际保留率（E 43/45、单向 102/105、O 59/60）。
- 补充：supplement.plan，保留率取试点实际值向下取到 0.05 的倍数；家族使用次数从试点和主轮算起（各 1 次），上限 2 次。
- 20% 预算的分母：试点和主轮实际派发的父样本按"一对展开成几行"相加（E、单向 2 行，O 1 行；每个生成行计一次，
  不按它能用于几折重复计）。补充轮下达的行也按同样的单位累计，两轮共用一个台账。
"""
import csv, json, math, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_full import load_world, ROWS_PER_PAIR, GROUP, sig_str  # noqa: E402
import supplement  # noqa: E402
import process_outputs as po  # noqa: E402

CLASSES = ("EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER")
ROWS_OF = {"EQUIVALENCE": {"EQUIVALENCE": 2}, "FORWARD_ENTAILMENT": {"FORWARD_ENTAILMENT": 1, "BACKWARD_ENTAILMENT": 1},
           "NEGATIVE_OTHER": {"NEGATIVE_OTHER": 1}}


def demand_of(world):
    d = {}
    for k, s in world["split"].items():
        c = Counter(world["byid"][i]["label"] for i in s["fit"])
        for cl in CLASSES: d[(int(k), cl)] = c[cl]
    return d


def pilot_supply(pilot_out):
    sup = Counter()
    with open(f"{pilot_out}/manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in r["usable_folds"].split(";"): sup[(int(k), r["label"])] += 1
    return sup


def main_parents(main_pkg):
    out = []
    for x in sorted(os.listdir(f"{main_pkg}/batches")):
        b = po.load(f"{main_pkg}/batches/{x}")
        out += [(p, b["usable_folds"]) for p in b["parents"]]
    return out


def run(off, pilot_pkg, pilot_out, main_pkg):
    world = load_world(off, pilot_pkg)
    dem = demand_of(world)
    pil = pilot_supply(pilot_out)
    mp = main_parents(main_pkg)
    pilot_keep = {"E": 43 / 45, "U": 102 / 105, "O": 59 / 60}
    res = {"demand_rows": {f"{k}-{c}": v for (k, c), v in sorted(dem.items())}}
    for name, keep in (("best_case_keep_1", {"E": 1, "U": 1, "O": 1}), ("pilot_keep", pilot_keep)):
        sup = Counter({k: float(v) for k, v in pil.items()})
        for p, folds in mp:
            for cl, n in ROWS_OF[p["label"]].items():
                for k in folds: sup[(k, cl)] += n * keep[GROUP[p["label"]]]
        deficit = {(k, c): max(0.0, dem[(k, c)] - sup[(k, c)]) for (k, c) in dem}
        surplus = {(k, c): max(0.0, sup[(k, c)] - dem[(k, c)]) for (k, c) in dem}
        res[name] = {"deficit_rows": {f"{k}-{c[:3]}": round(v, 1) for (k, c), v in sorted(deficit.items())},
                     "surplus_rows": {f"{k}-{c[:3]}": round(v, 1) for (k, c), v in sorted(surplus.items())}}
        if name != "pilot_keep": continue
        # ---------- 补充规划的干跑 ----------
        dg = defaultdict(float)
        for (k, c), v in deficit.items(): dg[(k, GROUP[c])] += v
        sig_families = defaultdict(lambda: defaultdict(list))
        for f, p in world["eligible"].items(): sig_families[sig_str(world["signature"][f])][GROUP[p["label"]]].append(f)
        sig_families = {s: dict(v) for s, v in sig_families.items()}
        used = {f: 1 for f in world["pilot_families"]} | {p["family_id"]: 1 for p, _ in mp}
        pilot_parents = []
        for x in sorted(os.listdir(f"{pilot_pkg}/batches")):
            pilot_parents += po.load(f"{pilot_pkg}/batches/{x}")["parents"]
        planned_rows = sum(ROWS_PER_PAIR[p["label"]] for p in pilot_parents) + sum(ROWS_PER_PAIR[p["label"]] for p, _ in mp)
        kr = {g: math.floor(v * 20) / 20 for g, v in pilot_keep.items()}
        batches, status, led = supplement.plan(dict(dg), sig_families, kr, planned_rows, supplement.new_ledger(used), 1)
        twice = sum(1 for f, n in led["uses"].items() if n >= 2) if status == "ok" else None
        res["supplement_round1"] = {
            "status": status, "keep_rate_used": kr, "budget_denominator_rows": planned_rows,
            "budget_rows": supplement.BUDGET * planned_rows, "budget_rule": "两轮累计下达的补充行数 > 0.20 × 分母 就停",
            "batches": len(batches), "pairs": sum(sum(b["quota_pairs"].values()) for b in batches),
            "rows_issued": led["rows_issued"] if status == "ok" else None,
            "families_used_twice": twice,
            "by_group": dict(Counter(b["group"] for b in batches)),
            "deficit_rows_by_fold_group": {f"{k}-{g}": round(v, 1) for (k, g), v in sorted(dg.items())},
            "plan": batches}
        res["supplement_round2"] = "期望下不需要：第 1 轮按预计保留率规划，预计补足；只在实际仍有缺口时启用，台账累计、预算共用"
    res["main_round"] = {"batches": len({x for x in os.listdir(f"{main_pkg}/batches")}), "parents": len(mp),
                         "by_label": dict(Counter(p["label"] for p, _ in mp))}
    return res


if __name__ == "__main__":
    r = run(*map(os.path.abspath, sys.argv[1:5]))
    with open(sys.argv[5], "w", encoding="utf-8") as f: json.dump(r, f, ensure_ascii=False, indent=1)
    show = {k: v for k, v in r.items() if k != "demand_rows"}
    show["supplement_round1"] = {k: v for k, v in r["supplement_round1"].items() if k != "plan"}
    print(json.dumps(show, ensure_ascii=False, indent=1))
