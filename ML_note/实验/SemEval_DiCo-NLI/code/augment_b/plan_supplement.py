"""按合并后的实际缺口规划补充批次（supplement.plan 的包装；第 1、2 轮共用台账）。

  python plan_supplement.py <官方仓库目录> <v6 试点包目录> <合并输出目录> <轮次 1|2> <上一轮计划.json 或 -> <输出计划.json> <各阶段包目录...>

- 缺口：merge_full.py 的 report.json 里 fold_quota_short（每折每组缺的行数，单向组 = F + B）。
- 保留率：已完成各阶段的 保留对数 / 返回对数，按组算，向下取到 0.05 的倍数。
- 家族使用次数：已派发的全部批次（试点、主轮、已下达的补充）里每个家族出现的次数；上限 supplement.MAX_USES。
- 预算分母：试点和主轮派发的父样本按一对展开的行数相加（E、单向 2 行，O 1 行），整数，写进计划；第 2 轮沿用第 1 轮的分母和台账。
"""
import json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import process_outputs as po  # noqa: E402
import supplement  # noqa: E402
from build_full import load_world, GROUP, ROWS_PER_PAIR, sig_str  # noqa: E402

LETTER = {"EQUIVALENCE": "E", "FORWARD_ENTAILMENT": "U", "NEGATIVE_OTHER": "O"}


def main(off, pilot_pkg, merged, rnd, prev, dest, pkgs):
    world = load_world(off, pilot_pkg)
    rep = po.load(f"{merged}/report.json")
    deficit = {}
    for k, v in rep["stats"]["fold_quota_short"].items():
        f, g = k.split("-", 1); deficit[(int(f), LETTER[g])] = v
    disp, base_rows = Counter(), 0
    for pkg in pkgs:
        for x in sorted(os.listdir(f"{pkg}/batches")):
            b = po.load(f"{pkg}/batches/{x}")
            for p in b["parents"]:
                disp[p["family_id"]] += 1
                if b["stage"] in ("pilot", "main"): base_rows += ROWS_PER_PAIR[p["label"]]
    returned, kept = Counter(), Counter()
    for pkg in pkgs:
        for x in sorted(os.listdir(f"{pkg}/batches")):
            b = po.load(f"{pkg}/batches/{x}")
            for p in b["parents"]: returned[GROUP[p["label"]]] += 1
    m = po.read_csv(f"{merged}/manifest.csv")
    for r in m:
        if r["direction"] == "original": kept[GROUP[r["label"]]] += 1
    keep = {g: (kept[g] / returned[g] if returned[g] else 0) for g in "EUO"}
    sig_families = defaultdict(lambda: defaultdict(list))
    for f, p in world["eligible"].items(): sig_families[sig_str(world["signature"][f])][GROUP[p["label"]]].append(f)
    sig_families = {s: dict(v) for s, v in sig_families.items()}
    if rnd == 1:
        ledger, denom = supplement.new_ledger(dict(disp)), base_rows
    else:
        pp = po.load(prev); ledger, denom = pp["ledger"], pp["budget_denominator_rows"]
    import math
    kr = {g: math.floor(v * 20) / 20 for g, v in keep.items()}
    batches, status, led = supplement.plan(deficit, sig_families, kr, denom, ledger, rnd)
    plan = {"round": rnd, "status": status, "batches": batches, "ledger": led, "budget_denominator_rows": denom,
            "keep_rate_observed": keep, "keep_rate_used": kr, "deficit_rows": {f"{k}-{g}": v for (k, g), v in sorted(deficit.items())},
            "merged_report_sha256": po.sha_file(f"{merged}/report.json")}
    po.dump(dest, plan)
    print(json.dumps({k: v for k, v in plan.items() if k not in ("batches", "ledger")} | {"batches": len(batches)}, ensure_ascii=False, indent=1))
    return plan


if __name__ == "__main__":
    a = sys.argv
    main(os.path.abspath(a[1]), os.path.abspath(a[2]), os.path.abspath(a[3]), int(a[4]), a[5] if a[5] != "-" else None,
         os.path.abspath(a[6]), [os.path.abspath(x) for x in a[7:]])
