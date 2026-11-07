"""规模轮（+2 / +4）的补充批次规划（Pan-Yan-0 11-07 定的规则，在看到规模轮输出之前写定）。

  python plan_scale_supp.py <官方仓库> <v6 试点包> <合并目录> <标签 s2a|s2b|s4a|s4b> <上一份台账.json 或 -> <输出计划.json> <各阶段包目录...>

- 缺口：merge_full.py --tiers 的 report.json 里 fold_quota_short（最高规模那一层每折每组缺的行数）。
- 家族上限：+k 每个家族最多 2k 次（+2：4 次，+4：8 次），按标签的数字决定。使用次数从全部已派发批次重新数。
- 保留率：已完成各阶段按组的 保留对数 / 返回对数，向下取到 0.05 的倍数（和 plan_supplement.py 相同）。
- 预算：+1 以来所有补充累计下达的行数 ≤ 20% × （+k 用到的各轮派发的行数：试点 + 主轮 + r2 … rk）。
  +1 的补充（supp1，224 行）计入累计。分母按一对展开的行数算（E、单向 2 行，O 1 行）。
- 同一规模最多两轮补充（a、b）；第二轮之后仍有缺口就停下报告。
"""
import json, math, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import process_outputs as po  # noqa: E402
import supplement  # noqa: E402
from build_full import load_world, GROUP, ROWS_PER_PAIR, sig_str  # noqa: E402

LETTER = {"EQUIVALENCE": "E", "FORWARD_ENTAILMENT": "U", "NEGATIVE_OTHER": "O"}
ROUND_STAGES = {"2": ("pilot", "main", "r2"), "4": ("pilot", "main", "r2", "r3", "r4")}


def main(off, pilot_pkg, merged, label, prev, dest, pkgs):
    k = label[1]; assert label in ("s2a", "s2b", "s4a", "s4b")
    world = load_world(off, pilot_pkg)
    rep = po.load(f"{merged}/report.json")
    assert max(rep["stats"]["tiers"]) == int(k), "合并目录的最高规模和补充标签不符"
    deficit = {}
    for key, v in rep["stats"]["fold_quota_short"].items():
        f, g = key.split("-", 1); deficit[(int(f), LETTER[g])] = v
    disp, denom, returned, issued_supp = Counter(), 0, Counter(), 0
    for pkg in pkgs:
        for x in sorted(os.listdir(f"{pkg}/batches")):
            b = po.load(f"{pkg}/batches/{x}")
            for p in b["parents"]:
                disp[p["family_id"]] += 1; returned[GROUP[p["label"]]] += 1
                if b["stage"] in ROUND_STAGES[k]: denom += ROWS_PER_PAIR[p["label"]]
                elif b["stage"].startswith("supp") or b["stage"][0] == "s": issued_supp += ROWS_PER_PAIR[p["label"]]
    kept = Counter(GROUP[r["label"]] for r in po.read_csv(f"{merged}/manifest.csv") if r["direction"] == "original")
    keep = {g: (kept[g] / returned[g] if returned[g] else 0) for g in "EUO"}
    kr = {g: math.floor(v * 20) / 20 for g, v in keep.items()}
    sig_families = defaultdict(lambda: defaultdict(list))
    for f, p in world["eligible"].items(): sig_families[sig_str(world["signature"][f])][GROUP[p["label"]]].append(f)
    sig_families = {s: dict(v) for s, v in sig_families.items()}
    if prev:
        ledger = po.load(prev)["ledger"]
        assert ledger["rows_issued"] == issued_supp, "台账和已派发的补充行数不符"
        ledger["uses"] = dict(disp)
    else:
        ledger = supplement.new_ledger(dict(disp)); ledger["rows_issued"] = issued_supp
    batches, status, led = supplement.plan(deficit, sig_families, kr, denom, ledger, None, max_uses=2 * int(k), label=label)
    plan = {"round": label, "stage": label, "status": status, "batches": batches, "ledger": led, "budget_denominator_rows": denom,
            "budget_cap_rows": supplement.BUDGET * denom, "max_uses": 2 * int(k),
            "keep_rate_observed": keep, "keep_rate_used": kr,
            "deficit_rows": {f"{a}-{g}": v for (a, g), v in sorted(deficit.items())},
            "merged_report_sha256": po.sha_file(f"{merged}/report.json")}
    po.dump(dest, plan)
    print(json.dumps({x: v for x, v in plan.items() if x not in ("batches", "ledger")} | {"batches": len(batches),
                      "rows_issued_total": led["rows_issued"]}, ensure_ascii=False, indent=1))
    return plan


if __name__ == "__main__":
    a = sys.argv
    main(os.path.abspath(a[1]), os.path.abspath(a[2]), os.path.abspath(a[3]), a[4], a[5] if a[5] != "-" else None,
         os.path.abspath(a[6]), [os.path.abspath(x) for x in a[7:]])
