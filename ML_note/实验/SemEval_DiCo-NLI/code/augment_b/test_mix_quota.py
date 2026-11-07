"""mix_quota 的自检：python3 -I test_mix_quota.py <官方仓库目录> [<split_manifest.json>]"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mix_quota as m

repo = sys.argv[1]
rows, fam_of, fold_of, inner, part, split_hash = m.load_split(repo)
man = m.split_manifest(rows, part)
if len(sys.argv) > 2:  # 和 8B 验收归档的划分逐折逐部分比对
    ref = json.load(open(sys.argv[2], encoding="utf-8"))
    for k in range(5):
        for s in ("fit", "val", "test"):
            assert man[str(k)][s] == ref[str(k)][s], (k, s)
    print("划分与 qwen8b_acceptance/split_manifest.json 逐项一致")
pairs0 = m.original_pairs(rows, fam_of, part, 0)
fr0 = m.fit_counts(pairs0)
assert fr0 == {"E": 574, "D": 1262, "O": 354}, fr0
assert man["0"]["fit"]["rows"] == 2190 and man["0"]["val"]["rows"] == 244 and man["0"]["test"]["rows"] == 608
checked = 0
for k in range(5):
    pairs = m.original_pairs(rows, fam_of, part, k)
    fr = m.fit_counts(pairs)
    n_add, q = m.fold_quotas(fr)
    assert n_add == sum(fr.values()) - sum(fr.values()) % 2
    assert q["R"]["rows"] == fr, (k, q["R"]["rows"], fr)       # 本折拟合比例 → R 正好等于拟合部分的计数
    assert q["E3"]["rows"] == {"E": n_add, "D": 0, "O": 0} and q["O3"]["rows"] == {"E": 0, "D": 0, "O": n_add}
    for p, v in q.items():
        r = v["rows"]
        assert sum(r.values()) == n_add and r["E"] % 2 == 0 and r["D"] % 2 == 0, (k, p)
        assert v["rows_by_label"]["F"] == v["rows_by_label"]["B"]
        t = v["target_rows"]
        assert all(abs(x - y) <= 2 for x, y in zip((r["E"], r["D"], r["O"]), t)), (k, p, r, t)  # 整数化误差 ≤ 2 行
        dup = m.dup_list(pairs, v, k)
        assert sum(n for _, n in dup) == n_add
        assert dup == m.dup_list(pairs, v, k)                  # 确定性
        ids = {i for _, mem in pairs["E"] + pairs["D"] + pairs["O"] for i in mem}
        assert all(i in ids for i, _ in dup)                    # 只用本折拟合部分的行
        checked += 1
# 并列规则与边界
assert m.integerize((1.0, 1.0, 0.0), 2) == (0, 2, 0) or m.integerize((1.0, 1.0, 0.0), 2) == (2, 0, 0)
# 池缺口
need = {0: {"E": 1, "D": 1, "O": 2}}
pool = [{"pair_uid": "a", "label": "NEGATIVE_OTHER", "usable_folds": [0]},
        {"pair_uid": "b", "label": "EQUIVALENCE", "usable_folds": [0]}, {"pair_uid": "b", "label": "EQUIVALENCE", "usable_folds": [0]}]
assert m.pool_gap(need, pool) == {0: {"E": 0, "D": 1, "O": 1}}
lam, tot, per = m.plan_supplement_uniform({0: 10, 1: 5}, {(0, 1): 5, (1,): 5})
assert abs(per[0] - 10) < 1e-9 and per[1] >= 5
plan = m.plan_supplement({0: 5, 1: 5}, {(0, 1): 3, (0,): 1}, 2)
assert plan == {(0, 1): 3}, plan
print(f"全部通过：{checked} 个（折 × 点）的配额和重复列表；划分哈希 {split_hash[:12]}")
