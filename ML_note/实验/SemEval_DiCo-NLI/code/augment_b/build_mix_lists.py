"""类别构成（线 2）事先定的主要比较 M1 的抽样列表：U4、BAL、E1、D1、O1 五个点，每点一个 Dup 组和一个 G 组。

  python build_mix_lists.py <官方仓库> <split_manifest.json> <输出.json> [<规模合并目录（含 pool_order.json）>]

- 配额：mix_quota.py（mixquota-v1，10-28 和 GPT 定的），每折新增行数 = fit 行数，各组相同。
- Dup：mix_quota.dup_list（每类固定顺序循环取整对，记重复次数）。
- G：规模合并的 pool_order.json 里每折每组的固定顺序取前缀（E 取 E 对、D 取单向对、O 取 O 对），
  和 +1 / +2 / +4 的选用是同一个顺序，所以 G_R 等于 B1-Haiku。只给了 Dup 时（Commit 1）不写 G。
- 核对：mix_quota 复现的划分和训练用的 split_manifest 每折 fit / val / test 逐项相同。
"""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mix_quota as mq

POINTS = ("U4", "BAL", "E1", "D1", "O1")
GROUP_OF = {"E": "EQUIVALENCE", "D": "FORWARD_ENTAILMENT", "O": "NEGATIVE_OTHER"}


def main(off, split_path, dest, scale_dir=None):
    rows, fam_of, fold_of, inner, part, _ = mq.load_split(off)
    split = json.load(open(split_path, encoding="utf-8"))["folds"]
    for k in range(5):
        for s in ("fit", "val", "test"):
            mine = sorted(r["instance_id"] for r in rows if part[(k, r["instance_id"])] == s)
            assert mine == sorted(split[str(k)][s]), f"fold {k} {s} 和 split_manifest 不同"
    order = json.load(open(f"{scale_dir}/pool_order.json", encoding="utf-8"))["order"] if scale_dir else None
    out = {"quota_version": mq.QUOTA_VERSION, "order_seed": mq.ORDER_SEED, "points": list(POINTS),
           "split_manifest_sha256": hashlib.sha256(open(split_path, "rb").read()).hexdigest(),
           "pool_order_sha256": hashlib.sha256(open(f"{scale_dir}/pool_order.json", "rb").read()).hexdigest() if scale_dir else None,
           "arms": {}}
    for k in range(5):
        pairs = mq.original_pairs(rows, fam_of, part, k)
        fr = mq.fit_counts(pairs)
        n_add, q = mq.fold_quotas(fr)
        for p in POINTS:
            dup = mq.dup_list(pairs, q[p], k)
            assert sum(n for _, n in dup) == n_add
            a = out["arms"].setdefault(f"Dup-{p}", {"kind": "dup", "point": p, "folds": {}})
            a["folds"][str(k)] = {"n_add": n_add, "rows_by_label": q[p]["rows_by_label"], "pairs": q[p]["pairs"], "dup": dup}
            if order:
                g = out["arms"].setdefault(f"G-{p}", {"kind": "gen", "point": p, "folds": {}})
                take = []
                for c, grp in GROUP_OF.items():
                    lst = order[f"{k}-{grp}"]; need = q[p]["pairs"][c]
                    assert len(lst) >= need, f"fold {k} {c} 的生成池不够 {p}：{len(lst)} < {need}"
                    take += lst[:need]
                g["folds"][str(k)] = {"n_add": n_add, "rows_by_label": q[p]["rows_by_label"], "pairs": q[p]["pairs"], "pair_uids": take}
    json.dump(out, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=0, sort_keys=True)
    print({a: {k: v["rows_by_label"] for k, v in x["folds"].items() if k == "0"} for a, x in out["arms"].items()})


if __name__ == "__main__":
    main(*map(os.path.abspath, sys.argv[1:4]), os.path.abspath(sys.argv[4]) if len(sys.argv) > 4 else None)
