"""A1 OOF 复核。主比较沿用 Commit 5 的家族配对 bootstrap，预先固定 m=3。"""
import random
from collections import Counter

from a1_data import ARMS, B2, GROUPS, base, reference_task, require, task_config
from train_a1 import read_completed

REPLICATES = 20000
BOOTSTRAP_SEED = 20261021
M = 3
ALPHA = 0.05
LETTERS = "EFBO"


def probability(row, prefix):
    p = [float(row[prefix + c]) for c in LETTERS]
    require(all(base._fin(x, 0, 1) for x in p) and abs(sum(p) - 1) <= 2e-9,
            f"预测概率不合法：{row['instance_id']} {prefix}")
    return p


def valid_argmax(label, p, rounding=0):
    return label in base.L2I and p[base.L2I[label]] >= max(p) - rounding


def verify_rows(inputs, dest, part, expected_ids):
    val = part == "val"
    rows = base.csvrows(dest / ("val_predictions.csv" if val else "predictions.csv"))
    require([r["instance_id"] for r in rows] == expected_ids, f"{dest} {part} 行或顺序不一致")
    for row in rows:
        iid = row["instance_id"]
        gold = inputs.original[iid]
        require(all(row[k] == gold[k] for k in ("label", "pair_id")), f"{dest} 金标或 pair_id 不一致：{iid}")
        p = probability(row, "p_")
        if val:
            logits = [float(row["logit_" + c]) for c in LETTERS]
            require(all(base._fin(x) for x in logits) and valid_argmax(row["pred"], logits),
                    f"{dest} val argmax 不一致：{iid}")
        else:
            require(row["reverse_pair_id"] == gold["reverse_pair_id"]
                    and row["family_id"] == inputs.family_of_pair[gold["pair_id"]], f"{dest} 家族或反向 ID 不一致：{iid}")
            q = probability(row, "r_p_")
            sym = [(p[k] + q[j]) / 2 for k, j in enumerate(base.SWAP)]
            require(max(abs(a - b) for a, b in zip(sym, probability(row, "sym_p_"))) <= 2e-9
                    and valid_argmax(row["plain"], p) and valid_argmax(row["sym"], sym, rounding=2e-10), f"{dest} sym 概率或 argmax 不一致：{iid}")
    return rows


def collect_task(inputs, out, info, group, seed, fold, reuse):
    archived = group == B2 and reuse
    root = inputs.bundle / "baseline" if archived else out
    dest = base.task_path(root, group, seed, fold)
    cfg = (reference_task(inputs, group, seed, fold, info, False) if archived
           else task_config(inputs, group, seed, fold, info))
    done = read_completed(dest, cfg, info)
    split = inputs.splits[str(fold)]
    val = verify_rows(inputs, dest, "val", split["val"])
    score = base.wf1([base.L2I[r["label"]] for r in val], [base.L2I[r["pred"]] for r in val])
    require(abs(score - done["best_val_original_wf1"]) <= 1e-12, f"{dest} 内层验证 WF1 不一致")
    test = verify_rows(inputs, dest, "test", split["test"])
    updates = {key: done[key] for key in ("best_epoch", "planned_updates", "actual_optimizer_updates",
                                           "skipped_fp16_updates", "runtime_sec", "environment")}
    return test, {"group": group, "seed": seed, "fold": fold, "reused": archived, **updates}


def descriptive(rows, mode):
    from sklearn.metrics import average_precision_score

    gold = [r["label"] for r in rows]
    preds = [r[mode] for r in rows]
    prefix = "sym_p_" if mode == "sym" else "p_"
    p = [probability(r, prefix) for r in rows]
    conf = Counter(zip(gold, preds))
    other = base.LABELS[3]
    num_o = sum(y == other for y in gold)
    pred_o = sum(y == other for y in preds)
    tp = conf[other, other]
    errors = sum(g != y for g, y in zip(gold, preds))
    high = [i for i, values in enumerate(p) if max(values) >= 0.99]
    high_errors = sum(gold[i] != preds[i] for i in high)
    return {"O_precision": tp / pred_o if pred_o else 0.0, "O_recall": tp / num_o,
            "O_AP": float(average_precision_score([y == other for y in gold], [q[3] for q in p])),
            "O_predicted": pred_o, "O_missed": num_o - tp, "O_false_alarm": pred_o - tp,
            "O_direction_confusions": {f"{a}->{b}": conf[base.LABELS[LETTERS.index(a)], base.LABELS[LETTERS.index(b)]]
                                       for a, b in ("OF", "OB", "FO", "BO")},
            "confidence_ge_0.99": {"rows": len(high), "errors": high_errors,
                                   "error_rate": high_errors / len(high) if high else None,
                                   "share_of_all_errors": high_errors / errors if errors else None}}


def bootstrap(preds, gold, family_indices, n_families, n_boot=REPLICATES):
    """每次对家族抽一次，对所有组和三个种子共用相同频数。"""
    import numpy as np

    groups = list(preds)
    counts = np.zeros((n_families, len(groups), len(base.SEEDS), 4, 4))
    for g, name in enumerate(groups):
        for s, p in enumerate(preds[name]):
            np.add.at(counts, (family_indices, g, s, gold, p), 1)
    flat = counts.reshape(n_families, -1)
    rng = random.Random(BOOTSTRAP_SEED)
    res = np.empty((n_boot, len(groups), len(base.SEEDS)))
    family_pool = range(n_families)
    for start in range(0, n_boot, 500):
        n = min(500, n_boot - start)
        weights = np.stack([np.bincount(rng.choices(family_pool, k=n_families), minlength=n_families)
                            for _ in range(n)])
        cm = (weights.astype(float) @ flat).reshape(n, len(groups), len(base.SEEDS), 4, 4)
        support, predicted = cm.sum(-1), cm.sum(-2)
        tp = np.diagonal(cm, axis1=-2, axis2=-1)
        denom = support + predicted
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom != 0)
        res[start:start + n] = (f1 * support).sum(-1) / support.sum(-1)
    return {g: res[:, i, :] for i, g in enumerate(groups)}


def difference(boot, points, a, b, m, primary):
    import numpy as np

    paired = np.array(points[a]) - np.array(points[b])
    deltas = np.sort((boot[a] - boot[b]).mean(axis=1))
    idx = int(len(deltas) * ALPHA / (2 * m))
    ci = [float(deltas[idx]), float(deltas[len(deltas) - 1 - idx])]
    mean = float(paired.mean())
    report = {"per_seed": dict(zip(map(str, base.SEEDS), map(float, paired))),
              "mean": mean, "level": 1 - ALPHA / m, "interval": ci}
    if primary:
        report["improvement"] = mean > 0 and ci[0] > 0
    return report


def score(inputs, out, info, reuse):
    import numpy as np

    ids = sorted(inputs.original)
    family = [inputs.family_of_pair[inputs.original[i]["pair_id"]] for i in ids]
    families = sorted(set(family))
    require(len(ids) == 3042 and len(families) == 1725, "OOF 原始行或家族数改变")
    index = {f: i for i, f in enumerate(families)}
    fam_indices = np.array([index[f] for f in family])
    gold = np.array([base.L2I[inputs.original[i]["label"]] for i in ids])
    preds = {g: [] for g in GROUPS}
    points = {g: [] for g in GROUPS}
    diagnostics, tasks = {}, []
    official = inputs.bundle / "official"
    for group in GROUPS:
        diagnostics[group] = {}
        for seed in base.SEEDS:
            rows = []
            for fold in base.FOLDS:
                records, rec = collect_task(inputs, out, info, group, seed, fold, reuse)
                rows.extend(records)
                tasks.append(rec)
            rows.sort(key=lambda r: r["instance_id"])
            require([r["instance_id"] for r in rows] == ids, f"{group}/{seed} OOF 行重复或缺失")
            dest = out / "oof" / group / f"seed{seed}"
            base.csvsave(dest / "predictions.csv", rows, list(rows[0]))
            scores = {}
            for mode in ("plain", "sym"):
                result = base.official_score(official, rows, mode, dest, mode)
                scores[mode] = {"official": result, **descriptive(rows, mode)}
            pred = np.array([base.L2I[r["sym"]] for r in rows])
            point = base.wf1(gold, pred)
            require(abs(point - scores["sym"]["official"]["weighted_f1"]) <= 1e-12, "自算与官方 WF1 不一致")
            preds[group].append(pred)
            points[group].append(point)
            diagnostics[group][str(seed)] = scores
    boot = bootstrap(preds, gold, fam_indices, len(families))
    comparisons = {g + "_minus_B2": difference(boot, points, g, B2, M, True) for g in ARMS}
    exploratory = {"B2-SL_minus_" + g: difference(boot, points, "B2-SL", g, 1, False) for g in ("B2-RL", "B2-LS")}
    report = {"status": "complete", "B2_reused": reuse, "n_rows": len(ids), "n_families": len(families),
              "bootstrap_replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
              "bonferroni_m": M, "interval_conditions_on_seeds": list(base.SEEDS), "primary_sym_wf1": comparisons,
              "exploratory_sym_wf1": exploratory, "scores": diagnostics, "tasks": tasks}
    base.jsave(out / "summary.json", report)
    for name, row in comparisons.items():
        print(name, f"{100 * row['mean']:+.2f} pp", [round(100 * x, 2) for x in row["interval"]],
              "通过" if row["improvement"] else "未过线", flush=True)
    return report
