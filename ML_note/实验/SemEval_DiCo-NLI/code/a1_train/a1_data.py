"""冻结数据、A0 目标和 A1 的训练身份。CPU 预检不加载模型。"""
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
# 源码在仓库中运行时，上游在同级目录。Kaggle 包里直接随本模块放置。
sys.path.insert(0, str(HERE if (HERE / "train_augb_v6.py").is_file() else HERE.parent / "augment_b_train"))
import train_augb_v6 as base

VERSION = "a1-train-v1"
BASE_SHA = "50c800620fbc9d5f53dcad5dffba3c8821b83aee28f7d567554dd887488e60ad"
TEACHER_SHA = "bd17653c3717a0ec36d41b3d89a19cec8182cb0a019c1112a1509221a8b981b0"
A0_SUMMARY_SHA = "29e1285e16471aedba1e0f5923cba80ae1d8f415283427421f3ad1b44386013a"
A0_STATS_SHA = "96178a5c9d2e87b7d8d1a921fbc599d96bbff8a62e9fea6b5fac1c267ebb4a27"
AUDIT_RESULT_SHA = "eebe58a68868938e61b8ec21e2989970751b83514cc27a962cf6ad83801674e7"
OFFICIAL_REV = "588968e610197ddc4c440314701cbc587afc4c1b"
MODEL_REV = "a0484667b22365f84929a935b5e50a51f71f159d"
B2 = "B2-Haiku"
ARMS = ("B2-RL", "B2-SL", "B2-LS")
GROUPS = (B2, *ARMS)
EPS = 0.1
POLICY = {B2: "generated_hard", "B2-RL": "teacher_argmax",
          "B2-SL": "teacher_sym_soft_ce", "B2-LS": "generated_smoothing_0.1"}
SOFT_ARMS = {"B2-SL", "B2-LS"}
SCRIPT_NAMES = ("a1_data.py", "train_a1.py", "run_a1.py", "score_a1.py", "train_augb_v6.py")
OFFICIAL_FILES = {
    "reference": "c8f4d8587711e0beaefc115606e5c29c3e78dc6660e074967428af0fed069ee4",
    "participant_labeled": "5e8dd9f3a7e000933fec3c9e64d064f9a25586f141f58d6e6afb47c4fc797211",
}


@dataclass
class Inputs:
    bundle: Path
    original: dict
    family_of_pair: dict
    splits: dict
    train: dict
    teacher: dict
    data_hashes: dict
    scale_hashes: dict
    order_hashes: dict
    fold_teacher_hashes: dict
    code_hashes: dict
    bundle_hash: str


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def file_hash(path, expected):
    require(base.digest(path) == expected, f"SHA-256 不一致：{path}")


def verify_bundle(bundle):
    """校验运行包及当前执行的代码，避免混用旧 notebook 目录或同名脚本。"""
    manifest = bundle / "SHA256SUMS"
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and name not in listed and name != manifest.name,
                f"清单路径不合法：{name}")
        file_hash(bundle / path, expected)
        listed.add(name)
    require(set(SCRIPT_NAMES) <= listed, "清单缺少 A1 执行代码")
    paths = {name: HERE / name for name in SCRIPT_NAMES}
    paths["train_augb_v6.py"] = Path(base.__file__)
    code_hashes = {name: base.digest(path) for name, path in paths.items()}
    require(code_hashes["train_augb_v6.py"] == BASE_SHA, "Commit 5 上游代码身份不一致")
    for name, expected in code_hashes.items():
        file_hash(bundle / name, expected)
    return code_hashes, base.digest(manifest)


def valid_distribution(p):
    return (isinstance(p, list) and len(p) == 4
            and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in p)
            and abs(sum(p) - 1.0) <= 1e-7)


def load_targets(path, train):
    """按 fold 和有向 sample_id 连接 A0。核对文本、原标签、种子均值及反向概率。"""
    file_hash(path, TEACHER_SHA)
    expected = {f: {r["sample_id"]: r for r in train[f] if r["source_group"] != "original"}
                for f in base.FOLDS}
    found = {f: {} for f in base.FOLDS}
    changes = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            t = json.loads(line)
            fold, sid = t["fold"], t["sample_id"]
            require(type(fold) is int and fold in found and sid in expected[fold] and sid not in found[fold],
                    f"教师行缺少对应样本、折号错误或重复：{fold}/{sid}")
            row = expected[fold][sid]
            require((t["pair_uid"], t["text1"], t["text2"], t["gen"])
                    == (row["pair_uid"], row["text1"], row["text2"], row["label"]),
                    f"教师行与 B2 有向生成行不一致：{fold}/{sid}")
            p, seeds = t["p"], t["p_seeds"]
            require(valid_distribution(p) and len(seeds) == 3 and all(valid_distribution(x) for x in seeds),
                    f"教师概率不合法：{fold}/{sid}")
            mean = [sum(x[k] for x in seeds) / 3 for k in range(4)]
            require(mean == p and t["teacher"] == base.LABELS[max(range(4), key=p.__getitem__)],
                    f"教师种子平均或 argmax 不一致：{fold}/{sid}")
            found[fold][sid] = t
            changes += t["teacher"] != t["gen"]
    for fold, items in found.items():
        require(items.keys() == expected[fold].keys(), f"fold {fold} 教师目标覆盖不全")
        pairs = {}
        for sid, target in items.items():
            pair = expected[fold][sid]["pair_uid"]
            reverse = pairs.get(pair)
            if reverse is not None:
                require((target["text1"], target["text2"]) == (reverse["text2"], reverse["text1"])
                        and target["p"] == [reverse["p"][k] for k in base.SWAP]
                        and all(p == [q[k] for k in base.SWAP] for p, q in zip(target["p_seeds"], reverse["p_seeds"])),
                        f"fold {fold} 正反向教师概率没有按 E/F/B/O 映射：{pair}")
            pairs[pair] = target
    require(sum(map(len, found.values())) == 21936 and changes == 1794, "教师目标数或硬改动数与 A0 封存不一致")
    hashes = {f: base.canonical_json_hash(items) for f, items in found.items()}
    return found, hashes


def preflight(bundle):
    bundle = Path(bundle).resolve()
    code_hashes, bundle_hash = verify_bundle(bundle)
    official = bundle / "official"
    for kind, sha in OFFICIAL_FILES.items():
        file_hash(official / f"final_data/train/dico_nli_train_track1_{kind}.csv", sha)
    _, selected, splits, _ = base.preflight_generated(bundle, None)
    with TemporaryDirectory(prefix="a1-preflight-") as scratch:
        original, families, _ = base.preflight_official(bundle, official, selected, splits, scratch)
    base.load_v3_data(bundle, "5", splits)
    train = {f: base.make_rows(B2, f, original, selected, splits) for f in base.FOLDS}
    order_hashes = {}
    for f, rows in train.items():
        n_fit = len(splits["folds"][str(f)]["fit"])
        ids = [r["row_id"] for r in rows]
        require(len(ids) == len(set(ids)) == 3 * n_fit and sum(r["source_group"] == "original" for r in rows) == n_fit,
                f"fold {f} B2 顺序或行数不完整")
        order_hashes[f] = base.canonical_json_hash(ids)

    file_hash(bundle / "teacher/audit_result.json", AUDIT_RESULT_SHA)
    file_hash(bundle / "teacher/A0_SUMMARY.json", A0_SUMMARY_SHA)
    file_hash(bundle / "teacher/a0_stats.json", A0_STATS_SHA)
    audit = base.jload(bundle / "teacher/audit_result.json")
    require(audit["RL"]["pass"] is True and audit["SL"]["pass"] is True, "冻结的 A1 入队记录不完整")
    teacher, fold_teacher_hashes = load_targets(bundle / "teacher/teacher_targets.jsonl", train)
    return Inputs(bundle, original, families, splits["folds"], train, teacher,
                  {n: base.digest(bundle / "data" / n) for n in base.INPUT_NAMES},
                  dict(base.V3["hashes"]), order_hashes, fold_teacher_hashes, code_hashes, bundle_hash)


def reference_task(inputs, group, seed, fold, info, smoke):
    """Commit 5 的同配方字段。B2 归档用原代码身份逐字段比较。"""
    n_train = len(inputs.train[fold])
    split = inputs.splits[str(fold)]
    epochs = 1 if smoke else base.CONFIG["max_epochs"]
    return {"group": group, "fold": fold, "seed": seed, "model": base.MODEL_ID, "revision": MODEL_REV,
            "train_rows": n_train, "val_rows": len(split["val"]), "test_rows": len(split["test"]),
            "steps": math.ceil(n_train / base.CONFIG["batch_size"]) * epochs, "epochs": epochs, "smoke": smoke,
            "config": base.CONFIG, "data_sha256": inputs.data_hashes, "v3_data_sha256": inputs.scale_hashes,
            "official_data_sha256": hashlib.sha256(OFFICIAL_FILES["reference"].encode("ascii")).hexdigest(),
            "code_sha256": BASE_SHA, "stable_environment": base.stable_env_identity(info),
            "seed_policy": base.CONFIG["seed_note"], "group_spec": base.GROUP_SPEC[B2], "class_weights": None, "loss": "ce"}


def task_config(inputs, group, seed, fold, info, smoke_batches=0):
    cfg = reference_task(inputs, group, seed, fold, info, bool(smoke_batches))
    loss = "soft_ce_fp32" if group in SOFT_ARMS else "ce"
    return {**cfg, "code_sha256": inputs.code_hashes["train_a1.py"], "code": inputs.code_hashes,
            "config": {**base.CONFIG, "code_version": VERSION}, "loss": loss,
            "target_policy": POLICY[group], "ordered_training_rows_sha256": inputs.order_hashes[fold],
            "teacher_fold_sha256": inputs.fold_teacher_hashes[fold] if group in ("B2-RL", "B2-SL") else None,
            "smoke_batches": smoke_batches}


def training_targets(inputs, group, fold):
    """与 B2 顺序一致。软目标只改变生成行，原始行仍为金标 one-hot。"""
    labels, soft = [], []
    for row in inputs.train[fold]:
        original = row["source_group"] == "original"
        label = row["label"]
        if group == "B2-RL" and not original:
            label = inputs.teacher[fold][row["sample_id"]]["teacher"]
        index = base.L2I[label]
        labels.append(index)
        if group in SOFT_ARMS:
            if original:
                target = [float(k == index) for k in range(4)]
            elif group == "B2-SL":
                target = inputs.teacher[fold][row["sample_id"]]["p"]
            else:
                target = [(1 - EPS) * (k == index) + EPS / 4 for k in range(4)]
            soft.append(target)
    return labels, soft


def target_counts(inputs, group, fold):
    labels, soft = training_targets(inputs, group, fold)
    n_orig = len(inputs.splits[str(fold)]["fit"])
    if soft:
        require(all(x == [float(k == y) for k in range(4)] for y, x in zip(labels[:n_orig], soft[:n_orig])),
                f"{group} fold {fold} 原始行必须保留 one-hot 金标")
    require(labels[:n_orig] == [base.L2I[r["label"]] for r in inputs.train[fold][:n_orig]],
            f"{group} fold {fold} 原始行标签改变")
    return {"original_one_hot": n_orig, "generated": len(labels) - n_orig,
            "soft_generated": len(soft) - n_orig if soft else 0,
            "hard_label_counts": dict(Counter(base.LABELS[k] for k in labels))}
