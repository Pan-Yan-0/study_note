"""从已归档的 A0 / B2 / 官方 588968e 组装 Kaggle 数据集，不打包模型权重。"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from a1_data import (A0_STATS_SHA, A0_SUMMARY_SHA, AUDIT_RESULT_SHA, BASE_SHA, B2, HERE, OFFICIAL_REV, SCRIPT_NAMES,
                     TEACHER_SHA, base, file_hash, preflight, reference_task, require)
from train_a1 import OUTPUTS, read_completed

SE = HERE.parent.parent
AB = SE / "metrics/augment_b"
A0 = SE / "metrics/a0_teacher"
Q0 = SE / "metrics/qwen4b_augb_s1/formal/Q0"
C5 = AB / "train_c5"
SCALE = AB / "haiku_scale/merged"
FULL = AB / "haiku_full/merged"
BUNDLE_NAME = "A1_Kaggle_Train_v1"


def copy(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def official_files(official, bundle):
    revision = subprocess.check_output(["git", "-C", official, "rev-parse", "HEAD"], text=True).strip()
    require(revision == OFFICIAL_REV, f"需要 DiCo-NLI 官方提交 {OFFICIAL_REV}，当前 {revision}")
    names = ["LICENSE", *[f"final_data/train/dico_nli_train_track1_{k}.csv"
                          for k in ("reference", "participant_labeled")]]
    scorer = subprocess.check_output(
        ["git", "-C", official, "ls-tree", "-r", "--name-only", OFFICIAL_REV, "evaluation_functions"], text=True)
    names += [name for name in scorer.splitlines() if name.endswith(".py")]
    for name in names:
        clean = subprocess.check_output(["git", "-C", official, "show", f"{OFFICIAL_REV}:{name}"])
        dest = bundle / "official" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(clean)


def write_signatures(bundle):
    data = bundle / "data"
    splits = base.jload(data / "split_manifest.json")
    original = base.csvrows(bundle / "official/final_data/train/dico_nli_train_track1_participant_labeled.csv")
    families = base.build_families(original)
    signature = {f: [] for f in sorted(set(families.values()))}
    for fold, split in splits["folds"].items():
        fit = set(split["fit"])
        for family in {families[r["pair_id"]] for r in original if r["instance_id"] in fit}:
            signature[family].append(int(fold))
    canonical = json.dumps(splits, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload = {"split_manifest_sha256": hashlib.sha256(canonical).hexdigest(), "signature_of_family": signature}
    path = data / "family_signatures.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    file_hash(path, base.EXPECTED_FAMILY_SIGNATURES_SHA)


def verify_a0_receipts(bundle):
    """核对 A0 正式回执、Q0 归档和每折输入顺序，再运行原 stats 重建。"""
    sys.path.insert(0, str(SE / "code/a0_teacher"))
    import teacher_a0 as a0

    full = A0 / "full"
    summary_path = full / "A0_SUMMARY.json"
    file_hash(summary_path, A0_SUMMARY_SHA)
    summary = base.jload(summary_path)
    identity = summary["run_identity"]
    require(summary["version"] == a0.VERSION and summary["problems"] == []
            and identity["gate"] == a0.GATE and identity["env"]["gpu"] == "Tesla T4"
            and identity["env"]["dtype"] == "torch.float16", "A0 运行回执不符")
    man = {r["sample_id"]: r for r in base.csvrows(SCALE / "manifest.csv")}
    selection = base.csvrows(SCALE / "fold_selection_scale.csv")
    split = base.jload(bundle / "data/split_manifest.json")["folds"]
    byid = {r["instance_id"]: r for r in base.csvrows(
        bundle / "official/final_data/train/dico_nli_train_track1_participant_labeled.csv")}
    ordered, counts = {}, {}
    for fold in base.FOLDS:
        rows = [man[r["sample_id"]] for r in selection if int(r["fold"]) == fold and int(r["tier"]) in (1, 2)]
        pairs = list(dict.fromkeys(pair for r in rows for pair in (
            (r["text1"], r["text2"]), (r["text2"], r["text1"]))))
        ordered[str(fold)], counts[str(fold)] = pairs, len(rows)
        for seed in base.SEEDS:
            q0 = Q0 / f"seed{seed}" / f"fold{fold}"
            complete = base.jload(q0 / "COMPLETE.json")
            files = complete["files"]
            for name in ("trained.json", "actual_split.json"):
                file_hash(q0 / name, files[name])
            if seed == 42:
                file_hash(q0 / "cache.jsonl", files["cache.jsonl"])
            orig_split = base.jload(q0 / "actual_split.json")
            require(all(sorted(orig_split[k]) == sorted(split[str(fold)][k]) for k in ("fit", "val", "test")),
                    f"Q0 fold{fold} seed{seed} 划分与 B2 不同")
            best = base.jload(q0 / "trained.json")["best_epoch"]
            adapters = {n: files[f"adapters/epoch{best}/{n}"] for n in ("adapter_model.safetensors", "adapter_config.json")}
            path, rec = a0.task_paths(full, fold, seed, False)
            receipt = base.jload(rec)
            want = {"status": "pass", "fold": fold, "seed": seed, "smoke": False, "best_epoch": best,
                    "adapter_sha256": adapters, "inputs": len(pairs), "run_identity": identity}
            val = receipt["val_recheck"]
            n_val = len({(byid[i]["text1"], byid[i]["text2"]) for i in orig_split["val"]})
            require(all(receipt.get(k) == v for k, v in want.items()) and val["pass"] is True
                    and val["rows"] == val["returned"] == n_val
                    and base._fin(val["argmax_agree"], a0.GATE["val_argmax_agree"], 1),
                    f"A0 fold{fold} seed{seed} 回执不匹配")
            if seed == 42:
                reproduction = receipt["reproduction"]
                with (q0 / "cache.jsonl").open(encoding="utf-8") as cache:
                    count = sum(1 for line in cache if line.strip())
                require(reproduction["pass"] is True and reproduction["inputs"] == count and a0.repro_ok(reproduction),
                        f"A0 fold{fold} 逐折复现未通过")
            file_hash(path, receipt["output_sha256"])
            with path.open(encoding="utf-8") as stream:
                predictions = a0.check_results([json.loads(line) for line in stream], len(pairs), str(path))
            require([(r["a"], r["b"]) for r in predictions] == a0.sort_inputs(pairs),
                    f"A0 fold{fold} seed{seed} 输入顺序不符")
    # 与 A0 build_bundle.py 原序列化格式相同。
    sources = {n: base.digest(SCALE / n) for n in ("manifest.csv", "fold_selection_scale.csv", "gate.json")}
    inputs = {"inputs": ordered, "rows_per_fold": counts,
              "rule": "fold_selection_scale.csv 的 tier 1、2（即 B2），两个方向，首次出现的顺序去重",
              "sources": sources}
    raw = json.dumps(inputs, ensure_ascii=False).encode()
    require(hashlib.sha256(raw).hexdigest() == identity["data"]["b2_inputs.json"], "A0 B2 输入身份不一致")


def build(bundle, official):
    for name in SCRIPT_NAMES:
        copy(Path(base.__file__) if name == "train_augb_v6.py" else HERE / name, bundle / name)
    for name in ("README.md", "A1_Kaggle_v1.ipynb", "test_a1.py"):
        copy(HERE / name, bundle / name)
    file_hash(bundle / "train_augb_v6.py", BASE_SHA)
    official_files(official, bundle)
    for name in ("manifest.csv", "fold_selection.csv", "gate.json", "sources.json", "report.json"):
        copy(FULL / name, bundle / "data" / name)
    copy(C5 / "input_data/split_manifest.json", bundle / "data/split_manifest.json")
    write_signatures(bundle)
    copy(AB / "train_c2/input_data/mix_lists.json", bundle / "data_v3/mix_lists.json")
    for name in ("gate.json", *base.jload(SCALE / "gate.json")["bound_to"]):
        copy(SCALE / name, bundle / "data_v3/scale" / name)
    verify_a0_receipts(bundle)
    with TemporaryDirectory(prefix="a1-targets-") as tmp:
        subprocess.run([sys.executable, "-I", SE / "code/a0_analysis.py", "stats", A0 / "full", tmp], check=True)
        targets = Path(tmp) / "teacher_targets.jsonl"
        file_hash(targets, TEACHER_SHA)
        file_hash(Path(tmp) / "a0_stats.json", A0_STATS_SHA)
        copy(targets, bundle / "teacher/teacher_targets.jsonl")
        copy(Path(tmp) / "a0_stats.json", bundle / "teacher/a0_stats.json")
    file_hash(A0 / "audit/audit_result.json", AUDIT_RESULT_SHA)
    copy(A0 / "audit/audit_result.json", bundle / "teacher/audit_result.json")
    copy(A0 / "full/A0_SUMMARY.json", bundle / "teacher/A0_SUMMARY.json")

    archived = C5 / "smoke" / B2
    for name in ("task.json", "done.json", "smoke.json", *OUTPUTS):
        copy(archived / name, bundle / "baseline/smoke" / name)
    for seed in base.SEEDS:
        for fold in base.FOLDS:
            for name in ("task.json", "done.json", *OUTPUTS):
                copy(base.task_path(C5, B2, seed, fold) / name,
                     base.task_path(bundle / "baseline", B2, seed, fold) / name)
    files = sorted(p for p in bundle.rglob("*") if p.is_file())
    (bundle / "SHA256SUMS").write_text("".join(f"{base.digest(p)}  {p.relative_to(bundle).as_posix()}\n" for p in files),
                                      encoding="utf-8")
    inputs = preflight(bundle)
    info = base.jload(archived / "done.json")["environment"]
    read_completed(bundle / "baseline/smoke", reference_task(inputs, B2, 42, 0, info, True), info)
    for seed in base.SEEDS:
        for fold in base.FOLDS:
            read_completed(base.task_path(bundle / "baseline", B2, seed, fold),
                           reference_task(inputs, B2, seed, fold, info, False), info)
    print(f"打包验证通过：{len(files)} 个文件，5 折 A0 目标，B2 SMOKE 和 15 个正式任务")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official", type=Path, help="已 checkout 到 588968e 的官方仓库")
    parser.add_argument("output", type=Path, help="新建的输出 ZIP")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.suffix == ".zip" and not output.exists(), "输出必须是新的 .zip 文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="a1-bundle-", dir=output.parent) as tmp:
        bundle = Path(tmp) / BUNDLE_NAME
        build(bundle, args.official.resolve())
        staged = Path(tmp) / output.name
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    info = zipfile.ZipInfo(path.relative_to(bundle.parent).as_posix())
                    archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        staged.rename(output)
    print(output, base.digest(output))


if __name__ == "__main__":
    main()
