"""A1 预检、SMOKE、双卡队列、续跑和打包。"""
import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from a1_data import ARMS, B2, GROUPS, MODEL_REV, VERSION, base, preflight, require, target_counts, task_config
from train_a1 import (SMOKE_LEGACY_BATCHES, SMOKE_MAX_BATCHES, b2_reuse, numeric_ok, numeric_smoke,
                      one_task, read_completed, smoke_done)

WORKERS = 2
RECEIPT = "a1_smoke.json"
RESULT_NAME = "a1_train_v1_results.zip"


def runtime_environment():
    import torch

    require(torch.cuda.device_count() == 1, "每个 worker 只绑定一张 T4。请用入口自动分配 CUDA_VISIBLE_DEVICES")
    info = base.envinfo()
    require(info["gpu"] == "Tesla T4" and info["precision"] == "fp16", f"A1 固定 T4 fp16：{info}")
    info["runner"] = "Kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "Linux"
    return info


def run_identity(inputs, info):
    return {"version": VERSION, "bundle_sha256": inputs.bundle_hash, "code": inputs.code_hashes,
            "model_revision": MODEL_REV, "stable_environment": base.stable_env_identity(info),
            "software_environment": base.soft_env(info)}


def prepare_run(inputs, out, info):
    identity = run_identity(inputs, info)
    path = out / "run_identity.json"
    if path.exists():
        require(base.jload(path) == identity, "旧运行身份已改变，请在新目录运行")
    else:
        base.jsave(path, identity)
    history_path = out / "environment_history.json"
    history = base.jload(history_path) if history_path.exists() else []
    record = {"environment": info, "model_revision": MODEL_REV, "code": inputs.code_hashes}
    if record not in history:
        base.jsave(history_path, [*history, record])
    base.jsave(out / "model_revision.json", {"model": base.MODEL_ID, "revision": MODEL_REV})
    for name, expected in inputs.data_hashes.items():
        src, dest = inputs.bundle / "data" / name, out / "input_data" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            require(base.digest(dest) == expected, f"结果目录的数据已改变：{name}")
        else:
            temp = dest.with_suffix(dest.suffix + ".tmp")
            shutil.copyfile(src, temp)
            temp.replace(dest)
    return identity


def bindings(out):
    names = ("smoke.json", "done.json")
    return {g: {name: base.digest(out / "smoke" / g / name) for name in names} for g in GROUPS}


def gate(inputs, out, info):
    receipt = base.jload(out / RECEIPT)
    identity = run_identity(inputs, info)
    require(base.jload(out / "run_identity.json") == receipt.get("run_identity") == identity
            and receipt.get("status") == "pass", "SMOKE 的包、代码、模型或环境身份已改变")
    require(receipt.get("fold_teacher_sha256") == {str(f): h for f, h in inputs.fold_teacher_hashes.items()},
            "SMOKE 教师目标身份不一致")
    for name, expected in inputs.data_hashes.items():
        require(base.digest(out / "input_data" / name) == expected, f"结果目录的数据已改变：{name}")
    require(receipt.get("bindings") == bindings(out), "SMOKE 回执内容已改变")
    for group in GROUPS:
        smoke_done(inputs, out, info, group)
    numeric = receipt.get("one_hot_equivalence")
    require(numeric_ok(numeric) and numeric.get("actual_updates_each") == 2, "CE / one-hot 数值回执不合格")
    reuse = b2_reuse(inputs, out, info)
    require(receipt.get("b2") == reuse, "B2 复用回执和当前核对结果不一致")
    return reuse["reuse"]


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base.MODEL_ID, revision=MODEL_REV)


def smoke(inputs, out, info):
    identity = prepare_run(inputs, out, info)
    (out / RECEIPT).unlink(missing_ok=True)
    shutil.rmtree(out / "smoke", ignore_errors=True)
    checks = numeric_smoke("cuda")
    tok = tokenizer()
    for group in GROUPS:
        batches = SMOKE_LEGACY_BATCHES if group == B2 else SMOKE_MAX_BATCHES
        ok = one_task(inputs, out, group, 42, 0, tok, info, smoke_batches=batches)
        if group == B2 and not ok:
            # 两步复现失败时，先验证 B2 本次的训练和输出路径，再走 60 项回退。
            dest = out / "smoke" / B2 / "failure.json"
            dest.rename(out / "smoke" / "B2_legacy_attempt.json")
            ok = one_task(inputs, out, B2, 42, 0, tok, info, smoke_batches=SMOKE_MAX_BATCHES)
        require(ok, f"{group} SMOKE 没有完成 2 次实际更新，详见 failure.json")
    reuse = b2_reuse(inputs, out, info)
    base.jsave(out / RECEIPT, {"status": "pass", "run_identity": identity, "bindings": bindings(out),
                              "fold_teacher_sha256": inputs.fold_teacher_hashes,
                              "one_hot_equivalence": checks, "b2": reuse})
    passed = gate(inputs, out, info)
    print("[SMOKE PASS]", "B2 复用" if passed else "B2 同队列重跑", "正式任务", len(queue(passed)),
          reuse["differences"], flush=True)


def queue(reuse):
    groups = ARMS if reuse else GROUPS
    return [(g, s, f) for g in groups for s in base.SEEDS for f in base.FOLDS]


def verify_tasks(inputs, out, info, jobs, complete=False):
    valid_paths = {base.task_path(out, *job): job for job in jobs}
    for dest in sorted((out / "tasks").glob("*/seed*/fold*")):
        if dest.is_dir():
            require(dest in valid_paths, f"旧任务属于不同队列：{dest}，请使用新结果目录")
    done = 0
    for dest, job in valid_paths.items():
        cfg = task_config(inputs, *job, info)
        if (dest / "done.json").exists():
            record = read_completed(dest, cfg, info)
            require(record.get("target_counts") == target_counts(inputs, job[0], job[2]),
                    f"任务目标计数不一致：{dest}")
            done += 1
        elif (dest / "task.json").exists():
            require(base.jload(dest / "task.json") == cfg, f"旧任务身份改变：{dest}")
    if complete:
        require(done == len(jobs), f"正式任务未齐：{done}/{len(jobs)}")
    return done


def worker(inputs, out, info, index):
    reuse = gate(inputs, out, info)
    jobs = queue(reuse)
    tok = tokenizer()
    print(f"[WORKER {index}] {len(jobs[index::WORKERS])} tasks", flush=True)
    for job in jobs[index::WORKERS]:
        one_task(inputs, out, *job, tok, info)


def pack(out, result, completed_only=False):
    result = result.resolve()
    require(not result.is_relative_to(out), "结果 ZIP 必须放在运行目录之外")
    result.parent.mkdir(parents=True, exist_ok=True)
    tmp = result.with_suffix(".zip.tmp")
    completed = {p.parent for p in (out / "tasks").glob("*/seed*/fold*/done.json")} if completed_only else set()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path.name.endswith(".lock") or path.suffix in {".pt", ".tmp"}:
                continue
            rel = path.relative_to(out)
            if completed_only and rel.parts[0] == "tasks" and path.parent not in completed:
                continue
            archive.write(path, Path("a1_run") / rel)
    tmp.replace(result)
    print(f"[结果包] {result}", flush=True)


def restore(source, out, validate):
    """只恢复经过身份预检的任务。旧 SMOKE 与正式报告不会复制进新会话。"""
    if source is None:
        return
    if (out / "tasks").exists():
        print("[RESUME] 使用当前会话已有任务", flush=True)
        return
    with TemporaryDirectory(prefix="a1-restore-", dir=out) as tmp:
        tmp = Path(tmp)
        if source.is_dir():
            root = source
        else:
            require(source.is_file(), f"找不到旧结果：{source}")
            with zipfile.ZipFile(source) as archive:
                members = []
                seen = set()
                for entry in archive.infolist():
                    path = Path(entry.filename)
                    require(not path.is_absolute() and ".." not in path.parts and entry.filename not in seen,
                            f"旧包路径非法：{path}")
                    seen.add(entry.filename)
                    if path.parts[:2] == ("a1_run", "tasks") or path == Path("a1_run/run_identity.json"):
                        members.append(entry)
                archive.extractall(tmp, members)
            root = tmp / "a1_run"
        require(base.jload(root / "run_identity.json") == base.jload(out / "run_identity.json"), "旧结果包的运行身份不同")
        tasks = root / "tasks"
        if tasks.is_dir():
            target = tmp / "tasks"
            shutil.copytree(tasks, target, ignore=shutil.ignore_patterns("*.pt", "*.tmp"))
            validate(tmp)
            target.rename(out / "tasks")


def full(inputs, out, info, resume, result):
    from score_a1 import score

    reuse = gate(inputs, out, info)
    jobs = queue(reuse)
    restore(resume, out, lambda tmp: verify_tasks(inputs, tmp, info, jobs))
    count = verify_tasks(inputs, out, info, jobs)
    print(f"[QUEUE] {count}/{len(jobs)} 已完成，B2 复用={reuse}", flush=True)
    (out / "summary.json").unlink(missing_ok=True)
    if count != len(jobs):
        launch_workers(inputs, out, result, len(jobs))
    verify_tasks(inputs, out, info, jobs, complete=True)
    score(inputs, out, info, reuse)


def launch_workers(inputs, out, result, total):
    root = out / "logs"
    root.mkdir(exist_ok=True)
    procs = []
    try:
        for index in range(WORKERS):
            cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--bundle", str(inputs.bundle),
                   "--out", str(out), "--worker", str(index)]
            with (root / f"worker{index}.log").open("a", encoding="utf-8") as log:
                procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(index)}))
        last_pack = time.monotonic()
        while True:
            codes = [p.poll() for p in procs]
            require(all(c is None or c == 0 for c in codes), f"worker 退出码 {codes}，见 logs/worker*.log")
            if None not in codes:
                return
            if time.monotonic() - last_pack >= 300:
                count = len(list((out / "tasks").glob("*/seed*/fold*/done.json")))
                print(f"[PROGRESS] {count}/{total}", flush=True)
                pack(out, result, completed_only=True)
                last_pack = time.monotonic()
            time.sleep(5)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "smoke", "full", "worker", "score"))
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("a1_run"))
    parser.add_argument("--result", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--worker", type=int, choices=range(WORKERS))
    args = parser.parse_args()
    inputs = preflight(args.bundle)
    if args.stage == "preflight":
        print("[PREFLIGHT PASS]", {f: len(rows) for f, rows in inputs.train.items()},
              "teacher folds", {f: h[:12] for f, h in inputs.fold_teacher_hashes.items()})
        return
    out = args.out.resolve()
    require(not out.is_relative_to(inputs.bundle), "运行目录须在训练包之外")
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "worker":
        require(args.worker is not None, "worker 需要明确编号")
        with (out / f".worker{args.worker}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(inputs, out, runtime_environment(), args.worker)
        return
    result = args.result or (out.parent / RESULT_NAME)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failure = out / "failure.json"
        failure.unlink(missing_ok=True)
        try:
            if args.stage == "score":
                from score_a1 import score

                info = base.jload(out / "environment_history.json")[-1]["environment"]
                reuse = gate(inputs, out, info)
                verify_tasks(inputs, out, info, queue(reuse), complete=True)
                score(inputs, out, info, reuse)
            else:
                import torch

                if args.stage == "full":
                    require(torch.cuda.device_count() == WORKERS, "正式队列需要 Kaggle GPU T4 x2")
                # 主进程只读环境；训练只在绑定 GPU 的子进程进行。
                if args.stage == "full":
                    info = base.envinfo()
                    require(all(torch.cuda.get_device_name(i) == "Tesla T4" for i in range(WORKERS)), "需要两张 Tesla T4")
                    full(inputs, out, info, args.resume, result)
                else:
                    smoke(inputs, out, runtime_environment())
        except Exception as error:
            base.jsave(failure, {"stage": args.stage, "error": str(error)})
            raise
        finally:
            pack(out, result)


if __name__ == "__main__":
    main()
