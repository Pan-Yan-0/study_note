#!/usr/bin/env python3
"""SST-2 情感二分类：微调 DeBERTa-v3-base，作为指令学习实验的对照组。

训练数据分别只用 16 / 64 / 256 / 1024 条（正负各半，从训练集随机抽）和全部 67,349 条，
每种用 3 个随机种子，都在同一份 872 条验证集上算准确率。验证集只用来报告结果，
不用来挑轮数或超参数，最终数字就是训练结束时那个模型的成绩；中途几次验证只记录曲线。

超参数固定，不在验证集上调：
  少样本（16–1024 条）：学习率 2e-5，batch 8，训练 1000 步
  全量：学习率 2e-5，batch 32，3 轮
  共同：AdamW，权重衰减 0.01（偏置和 LayerNorm 不衰减），前 10% 的步数线性 warmup，
        之后线性降到 0；梯度裁剪 1.0；最长 128 个词片；fp16 混合精度

用法：
  python deberta_sst2.py --val-file sst2_validation.tsv --out-dir deberta
  python deberta_sst2.py --val-file ... --out-dir ... --smoke     # 16 条、10 步、验证前 64 条
  python deberta_sst2.py --val-file ... --out-dir ... --dry-run   # 只读数据、打印计划，不加载模型

输出（--out-dir 下）：
  plan.json                    全部训练组的设置
  runs/<组名>/summary.json     一组的设置、状态、最终准确率、验证曲线、用时、抽到的训练样本 idx
  runs/<组名>/predictions.csv  872 条逐条预测：idx、真实标签、预测、正面概率
  runs.csv、summary.md         所有组一行一条，以及按训练条数汇总的均值 ± 标准差
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

MODEL = "microsoft/deberta-v3-base"
MIRROR = "https://hf-mirror.com"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def write_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


# ======================= 1. 数据 =======================
def read_tsv(path: str | Path) -> list[dict]:
    """读 sentence / label / idx 三列的 tsv，标签是 0（负面）/ 1（正面）。读法与评测脚本一致。"""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not {"sentence", "label"}.issubset(reader.fieldnames or []):
            raise SystemExit(f"{path} 必须包含 sentence 和 label 两列")
        for i, r in enumerate(reader):
            label = int(r["label"])
            if label not in (0, 1):
                raise SystemExit(f"{path} 第 {i + 2} 行的标签应为 0/1，读到 {label}")
            idx = int(r["idx"]) if (r.get("idx") or "").strip() else i
            rows.append({"idx": idx, "sentence": r["sentence"].strip(), "label": label})
    return rows


def rows_sha256(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r['idx']}\t{r['sentence']}\t{r['label']}\n".encode("utf-8"))
    return h.hexdigest()


def load_hf_split(split: str) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "sst2", split=split)
    return [{"idx": int(r["idx"]), "sentence": r["sentence"].strip(), "label": int(r["label"])}
            for r in ds]


def load_train(args) -> tuple[list[dict], dict]:
    if args.train_file:
        rows = read_tsv(args.train_file)
        return rows, {"source": f"file:{args.train_file}"}
    try:
        rows = load_hf_split("train")
    except Exception as e:  # 网络问题：换镜像整个重启一次（HF_ENDPOINT 要在导入前设好）
        if not os.environ.get("HF_ENDPOINT"):
            log(f"从 Hugging Face 下载训练集失败（{type(e).__name__}: {e}），改用 {MIRROR} 重试")
            os.environ["HF_ENDPOINT"] = MIRROR
            os.execv(sys.executable, [sys.executable] + sys.argv)
        raise
    info = {"source": "nyu-mll/glue sst2 train", "hf_endpoint": os.environ.get("HF_ENDPOINT", "default")}
    return rows, info


def check_validation(val: list[dict], args) -> dict:
    """确认内置的 872 条和 Hugging Face 上的验证集一致（只记录，不影响训练）。"""
    if args.train_file:
        return {"checked": False}
    try:
        hf = load_hf_split("validation")
    except Exception as e:
        return {"checked": False, "error": f"{type(e).__name__}: {e}"[:300]}
    same = [(r["sentence"], r["label"]) for r in val] == [(r["sentence"], r["label"]) for r in hf]
    return {"checked": True, "matches_hf_validation": same, "hf_rows": len(hf)}


def sample_balanced(train: list[dict], k: int, seed: int) -> list[dict]:
    """从训练集里抽 k 条，正负各半；同一个种子总是抽到同样的样本。"""
    rng = random.Random(seed)
    neg = [r for r in train if r["label"] == 0]
    pos = [r for r in train if r["label"] == 1]
    picked = rng.sample(neg, k // 2) + rng.sample(pos, k - k // 2)
    rng.shuffle(picked)
    return picked


# ======================= 2. 训练计划 =======================
def build_plan(args, n_train: int) -> list[dict]:
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    plan = []
    for size in sizes:
        for seed in seeds:
            if size == "full":
                n, batch, lr = n_train, args.full_batch, args.full_lr
                per_epoch = math.ceil(n / batch)
                steps = math.ceil(per_epoch * args.full_epochs)
                eval_at = sorted({min(per_epoch * e, steps)
                                  for e in range(1, math.ceil(args.full_epochs) + 1)} | {steps})
                schedule = f"{args.full_epochs:g} 轮"
            else:
                n = int(size)
                if n < 2 or n > n_train:
                    raise SystemExit(f"训练条数 {n} 不合理")
                batch, lr, steps = min(args.few_batch, n), args.few_lr, args.few_steps
                points = max(1, args.eval_points)
                eval_at = sorted({max(1, round(steps * j / points)) for j in range(1, points + 1)} | {steps})
                schedule = f"{steps} 步"
            plan.append({"name": f"n{size}_seed{seed}", "size": size, "n_train": n, "seed": seed,
                         "learning_rate": lr, "batch_size": batch, "steps": steps,
                         "warmup_steps": int(round(args.warmup * steps)), "schedule": schedule,
                         "eval_at": eval_at, "weight_decay": args.weight_decay,
                         "max_length": args.max_length, "model": args.model})
    return plan


def lr_factor(step: int, warmup: int, total: int) -> float:
    """线性 warmup 到 1，再线性降到 0。step 从 0 开始计（LambdaLR 的约定）。"""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


# ======================= 3. 训练一组 =======================
def collate(seqs: list[list[int]], pad_id: int, device):
    import torch
    width = max(len(s) for s in seqs)
    ids = [s + [pad_id] * (width - len(s)) for s in seqs]
    mask = [[1] * len(s) + [0] * (width - len(s)) for s in seqs]
    return (torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(mask, dtype=torch.long, device=device))


def evaluate(model, enc, labels, pad_id, device, use_fp16, batch_size):
    import torch
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(enc), batch_size):
            ids, mask = collate(enc[i:i + batch_size], pad_id, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    preds = [int(p >= 0.5) for p in probs]
    acc = sum(int(p == y) for p, y in zip(preds, labels, strict=True)) / len(labels)
    return acc, probs, preds


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(spec: dict, data: list[dict], val: list[dict], val_enc, tokenizer, args,
              deadline: float, use_fp16: bool) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification

    device = torch.device("cuda")
    set_seed(spec["seed"])
    texts = [r["sentence"] for r in data]
    enc = tokenizer(texts, truncation=True, max_length=spec["max_length"])["input_ids"]
    labels = [r["label"] for r in data]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    model = AutoModelForSequenceClassification.from_pretrained(spec["model"], num_labels=2)
    model = model.float().to(device)  # 参数保持 fp32，前向用 autocast 算 fp16
    commit = getattr(model.config, "_commit_hash", None)
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": spec["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=spec["learning_rate"])
    total, warmup = spec["steps"], spec["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, warmup, total))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    torch.cuda.reset_peak_memory_stats()

    rng = random.Random(spec["seed"] + 1)
    n, batch = len(data), spec["batch_size"]

    def batches():
        while True:
            order = list(range(n))
            rng.shuffle(order)
            for i in range(0, n, batch):
                yield order[i:i + batch]

    stream = batches()
    eval_at = set(spec["eval_at"])
    report_every = max(50, total // 10)
    curve, nonfinite, loss_window = [], 0, []
    status, t0 = "completed", time.time()
    model.train()
    step = 0
    for step in range(1, total + 1):
        chosen = next(stream)
        ids, mask = collate([enc[i] for i in chosen], pad_id, device)
        y = torch.tensor([labels[i] for i in chosen], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            logits = model(input_ids=ids, attention_mask=mask).logits
        loss = F.cross_entropy(logits.float(), y)
        value = loss.item()
        if not math.isfinite(value):
            nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            loss_window.append(value)
        scheduler.step()
        if step % report_every == 0 or step == total:
            mean_loss = statistics.fmean(loss_window) if loss_window else float("nan")
            loss_window = []
            elapsed = time.time() - t0
            log(f"    {spec['name']} 第 {step}/{total} 步  loss {mean_loss:.4f}  "
                f"学习率 {scheduler.get_last_lr()[0]:.2e}  已用 {elapsed / 60:.1f} 分钟，"
                f"预计还要 {elapsed / step * (total - step) / 60:.1f} 分钟")
        if step in eval_at:
            acc, _, _ = evaluate(model, val_enc, [r["label"] for r in val], pad_id, device,
                                 use_fp16, args.eval_batch)
            curve.append({"step": step, "accuracy": acc})
            log(f"    {spec['name']} 第 {step} 步验证准确率 {acc:.4f}（只记录，不用于挑模型）")
            model.train()
        if deadline and time.time() > deadline and step < total:
            status = "stopped_deadline"
            log(f"    {spec['name']} 到达截止时间，在第 {step} 步停止")
            break

    train_seconds = time.time() - t0
    acc, probs, preds = evaluate(model, val_enc, [r["label"] for r in val], pad_id, device,
                                 use_fp16, args.eval_batch)
    if not curve or curve[-1]["step"] != step:
        curve.append({"step": step, "accuracy": acc})
    result = {"status": status, "accuracy": acc, "correct": sum(int(p == r["label"]) for p, r in zip(preds, val, strict=True)),
              "n_eval": len(val), "steps_done": step, "train_seconds": round(train_seconds, 1),
              "steps_per_second": round(step / train_seconds, 3) if train_seconds else None,
              "nonfinite_losses": nonfinite, "fp16": use_fp16, "curve": curve,
              "model_commit": commit,
              "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
              "predictions": [{"idx": r["idx"], "label": r["label"], "pred": p, "prob_positive": round(q, 6)}
                              for r, p, q in zip(val, preds, probs, strict=True)]}
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_spec(spec: dict, train: list[dict], val: list[dict], val_enc, tokenizer, args,
             out_dir: Path, deadline: float, env: dict) -> dict:
    run_dir = out_dir / "runs" / spec["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    keys = ("size", "n_train", "seed", "learning_rate", "batch_size", "steps", "warmup_steps",
            "weight_decay", "max_length", "model")
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if old.get("status") == "completed" and all(old.get(k) == spec[k] for k in keys):
            log(f"[{spec['name']}] 已完成，跳过（准确率 {old['accuracy']:.4f}）")
            return old
    if deadline and time.time() > deadline - 60:
        record = {**spec, "status": "skipped_deadline"}
        write_json(summary_path, record)
        log(f"[{spec['name']}] 已到截止时间，跳过")
        return record

    data = train if spec["size"] == "full" else sample_balanced(train, spec["n_train"], spec["seed"])
    log(f"[{spec['name']}] 训练 {len(data)} 条（正面 {sum(r['label'] for r in data)}），"
        f"{spec['schedule']}，共 {spec['steps']} 步，batch {spec['batch_size']}，学习率 {spec['learning_rate']}")
    record = {**spec, **env, "train_label_counts": {"0": sum(r["label"] == 0 for r in data),
                                                    "1": sum(r["label"] == 1 for r in data)},
              "train_idx": None if spec["size"] == "full" else [r["idx"] for r in data],
              "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        result = train_one(spec, data, val, val_enc, tokenizer, args, deadline, use_fp16=not args.fp32)
        if result["fp16"] and result["nonfinite_losses"] > 0.05 * max(1, result["steps_done"]):
            log(f"[{spec['name']}] fp16 下有 {result['nonfinite_losses']} 步 loss 非有限，改用 fp32 重训一次")
            result = train_one(spec, data, val, val_enc, tokenizer, args, deadline, use_fp16=False)
            result["fp16_unstable_rerun_fp32"] = True
    except Exception as e:
        record.update(status="failed", error=f"{type(e).__name__}: {e}"[:2000],
                      traceback=traceback.format_exc()[-6000:])
        write_json(summary_path, record)
        log(f"[{spec['name']}] 失败：{record['error']}")
        try:
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        return record
    predictions = result.pop("predictions")
    with open(run_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "label", "pred", "prob_positive"])
        writer.writeheader()
        writer.writerows(predictions)
    record.update(result, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(summary_path, record)
    log(f"[{spec['name']}] {record['status']}：准确率 {record['accuracy']:.4f}"
        f"（{record['correct']}/{record['n_eval']}），训练 {record['train_seconds'] / 60:.1f} 分钟")
    return record


# ======================= 4. 汇总 =======================
def aggregate(out_dir: Path, plan: list[dict]) -> None:
    rows = []
    for spec in plan:
        path = out_dir / "runs" / spec["name"] / "summary.json"
        rec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {**spec, "status": "not_run"}
        rows.append({"name": spec["name"], "size": spec["size"], "n_train": spec["n_train"],
                     "seed": spec["seed"], "status": rec.get("status"),
                     "accuracy": rec.get("accuracy"), "correct": rec.get("correct"),
                     "steps_done": rec.get("steps_done"), "steps": spec["steps"],
                     "train_minutes": round(rec["train_seconds"] / 60, 2) if rec.get("train_seconds") else None,
                     "fp16": rec.get("fp16"), "nonfinite_losses": rec.get("nonfinite_losses"),
                     "error": (rec.get("error") or "")[:200]})
    with open(out_dir / "runs.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| 训练条数 | 完成的种子 | 准确率（均值 ± 标准差） | 最低 | 最高 | 各种子 |",
             "|---|---:|---:|---:|---:|---|"]
    for size in dict.fromkeys(r["size"] for r in rows):
        group = [r for r in rows if r["size"] == size]
        done = [r for r in group if r["status"] == "completed"]
        accs = [r["accuracy"] for r in done]
        label = f"全量 {group[0]['n_train']:,}" if size == "full" else size
        if accs:
            sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
            per_seed = "、".join(f"{r['seed']}: {r['accuracy']:.4f}" for r in done)
            lines.append(f"| {label} | {len(done)}/{len(group)} | {statistics.fmean(accs):.4f} ± {sd:.4f} | "
                         f"{min(accs):.4f} | {max(accs):.4f} | {per_seed} |")
        else:
            states = "、".join(sorted({str(r['status']) for r in group}))
            lines.append(f"| {label} | 0/{len(group)} | — | — | — | {states} |")
    note = ("\n准确率在同一份 872 条验证集上计算；标准差为样本标准差。"
            "验证集只用于报告，不用于挑选模型。\n")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")
    print("\n".join(lines), flush=True)


# ======================= 5. 命令行 =======================
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-file", required=True, help="872 条验证集 tsv（sentence、label 0/1、idx）")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-file", help="本地训练集 tsv；不给就从 Hugging Face 下载 nyu-mll/glue sst2")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--sizes", default="16,64,256,1024,full", help="训练条数，full 表示全部训练集")
    p.add_argument("--seeds", default="13,21,42")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--few-lr", type=float, default=2e-5)
    p.add_argument("--few-batch", type=int, default=8)
    p.add_argument("--few-steps", type=int, default=1000)
    p.add_argument("--full-lr", type=float, default=2e-5)
    p.add_argument("--full-batch", type=int, default=32)
    p.add_argument("--full-epochs", type=float, default=3.0)
    p.add_argument("--warmup", type=float, default=0.1, help="warmup 占总步数的比例")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--eval-points", type=int, default=5, help="少样本训练中途记录几次验证准确率")
    p.add_argument("--fp32", action="store_true", help="不用 fp16 混合精度")
    p.add_argument("--deadline-unix", type=float, default=0.0, help="到这个时间点就不再开始新的组")
    p.add_argument("--smoke", action="store_true", help="16 条、10 步、验证前 64 条，只检查流程")
    p.add_argument("--dry-run", action="store_true", help="只读数据、打印计划，不加载模型")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val = read_tsv(args.val_file)
    if args.smoke:
        val = val[:64]
        args.sizes, args.seeds, args.few_steps, args.eval_points = "16", args.seeds.split(",")[0], 10, 1
    train, train_info = load_train(args)
    plan = build_plan(args, len(train))
    data_info = {"train": {**train_info, "rows": len(train), "sha256": rows_sha256(train),
                           "label_counts": {"0": sum(r["label"] == 0 for r in train),
                                            "1": sum(r["label"] == 1 for r in train)}},
                 "validation": {"file": str(args.val_file), "rows": len(val), "sha256": rows_sha256(val),
                                **(check_validation(val, args) if not args.smoke else {})}}
    write_json(out_dir / "plan.json", {"data": data_info, "runs": plan, "args": vars(args)})
    log(f"训练集 {len(train):,} 条（{train_info['source']}），验证集 {len(val)} 条；共 {len(plan)} 组")
    for spec in plan:
        log(f"  {spec['name']}: {spec['n_train']} 条，{spec['schedule']}（{spec['steps']} 步），"
            f"batch {spec['batch_size']}，学习率 {spec['learning_rate']}")
    if args.dry_run:
        return

    import torch
    import transformers
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("没有可用的 CUDA GPU")
    env = {"torch": torch.__version__, "transformers": transformers.__version__,
           "gpu": torch.cuda.get_device_name(0),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    log(f"GPU：{env['gpu']}；torch {env['torch']}；transformers {env['transformers']}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    val_enc = tokenizer([r["sentence"] for r in val], truncation=True, max_length=args.max_length)["input_ids"]
    log(f"验证集最长 {max(len(e) for e in val_enc)} 个词片")

    t0 = time.time()
    for i, spec in enumerate(plan, 1):
        log(f"========== [{i}/{len(plan)}] {spec['name']} ==========")
        run_spec(spec, train, val, val_enc, tokenizer, args, out_dir, args.deadline_unix, env)
    aggregate(out_dir, plan)
    log(f"全部结束，用时 {(time.time() - t0) / 60:.1f} 分钟")
    if args.smoke:
        rec = json.loads((out_dir / "runs" / plan[0]["name"] / "summary.json").read_text(encoding="utf-8"))
        if rec.get("status") != "completed":
            raise SystemExit(f"冒烟测试失败：{rec.get('error')}")
        print(f"SMOKE OK accuracy={rec['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
