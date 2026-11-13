"""A1 共用训练循环。B2 的 batch、AMP、优化器和评估沿用 Commit 5。"""
import math
import time
from types import SimpleNamespace
from pathlib import Path

from a1_data import B2, MODEL_REV, SOFT_ARMS, base, require, target_counts, task_config, training_targets

OUTPUTS = ("predictions.csv", "val_predictions.csv", "extra_logits.csv")
SMOKE_LEGACY_BATCHES = 2
SMOKE_MAX_BATCHES = 6


class SoftCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, samples):
        import torch

        targets = torch.tensor([sample.pop("soft_targets") for sample in samples], dtype=torch.float32)
        return {**self.collator(samples), "soft_targets": targets}


def training_dataset(inputs, group, fold, tokenizer):
    labels, targets = training_targets(inputs, group, fold)
    rows = inputs.train[fold]
    if group == "B2-RL":
        rows = [{**row, "label": base.LABELS[label]} for row, label in zip(rows, labels)]
    encoded, collator = base.datasets_for(rows, tokenizer)
    if group not in SOFT_ARMS:
        return encoded, collator

    class SoftDataset:
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            return {**encoded[index], "soft_targets": targets[index]}

    return SoftDataset(), SoftCollator(collator)


def loss_for(net, batch, labels, targets, device):
    import torch
    import torch.nn.functional as F

    with torch.autocast(device, dtype=torch.float16):
        logits = net(**batch).logits
        if targets is None:
            return F.cross_entropy(logits.float(), labels)
    # 在 autocast 外对全部行做同一次 mean。原始行的目标是 one-hot。
    return -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def numeric_smoke(device):
    """同一组 logits 上核对 CE/one-hot soft CE，再用原优化器路径各更新两次。"""
    import torch
    import torch.nn.functional as F
    from transformers import get_linear_schedule_with_warmup

    values = torch.tensor([[1.0, -1.0, 0.25], [-0.5, 0.75, 1.0],
                           [0.25, -1.0, 0.5], [1.5, 0.5, -0.5]], device=device)
    labels = torch.arange(4, device=device)
    one_hot = F.one_hot(labels, 4).float()

    class TinyHead(torch.nn.Linear):
        def forward(self, x):
            return SimpleNamespace(logits=super().forward(x))

    # 单独保存和恢复 RNG；实际 SMOKE 每个任务也使用固定 init_seed。
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
        net = TinyHead(3, 4).to(device)
        with torch.no_grad():
            net.weight.copy_(torch.arange(12, device=device).reshape(4, 3) / 40)
            net.bias.zero_()
        soft_net = TinyHead(3, 4).to(device)
        soft_net.load_state_dict(net.state_dict())
    optimizer = [torch.optim.AdamW(x.parameters(), lr=base.CONFIG["lr"]) for x in (net, soft_net)]
    schedule = [get_linear_schedule_with_warmup(x, 0, 2) for x in optimizer]
    scaler = [torch.amp.GradScaler(device, init_scale=128) for _ in optimizer]
    deltas = {"loss_max_abs": 0.0, "grad_norm_max_abs": 0.0, "parameter_max_abs": 0.0}
    for _ in range(2):
        losses, norms = [], []
        for i, (model, target) in enumerate(((net, None), (soft_net, one_hot))):
            optimizer[i].zero_grad(set_to_none=True)
            loss = loss_for(model, {"x": values}, labels, target, device)
            status, norm = base.apply_update(loss, optimizer[i], schedule[i], scaler[i], list(model.parameters()))
            require(status == "updated" and math.isfinite(norm), "CE / one-hot SMOKE 未实际更新")
            losses.append(float(loss.detach()))
            norms.append(norm)
        require(schedule[0].get_last_lr() == schedule[1].get_last_lr(), "CE / one-hot 调度器步进不同")
        deltas["loss_max_abs"] = max(deltas["loss_max_abs"], abs(losses[0] - losses[1]))
        deltas["grad_norm_max_abs"] = max(deltas["grad_norm_max_abs"], abs(norms[0] - norms[1]))
        deltas["parameter_max_abs"] = max(deltas["parameter_max_abs"], max(
            float((a - b).detach().abs().max()) for a, b in zip(net.parameters(), soft_net.parameters())))
    require(numeric_ok(deltas), f"CE / one-hot 数值路径不一致：{deltas}")
    return {**deltas, "actual_updates_each": 2}


def numeric_ok(report):
    return (isinstance(report, dict)
            and all(base._fin(report.get(k), 0, tol) for k, tol in (
                ("loss_max_abs", 1e-6), ("grad_norm_max_abs", 1e-6), ("parameter_max_abs", 1e-7))))


def output_hashes(dest):
    return {name.removesuffix(".csv") + "_sha256": base.digest(dest / name) for name in OUTPUTS}


def read_completed(dest, expected, info):
    task = base.jload(dest / "task.json")
    done = base.jload(dest / "done.json")
    require(task == expected and done.get("task_fingerprint") == base.canonical_json_hash(task),
            f"任务配置或指纹不匹配：{dest}")
    identity = ("group", "seed", "fold", "train_rows", "val_rows", "test_rows", "class_weights", "seed_policy")
    require(done.get("status") == "pass" and all(done.get(k) == task[k] for k in identity)
            and done.get("repeat_counts") == {} and not done.get("repeated_original_row_ids")
            and done.get("source_count") == {"original": task["train_rows"] // 3,
                                             base.CONFIG["data_source"]: task["train_rows"] * 2 // 3},
            f"任务完成记录不匹配：{dest}")
    require(base.stable_env_identity(done["environment"]) == task["stable_environment"]
            and base.soft_env(done["environment"]) == base.soft_env(info)
            and done.get("inference_precision") == "fp32_no_autocast",
            f"任务环境或评估精度不匹配：{dest}")
    require(all(done.get(k) == v for k, v in output_hashes(dest).items()), f"任务输出损坏：{dest}")
    planned = 2 if task["smoke"] else task["steps"]
    epochs = done["epochs"]
    updates, skips = done["actual_optimizer_updates"], done["skipped_fp16_updates"]
    scores = [e["val_original_wf1"] for e in epochs]
    require([e["epoch"] for e in epochs] == list(range(1, task["epochs"] + 1))
            and all(base._fin(s, 0, 1) for s in scores), f"验证轮数或分数非法：{dest}")
    best = 0
    for ep, score in enumerate(scores):
        if score > scores[best] + 1e-12:
            best = ep
    require(done["planned_updates"] == planned and done["lr_schedule"] == base.learning_rate_trace(planned)
            and type(updates) is int and type(skips) is int and updates > 0 and skips >= 0
            and (updates == 2 if task["smoke"] else updates + skips == planned)
            and epochs[-1]["updates_total"] == updates and epochs[-1]["skips_total"] == skips
            and done["best_epoch"] == best + 1 and done["best_val_original_wf1"] == scores[best],
            f"更新计数或验证选轮不一致：{dest}")
    return done


def training_indices(rows, batch_size, smoke_batches):
    ranked = sorted(range(len(rows)),
                    key=lambda i: (len(rows[i]["text1"].split()) + len(rows[i]["text2"].split()), i), reverse=True)
    chosen = [next(i for i in ranked if rows[i]["label"] == label) for label in base.LABELS]
    chosen.extend(i for i in ranked if i not in chosen)
    return chosen[:smoke_batches * batch_size]


def one_task(inputs, out, group, seed, fold, tokenizer, info, smoke_batches=0):
    import torch
    from torch.utils.data import DataLoader, Subset
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    device = "cuda"
    cfg = task_config(inputs, group, seed, fold, info, smoke_batches)
    smoke = cfg["smoke"]
    dest = out / "smoke" / group if smoke else base.task_path(out, group, seed, fold)
    dest.mkdir(parents=True, exist_ok=True)
    if not smoke and (dest / "done.json").exists():
        read_completed(dest, cfg, info)
        print("[SKIP authenticated]", group, seed, fold, flush=True)
        return True
    if not smoke and (dest / "task.json").exists():
        require(base.jload(dest / "task.json") == cfg, f"任务身份已改变：{dest}")
    base.jsave(dest / "task.json", cfg)
    checkpoint = dest / "best_state.pt"
    checkpoint.unlink(missing_ok=True)
    (dest / "failure.json").unlink(missing_ok=True)

    split = inputs.splits[str(fold)]
    val, test = ([inputs.original[i] for i in split[k]] for k in ("val", "test"))
    train, collator = training_dataset(inputs, group, fold, tokenizer)
    val_ds, eval_collator = base.datasets_for(val, tokenizer)
    if smoke:
        indices = training_indices(inputs.train[fold], base.CONFIG["batch_size"], smoke_batches)
        train = Subset(train, indices)
    n_steps = 2 if smoke else cfg["steps"]
    init_seed = seed * 1000 + fold
    base.set_seed(init_seed)
    net = AutoModelForSequenceClassification.from_pretrained(
        base.MODEL_ID, revision=MODEL_REV, num_labels=4,
        id2label=dict(enumerate(base.LABELS)), label2id=base.L2I, ignore_mismatched_sizes=False).float().to(device)
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if param.requires_grad:
            (no_decay if name.endswith("bias") or "LayerNorm.weight" in name or "layer_norm.weight" in name else decay).append(param)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": base.CONFIG["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=base.CONFIG["lr"])
    schedule = get_linear_schedule_with_warmup(optimizer, int(n_steps * base.CONFIG["warmup_ratio"]), n_steps)
    scaler = torch.amp.GradScaler(device)
    updates, skips = 0, 0
    best, best_epoch, epochs, trace = -1.0, None, [], []
    started = time.monotonic()
    for epoch in range(1, cfg["epochs"] + 1):
        net.train()
        generator = torch.Generator().manual_seed(init_seed + 10000 + epoch - 1)
        loader = DataLoader(train, batch_size=base.CONFIG["batch_size"], shuffle=True,
                            generator=generator, collate_fn=collator, num_workers=0)
        total_loss, seen = 0.0, 0
        for step, batch in enumerate(loader):
            if smoke and updates == 2:
                break
            targets = batch.pop("soft_targets", None)
            if targets is not None:
                targets = targets.to(device)
            y = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(net, batch, y, targets, device)
            require(bool(torch.isfinite(loss)), f"训练 loss 非有限值：{group}/{seed}/{fold}")
            status, norm = base.apply_update(loss, optimizer, schedule, scaler, list(net.parameters()))
            if smoke:
                trace.append({"epoch": epoch, "batch": step, "status": status,
                              "grad_norm_unscaled_preclip": norm if math.isfinite(norm) else None,
                              "loss": float(loss.detach())})
            if status not in ("updated", "skipped"):
                base.jsave(dest / "failure.json", {"status": status, "epoch": epoch, "batch": step, "norm": str(norm)})
                raise RuntimeError(f"训练更新异常：{group}/{seed}/{fold} {status}")
            updates += status == "updated"
            skips += status == "skipped"
            total_loss += float(loss.detach()) * len(y)
            seen += len(y)
        if smoke and updates < 2:
            base.jsave(dest / "failure.json", {"status": "insufficient_smoke_updates", "trace": trace,
                                               "attempted_batches": smoke_batches})
            del net
            torch.cuda.empty_cache()
            return False
        logits = base.prediction(net, val_ds, eval_collator)
        val_score = base.wf1([base.L2I[r["label"]] for r in val], logits.argmax(axis=1))
        epochs.append({"epoch": epoch, "val_original_wf1": val_score, "train_loss": total_loss / seen,
                       "updates_total": updates, "skips_total": skips, "scaler_scale": scaler.get_scale()})
        print("[EPOCH]", group, seed, fold, epochs[-1], flush=True)
        if val_score > best + 1e-12:
            best, best_epoch = val_score, epoch
            torch.save(net.state_dict(), checkpoint)

    test_ds, _ = base.datasets_for(test, tokenizer)
    val_rev, _ = base.datasets_for(val, tokenizer, reverse=True)
    test_rev, _ = base.datasets_for(test, tokenizer, reverse=True)
    last = [base.prediction(net, ds, eval_collator) for ds in (val_ds, val_rev, test_ds, test_rev)]
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    checkpoint.unlink()
    scores = write_outputs(inputs, dest, net, (val_ds, val_rev, test_ds, test_rev), eval_collator, val, test, last)
    require(abs(scores["best_val_original_wf1"] - best) <= 1e-12, "恢复 checkpoint 后验证分数改变")
    done = {"status": "pass", "task_fingerprint": base.canonical_json_hash(cfg), **scores,
            **{k: cfg[k] for k in ("group", "seed", "fold", "train_rows", "val_rows", "test_rows", "class_weights", "seed_policy")},
            "best_epoch": best_epoch, "epochs": epochs, "planned_updates": n_steps,
            "actual_optimizer_updates": updates, "skipped_fp16_updates": skips,
            "lr_schedule": base.learning_rate_trace(n_steps),
            "source_count": {"original": len(split["fit"]), base.CONFIG["data_source"]: cfg["train_rows"] - len(split["fit"])},
            "repeated_original_row_ids": [], "repeat_counts": {}, "inference_precision": "fp32_no_autocast",
            "runtime_sec": round(time.monotonic() - started, 3), "environment": info, **output_hashes(dest),
            "target_counts": target_counts(inputs, group, fold)}
    base.jsave(dest / "done.json", done)
    if smoke:
        base.jsave(dest / "smoke.json", {"status": "pass", "group": group, "updates": updates, "skips": skips,
                                         "grad_trace": trace, "smoke_batches": smoke_batches, "output_path_checked": True,
                                         "task_fingerprint": done["task_fingerprint"]})
    print("[DONE]", group, seed, fold, "SMOKE" if smoke else "formal", flush=True)
    del net
    torch.cuda.empty_cache()
    return True


def write_outputs(inputs, dest, net, datasets, collator, val, test, last):
    import numpy as np

    val_ds, val_rev, test_ds, test_rev = datasets

    def softmax(logits):
        logits = logits.astype(float)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    plain, flipped = (softmax(base.prediction(net, ds, collator)) for ds in (test_ds, test_rev))
    sym = (plain + flipped[:, base.SWAP]) / 2.0
    val_logits = base.prediction(net, val_ds, collator)
    val_probs = softmax(val_logits)
    names = ("E", "F", "B", "O")

    def columns(prefix, values):
        return {prefix + name: format(float(v), ".10g") for name, v in zip(names, values)}

    validation = [{"instance_id": r["instance_id"], "pair_id": r["pair_id"], "label": r["label"],
                   "pred": base.LABELS[int(val_probs[i].argmax())],
                   **columns("p_", val_probs[i]), **columns("logit_", val_logits[i])} for i, r in enumerate(val)]
    base.csvsave(dest / "val_predictions.csv", validation, list(validation[0]))
    extra = []
    rev_logits = base.prediction(net, val_rev, collator)
    for part, rows, best_rev, last_ab, last_ba in (("val", val, rev_logits, last[0], last[1]),
                                                   ("test", test, None, last[2], last[3])):
        for i, r in enumerate(rows):
            extra.append({"part": part, "instance_id": r["instance_id"],
                          **(columns("best_rev_", best_rev[i]) if best_rev is not None else {}),
                          **columns("last_", last_ab[i]), **columns("last_rev_", last_ba[i])})
    base.csvsave(dest / "extra_logits.csv", extra, list(extra[0]))
    preds = [{"instance_id": r["instance_id"], "pair_id": r["pair_id"], "reverse_pair_id": r.get("reverse_pair_id", ""),
              "family_id": inputs.family_of_pair[r["pair_id"]], "label": r["label"],
              "plain": base.LABELS[int(plain[i].argmax())], "sym": base.LABELS[int(sym[i].argmax())],
              **columns("p_", plain[i]), **columns("r_p_", flipped[i]), **columns("sym_p_", sym[i])}
             for i, r in enumerate(test)]
    base.csvsave(dest / "predictions.csv", preds, list(preds[0]))
    y_val, y_test = ([base.L2I[r["label"]] for r in rows] for rows in (val, test))
    return {"best_val_original_wf1": base.wf1(y_val, val_probs.argmax(axis=1)),
            "test_original_wf1": base.wf1(y_test, plain.argmax(axis=1)),
            "test_sym_wf1": base.wf1(y_test, sym.argmax(axis=1))}


def smoke_done(inputs, out, info, group):
    dest = out / "smoke" / group
    smoke = base.jload(dest / "smoke.json")
    batches = smoke["smoke_batches"]
    require(batches in (SMOKE_LEGACY_BATCHES, SMOKE_MAX_BATCHES), f"{group} SMOKE batch 数不合法")
    done = read_completed(dest, task_config(inputs, group, 42, 0, info, batches), info)
    trace = smoke["grad_trace"]
    statuses = [x["status"] for x in trace]
    require(smoke["status"] == "pass" and smoke["group"] == group and smoke["output_path_checked"] is True
            and smoke["task_fingerprint"] == done["task_fingerprint"]
            and smoke["updates"] == done["actual_optimizer_updates"] == statuses.count("updated") == 2
            and smoke["skips"] == done["skipped_fp16_updates"] == statuses.count("skipped")
            and len(statuses) == 2 + smoke["skips"] <= batches and statuses[-1] == "updated"
            and all(base._fin(x["loss"], 0) and (x["status"] == "skipped" or base._fin(x["grad_norm_unscaled_preclip"], 0)) for x in trace)
            and done.get("target_counts") == target_counts(inputs, group, 0), f"{group} SMOKE 回执不完整或不匹配")
    return smoke, done


def b2_reuse(inputs, out, info):
    """Commit 5 对照逐格复现 + 15 份正式任务完整性。失败则在本队列重跑 B2。"""
    from a1_data import reference_task

    root = inputs.bundle / "baseline"
    issues, compared = [], {}
    try:
        archived = root / "smoke"
        reference = read_completed(archived, reference_task(inputs, B2, 42, 0, info, True), info)
        old = base.jload(archived / "smoke.json")
        smoke, current = smoke_done(inputs, out, info, B2)
        if not (smoke["smoke_batches"] == 2 and smoke["skips"] == old["skips"] == 0
                and old.get("status") == "pass" and old.get("output_path_checked") is True
                and [x["status"] for x in old["grad_trace"]] == ["updated", "updated"]):
            issues.append("B2 SMOKE 两步更新或梯度记录不同")
        for key in ("best_epoch", "best_val_original_wf1", "planned_updates", "actual_optimizer_updates",
                    "skipped_fp16_updates", "lr_schedule", "train_rows", "val_rows", "test_rows",
                    "class_weights", "repeat_counts", "source_count", "epochs"):
            if reference[key] != current[key]:
                issues.append("B2 SMOKE done." + key)
        for name in OUTPUTS:
            expected, actual = (base.csvrows(path / name) for path in (archived, out / "smoke" / B2))
            fields = ("part", "instance_id") if name == "extra_logits.csv" else ("instance_id",)
            maps = [{tuple(r[k] for k in fields): r for r in rows} for rows in (expected, actual)]
            left, right = maps
            if left.keys() != right.keys() or len(left) != len(expected) or len(right) != len(actual):
                issues.append(f"B2 SMOKE {name} 行标识不同")
                continue
            cells = sum(sum(a.get(k) != right[i].get(k) for k in a.keys() | right[i].keys()) for i, a in left.items())
            compared[name] = {"rows": len(left), "different_cells": cells}
            if cells:
                issues.append(f"B2 SMOKE {name} 有 {cells} 格不同")
    except (KeyError, ValueError, OSError, TypeError, RuntimeError) as exc:
        issues.append(f"B2 SMOKE 归档核对失败：{exc}")
    accepted = 0
    for seed in base.SEEDS:
        for fold in base.FOLDS:
            try:
                expected = reference_task(inputs, B2, seed, fold, info, False)
                read_completed(base.task_path(root, B2, seed, fold), expected, info)
                accepted += 1
            except (KeyError, ValueError, OSError, TypeError, RuntimeError) as exc:
                issues.append(f"B2 正式归档 seed{seed}/fold{fold}：{exc}")
    return {"reuse": not issues, "archive_tasks_accepted": accepted, "compared": compared, "differences": issues}
