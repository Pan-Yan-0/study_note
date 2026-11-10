"""实验登记表：从保存的指纹、运行记录和逐条预测自动生成，写出 实验登记表.md 和 metrics/registry.json。

每个配置一行（种子合在一行，列出已完成的种子）：模型、全部训练变量、数据划分、训练数据量、执行环境（平台、卡、精度、torch）、代码版本和哈希、分数。
每个比较一行，分两列归类（GPT 10-22 审阅后的写法）：
- 设计：两边训练变量差几处。0 处是重复运行，1 处是单一计划改动，多处是系统比较。
- 执行环境：一致 / 同卡同精度、软件或平台不同 / 卡或精度不同；有字段没记录时另标"信息不全"。
  平台名称、torch 和库版本的差别只标注，不要求补跑（Pan-Yan-0 10-22 定的）；但它们也没有被证明不影响训练，
  所以结论要写成"单一计划改动，在某环境差异下"，不能只看归类就当成严格的控制变量。
没记录的值（None、"未记录"）不当作相同，单独列为"未知"。
比较的数字是同种子 sym WF1 的差、三个种子平均，按家族配对重抽样 2,000 次（每次所有种子用同一批家族）。

新增运行：在 RUNS 里加一项（指向它的结果目录）；新增比较：在 COMPARISONS 里加一项。
用法：python3 code/registry.py <官方仓库>
"""
import csv
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
M = os.path.join(EXP, "metrics")
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
B, BOOT_SEED = 2000, 20261021


def rj(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def preds(p):
    return {r["instance_id"]: r["label"] for r in rows(p)}


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def short(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


# ---------- 每类运行怎么读 ----------

def qwen_run(rid, mdir, track, kind, platform_name, desc):
    """第四组的 Qwen 运行（零样本或 SFT）。变量来自 fingerprint.json，平台来自 run_meta.json。"""
    root = os.path.join(M, mdir)
    fp = rj(os.path.join(root, "fingerprint.json"))
    metas = [r for f in sorted(os.listdir(root)) if f.startswith("run_meta") for r in rj(os.path.join(root, f))]
    seeds_dir = {}
    for name in sorted(os.listdir(os.path.join(root, f"track{track}"))):
        d = os.path.join(root, f"track{track}", name)
        if os.path.exists(os.path.join(d, "done.json")):
            if kind == "zero_shot" and name == "zero_shot":
                seeds_dir["-"] = d
            elif kind == "sft" and name.startswith("seed"):
                seeds_dir[name[4:]] = d
    variables = {
        "model": fp["model"], "model_revision": fp.get("model_revision"),
        "prompt": short(fp["prompt_text"]), "split": fp["split"],
        "train_data": f"track{track} fold 0 划分的训练部分" if kind == "sft" else "不训练",
        "method": "LoRA SFT，只算答案字母的交叉熵" if kind == "sft" else "零样本，读 A–D 的概率",
        "inference": "plain 和 sym（F / B 互换后平均）",
    }
    if kind == "sft":
        variables.update({"lora": fp["lora"], "train": fp["train"], "selection": "内层验证原方向 WF1，同分取早的",
                          "lib": fp["pins"]})
    else:
        variables["lib"] = fp["pins"]
    platform = {"platform": platform_name, "gpu": sorted({m.get("gpu") for m in metas}),
                "dtype": sorted({m.get("dtype") for m in metas}), "torch": sorted({m.get("torch") for m in metas})}
    evals = {}
    for s, d in seeds_dir.items():
        for name in ("fold0", "dev"):
            evals.setdefault(name, {})[s] = {m: preds(os.path.join(d, f"pred_{name}_{m}.csv")) for m in ("plain", "sym")}
    return {"id": rid, "desc": desc, "track": track, "variables": variables, "platform": platform,
            "code": {"version": fp["version"], "code_sha256": sorted({str(m.get("code_sha256"))[:12] for m in metas})},
            "metrics_dir": f"metrics/{mdir}", "seeds": sorted(seeds_dir), "evals": evals}


def mdeberta_run(rid, recipe, part, desc):
    """第二组的 mDeBERTa（英语）。part = "oof"：5 折折外模型（fold 0 那折用 2,190 行训练）；"dev"：全训练集的 dev 模型。"""
    fp = rj(os.path.join(M, "diagnostics", "fingerprint.json"))
    seeds = [str(s) for s in fp["seeds"]]
    if part == "oof":
        env = rj(os.path.join(M, "diagnostics", "oof", recipe, "seed42", "fold0", "train_log.json"))["env"]
        train_data = "track1 5 折折外：每折用其余 4 折去掉内层验证后训练（fold 0 那折是 2,190 行）"
    else:
        env = rj(os.path.join(M, "diagnostics", "dev", recipe, "seed42", "train_log.json"))["env"]
        train_data = "track1 全训练集去掉内层验证（2,737 行）"
    variables = {
        "model": fp["model"], "model_revision": "未记录", "prompt": "不适用",
        "split": {"folds": fp["folds"], "fold_seed": fp["fold_seed"], "inner_fraction": fp["inner_fraction"], "holdout_fold": 0},
        "train_data": train_data, "method": "全参数微调，[CLS] text1 [SEP] text2 [SEP]，交叉熵",
        "inference": "plain 和 sym（F / B 互换后平均）",
        "train": {"max_epochs": fp["recipes"][recipe], "lr": fp["lr"], "batch": fp["batch_size"], "max_len": fp["max_len"],
                  "warmup": fp["warmup_ratio"], "weight_decay": fp["weight_decay"]},
        "selection": "内层验证原方向 WF1，同分取早的", "lib": {"transformers": env.get("transformers")},
    }
    platform = {"platform": "T4（平台名称未记录）", "gpu": [env.get("device")], "dtype": ["fp16 autocast"], "torch": [env.get("torch")]}
    evals = {}
    for s in seeds:
        if part == "oof":
            d = os.path.join(M, "diagnostics", "oof", recipe, f"seed{s}")
            evals.setdefault("oof", {})[s] = {m: preds(os.path.join(d, f"oof_pred_{m}.csv")) for m in ("plain", "sym")}
        else:
            d = os.path.join(M, "diagnostics", "dev", recipe, f"seed{s}")
            evals.setdefault("dev", {})[s] = {m: preds(os.path.join(d, f"dev_pred_{m}.csv")) for m in ("plain", "sym")}
    if part == "oof":  # fold 0 是折外预测里 fold 0 那一折
        fam = rj(os.path.join(M, "diagnostics", "families.json"))
        for s in seeds:
            evals.setdefault("fold0", {})[s] = {m: {i: l for i, l in evals["oof"][s][m].items()
                                                    if fam["fold_of_family"][fam["family_of_pair"][i.split("__")[0]]] == 0}
                                                for m in ("plain", "sym")}
    return {"id": rid, "desc": desc, "track": 1, "variables": variables, "platform": platform,
            "code": {"version": fp["version"], "code_sha256": ["未记录"]}, "metrics_dir": "metrics/diagnostics", "seeds": seeds,
            "evals": evals}


def qwen_methods_run(rid, method, desc):
    """第六组正式运行；保留损失、正则参数和数据身份，SMOKE 不登记为成绩。"""
    mdir = f"qwen_methods_v2/{method}_results"
    fp = rj(os.path.join(M, mdir, "fingerprint.json"))
    assert not fp["smoke"] and fp["method"] == method
    metas = rj(os.path.join(M, mdir, "run_meta.json"))
    platforms = sorted({m["platform"] for m in metas})
    assert len(platforms) == 1
    r = qwen_run(rid, mdir, 1, "sft", platforms[0], desc)
    # Method names and the repeated dropout field are metadata aliases, not extra interventions.
    assert fp["method_params"]["lora_dropout"] == fp["lora"]["dropout"]
    params = {k:v for k,v in fp["method_params"].items() if k != "lora_dropout"}
    r["variables"].update({"method": "LoRA SFT", "method_params": params,
                           "data_sha256": fp["data_sha256"]})
    assert r["seeds"] == ["42", "43", "44"]
    return r


def mdeberta_methods_run(rid, track, method, desc):
    """第五组已完成的第一阶段，fold 0 三种子；不把孤立 fold 1 当 OOF。"""
    mdir = "mdeberta_methods_stage1"
    root = os.path.join(M, mdir)
    directories = [os.path.join(root, f"track{track}", method, f"seed{s}", "fold0") for s in [42,43,44]]
    tasks = [rj(os.path.join(d, "task.json")) for d in directories]
    done = [rj(os.path.join(d, "done.json")) for d in directories]
    t = tasks[0]
    for task in tasks:
        assert all(task[k] == t[k] for k in ["model", "model_revision", "train", "split", "method", "method_params", "data_sha256"])
        assert not task["smoke"] and task["fold"] == 0 and task["track"] == track
    env = done[0]["env"]
    assert all(x["env"] == env for x in done)
    evals = {part: {str(seed): {mode: preds(os.path.join(d, f"pred_{saved}_{mode}.csv")) for mode in ["plain", "sym"]}
                    for seed,d in zip([42,43,44],directories)} for part,saved in [("fold0","test"),("dev","dev")]}
    variables = {"model":t["model"], "model_revision":t["model_revision"], "prompt":"不适用",
                 "train":t["train"], "split":dict(t["split"],holdout_fold=0), "method":method,
                 "method_params":t["method_params"], "data_sha256":t["data_sha256"],
                 "train_data":f"track{track} fold 0 划分的训练部分", "selection":t["train"]["selection"],
                 "inference":"plain 和 sym（F / B 互换后平均）", "lib":{"transformers":env["transformers"]}}
    return {"id":rid, "desc":desc, "track":track, "variables":variables,
            "platform":{"platform":env["platform"],"gpu":[env["gpu"]],"dtype":[env["amp"]],"torch":[env["torch"]]},
            "code":{"version":rj(os.path.join(root,"run_meta.json"))[0]["version"],"code_sha256":sorted({x["code_sha256"][:12] for x in done})},
            "metrics_dir":f"metrics/{mdir}/track{track}/{method}","seeds":["42","43","44"],"evals":evals}


def joint_run(rid, method):
    root = os.path.join(M, "joint_validation")
    seeds = [42,43,44] if method in ("qwen","kd") else list(range(42,47))
    if method == "qwen":
        r = qwen_run(rid, "joint_validation/formal/qwen/fold0/dico_qwen_ce_results", 1, "sft", "阿里云 PAI-DSW", "Qwen CE SFT 英语五折三种子")
        r["variables"]["split"] = dict(r["variables"]["split"], holdout_fold="0–4")
        r["variables"]["data_sha256"] = rj(os.path.join(root,"formal/qwen/fold0/dico_qwen_ce_results/fingerprint.json"))["data_sha256"]
        name = "qwen"
    else:
        base = "formal/kd/dico_mdeberta_results" if method == "kd" else "mdeberta/dico_mdeberta_results"
        d = os.path.join(root,base,"track1",method,"seed42","fold0")
        t,done = rj(os.path.join(d,"task.json")),rj(os.path.join(d,"done.json"))
        e=done["env"]
        r={"id":rid,"desc":f"英语五折 {method}，{len(seeds)}种子","track":1,
           "variables":{"model":t["model"],"model_revision":t["model_revision"],"train":t["train"],
                        "split":dict(t["split"],holdout_fold="0–4"),"method":"编码器分类训练",
                        "method_params":t["method_params"],"data_sha256":t["data_sha256"],
                        "selection":t["train"]["selection"],"inference":"plain 和 sym（F / B 互换后平均）",
                        "lib":{"transformers":e["transformers"]}},
           "platform":{"platform":e["platform"],"gpu":[e["gpu"]],"dtype":["torch.bfloat16"],"torch":[e["torch"]]},
           "code":{"version":"joint_validation","code_sha256":[done["code_sha256"][:12]]}}
        name="kd" if method=="kd" else "mdeberta/"+method
    r["variables"]["train_data"]="track1 按家族五折，每折只训练fit，内层验证选轮"
    r["metrics_dir"]="metrics/joint_validation"
    r["seeds"]=[str(x) for x in seeds]
    r["evals"]={"oof":{str(seed):{mode:preds(os.path.join(root,"oof",name,f"seed{seed}",f"pred_{mode}.csv"))
                                  for mode in ("plain","sym")} for seed in seeds}}
    return r


def qwen8b_run(rid):
    """实验 A：Qwen3-8B 英语五折三种子（A10，v5.5 队列，上游训练核心和 JOINT-QWEN 相同）。变量来自每个任务的 task.json。"""
    root = os.path.join(M, "qwen8b_a1")
    seeds = [42, 43, 44]
    tasks = [rj(os.path.join(root, "formal", f"seed{s}", f"fold{k}", "task.json")) for s in seeds for k in range(5)]
    t = tasks[0]
    for task in tasks:
        assert not task["smoke"] and task["pipeline"] == t["pipeline"] and task["settings"]["lora"] == t["settings"]["lora"]
        assert task["settings"]["train"] == t["settings"]["train"] and task["runtime"] == t["runtime"]
    st, env = t["settings"], t["runtime"]
    variables = {"model": t["model"], "model_revision": t["revision"], "prompt": short(st["prompt_text"]),
                 "split": dict(st["split"], holdout_fold="0–4"),
                 "train_data": "track1 按家族五折，每折只训练fit，内层验证选轮",
                 "method": "LoRA SFT，只算答案字母的交叉熵", "inference": "plain 和 sym（F / B 互换后平均）",
                 "lora": st["lora"], "train": st["train"], "selection": "内层验证原方向 WF1，同分取早的",
                 "lib": st["pins"], "data_sha256": t["data"]}
    return {"id": rid, "desc": "Qwen3-8B CE SFT 英语五折三种子（实验 A）", "track": 1, "variables": variables,
            "platform": {"platform": "阿里云 PAI-DSW", "gpu": [env["gpu"]], "dtype": ["torch.bfloat16"], "torch": [env["torch"]]},
            "code": {"version": st["version"] + "（qwen8b_v5.5 队列，代码身份 " + t["pipeline"][:12] + "）",
                     "code_sha256": [rj(os.path.join(root, "MANIFEST.json"))["files"]["source/upstream_qwen.py"][:12]]},
            "metrics_dir": "metrics/qwen8b_a1", "seeds": [str(s) for s in seeds],
            "evals": {"oof": {str(s): {m: preds(os.path.join(root, "oof", f"seed{s}_{m}.csv")) for m in ("plain", "sym")} for s in seeds}}}


def augb_run(rid, group):
    """实验 B +1：B0 / B1-Haiku / step+1，英语五折三种子（Kaggle T4 fp16，GPT 的 train_augb.py v2）。
    OOF 由 code/claude_recheck_augb.py 从每个任务的四类概率重建。"""
    root = os.path.join(M, "augment_b", "train_plus1")
    d = {"B0": "B0", "B1-Haiku": "B1-Haiku", "step+1": "stepplus1"}[group]
    seeds = [42, 43, 44]
    tasks = [rj(os.path.join(root, "tasks", d, f"seed{s}", f"fold{k}", "task.json")) for s in seeds for k in range(5)]
    done = [rj(os.path.join(root, "tasks", d, f"seed{s}", f"fold{k}", "done.json")) for s in seeds for k in range(5)]
    t = tasks[0]
    for task in tasks:
        assert all(task[k] == t[k] for k in ["model", "revision", "config", "data_sha256", "code_sha256", "stable_environment"])
        assert not task["smoke"] and task["group"] == group
    env = done[0]["environment"]
    assert all(x["environment"] == env for x in done)
    c = t["config"]
    data = {"B0": "track1 按家族五折，每折只训练fit，内层验证选轮",
            "B1-Haiku": "track1 按家族五折，每折 fit + 该折选用的 Haiku 生成行（+1，清单 b380b4a6），内层验证只用原数据",
            "step+1": "track1 按家族五折，每折 fit 每行重复两次（行数和 B1-Haiku 相同），内层验证选轮"}[group]
    variables = {"model": t["model"], "model_revision": t["revision"], "prompt": "不适用",
                 "train": {"max_epochs": c["max_epochs"], "lr": c["lr"], "batch_size": c["batch_size"], "max_length": c["max_length"],
                           "warmup_ratio": c["warmup_ratio"], "weight_decay": c["weight_decay"], "gradient_clip": c["gradient_clip"],
                           "precision": c["precision"], "seed_policy": c["seed_note"], "selection": c["validation"] + "，" + c["selection_tie"]},
                 "split": {"folds": 5, "fold_seed": c["split_seed"], "inner_fraction": c["inner_family_fraction"], "holdout_fold": "0–4"},
                 "method": "ce", "method_params": "无", "data_sha256": {k: v[:12] for k, v in t["data_sha256"].items()},
                 "train_data": data, "selection": c["validation"], "inference": "plain 和 sym（F / B 互换后平均）",
                 "lib": {"transformers": env["transformers"]}}
    return {"id": rid, "desc": f"实验 B +1：{group}，mDeBERTa CE 英语五折三种子（Kaggle）", "track": 1, "variables": variables,
            "platform": {"platform": "Kaggle", "gpu": [env["gpu"]], "dtype": [env["precision"]], "torch": [env["torch"]]},
            "code": {"version": "train_augb.py v2（GPT）", "code_sha256": [t["code_sha256"][:12]]},
            "metrics_dir": "metrics/augment_b/train_plus1", "seeds": [str(s) for s in seeds],
            "evals": {"oof": {str(s): {m: preds(os.path.join(root, "oof", f"{d}_seed{s}_{m}.csv")) for m in ("plain", "sym")} for s in seeds}}}


AUGB_C1 = {"B0": ("B0", "track1 按家族五折，每折只训练 fit"),
           "W2": ("W2", "同 B0；loss = mean_i w_{y_i}·CE_i，w_O = 2/Z、其余 1/Z、Z = 1 + p_O(fit)"),
           "W4": ("W4", "同 B0；loss = mean_i w_{y_i}·CE_i，w_O = 4/Z、其余 1/Z、Z = 1 + 3·p_O(fit)"),
           "step+2": ("stepplus2", "fit 每行重复三次（+2 的行数）"),
           "step+4": ("stepplus4", "fit 每行重复五次（+4 的行数）"),
           "Dup-U4": ("Dup-U4", "fit + 按 mixquota-v1 的 U4 构成重复采样 fit（行数 = 2·fit）"),
           "Dup-BAL": ("Dup-BAL", "fit + 按 mixquota-v1 的 BAL 构成重复采样 fit（行数 = 2·fit）"),
           "Dup-E1": ("Dup-E1", "fit + 按 mixquota-v1 的 E1 构成重复采样 fit（行数 = 2·fit）"),
           "Dup-D1": ("Dup-D1", "fit + 按 mixquota-v1 的 D1 构成重复采样 fit（行数 = 2·fit）"),
           "Dup-O1": ("Dup-O1", "fit + 按 mixquota-v1 的 O1 构成重复采样 fit（行数 = 2·fit）")}
AUGB_C2 = {"B2-Haiku": ("B2-Haiku", "fit + 规模轮清单 24704e11 的第 1、2 层（+2，嵌套，第 1 层 = +1 的选用）"),
           "B4-Haiku": ("B4-Haiku", "fit + 规模轮清单 24704e11 的第 1、2、4 层（+4，嵌套）"),
           **{f"G-{p}": (f"G-{p}", f"fit + 按 mixquota-v1 的 {p} 构成取 Haiku 生成行（pool_order 7d373345，每类行数和 Dup-{p} 相同）")
              for p in ("U4", "BAL", "E1", "D1", "O1")}}
AUGB_C3 = {"Focal": ("Focal", "track1 按家族五折，每折只训练 fit"),
           "Dice": ("Dice", "track1 按家族五折，每折只训练 fit")}
LOSS_DESC = {"focal_softmax_a1_g2": ("focal", "四分类 softmax focal，alpha=1，gamma=2，fp32、autocast 之外"),
             "soft_dice_macro4_batch": ("soft_dice", "四类 macro soft Dice，batch 级、缺席类也平均，eps=1e-6，fp32、autocast 之外")}


def augb_c1_run(rid, group, bias=None, commit="1"):
    """实验 B Commit 1：Claude 的 train_augb_v3.py（v2 的训练核心 + 新组），英语五折三种子（Kaggle T4 fp16）。
    OOF 由 code/claude_recheck_augb_c1.py 从四类概率重建；bias=λ 时是 W 组在推理时把 O 的 logit 减 log λ。"""
    root = os.path.join(M, "augment_b", f"train_c{commit}")
    d, data = {"1": AUGB_C1, "2": AUGB_C2, "3": AUGB_C3}[commit][group]
    seeds = [42, 43, 44]
    tasks = [rj(os.path.join(root, "tasks", d, f"seed{s}", f"fold{k}", "task.json")) for s in seeds for k in range(5)]
    done = [rj(os.path.join(root, "tasks", d, f"seed{s}", f"fold{k}", "done.json")) for s in seeds for k in range(5)]
    t = tasks[0]
    for task in tasks:
        assert all(task[k] == t[k] for k in ["model", "revision", "config", "data_sha256", "code_sha256", "stable_environment"])
        assert not task["smoke"] and task["group"] == group
    env = done[0]["environment"]
    assert all(x["environment"] == env for x in done)
    c = t["config"]
    lam = t["group_spec"].get("lambda_O", 1.0)
    variables = {"model": t["model"], "model_revision": t["revision"], "prompt": "不适用",
                 "train": {"max_epochs": c["max_epochs"], "lr": c["lr"], "batch_size": c["batch_size"], "max_length": c["max_length"],
                           "warmup_ratio": c["warmup_ratio"], "weight_decay": c["weight_decay"], "gradient_clip": c["gradient_clip"],
                           "precision": c["precision"], "seed_policy": c["seed_note"], "selection": c["validation"] + "，" + c["selection_tie"]},
                 "split": {"folds": 5, "fold_seed": c["split_seed"], "inner_fraction": c["inner_family_fraction"], "holdout_fold": "0–4"},
                 "method": LOSS_DESC[t["loss"]][0] if t.get("loss") in LOSS_DESC else ("ce" if lam == 1.0 else "weighted_ce"),
                 "method_params": LOSS_DESC[t["loss"]][1] if t.get("loss") in LOSS_DESC else ("无" if lam == 1.0 else f"lambda_O={lam}"),
                 "data_sha256": {k: v[:12] for k, v in t["data_sha256"].items()},
                 "train_data": data, "selection": c["validation"],
                 "inference": "plain 和 sym（F / B 互换后平均）" + (f"；O 的 logit 减 log {lam:g}（两个方向都减，再对称化）" if bias else ""),
                 "lib": {"transformers": env["transformers"]}}
    name = d if not bias else f"{group}-bias"
    return {"id": rid, "desc": f"实验 B Commit {commit}：{group}{'（偏置对照）' if bias else ''}，mDeBERTa 英语五折三种子（Kaggle）", "track": 1,
            "variables": variables,
            "platform": {"platform": "Kaggle", "gpu": [env["gpu"]], "dtype": [env["precision"]], "torch": [env["torch"]]},
            "code": {"version": "train_augb_v4.py v4.1（Claude）" if commit == "3" else "train_augb_v3.py（Claude）", "code_sha256": [t["code_sha256"][:12]]},
            "metrics_dir": f"metrics/augment_b/train_c{commit}", "seeds": [str(s) for s in seeds],
            "evals": {"oof": {str(s): {m: preds(os.path.join(root, "oof", f"{name}_seed{s}_{m}.csv")) for m in ("plain", "sym")} for s in seeds}}}


def multilang_run(rid, track, fold0_only=False):
    """Kaggle T4 上 mDeBERTa CE（第五组 C0 配方）的西语 / 巴语 / 跨语言五折三种子；dev 只有 fold 0 的模型评过。"""
    root = os.path.join(M, "mdeberta_multilang_oof")
    seeds = [42, 43, 44]
    task_dir = lambda s, k: os.path.join(root, "formal", f"t{track}_s{s}_f{k}", f"track{track}", "ce", f"seed{s}", f"fold{k}")
    tasks = [rj(os.path.join(task_dir(s, k), "task.json")) for s in seeds for k in range(5)]
    done = [rj(os.path.join(task_dir(s, k), "done.json")) for s in seeds for k in range(5)]
    t = tasks[0]
    for task in tasks:
        assert all(task[k] == t[k] for k in ["model", "model_revision", "train", "split", "method", "method_params", "data_sha256"])
        assert not task["smoke"] and task["track"] == track
    env = done[0]["env"]
    assert all(x["env"]["gpu"] == env["gpu"] and x["env"]["amp"] == env["amp"] and x["env"]["torch"] == env["torch"] for x in done)
    names = {2: "西班牙语", 3: "巴斯克语", 4: "跨语言"}
    variables = {"model": t["model"], "model_revision": t["model_revision"], "prompt": "不适用",
                 "train": t["train"], "split": dict(t["split"], holdout_fold="0–4"), "method": "ce",
                 "method_params": t["method_params"], "data_sha256": t["data_sha256"],
                 "train_data": f"track{track} 按家族五折，每折只训练fit，内层验证选轮", "selection": t["train"]["selection"],
                 "inference": "plain 和 sym（F / B 互换后平均）", "lib": {"transformers": env["transformers"]}}
    evals = {"oof": {str(s): {mode: preds(os.path.join(root, "aggregate", f"track{track}", f"seed{s}", f"pred_oof_{mode}.csv"))
                              for mode in ("plain", "sym")} for s in seeds},
             "fold0": {str(s): {mode: preds(os.path.join(task_dir(s, 0), f"pred_test_{mode}.csv")) for mode in ("plain", "sym")} for s in seeds},
             "dev": {str(s): {mode: preds(os.path.join(task_dir(s, 0), f"pred_dev_{mode}.csv")) for mode in ("plain", "sym")} for s in seeds}}
    r = {"id": rid, "desc": f"mDeBERTa CE {names[track]}五折三种子（Kaggle）", "track": track, "variables": variables,
            "platform": {"platform": env["platform"], "gpu": [env["gpu"]], "dtype": [env["amp"]], "torch": [env["torch"]]},
            "code": {"version": "dico_mdeberta_methods-3 + multilang 驱动", "code_sha256": sorted({x["code_sha256"][:12] for x in done})},
            "metrics_dir": "metrics/mdeberta_multilang_oof", "seeds": [str(s) for s in seeds], "evals": evals}
    if fold0_only:  # 只看 fold 0 那三个模型：记录方式和第五组第一阶段相同，便于逐项比较
        r["variables"].update(split=dict(t["split"], holdout_fold=0), train_data=f"track{track} fold 0 划分的训练部分")
        r["evals"] = {k: v for k, v in evals.items() if k != "oof"}
        r["id"], r["desc"] = rid, f"mDeBERTa CE {names[track]} fold 0 三种子（Kaggle，五折运行中的 fold 0）"
    return r


RUNS = [
    ("ML-T2-CE", lambda: multilang_run("ML-T2-CE", 2)),
    ("ML-T3-CE", lambda: multilang_run("ML-T3-CE", 3)),
    ("ML-T4-CE", lambda: multilang_run("ML-T4-CE", 4)),
    ("ML-T2-CE-F0", lambda: multilang_run("ML-T2-CE-F0", 2, True)),
    ("ML-T3-CE-F0", lambda: multilang_run("ML-T3-CE-F0", 3, True)),
    ("ML-T4-CE-F0", lambda: multilang_run("ML-T4-CE-F0", 4, True)),
    ("JOINT-CE", lambda: joint_run("JOINT-CE", "ce")),
    ("JOINT-RDROP", lambda: joint_run("JOINT-RDROP", "rdrop")),
    ("JOINT-DIR_RDROP", lambda: joint_run("JOINT-DIR_RDROP", "dir_rdrop")),
    ("JOINT-SCL", lambda: joint_run("JOINT-SCL", "scl")),
    ("JOINT-MOCO_M0.999", lambda: joint_run("JOINT-MOCO_M0.999", "moco_m0.999")),
    ("A1-QWEN8B", lambda: qwen8b_run("A1-QWEN8B")),
    ("AUGB-B0", lambda: augb_run("AUGB-B0", "B0")),
    ("AUGB-B1-HAIKU", lambda: augb_run("AUGB-B1-HAIKU", "B1-Haiku")),
    ("AUGB-STEP1", lambda: augb_run("AUGB-STEP1", "step+1")),
    ("AUGB-C1-B0", lambda: augb_c1_run("AUGB-C1-B0", "B0")),
    ("AUGB-W2", lambda: augb_c1_run("AUGB-W2", "W2")),
    ("AUGB-W4", lambda: augb_c1_run("AUGB-W4", "W4")),
    ("AUGB-W2-BIAS", lambda: augb_c1_run("AUGB-W2-BIAS", "W2", bias=2)),
    ("AUGB-W4-BIAS", lambda: augb_c1_run("AUGB-W4-BIAS", "W4", bias=4)),
    ("AUGB-STEP2", lambda: augb_c1_run("AUGB-STEP2", "step+2")),
    ("AUGB-STEP4", lambda: augb_c1_run("AUGB-STEP4", "step+4")),
    ("AUGB-DUP-U4", lambda: augb_c1_run("AUGB-DUP-U4", "Dup-U4")),
    ("AUGB-DUP-BAL", lambda: augb_c1_run("AUGB-DUP-BAL", "Dup-BAL")),
    ("AUGB-DUP-E1", lambda: augb_c1_run("AUGB-DUP-E1", "Dup-E1")),
    ("AUGB-DUP-D1", lambda: augb_c1_run("AUGB-DUP-D1", "Dup-D1")),
    ("AUGB-DUP-O1", lambda: augb_c1_run("AUGB-DUP-O1", "Dup-O1")),
    ("AUGB-B2-HAIKU", lambda: augb_c1_run("AUGB-B2-HAIKU", "B2-Haiku", commit="2")),
    ("AUGB-B4-HAIKU", lambda: augb_c1_run("AUGB-B4-HAIKU", "B4-Haiku", commit="2")),
    ("AUGB-G-U4", lambda: augb_c1_run("AUGB-G-U4", "G-U4", commit="2")),
    ("AUGB-G-BAL", lambda: augb_c1_run("AUGB-G-BAL", "G-BAL", commit="2")),
    ("AUGB-G-E1", lambda: augb_c1_run("AUGB-G-E1", "G-E1", commit="2")),
    ("AUGB-G-D1", lambda: augb_c1_run("AUGB-G-D1", "G-D1", commit="2")),
    ("AUGB-G-O1", lambda: augb_c1_run("AUGB-G-O1", "G-O1", commit="2")),
    ("AUGB-FOCAL", lambda: augb_c1_run("AUGB-FOCAL", "Focal", commit="3")),
    ("AUGB-DICE", lambda: augb_c1_run("AUGB-DICE", "Dice", commit="3")),
    ("JOINT-QWEN", lambda: joint_run("JOINT-QWEN", "qwen")),
    ("JOINT-KD", lambda: joint_run("JOINT-KD", "kd")),
    ("MDB-G5-T1-CE", lambda: mdeberta_methods_run("MDB-G5-T1-CE", 1, "ce", "第五组 track1 ce，fold 0 三种子")),
    ("MDB-G5-T1-KD", lambda: mdeberta_methods_run("MDB-G5-T1-KD", 1, "kd", "第五组 track1 kd，fold 0 三种子")),
    ("MDB-G5-T1-RDROP", lambda: mdeberta_methods_run("MDB-G5-T1-RDROP", 1, "rdrop", "第五组 track1 rdrop，fold 0 三种子")),
    ("MDB-G5-T1-DIR_RDROP", lambda: mdeberta_methods_run("MDB-G5-T1-DIR_RDROP", 1, "dir_rdrop", "第五组 track1 dir_rdrop，fold 0 三种子")),
    ("MDB-G5-T1-SCL", lambda: mdeberta_methods_run("MDB-G5-T1-SCL", 1, "scl", "第五组 track1 scl，fold 0 三种子")),
    ("MDB-G5-T1-MOCO_M0.99", lambda: mdeberta_methods_run("MDB-G5-T1-MOCO_M0.99", 1, "moco_m0.99", "第五组 track1 moco_m0.99，fold 0 三种子")),
    ("MDB-G5-T1-MOCO_M0.999", lambda: mdeberta_methods_run("MDB-G5-T1-MOCO_M0.999", 1, "moco_m0.999", "第五组 track1 moco_m0.999，fold 0 三种子")),
    ("MDB-G5-T2-CE", lambda: mdeberta_methods_run("MDB-G5-T2-CE", 2, "ce", "第五组 track2 ce，fold 0 三种子")),
    ("MDB-G5-T3-CE", lambda: mdeberta_methods_run("MDB-G5-T3-CE", 3, "ce", "第五组 track3 ce，fold 0 三种子")),
    ("MDB-G5-T4-CE", lambda: mdeberta_methods_run("MDB-G5-T4-CE", 4, "ce", "第五组 track4 ce，fold 0 三种子")),
    ("QWEN-Q1-DROP", lambda: qwen_methods_run("QWEN-Q1-DROP", "ce_drop", "第六组 Q1，仅增加 LoRA dropout 0.1")),
    ("QWEN-Q4-MOCO", lambda: qwen_methods_run("QWEN-Q4-MOCO", "moco", "第六组 Q4，动量监督对比学习 MoCo")),
    ("QWEN-Q3-SCL", lambda: qwen_methods_run("QWEN-Q3-SCL", "scl", "第六组 Q3，监督对比学习 SCL")),
    ("QWEN-Q0-CE", lambda: qwen_methods_run("QWEN-Q0-CE", "ce", "第六组 Q0，同代码 CE 基线")),
    ("QWEN-Q2-RDROP", lambda: qwen_methods_run("QWEN-Q2-RDROP", "rdrop", "第六组 Q2，dropout 0.1 + R-Drop")),
    ("MDB-E6-OOF", lambda: mdeberta_run("MDB-E6-OOF", "e6", "oof", "mDeBERTa 最多 6 轮，英语 5 折折外（第二组）")),
    ("MDB-E12-OOF", lambda: mdeberta_run("MDB-E12-OOF", "e12", "oof", "mDeBERTa 最多 12 轮，英语 5 折折外（第二组）")),
    ("MDB-E12-DEV", lambda: mdeberta_run("MDB-E12-DEV", "e12", "dev", "mDeBERTa 最多 12 轮，全训练集的 dev 模型（第二组）")),
    ("QWEN-ZS-T1", lambda: qwen_run("QWEN-ZS-T1", "qwen_sft", 1, "zero_shot", "Kaggle", "Qwen 零样本，英语")),
    ("QWEN-E3-T1", lambda: qwen_run("QWEN-E3-T1", "qwen_sft", 1, "sft", "Kaggle", "Qwen SFT 最多 3 轮，英语")),
    ("QWEN-E5-T1", lambda: qwen_run("QWEN-E5-T1", "qwen_sft_e5", 1, "sft", "腾讯 Cloud Studio", "Qwen SFT 最多 5 轮，英语")),
    ("QWEN-ZS-T2", lambda: qwen_run("QWEN-ZS-T2", "qwen_sft", 2, "zero_shot", "Kaggle", "Qwen 零样本，西班牙语")),
    ("QWEN-E3-T2", lambda: qwen_run("QWEN-E3-T2", "qwen_sft", 2, "sft", "Kaggle", "Qwen SFT 最多 3 轮，西班牙语")),
    ("QWEN-ZS-T3", lambda: qwen_run("QWEN-ZS-T3", "qwen_sft", 3, "zero_shot", "Kaggle", "Qwen 零样本，巴斯克语")),
    ("QWEN-E3-T3", lambda: qwen_run("QWEN-E3-T3", "qwen_sft", 3, "sft", "Kaggle", "Qwen SFT 最多 3 轮，巴斯克语")),
    ("QWEN-ZS-T1-A10", lambda: qwen_run("QWEN-ZS-T1-A10", "qwen_sft_a10_t1", 1, "zero_shot", "阿里云 PAI-DSW", "Qwen 零样本，英语（A10）")),
    ("QWEN-E3-T1-A10", lambda: qwen_run("QWEN-E3-T1-A10", "qwen_sft_a10_t1", 1, "sft", "阿里云 PAI-DSW", "Qwen SFT 最多 3 轮，英语（A10，历史 SFT 基线）")),
    ("QWEN-ZS-T4", lambda: qwen_run("QWEN-ZS-T4", "qwen_sft_a10", 4, "zero_shot", "阿里云 PAI-DSW", "Qwen 零样本，跨语言")),
    ("QWEN-E3-T4", lambda: qwen_run("QWEN-E3-T4", "qwen_sft_a10", 4, "sft", "阿里云 PAI-DSW", "Qwen SFT 最多 3 轮，跨语言")),
]

# (编号, 新, 旧, 评测集, 目的)
COMPARISONS = [
    ("JOINT-CMP-RDROP", "JOINT-RDROP", "JOINT-CE", "oof", "英语五折配对比较：rdrop 对 CE"),
    ("JOINT-CMP-DIR_RDROP", "JOINT-DIR_RDROP", "JOINT-CE", "oof", "英语五折配对比较：dir_rdrop 对 CE"),
    ("JOINT-CMP-SCL", "JOINT-SCL", "JOINT-CE", "oof", "英语五折配对比较：scl 对 CE"),
    ("JOINT-CMP-MOCO_M0.999", "JOINT-MOCO_M0.999", "JOINT-CE", "oof", "英语五折配对比较：moco_m0.999 对 CE"),
    ("AUGB-CMP-B1-B0", "AUGB-B1-HAIKU", "AUGB-B0", "oof", "实验 B +1：加 Haiku 生成数据 对 只用原数据"),
    ("AUGB-CMP-B1-STEP1", "AUGB-B1-HAIKU", "AUGB-STEP1", "oof", "实验 B +1：加 Haiku 生成数据 对 原数据重复到同样行数（同计划步数）"),
    ("AUGB-CMP-STEP1-B0", "AUGB-STEP1", "AUGB-B0", "oof", "实验 B +1：原数据重复两遍 对 一遍（只看步数加倍的作用，描述）"),
    ("AUGB-CMP-B0-JOINTCE", "AUGB-B0", "JOINT-CE", "oof", "实验 B 的 B0（Kaggle T4 fp16）对第五组 CE（A10 bf16），同划分同种子 42–44（执行环境迁移，描述）"),
    ("AUGB-CMP-C1B0-B0", "AUGB-C1-B0", "AUGB-B0", "oof", "Commit 1 的 B0（v3 代码）对 +1 的 B0（v2 代码）：代码路径对照"),
    ("AUGB-CMP-W2-B0", "AUGB-W2", "AUGB-C1-B0", "oof", "类别加权：O 权重 λ=2 对 λ=1（同数据、同步数）"),
    ("AUGB-CMP-W4-B0", "AUGB-W4", "AUGB-C1-B0", "oof", "类别加权：O 权重 λ=4 对 λ=1（同数据、同步数）"),
    ("AUGB-CMP-W2BIAS-B0", "AUGB-W2-BIAS", "AUGB-C1-B0", "oof", "类别加权 λ=2，推理时 O 的 logit 减 log 2（事先声明的解析偏置对照）对 λ=1"),
    ("AUGB-CMP-W4BIAS-B0", "AUGB-W4-BIAS", "AUGB-C1-B0", "oof", "类别加权 λ=4，推理时 O 的 logit 减 log 4（事先声明的解析偏置对照）对 λ=1"),
    ("AUGB-CMP-STEP2-B0", "AUGB-STEP2", "AUGB-C1-B0", "oof", "原数据重复三遍（+2 的行数和步数）对 一遍（描述，B2 的对照）"),
    ("AUGB-CMP-STEP4-B0", "AUGB-STEP4", "AUGB-C1-B0", "oof", "原数据重复五遍（+4 的行数和步数）对 一遍（描述，B4 的对照）"),
    ("AUGB-CMP-DUPU4-STEP1", "AUGB-DUP-U4", "AUGB-STEP1", "oof", "类别构成 U4 的重复采样 对 按原比例重复一遍（Dup_c − Dup_R，描述；跨 v3 / v2 代码）"),
    ("AUGB-CMP-DUPBAL-STEP1", "AUGB-DUP-BAL", "AUGB-STEP1", "oof", "类别构成 BAL 的重复采样 对 按原比例重复一遍（Dup_c − Dup_R，描述；跨 v3 / v2 代码）"),
    ("AUGB-CMP-DUPE1-STEP1", "AUGB-DUP-E1", "AUGB-STEP1", "oof", "类别构成 E1 的重复采样 对 按原比例重复一遍（Dup_c − Dup_R，描述；跨 v3 / v2 代码）"),
    ("AUGB-CMP-DUPD1-STEP1", "AUGB-DUP-D1", "AUGB-STEP1", "oof", "类别构成 D1 的重复采样 对 按原比例重复一遍（Dup_c − Dup_R，描述；跨 v3 / v2 代码）"),
    ("AUGB-CMP-DUPO1-STEP1", "AUGB-DUP-O1", "AUGB-STEP1", "oof", "类别构成 O1 的重复采样 对 按原比例重复一遍（Dup_c − Dup_R，描述；跨 v3 / v2 代码）"),
    ("AUGB-CMP-B2-B0", "AUGB-B2-HAIKU", "AUGB-C1-B0", "oof", "实验 B +2：加 Haiku 生成数据 对 只用原数据"),
    ("AUGB-CMP-B2-STEP2", "AUGB-B2-HAIKU", "AUGB-STEP2", "oof", "实验 B +2：加 Haiku 生成数据 对 原数据重复到同样行数（同计划步数）"),
    ("AUGB-CMP-B4-B0", "AUGB-B4-HAIKU", "AUGB-C1-B0", "oof", "实验 B +4：加 Haiku 生成数据 对 只用原数据"),
    ("AUGB-CMP-B4-STEP4", "AUGB-B4-HAIKU", "AUGB-STEP4", "oof", "实验 B +4：加 Haiku 生成数据 对 原数据重复到同样行数（同计划步数）"),
    ("AUGB-CMP-B4-B2", "AUGB-B4-HAIKU", "AUGB-B2-HAIKU", "oof", "实验 B 规模曲线：+4 对 +2（描述；+2 ⊂ +4，步数不同）"),
    ("AUGB-CMP-FOCAL-B0", "AUGB-FOCAL", "AUGB-C1-B0", "oof", "loss 对照：Focal（γ=2）对 CE（同数据、同步数；B0 复用 Commit 1，SMOKE 逐字节相同）"),
    ("AUGB-CMP-DICE-B0", "AUGB-DICE", "AUGB-C1-B0", "oof", "loss 对照：四类 macro soft Dice 对 CE（同数据、同步数；B0 复用 Commit 1）"),
    ("AUGB-CMP-FOCAL-W4", "AUGB-FOCAL", "AUGB-W4", "oof", "Focal 对 O 类加权 λ=4（描述，两种处理类别不平衡的 loss）"),
    ("AUGB-CMP-B2-B1", "AUGB-B2-HAIKU", "AUGB-B1-HAIKU", "oof", "实验 B 规模曲线：+2 对 +1（描述；跨 v3 / v2 代码，步数不同）"),
    ("AUGB-CMP-GU4-DUPU4", "AUGB-G-U4", "AUGB-DUP-U4", "oof", "类别构成 U4：生成数据 对 同构成的原数据重复采样（G_c − Dup_c，同代码同步数）"),
    ("AUGB-CMP-GU4-B1", "AUGB-G-U4", "AUGB-B1-HAIKU", "oof", "类别构成 U4 的生成数据 对 按原比例的生成数据（G_c − G_R；跨 v3 / v2 代码）"),
    ("AUGB-CMP-GBAL-DUPBAL", "AUGB-G-BAL", "AUGB-DUP-BAL", "oof", "类别构成 BAL：生成数据 对 同构成的原数据重复采样（G_c − Dup_c，同代码同步数）"),
    ("AUGB-CMP-GBAL-B1", "AUGB-G-BAL", "AUGB-B1-HAIKU", "oof", "类别构成 BAL 的生成数据 对 按原比例的生成数据（G_c − G_R；跨 v3 / v2 代码）"),
    ("AUGB-CMP-GE1-DUPE1", "AUGB-G-E1", "AUGB-DUP-E1", "oof", "类别构成 E1：生成数据 对 同构成的原数据重复采样（G_c − Dup_c，同代码同步数）"),
    ("AUGB-CMP-GE1-B1", "AUGB-G-E1", "AUGB-B1-HAIKU", "oof", "类别构成 E1 的生成数据 对 按原比例的生成数据（G_c − G_R；跨 v3 / v2 代码）"),
    ("AUGB-CMP-GD1-DUPD1", "AUGB-G-D1", "AUGB-DUP-D1", "oof", "类别构成 D1：生成数据 对 同构成的原数据重复采样（G_c − Dup_c，同代码同步数）"),
    ("AUGB-CMP-GD1-B1", "AUGB-G-D1", "AUGB-B1-HAIKU", "oof", "类别构成 D1 的生成数据 对 按原比例的生成数据（G_c − G_R；跨 v3 / v2 代码）"),
    ("AUGB-CMP-GO1-DUPO1", "AUGB-G-O1", "AUGB-DUP-O1", "oof", "类别构成 O1：生成数据 对 同构成的原数据重复采样（G_c − Dup_c，同代码同步数）"),
    ("AUGB-CMP-GO1-B1", "AUGB-G-O1", "AUGB-B1-HAIKU", "oof", "类别构成 O1 的生成数据 对 按原比例的生成数据（G_c − G_R；跨 v3 / v2 代码）"),
    ("A1-CMP-8B-4B", "A1-QWEN8B", "JOINT-QWEN", "oof", "实验 A：Qwen3-8B 对 Qwen3-4B-Instruct-2507（同划分、同种子、同配方；系统比较，模型不同）"),
    ("JOINT-CMP-QWEN", "JOINT-QWEN", "JOINT-CE", "oof", "英语五折配对比较：qwen 对 CE"),
    ("JOINT-CMP-KD", "JOINT-KD", "JOINT-CE", "oof", "英语五折配对比较：kd 对 CE"),
    ("CMP-12", "QWEN-Q1-DROP", "QWEN-Q0-CE", "fold0", "只加 dropout 对 CE"),
    ("CMP-13", "QWEN-Q1-DROP", "QWEN-Q0-CE", "dev", "只加 dropout 对 CE；dev 仅报告"),
    ("CMP-14", "QWEN-Q2-RDROP", "QWEN-Q1-DROP", "fold0", "R-Drop 对同 dropout 的 CE；增加双前向平均与 KL"),
    ("CMP-15", "QWEN-Q2-RDROP", "QWEN-Q1-DROP", "dev", "R-Drop 对同 dropout 的 CE；dev 仅报告"),
    ("CMP-10", "QWEN-Q4-MOCO", "QWEN-Q0-CE", "fold0", "第六组 MoCo 对 CE；评估整体 MoCo 配方"),
    ("CMP-11", "QWEN-Q4-MOCO", "QWEN-Q0-CE", "dev", "第六组 MoCo 对 CE；dev 仅报告，不选方法"),
    ("CMP-08", "QWEN-Q3-SCL", "QWEN-Q0-CE", "fold0", "第六组 SCL 对 CE；评估整体 SCL 配方"),
    ("CMP-09", "QWEN-Q3-SCL", "QWEN-Q0-CE", "dev", "第六组 SCL 对 CE；dev 仅报告，不选方法"),
    ("CMP-06", "QWEN-Q2-RDROP", "QWEN-Q0-CE", "fold0", "第六组 R-Drop 对 CE；dropout 和损失同时改变"),
    ("CMP-07", "QWEN-Q2-RDROP", "QWEN-Q0-CE", "dev", "第六组 R-Drop 对 CE；dev 仅报告，不选方法"),
    ("CMP-01", "MDB-E12-OOF", "MDB-E6-OOF", "oof", "mDeBERTa 训练轮数上限：12 轮 对 6 轮（第二组定配方用的比较）"),
    ("CMP-02", "QWEN-E3-T1", "QWEN-ZS-T1", "fold0", "Qwen 微调 对 不微调"),
    ("CMP-03", "QWEN-E3-T1", "MDB-E12-OOF", "fold0", "Qwen SFT 对 mDeBERTa（同训练 / 验证 / 留出数据；系统对比，模型不同）"),
    ("CMP-04", "QWEN-E5-T1", "QWEN-E3-T1", "fold0", "Qwen SFT 轮数上限：5 轮 对 3 轮"),
    ("CMP-05", "QWEN-E3-T1-A10", "QWEN-E3-T1", "fold0", "Qwen SFT 3 轮配方换到 A10（执行环境迁移复现）"),
    ("ML-CMP-T2", "ML-T2-CE-F0", "MDB-G5-T2-CE", "fold0", "西语 mDeBERTa CE：Kaggle T4 fp16 对 A10 bf16（同配方、同划分、同种子）"),
    ("ML-CMP-T3", "ML-T3-CE-F0", "MDB-G5-T3-CE", "fold0", "巴语 mDeBERTa CE：Kaggle T4 fp16 对 A10 bf16（同配方、同划分、同种子）"),
    ("ML-CMP-T4", "ML-T4-CE-F0", "MDB-G5-T4-CE", "fold0", "跨语言 mDeBERTa CE：Kaggle T4 fp16 对 A10 bf16（同配方、同划分、同种子）"),
    ("ML-CMP-T1", "MDB-E12-OOF", "MDB-G5-T1-CE", "fold0", "英语 mDeBERTa 12 轮：第二组 T4 fp16 对第五组 A10 bf16（同划分、同种子，代码版本不同）"),
    ("ML-CMP-QT2", "QWEN-E3-T2", "ML-T2-CE-F0", "fold0", "西语 Qwen SFT 对 mDeBERTa CE（同数据、同为 Kaggle T4 fp16；模型不同）"),
    ("ML-CMP-QT3", "QWEN-E3-T3", "ML-T3-CE-F0", "fold0", "巴语 Qwen SFT 对 mDeBERTa CE（同数据、同为 Kaggle T4 fp16；模型不同）"),
    ("ML-CMP-QT4", "QWEN-E3-T4", "ML-T4-CE-F0", "fold0", "跨语言 Qwen SFT（A10）对 mDeBERTa CE（Kaggle T4）（同数据；模型和执行环境都不同）"),
    ("CMP-16", "QWEN-Q0-CE", "QWEN-E3-T1-A10", "fold0", "Qwen SFT 3 轮配方在 A10 上重跑（第六组 Q0 对第四组 R3，代码版本不同）"),
]

IGNORE_FOR_CONTROL = {"lib.transformers", "lib.peft", "lib.accelerate", "lib.datasets"}  # 库版本单独列，不算训练变量
CARD_KEYS = ("gpu", "dtype")  # 卡和精度分开记录、分开比较
UNKNOWN = (None, "未记录", "None")
NOTES = {
    "JOINT-CMP-RDROP": "使用共同种子；fold0已用于探索，五折并非独立隐藏测试。区间不包含重新训练随机性。",
    "JOINT-CMP-DIR_RDROP": "使用共同种子；fold0已用于探索，五折并非独立隐藏测试。区间不包含重新训练随机性。",
    "JOINT-CMP-SCL": "使用共同种子；fold0已用于探索，五折并非独立隐藏测试。区间不包含重新训练随机性。",
    "JOINT-CMP-MOCO_M0.999": "使用共同种子；fold0已用于探索，五折并非独立隐藏测试。区间不包含重新训练随机性。",
    "AUGB-CMP-B1-B0": "按冻结计划第 1 节的规则算提升；三个种子都升。和 AUGB-CMP-B1-STEP1 一起都通过，才算扩充有效。审核不是独立双盲、E 的放行门槛在第一次 P1 后改定（见 metrics/augment_b/train_plus1/README.md）。区间不包含重新训练随机性。",
    "AUGB-CMP-B1-STEP1": "两边每折行数、计划步数、学习率日程相同，只差多出来的那一半是生成数据还是原数据的重复；三个种子都升，按规则算提升。实际更新次数因 fp16 跳步差 0–2 次，只报告。区间不包含重新训练随机性。",
    "AUGB-CMP-STEP1-B0": "步数加倍本身也有作用（seed 43 略降），所以 B1-Haiku 对 B0 的提升里有一部分来自更多更新；扩充是否有效看 AUGB-CMP-B1-STEP1。",
    "AUGB-CMP-C1B0-B0": "三个种子的对称化预测逐条 100% 相同（λ=1 走原 CE 分支），所以 v3 的训练路径在 B0 上和 v2 等价；Dup 组拿 +1 的 step+1 当 Dup_R 依赖这一点，但 v3 的重复采样路径本身没有和 v2 的 step+1 逐条对过。",
    "AUGB-CMP-W2-B0": "按事先规则不算提升（种子 43 降）。",
    "AUGB-CMP-W4-B0": "均值为正但区间含 0，按事先规则不算提升；O 的 F1 升约 2 个百分点。",
    "AUGB-CMP-W2BIAS-B0": "不需要选择的解析对照；不算提升。",
    "AUGB-CMP-W4BIAS-B0": "区间下限 +0.02，刚过规则线；四个主比较没有做多重比较校正。撤销 log λ 的偏置后仍比 B0 高，说明增益不只是 O 的判定阈值移动，但不说明排序能力提高（GPT 的更正）。",
    "AUGB-CMP-STEP2-B0": "原数据多重复几遍也有作用，+2 和 +1（AUGB-CMP-STEP1-B0）差不多、+4 不再升；B2 / B4 是否有效看它们对 step+2 / step+4。",
    "AUGB-CMP-STEP4-B0": "见 AUGB-CMP-STEP2-B0。",
    "AUGB-CMP-DUPU4-STEP1": "两边代码版本不同（v3 对 v2），训练路径在 B0 上逐位相同（AUGB-CMP-C1B0-B0）；只作描述，不归因。",
    "AUGB-CMP-DUPBAL-STEP1": "同 AUGB-CMP-DUPU4-STEP1。",
    "AUGB-CMP-DUPE1-STEP1": "同 AUGB-CMP-DUPU4-STEP1。",
    "AUGB-CMP-DUPD1-STEP1": "同 AUGB-CMP-DUPU4-STEP1。",
    "AUGB-CMP-DUPO1-STEP1": "同 AUGB-CMP-DUPU4-STEP1。",
    "AUGB-CMP-B2-STEP2": "两边每折行数、计划步数相同，只差多出来的是生成数据还是原数据重复；和 AUGB-CMP-B2-B0 都通过，按冻结计划 3.5 节 +2 扩充有效。区间不包含重新训练随机性。",
    "AUGB-CMP-B4-STEP4": "同 AUGB-CMP-B2-STEP2；+4 扩充有效。B4 的第 4 层有 25.6% 来自 reasoning effort 为 high 的批次。",
    "AUGB-CMP-B4-B2": "+2 ⊂ +4，每轮步数也随行数增加；差约为 0，规模曲线在 +2 饱和。",
    "AUGB-CMP-B2-B1": "跨 v3 / v2 代码，只作描述。",
    "AUGB-CMP-GU4-DUPU4": "同代码、同行数、同步数、每类新增行数相同；五个构成都通过。构成 v2 第 4 节要求它和 G_c − G_R 都通过才说构成值得生成。",
    "AUGB-CMP-GU4-B1": "跨 v3 / v2 代码；五个构成都没通过，按构成 v2 没有构成值得生成，事先定的 +2 后续不触发。",
    "AUGB-CMP-GBAL-DUPBAL": "同 AUGB-CMP-GU4-DUPU4。",
    "AUGB-CMP-GBAL-B1": "同 AUGB-CMP-GU4-B1。",
    "AUGB-CMP-GE1-DUPE1": "同 AUGB-CMP-GU4-DUPU4。",
    "AUGB-CMP-GE1-B1": "同 AUGB-CMP-GU4-B1。",
    "AUGB-CMP-GD1-DUPD1": "同 AUGB-CMP-GU4-DUPU4。",
    "AUGB-CMP-GD1-B1": "同 AUGB-CMP-GU4-B1。",
    "AUGB-CMP-GO1-DUPO1": "同 AUGB-CMP-GU4-DUPU4。",
    "AUGB-CMP-GO1-B1": "同 AUGB-CMP-GU4-B1。",
    "AUGB-CMP-FOCAL-B0": "设计上只改 loss（method 和 method_params 是同一处改动的两个字段，所以自动归为系统比较）；Bonferroni 97.5% 区间 [−0.19, +1.50] 含 0，按事先规则不算提升。区间不包含重新训练随机性。",
    "AUGB-CMP-DICE-B0": "同 AUGB-CMP-FOCAL-B0；Bonferroni 97.5% 区间 [−2.38, −0.25] 在 0 以下，Dice 显著更差（种子 43 的 fold 3 选在第 3 轮，留出 0.692）。两种 loss 都没过线，按冻结计划不做“增广 + 新 loss”的叠加。",
    "AUGB-CMP-B0-JOINTCE":"训练变量的记录方式不同，自动列出的差别多是写法；实际差别是卡和精度（T4 fp16 对 A10 bf16）、库版本，以及初始化和打乱的随机数流（B0 用 seed×1000+fold、每轮独立生成器）。B0 不是 JOINT-CE 的复现，只作参照。",
    "A1-CMP-8B-4B": "登记的变量只差模型；未登记的分词器、聊天模板（8B 带空 think 块）和后训练方式也不同，所以是系统比较。按事先规则（均值 > 0 且区间下限 > 0）不算提升。区间不包含重新训练随机性。",
    "JOINT-CMP-QWEN": "使用共同种子；fold0已用于探索，五折并非独立隐藏测试。区间不包含重新训练随机性。",
    "JOINT-CMP-KD": "使用共同种子；fold0已用于探索，五折并非独立隐藏测试。区间不包含重新训练随机性。",
    "ML-CMP-T2": "执行环境迁移复现：同一 fold 0 只有 608 行，单个种子的差可到 3 个百分点，逐条预测 87–89% 相同",
    "ML-CMP-T3": "执行环境迁移复现：单个种子的差在 −3.3 到 +3.7 个百分点之间，逐条预测 78–86% 相同；单折单种子的数字波动很大",
    "ML-CMP-T4": "执行环境迁移复现；逐条预测 86–90% 相同",
    "ML-CMP-T1": "和 ML-CMP-T2/T3/T4 一起看：四个赛道的 fold 0 上，T4 fp16 都比 A10 bf16 平均高约 1 个百分点。卡、精度、torch 和代码版本同时不同，不能归因到 bf16；要归因需在同一张卡上只改精度补跑",
    "ML-CMP-QT2": "同为 Kaggle T4 的 fp16（两边精度字段的写法不同，所以被归为卡或精度不同）；模型、训练方案都不同",
    "ML-CMP-QT3": "同为 Kaggle T4 的 fp16（两边精度字段的写法不同）；巴斯克语上 Qwen SFT 没有比 mDeBERTa 好，三个种子方向不一",
    "ML-CMP-QT4": "Qwen 在 A10 bf16、mDeBERTa 在 T4 fp16；模型和执行环境都不同",
    "CMP-16": "自动列出的差别是记录方式不同：方法的描述文字、多记的 loss 字段，以及这一侧登记时没读到数据哈希（第六组 ce_results/fingerprint.json 里的数据哈希和 R3 逐项相同）；训练变量相同、同卡同精度，代码版本不同（第六组加了各方法的分支，交叉熵路径没改）。三个种子都比 R3 低，说明重复运行本身就有约 0.4 个百分点的差，和第六组 SCL、MoCo 相对 Q0 的差同量级；单折三种子下这些方法分不出是方法效应还是重复运行的波动",
    "CMP-12": "本轮三种子均提升；仍只有一个留出折，不能用来保证跨折泛化",
    "CMP-13": "dev 仅报告，不据此重选模型",
    "CMP-14": "控制 dropout 后评估 R-Drop 整体训练规则；若要单独归因于 KL，还需双前向平均但 alpha=0 的消融",
    "CMP-15": "留出折和 dev 趋势不同；不按 dev 挑选方法",
    "CMP-10": "仅一个留出折和三个种子；同时引入投影头、目标模型、队列及混合损失，不归因于单个组件",
    "CMP-11": "区间仅针对已训练模型的样本重抽样，不包含训练随机性；不按 dev 选方法",
    "CMP-08": "SCL 同时引入投影头和混合损失，不单独归因于某个组件；种子44下降，需报告波动",
    "CMP-09": "仅三个种子；区间不包含训练随机性，不依据 dev 重选模型",
    "CMP-06": "CE+dropout 已补齐，见 CMP-12 至 CMP-15；本比较不能单独归因于 KL",
    "CMP-07": "CE+dropout 已补齐；区间仅反映已训练模型的样本重抽样，不包含训练随机性",  # 人工补充的核对结果，原样写进比较表
    "CMP-04": "两边零样本的 1,448 个输入概率逐条相同（推理一致）；训练过程是否等价没有验证",
    "CMP-05": "训练变量相同，执行环境不同（A10 bf16、torch 2.6 对 T4 fp16、torch 2.11）；代码版本 sft-4 对 sft-3（v4 加了 PAI-DSW 的支持、ModelScope 下载和权重包）。零样本 1,448 个输入里 16 个最大概率的类不同；三个种子各有 4–5% 的输入最大概率的类不同",
}


def families_for(dico):
    fam = rj(os.path.join(M, "diagnostics", "families.json"))["family_of_pair"]
    dev = rows(os.path.join(dico, "final_data/dev/dico_nli_dev_track1_reference.csv"))
    parent, first = {}, {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    norm = lambda s: " ".join(s.lower().split())
    for r in dev:
        p = r["pair_id"]
        parent.setdefault(p, p)
        k = tuple(sorted((norm(r["text1"]), norm(r["text2"]))))
        if k in first:
            a, b = find(p), find(first[k])
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            first[k] = p
    out = dict(fam)
    out.update({p: "dev:" + find(p) for p in parent})
    return out


def weighted_f1(gold, pred):
    total, out = len(gold), 0.0
    for l in LABELS:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        out += (2 * tp / (2 * tp + fp + fn) if tp else 0.0) * sum(g == l for g in gold) / total
    return out


def gold_for(dico, track, eval_set):
    split = "dev" if eval_set == "dev" else "train"
    return {r["instance_id"]: r["label"] for r in rows(os.path.join(dico, f"final_data/{split}/dico_nli_{split}_track{track}_reference.csv"))}


def compare(dico, fam_of, a, b, eval_set):
    track = a["track"]
    assert track == b["track"]
    seeds = sorted(set(a["evals"][eval_set]) & set(b["evals"][eval_set]))
    if not seeds and ("-" in a["evals"][eval_set] or "-" in b["evals"][eval_set]):  # 零样本只有一份，和每个种子配对
        sa = a["evals"][eval_set]; sb = b["evals"][eval_set]
        pairs = [(s, s if s in sa else "-", s if s in sb else "-") for s in sorted(set(sa) | set(sb)) if s != "-"]
    else:
        pairs = [(s, s, s) for s in seeds]
    gold = gold_for(dico, track, eval_set)
    ids = sorted(set.intersection(*[set(a["evals"][eval_set][x]["sym"]) for _, x, _ in pairs],
                                  *[set(b["evals"][eval_set][y]["sym"]) for _, _, y in pairs]))
    by_fam = {}
    for i in ids:
        by_fam.setdefault(fam_of[i.split("__")[0]], []).append(i)
    fams = sorted(by_fam)

    def diff(sel):
        g = [gold[i] for i in sel]
        return sum(weighted_f1(g, [a["evals"][eval_set][x]["sym"][i] for i in sel]) -
                   weighted_f1(g, [b["evals"][eval_set][y]["sym"][i] for i in sel]) for _, x, y in pairs) / len(pairs)

    per_seed = {s: round(weighted_f1([gold[i] for i in ids], [a["evals"][eval_set][x]["sym"][i] for i in ids]) -
                         weighted_f1([gold[i] for i in ids], [b["evals"][eval_set][y]["sym"][i] for i in ids]), 4) for s, x, y in pairs}
    rng = random.Random(BOOT_SEED)
    # Precompute per-family confusion counts; same Python RNG draws and percentile rule.
    # F1 depends only on these counts, so repeated text-level scans are unnecessary.
    import numpy as np
    label_id = {label:i for i,label in enumerate(LABELS)}
    counts = np.zeros((len(fams),len(pairs),2,4,4),dtype=np.int64)
    for fi,f in enumerate(fams):
        for si,(_,x,y) in enumerate(pairs):
            for i in by_fam[f]:
                gi=label_id[gold[i]]
                counts[fi,si,0,gi,label_id[a["evals"][eval_set][x]["sym"][i]]]+=1
                counts[fi,si,1,gi,label_id[b["evals"][eval_set][y]["sym"][i]]]+=1
    family_index={f:i for i,f in enumerate(fams)}
    boot=[]
    for _ in range(B):
        weights=np.bincount([family_index[f] for f in rng.choices(fams,k=len(fams))],minlength=len(fams))
        cm=np.tensordot(weights,counts,axes=(0,0))
        support=cm.sum(axis=-1); predicted=cm.sum(axis=-2)
        den=support+predicted; tp=np.diagonal(cm,axis1=-2,axis2=-1)
        f1=np.divide(2*tp,den,out=np.zeros_like(den,dtype=float),where=den!=0)
        wf1=(f1*support).sum(axis=-1)/support.sum(axis=-1)
        boot.append(float((wf1[:,0]-wf1[:,1]).mean()))
    boot.sort()
    return {"seeds": [s for s, _, _ in pairs], "rows": len(ids), "families": len(fams), "per_seed": per_seed,
            "mean": round(diff(ids), 4), "ci95": [round(boot[int(.025 * B)], 4), round(boot[int(.975 * B) - 1], 4)]}


def run_scores(dico, r):
    out = {}
    for ev, by_seed in r["evals"].items():
        gold = gold_for(dico, r["track"], ev)
        vals = [weighted_f1([gold[i] for i in p["sym"]], list(p["sym"].values())) for p in by_seed.values()]
        out[ev] = round(sum(vals) / len(vals), 4)
    return out


def main(dico):
    fam_of = families_for(dico)
    runs = {rid: f() for rid, f in RUNS}
    reg = {"runs": [], "comparisons": []}
    for r in runs.values():
        r["scores_sym_wf1_mean"] = run_scores(dico, r)
        reg["runs"].append({k: v for k, v in r.items() if k != "evals"})
    for cid, new, old, ev, purpose in COMPARISONS:
        a, b = runs[new], runs[old]
        va, vb = flat(a["variables"]), flat(b["variables"])
        keys = set(va) | set(vb)
        unknown = sorted(k for k in keys if (k in va and va[k] in UNKNOWN) or (k in vb and vb[k] in UNKNOWN))
        va = {k: va.get(k, "（无此项）") for k in keys}  # 一边没有这一项（例如零样本没有 LoRA）：算作不同，不算未知
        vb = {k: vb.get(k, "（无此项）") for k in keys}
        changed = sorted(k for k in keys if va.get(k) != vb.get(k) and k not in IGNORE_FOR_CONTROL and k not in unknown)
        libs = sorted(k for k in IGNORE_FOR_CONTROL if va.get(k) != vb.get(k) and k not in unknown)
        plat = sorted(k for k in a["platform"] if a["platform"][k] != b["platform"][k])
        card = [k for k in plat if k in CARD_KEYS]
        as_list = lambda v: v if isinstance(v, list) else [v]
        env_unknown = sorted(k for k in a["platform"] if any(x in UNKNOWN or "未记录" in str(x) for x in
                                                              as_list(a["platform"][k]) + as_list(b["platform"][k])))
        code_unknown = any("未记录" in str(x) for x in a["code"]["code_sha256"] + b["code"]["code_sha256"])
        res = compare(dico, fam_of, a, b, ev)
        design = ("重复运行（训练变量相同）" if not changed else "单一计划改动" if len(changed) == 1 else
                  "系统比较（多处不同，不能归因到单个变量）")
        env = ("卡或精度不同" if card else "同卡同精度，软件或平台不同" if plat or libs else "一致")
        missing = unknown + [f"platform.{k}" for k in env_unknown] + (["code_sha256"] if code_unknown else [])
        if missing:
            env += "；信息不全"
        reg["comparisons"].append({"id": cid, "new": new, "old": old, "eval": ev, "purpose": purpose,
                                   "design": design, "environment": env, "unknown": missing, "note": NOTES.get(cid, ""),
                                   "changed": {k: [va.get(k), vb.get(k)] for k in changed},
                                   "lib_diff": {k: [va.get(k), vb.get(k)] for k in libs},
                                   "platform_diff": {k: [a["platform"][k], b["platform"][k]] for k in plat}, "result": res})
    json.dump(reg, open(os.path.join(M, "registry.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    write_md(reg)
    print(f"{len(reg['runs'])} 次运行，{len(reg['comparisons'])} 个比较")


def fmt(v):
    if isinstance(v, list):
        return " / ".join(map(str, v))
    return str(v)


def write_md(reg):
    L = ["# 实验登记表", "",
         "由 [code/registry.py](code/registry.py) 从保存的指纹、运行记录和逐条预测自动生成，不要手改；完整内容在 `metrics/registry.json`。",
         "", "分数是对称化 WF1、所有种子的均值。比较是同种子的 sym WF1 差、种子平均，按家族配对重抽样 2,000 次的近似 95% 区间（每次所有种子用同一批家族）。"
         "\"改了什么\"由脚本对比两边记录下来的变量自动列出，只能发现记录了的差别；没记录的值列为\"未知\"，不当作相同。", "",
         "每个比较分两列归类：**设计**看训练变量差几处（重复运行 / 单一计划改动 / 系统比较）；**执行环境**看平台、卡、精度和软件版本（一致 / 同卡同精度、软件或平台不同 / 卡或精度不同，没记录的另标信息不全）。环境差别只标注、不要求补跑，但结论里要写明。", "",
         "## 运行", "",
         "| 编号 | 内容 | 关键变量 | 执行环境 | 代码 | 已完成的种子 | fold 0 | 5 折折外 | dev |", "|---|---|---|---|---|---|---:|---:|---:|"]
    for r in reg["runs"]:
        v = r["variables"]
        key = []
        if v.get("train"):
            t = v["train"]
            key.append(f"最多 {t['max_epochs']} 轮，lr {t['lr']}")
        if v.get("lora"):
            key.append(f"LoRA r{v['lora']['r']} / α{v['lora']['alpha']} / dropout {v['lora']['dropout']}")
        key.append(v["train_data"])
        p = r["platform"]
        sc = r["scores_sym_wf1_mean"]
        L.append(f"| {r['id']} | {r['desc']} | {'；'.join(key)} | {p['platform']}，{fmt(p['gpu'])}，{fmt(p['dtype'])}，torch {fmt(p['torch'])} | "
                 f"{r['code']['version']}（{fmt(r['code']['code_sha256'])}） | {', '.join(r['seeds'])} | "
                 f"{sc.get('fold0', '—')} | {sc.get('oof', '—')} | {sc.get('dev', '—')} |")
    L += ["", "## 比较", "", "| 编号 | 新 对 旧 | 评测 | 改了什么（自动） | 设计 | 环境的差别 | 执行环境 | 未知 | 逐种子的差 | 平均差 [区间] | 备注 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in reg["comparisons"]:
        ch = "；".join(f"{k}: {fmt(b)} → {fmt(a)}" for k, (a, b) in c["changed"].items())  # 旧 → 新
        if len(c["changed"]) > 4:
            ch = f"{len(c['changed'])} 处（{', '.join(list(c['changed'])[:4])} 等）"
        pl = "；".join([f"{k}: {fmt(b)} → {fmt(a)}" for k, (a, b) in c["platform_diff"].items()] +
                      [f"{k}: {b} → {a}" for k, (a, b) in c["lib_diff"].items()]) or "无"
        r = c["result"]
        L.append(f"| {c['id']} | {c['new']} 对 {c['old']}：{c['purpose']} | {c['eval']}（{r['rows']} 行，{r['families']} 个家族） | {ch or '无'} | "
                 f"{c['design']} | {pl} | {c['environment']} | {', '.join(c['unknown']) or '无'} | "
                 f"{' / '.join(f'{v:+.3f}' for v in r['per_seed'].values())}（种子 {', '.join(r['seeds'])}） | "
                 f"{r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {c['note'] or '—'} |")
    L += ["", "## 怎么用", "",
          "- 新的运行：结果目录放进 `metrics/` 以后，在 `RUNS` 里加一项；新的比较在 `COMPARISONS` 里加一项，然后重新运行脚本。",
          "- 要把差别归因到某个变量，设计必须是\"单一计划改动\"，并在结论里写明执行环境的差别（例如\"同 T4，torch 不同\"）。卡或精度不同时，在同一环境里补跑对照再归因。",
          "- 系统比较（例如换模型）可以写进结论，但只说\"在给定设置下 A 比 B 好\"，不说是哪个变量带来的。",
          "- 零样本概率逐条相同只说明推理一致，不说明训练等价；这类核对写在备注里。",
          "- 区间是对已训练模型、按家族重抽样得到的近似区间，不含训练的随机性；种子少时不下结论。"]
    open(os.path.join(EXP, "实验登记表.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
