"""实验登记表：从保存的指纹、运行记录和逐条预测自动生成，写出 实验登记表.md 和 metrics/registry.json。

每次运行一行：模型、全部训练变量、数据划分、训练数据量、平台（卡、精度、torch）、代码版本和哈希、种子、分数。
每个比较一行：两边哪些变量不同由脚本自动算出；只差一个训练变量、卡（和随卡定的精度）也相同的才算"控制变量对照"。
平台名称（Kaggle / 腾讯云 / PAI-DSW）、torch 和库版本只标注，不影响判定（Pan-Yan-0 10-22 定的）。
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
        "model": fp["model"], "model_revision": None, "prompt": None,
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


RUNS = [
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
    ("QWEN-ZS-T4", lambda: qwen_run("QWEN-ZS-T4", "qwen_sft_a10", 4, "zero_shot", "阿里云 PAI-DSW", "Qwen 零样本，跨语言")),
    ("QWEN-E3-T4", lambda: qwen_run("QWEN-E3-T4", "qwen_sft_a10", 4, "sft", "阿里云 PAI-DSW", "Qwen SFT 最多 3 轮，跨语言")),
]

# (编号, 新, 旧, 评测集, 目的)
COMPARISONS = [
    ("CMP-01", "MDB-E12-OOF", "MDB-E6-OOF", "oof", "mDeBERTa 训练轮数上限：12 轮 对 6 轮（第二组定配方用的比较）"),
    ("CMP-02", "QWEN-E3-T1", "QWEN-ZS-T1", "fold0", "Qwen 微调 对 不微调"),
    ("CMP-03", "QWEN-E3-T1", "MDB-E12-OOF", "fold0", "Qwen SFT 对 mDeBERTa（同训练 / 验证 / 留出数据；系统对比，模型不同）"),
    ("CMP-04", "QWEN-E5-T1", "QWEN-E3-T1", "fold0", "Qwen SFT 轮数上限：5 轮 对 3 轮"),
]

IGNORE_FOR_CONTROL = {"lib.transformers", "lib.peft", "lib.accelerate", "lib.datasets"}  # 库版本单独列，不算训练变量
CARD_KEYS = ("gpu", "dtype")  # 判定控制变量时只看卡和精度；平台名称和 torch 版本只标注


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
    boot = sorted(diff([i for f in rng.choices(fams, k=len(fams)) for i in by_fam[f]]) for _ in range(B))
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
        changed = sorted(k for k in set(va) | set(vb) if va.get(k) != vb.get(k) and k not in IGNORE_FOR_CONTROL)
        libs = sorted(k for k in IGNORE_FOR_CONTROL if va.get(k) != vb.get(k))
        plat = sorted(k for k in a["platform"] if a["platform"][k] != b["platform"][k])
        card = [k for k in plat if k in CARD_KEYS]
        res = compare(dico, fam_of, a, b, ev)
        kind = ("控制变量对照" if len(changed) == 1 and not card else
                "单变量，但卡不同" if len(changed) == 1 else
                "只差卡（换卡的复现检查）" if not changed and card else
                "设置相同（重复运行）" if not changed else "多变量（系统对比，不能归因到单个变量）")
        reg["comparisons"].append({"id": cid, "new": new, "old": old, "eval": ev, "purpose": purpose, "kind": kind,
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
         "\"改了什么\"由脚本对比两边的变量自动列出：只差一个训练变量、卡（和随卡定的精度）也相同的，才算控制变量对照；平台名称、torch 和库版本只标注，不影响判定。", "",
         "## 运行", "",
         "| 编号 | 内容 | 关键变量 | 平台 | 代码 | 种子 | fold 0 | 5 折折外 | dev |", "|---|---|---|---|---|---|---:|---:|---:|"]
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
    L += ["", "## 比较", "", "| 编号 | 新 对 旧 | 评测 | 改了什么（自动） | 卡 / 平台 / 版本的差别（只有卡影响类型） | 类型 | 逐种子的差 | 平均差 [区间] |",
          "|---|---|---|---|---|---|---|---|"]
    for c in reg["comparisons"]:
        ch = "；".join(f"{k}: {fmt(b)} → {fmt(a)}" for k, (a, b) in c["changed"].items())  # 旧 → 新
        if len(c["changed"]) > 4:
            ch = f"{len(c['changed'])} 处（{', '.join(list(c['changed'])[:4])} 等）"
        pl = "；".join([f"{k}: {fmt(b)} → {fmt(a)}" for k, (a, b) in c["platform_diff"].items()] +
                      [f"{k}: {b} → {a}" for k, (a, b) in c["lib_diff"].items()]) or "无"
        r = c["result"]
        L.append(f"| {c['id']} | {c['new']} 对 {c['old']}：{c['purpose']} | {c['eval']}（{r['rows']} 行，{r['families']} 个家族） | {ch} | {pl} | {c['kind']} | "
                 f"{' / '.join(f'{v:+.3f}' for v in r['per_seed'].values())}（种子 {', '.join(r['seeds'])}） | "
                 f"{r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] |")
    L += ["", "## 怎么用", "",
          "- 新的运行：结果目录放进 `metrics/` 以后，在 `RUNS` 里加一项；新的比较在 `COMPARISONS` 里加一项，然后重新运行脚本。",
          "- 要写进结论的比较，类型应是\"控制变量对照\"。卡不同时，在同一种卡上补跑对照。",
          "- 平台名称、torch 和库版本只标注。同一种卡、不同平台的两次运行，零样本概率应逐条相同（Kaggle 和腾讯云的 T4 已核对过）；不同时要查原因。",
          "- 系统对比（换模型）只回答\"哪个更好\"，不说明是哪个变量带来的。"]
    open(os.path.join(EXP, "实验登记表.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
