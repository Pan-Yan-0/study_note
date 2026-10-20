# %% [markdown]
# # DiCo-NLI 第二组实验：英语折外诊断与训练配方比较（mDeBERTa）
#
# 只有两个代码单元：先运行配置单元，再运行第二个单元（定义 + 训练 + 分析 + 打包），也可以直接 Run All。
#
# 这一组只做英语（Track 1）和 mDeBERTa-v3-base，回答两个事先定好的问题：
#
# 1. **单词替换的诊断**（固定用 `e6` 配方）：两侧只差一个词的短语对，模型的判断靠的是被替换的两个词，还是也用到了短语里其余的词？同一个检查点分别输入完整短语和只输入被替换的两个词，列 2×2 配对表。
# 2. **训练配方 e6 对 e12**：最多 6 轮和最多 12 轮，都在内部验证上选轮。学习率按总步数线性衰减，所以这是两个配方的比较，不只是轮数不同。
#
# 两个问题都只用训练集上的**折外预测**来回答：训练集按"家族"分成 5 折，每一折的预测都来自没见过它的模型。官方 dev 只用来报告两个配方的结果，不用来做选择。
#
# **事先定好的决定规则**：3 个种子在折外预测上的配对差值 ΔWF1（12 轮 − 6 轮，对称化后）取平均，平均值大于 0 且按家族自助法的 95% 区间下限也大于 0，就改用 12 轮；否则保留 6 轮。
#
# **和第一组 notebook（`dico_encoders`）的区别**
#
# - **划分**：先把文本相同的 pair_id 合成一个家族（转小写、合并空白、不计两侧顺序），再按家族分组。第一组按 pair_id 分组，内部验证集 176 对里有 11 对和训练部分文本相同。
# - **推理**：相同的有序输入只算一次，按长度排序后成批，用 fp32 推理（训练仍用 fp16 autocast）。第一组在 Track 4 上，同一个输入落在不同批次里时 logits 最多差 0.48，所以改成这样。
# - **选轮**：每个外层训练折里再按家族分出 10% 作内部验证，用它选轮，被预测的那一折完全不参与。
# - **标准差**：汇总里用样本标准差（除以 n − 1）。第一组的 `summary.json` 除以 n。
#
# **规模和时间**：2 个配方 × 5 折 × 3 个种子 = 30 次折外训练，再加 2 个配方 × 3 个种子 = 6 次全量训练（用来报告 dev）。T4 上预计 1.5–2 小时（按第一组每轮约 19 秒估算，未实测）。先跑完全部 `e6` 再跑 `e12`，中途停下也能先拿到诊断结果。
#
# **断点续跑**：每次训练完成都写 `done.json`，重新运行会跳过已完成的。换平台时把上一次的 `dico_diag_results_*.zip` 放到工作目录（Kaggle 上作为数据集挂到 `/kaggle/input`），会自动解压接着跑。设置和数据必须和原来一致，否则会停下并提示。
#
# **SMOKE = True**：每次训练只用 64 条、种子 1 个、轮数 1 / 2，几分钟内查完流程是否通。
#
# 在 Kaggle 上要打开 Internet（克隆官方仓库、下载模型）。Colab、腾讯 Cloud Studio 直接运行即可。

# %%
import os

SMOKE = False
MODEL = "microsoft/mdeberta-v3-base"
TRACK = 1
SEEDS = [42, 43, 44]
FOLDS = 5
RECIPES = {"e6": 6, "e12": 12}  # 最多训练几轮，都按内部验证选轮
PRIMARY = "e6"  # 单词替换诊断事先固定用这个配方
LR = 2e-5
BATCH_SIZE = 32
EVAL_BATCH = 64
MAX_LEN = 96
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
INNER_FRACTION = 0.1
FOLD_SEED = 2027
BOOTSTRAP = 2000
REVIEW_PER_CELL = 15
SAVE_CHECKPOINTS = False  # True 时保存全量训练模型的半精度权重（每个约 0.55 GB，不放进结果包）

WORK = os.path.abspath(os.environ.get("DICO_WORK", "/kaggle/working" if os.path.isdir("/kaggle/working") else "."))
DICO_DIR = os.environ.get("DICO_DIR", os.path.join(WORK, "SemEval-2027-Task-2-DiCo-NLI"))
SMOKE_TRAIN_ROWS = 64
if SMOKE:
    SEEDS, RECIPES, BOOTSTRAP = SEEDS[:1], {"e6": 1, "e12": 2}, 200
OUT = os.path.join(WORK, "dico_diag_smoke" if SMOKE else "dico_diag_results")  # SMOKE 的结果单独放，不会被当成正式结果续跑

# %%
import csv
import glob
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
SWAP = [0, 2, 1, 3]  # 对调两句后：FORWARD ↔ BACKWARD，其余不变
VERSION = "dico_diagnostics-1"
TOKEN = re.compile(r"\w+")


# ---------- 环境与数据 ----------

def ensure_packages():
    try:
        import sentencepiece  # noqa: F401  mDeBERTa 的分词器需要
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "protobuf"], check=True)


def ensure_official_repo():
    if not os.path.isdir(os.path.join(DICO_DIR, "final_data")):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI.git", DICO_DIR], check=True)
    print("官方仓库：", DICO_DIR)


def restore_previous_results():
    if SMOKE or os.path.isdir(OUT):
        return
    zips = sorted(glob.glob(os.path.join(WORK, "dico_diag_results_*.zip"))
                  + glob.glob("/kaggle/input/*/dico_diag_results_*.zip"))
    if not zips:
        return
    best = max(zips, key=lambda z: sum(n.endswith("done.json") for n in zipfile.ZipFile(z).namelist()))
    zipfile.ZipFile(best).extractall(WORK)
    print("已从上一次的结果包恢复：", best)


def data_files():
    d = os.path.join(DICO_DIR, "final_data")
    return {
        "train": os.path.join(d, f"train/dico_nli_train_track{TRACK}_participant_labeled.csv"),
        "train_gold": os.path.join(d, f"train/dico_nli_train_track{TRACK}_reference.csv"),
        "dev": os.path.join(d, f"dev/dico_nli_dev_track{TRACK}_participant_labeled.csv"),
        "dev_gold": os.path.join(d, f"dev/dico_nli_dev_track{TRACK}_reference.csv"),
    }


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fingerprint():
    return {
        "version": VERSION, "model": MODEL, "track": TRACK, "seeds": SEEDS, "folds": FOLDS, "recipes": RECIPES,
        "primary": PRIMARY, "lr": LR, "batch_size": BATCH_SIZE, "eval_batch": EVAL_BATCH, "max_len": MAX_LEN,
        "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY, "inner_fraction": INNER_FRACTION,
        "fold_seed": FOLD_SEED, "smoke": SMOKE, "smoke_train_rows": SMOKE_TRAIN_ROWS if SMOKE else None,
        "data_sha256": {k: sha256(p) for k, p in data_files().items()},
    }


def check_fingerprint():
    fp, path = fingerprint(), os.path.join(OUT, "fingerprint.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        diff = sorted(k for k in set(fp) | set(old) if fp.get(k) != old.get(k))
        if diff:
            raise SystemExit(f"当前设置和已有结果不一致：{diff}。要接着跑，请把设置改回去；要重新开始，请删除或改名 {OUT}。")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fp, f, indent=2)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- 家族、折、切片 ----------

def norm(s):
    return " ".join(s.lower().split())


def build_families(rows):
    """规范化后文本相同（不计两侧顺序）的 pair_id 合成一个家族；家族编号取其中最小的 pair_id。"""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first = {}
    for r in rows:
        p = r["pair_id"]
        parent.setdefault(p, p)
        key = tuple(sorted((norm(r["text1"]), norm(r["text2"]))))
        if key in first:
            a, b = find(p), find(first[key])
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            first[key] = p
    return {p: find(p) for p in parent}


def pick_families(families, fraction, seed):
    fams = sorted(set(families))
    random.Random(seed).shuffle(fams)
    return set(fams[: max(1, int(len(fams) * fraction))])


def assign_folds(rows, fam_of):
    """家族随机排序后依次放进当前行数最少的折，让各折行数接近。"""
    size = Counter(fam_of[r["pair_id"]] for r in rows)
    fams = sorted(size)
    random.Random(FOLD_SEED).shuffle(fams)
    load, fold_of = [0] * FOLDS, {}
    for f in fams:
        k = min(range(FOLDS), key=lambda i: (load[i], i))
        fold_of[f] = k
        load[k] += size[f]
    return fold_of


def shape(a, b):
    """和 code/error_analysis.py 的定义一致。"""
    A, B = Counter(t.lower() for t in TOKEN.findall(a)), Counter(t.lower() for t in TOKEN.findall(b))
    da, db = sum((A - B).values()), sum((B - A).values())
    if not (A & B):
        return "disjoint"
    if da == 0 and db == 0:
        return "same_bag"
    if da == 0 or db == 0:
        return "insert"
    if da == 1 and db == 1:
        return "sub1"
    return "other"


def substitution(a, b):
    """两侧只差一个词时，返回 (text1 里被替换的词, text2 里替换成的词, 这个词在短语里是否出现不止一次)；否则返回 None。"""
    if shape(a, b) != "sub1":
        return None
    ta, tb = TOKEN.findall(a), TOKEN.findall(b)
    A, B = Counter(t.lower() for t in ta), Counter(t.lower() for t in tb)
    wa, wb = next(iter(A - B)), next(iter(B - A))
    sa = [t for t in ta if t.lower() == wa]
    sb = [t for t in tb if t.lower() == wb]
    return sa[-1], sb[-1], len(sa) > 1 or len(sb) > 1


def needed_inputs(rows, with_pairs):
    """要算 logits 的有序输入：每行的两个方向；with_pairs 时再加单词替换行的两个词（两个方向）。"""
    need, seen = [], set()

    def add(x):
        if x not in seen:
            seen.add(x)
            need.append(x)

    for r in rows:
        add((r["text1"], r["text2"]))
        add((r["text2"], r["text1"]))
        if with_pairs:
            sub = substitution(r["text1"], r["text2"])
            if sub:
                add((sub[0], sub[1]))
                add((sub[1], sub[0]))
    return need


# ---------- 概率、预测、指标 ----------

def softmax(row):
    m = max(row)
    e = [math.exp(x - m) for x in row]
    s = sum(e)
    return [x / s for x in e]


def probs_plain(lm, a, b):
    return softmax(lm[(a, b)])


def probs_sym(lm, a, b):
    """p_sym(a,b) = (p(a,b) + p(b,a) 对调 FORWARD/BACKWARD) / 2。两个方向用同一份 logits，所以严格互为对调。"""
    pa, pb = softmax(lm[(a, b)]), softmax(lm[(b, a)])
    return [(pa[i] + pb[SWAP[i]]) / 2 for i in range(4)]


PROBS = {"plain": probs_plain, "sym": probs_sym}


def argmax_label(p):
    return LABELS[max(range(4), key=lambda i: p[i])]


def weighted_f1(gold, pred):
    total, out = len(gold), 0.0
    for l in LABELS:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        out += f1 * sum(g == l for g in gold) / total
    return out


def write_predictions(path, ids, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "label"])
        w.writerows(zip(ids, labels))


def official_score(gold, pred, out_dir):
    gold, pred, out_dir = (os.path.abspath(p) for p in (gold, pred, out_dir))  # 评分脚本在官方仓库目录下运行
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", gold,
                    "--predictions", pred, "--output-dir", out_dir],
                   cwd=DICO_DIR, check=True, capture_output=True)
    s = read_json(os.path.join(out_dir, "scores.json"))
    return {k: s[k] for k in ("weighted_f1", "soft_cons", "hard_cons")}


def mean_std(vals):
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else None
    return {"mean": round(m, 4), "std": None if sd is None else round(sd, 4), "n": len(vals)}


# ---------- 训练（唯一用到 torch 的部分） ----------

_TOKENIZER = None


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
    return _TOKENIZER


def train_and_predict(fit_rows, val_rows, epochs, seed, inputs, checkpoint=None):
    """最多训练 epochs 轮，按内部验证 WF1 选轮（最早的最大值）；返回 inputs 里每个有序输入的 logits。"""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = tokenizer()

    def collate(batch):
        enc = tok([r["text1"] for r in batch], [r["text2"] for r in batch], truncation=True,
                  max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([L2I[r["label"]] for r in batch])
        return enc

    @torch.no_grad()
    def logits_for(model, pairs):
        """相同输入只算一次；按长度排序成批，尽量少补齐；fp32 推理。"""
        model.eval()
        lens = [len(x) for x in tok([a for a, _ in pairs], [b for _, b in pairs], truncation=True,
                                    max_length=MAX_LEN)["input_ids"]]
        order = sorted(range(len(pairs)), key=lambda i: (lens[i], i))
        out = [None] * len(pairs)
        for s in range(0, len(order), EVAL_BATCH):
            idx = order[s: s + EVAL_BATCH]
            enc = tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            for i, row in zip(idx, model(**enc).logits.float().cpu().tolist()):
                out[i] = row
        return out

    val_inputs = list(dict.fromkeys((r["text1"], r["text2"]) for r in val_rows))

    def run(use_fp16):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=4)
        model = model.float().to(device)  # 参数保持 fp32；训练前向用 autocast 半精度
        no_decay = ("bias", "LayerNorm.weight", "layernorm.weight")
        params = [
            {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": WEIGHT_DECAY},
            {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(params, lr=LR)
        loader = DataLoader(fit_rows, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(seed))
        total = epochs * len(loader)
        sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * total), total)
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
        best, best_epoch, best_state, history = -1.0, 0, None, []
        for epoch in range(epochs):
            model.train()
            t0, losses = time.time(), []
            for enc in loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                    loss = model(**enc).loss
                if not torch.isfinite(loss):
                    return None  # 半精度溢出，换 fp32 重跑
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                losses.append(loss.item())
            lm = dict(zip(val_inputs, logits_for(model, val_inputs)))
            f1 = weighted_f1([r["label"] for r in val_rows],
                             [argmax_label(softmax(lm[(r["text1"], r["text2"])])) for r in val_rows])
            history.append({"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), "val_weighted_f1": f1,
                            "seconds": round(time.time() - t0, 1)})
            print(f"    epoch {epoch + 1}: loss {history[-1]['train_loss']:.4f}  内部验证 WF1 {f1:.4f}")
            if f1 > best:
                best, best_epoch = f1, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        return model, history, best_epoch, best_state

    result = run(use_fp16=(device == "cuda"))
    fp16 = result is not None and device == "cuda"
    if result is None:
        print("    半精度出现非有限损失，改用 fp32 重跑")
        result = run(use_fp16=False)
    model, history, best_epoch, best_state = result
    logits = dict(zip(inputs, logits_for(model, inputs)))
    if checkpoint:
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in best_state.items()}, checkpoint)
    env = {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
           "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    del model, best_state, result
    torch.cuda.empty_cache()
    return {"logits": logits, "history": history, "best_epoch": best_epoch, "fp16": fp16, "env": env}


# ---------- 两类训练任务 ----------

def save_logits(path, logits):
    write_json(path, {"inputs": [list(k) for k in logits], "logits": list(logits.values())})


def load_logits(path):
    d = read_json(path)
    return {tuple(k): v for k, v in zip(d["inputs"], d["logits"])}


def oof_dir(recipe, seed, fold):
    return os.path.join(OUT, "oof", recipe, f"seed{seed}", f"fold{fold}")


def dev_dir(recipe, seed):
    return os.path.join(OUT, "dev", recipe, f"seed{seed}")


def limit(fit, val):
    return (fit[:SMOKE_TRAIN_ROWS], val[:32]) if SMOKE else (fit, val)


def run_oof(train, fam_of, fold_of, recipe, seed, fold):
    d = oof_dir(recipe, seed, fold)
    if os.path.exists(os.path.join(d, "done.json")):
        print(f"跳过（已完成）：折外 {recipe} seed{seed} fold{fold}")
        return
    print(f"开始：折外 {recipe} seed{seed} fold{fold}")
    t0 = time.time()
    test = [r for r in train if fold_of[fam_of[r["pair_id"]]] == fold]
    rest = [r for r in train if fold_of[fam_of[r["pair_id"]]] != fold]
    inner = pick_families({fam_of[r["pair_id"]] for r in rest}, INNER_FRACTION, FOLD_SEED * 100 + fold + 1)
    fit, val = limit([r for r in rest if fam_of[r["pair_id"]] not in inner], [r for r in rest if fam_of[r["pair_id"]] in inner])
    res = train_and_predict(fit, val, RECIPES[recipe], seed, needed_inputs(test, with_pairs=True))
    save_logits(os.path.join(d, "logits.json"), res["logits"])
    write_json(os.path.join(d, "train_log.json"), {
        "history": res["history"], "best_epoch": res["best_epoch"], "fp16": res["fp16"], "env": res["env"],
        "fit_rows": len(fit), "val_rows": len(val), "test_rows": len(test), "inner_val_families": sorted(inner)})
    write_json(os.path.join(d, "done.json"), {"best_epoch": res["best_epoch"], "minutes": round((time.time() - t0) / 60, 1)})
    print(f"完成：折外 {recipe} seed{seed} fold{fold}  选中第 {res['best_epoch']} 轮  用时 {(time.time() - t0) / 60:.1f} 分钟")


def run_dev(train, dev, fam_of, recipe, seed):
    d = dev_dir(recipe, seed)
    if os.path.exists(os.path.join(d, "done.json")):
        print(f"跳过（已完成）：全量 {recipe} seed{seed}")
        return
    print(f"开始：全量 {recipe} seed{seed}（报告 dev）")
    t0 = time.time()
    inner = pick_families(set(fam_of.values()), INNER_FRACTION, FOLD_SEED)
    fit, val = limit([r for r in train if fam_of[r["pair_id"]] not in inner], [r for r in train if fam_of[r["pair_id"]] in inner])
    ckpt = os.path.join(d, "model_fp16.pt") if SAVE_CHECKPOINTS else None
    os.makedirs(d, exist_ok=True)
    res = train_and_predict(fit, val, RECIPES[recipe], seed, needed_inputs(dev, with_pairs=False), ckpt)
    save_logits(os.path.join(d, "dev_logits.json"), res["logits"])
    ids, scores = [r["instance_id"] for r in dev], {}
    for mode, fn in PROBS.items():
        path = os.path.join(d, f"dev_pred_{mode}.csv")
        write_predictions(path, ids, [argmax_label(fn(res["logits"], r["text1"], r["text2"])) for r in dev])
        scores[mode] = official_score(data_files()["dev_gold"], path, os.path.join(d, f"scores_{mode}"))
    write_json(os.path.join(d, "train_log.json"), {
        "history": res["history"], "best_epoch": res["best_epoch"], "fp16": res["fp16"], "env": res["env"],
        "fit_rows": len(fit), "val_rows": len(val), "inner_val_families": sorted(inner)})
    write_json(os.path.join(d, "done.json"), {"best_epoch": res["best_epoch"], "scores": scores,
                                              "minutes": round((time.time() - t0) / 60, 1)})
    print(f"完成：全量 {recipe} seed{seed}  dev {scores}  用时 {(time.time() - t0) / 60:.1f} 分钟")


# ---------- 分析 ----------

def oof_logits(recipe, seed, rows, fam_of, fold_of):
    """5 折都完成时，返回每一行所在折的 logits 表（同一个词对可能出现在不同折，不能合并成一张表）；否则返回 None。"""
    by_fold = []
    for fold in range(FOLDS):
        d = oof_dir(recipe, seed, fold)
        if not os.path.exists(os.path.join(d, "done.json")):
            return None
        by_fold.append(load_logits(os.path.join(d, "logits.json")))
    return [by_fold[fold_of[fam_of[r["pair_id"]]]] for r in rows]


def check_swap(rows, row_lm):
    """对称化后，每对反向行的概率必须严格互为对调（反向行和本行在同一折）。"""
    pos = {r["instance_id"]: i for i, r in enumerate(rows)}
    bad = 0
    for i, r in enumerate(rows):
        j = pos.get(r.get("reverse_pair_id") or "")
        if j is None:
            continue
        assert row_lm[i] is row_lm[j]
        q, qr = probs_sym(row_lm[i], r["text1"], r["text2"]), probs_sym(row_lm[j], rows[j]["text1"], rows[j]["text2"])
        bad += any(q[x] != qr[SWAP[x]] for x in range(4))
    return bad


def family_draws(n, B):
    import numpy as np
    rng = np.random.default_rng(0)
    return [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]


def wf1_weighted(gold_idx, pred_idx, w):
    import numpy as np
    cm = np.bincount(gold_idx * 4 + pred_idx, weights=w, minlength=16).reshape(4, 4)
    support, tp = cm.sum(1), np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (support - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros(4), where=denom > 0)
    return float((f1 * support).sum() / support.sum())


def quantiles(xs):
    xs = sorted(xs)
    return [round(xs[int(0.025 * (len(xs) - 1))], 4), round(xs[int(0.975 * (len(xs) - 1))], 4)]


def analyze(train, dev, fam_of, fold_of):
    import numpy as np
    files = data_files()
    gold_train = read_rows(files["train_gold"])  # 和训练文件同序同内容，另外多一列 reverse_pair_id，只用于检查
    assert all(a[k] == b[k] for a, b in zip(gold_train, train) for k in ("instance_id", "pair_id", "text1", "text2", "label"))
    assert len(gold_train) == len(train)
    train = gold_train
    fam_list = sorted(set(fam_of.values()))
    fam_idx = {f: i for i, f in enumerate(fam_list)}
    row_fam = np.array([fam_idx[fam_of[r["pair_id"]]] for r in train])
    gold_idx = np.array([L2I[r["label"]] for r in train])
    draws = family_draws(len(fam_list), BOOTSTRAP)
    summary = {"version": VERSION, "missing": []}

    # 1. 折外预测：每个配方、种子、两种预测的官方三个指标
    oof, preds = {}, {}
    for recipe in RECIPES:
        for seed in SEEDS:
            row_lm = oof_logits(recipe, seed, train, fam_of, fold_of)
            if row_lm is None:
                summary["missing"].append(f"折外 {recipe} seed{seed}")
                continue
            bad = check_swap(train, row_lm)
            assert bad == 0, f"{recipe} seed{seed}: {bad} 行的对称化概率和反向行不互为对调"
            for mode, fn in PROBS.items():
                labels = [argmax_label(fn(lm, r["text1"], r["text2"])) for lm, r in zip(row_lm, train)]
                d = os.path.join(OUT, "oof", recipe, f"seed{seed}")
                path = os.path.join(d, f"oof_pred_{mode}.csv")
                write_predictions(path, [r["instance_id"] for r in train], labels)
                s = official_score(files["train_gold"], path, os.path.join(d, f"scores_{mode}"))
                local = weighted_f1([r["label"] for r in train], labels)
                assert abs(local - s["weighted_f1"]) < 1e-9, (local, s["weighted_f1"])
                oof.setdefault(recipe, {}).setdefault(mode, {})[seed] = s
                preds[(recipe, seed, mode)] = np.array([L2I[l] for l in labels])
    summary["oof_scores"] = {
        recipe: {mode: {"per_seed": by_seed, **{k: mean_std([s[k] for s in by_seed.values()])
                                               for k in ("weighted_f1", "soft_cons", "hard_cons")}}
                 for mode, by_seed in modes.items()}
        for recipe, modes in oof.items()}

    # 2. 配方比较（事先定好的决定规则，只看折外预测）
    if len(RECIPES) == 2:
        base, alt = list(RECIPES)
        seeds = [s for s in SEEDS if all((r, s, m) in preds for r in (base, alt) for m in PROBS)]
        cmp = {"base": base, "alt": alt, "seeds": seeds}
        for mode in PROBS:
            per_seed = {s: {k: round(oof[alt][mode][s][k] - oof[base][mode][s][k], 4)
                            for k in ("weighted_f1", "soft_cons", "hard_cons")} for s in seeds}
            boot = [float(np.mean([wf1_weighted(gold_idx, preds[(alt, s, mode)], w[row_fam])
                                   - wf1_weighted(gold_idx, preds[(base, s, mode)], w[row_fam]) for s in seeds]))
                    for w in draws] if seeds else []
            mean_d = float(np.mean([per_seed[s]["weighted_f1"] for s in seeds])) if seeds else None
            cmp[mode] = {"delta_per_seed": per_seed,
                         "delta_wf1_mean": None if mean_d is None else round(mean_d, 4),
                         "delta_wf1_ci95_family_bootstrap": quantiles(boot) if boot else None}
        if seeds == SEEDS:
            ok = cmp["sym"]["delta_wf1_mean"] > 0 and cmp["sym"]["delta_wf1_ci95_family_bootstrap"][0] > 0
            cmp["decision"] = alt if ok else base
            cmp["rule"] = "折外 ΔWF1（对称化）的种子均值 > 0 且家族自助法 95% 区间下限 > 0 时改用 alt，否则保留 base"
        else:
            cmp["decision"] = None
        summary["recipe_comparison"] = cmp

    # 3. 选中的轮数
    summary["best_epochs"] = {
        recipe: {f"seed{s}": [read_json(os.path.join(oof_dir(recipe, s, k), "done.json"))["best_epoch"]
                              if os.path.exists(os.path.join(oof_dir(recipe, s, k), "done.json")) else None
                              for k in range(FOLDS)] for s in SEEDS} for recipe in RECIPES}

    # 4. dev：只报告，不用来选择
    dev_scores = {}
    for recipe in RECIPES:
        for seed in SEEDS:
            p = os.path.join(dev_dir(recipe, seed), "done.json")
            if not os.path.exists(p):
                summary["missing"].append(f"全量 {recipe} seed{seed}")
                continue
            d = read_json(p)
            for mode, s in d["scores"].items():
                dev_scores.setdefault(recipe, {}).setdefault(mode, {})[seed] = {**s, "best_epoch": d["best_epoch"]}
    summary["dev_scores_report_only"] = {
        recipe: {mode: {"per_seed": by_seed, **{k: mean_std([s[k] for s in by_seed.values()])
                                               for k in ("weighted_f1", "soft_cons", "hard_cons")}}
                 for mode, by_seed in modes.items()}
        for recipe, modes in dev_scores.items()}

    # 5. 单词替换诊断：同一检查点，完整短语 vs 只输入被替换的两个词
    subs = [(i, r, substitution(r["text1"], r["text2"])) for i, r in enumerate(train)]
    subs = [(i, r, s) for i, r, s in subs if s]
    summary["slice"] = {
        "rule": "\\w+ 分词、转小写后，两侧各恰好多出 1 个词且至少有 1 个共同词（和 code/error_analysis.py 的 sub1 相同）",
        "rows": len(subs), "families": len({fam_of[r["pair_id"]] for _, r, _ in subs}),
        "surface_ambiguous_rows": sum(s[2] for _, _, s in subs),
        "shape_counts_rows": dict(Counter(shape(r["text1"], r["text2"]) for r in train)),
    }
    diag = {}
    sub_fam = np.array([fam_idx[fam_of[r["pair_id"]]] for _, r, _ in subs])
    for recipe in RECIPES:
        lms = {s: oof_logits(recipe, s, train, fam_of, fold_of) for s in SEEDS}
        seeds = [s for s in SEEDS if lms[s] is not None]
        if not seeds:
            continue
        diag[recipe] = {"seeds": seeds}
        for mode, fn in PROBS.items():
            full_ok, pair_ok, per_seed = [], [], {}
            for s in seeds:
                cells, by_label, agree, dconf = Counter(), defaultdict(Counter), 0, []
                fo, po = [], []
                for i, r, (wa, wb, _) in subs:
                    lm = lms[s][i]
                    pf, pp = fn(lm, r["text1"], r["text2"]), fn(lm, wa, wb)
                    lf, lp = argmax_label(pf), argmax_label(pp)
                    cf, cp = lf == r["label"], lp == r["label"]
                    cell = {(True, True): "都对", (True, False): "去掉其余词后变错",
                            (False, True): "去掉其余词后变对", (False, False): "都错"}[(cf, cp)]
                    cells[cell] += 1
                    by_label[r["label"]][cell] += 1
                    agree += lf == lp
                    dconf.append(pp[L2I[r["label"]]] - pf[L2I[r["label"]]])
                    fo.append(cf)
                    po.append(cp)
                n = len(subs)
                per_seed[s] = {"cells": dict(cells), "cells_by_gold": {k: dict(v) for k, v in by_label.items()},
                               "acc_full": round(sum(fo) / n, 4), "acc_pair": round(sum(po) / n, 4),
                               "delta_acc": round((sum(po) - sum(fo)) / n, 4), "agreement": round(agree / n, 4),
                               "mean_delta_p_gold": round(sum(dconf) / n, 4)}
                full_ok.append(np.array(fo, dtype=float))
                pair_ok.append(np.array(po, dtype=float))
            diff = np.mean([p - f for p, f in zip(pair_ok, full_ok)], axis=0)  # 每行在各种子上的平均差
            boot = []
            for w in draws:
                rw = w[sub_fam]
                boot.append(float((diff * rw).sum() / rw.sum()) if rw.sum() else 0.0)
            diag[recipe][mode] = {"per_seed": per_seed, "delta_acc_mean": round(float(diff.mean()), 4),
                                  "delta_acc_ci95_family_bootstrap": quantiles(boot)}
    summary["word_pair_diagnostic"] = diag

    # 6. 人工审核样本（事先固定的配方、对称化预测、种子多数票）
    if PRIMARY in diag:
        lms = {s: oof_logits(PRIMARY, s, train, fam_of, fold_of) for s in diag[PRIMARY]["seeds"]}
        rows_out, by_cell = [], defaultdict(set)
        for i, r, (wa, wb, amb) in subs:
            fp = {s: argmax_label(probs_sym(lm[i], r["text1"], r["text2"])) for s, lm in lms.items()}
            pp = {s: argmax_label(probs_sym(lm[i], wa, wb)) for s, lm in lms.items()}
            k = len(lms) / 2
            cf, cp = sum(v == r["label"] for v in fp.values()) > k, sum(v == r["label"] for v in pp.values()) > k
            cell = {(True, True): "都对", (True, False): "去掉其余词后变错",
                    (False, True): "去掉其余词后变对", (False, False): "都错"}[(cf, cp)]
            rows_out.append((r, wa, wb, amb, fp, pp, cell))
            if cell != "都对":
                by_cell[cell].add(fam_of[r["pair_id"]])
        rng, chosen = random.Random(FOLD_SEED), set()
        for cell in sorted(by_cell):
            fams = sorted(by_cell[cell])
            chosen |= set(rng.sample(fams, min(REVIEW_PER_CELL, len(fams))))
        path = os.path.join(OUT, "review_sample.csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["家族", "instance_id", "text1", "text2", "被替换的词", "替换成的词", "原词形不唯一", "金标",
                        *[f"完整_seed{s}" for s in lms], *[f"只给词对_seed{s}" for s in lms], "格子（多数票）",
                        "人工判断（词汇关系 / 搭配或结构 / 歧义 / 疑似标注问题 / 无法判断）", "备注"])
            for r, wa, wb, amb, fp, pp, cell in sorted(rows_out, key=lambda x: (x[6], fam_of[x[0]["pair_id"]], x[0]["instance_id"])):
                if fam_of[r["pair_id"]] in chosen:
                    w.writerow([fam_of[r["pair_id"]], r["instance_id"], r["text1"], r["text2"], wa, wb, int(amb), r["label"],
                                *fp.values(), *pp.values(), cell, "", ""])
        summary["review_sample"] = {"file": "review_sample.csv", "families": len(chosen),
                                    "families_per_cell": {c: len(v) for c, v in by_cell.items()}}

    write_json(os.path.join(OUT, "summary.json"), summary)
    return summary


def fmt(m):
    return f"{m['mean']:.4f}" + (f" ± {m['std']:.4f}" if m["std"] is not None else "") + f"（{m['n']} 个种子）"


def report(summary):
    print("\n========== 结果摘要 ==========")
    for recipe, modes in summary.get("oof_scores", {}).items():
        for mode, s in modes.items():
            print(f"折外 {recipe} {mode}: WF1 {fmt(s['weighted_f1'])}  SoftCons {fmt(s['soft_cons'])}  HardCons {fmt(s['hard_cons'])}")
    cmp = summary.get("recipe_comparison")
    if cmp and cmp.get("seeds"):
        for mode in PROBS:
            print(f"配方比较 {cmp['alt']} − {cmp['base']}（折外，{mode}）：ΔWF1 均值 {cmp[mode]['delta_wf1_mean']}，"
                  f"95% 区间 {cmp[mode]['delta_wf1_ci95_family_bootstrap']}")
        print("按事先定好的规则，采用的配方：", cmp["decision"] or "（种子还没跑全，暂不决定）")
    for recipe, modes in summary.get("dev_scores_report_only", {}).items():
        for mode, s in modes.items():
            print(f"dev（只报告）{recipe} {mode}: WF1 {fmt(s['weighted_f1'])}  SoftCons {fmt(s['soft_cons'])}  HardCons {fmt(s['hard_cons'])}")
    for recipe, d in summary.get("word_pair_diagnostic", {}).items():
        for mode in PROBS:
            if mode in d:
                cells = Counter()
                for v in d[mode]["per_seed"].values():
                    cells.update(v["cells"])
                print(f"单词替换诊断 {recipe} {mode}：只给词对 − 完整短语 的准确率差 {d[mode]['delta_acc_mean']}，"
                      f"95% 区间 {d[mode]['delta_acc_ci95_family_bootstrap']}；各格合计（{len(d['seeds'])} 个种子）{dict(cells)}")
    if summary["missing"]:
        print("还没完成：", "、".join(summary["missing"]))


def pack():
    stamp = time.strftime("%Y%m%d-%H%M")
    zip_path = os.path.join(WORK, f"{os.path.basename(OUT)}_{stamp}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in glob.glob(os.path.join(OUT, "**", "*"), recursive=True):
            if os.path.isfile(path) and not path.endswith(".pt"):
                z.write(path, os.path.relpath(path, WORK))
    return zip_path


def main():
    ensure_packages()
    ensure_official_repo()
    restore_previous_results()
    os.makedirs(OUT, exist_ok=True)
    check_fingerprint()
    files = data_files()
    train, dev = read_rows(files["train"]), read_rows(files["dev"])
    fam_of = build_families(train)
    fold_of = assign_folds(train, fam_of)
    write_json(os.path.join(OUT, "families.json"), {
        "rule": "转小写、合并空白、不计两侧顺序后文本相同的 pair_id 合成一个家族；家族随机排序后依次放进行数最少的折",
        "family_of_pair": fam_of, "fold_of_family": fold_of,
        "rows_per_fold": dict(Counter(fold_of[fam_of[r["pair_id"]]] for r in train))})
    print(f"训练集 {len(train)} 行、{len(set(fam_of))} 个 pair_id、{len(set(fam_of.values()))} 个家族；"
          f"每折行数 {[sum(fold_of[fam_of[r['pair_id']]] == k for r in train) for k in range(FOLDS)]}")
    for recipe in RECIPES:  # 先跑完 e6（诊断用），再跑 e12
        for seed in SEEDS:
            for fold in range(FOLDS):
                run_oof(train, fam_of, fold_of, recipe, seed, fold)
    for recipe in RECIPES:
        for seed in SEEDS:
            run_dev(train, dev, fam_of, recipe, seed)
    summary = analyze(train, dev, fam_of, fold_of)
    report(summary)
    zip_path = pack()
    print("\n结果包已保存到：", zip_path)
    if summary["missing"]:
        print("还有没跑完的部分。重新运行本 notebook 会接着跑；换平台时把这个 zip 放到新平台的工作目录。")
    print("请把这个 zip 下载下来发给 Claude。review_sample.csv 是给人工看的样本，可以先打开看看。")


if __name__ == "__main__":
    main()
