# %% [markdown]
# # DiCo-NLI 第一组实验：mBERT 与 mDeBERTa
#
# 只有两个代码单元：先运行配置单元，再运行第二个单元（定义 + 训练 + 打包），也可以直接 Run All。
#
# 把两个短语拼成 `[CLS] text1 [SEP] text2 [SEP]`，微调做四分类。四个赛道，每个模型、每个赛道跑 3 个种子。
#
# - **训练和选轮数**：只用训练集。从训练集里按 pair_id 分出 10% 作内部验证（同一个 pair 的正反两条分到同一边），用它选最好的一轮。
# - **报告**：官方 dev，官方评分脚本，三个指标按赛道分开。
# - **两种预测**：
#   - `plain`：只看 (text1, text2) 这一个方向。
#   - `sym`：再把两句对调算一次，把对调结果的 FORWARD / BACKWARD 概率互换后，和原方向取平均。只用到文本本身，不用 pair_id 的结构。
# - **断点续跑**：每个 (模型, 赛道, 种子) 跑完就写 `done.json`，重新运行会跳过已完成的。换平台时把上一次的 `dico_results_*.zip` 放到工作目录，会自动解压接着跑。
# - **时间估计（T4）**：单语赛道每组约 2–3 分钟，Track 4 每组约 10–15 分钟；两个模型全部跑完约 2 小时。
# - **SMOKE = True**：每个赛道只取 64 条、1 个种子、1 轮，几分钟内查完流程是否通。
#
# 在 Kaggle 上要打开 Internet（克隆官方仓库、下载模型）。Colab、腾讯 Cloud Studio 直接运行即可。

# %%
import os

SMOKE = False
MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "mdeberta": "microsoft/mdeberta-v3-base",
}
TRACKS = [1, 2, 3, 4]
SEEDS = [42, 43, 44]
EPOCHS = 6
LR = {"mbert": 3e-5, "mdeberta": 2e-5}
BATCH_SIZE = 32
MAX_LEN = 96
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
VAL_FRACTION = 0.1
SPLIT_SEED = 2027

WORK = os.path.abspath(os.environ.get("DICO_WORK", "/kaggle/working" if os.path.isdir("/kaggle/working") else "."))
DICO_DIR = os.environ.get("DICO_DIR", os.path.join(WORK, "SemEval-2027-Task-2-DiCo-NLI"))
OUT = os.path.join(WORK, "dico_results")

if SMOKE:
    SEEDS, EPOCHS = SEEDS[:1], 1

# %%
import csv
import glob
import json
import math
import random
import subprocess
import sys
import time
import zipfile

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
SWAP = [0, 2, 1, 3]  # 对调两句后：FORWARD ↔ BACKWARD，其余不变


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
    if os.path.isdir(OUT):
        return
    zips = sorted(glob.glob(os.path.join(WORK, "dico_results_*.zip")) + glob.glob("/kaggle/input/*/dico_results_*.zip"))
    if not zips:
        return
    best = max(zips, key=lambda z: sum(n.endswith("done.json") for n in zipfile.ZipFile(z).namelist()))
    zipfile.ZipFile(best).extractall(WORK)
    print("已从上一次的结果包恢复：", best)


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_track(track):
    d = os.path.join(DICO_DIR, "final_data")
    train = read_rows(os.path.join(d, f"train/dico_nli_train_track{track}_participant_labeled.csv"))
    dev = read_rows(os.path.join(d, f"dev/dico_nli_dev_track{track}_participant_labeled.csv"))
    gold = os.path.join(d, f"dev/dico_nli_dev_track{track}_reference.csv")
    return train, dev, gold


def group_split(rows, fraction, seed):
    pairs = sorted({r["pair_id"] for r in rows})
    random.Random(seed).shuffle(pairs)
    val_pairs = set(pairs[: max(1, int(len(pairs) * fraction))])
    return [r for r in rows if r["pair_id"] not in val_pairs], [r for r in rows if r["pair_id"] in val_pairs]


def write_predictions(path, ids, labels):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "label"])
        w.writerows(zip(ids, labels))


def official_score(gold, pred, out_dir):
    gold, pred, out_dir = (os.path.abspath(p) for p in (gold, pred, out_dir))  # 评分脚本在官方仓库目录下运行
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", gold,
                    "--predictions", pred, "--output-dir", out_dir],
                   cwd=DICO_DIR, check=True, capture_output=True)
    with open(os.path.join(out_dir, "scores.json"), encoding="utf-8") as f:
        s = json.load(f)
    return {k: s[k] for k in ("weighted_f1", "soft_cons", "hard_cons")}


def softmax(row):
    m = max(row)
    e = [math.exp(x - m) for x in row]
    return [x / sum(e) for x in e]


def symmetrize(logits_ab, logits_ba):
    """p_sym(A,B) = (p(A,B) + p(B,A) 对调 FORWARD/BACKWARD) / 2。"""
    out = []
    for la, lb in zip(logits_ab, logits_ba):
        pa, pb = softmax(la), softmax(lb)
        out.append([(pa[i] + pb[SWAP[i]]) / 2 for i in range(4)])
    return out


def argmax_labels(scores):
    return [LABELS[max(range(4), key=lambda i: s[i])] for s in scores]


def weighted_f1(gold, pred):
    total, f1s = len(gold), 0.0
    for l in LABELS:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        f1s += f1 * sum(g == l for g in gold) / total
    return f1s


def train_and_predict(model_tag, track, seed, run_dir):
    """训练一组；返回 dev 上两个方向的 logits。"""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = MODELS[model_tag]
    tok = AutoTokenizer.from_pretrained(name)

    train_rows, dev_rows, _ = load_track(track)
    fit_rows, val_rows = group_split(train_rows, VAL_FRACTION, SPLIT_SEED)
    if SMOKE:
        fit_rows, val_rows, dev_rows = fit_rows[:64], val_rows[:32], dev_rows[:64]

    def batches(rows, shuffle, swap=False):
        def collate(batch):
            a = [r["text2"] if swap else r["text1"] for r in batch]
            b = [r["text1"] if swap else r["text2"] for r in batch]
            enc = tok(a, b, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            if "label" in batch[0]:
                enc["labels"] = torch.tensor([L2I[r["label"]] for r in batch])
            return enc
        g = torch.Generator().manual_seed(seed)
        return DataLoader(rows, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate, generator=g)

    @torch.no_grad()
    def logits_of(model, rows, swap=False):
        model.eval()
        out = []
        for enc in batches(rows, False, swap):
            enc.pop("labels", None)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out.append(model(**enc).logits.float().cpu())
        return torch.cat(out).tolist()

    def run(use_fp16):
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=4)
        model = model.float().to(device)  # 新版 transformers 可能按半精度加载；参数保持 fp32，前向再用 autocast 半精度
        no_decay = ("bias", "LayerNorm.weight", "layernorm.weight")
        params = [
            {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": WEIGHT_DECAY},
            {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(params, lr=LR[model_tag])
        loader = batches(fit_rows, True)
        total = EPOCHS * len(loader)
        sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * total), total)
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
        best, best_state, history = -1.0, None, []
        for epoch in range(EPOCHS):
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
            val_pred = argmax_labels(logits_of(model, val_rows))
            f1 = weighted_f1([r["label"] for r in val_rows], val_pred)
            history.append({"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), "val_weighted_f1": f1,
                            "seconds": round(time.time() - t0, 1)})
            print(f"  epoch {epoch + 1}: loss {history[-1]['train_loss']:.4f}  内部验证 WF1 {f1:.4f}")
            if f1 > best:
                best = f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        return model, history

    result = run(use_fp16=(device == "cuda"))
    fp16 = result is not None and device == "cuda"
    if result is None:
        print("  半精度出现非有限损失，改用 fp32 重跑")
        result = run(use_fp16=False)
    model, history = result
    ab, ba = logits_of(model, dev_rows), logits_of(model, dev_rows, swap=True)
    with open(os.path.join(run_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"model": MODELS[model_tag], "fp16": fp16, "history": history,
                   "fit_rows": len(fit_rows), "val_rows": len(val_rows)}, f, indent=2)
    del model
    torch.cuda.empty_cache()
    return dev_rows, ab, ba


def run_one(model_tag, track, seed):
    run_dir = os.path.join(OUT, model_tag, f"track{track}", f"seed{seed}")
    if os.path.exists(os.path.join(run_dir, "done.json")):
        print(f"跳过（已完成）：{model_tag} track{track} seed{seed}")
        return
    os.makedirs(run_dir, exist_ok=True)
    print(f"开始：{model_tag} track{track} seed{seed}")
    t0 = time.time()
    dev_rows, ab, ba = train_and_predict(model_tag, track, seed, run_dir)
    _, _, gold = load_track(track)
    ids = [r["instance_id"] for r in dev_rows]
    with open(os.path.join(run_dir, "dev_logits.json"), "w", encoding="utf-8") as f:
        json.dump({"instance_id": ids, "ab": ab, "ba": ba}, f)
    scores = {}
    for mode, preds in (("plain", argmax_labels(ab)), ("sym", argmax_labels(symmetrize(ab, ba)))):
        pred_path = os.path.join(run_dir, f"dev_pred_{mode}.csv")
        write_predictions(pred_path, ids, preds)
        if SMOKE:
            scores[mode] = {"weighted_f1": weighted_f1([r["label"] for r in dev_rows], preds)}
        else:
            scores[mode] = official_score(gold, pred_path, os.path.join(run_dir, f"scores_{mode}"))
    with open(os.path.join(run_dir, "done.json"), "w", encoding="utf-8") as f:
        json.dump({"scores": scores, "minutes": round((time.time() - t0) / 60, 1)}, f, indent=2)
    print(f"完成：{model_tag} track{track} seed{seed}  {scores}  用时 {(time.time() - t0) / 60:.1f} 分钟")


def summarize():
    rows = []
    for done in sorted(glob.glob(os.path.join(OUT, "*", "track*", "seed*", "done.json"))):
        parts = done.split(os.sep)
        model_tag, track, seed = parts[-4], parts[-3], parts[-2]
        with open(done, encoding="utf-8") as f:
            d = json.load(f)
        for mode, s in d["scores"].items():
            rows.append({"model": model_tag, "track": track, "seed": seed, "mode": mode, **s})
    table = {}
    for r in rows:
        table.setdefault((r["model"], r["track"], r["mode"]), []).append(r)
    summary = []
    for (m, t, mode), rs in sorted(table.items()):
        item = {"model": m, "track": t, "mode": mode, "n_seeds": len(rs)}
        for k in ("weighted_f1", "soft_cons", "hard_cons"):
            vals = [r[k] for r in rs if k in r]
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                item[k] = round(mean, 4)
                item[k + "_std"] = round(std, 4)
        summary.append(item)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": rows, "summary": summary}, f, indent=2)
    for s in summary:
        print(s)
    return summary


ensure_packages()
ensure_official_repo()
restore_previous_results()
os.makedirs(OUT, exist_ok=True)
for track in TRACKS:
    for model_tag in MODELS:
        for seed in SEEDS:
            run_one(model_tag, track, seed)
summarize()

stamp = time.strftime("%Y%m%d-%H%M")
zip_path = os.path.join(WORK, f"dico_results_{stamp}.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for path in glob.glob(os.path.join(OUT, "**", "*"), recursive=True):
        if os.path.isfile(path):
            z.write(path, os.path.relpath(path, WORK))
print("结果包已保存到：", zip_path)
print("请把这个 zip 下载下来发给 Claude；换平台续跑时，把它放到新平台的工作目录里再运行本 notebook。")
