# %% [markdown]
# # GMCM 2026 E 题：缺失门控低秩补偿（MG-LoRA）——问题二、问题三 Kaggle 全流程
#
# **怎么跑**
# 1. Kaggle 新建 Notebook → File → Import Notebook，导入本文件；右侧 Accelerator 选 **GPU T4 x2**（或 P100）。
# 2. Add Input：比赛数据集（含 附件2 / 附件3 / 附件4 文件夹，附件1 可以不加）。
#    BERT 二选一：打开右侧 **Internet**（自动下载 bert-base-uncased），或者 Add Input 搜 “bert-base-uncased”。
# 3. 先把下面 `SMOKE = True`，Run All，大约 5–10 分钟，确认每一节都有输出、没有报错；
#    再改成 `SMOKE = False`，用 “Save Version → Save & Run All (Commit)” 后台全量跑（T4 上约 2.5–3 小时）。
# 4. 跑完下载 `/kaggle/working/outputs.zip` 发回即可（检查点不在 zip 里，在 outputs/ckpt 下）。
#
# **方法一句话**：阶段一只用完整输入训练 “LoRA-BERT + 掩码池化双头”；阶段二冻结阶段一，
# 在 BERT 与融合层里加一组 **只在有缺失时才打开的低秩补偿**（门控 g=0 时与阶段一逐位相同），
# 用“完整视角教师”做自蒸馏；缺失模拟只依据题面定义（一个或多个模态中的连续区间），不参考附件3。
#
# **评估**：完整输入 / PM 面板（7 种模态组合 × 3 比例 × 头中尾）/ LH 压力面板（整路关闭）/
# SC 散点缺失（训练分布外）/ 扩展扫描到 100% / 多种子 / 视频配对重抽样 / 消融（常规联合训练、无门控、
# 无教师、散点训练）/ 规则层分组交叉验证 / 错误归因 / 问题三精确 Shapley、逐位置重要性、删除检验、稳定性。

# %% [cfg] 0. 配置
SMOKE = False          # ← 第一次先改成 True 冒烟测试

import os, re, sys, json, math, time, glob, pickle, random, zipfile, hashlib, warnings, copy, shutil, subprocess
from pathlib import Path
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CFG = dict(
    SMOKE=SMOKE,
    INPUT_ROOT="/kaggle/input",
    OUT_DIR="/kaggle/working/outputs",
    ATT2_PATH=None,               # 附件2 aligned_50.pkl（文件）；None 表示自动查找
    ATT3_PATH=None,               # 附件3 对齐版本（文件或文件夹）；None 表示自动查找
    ATT4_PATH=None,               # 附件4 对齐版本（文件或文件夹）；None 表示自动查找
    VIDEO4_DIR=None,              # 附件4 视频目录；None 表示自动查找
    BERT_PATH=None,               # 本地 bert-base-uncased 目录；None 表示自动查找
    BERT_HUB="bert-base-uncased", # 找不到本地权重时从 HuggingFace 下载（需开 Internet）
    CLS_ID=101, SEP_ID=102, T=50,
    # 随机种子：主配置 5 个，消融 3 个；部署种子事先固定，不按结果挑
    SEEDS_MAIN=[17, 29, 43, 71, 97],
    SEEDS_ABL=[17, 29, 43],
    DEPLOY_SEED=17,
    ABLATIONS=["joint", "mg_nogate", "mg_nokd", "mg_scatter"],
    # 阶段一：LoRA 文本 + 池化双头（只用完整输入）
    S1_EPOCHS=12, S1_BS=32, S1_LR_LORA=2e-4, S1_LR_HEAD=1e-3, S1_WD=0.01,
    LORA_R=8, LORA_ALPHA=16, LORA_DROPOUT=0.05,
    HIDDEN=256, FUSE_DIM=64, DROPOUT=0.2,
    CLASS_WEIGHT="sqrt_inv",      # 类别数平方根倒数加权；None 关闭
    # 阶段二：缺失门控 LoRA
    S2_EPOCHS=10, S2_BS=32, S2_LR=5e-4,
    MISS_R_TEXT=4, MISS_R_FUSE=8, MISS_ALPHA=8,
    KD_T=2.0, LAMBDA_KD=0.5, LAMBDA_FEAT=0.5,
    NOGATE_COMPLETE_P=0.2,        # 无门控消融里保留完整样本的比例
    JOINT_MASK_P=0.5,             # 常规联合训练里缺失样本的比例
    # 训练用缺失族（题面定义：一个或多个模态中的连续区间）
    FAM_WHOLE_P=0.1, FAM_R_MIN=0.05, FAM_R_MAX=0.9, FAM_MAX_SEG=4,   # 每个被选模态 1~4 段连续区间
    # 评估面板
    PANEL_RATIOS=[0.1, 0.3, 0.5],
    SC_RATIOS=[0.1, 0.3, 0.5],
    MS_RATIOS=[0.1, 0.3, 0.5], MS_SEGS=3,   # MS：多段短区间面板
    EXT_RATIOS=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    EVAL_BS=256, BOOT_REPS=1000,
    # 规则层
    RULE_CV_FOLDS=5, RULE_MIN_GAIN=0.003, REVIEW_PMAX=0.5, COHERENT_MAX_MAE_LOSS=0.01,
    # 问题三
    N_EXPLAIN_VALID=96, LOCAL_WIN=3, EVID_TOPK=3,
    FAITH_BUDGETS=[0.1, 0.2, 0.3], FAITH_RAND=20, STAB_NOISE=0.05,
    N_TYPICAL=4,                  # 画逐位置重要性图的典型样本数
    ATT4_TIMING="auto",           # auto：先试 MMS 强制对齐（需 Internet），失败则按词位比例；proportional；off
    ATT4_ALIGN_CSV=None,          # 可选：外部词—秒对齐表（列 sample,position,start_s,end_s）
    # 步骤开关
    RUN_TRAIN=True, RUN_ABLATIONS=True, RUN_EXT_SCAN=True, RUN_RULES=True,
    RUN_ATT3=True, RUN_Q3=True, RUN_TEST=True,
)

COMBOS = ["T", "A", "V", "TA", "TV", "AV", "TAV"]
POSITIONS = ["head", "mid", "tail"]
MODS = ["T", "A", "V"]
CLASS_NAMES = ["Negative", "Neutral", "Positive"]

if CFG["SMOKE"]:
    CFG.update(S1_EPOCHS=1, S2_EPOCHS=1, SEEDS_MAIN=[17], SEEDS_ABL=[17], BOOT_REPS=50,
               N_EXPLAIN_VALID=12, FAITH_RAND=3, EXT_RATIOS=[0.5, 1.0], OUT_DIR=CFG["OUT_DIR"] + "_smoke")

OUT = Path(CFG["OUT_DIR"])
for sub in ("", "figures", "ckpt", "runs", "tables", "q3", "att4_keyframes"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(*a):
    msg = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, Path): return str(o)
    return str(o)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


# %% [data] 1. 数据发现与读取（兼容：整文件 train/valid/test、单样本文件、样本字典、npz）
FEAT_EXT = (".pkl", ".pickle", ".pk", ".npz")
ARRAY_FIELDS = dict(text=3, text_bert=3, audio=3, vision=3, regression_labels=1, classification_labels=1,
                    audio_lengths=1, vision_lengths=1)
STR_FIELDS = ("id", "raw_text", "annotations")
ATT_KEYS = dict(att2=["附件2", "attachment2", "attachment_2", "att2"],
                att3=["附件3", "attachment3", "attachment_3", "att3"],
                att4=["附件4", "attachment4", "attachment_4", "att4"])


def is_unaligned(p):
    return ("unaligned" in p.lower()) or ("未对齐" in p)


def att_of(p):
    low = p.lower()
    hit = [k for k, ws in ATT_KEYS.items() if any(w in low for w in ws)]
    return hit[0] if len(hit) == 1 else None


class _NumpyCompatUnpickler(pickle.Unpickler):
    """附件4 的 pkl 是用 numpy 2.x 存的（模块路径 numpy._core.*），Kaggle 镜像是 numpy 1.26（numpy.core.*）。"""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            try:
                __import__(module)
            except ImportError:
                module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_obj(path):
    if path.lower().endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        d = {}
        for k in z.files:
            v = z[k]
            d[k] = v.item() if (v.dtype == object and v.ndim == 0) else v
        return d
    with open(path, "rb") as f:
        try:
            return _NumpyCompatUnpickler(f).load()
        except UnicodeDecodeError:
            f.seek(0)
            return _NumpyCompatUnpickler(f, encoding="latin1").load()


def _is_field_dict(d):
    return isinstance(d, dict) and any(k in d for k in ("audio", "vision", "text_bert"))


def _fix_dims(k, arr, single):
    arr = np.asarray(arr)
    want = ARRAY_FIELDS[k]
    if want == 1:
        return arr.reshape(-1)
    if single and arr.ndim == want - 1:
        arr = arr[None]
    while arr.ndim > want and arr.shape[1] == 1:       # 列表里每个元素是 (1, ...)：挤掉多余的 1
        arr = arr[:, 0]
    return arr


def _as_batch(d, default_id):
    key = next(k for k in ("audio", "vision", "text_bert") if k in d)
    try:
        single = np.asarray(d[key]).ndim == 2           # 单样本文件里存成 (50,74) / (3,50) 的情形
    except Exception:
        single = False
    out = {}
    for k, v in d.items():
        if k in ARRAY_FIELDS:
            try:
                out[k] = _fix_dims(k, v, single)
            except Exception:
                pass
        elif k in STR_FIELDS:
            a = np.asarray(v, dtype=object).reshape(-1)
            out[k] = [x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x) for x in a]
    n = len(out["audio"]) if "audio" in out else len(out["text_bert"])
    if "id" not in out or len(out["id"]) != n:
        out["id"] = [str(default_id)] if n == 1 else [f"{default_id}_{i:03d}" for i in range(n)]
    for k in ("raw_text", "annotations"):
        if k in out and len(out[k]) != n:
            del out[k]
    return out


def _concat_batches(parts):
    parts = [p for p in parts if p]
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    keys = set(parts[0])
    for p in parts[1:]:
        keys &= set(p)
    out = {}
    for k in keys:
        if k in STR_FIELDS:
            out[k] = sum([list(p[k]) for p in parts], [])
        else:
            try:
                out[k] = np.concatenate([np.asarray(p[k]) for p in parts], 0)
            except Exception:
                pass
    return out


def extract_batches(obj, name):
    """任意结构 → {划分名: 字段字典}。划分名为 train/valid/test，或 all。"""
    if isinstance(obj, dict):
        sk = [k for k in ("train", "valid", "test") if k in obj and isinstance(obj[k], dict)]
        if sk:
            return {k: _as_batch(obj[k], f"{name}_{k}") for k in sk}
        if _is_field_dict(obj):
            return {"all": _as_batch(obj, name)}
        if len(obj) == 1:
            return extract_batches(next(iter(obj.values())), name)
        parts = []
        for k in sorted(obj.keys(), key=natural_key):
            v = obj[k]
            if isinstance(v, (dict, list, tuple)):
                parts += list(extract_batches(v, k).values())
        return {"all": _concat_batches(parts)} if parts else {}
    if isinstance(obj, (list, tuple)) and len(obj) and isinstance(obj[0], dict):
        parts = []
        for i, o in enumerate(obj):
            parts += list(extract_batches(o, f"{name}_{i:03d}").values())
        return {"all": _concat_batches(parts)}
    return {}


def _T_of(b):
    return np.asarray(b["audio"]).shape[1] if "audio" in b else np.asarray(b["text_bert"]).shape[-1]


def _stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def load_folder(paths, name):
    """一个附件下的多个特征文件 → 一个字段字典（按自然顺序拼接）。"""
    parts = []
    for p in sorted(paths, key=natural_key):
        try:
            obj = load_obj(p)
        except Exception as e:
            log("   读取失败", p, e); continue
        bs = extract_batches(obj, _stem(p))
        for k, b in bs.items():
            if not b:
                continue
            if _T_of(b) != CFG["T"]:
                log(f"   跳过 {p}（序列长度 {_T_of(b)}，不是对齐版 {CFG['T']}）"); continue
            parts.append(b)
        del obj
    return _concat_batches(parts)


def describe_batch(b):
    return {k: (np.asarray(v).shape if k not in STR_FIELDS else f"list[{len(v)}]") for k, v in b.items()}


_ATT2_CACHE = {}


def discover_inputs():
    root = CFG["INPUT_ROOT"]
    files = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
             if p.lower().endswith(FEAT_EXT) and os.path.isfile(p)]
    n_un = sum(is_unaligned(p) for p in files)
    files = [p for p in files if not is_unaligned(p)]
    log(f"在 {root} 下找到 {len(files) + n_un} 个特征文件（其中未对齐版 {n_un} 个，已跳过）：")
    for p in sorted(files, key=natural_key)[:60]:
        log(f"   {p}  {os.path.getsize(p) / 1e6:.2f} MB")
    if len(files) > 60:
        log(f"   ……共 {len(files)} 个")
    groups = {k: [] for k in ATT_KEYS}
    for p in files:
        a = att_of(p)
        if a:
            groups[a].append(p)
    # 附件2：优先手动指定；否则在附件2目录里找含 train/valid/test 的对齐文件
    att2 = CFG["ATT2_PATH"]
    if att2 is None:
        cands = groups["att2"] or files
        cands = sorted(cands, key=lambda p: (0 if "aligned_50" in os.path.basename(p).lower() else 1, -os.path.getsize(p)))
        for p in cands:
            try:
                obj = load_obj(p)
            except Exception:
                continue
            bs = extract_batches(obj, _stem(p))
            del obj
            if "train" in bs and _T_of(bs["train"]) == CFG["T"]:
                att2 = p
                _ATT2_CACHE[p] = bs
                break
            del bs
        if att2 is None:
            raise FileNotFoundError("没有找到附件2 对齐版（含 train/valid/test 且长度 50）。请设置 CFG['ATT2_PATH']")

    def pick(key, manual):
        if manual:
            if os.path.isdir(manual):
                return [p for p in glob.glob(os.path.join(manual, "**", "*"), recursive=True)
                        if p.lower().endswith(FEAT_EXT) and not is_unaligned(p)]
            return [manual]
        return [p for p in groups[key] if p != att2]

    att3 = pick("att3", CFG["ATT3_PATH"]); att4 = pick("att4", CFG["ATT4_PATH"])
    log("附件2：", att2)
    att3 = sorted(att3, key=natural_key); att4 = sorted(att4, key=natural_key)
    log(f"附件3：{len(att3)} 个文件", [os.path.basename(p) for p in att3[:3]], "..." if len(att3) > 3 else "")
    log(f"附件4：{len(att4)} 个文件", [os.path.basename(p) for p in att4[:3]], "..." if len(att4) > 3 else "")
    return att2, att3, att4


def find_videos4():
    if CFG["VIDEO4_DIR"]:
        roots = [CFG["VIDEO4_DIR"]]
    else:
        roots = [CFG["INPUT_ROOT"]]
    vids = []
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", "*"), recursive=True):
            if not p.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue
            if CFG["VIDEO4_DIR"] or (att_of(p) == "att4" and not is_unaligned(p)):
                vids.append(p)
    return sorted(vids, key=natural_key)


def find_bert():
    if CFG["BERT_PATH"]:
        return CFG["BERT_PATH"]
    for cfgp in glob.glob(os.path.join(CFG["INPUT_ROOT"], "**", "config.json"), recursive=True):
        d = os.path.dirname(cfgp)
        fs = os.listdir(d)
        if "vocab.txt" not in fs or not any(f in fs for f in ("pytorch_model.bin", "model.safetensors")):
            continue
        try:
            c = json.load(open(cfgp))
        except Exception:
            continue
        if c.get("model_type") == "bert" and c.get("hidden_size") == 768 and c.get("vocab_size") == 30522:
            log("使用本地 BERT：", d)
            return d
    log("没找到本地 BERT，改用 HuggingFace：", CFG["BERT_HUB"])
    return CFG["BERT_HUB"]


def _split_text_bert(tb):
    """text_bert → (input_ids, attention_mask, token_type_ids)，自动识别通道顺序。"""
    tb = np.asarray(tb)
    if tb.ndim == 3 and tb.shape[1] != 3 and tb.shape[2] == 3:
        tb = tb.transpose(0, 2, 1)
    tb = np.rint(tb).astype(np.int64)
    chans = [tb[:, c, :] for c in range(3)]
    i_ids = int(np.argmax([c.max() for c in chans]))
    rest = [c for c in range(3) if c != i_ids]
    i_mask = rest[int(np.argmax([chans[c].sum() for c in rest]))]
    i_tt = [c for c in rest if c != i_mask][0]
    return chans[i_ids].copy(), chans[i_mask].copy(), chans[i_tt].copy()


def _longest_run(b):
    n, L = b.shape
    best = np.zeros(n, dtype=np.int64); cur = np.zeros(n, dtype=np.int64)
    for t in range(L):
        cur = np.where(b[:, t], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def _last_true(b):
    """每行最后一个 True 的位置，没有则 -1。"""
    L = b.shape[1]
    has = b.any(1)
    return np.where(has, L - 1 - np.argmax(b[:, ::-1], 1), -1)


def _first_true(b):
    has = b.any(1)
    return np.where(has, np.argmax(b, 1), -1)


_UID = [0]


class MMData:
    """一个划分（或一个专项测试集）的全部数组与掩码。
    词范围 W = 位置 1..sep-1；文本缺失以 W 为参照；声学/视觉缺失以参照区间 R_A、R_V 为参照
    （R = W 按训练集估计的整体偏移平移，正常数据下偏移为 0）。"""

    def __init__(self, d, name, has_label=True):
        _UID[0] += 1; self.uid = _UID[0]
        self.name = name
        audio = np.asarray(d["audio"], dtype=np.float32)
        vision = np.asarray(d["vision"], dtype=np.float32)
        n, T = audio.shape[0], audio.shape[1]
        assert T == CFG["T"], f"{name}: 序列长度为 {T}，本流程只支持对齐版 aligned_50"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        vision = np.nan_to_num(vision, nan=0.0, posinf=0.0, neginf=0.0)
        self.n, self.T = n, T
        self.ids = [str(x) for x in d["id"]]
        self.videos = [s.split("$_$")[0] for s in self.ids]
        if "text_bert" not in d:
            raise KeyError(f"{name} 缺少 text_bert 字段（本流程在线编码文本，需要词元编号）")
        ids, attn, self.token_type = _split_text_bert(d["text_bert"])
        self.raw_text = [str(x) for x in d["raw_text"]] if "raw_text" in d else [""] * n
        nz_text = None
        if "text" in d and d["text"] is not None:
            tx = np.asarray(d["text"], dtype=np.float32)
            if tx.ndim == 3 and tx.shape[1] == T:
                nz_text = np.linalg.norm(tx, axis=2) > 1e-8
            del tx
        nzA = np.linalg.norm(audio, axis=2) > 1e-8
        nzV = np.linalg.norm(vision, axis=2) > 1e-8
        # 结构词元：[CLS] 被置零的补回；找不到 [SEP] 时用 “词元/注意力/声学/视觉” 中最靠后的非零位置推断句长
        self.fix = dict(cls_restored=0, sep_restored=0)
        miss_cls = ids[:, 0] == 0
        ids[miss_cls, 0] = CFG["CLS_ID"]; self.fix["cls_restored"] = int(miss_cls.sum())
        sep = np.full(n, -1, dtype=np.int64)
        last_any = np.max(np.stack([_last_true(ids > 0), _last_true(attn > 0), _last_true(nzA), _last_true(nzV)]), 0)
        for i in range(n):
            w = np.flatnonzero(ids[i] == CFG["SEP_ID"])
            if len(w):
                sep[i] = w[0]
            else:
                sep[i] = int(np.clip(last_any[i] + 1, 1, T - 1))
                if ids[i, sep[i]] == 0:
                    ids[i, sep[i]] = CFG["SEP_ID"]; attn[i, sep[i]] = 1
                self.fix["sep_restored"] += 1
        self.input_ids = ids
        self.sep = sep
        pos = np.arange(T)[None, :]
        self.W = (pos >= 1) & (pos < sep[:, None])
        tmiss = self.W & ((ids == 0) | (attn == 0))
        if nz_text is not None:
            tmiss |= self.W & (~nz_text)
        self.oT = self.W & ~tmiss
        attn = attn.copy(); attn[tmiss] = 0; attn[:, 0] = 1; attn[np.arange(n), sep] = 1
        self.attn = attn
        self.oA, self.oV = nzA, nzV
        self.audio_raw, self.vision_raw = audio, vision
        self.audio = self.vision = None
        self.RT = self.W
        self.RA = self.RV = self.W
        self.has_label = has_label and ("regression_labels" in d)
        self.label_check = {}
        if self.has_label:
            y = np.asarray(d["regression_labels"], dtype=np.float32).reshape(-1)
            self.y = np.clip(y, -3, 3)
            self.c = np.where(y < 0, 0, np.where(y > 0, 2, 1)).astype(np.int64)
            if "annotations" in d:
                m = {"negative": 0, "neutral": 1, "positive": 2}
                a = np.array([m.get(str(x).strip().lower(), -1) for x in d["annotations"]])
                ok = a >= 0
                self.label_check["annotations_agree"] = float((a[ok] == self.c[ok]).mean()) if ok.any() else None
            if "classification_labels" in d:
                cl = np.asarray(d["classification_labels"]).reshape(-1)
                for tag, mp in (("cls_0_1_2", {0: 0, 1: 1, 2: 2}), ("cls_neg1_0_1", {-1: 0, 0: 1, 1: 2})):
                    a = np.array([mp.get(int(round(float(x))), -1) for x in cl])
                    ok = a >= 0
                    if ok.mean() > 0.99:
                        self.label_check[f"classification_labels_agree_{tag}"] = float((a[ok] == self.c[ok]).mean())
        else:
            self.y = None; self.c = None

    def set_ref(self, offs):
        pos = np.arange(self.T)[None, :]
        for m, attr in (("A", "RA"), ("V", "RV")):
            d0, d1 = offs[m]
            lo = 1 + d0; hi = self.sep[:, None] - 1 + d1
            setattr(self, attr, (pos >= lo) & (pos <= hi))

    def o(self, m):
        return {"T": self.oT, "A": self.oA, "V": self.oV}[m]

    def R(self, m):
        return {"T": self.RT, "A": self.RA, "V": self.RV}[m]

    def standardize(self, stats):
        (muA, sdA), (muV, sdV) = stats
        self.audio = ((self.audio_raw - muA) / sdA * self.oA[..., None]).astype(np.float32)
        self.vision = ((self.vision_raw - muV) / sdV * self.oV[..., None]).astype(np.float32)

    def subset(self, idx, name=None):
        s = copy.copy(self)
        _UID[0] += 1; s.uid = _UID[0]
        idx = np.asarray(idx)
        for k in ("input_ids", "token_type", "attn", "sep", "W", "oT", "oA", "oV", "RT", "RA", "RV",
                  "audio_raw", "vision_raw"):
            setattr(s, k, getattr(self, k)[idx])
        if self.audio is not None:
            s.audio = self.audio[idx]; s.vision = self.vision[idx]
        s.ids = [self.ids[i] for i in idx]; s.videos = [self.videos[i] for i in idx]
        s.raw_text = [self.raw_text[i] for i in idx]
        if self.has_label:
            s.y = self.y[idx]; s.c = self.c[idx]
        s.n = len(idx); s.name = name or self.name
        return s


def estimate_offsets(tr):
    """声学/视觉的首末非零位置相对词范围 [1, sep-1] 的典型偏移（众数）。正常对齐数据应为 (0, 0)。"""
    offs, info = {}, {}
    for m in ("A", "V"):
        o = tr.o(m)
        f, l = _first_true(o), _last_true(o)
        ok = f >= 0
        d0 = (f - 1)[ok]; d1 = (l - (tr.sep - 1))[ok]
        res = []
        for d in (d0, d1):
            c = Counter(d.tolist()).most_common(1)
            val, share = (c[0][0], c[0][1] / max(1, len(d))) if c else (0, 0)
            res.append((int(val) if share >= 0.5 else 0, float(share)))
        offs[m] = (res[0][0], res[1][0])
        info[m] = dict(first_offset_mode=res[0][0], first_share=res[0][1], last_offset_mode=res[1][0], last_share=res[1][1])
    log("声学/视觉参照区间偏移（相对词范围）：", info)
    return offs, info


def train_stats(ds):
    out = []
    for x, o in ((ds.audio_raw, ds.oA), (ds.vision_raw, ds.oV)):
        v = x[o]
        mu = v.mean(0) if len(v) else np.zeros(x.shape[2], np.float32)
        sd = v.std(0) if len(v) else np.ones(x.shape[2], np.float32)
        sd = np.where(sd < 1e-6, 1.0, sd)
        out.append((mu.astype(np.float32), sd.astype(np.float32)))
    return out


def load_all():
    p2, p3, p4 = discover_inputs()
    bs = _ATT2_CACHE.pop(p2, None) or extract_batches(load_obj(p2), "att2")
    log("附件2 结构：", {k: describe_batch(v) for k, v in bs.items()})
    data = {}
    for k in ("train", "valid", "test"):
        if k in bs:
            data[k] = MMData(bs[k], k, has_label=True)
    del bs
    for key, paths in (("att3", p3), ("att4", p4)):
        if not paths:
            log(f"未找到 {key}，相关步骤将跳过"); continue
        b = load_folder(paths, key)
        if not b:
            log(f"{key} 读取为空，相关步骤将跳过"); continue
        log(f"{key} 结构：", describe_batch(b))
        data[key] = MMData(b, key, has_label=False)
    offs, off_info = estimate_offsets(data["train"])
    for ds in data.values():
        ds.set_ref(offs)
    stats = train_stats(data["train"])
    for ds in data.values():
        ds.standardize(stats)
    if CFG["SMOKE"]:
        rng = np.random.default_rng(0)
        data["train"] = data["train"].subset(np.sort(rng.choice(data["train"].n, min(256, data["train"].n), replace=False)))
        for k in ("valid", "test"):
            if k in data:
                data[k] = data[k].subset(np.arange(min(96, data[k].n)))
    return data, stats, off_info


def audit(data, off_info=None):
    rows = []
    for k, ds in data.items():
        r = dict(split=k, n=ds.n, words_mean=float(ds.W.sum(1).mean()), sep_restored=ds.fix["sep_restored"],
                 cls_restored=ds.fix["cls_restored"])
        for m in MODS:
            o, R = ds.o(m), ds.R(m)
            miss = R & ~o
            r[f"{m}_obs_mean"] = float(o.sum(1).mean())
            r[f"{m}_samples_with_missing"] = int((miss.sum(1) > 0).sum())
            r[f"{m}_missing_frac_mean"] = float((miss.sum(1) / np.maximum(1, R.sum(1))).mean())
            r[f"{m}_longest_gap_mean"] = float(_longest_run(miss).mean())
            r[f"{m}_empty_samples"] = int((o.sum(1) == 0).sum())
            r[f"{m}_obs_outside_ref"] = int((o & ~R).sum())
        if ds.has_label:
            cnt = np.bincount(ds.c, minlength=3)
            r.update(neg=int(cnt[0]), neu=int(cnt[1]), pos=int(cnt[2]), y_mean=float(ds.y.mean()))
            r.update(ds.label_check)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "data_audit.csv", index=False)
    if off_info:
        save_json(off_info, OUT / "tables" / "ref_offsets.json")
    log("数据审计：\n" + df.T.to_string())
    return df

# %% [masks] 2. 缺失掩码：训练缺失族、PM 面板、LH 压力面板、SC 散点面板、扩展扫描
def _empty_masks(n, T):
    return {m: np.zeros((n, T), dtype=bool) for m in MODS}


def window_mask(o_row, r, pos, allow_clear=False):
    """在一行可观测位置上按“名义比例 r、头/中/尾”截取一段连续窗口（与第二篇的 PM 规则一致）。"""
    s = np.zeros_like(o_row)
    P = np.flatnonzero(o_row)
    if len(P) == 0:
        return s
    if r >= 1.0 and allow_clear:
        s[P] = True
        return s
    if len(P) < 2:
        return s
    f, l = P[0], P[-1]
    L = l - f + 1
    w = min(max(1, int(np.floor(r * L))), L)
    st = f if pos == "head" else (f + (L - w) // 2 if pos == "mid" else l - w + 1)
    s[st:st + w] = True
    s &= o_row
    if not allow_clear and s.sum() >= o_row.sum():      # 不把原本可观测的模态整个删空
        s[P[-1] if pos != "tail" else P[0]] = False
    return s


def build_panel(ds, kind="PM", ratios=None, allow_clear=False):
    """返回 [(bucket_name, combo, ratio, pos, masks_dict)]。
    PM：被遮蔽模态内删一段连续窗口，其余模态不动；
    LH：只保留子集 S，S 以外的模态整路关闭，S 内再删一段连续窗口。"""
    ratios = ratios or CFG["PANEL_RATIOS"]
    cache = {}

    def win(m, r, pos, clear):
        key = (m, r, pos, clear)
        if key not in cache:
            o = ds.o(m)
            cache[key] = np.stack([window_mask(o[i], r, pos, clear) for i in range(ds.n)]) if ds.n else \
                np.zeros((0, ds.T), dtype=bool)
        return cache[key]

    zero = np.zeros((ds.n, ds.T), dtype=bool)
    out = []
    for combo in COMBOS:
        for r in ratios:
            for pos in POSITIONS:
                ms = {}
                for m in MODS:
                    if kind == "PM":
                        ms[m] = win(m, r, pos, allow_clear) if m in combo else zero
                    else:
                        ms[m] = win(m, r, pos, False) if m in combo else ds.o(m)
                out.append((f"{kind}|{combo}|{r}|{pos}", combo, r, pos, ms))
    return out


def build_sc_panel(ds, ratios=None, seed=2026):
    """SC：训练分布外的“散点缺失”——被选模态的每个可观测位置独立以概率 r 丢弃（至少保留 1 个）。"""
    ratios = ratios or CFG["SC_RATIOS"]
    rng = np.random.default_rng(seed)
    zero = np.zeros((ds.n, ds.T), dtype=bool)
    base = {}
    for m in MODS:
        u = rng.random((ds.n, ds.T))
        base[m] = u
    out = []
    for combo in COMBOS:
        for r in ratios:
            ms = {}
            for m in MODS:
                if m not in combo:
                    ms[m] = zero; continue
                o = ds.o(m)
                s = (base[m] < r) & o
                full = s.sum(1) >= o.sum(1)
                for i in np.flatnonzero(full & (o.sum(1) > 0)):
                    s[i, np.flatnonzero(o[i])[0]] = False
                ms[m] = s
            out.append((f"SC|{combo}|{r}|scatter", combo, r, "scatter", ms))
    return out


def build_ms_panel(ds, ratios=None, n_seg=None, seed=2027):
    """MS：多段短区间——被选模态里放 n_seg 段互不重叠的连续区间，总长约为比例 r（每段至少 1 个位置）。"""
    ratios = ratios or CFG["MS_RATIOS"]; n_seg = n_seg or CFG["MS_SEGS"]
    rng = np.random.default_rng(seed)
    zero = np.zeros((ds.n, ds.T), dtype=bool)
    out = []
    for combo in COMBOS:
        for r in ratios:
            ms = {}
            for m in MODS:
                if m not in combo:
                    ms[m] = zero; continue
                o = ds.o(m); s = np.zeros_like(o)
                for i in range(ds.n):
                    P = np.flatnonzero(o[i])
                    if len(P) < 2:
                        continue
                    total = min(max(n_seg, int(round(r * len(P)))), len(P) - 1)
                    k = min(n_seg, total)
                    lens = np.full(k, total // k); lens[: total % k] += 1
                    free = np.ones(len(P), bool)
                    for ln in lens:
                        cand = [j for j in range(len(P) - ln + 1) if free[max(0, j - 1):j + ln + 1].all()]
                        if not cand:
                            break
                        j = cand[rng.integers(len(cand))]
                        s[i, P[j:j + ln]] = True; free[j:j + ln] = False
                ms[m] = s
            out.append((f"MS|{combo}|{r}|multi", combo, r, "multi", ms))
    return out


def sample_family_masks(ds, idx, rng, p_masked=1.0, family="contig"):
    """训练用缺失族。contig：按题面定义（“一个或多个模态中存在随机连续序列区间”），
    在随机选中的模态里删除 1~FAM_MAX_SEG 段互不重叠的连续区间，总比例 U(FAM_R_MIN, FAM_R_MAX)，小概率整路删除；
    scatter（消融用）：删除比例同分布，但位置逐点独立，不成段。两者都只用题面的文字定义，没有参考附件3。"""
    n = len(idx)
    ms = _empty_masks(n, ds.T)
    masked = np.zeros(n, dtype=bool)
    for i, j in enumerate(idx):
        if rng.random() >= p_masked:
            continue
        combo = COMBOS[rng.integers(len(COMBOS))]
        for m in combo:
            o = ds.o(m)[j]
            P = np.flatnonzero(o)
            if len(P) == 0:
                continue
            if rng.random() < CFG["FAM_WHOLE_P"]:
                ms[m][i, P] = True; masked[i] = True
                continue
            if len(P) < 2:
                continue
            r = rng.uniform(CFG["FAM_R_MIN"], CFG["FAM_R_MAX"])
            if family == "scatter":
                s = np.zeros(ds.T, dtype=bool)
                s[P[rng.random(len(P)) < r]] = True
                if s.sum() >= len(P):
                    s[P[0]] = False
                ms[m][i] = s
                masked[i] |= s.any()
                continue
            f, l = P[0], P[-1]
            L = l - f + 1
            total = max(1, int(round(r * L)))
            k = int(rng.integers(1, CFG["FAM_MAX_SEG"] + 1))
            k = max(1, min(k, total))
            cuts = np.sort(rng.choice(np.arange(1, total), k - 1, replace=False)) if k > 1 else np.array([], int)
            lens = np.diff(np.concatenate([[0], cuts, [total]])).astype(int)
            used = np.zeros(ds.T, dtype=bool)
            for ln in rng.permutation(lens):
                for _ in range(20):
                    st = int(rng.integers(f, max(f, l - ln + 1) + 1))
                    seg = np.zeros(ds.T, dtype=bool); seg[st:st + ln] = True
                    grow = seg.copy(); grow[max(0, st - 1):st + ln + 1] = True      # 段与段之间至少隔 1 个位置
                    if not (grow & used).any():
                        used |= seg
                        break
            ms[m][i] = used & o
            masked[i] |= ms[m][i].any()
    return ms, masked


def ref_rows(ds, idx):
    return ds.RT[idx], ds.RA[idx], ds.RV[idx]


def descriptors(aT, aA, aV, R):
    """缺失描述量 φ（9 维）：每个模态 [缺失比例, 最长缺口比例, 是否有缺失]，以各自参照区间为分母。"""
    feats, anys = [], []
    for a, Rm in zip((aT, aA, aV), R):
        Rn = np.maximum(1, Rm.sum(1))
        miss = Rm & ~a
        frac = miss.sum(1) / Rn
        gap = _longest_run(miss) / Rn
        anym = (miss.sum(1) > 0).astype(np.float32)
        feats += [frac, gap, anym]
        anys.append(anym)
    phi = np.stack(feats, 1).astype(np.float32)
    indT = anys[0]
    indF = np.clip(anys[0] + anys[1] + anys[2], 0, 1).astype(np.float32)
    return phi, indT, indF


def apply_masks(ds, idx, ms=None):
    """原生可观测掩码 − 删除掩码 = 实际可用掩码 a；文本删除位置的注意力置 0。"""
    oT, oA, oV = ds.oT[idx], ds.oA[idx], ds.oV[idx]
    attn = ds.attn[idx].copy()
    if ms is not None:
        aT = oT & ~ms["T"]; aA = oA & ~ms["A"]; aV = oV & ~ms["V"]
        attn[ms["T"] & ds.W[idx]] = 0
    else:
        aT, aA, aV = oT, oA, oV
    attn[:, 0] = 1
    return aT, aA, aV, attn


# %% [metrics] 3. 指标与视频配对重抽样
def prior_of(tr):
    """全部模态都缺失时的兜底输出：训练集类别频率与强度中位数。"""
    p = np.bincount(tr.c, minlength=3) / tr.n
    return p.astype(np.float32), float(np.median(tr.y))


def conf_mat(c_true, c_pred, w=None):
    return np.bincount(c_true * 3 + c_pred, weights=w, minlength=9).reshape(3, 3)


def f1_from_cm(C):
    f1 = []
    for k in range(3):
        tp = C[k, k]; fp = C[:, k].sum() - tp; fn = C[k, :].sum() - tp
        d = 2 * tp + fp + fn
        f1.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(f1)), f1


def _weighted_f1_binary(t, p):
    out, tot = 0.0, len(t)
    for k in (0, 1):
        tp = np.sum((p == k) & (t == k)); fp = np.sum((p == k) & (t != k)); fn = np.sum((p != k) & (t == k))
        d = 2 * tp + fp + fn
        out += (np.sum(t == k) / max(1, tot)) * (0.0 if d == 0 else 2 * tp / d)
    return float(out)


def ece_score(probs, c_true, bins=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == c_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def metrics(c_true, y_true, probs, yhat, c_pred=None):
    if c_pred is None:
        c_pred = probs.argmax(1)
    C = conf_mat(c_true, c_pred)
    f1, per = f1_from_cm(C)
    corr = np.nan
    if np.std(yhat) > 1e-9 and np.std(y_true) > 1e-9:
        corr = float(np.corrcoef(yhat, y_true)[0, 1])
    rec = [C[k, k] / C[k].sum() if C[k].sum() else 0.0 for k in range(3)]
    prec = [C[k, k] / C[:, k].sum() if C[:, k].sum() else 0.0 for k in range(3)]
    nz = y_true != 0
    t2 = (y_true[nz] > 0).astype(int); p2 = (yhat[nz] > 0).astype(int)
    y7 = np.clip(np.round(y_true), -3, 3); p7 = np.clip(np.round(yhat), -3, 3)
    y5 = np.clip(np.round(y_true), -2, 2); p5 = np.clip(np.round(yhat), -2, 2)
    return dict(acc=float((c_pred == c_true).mean()), f1=f1, mae=float(np.abs(yhat - y_true).mean()), corr=corr,
                f1_neg=per[0], f1_neu=per[1], f1_pos=per[2], rec_neg=rec[0], rec_neu=rec[1], rec_pos=rec[2],
                prec_neu=prec[1],
                acc2=float((t2 == p2).mean()) if nz.any() else np.nan,
                f1_2w=_weighted_f1_binary(t2, p2) if nz.any() else np.nan,
                acc5=float((y5 == p5).mean()), acc7=float((y7 == p7).mean()),
                ece=ece_score(probs, c_true))


def panel_table(ds, panel_preds):
    rows = []
    for name, combo, r, pos, pr in panel_preds:
        m = metrics(ds.c, ds.y, pr["probs"], pr["yhat"])
        m.update(bucket=name, combo=combo, ratio=r, pos=pos, gateF=float(pr["gF"].mean()), gateT=float(pr["gT"].mean()))
        rows.append(m)
    return pd.DataFrame(rows)


def panel_summary(df, complete=None):
    s = dict(avg_f1=float(df.f1.mean()), avg_acc=float(df.acc.mean()), avg_mae=float(df.mae.mean()),
             avg_corr=float(df["corr"].mean()), avg_acc2=float(df.acc2.mean()), avg_ece=float(df.ece.mean()),
             worst_f1=float(df.f1.min()), worst_bucket=df.loc[df.f1.idxmin(), "bucket"], worst_mae=float(df.mae.max()))
    if complete is not None:
        s["gap_f1"] = complete["f1"] - s["avg_f1"]
        s["gap_mae"] = s["avg_mae"] - complete["mae"]
        s["gap_worst_f1"] = complete["f1"] - s["worst_f1"]
    return s


def paired_bootstrap(ds, predsA, predsB, reps=None, seed=0):
    """按源视频有放回抽样，比较两组 bucket 预测（同一批样本）的平均 F1、平均 MAE 与最差档 F1。"""
    reps = reps or CFG["BOOT_REPS"]
    rng = np.random.default_rng(seed)
    uv, inv = np.unique(np.array(ds.videos), return_inverse=True)
    G = len(uv)
    cA = [p["probs"].argmax(1) if "c" not in p else p["c"] for p in predsA]
    cB = [p["probs"].argmax(1) if "c" not in p else p["c"] for p in predsB]
    yA = [p["yhat"] for p in predsA]; yB = [p["yhat"] for p in predsB]

    def stat(w):
        fA = [f1_from_cm(conf_mat(ds.c, c, w))[0] for c in cA]
        fB = [f1_from_cm(conf_mat(ds.c, c, w))[0] for c in cB]
        ww = np.ones(ds.n) if w is None else w
        mA = [np.sum(ww * np.abs(y - ds.y)) / ww.sum() for y in yA]
        mB = [np.sum(ww * np.abs(y - ds.y)) / ww.sum() for y in yB]
        return [np.mean(fA) - np.mean(fB), np.mean(mA) - np.mean(mB), np.min(fA) - np.min(fB)]

    diffs = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, G, G), minlength=G)[inv].astype(np.float64)
        if w.sum() > 0:
            diffs.append(stat(w))
    d = np.array(diffs); pt = stat(None)
    out = {}
    for k, name in enumerate(["avg_f1", "avg_mae", "worst_f1"]):
        out[name] = dict(point=float(pt[k]), lo=float(np.percentile(d[:, k], 2.5)), hi=float(np.percentile(d[:, k], 97.5)))
    out["groups"] = int(G)
    return out

# %% [model] 4. 模型：LoRA 文本编码 + 掩码池化双头 + 缺失门控 LoRA（MG-LoRA）
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
log("设备：", DEVICE, "| torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0) if USE_AMP else "-")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def autocast():
    return torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else contextlib.nullcontext()


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=USE_AMP)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=USE_AMP)


def load_tokenizer(path):
    return BertTokenizerFast.from_pretrained(path)


class GateHolder:
    """一次前向里暂存每个样本的门控值 g（形状 (B,)），供各层的缺失 LoRA 读取。"""
    def __init__(self):
        self.g = None


class GatedLoRALinear(nn.Module):
    """y = W0 x + s_task·B A x + g·s_miss·B' A' x。
    任务 LoRA（阶段一训练）与缺失 LoRA（阶段二训练）共用一层；g=0 时本层与阶段一逐位相同。"""

    def __init__(self, base, r_task=0, alpha_task=1.0, r_miss=0, alpha_miss=1.0, dropout=0.0, holder=None):
        super().__init__()
        self.base = base
        self.r_task, self.r_miss = r_task, r_miss
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.holder = holder
        fin, fout = base.in_features, base.out_features
        if r_task > 0:
            self.task_A = nn.Parameter(torch.empty(r_task, fin)); nn.init.kaiming_uniform_(self.task_A, a=math.sqrt(5))
            self.task_B = nn.Parameter(torch.zeros(fout, r_task)); self.s_task = alpha_task / r_task
        if r_miss > 0:
            self.miss_A = nn.Parameter(torch.empty(r_miss, fin)); nn.init.kaiming_uniform_(self.miss_A, a=math.sqrt(5))
            self.miss_B = nn.Parameter(torch.zeros(fout, r_miss)); self.s_miss = alpha_miss / r_miss

    def forward(self, x):
        y = self.base(x)
        if self.r_task > 0:
            y = y + F.linear(F.linear(self.drop(x), self.task_A), self.task_B) * self.s_task
        g = self.holder.g if self.holder is not None else None
        if self.r_miss > 0 and g is not None:
            g = g.to(y.dtype).view(-1, *([1] * (x.dim() - 1)))
            y = y + F.linear(F.linear(self.drop(x), self.miss_A), self.miss_B) * (self.s_miss * g)
        return y


def masked_mean_std(x, a):
    """在可用位置上求均值与标准差；可用位置为空时两者都取 0。"""
    x = x.float(); a = a.float().unsqueeze(-1)
    cnt = a.sum(1)
    has = (cnt > 0).float()
    c = cnt.clamp(min=1.0)
    mu = (x * a).sum(1) / c
    var = (((x - mu.unsqueeze(1)) ** 2) * a).sum(1) / c
    sd = torch.sqrt(var + 1e-8) * has
    return torch.cat([mu * has, sd], dim=-1)


class MGModel(nn.Module):
    def __init__(self, dA, dV):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH, add_pooling_layer=False)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.hT, self.hF = GateHolder(), GateHolder()
        for layer in self.bert.encoder.layer:
            att = layer.attention.self
            for nm in ("query", "value"):
                setattr(att, nm, GatedLoRALinear(getattr(att, nm), CFG["LORA_R"], CFG["LORA_ALPHA"],
                                                 CFG["MISS_R_TEXT"], CFG["MISS_ALPHA"], CFG["LORA_DROPOUT"], self.hT))
        Hd = self.bert.config.hidden_size
        din = 2 * (Hd + dA + dV)
        rF, aF = CFG["MISS_R_FUSE"], CFG["MISS_ALPHA"]
        self.fc1 = GatedLoRALinear(nn.Linear(din, CFG["HIDDEN"]), 0, 1, rF, aF, 0.0, self.hF)
        self.fc2 = GatedLoRALinear(nn.Linear(CFG["HIDDEN"], CFG["FUSE_DIM"]), 0, 1, rF, aF, 0.0, self.hF)
        self.head_c = GatedLoRALinear(nn.Linear(CFG["FUSE_DIM"], 3), 0, 1, 4, aF, 0.0, self.hF)
        self.head_r = GatedLoRALinear(nn.Linear(CFG["FUSE_DIM"], 1), 0, 1, 4, aF, 0.0, self.hF)
        self.drop = nn.Dropout(CFG["DROPOUT"])
        self.gate_T = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        self.gate_F = nn.Sequential(nn.Linear(9, 16), nn.ReLU(), nn.Linear(16, 1))
        self.gate_mode = "off"      # off：阶段一/基础模型；gated：MG-LoRA；always：无门控消融

    def gate(self, net, phi, ind):
        if self.gate_mode == "off":
            return None
        if self.gate_mode == "always":
            return torch.ones(phi.shape[0], device=phi.device)
        return torch.sigmoid(net(phi).squeeze(-1)) * ind

    def encode_text(self, ids, attn, tt, aT, phiT, indT):
        self.hT.g = self.gate(self.gate_T, phiT, indT)
        try:
            with autocast():
                hs = self.bert(input_ids=ids, attention_mask=attn, token_type_ids=tt).last_hidden_state
        finally:
            self.hT.g = None
        return masked_mean_std(hs, aT)

    def fuse(self, zT, audio, vision, aA, aV, phi, indF):
        z = torch.cat([zT, masked_mean_std(audio, aA), masked_mean_std(vision, aV)], dim=-1)
        g = self.gate(self.gate_F, phi, indF)
        self.hF.g = g
        try:
            with autocast():
                h = self.drop(F.relu(self.fc1(z)))
                h = self.drop(F.relu(self.fc2(h)))
                logits = self.head_c(h)
                u = self.head_r(h).squeeze(-1)
        finally:
            self.hF.g = None
        return logits.float(), 3 * torch.tanh(u.float()), h.float(), g

    def forward_batch(self, b):
        zT = self.encode_text(b["ids"], b["attn"], b["tt"], b["aT"], b["phiT"], b["indT"])
        return self.fuse(zT, b["audio"], b["vision"], b["aA"], b["aV"], b["phi"], b["indF"])


def _t(x, dt=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dt, device=DEVICE)


def to_batch(ds, idx, ms=None, noise=None):
    """ms：与 idx 行对齐的删除掩码 {'T','A','V'}；noise：(rng, sd) 给声学、视觉可用位置加噪声（稳定性检验用）。"""
    idx = np.asarray(idx)
    aT, aA, aV, attn = apply_masks(ds, idx, ms)
    phi, indT, indF = descriptors(aT, aA, aV, ref_rows(ds, idx))
    audio, vision = ds.audio[idx], ds.vision[idx]
    if noise is not None:
        rng, sd = noise
        audio = audio + rng.normal(0, sd, audio.shape).astype(np.float32) * aA[..., None]
        vision = vision + rng.normal(0, sd, vision.shape).astype(np.float32) * aV[..., None]
    return dict(ids=_t(ds.input_ids[idx], torch.long), attn=_t(attn, torch.long), tt=_t(ds.token_type[idx], torch.long),
                aT=_t(aT), aA=_t(aA), aV=_t(aV), audio=_t(audio), vision=_t(vision),
                phi=_t(phi), phiT=_t(phi[:, :3]), indT=_t(indT), indF=_t(indF),
                emptyT=aT.sum(1) == 0, emptyAll=(aT.sum(1) + aA.sum(1) + aV.sum(1)) == 0)


def load_trainable(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if (not k.startswith("bert.")) or ("task_" in k) or ("miss_" in k)]
    if unexpected or bad:
        raise RuntimeError(f"检查点与模型不匹配：unexpected={unexpected[:5]} missing={bad[:5]}")


class Shared:
    """所有已训练配置共用一个 GPU 上的模型实例：冻结的 BERT 主干完全相同，切换配置只换 LoRA、头部与门控参数。"""
    model = None
    key = None
    dims = None


class TorchPredictor:
    """统一预测接口：predict(ds, idx, ms=None, noise=None) → dict(probs, yhat, gT, gF)。
    完整文本的编码按数据集缓存；只有文本被删改的行才重新过 BERT，文本整段缺失的行直接取 0。"""

    def __init__(self, state=None, gate_mode="off", key=None, model=None, prior=None):
        self.state, self.gate_mode, self.key, self.own = state, gate_mode, key, model
        pp, py = prior if prior is not None else PRIOR
        self.prior_probs = np.asarray(pp, np.float32); self.prior_y = float(py)
        self._cache = {}

    def _model(self):
        if self.own is not None:
            return self.own
        if Shared.model is None:
            Shared.model = MGModel(*Shared.dims).to(DEVICE)
        if Shared.key != self.key:
            load_trainable(Shared.model, self.state)
            Shared.key = self.key
        Shared.model.gate_mode = self.gate_mode
        return Shared.model

    @torch.no_grad()
    def _text_full(self, ds, model):
        if ds.uid not in self._cache:
            zs = []
            for st in range(0, ds.n, CFG["EVAL_BS"]):
                idx = np.arange(st, min(ds.n, st + CFG["EVAL_BS"]))
                b = to_batch(ds, idx)
                z = model.encode_text(b["ids"], b["attn"], b["tt"], b["aT"], b["phiT"], b["indT"])
                z[_t(np.flatnonzero(b["emptyT"]), torch.long)] = 0
                zs.append(z)
            self._cache[ds.uid] = torch.cat(zs) if zs else None
        return self._cache[ds.uid]

    @torch.no_grad()
    def predict(self, ds, idx, ms=None, noise=None):
        model = self._model()
        model.eval()
        idx = np.asarray(idx, dtype=np.int64)
        n = len(idx)
        out = dict(probs=np.zeros((n, 3), np.float32), yhat=np.zeros(n, np.float32),
                   gT=np.zeros(n, np.float32), gF=np.zeros(n, np.float32))
        if n == 0:
            return out
        zfull = self._text_full(ds, model)
        for st in range(0, n, CFG["EVAL_BS"]):
            sl = slice(st, min(n, st + CFG["EVAL_BS"]))
            bidx = idx[sl]
            bms = None if ms is None else {m: ms[m][sl] for m in MODS}
            b = to_batch(ds, bidx, bms, noise)
            zT = zfull[_t(bidx, torch.long)].clone()
            if bms is not None:
                changed = bms["T"].any(1)
                empty = b["emptyT"]
                rows = np.flatnonzero(changed & ~empty)
                if len(rows):
                    ri = _t(rows, torch.long)
                    zT[ri] = model.encode_text(b["ids"][ri], b["attn"][ri], b["tt"][ri], b["aT"][ri],
                                               b["phiT"][ri], b["indT"][ri])
                if empty.any():
                    zT[_t(np.flatnonzero(empty), torch.long)] = 0
            gT = model.gate(model.gate_T, b["phiT"], b["indT"])
            logits, yhat, _, gF = model.fuse(zT, b["audio"], b["vision"], b["aA"], b["aV"], b["phi"], b["indF"])
            p = torch.softmax(logits, -1).cpu().numpy()
            y = yhat.cpu().numpy()
            ea = b["emptyAll"]
            if ea.any():
                p[ea] = self.prior_probs; y[ea] = self.prior_y
            out["probs"][sl] = p; out["yhat"][sl] = y
            if gT is not None: out["gT"][sl] = gT.float().cpu().numpy()
            if gF is not None: out["gF"][sl] = gF.float().cpu().numpy()
        return out


# %% [train] 5. 训练：阶段一（base / joint）与阶段二（mg / mg_nogate / mg_nokd / mg_scatter）
def class_weights(tr):
    cnt = np.bincount(tr.c, minlength=3).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1)) if CFG["CLASS_WEIGHT"] == "sqrt_inv" else np.ones(3)
    return _t(w / w.mean())


def lr_lambda(total, warm):
    def f(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return max(0.0, (total - step) / max(1, total - warm))
    return f


def trainable_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if (not k.startswith("bert.")) or ("task_" in k) or ("miss_" in k)}


def param_counts(model):
    cnt = lambda f: int(sum(p.numel() for n, p in model.named_parameters() if f(n)))
    return dict(total=cnt(lambda n: True), bert_frozen=cnt(lambda n: n.startswith("bert.") and "task_" not in n and "miss_" not in n),
                stage1_task_lora=cnt(lambda n: "task_" in n),
                stage1_heads=cnt(lambda n: n.startswith(("fc1.base", "fc2.base", "head_c.base", "head_r.base"))),
                stage2_miss_lora_text=cnt(lambda n: "miss_" in n and n.startswith("bert.")),
                stage2_miss_lora_fuse=cnt(lambda n: "miss_" in n and not n.startswith("bert.")),
                stage2_gates=cnt(lambda n: n.startswith(("gate_T", "gate_F"))))


def quick_eval(model, data, with_panel):
    pr = TorchPredictor(model=model)
    va = data["valid"]
    full = pr.predict(va, np.arange(va.n))
    m = metrics(va.c, va.y, full["probs"], full["yhat"])
    res = dict(f1=m["f1"], mae=m["mae"], acc=m["acc"])
    if with_panel:
        preds = [(nm, cb, r, pos, pr.predict(va, np.arange(va.n), ms)) for nm, cb, r, pos, ms in PANELS["valid"]["PM"]]
        df = panel_table(va, preds)
        res.update(pm_f1=float(df.f1.mean()), pm_mae=float(df.mae.mean()), pm_worst=float(df.f1.min()))
    return res


def _step(opt, sch, scaler, loss, params):
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update(); sch.step()


def train_stage1(data, seed, mode="base"):
    """mode=base：只用完整输入；mode=joint：常规做法，把缺失样本混进来一起训练（对照）。"""
    set_seed(seed)
    tr = data["train"]
    model = MGModel(tr.audio.shape[2], tr.vision.shape[2]).to(DEVICE)
    model.gate_mode = "off"
    task = [p for n, p in model.named_parameters() if "task_" in n]
    head = [p for n, p in model.named_parameters() if n.startswith(("fc1.base", "fc2.base", "head_c.base", "head_r.base"))]
    for p in task + head:
        p.requires_grad = True
    opt = torch.optim.AdamW([{"params": task, "lr": CFG["S1_LR_LORA"]}, {"params": head, "lr": CFG["S1_LR_HEAD"]}],
                            weight_decay=CFG["S1_WD"])
    bs = CFG["S1_BS"]; steps = CFG["S1_EPOCHS"] * math.ceil(tr.n / bs)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, int(0.1 * steps)))
    scaler = make_scaler(); cw = class_weights(tr); rng = np.random.default_rng(seed)
    best_key, best_state, hist = None, None, []
    for ep in range(CFG["S1_EPOCHS"]):
        model.train(); perm = rng.permutation(tr.n); tl = []
        for st in range(0, tr.n, bs):
            idx = perm[st:st + bs]
            ms = sample_family_masks(tr, idx, rng, p_masked=CFG["JOINT_MASK_P"])[0] if mode == "joint" else None
            logits, yhat, _, _ = model.forward_batch(to_batch(tr, idx, ms))
            loss = F.cross_entropy(logits, _t(tr.c[idx], torch.long), weight=cw) + F.l1_loss(yhat, _t(tr.y[idx]))
            _step(opt, sch, scaler, loss, task + head)
            tl.append(loss.item())
        ev = quick_eval(model, data, with_panel=(mode == "joint"))
        ev.update(epoch=ep + 1, train_loss=float(np.mean(tl)))
        hist.append(ev)
        key = (ev["pm_f1"], -ev["pm_mae"]) if mode == "joint" else (ev["f1"], -ev["mae"])
        log(f"  [{mode} s{seed}] ep{ep + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in ev.items() if isinstance(v, float)))
        if best_key is None or key > best_key:
            best_key, best_state = key, trainable_state(model)
    load_trainable(model, best_state)
    return model, hist, best_state


def train_stage2(data, seed, base_state, variant="mg"):
    """冻结阶段一的全部参数，只训练缺失 LoRA 与门控网络。
    mg：门控 + 完整视角自蒸馏；mg_nogate：门控恒为 1；mg_nokd：去掉教师（蒸馏与表示一致项）；
    mg_scatter：同 mg，但训练缺失改为散点（检验“成段”这一题面约束是否必要）。"""
    set_seed(seed + 1000)
    tr = data["train"]
    model = MGModel(tr.audio.shape[2], tr.vision.shape[2]).to(DEVICE)
    load_trainable(model, base_state)
    for n, p in model.named_parameters():
        p.requires_grad = ("miss_" in n) or n.startswith(("gate_T", "gate_F"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["S2_LR"], weight_decay=0.01)
    bs = CFG["S2_BS"]; steps = CFG["S2_EPOCHS"] * math.ceil(tr.n / bs)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, int(0.1 * steps)))
    scaler = make_scaler(); cw = class_weights(tr); rng = np.random.default_rng(seed + 1000)
    mode = "always" if variant == "mg_nogate" else "gated"
    p_masked = 1.0 - CFG["NOGATE_COMPLETE_P"] if variant == "mg_nogate" else 1.0
    family = "scatter" if variant == "mg_scatter" else "contig"
    lam_kd = 0.0 if variant == "mg_nokd" else CFG["LAMBDA_KD"]
    lam_ft = 0.0 if variant == "mg_nokd" else CFG["LAMBDA_FEAT"]
    Tk = CFG["KD_T"]
    best_key, best_state, hist = None, None, []
    for ep in range(CFG["S2_EPOCHS"]):
        perm = rng.permutation(tr.n); tl = []
        for st in range(0, tr.n, bs):
            idx = perm[st:st + bs]
            ms, _ = sample_family_masks(tr, idx, rng, p_masked=p_masked, family=family)
            with torch.no_grad():
                model.eval(); model.gate_mode = "off"
                lt, yt, ht, _ = model.forward_batch(to_batch(tr, idx))
            model.train(); model.gate_mode = mode
            ls, ys, hs, _ = model.forward_batch(to_batch(tr, idx, ms))
            loss = F.cross_entropy(ls, _t(tr.c[idx], torch.long), weight=cw) + F.l1_loss(ys, _t(tr.y[idx]))
            if lam_kd > 0:
                kd = F.kl_div(F.log_softmax(ls / Tk, -1), F.softmax(lt / Tk, -1), reduction="batchmean") * Tk * Tk
                loss = loss + lam_kd * (kd + F.l1_loss(ys, yt))
            if lam_ft > 0:
                loss = loss + lam_ft * F.mse_loss(hs, ht)
            _step(opt, sch, scaler, loss, params)
            tl.append(loss.item())
        model.gate_mode = mode
        ev = quick_eval(model, data, with_panel=True)
        ev.update(epoch=ep + 1, train_loss=float(np.mean(tl)))
        hist.append(ev)
        log(f"  [{variant} s{seed}] ep{ep + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in ev.items() if isinstance(v, float)))
        key = (ev["pm_f1"], -ev["pm_mae"])
        if best_key is None or key > best_key:
            best_key, best_state = key, trainable_state(model)
    load_trainable(model, best_state)
    model.gate_mode = mode
    return model, hist, best_state


GATE_MODE = dict(base="off", joint="off", mg="gated", mg_nokd="gated", mg_scatter="gated", mg_nogate="always")
TRAIN_KEYS = ["S1_EPOCHS", "S1_BS", "S1_LR_LORA", "S1_LR_HEAD", "S1_WD", "LORA_R", "LORA_ALPHA", "LORA_DROPOUT",
              "HIDDEN", "FUSE_DIM", "DROPOUT", "CLASS_WEIGHT", "S2_EPOCHS", "S2_BS", "S2_LR", "MISS_R_TEXT",
              "MISS_R_FUSE", "MISS_ALPHA", "KD_T", "LAMBDA_KD", "LAMBDA_FEAT", "NOGATE_COMPLETE_P", "JOINT_MASK_P",
              "FAM_WHOLE_P", "FAM_R_MIN", "FAM_R_MAX", "FAM_MAX_SEG", "PANEL_RATIOS", "SMOKE"]


def ckpt_dir(data):
    sig_src = {k: CFG[k] for k in TRAIN_KEYS}
    sig_src["n_train"] = data["train"].n; sig_src["n_valid"] = data["valid"].n
    sig = hashlib.md5(json.dumps(sig_src, sort_keys=True, default=str).encode()).hexdigest()[:8]
    d = OUT / "ckpt" / sig
    d.mkdir(parents=True, exist_ok=True)
    for p in glob.glob(os.path.join(CFG["INPUT_ROOT"], "**", "ckpt", sig, "*.pt"), recursive=True):
        q = d / os.path.basename(p)
        if not q.exists():
            shutil.copy(p, q); log("  续跑：复制已有检查点", p)
    save_json(sig_src, d / "train_config.json")
    return d


def _load_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_training(data):
    """训练（或从检查点续跑）全部配置，返回 {(config, seed): TorchPredictor} 与训练曲线。"""
    tr = data["train"]; Shared.dims = (tr.audio.shape[2], tr.vision.shape[2])
    ck_dir = ckpt_dir(data)
    plan = [("base", s) for s in CFG["SEEDS_MAIN"]] + [("mg", s) for s in CFG["SEEDS_MAIN"]]
    if CFG["RUN_ABLATIONS"]:
        plan += [(v, s) for v in CFG["ABLATIONS"] for s in CFG["SEEDS_ABL"]]
    preds, hists, timing = {}, {}, {}
    for cfg_name, seed in plan:
        ck = ck_dir / f"{cfg_name}_s{seed}.pt"
        if ck.exists():
            log(f"读取已有检查点 {ck.name}")
            state = _load_state(ck)
        else:
            if not CFG["RUN_TRAIN"]:
                raise FileNotFoundError(f"缺少检查点 {ck}，且 RUN_TRAIN=False")
            log(f"==== 训练 {cfg_name} seed {seed} ====")
            t1 = time.time()
            if cfg_name in ("base", "joint"):
                model, hist, state = train_stage1(data, seed, mode=cfg_name)
            else:
                base_ck = ck_dir / f"base_s{seed}.pt"
                if not base_ck.exists():
                    m0, h0, st0 = train_stage1(data, seed, "base"); torch.save(st0, base_ck)
                    pd.DataFrame(h0).to_csv(OUT / "runs" / f"history_base_s{seed}.csv", index=False)
                    del m0
                model, hist, state = train_stage2(data, seed, _load_state(base_ck), cfg_name)
            torch.save(state, ck)
            hists[(cfg_name, seed)] = hist
            timing[f"{cfg_name}_s{seed}"] = time.time() - t1
            pd.DataFrame(hist).to_csv(OUT / "runs" / f"history_{cfg_name}_s{seed}.csv", index=False)
            log(f"  用时 {time.time() - t1:.0f}s")
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        preds[(cfg_name, seed)] = TorchPredictor(state, GATE_MODE[cfg_name], key=(cfg_name, seed))
    save_json(timing, OUT / "tables" / "train_time_seconds.json")
    try:
        pc = param_counts(preds[("mg", CFG["DEPLOY_SEED"])]._model())
        save_json(pc, OUT / "tables" / "param_counts.json"); log("参数量：", pc)
    except Exception as e:
        log("参数量统计失败：", e)
    return preds, hists

# %% [evaluate] 6. 统一评估：完整输入 + PM 面板 + LH 压力面板 + SC 散点面板
PANEL_KINDS = ("PM", "LH", "SC", "MS")


def evaluate_predictor(pred, ds, tag, save=True):
    full = pred.predict(ds, np.arange(ds.n))
    res = dict(full=full)
    if ds.has_label:
        res["complete"] = metrics(ds.c, ds.y, full["probs"], full["yhat"])
    for kind in PANEL_KINDS:
        preds = [(nm, cb, r, pos, pred.predict(ds, np.arange(ds.n), ms)) for nm, cb, r, pos, ms in PANELS[ds.name][kind]]
        res[kind] = preds
        if ds.has_label:
            df = panel_table(ds, preds)
            res[kind + "_df"] = df
            res[kind + "_sum"] = panel_summary(df, res["complete"])
            if save:
                df.to_csv(OUT / "runs" / f"panel_{kind}_{tag}.csv", index=False)
    return res


def evaluate_all(preds, ds):
    results, rows = {}, []
    for (cfg_name, seed), pr in preds.items():
        tag = f"{ds.name}_{cfg_name}_s{seed}"
        r = evaluate_predictor(pr, ds, tag)
        results[(cfg_name, seed)] = r
        row = dict(split=ds.name, config=cfg_name, seed=seed)
        row.update({f"complete_{k}": v for k, v in r["complete"].items()})
        for kind in PANEL_KINDS:
            row.update({f"{kind}_{k}": v for k, v in r[kind + "_sum"].items()})
        rows.append(row)
        log(f"  {tag}: 完整F1={r['complete']['f1']:.4f} MAE={r['complete']['mae']:.4f} | "
            f"PM均F1={r['PM_sum']['avg_f1']:.4f} 最差={r['PM_sum']['worst_f1']:.4f} | "
            f"LH均F1={r['LH_sum']['avg_f1']:.4f} | SC均F1={r['SC_sum']['avg_f1']:.4f} | MS均F1={r['MS_sum']['avg_f1']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / f"summary_runs_{ds.name}.csv", index=False)
    num = [c for c in df.columns if c not in ("split", "config", "seed") and not c.endswith("worst_bucket")]
    agg = df.groupby("config")[num].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.to_csv(OUT / "tables" / f"summary_by_config_{ds.name}.csv")
    return results, df, agg


SEED_METRICS = ["complete_f1", "complete_acc", "complete_mae", "complete_corr", "PM_avg_f1", "PM_avg_mae", "PM_worst_f1",
                "LH_avg_f1", "LH_avg_mae", "SC_avg_f1", "SC_avg_mae", "MS_avg_f1", "MS_avg_mae"]


def seed_level_diffs(df, split):
    """同种子配对：每个种子上 A − B 的差值，给出均值、标准差与 t 分布 95% 区间（种子层面的不确定性）。"""
    from scipy.stats import t as tdist
    pairs = [("mg", "base"), ("mg", "joint"), ("mg", "mg_nogate"), ("mg", "mg_nokd"), ("mg", "mg_scatter"), ("joint", "base")]
    rows = []
    for a, b in pairs:
        A = df[df.config == a].set_index("seed"); B = df[df.config == b].set_index("seed")
        seeds = sorted(set(A.index) & set(B.index))
        if not seeds:
            continue
        for met in SEED_METRICS:
            d = (A.loc[seeds, met] - B.loc[seeds, met]).values.astype(float)
            k = len(d); mu = float(d.mean()); sd = float(d.std(ddof=1)) if k > 1 else float("nan")
            half = float(tdist.ppf(0.975, k - 1) * sd / math.sqrt(k)) if k > 1 else float("nan")
            rows.append(dict(split=split, compare=f"{a} - {b}", metric=met, n_seeds=k, mean=mu, sd=sd,
                             lo=mu - half, hi=mu + half, n_positive=int((d > 0).sum()), per_seed=[round(float(x), 5) for x in d]))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tables" / f"seed_diffs_{split}.csv", index=False)
    return out


def bootstrap_comparisons(results, ds, seed):
    pairs = [("mg", "base"), ("mg", "joint"), ("mg", "mg_nogate"), ("mg", "mg_nokd"), ("mg", "mg_scatter"), ("joint", "base")]
    rows = []
    for a, b in pairs:
        ka, kb = (a, seed), (b, seed)
        if ka not in results or kb not in results:
            continue
        for kind in ("complete",) + PANEL_KINDS:
            if kind == "complete":
                pa = [results[ka]["full"]]; pb = [results[kb]["full"]]
            else:
                pa = [x[4] for x in results[ka][kind]]; pb = [x[4] for x in results[kb][kind]]
            bs = paired_bootstrap(ds, pa, pb, seed=int(hashlib.md5(f"{a}|{b}|{kind}".encode()).hexdigest()[:6], 16))
            for met in ("avg_f1", "avg_mae", "worst_f1"):
                if kind == "complete" and met == "worst_f1":
                    continue
                rows.append(dict(compare=f"{a} - {b}", panel=kind, metric=met, **bs[met], groups=bs["groups"]))
    df = pd.DataFrame(rows)
    if len(df):
        df["excludes_zero"] = (df.lo > 0) | (df.hi < 0)
    df.to_csv(OUT / "tables" / f"bootstrap_{ds.name}.csv", index=False)
    log(f"视频配对重抽样（{ds.name}）：\n" + (df.to_string() if len(df) else "无"))
    return df


def law_tables(results, key, ds, name):
    """缺失规律：按被遮蔽模态、比例、位置分组平均；另给 “模态 × 比例” 与 “模态 × 位置” 交叉表。"""
    tabs = []
    for kind in PANEL_KINDS:
        df = results[key][kind + "_df"]
        facs = ["combo", "ratio"] + (["pos"] if kind in ("PM", "LH") else [])
        for fac in facs:
            g = df.groupby(fac)[["f1", "acc", "mae", "corr", "gateF", "gateT"]].mean().reset_index()
            g.insert(0, "factor", fac); g.insert(0, "panel", kind); g = g.rename(columns={fac: "level"})
            tabs.append(g)
        for f2 in (["ratio", "pos"] if kind in ("PM", "LH") else ["ratio"]):
            g = df.groupby(["combo", f2])[["f1", "mae"]].mean().reset_index()
            g["level"] = g["combo"] + "|" + g[f2].astype(str); g["factor"] = f"combo×{f2}"; g["panel"] = kind
            tabs.append(g[["panel", "factor", "level", "f1", "mae"]])
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(OUT / "tables" / f"law_{name}_{ds.name}.csv", index=False)
    return tab


def extended_scan(pred, ds, name):
    """扩展扫描：比例扩展到 1.0（整模态删空），看 “全有或全无” 现象。"""
    panel = build_panel(ds, "PM", ratios=CFG["EXT_RATIOS"], allow_clear=True)
    preds = [(nm, cb, r, pos, pred.predict(ds, np.arange(ds.n), ms)) for nm, cb, r, pos, ms in panel]
    df = panel_table(ds, preds)
    df.to_csv(OUT / "tables" / f"ext_scan_{name}_{ds.name}.csv", index=False)
    return df


def gate_analysis(results, key, ds):
    """门控值与实际缺失程度的关系：门控能否读懂 “缺得有多严重”。"""
    rows = []
    for kind in ("PM", "SC", "MS"):
        for (nm, cb, r, pos, ms), (_, _, _, _, pr) in zip(PANELS[ds.name][kind], results[key][kind]):
            aT, aA, aV, _ = apply_masks(ds, np.arange(ds.n), ms)
            phi, _, _ = descriptors(aT, aA, aV, ref_rows(ds, np.arange(ds.n)))
            rows.append(pd.DataFrame(dict(panel=kind, bucket=nm, combo=cb, ratio=r, pos=pos,
                                          miss_T=phi[:, 0], miss_A=phi[:, 3], miss_V=phi[:, 6],
                                          gap_T=phi[:, 1], gap_A=phi[:, 4], gap_V=phi[:, 7],
                                          gateT=pr["gT"], gateF=pr["gF"])))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT / "tables" / f"gate_analysis_{ds.name}.csv", index=False)
    mm = df[["miss_T", "miss_A", "miss_V"]].max(1)
    corr = dict(gateF_vs_missmax=float(np.corrcoef(mm, df.gateF)[0, 1]) if df.gateF.std() > 1e-9 else None,
                gateT_vs_missT=float(np.corrcoef(df.miss_T, df.gateT)[0, 1]) if df.gateT.std() > 1e-9 else None,
                gateF_mean_by_combo=df.groupby("combo").gateF.mean().round(4).to_dict())
    return df, corr


# %% [rules] 7. 规则层：强度校准、中性判定、极性—强度一致性（验证集按视频分组交叉验证后再决定是否启用）
def fit_calib(yhat, y):
    a, b = np.polyfit(yhat.astype(np.float64), y.astype(np.float64), 1)
    return float(a), float(b)


def apply_calib(yhat, ab):
    return np.clip(ab[0] * yhat + ab[1], -3, 3) if ab is not None else yhat


def neutral_rule(probs, ycal, prm):
    """中性判定：非中性但概率不高且强度接近 0 → 改判中性；判为中性但强度明显偏离 0 → 按符号改判。"""
    c = probs.argmax(1)
    if prm is None:
        return c
    th, tau, tau2 = prm
    pmax = probs.max(1)
    c2 = c.copy()
    c2[(c != 1) & (pmax < th) & (np.abs(ycal) < tau)] = 1
    if tau2 is not None:
        m = (c == 1) & (np.abs(ycal) > tau2)
        c2[m] = np.where(ycal[m] > 0, 2, 0)
    return c2


def fit_neutral(probs, ycal, c_true):
    best = (f1_from_cm(conf_mat(c_true, probs.argmax(1)))[0], None)
    for th in np.arange(0.35, 0.91, 0.05):
        for tau in np.arange(0.05, 1.01, 0.05):
            for tau2 in [None] + list(np.arange(0.3, 1.51, 0.1)):
                f = f1_from_cm(conf_mat(c_true, neutral_rule(probs, ycal, (th, tau, tau2))))[0]
                if f > best[0] + 1e-9:
                    best = (f, (float(th), float(tau), None if tau2 is None else float(tau2)))
    return best[1]


def coherent_intensity(c, y, eps):
    """题面定义：强度 [-3,0) 负向、0 中性、(0,3] 正向。让输出的强度与极性自洽。"""
    y = y.copy()
    y[c == 1] = 0.0
    y[c == 0] = np.minimum(y[c == 0], -eps)
    y[c == 2] = np.maximum(y[c == 2], eps)
    return y


def _variants(probs, yhat, calib, neutral, eps):
    ycal = apply_calib(yhat, calib)
    c0 = probs.argmax(1); cn = neutral_rule(probs, ycal, neutral)
    return {"raw": (c0, yhat), "calib": (c0, ycal), "calib+neutral": (cn, ycal),
            "raw+coherent": (c0, coherent_intensity(c0, yhat, eps)),
            "calib+coherent": (c0, coherent_intensity(c0, ycal, eps)),
            "calib+neutral+coherent": (cn, coherent_intensity(cn, ycal, eps))}


def rules_cv(full, pm_preds, ds, eps):
    """按源视频分组 K 折：折内拟合（只用完整输入的预测），折外评估完整输入与 PM 面板上的效果。"""
    vids = np.array(ds.videos); uv = np.unique(vids)
    rng = np.random.default_rng(2026); rng.shuffle(uv)
    folds = np.array_split(uv, CFG["RULE_CV_FOLDS"])
    names = list(_variants(full["probs"][:1], full["yhat"][:1], None, None, eps).keys())
    oof_c = {v: np.zeros(ds.n, np.int64) for v in names}; oof_y = {v: np.zeros(ds.n, np.float32) for v in names}
    pm_c = [{v: np.zeros(ds.n, np.int64) for v in names} for _ in pm_preds]
    pm_y = [{v: np.zeros(ds.n, np.float32) for v in names} for _ in pm_preds]
    for fv in folds:
        te = np.isin(vids, fv); trm = ~te
        ab = fit_calib(full["yhat"][trm], ds.y[trm])
        prm = fit_neutral(full["probs"][trm], apply_calib(full["yhat"][trm], ab), ds.c[trm])
        for v, (c, y) in _variants(full["probs"][te], full["yhat"][te], ab, prm, eps).items():
            oof_c[v][te] = c; oof_y[v][te] = y
        for b, pr in enumerate(pm_preds):
            for v, (c, y) in _variants(pr["probs"][te], pr["yhat"][te], ab, prm, eps).items():
                pm_c[b][v][te] = c; pm_y[b][v][te] = y
    rows = []
    for v in names:
        mc = metrics(ds.c, ds.y, full["probs"], oof_y[v], c_pred=oof_c[v])
        pmm = [metrics(ds.c, ds.y, pr["probs"], pm_y[b][v], c_pred=pm_c[b][v]) for b, pr in enumerate(pm_preds)]
        rows.append(dict(variant=v, complete_f1=mc["f1"], complete_acc=mc["acc"], complete_mae=mc["mae"],
                         complete_corr=mc["corr"], complete_rec_neu=mc["rec_neu"],
                         PM_avg_f1=float(np.mean([m["f1"] for m in pmm])) if pmm else np.nan,
                         PM_avg_mae=float(np.mean([m["mae"] for m in pmm])) if pmm else np.nan))
    return pd.DataFrame(rows).set_index("variant")


class RuleLayer:
    def __init__(self, eps):
        self.calib = None; self.neutral = None; self.coherent = False; self.eps = eps; self.info = {}

    def fit(self, full, pm_preds, ds):
        cv = rules_cv(full, pm_preds, ds, self.eps)
        cv.to_csv(OUT / "tables" / "rules_cv.csv")
        log("规则层交叉验证（折外）：\n" + cv.round(4).to_string())
        raw, cal, neu = cv.loc["raw"], cv.loc["calib"], cv.loc["calib+neutral"]
        use_cal = (cal.complete_mae < raw.complete_mae - 1e-4) and not (cal.PM_avg_mae > raw.PM_avg_mae + 0.002)
        base = cal if use_cal else raw
        use_neu = (neu.complete_f1 >= base.complete_f1 + CFG["RULE_MIN_GAIN"]) and not (neu.PM_avg_f1 < base.PM_avg_f1 - 0.002)
        key = ("calib" if use_cal else "raw") if not use_neu else "calib+neutral"
        coh_key = {"raw": "raw+coherent", "calib": "calib+coherent", "calib+neutral": "calib+neutral+coherent"}[key]
        use_coh = cv.loc[coh_key].complete_mae <= cv.loc[key].complete_mae + CFG["COHERENT_MAX_MAE_LOSS"]
        if use_cal or use_neu:
            self.calib = fit_calib(full["yhat"], ds.y)
        if use_neu:
            self.neutral = fit_neutral(full["probs"], apply_calib(full["yhat"], self.calib), ds.c)
        self.use_cal_output = bool(use_cal or use_neu)   # 与交叉验证中评估的变体保持一致
        self.coherent = bool(use_coh)
        chosen = coh_key if use_coh else key
        self.info = dict(use_calib=bool(use_cal), use_neutral=bool(use_neu), use_coherent=bool(use_coh),
                         chosen_variant=chosen, calib=self.calib, neutral=self.neutral, eps=self.eps,
                         cv=cv.round(5).reset_index().to_dict("records"))
        log("规则层决定：", {k: v for k, v in self.info.items() if k != "cv"})
        return self

    def apply(self, probs, yhat):
        ycal = apply_calib(yhat, self.calib)
        c = neutral_rule(probs, ycal, self.neutral)
        y = ycal if self.use_cal_output else yhat
        y_out = coherent_intensity(c, y, self.eps) if self.coherent else y
        return c, y_out, y


def review_flags(full, shap_df):
    """人工复核标记：最大类别概率 < 阈值，或分类主导模态不是文本。"""
    pmax = full["probs"].max(1)
    return (pmax < CFG["REVIEW_PMAX"]) | (shap_df.dom_cls.values != "T")


def review_rule_eval(full, shap_df, ds):
    flag = review_flags(full, shap_df)
    err = full["probs"].argmax(1) != ds.c
    nt = shap_df.dom_cls.values != "T"
    lowp = full["probs"].max(1) < CFG["REVIEW_PMAX"]
    r = dict(coverage=float(flag.mean()),
             err_rate_all=float(err.mean()),
             err_rate_flagged=float(err[flag].mean()) if flag.any() else None,
             err_rate_unflagged=float(err[~flag].mean()) if (~flag).any() else None,
             share_of_errors_caught=float(err[flag].sum() / max(1, err.sum())),
             n_lowp=int(lowp.sum()), err_rate_lowp=float(err[lowp].mean()) if lowp.any() else None,
             n_nontext_dominant=int(nt.sum()), err_rate_nontext=float(err[nt].mean()) if nt.any() else None,
             err_rate_text=float(err[~nt].mean()) if (~nt).any() else None)
    return flag, r


# %% [q3] 8. 问题三：精确 Shapley、逐位置重要性、删除检验、稳定性、解释卡
SUBSETS = [(), ("T",), ("A",), ("V",), ("T", "A"), ("T", "V"), ("A", "V"), ("T", "A", "V")]


def _remove_except(ds, idx, keep):
    n = len(idx)
    return {m: (np.zeros((n, ds.T), bool) if m in keep else ds.o(m)[idx].copy()) for m in MODS}


def exact_shapley(pred, ds, idx=None):
    """三个模态只有 8 个子集，Shapley 值可以精确计算（不用抽样近似）；分类头与强度头各算一套。"""
    idx = np.arange(ds.n) if idx is None else np.asarray(idx)
    n = len(idx)
    full = pred.predict(ds, idx)
    cstar = full["probs"].argmax(1)
    vc, vr = {}, {}
    for S in SUBSETS:
        p = full if len(S) == 3 else pred.predict(ds, idx, _remove_except(ds, idx, S))
        vc[S] = p["probs"][np.arange(n), cstar]; vr[S] = p["yhat"]
    phi = {h: {m: np.zeros(n) for m in MODS} for h in ("cls", "reg")}
    for m in MODS:
        others = [x for x in MODS if x != m]
        for k in range(0, 3):
            for S in combinations(others, k):
                w = math.factorial(k) * math.factorial(2 - k) / 6.0
                Sm = tuple(x for x in MODS if x in S or x == m)
                S0 = tuple(x for x in MODS if x in S)
                phi["cls"][m] += w * (vc[Sm] - vc[S0])
                phi["reg"][m] += w * (vr[Sm] - vr[S0])
    rows = dict(id=[ds.ids[i] for i in idx], cstar=cstar, p_cstar=vc[("T", "A", "V")], yhat=vr[("T", "A", "V")],
                v0_cls=vc[()], v0_reg=vr[()])
    for h in ("cls", "reg"):
        mat = np.stack([phi[h][m] for m in MODS], 1)
        for j, m in enumerate(MODS):
            rows[f"phi_{h}_{m}"] = mat[:, j]
        absm = np.abs(mat); tot = absm.sum(1, keepdims=True)
        share = np.where(tot > 1e-9, absm / np.maximum(tot, 1e-12), np.nan)
        for j, m in enumerate(MODS):
            rows[f"share_{h}_{m}"] = share[:, j]
        srt = np.sort(np.nan_to_num(share), 1)
        rows[f"margin_{h}"] = srt[:, -1] - srt[:, -2]
        rows[f"dom_{h}"] = np.array(MODS)[absm.argmax(1)]
        v_full = vc[("T", "A", "V")] if h == "cls" else vr[("T", "A", "V")]
        v_0 = vc[()] if h == "cls" else vr[()]
        rows[f"eff_err_{h}"] = np.abs(mat.sum(1) - (v_full - v_0))
    return pd.DataFrame(rows), full


def local_importance(pred, ds, i, cstar, noise=None, mods=MODS):
    """样本 i 的每个模态：逐位置遮蔽以该位置为中心、宽 LOCAL_WIN 的窗口，得分 = |Δ强度| + |Δp(c*)|。"""
    h = CFG["LOCAL_WIN"] // 2
    base = pred.predict(ds, [i], None, noise)
    b_p = base["probs"][0, cstar]; b_y = base["yhat"][0]
    res = {}
    for m in mods:
        P = np.flatnonzero(ds.o(m)[i])
        if len(P) == 0:
            res[m] = (P, np.zeros(0)); continue
        R = len(P)
        ms = {x: np.zeros((R, ds.T), bool) for x in MODS}
        for r_, t in enumerate(P):
            ms[m][r_, P[(P >= t - h) & (P <= t + h)]] = True
        p = pred.predict(ds, np.full(R, i), ms, noise)
        res[m] = (P, np.abs(p["yhat"] - b_y) + np.abs(p["probs"][:, cstar] - b_p))
    return res


def faithfulness(pred, ds, idx, imps, cstars, rng):
    """删除检验与保留检验：高分位置 vs 低分位置 vs 随机位置（同一样本内重复 FAITH_RAND 次取平均）。"""
    rows = []
    for i, cs in zip(idx, cstars):
        base = pred.predict(ds, [i])["probs"][0, cs]
        for m in MODS:
            P, imp = imps[i][m]
            if len(P) < 2:
                continue
            order = np.argsort(-imp)
            for bud in CFG["FAITH_BUDGETS"]:
                k = max(1, int(round(bud * len(P))))
                sets = [("top", P[order[:k]]), ("low", P[order[-k:]])]
                sets += [("rand", rng.choice(P, k, replace=False)) for _ in range(CFG["FAITH_RAND"])]
                R = 2 * len(sets)
                ms = {x: np.zeros((R, ds.T), bool) for x in MODS}
                for j, (_, S) in enumerate(sets):
                    ms[m][2 * j, S] = True
                    ms[m][2 * j + 1, np.setdiff1d(P, S)] = True
                p = pred.predict(ds, np.full(R, i), ms)["probs"][:, cs]
                d_del = base - p[0::2]; d_keep = base - p[1::2]
                rows.append(dict(id=ds.ids[i], modality=m, budget=bud, k=k, n_pos=len(P),
                                 del_top=d_del[0], del_low=d_del[1], del_rand=float(d_del[2:].mean()),
                                 keep_top=d_keep[0], keep_low=d_keep[1], keep_rand=float(d_keep[2:].mean())))
    return pd.DataFrame(rows)


def words_of(ds, i, tokenizer):
    """子词位置 → 整词（处理 ## 续接）。返回每个位置对应的整词（非词位置为空串）与子词序列。"""
    ids = [int(x) for x in ds.input_ids[i]]
    toks = tokenizer.convert_ids_to_tokens(ids) if tokenizer is not None else [f"<{x}>" for x in ids]
    words = [""] * ds.T
    spans = []
    for t in np.flatnonzero(ds.W[i]):
        tk = toks[t]
        if tk.startswith("##") and spans and spans[-1][-1] == t - 1:
            spans[-1].append(t)
        else:
            spans.append([t])
    for sp in spans:
        w = "".join(toks[t][2:] if toks[t].startswith("##") else toks[t] for t in sp)
        for t in sp:
            words[t] = w
    return words, toks, spans


def evidence_of(imp_res, words, mod, topk):
    """取该模态得分最高的 topk 个位置，相邻的合并成片段。"""
    P, imp = imp_res[mod]
    if len(P) == 0:
        return []
    order = np.argsort(-imp)[:topk]
    pos = np.sort(P[order])
    spans, cur = [], [pos[0]]
    for t in pos[1:]:
        if t == cur[-1] + 1:
            cur.append(t)
        else:
            spans.append(cur); cur = [t]
    spans.append(cur)
    out = []
    for sp in spans:
        ws = []
        for t in sp:
            if words[t] and (not ws or ws[-1] != words[t]):
                ws.append(words[t])
        out.append(dict(modality=mod, positions=[int(t) for t in sp], words=" ".join(ws),
                        score=float(imp[np.isin(P, sp)].max())))
    return sorted(out, key=lambda e: -e["score"])


def explain_set(pred, ds, idx, tokenizer, rng, shap_df=None, noise_check=True):
    """对一组样本输出解释卡（含逐位置重要性），并做删除检验和稳定性检验。"""
    from scipy.stats import spearmanr
    if shap_df is None:
        shap_df, _ = exact_shapley(pred, ds, idx)
    sd = shap_df.set_index("id")
    cards, imps, spear = [], {}, []
    for i in idx:
        r = sd.loc[ds.ids[i]]
        cs = int(r.cstar)
        imp = local_importance(pred, ds, i, cs)
        imps[i] = imp
        if noise_check:
            imp_n = local_importance(pred, ds, i, cs, noise=(np.random.default_rng(int(i)), CFG["STAB_NOISE"]))
            for m in MODS:
                a, b = imp[m][1], imp_n[m][1]
                if len(a) >= 3 and np.std(a) > 1e-9 and np.std(b) > 1e-9:
                    spear.append(dict(id=ds.ids[i], modality=m, rho=float(spearmanr(a, b).correlation)))
        words, toks, _ = words_of(ds, i, tokenizer)
        card = dict(id=ds.ids[i], pred_class=CLASS_NAMES[cs], p_pred=float(r.p_cstar), intensity_raw=float(r.yhat),
                    shapley_cls={m: float(r[f"phi_cls_{m}"]) for m in MODS},
                    shapley_reg={m: float(r[f"phi_reg_{m}"]) for m in MODS},
                    share_cls={m: float(r[f"share_cls_{m}"]) for m in MODS},
                    share_reg={m: float(r[f"share_reg_{m}"]) for m in MODS},
                    dominant_cls=r.dom_cls, dominant_reg=r.dom_reg, margin_cls=float(r.margin_cls),
                    evidence={m: evidence_of(imp, words, m, CFG["EVID_TOPK"]) for m in MODS},
                    local_importance={m: dict(positions=[int(t) for t in imp[m][0]],
                                              words=[words[t] for t in imp[m][0]],
                                              score=[float(s) for s in imp[m][1]]) for m in MODS})
        cards.append(card)
    faith = faithfulness(pred, ds, idx, imps, [int(sd.loc[ds.ids[i]].cstar) for i in idx], rng)
    stab = pd.DataFrame(spear)
    return cards, faith, stab, imps


def pick_explain_valid(ds, n_total):
    """验证集解释样本：按真实类别分层，用 SHA-256(盐+样本号) 排序选取，不看模型预测。"""
    per = n_total // 3
    out = []
    for c in range(3):
        cand = [i for i in range(ds.n) if ds.c[i] == c]
        cand.sort(key=lambda i: hashlib.sha256(("gmcm2026E-" + ds.ids[i]).encode()).hexdigest())
        out += cand[:per]
    return np.array(sorted(out))

# %% [analysis] 9. 补充检验：完整输入等价性、训练缺失族覆盖性、错误归因、效率
def native_indicators(ds, idx=None):
    idx = np.arange(ds.n) if idx is None else np.asarray(idx)
    aT, aA, aV, _ = apply_masks(ds, idx, None)
    phi, indT, indF = descriptors(aT, aA, aV, ref_rows(ds, idx))
    return phi, indT, indF


def identity_check(results, ds):
    """结构保证的实测：无原生缺失的样本上，MG-LoRA 与同种子阶段一模型的输出应当逐位相同。"""
    _, _, indF = native_indicators(ds)
    clean = indF == 0
    rows = []
    for (cfg_name, seed) in results:
        if cfg_name != "mg" or ("base", seed) not in results:
            continue
        a, b = results[("mg", seed)]["full"], results[("base", seed)]["full"]
        dp = np.abs(a["probs"] - b["probs"]).max(1); dy = np.abs(a["yhat"] - b["yhat"])
        rows.append(dict(split=ds.name, seed=seed, n_clean=int(clean.sum()), n_native_missing=int((~clean).sum()),
                         max_abs_dprob_clean=float(dp[clean].max()) if clean.any() else None,
                         max_abs_dy_clean=float(dy[clean].max()) if clean.any() else None,
                         mean_abs_dprob_native_missing=float(dp[~clean].mean()) if (~clean).any() else None,
                         pred_changed_native_missing=int((a["probs"].argmax(1) != b["probs"].argmax(1))[~clean].sum())))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / f"identity_check_{ds.name}.csv", index=False)
    return df


def family_coverage(data, n_draw=20):
    """覆盖性检查（事后，只读附件3的缺失描述量，不参与训练与选择）：
    附件3 的缺失比例、最长缺口是否落在训练缺失族在验证集上生成的分布范围内。"""
    if "att3" not in data:
        return None
    va = data["valid"]; rng = np.random.default_rng(7)
    fam = []
    for _ in range(n_draw):
        idx = np.arange(va.n)
        ms, masked = sample_family_masks(va, idx, rng, 1.0)
        aT, aA, aV, _ = apply_masks(va, idx, ms)
        fam.append(descriptors(aT, aA, aV, ref_rows(va, idx))[0][masked])
    fam = np.concatenate(fam)
    phi3, _, _ = native_indicators(data["att3"])
    names = ["frac_T", "gap_T", "any_T", "frac_A", "gap_A", "any_A", "frac_V", "gap_V", "any_V"]
    rows = []
    for j, nm in enumerate(names):
        if nm.startswith("any"):
            rows.append(dict(feature=nm, att3_mean=float(phi3[:, j].mean()), family_mean=float(fam[:, j].mean())))
            continue
        sel = fam[:, j][fam[:, j] > 0]
        lo, hi = (np.percentile(sel, 1), np.percentile(sel, 99)) if len(sel) else (0, 0)
        v = phi3[:, j][phi3[:, j] > 0]
        rows.append(dict(feature=nm, att3_n_nonzero=int(len(v)), att3_min=float(v.min()) if len(v) else None,
                         att3_max=float(v.max()) if len(v) else None, family_p01=float(lo), family_p99=float(hi),
                         att3_share_inside=float(((v >= lo) & (v <= hi)).mean()) if len(v) else None,
                         att3_quantile_in_family=[float((sel <= x).mean()) for x in v] if len(v) else []))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "att3_family_coverage.csv", index=False)
    return df


def _bins(x, edges, labels):
    return pd.cut(x, edges, labels=labels, include_lowest=True).astype(str)


def error_attribution(ds, full, shap_df, tag, c_pred=None, y_pred=None):
    """验证集错误归因：按真实类别、强度大小、句长、原生缺失、主导模态、置信度分组看错误率与 MAE。"""
    c_pred = full["probs"].argmax(1) if c_pred is None else c_pred
    y_pred = full["yhat"] if y_pred is None else y_pred
    pmax = full["probs"].max(1)
    _, _, indF = native_indicators(ds)
    df = pd.DataFrame(dict(id=ds.ids, video=ds.videos, y=ds.y, c=ds.c, c_pred=c_pred, y_pred=y_pred, pmax=pmax,
                           words=ds.W.sum(1), native_missing=np.where(indF > 0, "yes", "no"),
                           raw_text=ds.raw_text))
    df["err"] = (df.c != df.c_pred).astype(int)
    df["abs_err_y"] = np.abs(df.y - df.y_pred)
    df["true_class"] = [CLASS_NAMES[k] for k in df.c]
    df["pred_class"] = [CLASS_NAMES[k] for k in df.c_pred]
    df["abs_y_bin"] = np.where(df.y == 0, "=0", _bins(np.abs(df.y), [0, 0.5, 1, 2, 3.01], ["(0,0.5]", "(0.5,1]", "(1,2]", "(2,3]"]))
    df["len_bin"] = _bins(df.words, [0, 10, 20, 30, 100], ["<=10", "11-20", "21-30", ">30"])
    df["conf_bin"] = _bins(df.pmax, [0, 0.5, 0.7, 0.9, 1.0], ["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"])
    if shap_df is not None:
        sd = shap_df.set_index("id").loc[df.id]
        df["dom_cls"] = sd.dom_cls.values; df["margin_cls"] = sd.margin_cls.values
        df["share_T"] = sd.share_cls_T.values
        df["dom_group"] = np.where(df.dom_cls == "T", "text", "non-text")
    order = dict(true_class=CLASS_NAMES, abs_y_bin=["=0", "(0,0.5]", "(0.5,1]", "(1,2]", "(2,3]"],
                 len_bin=["<=10", "11-20", "21-30", ">30"], conf_bin=["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"],
                 native_missing=["no", "yes"], dom_group=["text", "non-text"])
    rows = []
    for fac in ["true_class", "abs_y_bin", "len_bin", "native_missing", "conf_bin"] + (["dom_group"] if shap_df is not None else []):
        groups = dict(list(df.groupby(fac)))
        for lv in [x for x in order[fac] if x in groups] + [x for x in groups if x not in order[fac]]:
            g = groups[lv]
            rows.append(dict(factor=fac, level=lv, n=len(g), share=len(g) / len(df), err_rate=g.err.mean(),
                             mae=g.abs_err_y.mean(), mean_pmax=g.pmax.mean()))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "tables" / f"error_attribution_{tag}.csv", index=False)
    cm = pd.DataFrame(conf_mat(ds.c, c_pred), index=[f"true_{c}" for c in CLASS_NAMES], columns=[f"pred_{c}" for c in CLASS_NAMES])
    cm.to_csv(OUT / "tables" / f"confusion_{tag}.csv")
    wrong = df[df.err == 1].copy()
    wrong["wrong_conf"] = wrong.pmax
    cols = ["id", "true_class", "pred_class", "y", "y_pred", "pmax", "words"] + (["dom_cls", "share_T"] if shap_df is not None else []) + ["raw_text"]
    wrong.sort_values("wrong_conf", ascending=False)[cols].head(40).to_csv(OUT / "tables" / f"error_cases_{tag}.csv", index=False)
    # 错误模式计数
    pat = wrong.groupby(["true_class", "pred_class"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    pat.to_csv(OUT / "tables" / f"error_patterns_{tag}.csv", index=False)
    return tab, df, cm, pat


def efficiency(preds, ds):
    rows = []
    if "TorchPredictor" not in globals() or not hasattr(next(iter(preds.values())), "state"):
        return pd.DataFrame()
    for key in [("base", CFG["DEPLOY_SEED"]), ("mg", CFG["DEPLOY_SEED"])]:
        if key not in preds:
            continue
        p0 = preds[key]
        p = TorchPredictor(p0.state, p0.gate_mode, key=p0.key)
        p._model()
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t = time.time(); p.predict(ds, np.arange(ds.n))
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t_full = time.time() - t
        nm, cb, r, pos, ms = PANELS[ds.name]["PM"][-1]
        t = time.time(); p.predict(ds, np.arange(ds.n), ms)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        rows.append(dict(config=key[0], n=ds.n, ms_per_sample_complete=1000 * t_full / ds.n,
                         ms_per_sample_missing_TAV=1000 * (time.time() - t) / ds.n))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "efficiency.csv", index=False)
    return df


# %% [att3] 10. 附件3 推理（部署模型：MG-LoRA，部署种子事先固定）
def run_att3(preds, rules, ds, deploy_seed):
    pr = preds[("mg", deploy_seed)]
    full = pr.predict(ds, np.arange(ds.n))
    shap_df, _ = exact_shapley(pr, ds)
    c, y_out, y_cal = rules.apply(full["probs"], full["yhat"]) if rules is not None else \
        (full["probs"].argmax(1), full["yhat"], full["yhat"])
    phi, indT, indF = native_indicators(ds)
    flag = review_flags(full, shap_df)
    votes = []
    for (cfg_name, s), p in preds.items():
        if cfg_name == "mg":
            f = p.predict(ds, np.arange(ds.n))
            votes.append(rules.apply(f["probs"], f["yhat"])[0] if rules is not None else f["probs"].argmax(1))
    votes = np.stack(votes, 1)
    agree = (votes == c[:, None]).mean(1)
    base = preds[("base", deploy_seed)].predict(ds, np.arange(ds.n))
    cb, yb, _ = rules.apply(base["probs"], base["yhat"]) if rules is not None else (base["probs"].argmax(1), base["yhat"], None)
    sub = pd.DataFrame(dict(sample_id=ds.ids, polarity=[CLASS_NAMES[k] for k in c], polarity_code=c - 1,
                            intensity=np.round(y_out, 4), p_negative=full["probs"][:, 0], p_neutral=full["probs"][:, 1],
                            p_positive=full["probs"][:, 2], confidence=full["probs"].max(1),
                            review_flag=flag.astype(int)))
    sub.to_csv(OUT / "attachment3_predictions.csv", index=False, encoding="utf-8-sig")
    det = sub.copy()
    det["intensity_model_raw"] = full["yhat"]; det["intensity_calibrated"] = y_cal
    det["gate_text"] = full["gT"]; det["gate_fusion"] = full["gF"]
    for j, nm in enumerate(["miss_frac_T", "longest_gap_T", "has_miss_T", "miss_frac_A", "longest_gap_A", "has_miss_A",
                            "miss_frac_V", "longest_gap_V", "has_miss_V"]):
        det[nm] = phi[:, j]
    for m in MODS:
        det[f"share_cls_{m}"] = shap_df[f"share_cls_{m}"].values; det[f"phi_reg_{m}"] = shap_df[f"phi_reg_{m}"].values
    det["dominant_modality"] = shap_df.dom_cls.values
    det["seed_agreement"] = agree; det["n_seeds"] = votes.shape[1]
    det["base_polarity"] = [CLASS_NAMES[k] for k in cb]; det["base_intensity"] = yb
    det["changed_vs_base"] = (cb != c).astype(int)
    det["words"] = ds.W.sum(1); det["raw_text"] = ds.raw_text
    det.to_csv(OUT / "attachment3_predictions_detail.csv", index=False, encoding="utf-8-sig")
    summ = dict(n=ds.n, polarity_counts=sub.polarity.value_counts().to_dict(),
                intensity_mean=float(y_out.mean()), confidence_mean=float(sub.confidence.mean()),
                review_flags=int(flag.sum()), dominant_counts=det.dominant_modality.value_counts().to_dict(),
                seed_agreement_mean=float(agree.mean()), changed_vs_base=int(det.changed_vs_base.sum()),
                gate_fusion_mean=float(full["gF"].mean()), gate_text_mean=float(full["gT"].mean()),
                has_miss=dict(T=int(phi[:, 2].sum()), A=int(phi[:, 5].sum()), V=int(phi[:, 8].sum())),
                miss_frac_mean=dict(T=float(phi[:, 0].mean()), A=float(phi[:, 3].mean()), V=float(phi[:, 6].mean())))
    save_json(summ, OUT / "tables" / "att3_summary.json")
    log("附件3 汇总：", summ)
    return det, summ


# %% [att4] 11. 附件4：证据定位到原始视频（词位 → 秒 → 帧）
_MMS = {}


def match_videos(ds, vids):
    by_stem = {}
    for p in vids:
        by_stem.setdefault(_stem(p).lower(), []).append(p)
    out = {}
    for i, sid in enumerate(ds.ids):
        s = sid.lower()
        parts = s.split("$_$")
        vid, clip = (parts[0], parts[1]) if len(parts) == 2 else ("", "")
        cands = []
        for key in (s, s.replace("$_$", "_"), s.replace("$_$", "-"), clip):
            if key and key in by_stem:
                cands += by_stem[key]
        if vid and cands:
            cands = [p for p in cands if vid in p.lower()] or cands
        if not cands:
            nums = re.findall(r"\d+", s)
            if nums:
                k = nums[-1].lstrip("0") or "0"
                hit = [p for p in vids if (re.findall(r"\d+", _stem(p)) or ["x"])[-1].lstrip("0") == k]
                if len(hit) == 1:
                    cands = hit
        if cands:
            out[i] = (cands[0], "name")
    if len(out) < ds.n and len(vids) == ds.n:
        for i in range(ds.n):
            if i not in out:
                out[i] = (vids[i], "order")
    return out


def video_info(p):
    import cv2
    cap = cv2.VideoCapture(p)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0; nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(fps), int(nfr), (float(nfr / fps) if fps > 0 else None)


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _norm_word(w):
    return re.sub(r"[^a-z']", "", w.lower())


def mms_align(video, words):
    """MMS 强制对齐（torchaudio 自带，需联网下载权重）：已知词序列 → 每个词的起止秒。"""
    if _MMS.get("failed"):
        raise RuntimeError("MMS 模型不可用（此前加载失败），跳过")
    if "model" not in _MMS:
        try:
            import torchaudio
            b = torchaudio.pipelines.MMS_FA
            _MMS.update(model=b.get_model(with_star=False).to(DEVICE).eval(), tok=b.get_tokenizer(),
                        aligner=b.get_aligner(), sr=b.sample_rate)
            log("   MMS 强制对齐模型已加载")
        except Exception:
            _MMS["failed"] = True
            raise
    import soundfile as sf
    ff = _ffmpeg()
    if ff is None:
        raise RuntimeError("没有 ffmpeg")
    wav = OUT / "att4_keyframes" / "_tmp.wav"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", video, "-ac", "1", "-ar", "16000", str(wav)], check=True)
    x, sr = sf.read(str(wav), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == _MMS["sr"], f"采样率 {sr}"
    xt = torch.tensor(x)[None].to(DEVICE)
    with torch.inference_mode():
        em, _ = _MMS["model"](xt)
    spans = _MMS["aligner"](em[0].float().cpu(), _MMS["tok"](words))
    ratio = xt.shape[1] / em.shape[1] / sr
    return [(float(s[0].start * ratio), float(s[-1].end * ratio)) for s in spans]


def att4_timing(ds, tokenizer):
    """每个样本：词片段（子词合并后的整词）→ 起止秒。优先 MMS 强制对齐；否则按词序在时长内均分（标注为近似）。"""
    vids = find_videos4()
    vm = match_videos(ds, vids)
    log(f"附件4 视频：找到 {len(vids)} 个，匹配 {len(vm)}/{ds.n}")
    ext = None
    if CFG["ATT4_ALIGN_CSV"] and os.path.exists(CFG["ATT4_ALIGN_CSV"]):
        ext = pd.read_csv(CFG["ATT4_ALIGN_CSV"])
    out = {}
    for i in range(ds.n):
        words, toks, spans = words_of(ds, i, tokenizer)
        span_words = [words[sp[0]] for sp in spans]
        info = dict(video=None, match=None, fps=None, n_frames=None, duration=None, method="none", pos_time={})
        if i in vm:
            info["video"], info["match"] = vm[i]
            try:
                info["fps"], info["n_frames"], info["duration"] = video_info(info["video"])
            except Exception as e:
                log("   读视频失败", info["video"], e)
        times = None
        if ext is not None:
            sub = ext[ext["sample"].astype(str) == ds.ids[i]]
            if len(sub):
                info["method"] = "external_csv"
                for _, r in sub.iterrows():
                    info["pos_time"][int(r.position)] = (float(r.start_s), float(r.end_s))
        if not info["pos_time"] and info["video"] and CFG["ATT4_TIMING"] == "auto":
            try:
                norm = [_norm_word(w) for w in span_words]
                keep = [j for j, w in enumerate(norm) if w]
                al = mms_align(info["video"], [norm[j] for j in keep])
                times = [None] * len(spans)
                for j, t in zip(keep, al):
                    times[j] = t
                info["method"] = "mms_forced_alignment"
            except Exception as e:
                if not _MMS.get("warned"):
                    log(f"   {ds.ids[i]}: MMS 强制对齐失败，改用按词位比例近似：", repr(e)[:160])
                    _MMS["warned"] = _MMS.get("failed", False)
                times = None
        if not info["pos_time"] and times is None and info["duration"] and CFG["ATT4_TIMING"] != "off":
            K = max(1, len(spans)); dur = info["duration"]
            times = [(dur * j / K, dur * (j + 1) / K) for j in range(K)]
            info["method"] = "proportional_approx"
        if times is not None:
            # 标点等无法对齐的词：取前后词时间插值
            for j in range(len(times)):
                if times[j] is None:
                    prev = next((times[k] for k in range(j - 1, -1, -1) if times[k] is not None), None)
                    nxt = next((times[k] for k in range(j + 1, len(times)) if times[k] is not None), None)
                    a = prev[1] if prev else (nxt[0] if nxt else 0.0); b = nxt[0] if nxt else a
                    times[j] = (a, max(a, b))
            for sp, t in zip(spans, times):
                for p in sp:
                    info["pos_time"][int(p)] = t
        out[i] = info
    return out


def pos_to_time(info, positions):
    pt = info["pos_time"]
    if not pt:
        return None
    keys = np.array(sorted(pt))
    ts = []
    for p in positions:
        k = keys[np.argmin(np.abs(keys - p))]           # 非词位置（如 [SEP] 处的声学/视觉帧）取最近的词
        ts.append(pt[int(k)])
    return (min(t[0] for t in ts), max(t[1] for t in ts))


def grab_frame(video, t_sec, out_path):
    import cv2
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000)
    ok, fr = cap.read()
    fidx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    cap.release()
    if not ok:
        return None
    h, w = fr.shape[:2]
    if w > 360:
        fr = cv2.resize(fr, (360, int(h * 360 / w)))
    cv2.imwrite(str(out_path), fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return fidx


def locate_evidence(cards, ds, timing):
    """给解释卡里的每条证据补上：原文词、起止秒、视频帧号；视觉证据额外截一张关键帧。"""
    for card in cards:
        i = ds.ids.index(card["id"])
        info = timing.get(i, {}) if timing else {}
        card["video"] = os.path.basename(info["video"]) if info.get("video") else None
        card["timing_method"] = info.get("method", "none")
        fps = info.get("fps")
        for m in MODS:
            for k, e in enumerate(card["evidence"][m]):
                t = pos_to_time(info, e["positions"]) if info else None
                if t is not None:
                    e["start_s"], e["end_s"] = round(t[0], 3), round(t[1], 3)
                    if fps:
                        e["frame_start"], e["frame_end"] = int(t[0] * fps), int(t[1] * fps)
                    if k == 0 and m in ("V", card["dominant_cls"]) and info.get("video"):
                        # 视觉证据取得分最高位置对应时刻的一帧；主导模态的首条证据也截一帧作为上下文
                        P = card["local_importance"][m]["positions"]; S = card["local_importance"][m]["score"]
                        best = [p for p in e["positions"]]
                        bp = max(best, key=lambda p: S[P.index(p)] if p in P else -1)
                        tb = pos_to_time(info, [bp])
                        fn = OUT / "att4_keyframes" / f"{re.sub(r'[^0-9A-Za-z_-]', '_', card['id'])}_{m}.jpg"
                        try:
                            fidx = grab_frame(info["video"], 0.5 * (tb[0] + tb[1]), fn)
                            if fidx is not None:
                                e["keyframe_index"] = fidx; e["keyframe_file"] = fn.name
                        except Exception as ex:
                            log("   截帧失败", ex)
    return cards


# %% [q3run] 12. 问题三：验证集（性能 + 解释质量）与附件4（全量预测 + 解释）
def summarize_faith(faith):
    if faith is None or not len(faith):
        return pd.DataFrame()
    g = faith.groupby(["modality", "budget"])[["del_top", "del_rand", "del_low", "keep_top", "keep_rand", "keep_low"]].mean()
    g["del_ratio_top_vs_rand"] = g.del_top / g.del_rand.replace(0, np.nan)
    g["top_beats_rand_share"] = faith.assign(w=faith.del_top > faith.del_rand).groupby(["modality", "budget"]).w.mean()
    return g.reset_index()


def run_q3_valid(pred, ds, tokenizer):
    shap_df, full = exact_shapley(pred, ds)
    shap_df.to_csv(OUT / "q3" / "valid_shapley_all.csv", index=False)
    eff = dict(eff_err_cls_max=float(shap_df.eff_err_cls.max()), eff_err_reg_max=float(shap_df.eff_err_reg.max()))
    dom = dict(dominant_counts=shap_df.dom_cls.value_counts().to_dict(),
               dominant_reg_counts=shap_df.dom_reg.value_counts().to_dict(),
               mean_share_cls={m: float(shap_df[f"share_cls_{m}"].mean()) for m in MODS},
               mean_share_reg={m: float(shap_df[f"share_reg_{m}"].mean()) for m in MODS})
    by_class = shap_df.assign(true_class=[CLASS_NAMES[k] for k in ds.c]).groupby("true_class")[
        [f"share_cls_{m}" for m in MODS] + [f"share_reg_{m}" for m in MODS]].mean()
    by_class.to_csv(OUT / "q3" / "valid_share_by_class.csv")
    flag, rev = review_rule_eval(full, shap_df, ds)
    idx = pick_explain_valid(ds, CFG["N_EXPLAIN_VALID"])
    cards, faith, stab, imps = explain_set(pred, ds, idx, tokenizer, np.random.default_rng(11), shap_df)
    faith.to_csv(OUT / "q3" / "valid_faithfulness_raw.csv", index=False)
    fs = summarize_faith(faith); fs.to_csv(OUT / "q3" / "valid_faithfulness_summary.csv", index=False)
    stab.to_csv(OUT / "q3" / "valid_stability.csv", index=False)
    save_json(cards, OUT / "q3" / "valid_explanation_cards.json")
    res = dict(efficiency_check=eff, dominance=dom, review_rule=rev,
               stability_spearman_mean=float(stab.rho.mean()) if len(stab) else None,
               stability_by_modality=stab.groupby("modality").rho.mean().to_dict() if len(stab) else {},
               n_explained=len(idx))
    save_json(res, OUT / "q3" / "valid_q3_summary.json")
    log("问题三（验证集）：", res)
    return shap_df, full, cards, faith, stab, res, imps


def run_q3_att4(pred, rules, ds, tokenizer, data):
    shap_df, full = exact_shapley(pred, ds)
    idx = np.arange(ds.n)
    cards, faith, stab, imps = explain_set(pred, ds, idx, tokenizer, np.random.default_rng(12), shap_df)
    timing = att4_timing(ds, tokenizer) if CFG["ATT4_TIMING"] != "off" else {}
    cards = locate_evidence(cards, ds, timing)
    c, y_out, y_cal = rules.apply(full["probs"], full["yhat"]) if rules is not None else \
        (full["probs"].argmax(1), full["yhat"], full["yhat"])
    # 附件4 中能在附件2 找回的样本（只做一致性参考，不作为评价依据）
    # 先按样本号匹配，再按词元序列（[CLS]…[SEP]）逐位相同匹配
    lookup, by_tok = {}, {}
    for k in ("train", "valid", "test"):
        if k in data:
            d2 = data[k]
            for j, sid in enumerate(d2.ids):
                lookup[sid] = (k, float(d2.y[j]), sid)
                by_tok[tuple(d2.input_ids[j, : d2.sep[j] + 1].tolist())] = (k, float(d2.y[j]), sid)
    for j, sid in enumerate(ds.ids):
        if sid not in lookup:
            hit = by_tok.get(tuple(ds.input_ids[j, : ds.sep[j] + 1].tolist()))
            if hit:
                lookup[sid] = hit
    rows = []
    for card, ci, yi in zip(cards, c, y_out):
        card["pred_class"] = CLASS_NAMES[ci]; card["intensity"] = float(yi)
        ev = card["evidence"]
        def fmt(m, key):
            e = ev[m][0] if ev[m] else {}
            return e.get(key)
        i = ds.ids.index(card["id"])
        rows.append(dict(
            sample_id=card["id"], polarity=CLASS_NAMES[ci], intensity=round(float(yi), 4), confidence=card["p_pred"],
            main_modality=card["dominant_cls"], main_modality_intensity=card["dominant_reg"],
            contrib_T=card["share_cls"]["T"], contrib_A=card["share_cls"]["A"], contrib_V=card["share_cls"]["V"],
            shapley_cls_T=card["shapley_cls"]["T"], shapley_cls_A=card["shapley_cls"]["A"], shapley_cls_V=card["shapley_cls"]["V"],
            shapley_reg_T=card["shapley_reg"]["T"], shapley_reg_A=card["shapley_reg"]["A"], shapley_reg_V=card["shapley_reg"]["V"],
            dominance_margin=card["margin_cls"],
            text_evidence=" | ".join(e["words"] for e in ev["T"]), text_positions="; ".join(str(e["positions"]) for e in ev["T"]),
            text_time_s=(fmt("T", "start_s"), fmt("T", "end_s")) if fmt("T", "start_s") is not None else None,
            audio_positions=fmt("A", "positions"), audio_words=fmt("A", "words"),
            audio_time_s=(fmt("A", "start_s"), fmt("A", "end_s")) if fmt("A", "start_s") is not None else None,
            visual_positions=fmt("V", "positions"), visual_words=fmt("V", "words"),
            visual_time_s=(fmt("V", "start_s"), fmt("V", "end_s")) if fmt("V", "start_s") is not None else None,
            visual_frames=(fmt("V", "frame_start"), fmt("V", "frame_end")) if fmt("V", "frame_start") is not None else None,
            visual_keyframe=fmt("V", "keyframe_file"), timing_method=card.get("timing_method"), video=card.get("video"),
            in_att2=lookup.get(card["id"], (None, None, None))[0], att2_label=lookup.get(card["id"], (None, None, None))[1],
            att2_id=lookup.get(card["id"], (None, None, None))[2],
            raw_text=ds.raw_text[i]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "attachment4_predictions_explanations.csv", index=False, encoding="utf-8-sig")
    save_json(cards, OUT / "attachment4_explanations.json")
    faith.to_csv(OUT / "q3" / "att4_faithfulness_raw.csv", index=False)
    summarize_faith(faith).to_csv(OUT / "q3" / "att4_faithfulness_summary.csv", index=False)
    stab.to_csv(OUT / "q3" / "att4_stability.csv", index=False)
    shap_df.to_csv(OUT / "q3" / "att4_shapley.csv", index=False)
    summ = dict(n=ds.n, polarity_counts=df.polarity.value_counts().to_dict(),
                main_modality_counts=df.main_modality.value_counts().to_dict(),
                mean_contrib={m: float(df[f"contrib_{m}"].mean()) for m in MODS},
                timing_methods=df.timing_method.value_counts().to_dict(),
                n_keyframes=int(df.visual_keyframe.notna().sum()),
                recoverable_in_att2=int(df.in_att2.notna().sum()),
                stability_spearman_mean=float(stab.rho.mean()) if len(stab) else None,
                efficiency_err_max=float(shap_df.eff_err_cls.max()))
    rec = df[df.in_att2.notna()]
    if len(rec):
        cls_true = np.where(rec.att2_label < 0, 0, np.where(rec.att2_label > 0, 2, 1))
        summ["recoverable_polarity_match"] = int((cls_true == np.array([CLASS_NAMES.index(p) for p in rec.polarity])).sum())
        summ["recoverable_mae"] = float(np.abs(rec.att2_label - rec.intensity).mean())
        summ["recoverable_splits"] = rec.in_att2.value_counts().to_dict()
    save_json(summ, OUT / "q3" / "att4_summary.json")
    log("附件4 汇总：", summ)
    return df, cards, shap_df, imps, summ


# %% [test] 13. 测试集一次性评估（所有选择都已在验证集上完成）
def run_test(preds, rules, deploy_seed, te):
    res, df, agg = evaluate_all(preds, te)
    boot = bootstrap_comparisons(res, te, deploy_seed)
    sdiff = seed_level_diffs(df, "test")
    rows = []
    for cfg_name in ("base", "mg"):
        for (cn, s), r in res.items():
            if cn != cfg_name:
                continue
            f = r["full"]
            m0 = r["complete"]
            row = dict(config=cn, seed=s, complete_f1=m0["f1"], complete_acc=m0["acc"], complete_mae=m0["mae"], complete_corr=m0["corr"])
            if rules is not None:
                c, y, _ = rules.apply(f["probs"], f["yhat"])
                m1 = metrics(te.c, te.y, f["probs"], y, c_pred=c)
                row.update(rules_f1=m1["f1"], rules_acc=m1["acc"], rules_mae=m1["mae"], rules_corr=m1["corr"])
                pm = []
                for (_, _, _, _, p) in r["PM"]:
                    c2, y2, _ = rules.apply(p["probs"], p["yhat"])
                    pm.append(metrics(te.c, te.y, p["probs"], y2, c_pred=c2))
                row.update(PM_avg_f1=r["PM_sum"]["avg_f1"], PM_avg_mae=r["PM_sum"]["avg_mae"],
                           rules_PM_avg_f1=float(np.mean([m["f1"] for m in pm])), rules_PM_avg_mae=float(np.mean([m["mae"] for m in pm])))
            rows.append(row)
    rt = pd.DataFrame(rows)
    rt.to_csv(OUT / "tables" / "test_rules_effect.csv", index=False)
    return res, df, agg, boot, sdiff, rt

# %% [figures] 14. 作图（英文标签，避免 Kaggle 上缺中文字体）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})
CFG_COLORS = dict(base="#7f7f7f", joint="#1f77b4", mg="#d62728", mg_nogate="#ff7f0e", mg_nokd="#2ca02c", mg_scatter="#9467bd")
CFG_ORDER = ["base", "joint", "mg_nogate", "mg_nokd", "mg_scatter", "mg"]
FIG = OUT / "figures"


def safe(fn):
    def wrap(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            import traceback
            log(f"  [作图跳过] {fn.__name__}: {e!r}")
            traceback.print_exc()
            plt.close("all")
    wrap.__name__ = fn.__name__
    return wrap


def _esc(x):
    return str(x).replace("$", r"\$")


def _save(fig, name):
    fig.savefig(FIG / f"{name}.png"); plt.close(fig)


@safe
def fig_data_structure(data):
    tr = data["train"]
    order = np.argsort(tr.sep)[:: max(1, tr.n // 400)]
    panels = [("train: text observed", tr.oT[order]), ("train: audio observed", tr.oA[order]),
              ("train: vision observed", tr.oV[order])]
    if "att3" in data:
        a3 = data["att3"]
        panels += [("att3: audio observed", a3.oA), ("att3: vision observed", a3.oV)]
    fig, axs = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 3.2))
    for ax, (t, M) in zip(np.atleast_1d(axs), panels):
        ax.imshow(M, aspect="auto", cmap="Greys", interpolation="nearest")
        ax.set_title(t); ax.set_xlabel("position"); ax.set_ylabel("sample")
    fig.suptitle("Observed (dark) vs zero (light) positions")
    _save(fig, "fig01_data_structure")


@safe
def fig_training_curves(dep):
    fs = sorted(glob.glob(str(OUT / "runs" / f"history_*_s{dep}.csv")))
    if not fs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    for f in fs:
        nm = os.path.basename(f)[8:].rsplit("_s", 1)[0]
        h = pd.read_csv(f)
        axs[0].plot(h.epoch, h.f1, "-o", ms=3, label=nm, color=CFG_COLORS.get(nm))
        if "pm_f1" in h:
            axs[1].plot(h.epoch, h.pm_f1, "-o", ms=3, label=nm, color=CFG_COLORS.get(nm))
    axs[0].set_title("Valid complete-input macro-F1"); axs[1].set_title("Valid PM-panel mean macro-F1")
    for ax in axs:
        ax.set_xlabel("epoch"); ax.legend(fontsize=7)
    _save(fig, "fig02_training_curves")


@safe
def fig_config_compare(sums):
    mets = ["complete_f1", "PM_avg_f1", "PM_worst_f1", "LH_avg_f1", "SC_avg_f1", "MS_avg_f1"]
    fig, axs = plt.subplots(1, len(sums), figsize=(5.2 * len(sums), 3.2), squeeze=False)
    for ax, (split, df) in zip(axs[0], sums.items()):
        cfgs = [c for c in CFG_ORDER if c in set(df.config)]
        w = 0.8 / len(cfgs)
        for j, c in enumerate(cfgs):
            d = df[df.config == c]
            mu = [d[m].mean() for m in mets]; sd = [d[m].std(ddof=1) if len(d) > 1 else 0 for m in mets]
            ax.bar(np.arange(len(mets)) + j * w, mu, w, yerr=sd, capsize=2, label=f"{c} (n={len(d)})", color=CFG_COLORS[c])
        ax.set_xticks(np.arange(len(mets)) + 0.4 - w / 2); ax.set_xticklabels(mets, rotation=20)
        lo = min(df[m].min() for m in mets); ax.set_ylim(max(0, lo - 0.05), None)
        ax.set_title(f"{split}: macro-F1 by condition (mean ± sd over seeds)"); ax.legend(fontsize=6, ncol=2)
    _save(fig, "fig03_config_compare")


@safe
def fig_pm_heatmap(res, dep, split):
    if ("mg", dep) not in res or ("base", dep) not in res:
        return
    def piv(k):
        d = res[k]["PM_df"].copy(); d["col"] = d.ratio.astype(str) + "/" + d.pos
        return d.pivot(index="combo", columns="col", values="f1").loc[COMBOS]
    B, M = piv(("base", dep)), piv(("mg", dep))
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (t, D, cm) in zip(axs, [("base", B, "viridis"), ("MG-LoRA", M, "viridis"), ("MG-LoRA - base", M - B, "RdBu")]):
        v = np.nanmax(np.abs(D.values)) if cm == "RdBu" else None
        im = ax.imshow(D.values, aspect="auto", cmap=cm, vmin=-v if v else None, vmax=v)
        ax.set_yticks(range(len(D.index))); ax.set_yticklabels(D.index)
        ax.set_xticks(range(len(D.columns))); ax.set_xticklabels(D.columns, rotation=60, fontsize=6)
        ax.set_title(f"{split} PM macro-F1: {t}"); fig.colorbar(im, ax=ax, fraction=0.04)
        for (r_, c_), val in np.ndenumerate(D.values):
            ax.text(c_, r_, f"{val:.2f}" if cm != "RdBu" else f"{val:+.2f}", ha="center", va="center", fontsize=5,
                    color="w" if cm != "RdBu" else "k")
    _save(fig, f"fig04_pm_heatmap_{split}")


@safe
def fig_ext_scan(ext):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for nm, df in ext.items():
        g = df.groupby(["combo", "ratio"])[["f1", "mae"]].mean().reset_index()
        for cb, st in zip(["T", "A", "V", "AV", "TAV"], ["-o", "-s", "-^", "-v", "-d"]):
            d = g[g.combo == cb]
            ls = st if nm == "mg" else st.replace("-", "--")
            axs[0].plot(d.ratio, d.f1, ls, ms=3, label=f"{nm}:{cb}", color=CFG_COLORS.get(nm), alpha=0.5 + 0.1 * len(cb))
            axs[1].plot(d.ratio, d.mae, ls, ms=3, label=f"{nm}:{cb}", color=CFG_COLORS.get(nm), alpha=0.5 + 0.1 * len(cb))
    axs[0].set_ylabel("macro-F1"); axs[1].set_ylabel("MAE")
    for ax in axs:
        ax.set_xlabel("missing ratio within modality (1.0 = whole modality)"); ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Extended scan (valid): all-or-nothing behaviour of text")
    _save(fig, "fig05_extended_scan")


@safe
def fig_gate(gdf):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    d = gdf[gdf.panel == "PM"]
    mm = d[["miss_T", "miss_A", "miss_V"]].max(1)
    b = pd.cut(mm, [-0.01, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    g = d.groupby(b).gateF.agg(["mean", "std", "count"])
    axs[0].errorbar(range(len(g)), g["mean"], g["std"], fmt="-o", ms=3)
    axs[0].set_xticks(range(len(g))); axs[0].set_xticklabels([str(x) for x in g.index], rotation=45, fontsize=6)
    axs[0].set_title("fusion gate vs max missing fraction"); axs[0].set_ylabel("gate g_F")
    bt = pd.cut(d.miss_T, [-0.01, 0, 0.1, 0.2, 0.3, 0.5, 1.0])
    g2 = d.groupby(bt).gateT.agg(["mean", "std"])
    axs[1].errorbar(range(len(g2)), g2["mean"], g2["std"], fmt="-o", ms=3, color="C3")
    axs[1].set_xticks(range(len(g2))); axs[1].set_xticklabels([str(x) for x in g2.index], rotation=45, fontsize=6)
    axs[1].set_title("text gate vs text missing fraction")
    gc = d.groupby("combo").gateF.mean().loc[COMBOS]
    axs[2].bar(gc.index, gc.values, color="C1"); axs[2].set_title("mean fusion gate by masked modalities")
    _save(fig, "fig06_gate")


@safe
def fig_error_attr(tab, tag):
    facs = [f for f in ["true_class", "abs_y_bin", "len_bin", "conf_bin", "dom_group", "native_missing"] if f in set(tab.factor)]
    fig, axs = plt.subplots(1, len(facs), figsize=(2.3 * len(facs), 2.8))
    for ax, f in zip(np.atleast_1d(axs), facs):
        d = tab[tab.factor == f]
        ax.bar(d.level.astype(str), d.err_rate, color="C0")
        for x, (e, n) in enumerate(zip(d.err_rate, d.n)):
            ax.text(x, e, f"n={n}", ha="center", va="bottom", fontsize=6)
        ax.set_title(f); ax.tick_params(axis="x", rotation=45)
    np.atleast_1d(axs)[0].set_ylabel("error rate (3-class)")
    _save(fig, f"fig07_error_attribution_{tag}")


@safe
def fig_confusions(cms, name):
    fig, axs = plt.subplots(1, len(cms), figsize=(2.9 * len(cms), 2.6))
    for ax, (t, C) in zip(np.atleast_1d(axs), cms.items()):
        Cn = C / np.maximum(1, C.sum(1, keepdims=True))
        ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(C):
            ax.text(j, i, f"{int(v)}\n{Cn[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(3)); ax.set_xticklabels(["neg", "neu", "pos"]); ax.set_yticks(range(3)); ax.set_yticklabels(["neg", "neu", "pos"])
        ax.set_xlabel("pred"); ax.set_ylabel("true"); ax.set_title(t)
    _save(fig, name)


@safe
def fig_att3(det):
    fig, axs = plt.subplots(2, 2, figsize=(9, 5.5))
    x = np.arange(len(det))
    col = det.polarity.map({"Negative": "C0", "Neutral": "C7", "Positive": "C3"})
    axs[0, 0].bar(x, det.intensity, color=col); axs[0, 0].axhline(0, color="k", lw=0.5)
    axs[0, 0].set_title("att3: predicted intensity (color = polarity)"); axs[0, 0].set_xlabel("sample")
    axs[0, 1].bar(x, det.confidence, color=col); axs[0, 1].axhline(CFG["REVIEW_PMAX"], ls="--", color="k", lw=0.6)
    axs[0, 1].set_title("att3: confidence (max class prob)")
    w = 0.28
    for j, m in enumerate(MODS):
        axs[1, 0].bar(x + (j - 1) * w, det[f"miss_frac_{m}"], w, label=m)
    axs[1, 0].set_title("att3: missing fraction per modality"); axs[1, 0].legend(fontsize=7)
    axs[1, 1].scatter(det[["miss_frac_T", "miss_frac_A", "miss_frac_V"]].max(1), det.gate_fusion, s=12, c=col)
    axs[1, 1].set_xlabel("max missing fraction"); axs[1, 1].set_ylabel("fusion gate"); axs[1, 1].set_title("att3: gate opens with missingness")
    _save(fig, "fig08_attachment3")


@safe
def fig_q3_modality(shap_va, df4):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    axs[0].boxplot([shap_va[f"share_cls_{m}"].dropna() for m in MODS], labels=["text", "audio", "vision"], showfliers=False)
    axs[0].set_title("valid: Shapley share (classification head)")
    if df4 is not None and len(df4):
        x = np.arange(len(df4)); bottom = np.zeros(len(df4))
        for m, c in zip(MODS, ["C3", "C0", "C2"]):
            axs[1].bar(x, df4[f"contrib_{m}"], bottom=bottom, color=c, label=m); bottom += df4[f"contrib_{m}"].fillna(0).values
        axs[1].set_xticks(x); axs[1].set_xticklabels([_esc(s[-10:]) for s in df4.sample_id], rotation=90, fontsize=5)
        axs[1].set_title("att4: modality contribution per sample"); axs[1].legend(fontsize=7)
    _save(fig, "fig09_q3_modality_share")


@safe
def fig_q3_local(card, tag):
    fig, axs = plt.subplots(3, 1, figsize=(9, 5.2))
    for ax, m, c in zip(axs, MODS, ["C3", "C0", "C2"]):
        li = card["local_importance"][m]
        if not li["positions"]:
            ax.set_title(f"{m}: no observed positions"); continue
        ax.bar(li["positions"], li["score"], color=c)
        ev = card["evidence"][m]
        for e in ev:
            ax.axvspan(min(e["positions"]) - 0.5, max(e["positions"]) + 0.5, color="gold", alpha=0.3)
        if m == "T":
            ax.set_xticks(li["positions"]); ax.set_xticklabels([_esc(w) for w in li["words"]], rotation=70, fontsize=6)
        ax.set_ylabel(m)
    axs[0].set_title(f"{_esc(card['id'])}  pred={card['pred_class']} p={card['p_pred']:.2f}  "
                     f"share T/A/V = {card['share_cls']['T']:.2f}/{card['share_cls']['A']:.2f}/{card['share_cls']['V']:.2f}")
    axs[-1].set_xlabel("aligned position (word index)")
    _save(fig, f"fig10_local_{tag}")


@safe
def fig_q3_faith(fsum, tag):
    mods = [m for m in MODS if m in set(fsum.modality)]
    fig, axs = plt.subplots(1, len(mods), figsize=(3.2 * len(mods), 2.8), squeeze=False)
    for ax, m in zip(axs[0], mods):
        d = fsum[fsum.modality == m]
        for k, st in (("del_top", "-o"), ("del_rand", "--s"), ("del_low", ":^")):
            ax.plot(d.budget, d[k], st, ms=3, label=k)
        ax.set_title(f"{m}: drop of p(c*) after deletion"); ax.set_xlabel("deleted share"); ax.legend(fontsize=6)
    _save(fig, f"fig11_faithfulness_{tag}")


@safe
def fig_seed_diffs(sd_all):
    mets = ["complete_f1", "PM_avg_f1", "PM_worst_f1", "LH_avg_f1", "SC_avg_f1", "MS_avg_f1"]
    d = sd_all[(sd_all["compare"] == "mg - base") & (sd_all["metric"].isin(mets))]
    if not len(d):
        return
    splits = list(d.split.unique())
    fig, axs = plt.subplots(1, len(splits), figsize=(4.5 * len(splits), 2.8), squeeze=False)
    for ax, sp in zip(axs[0], splits):
        dd = d[d.split == sp].set_index("metric").loc[[m for m in mets if m in set(d[d.split == sp].metric)]]
        for x, (m, r) in enumerate(dd.iterrows()):
            ps = r.per_seed if isinstance(r.per_seed, list) else json.loads(str(r.per_seed))
            ax.scatter([x] * len(ps), ps, s=12, color="C3", alpha=0.7)
            ax.errorbar(x, r["mean"], [[r["mean"] - r.lo], [r.hi - r["mean"]]] if np.isfinite(r.lo) else None, fmt="k_", capsize=4)
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(len(dd))); ax.set_xticklabels(dd.index, rotation=20)
        ax.set_title(f"{sp}: MG-LoRA - base per seed (95% t-CI)")
    _save(fig, "fig12_seed_diffs")


# %% [report] 15. 汇总报告
def _md(df, floatfmt=".4f"):
    if df is None or not len(df):
        return "（无）\n"
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt) + "\n"
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```\n"


def _cfg_table(agg):
    cols = [("complete_f1", "完整F1"), ("complete_acc", "完整Acc"), ("complete_mae", "完整MAE"), ("complete_corr", "完整Corr"),
            ("complete_acc2", "Acc-2"), ("complete_acc7", "Acc-7"), ("PM_avg_f1", "PM均F1"), ("PM_worst_f1", "PM最差F1"),
            ("PM_avg_mae", "PM均MAE"), ("LH_avg_f1", "LH均F1"), ("SC_avg_f1", "SC均F1"), ("MS_avg_f1", "MS均F1"), ("complete_ece", "ECE")]
    rows = []
    for cfg_name in [c for c in CFG_ORDER if c in agg.index]:
        r = {"配置": cfg_name, "种子数": int(agg.loc[cfg_name, "complete_f1_count"])}
        for k, nm in cols:
            mu, sd = agg.loc[cfg_name, f"{k}_mean"], agg.loc[cfg_name, f"{k}_std"]
            r[nm] = f"{mu:.4f}" + (f"±{sd:.4f}" if np.isfinite(sd) else "")
        rows.append(r)
    return pd.DataFrame(rows)


def write_report(S):
    L = ["# GMCM 2026 E 题 · MG-LoRA 运行报告\n",
         f"- 运行时间：{time.strftime('%Y-%m-%d %H:%M')}，总用时 {(time.time() - T0) / 60:.1f} 分钟；SMOKE={CFG['SMOKE']}",
         f"- 部署种子（事先固定）：{CFG['DEPLOY_SEED']}；主配置种子 {CFG['SEEDS_MAIN']}，消融种子 {CFG['SEEDS_ABL']}",
         f"- 环境：{S.get('env')}\n"]
    L += ["## 1 数据审计\n", _md(S["audit"].round(4).T.reset_index().rename(columns={"index": "项"}), ".3f"),
          f"参照区间偏移：`{json.dumps(S.get('offsets'), ensure_ascii=False, default=_json_default)}`\n"]
    for split in ("valid", "test"):
        if split in S.get("agg", {}):
            L += [f"## 2 各配置汇总（{split}，均值±标准差）\n", _md(_cfg_table(S["agg"][split]))]
    for split in ("valid", "test"):
        if split in S.get("seed_diffs", {}):
            d = S["seed_diffs"][split]
            d = d[d.metric.isin(["complete_f1", "complete_mae", "PM_avg_f1", "PM_worst_f1", "LH_avg_f1", "SC_avg_f1", "MS_avg_f1"])]
            L += [f"## 3 种子配对差值（{split}，A−B，95% t 区间）\n", _md(d.drop(columns=["per_seed"]))]
    for split in ("valid", "test"):
        if split in S.get("boot", {}):
            L += [f"## 4 视频配对重抽样（{split}，部署种子）\n", _md(S["boot"][split])]
    if "identity" in S:
        L += ["## 5 完整输入等价性（结构保证的实测）\n", _md(S["identity"])]
    if "gate_corr" in S:
        L += ["## 6 门控分析\n", f"`{json.dumps(S['gate_corr'], ensure_ascii=False, default=_json_default)}`\n"]
    if "law" in S:
        law = S["law"]
        L += ["## 7 缺失规律（验证集，MG-LoRA，部署种子）\n", _md(law[law.factor.isin(["combo", "ratio", "pos"])])]
    if "ext" in S:
        for nm, df in S["ext"].items():
            g = df.groupby(["combo", "ratio"])[["f1", "mae"]].mean().reset_index()
            L += [f"### 扩展扫描：{nm}\n", _md(g.pivot(index="combo", columns="ratio", values="f1").reset_index())]
    if "rules" in S:
        L += ["## 8 规则层（验证集按视频分组 5 折，折外结果）\n", _md(pd.DataFrame(S["rules"]["cv"])),
              f"决定：`{json.dumps({k: v for k, v in S['rules'].items() if k != 'cv'}, ensure_ascii=False, default=_json_default)}`\n"]
    if "err_tab" in S:
        L += ["## 9 验证集错误归因（部署模型，完整输入）\n", _md(S["err_tab"]), _md(S.get("err_pat"))]
    if "att3" in S:
        L += ["## 10 附件3\n", f"`{json.dumps(S['att3'], ensure_ascii=False, default=_json_default)}`\n"]
        if S.get("coverage") is not None:
            L += ["覆盖性检查（事后，只读附件3缺失描述量）：\n", _md(S["coverage"].drop(columns=["att3_quantile_in_family"], errors="ignore"))]
    if "q3_valid" in S:
        L += ["## 11 问题三（验证集）\n", f"`{json.dumps(S['q3_valid'], ensure_ascii=False, default=_json_default)}`\n",
              _md(S.get("faith_valid"))]
    if "att4" in S:
        L += ["## 12 附件4\n", f"`{json.dumps(S['att4'], ensure_ascii=False, default=_json_default)}`\n"]
        if S.get("att4_df") is not None:
            L += [_md(S["att4_df"][["sample_id", "polarity", "intensity", "confidence", "main_modality", "contrib_T",
                                    "contrib_A", "contrib_V", "text_evidence", "timing_method"]])]
    if "test_rules" in S:
        L += ["## 13 测试集：规则层效果\n", _md(S["test_rules"])]
    if "eff" in S and S["eff"] is not None and len(S["eff"]):
        L += ["## 14 效率\n", _md(S["eff"]), f"参数量：`{json.dumps(S.get('params'), default=_json_default)}`\n"]
    L += ["## 附：文件\n", "\n".join(f"- {p.relative_to(OUT)}" for p in sorted(OUT.rglob("*"))
                                     if p.is_file() and "ckpt" not in p.parts and p.suffix in (".csv", ".json", ".png", ".md"))]
    txt = "\n".join(L)
    (OUT / "REPORT.md").write_text(txt, encoding="utf-8")
    log("报告已写入 REPORT.md")
    return txt


def zip_outputs():
    zp = Path(CFG["OUT_DIR"]).parent / (Path(CFG["OUT_DIR"]).name + ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and "ckpt" not in p.parts and not p.name.startswith("_tmp"):
                z.write(p, p.relative_to(OUT.parent))
    log(f"已打包 {zp}（{zp.stat().st_size / 1e6:.1f} MB，不含检查点）")
    return zp

# %% [runA] A. 读数据 → 审计 → 构造评估面板
data, STATS, OFF_INFO = load_all()
AUDIT = audit(data, OFF_INFO)
PRIOR = prior_of(data["train"])
_ynz = np.abs(data["train"].y[data["train"].y != 0])
EPS_Y = float(min(1 / 3, max(0.05, _ynz.min()))) if len(_ynz) else 0.1     # 非零强度的最小幅度（极性—强度一致性规则用）
PANELS = {}
for _k in ("valid", "test"):
    if _k in data:
        PANELS[_k] = dict(PM=build_panel(data[_k], "PM"), LH=build_panel(data[_k], "LH"), SC=build_sc_panel(data[_k]),
                          MS=build_ms_panel(data[_k]))
DEP = CFG["DEPLOY_SEED"]
S = dict(audit=AUDIT, offsets=OFF_INFO, agg={}, seed_diffs={}, boot={})
fig_data_structure(data)
log("面板：", {k: {kk: len(v) for kk, v in p.items()} for k, p in PANELS.items()}, "| 先验", PRIOR, "| eps_y", EPS_Y)

# %% [runB] B. 训练全部配置（有检查点则直接读取）
BERT_PATH = find_bert()
TOKENIZER = load_tokenizer(BERT_PATH)
PREDS, HISTS = run_training(data)
try:
    import transformers
    S["env"] = dict(torch=torch.__version__, transformers=transformers.__version__, python=sys.version.split()[0],
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", bert=str(BERT_PATH))
except Exception:
    S["env"] = dict(python=sys.version.split()[0])
fig_training_curves(DEP)

# %% [runC] C. 问题二 · 验证集：各配置 × 三套面板、种子配对差、视频重抽样、等价性、缺失规律、门控
VA = data["valid"]
RES, SUMS, AGGS = {}, {}, {}
RES["valid"], SUMS["valid"], AGGS["valid"] = evaluate_all(PREDS, VA)
S["agg"]["valid"] = AGGS["valid"]
S["seed_diffs"]["valid"] = seed_level_diffs(SUMS["valid"], "valid")
S["boot"]["valid"] = bootstrap_comparisons(RES["valid"], VA, DEP)
S["identity"] = identity_check(RES["valid"], VA)
S["law"] = law_tables(RES["valid"], ("mg", DEP), VA, "mg")
law_tables(RES["valid"], ("base", DEP), VA, "base")
if CFG["RUN_EXT_SCAN"]:
    S["ext"] = {nm: extended_scan(PREDS[(nm, DEP)], VA, nm) for nm in ("base", "mg")}
    fig_ext_scan(S["ext"])
GATE_DF, S["gate_corr"] = gate_analysis(RES["valid"], ("mg", DEP), VA)
fig_gate(GATE_DF)
fig_pm_heatmap(RES["valid"], DEP, "valid")
log("等价性检查：\n" + S["identity"].to_string())

# %% [runD] D. 规则层（分组交叉验证后决定启用哪几条）+ 问题三验证集 + 错误归因
PRED = PREDS[("mg", DEP)]
FULL_VA = RES["valid"][("mg", DEP)]["full"]
RULES = None
if CFG["RUN_RULES"]:
    RULES = RuleLayer(EPS_Y).fit(FULL_VA, [x[4] for x in RES["valid"][("mg", DEP)]["PM"]], VA)
    S["rules"] = RULES.info
SHAP_VA, _, CARDS_VA, FAITH_VA, STAB_VA, S["q3_valid"], IMPS_VA = run_q3_valid(PRED, VA, TOKENIZER)
S["faith_valid"] = summarize_faith(FAITH_VA)
fig_q3_faith(S["faith_valid"], "valid")
_c, _y = (RULES.apply(FULL_VA["probs"], FULL_VA["yhat"])[:2] if RULES is not None else (None, None))
S["err_tab"], ERR_DF, _, S["err_pat"] = error_attribution(VA, FULL_VA, SHAP_VA, "valid_mg", _c, _y)
fig_error_attr(S["err_tab"], "valid_mg")
# 验证集典型样本：一条判对且置信度高、一条判错
_ok = [c for c in CARDS_VA if CLASS_NAMES[VA.c[VA.ids.index(c["id"])]] == c["pred_class"]]
_bad = [c for c in CARDS_VA if CLASS_NAMES[VA.c[VA.ids.index(c["id"])]] != c["pred_class"]]
for _tag, _cs in (("valid_correct", _ok), ("valid_error", _bad)):
    if _cs:
        fig_q3_local(max(_cs, key=lambda c: c["p_pred"]), _tag)

# %% [runE] E. 附件3 推理（MG-LoRA 部署种子 + 规则层 + 复核标记 + 多种子一致率）
DET3 = None
if CFG["RUN_ATT3"] and "att3" in data:
    DET3, S["att3"] = run_att3(PREDS, RULES, data["att3"], DEP)
    S["coverage"] = family_coverage(data)
    fig_att3(DET3)

# %% [runF] F. 问题三 · 附件4：精确 Shapley + 逐位置重要性 + 证据定位到秒/帧 + 解释卡
DF4 = None
if CFG["RUN_Q3"] and "att4" in data:
    DF4, CARDS4, SHAP4, IMPS4, S["att4"] = run_q3_att4(PRED, RULES, data["att4"], TOKENIZER, data)
    S["att4_df"] = DF4
    _pick = []
    if len(CARDS4):
        _pick.append(max(CARDS4, key=lambda c: c["p_pred"]))
        _pick.append(max(CARDS4, key=lambda c: 1 - c["share_cls"]["T"] if np.isfinite(c["share_cls"]["T"]) else -1))
        _pick.append(min(CARDS4, key=lambda c: c["p_pred"]))
        _pick.append(sorted(CARDS4, key=lambda c: c["p_pred"])[len(CARDS4) // 2])
    _seen = set()
    for c in _pick[:CFG["N_TYPICAL"]]:
        if c["id"] not in _seen:
            _seen.add(c["id"]); fig_q3_local(c, "att4_" + re.sub(r"[^0-9A-Za-z_-]", "_", c["id"]))
fig_q3_modality(SHAP_VA, DF4)

# %% [runG] G. 测试集一次性评估（模型、阈值、规则都已在验证集上定好）
if CFG["RUN_TEST"] and "test" in data:
    TE = data["test"]
    RES["test"], SUMS["test"], AGGS["test"], S["boot"]["test"], S["seed_diffs"]["test"], S["test_rules"] = \
        run_test(PREDS, RULES, DEP, TE)
    S["agg"]["test"] = AGGS["test"]
    S["identity_test"] = identity_check(RES["test"], TE)
    _fb = RES["test"][("base", DEP)]["full"]; _fm = RES["test"][("mg", DEP)]["full"]
    _cms = {"base": conf_mat(TE.c, _fb["probs"].argmax(1)), "MG-LoRA": conf_mat(TE.c, _fm["probs"].argmax(1))}
    if RULES is not None:
        _cms["MG-LoRA + rules"] = conf_mat(TE.c, RULES.apply(_fm["probs"], _fm["yhat"])[0])
    fig_confusions(_cms, "fig13_confusion_test")
    error_attribution(TE, _fm, None, "test_mg")
    fig_pm_heatmap(RES["test"], DEP, "test")
fig_config_compare(SUMS)
fig_seed_diffs(pd.concat(list(S["seed_diffs"].values()), ignore_index=True))

# %% [runH] H. 效率、汇总、打包
S["eff"] = efficiency(PREDS, VA)
_pc = OUT / "tables" / "param_counts.json"
S["params"] = json.load(open(_pc)) if _pc.exists() else None
SUMMARY = dict(
    config={k: v for k, v in CFG.items()},
    rules=S.get("rules"), att3=S.get("att3"), att4=S.get("att4"), q3_valid=S.get("q3_valid"),
    identity=S["identity"].to_dict("records") if "identity" in S else None,
    by_config={sp: {c: {m: float(a.loc[c, f"{m}_mean"]) for m in ("complete_f1", "complete_mae", "PM_avg_f1", "PM_worst_f1",
                                                                   "LH_avg_f1", "SC_avg_f1", "MS_avg_f1") if f"{m}_mean" in a.columns}
                    for c in a.index} for sp, a in S["agg"].items()},
    minutes=(time.time() - T0) / 60)
save_json(SUMMARY, OUT / "summary.json")
write_report(S)
zip_outputs()
log("全部完成。请下载 outputs.zip（或 outputs_smoke.zip）发回。")
